//! Concurrent workgroups with explicit asynchronous DMA completion and shared ports.
use crate::{
    Result,
    isa::{Arg, Instr, Op, View, Wave},
    power::Power,
    v09_hardware::{ComputeProfile, Engine, Hardware},
    v09_memory::{Memory, MemoryConfig, Request},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Run { instruction: Instr },
    Async { token: u32, instruction: Instr },
    Wait { tokens: Vec<u32> },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Group {
    pub sm: usize,
    pub rf_kib: usize,
    pub sh_kib: usize,
    pub commands: Vec<Command>,
}
#[derive(Default, Clone, Serialize)]
pub struct Stats {
    pub cycles: u64,
    pub waves: u64,
    pub instructions: u64,
    pub groups: u64,
    pub physical_fmas: u64,
    pub rf_bytes: u64,
    pub sh_bytes: u64,
    pub max_active_dmas: usize,
    pub max_return_queue: usize,
    pub max_frontend: usize,
    pub max_resident_groups: usize,
    pub scheduler_iterations: u64,
    pub max_descriptors_per_engine: usize,
}
#[derive(Serialize)]
pub struct Report {
    pub model: &'static str,
    pub area_au: f64,
    pub base_power_w: f64,
    pub short_power_w: f64,
    pub long_power_w: f64,
    pub power_pass: bool,
    pub energy_j: f64,
    pub stats: Stats,
    pub memory: crate::v09_memory::Stats,
}
#[derive(Default)]
struct Ports {
    read: u64,
    write: u64,
    sh_read: u64,
    sh_write: u64,
    interface: u64,
    issue: u64,
}
struct Running {
    profile: Arc<ComputeProfile>,
    step: usize,
    phase: u8,
    engine: usize,
}
struct State {
    pc: usize,
    ready: u64,
    run: Option<Running>,
    blocking: Option<usize>,
    tokens: HashMap<u32, bool>,
}
#[derive(Clone)]
struct Tx {
    id: u64,
    dma: usize,
    line: u64,
    words: Vec<usize>,
    ready: u64,
    staged: bool,
}
struct Dma {
    group: usize,
    token: Option<u32>,
    engine: usize,
    global: View,
    local: View,
    base: usize,
    shared: bool,
    write: bool,
    next: usize,
    ready: u64,
    front: VecDeque<Tx>,
    inflight: usize,
}
struct Return {
    tx: Tx,
    done: Option<u64>,
}
// This memo contains host-derived immutable timing profiles only. It is not
// architectural cache storage and does not change any simulated resource.
const PROFILE_CACHE_ITEMS: usize = 128;
const PROFILE_CACHE_BYTES: usize = 16 * 1024 * 1024;
struct ProfileCache {
    enabled: bool,
    hardware_key: Vec<u8>,
    entries: HashMap<Vec<u8>, (Arc<ComputeProfile>, usize)>,
    fifo: VecDeque<Vec<u8>>,
    bytes: usize,
}
impl ProfileCache {
    fn new() -> Self {
        Self {
            enabled: std::env::var("VNEXT_PROFILE_CACHE").as_deref() != Ok("0"),
            hardware_key: Vec::new(),
            entries: HashMap::new(),
            fifo: VecDeque::new(),
            bytes: 0,
        }
    }
    fn prepare(&mut self, hardware: &Hardware) -> Result<()> {
        let key = serde_json::to_vec(hardware).map_err(|e| e.to_string())?;
        if key != self.hardware_key {
            self.entries.clear();
            self.fifo.clear();
            self.bytes = 0;
            self.hardware_key = key;
        }
        Ok(())
    }
    fn profile(&mut self, hardware: &Hardware, instruction: &Instr) -> Result<Arc<ComputeProfile>> {
        if !self.enabled {
            return Ok(Arc::new(hardware.compute_profile(instruction)?));
        }
        // Keep all fields, including absolute SH addresses and affine strides:
        // replacing this key with just shape would silently erase bank conflicts.
        let mut key = serde_json::to_vec(instruction).map_err(|e| e.to_string())?;
        // JSON maps nonfinite floats to null. Preserve their exact bits too,
        // even though normal public-input validation rejects nonfinite values.
        match instruction {
            Instr::Fill { value, .. } => key.extend(value.to_bits().to_le_bytes()),
            Instr::Vector { a, b, .. } => {
                for arg in [a, b] {
                    if let Arg::Imm { value } = arg {
                        key.extend(value.to_bits().to_le_bytes());
                    }
                }
            }
            _ => {}
        }
        if let Some((profile, _)) = self.entries.get(&key) {
            return Ok(Arc::clone(profile));
        }
        let profile = Arc::new(hardware.compute_profile(instruction)?);
        let bytes = std::mem::size_of::<ComputeProfile>()
            + 2 * std::mem::size_of::<usize>()
            + profile.steps.capacity() * std::mem::size_of::<crate::v09_hardware::ComputeStep>()
            + profile
                .steps
                .iter()
                .map(|step| {
                    (step.sh_read_service.capacity() + step.sh_write_service.capacity())
                        * std::mem::size_of::<u64>()
                        + 64
                })
                .sum::<usize>()
            + 2 * key.capacity()
            + 256;
        // Reserve 16 KiB for map/deque backing storage and allocator bookkeeping.
        // Profiles larger than the remaining cap are used once, never retained.
        let budget = PROFILE_CACHE_BYTES - 16384;
        if bytes <= budget {
            while self.entries.len() >= PROFILE_CACHE_ITEMS || self.bytes + bytes > budget {
                let old = self
                    .fifo
                    .pop_front()
                    .expect("memo accounting agrees with FIFO");
                if let Some((_, old_bytes)) = self.entries.remove(&old) {
                    self.bytes -= old_bytes;
                }
            }
            self.bytes += bytes;
            self.fifo.push_back(key.clone());
            self.entries.insert(key, (Arc::clone(&profile), bytes));
        }
        Ok(profile)
    }
}
pub struct Scheduler {
    pub hardware: Hardware,
    pub memory: Memory,
    pub power: Power,
    pub stats: Stats,
    pub async_native: bool,
    next_token: u64,
    round: usize,
    profile_cache: ProfileCache,
}
fn engine_index(e: Engine) -> usize {
    match e {
        Engine::Vector => 0,
        Engine::Sfu => 1,
        Engine::Reduction => 2,
        Engine::Tc => 3,
        Engine::Copy => 4,
    }
}
fn dma_info(i: &Instr) -> Option<(usize, View, View, bool, bool)> {
    match *i {
        Instr::Load {
            tensor,
            global,
            local,
        } => Some((tensor, global, local, false, false)),
        Instr::Store {
            tensor,
            global,
            local,
        } => Some((tensor, global, local, true, false)),
        Instr::LoadShared {
            tensor,
            global,
            local,
        } => Some((tensor, global, local, false, true)),
        Instr::StoreShared {
            tensor,
            global,
            local,
        } => Some((tensor, global, local, true, true)),
        _ => None,
    }
}
#[derive(Clone)]
struct Access {
    space: usize,
    view: View,
    write: bool,
}
fn validate_instruction_shape(i: &Instr) -> Result<()> {
    let view = |v: View| -> Result<()> {
        if v.rows
            .checked_mul(v.cols)
            .is_none_or(|n| n == 0 || n > 16384)
            || v.end().is_none()
        {
            Err("invalid or overflowing instruction view".into())
        } else {
            Ok(())
        }
    };
    if let Some((tensor, g, l, _, _)) = dma_info(i) {
        tensor.checked_add(2).ok_or("tensor identifier overflow")?;
        view(g)?;
        view(l)?;
        if g.len() != l.len() {
            return Err("DMA source/destination length mismatch".into());
        }
    } else {
        match *i {
            Instr::Mma { m, n, k, .. } | Instr::MmaShared { m, n, k, .. } => {
                if !(1..=64).contains(&m) || !(1..=64).contains(&n) || !(1..=256).contains(&k) {
                    return Err("MMA dimensions exceed M,N<=64 and K<=256".into());
                }
                if let Instr::MmaShared { b, .. } = i {
                    view(*b)?;
                }
            }
            Instr::Fill { len, .. } | Instr::Vector { len, .. } | Instr::Reduce { len, .. } => {
                if !(1..=8192).contains(&len) {
                    return Err("invalid vector/reduction length".into());
                }
            }
            Instr::SharedRead { shared, local } | Instr::SharedWrite { shared, local } => {
                view(shared)?;
                view(local)?;
                if shared.len() != local.len() {
                    return Err("SH copy length mismatch".into());
                }
            }
            _ => {}
        }
    }
    // Only construct derived MMA extents after bounding all products above.
    for a in accesses(i) {
        view(a.view)?;
    }
    Ok(())
}
fn accesses(i: &Instr) -> Vec<Access> {
    let mut a = vec![];
    let mut push = |space, view, write| a.push(Access { space, view, write });
    if let Some((t, g, l, w, s)) = dma_info(i) {
        push(t + 2, g, w);
        push(usize::from(s), l, !w);
        return a;
    }
    match *i {
        Instr::SharedRead { shared, local } => {
            push(1, shared, false);
            push(0, local, true)
        }
        Instr::SharedWrite { shared, local } => {
            push(0, local, false);
            push(1, shared, true)
        }
        Instr::Fill { dst, len, .. } => push(0, View::contiguous(dst, 1, len), true),
        Instr::Vector {
            dst,
            len,
            a,
            b,
            kind,
        } => {
            for arg in [
                Some(a),
                if matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt | Op::Square) {
                    None
                } else {
                    Some(b)
                },
            ]
            .into_iter()
            .flatten()
            {
                if let Arg::Reg { base, stride } = arg {
                    push(
                        0,
                        View {
                            base,
                            rows: 1,
                            cols: len,
                            row_stride: 0,
                            col_stride: stride,
                        },
                        false,
                    )
                }
            }
            push(0, View::contiguous(dst, 1, len), true)
        }
        Instr::Reduce {
            src,
            dst,
            scratch,
            len,
            ..
        } => {
            push(0, View::contiguous(src, 1, len), false);
            push(0, View::contiguous(dst, 1, 1), true);
            push(0, View::contiguous(scratch, 1, len), true)
        }
        Instr::Mma { a, b, c, m, n, k } => {
            push(0, View::contiguous(a, 1, m * k), false);
            push(0, View::contiguous(b, 1, k * n), false);
            push(0, View::contiguous(c, 1, m * n), true)
        }
        Instr::MmaShared { a, b, c, m, n, k } => {
            push(0, View::contiguous(a, 1, m * k), false);
            push(1, b, false);
            push(0, View::contiguous(c, 1, m * n), true)
        }
        _ => {}
    }
    a
}
fn overlaps(a: &Access, b: &Access) -> bool {
    if a.space != b.space || !(a.write || b.write) {
        return false;
    }
    let (Some(ae), Some(be)) = (a.view.end(), b.view.end()) else {
        return true;
    };
    if ae <= b.view.base || be <= a.view.base {
        return false;
    }
    let aa = &a.view;
    let bb = &b.view;
    if aa.col_stride == 1
        && aa.row_stride == aa.cols
        && bb.col_stride == 1
        && bb.row_stride == bb.cols
    {
        return true;
    }
    let set: HashSet<_> = (0..aa.len()).map(|i| aa.at(i)).collect();
    (0..bb.len()).any(|i| set.contains(&bb.at(i)))
}
pub fn native_groups(wave: &Wave, hw: &Hardware, asynchronous: bool) -> Result<Vec<Group>> {
    wave.iter()
        .enumerate()
        .filter(|(_, p)| !p.is_empty())
        .map(|(index, p)| {
            let (mut r, mut s) = (0, 0);
            let mut commands = vec![];
            let mut tokens = vec![];
            let mut token = 0;
            for i in p {
                validate_instruction_shape(i)?;
                for a in accesses(i) {
                    let end = a.view.end().ok_or("invalid local extent")?;
                    if a.space == 0 {
                        r = r.max(end)
                    } else if a.space == 1 {
                        s = s.max(end)
                    }
                }
                let is_load = matches!(i, Instr::Load { .. } | Instr::LoadShared { .. });
                if asynchronous && hw.dma_depth > 0 && is_load {
                    // Consecutive loads may alias: conservatively join before the next conflicting load.
                    let conflict = commands
                        .iter()
                        .rev()
                        .take_while(|c| !matches!(c, Command::Wait { .. }))
                        .any(|c| match c {
                            Command::Async { instruction, .. } => accesses(i)
                                .iter()
                                .any(|a| accesses(instruction).iter().any(|b| overlaps(a, b))),
                            _ => false,
                        });
                    if (conflict || tokens.len() >= hw.dma_engines * (1 + hw.dma_depth))
                        && !tokens.is_empty()
                    {
                        commands.push(Command::Wait {
                            tokens: std::mem::take(&mut tokens),
                        });
                    }
                    token += 1;
                    tokens.push(token);
                    commands.push(Command::Async {
                        token,
                        instruction: i.clone(),
                    });
                } else {
                    if !tokens.is_empty() {
                        commands.push(Command::Wait {
                            tokens: std::mem::take(&mut tokens),
                        });
                    }
                    commands.push(Command::Run {
                        instruction: i.clone(),
                    });
                }
            }
            if !tokens.is_empty() {
                commands.push(Command::Wait { tokens });
            }
            Ok(Group {
                sm: index % hw.base.sms,
                rf_kib: (r * 4).div_ceil(1024).max(1),
                sh_kib: (s * 4).div_ceil(4096) * 4,
                commands,
            })
        })
        .collect()
}
impl Scheduler {
    pub fn new(hardware: Hardware, async_native: bool) -> Result<Self> {
        hardware.validate()?;
        let b = &hardware.base;
        let memory = Memory::new(MemoryConfig {
            sms: b.sms,
            hbm_channels: b.hbm_channels,
            hbm_queue_depth: b.hbm_queue,
            sm_noc_bytes_per_cycle: b.sm_noc,
            noc_bytes_per_cycle: b.global_noc,
            cache_mib: hardware.cache_mib,
            cache_mshrs: hardware.cache_mshrs(),
            cache_bytes_per_cycle: hardware.cache_bandwidth(),
            cache_latency: hardware.cache_latency(),
            multicast: hardware.multicast,
            max_pending_per_sm: hardware.memory_pending_limit(),
        })?;
        Ok(Self {
            hardware,
            memory,
            power: Power::default(),
            stats: Stats::default(),
            async_native,
            next_token: 1,
            round: 0,
            profile_cache: ProfileCache::new(),
        })
    }
    pub fn native_wave(&mut self, wave: &Wave, bases: &[usize]) -> Result<()> {
        let groups = native_groups(wave, &self.hardware, self.async_native)?;
        self.wave(&groups, bases)
    }
    pub fn validate_groups(&self, groups: &[Group]) -> Result<Wave> {
        let h = &self.hardware;
        let (mut r, mut s, mut n) = (
            vec![0; h.base.sms],
            vec![0; h.base.sms],
            vec![0; h.base.sms],
        );
        let mut wave = vec![];
        let mut count = 0usize;
        for g in groups {
            if g.sm >= h.base.sms
                || g.rf_kib == 0
                || g.rf_kib > h.base.rf_kib
                || g.sh_kib > h.base.sh_kib
                || g.sh_kib % 4 != 0
            {
                return Err("invalid workgroup reservation".into());
            }
            r[g.sm] += g.rf_kib;
            s[g.sm] += g.sh_kib;
            n[g.sm] += 1;
            if r[g.sm] > h.base.rf_kib || s[g.sm] > h.base.sh_kib || n[g.sm] > h.resident_groups {
                return Err("resident workgroups exceed SM RF/SH/slot capacity".into());
            }
            let mut pending: HashMap<u32, Vec<Access>> = HashMap::new();
            let mut seen = HashSet::new();
            let mut p = vec![];
            for c in &g.commands {
                count += 1;
                if count > crate::MAX_WAVE_COMMANDS {
                    return Err("too many commands per wave".into());
                }
                match c {
                    Command::Wait { tokens } => {
                        if tokens.is_empty() {
                            return Err("empty wait".into());
                        }
                        for t in tokens {
                            if pending.remove(t).is_none() {
                                return Err("unknown/repeated wait token".into());
                            }
                        }
                    }
                    Command::Run { instruction } | Command::Async { instruction, .. } => {
                        validate_instruction_shape(instruction)?;
                        let aa = accesses(instruction);
                        for a in &aa {
                            let limit = match a.space {
                                0 => g.rf_kib * 256,
                                1 => g.sh_kib * 256,
                                _ => usize::MAX,
                            };
                            if a.view.end().is_none_or(|e| e > limit)
                                || a.view
                                    .rows
                                    .checked_mul(a.view.cols)
                                    .is_none_or(|n| n > 16384)
                            {
                                return Err("workgroup access exceeds reservation".into());
                            }
                            if pending.values().any(|v| v.iter().any(|b| overlaps(a, b))) {
                                return Err("asynchronous dependency requires explicit wait".into());
                            }
                        }
                        if let Command::Async { token, .. } = c {
                            if h.dma_depth == 0
                                || dma_info(instruction).is_none()
                                || !seen.insert(*token)
                            {
                                return Err("invalid async instruction/token/depth".into());
                            }
                            pending.insert(*token, aa);
                            if pending.len() > h.dma_engines * (1 + h.dma_depth) {
                                return Err("too many live asynchronous tokens".into());
                            }
                        }
                        p.push(instruction.clone());
                    }
                }
            }
            if !pending.is_empty() {
                return Err("every asynchronous token requires an explicit wait".into());
            }
            wave.push(p);
        }
        if wave.iter().map(Vec::len).sum::<usize>() > crate::MAX_WAVE_PRIMITIVES {
            return Err("wave exceeds instruction limit".into());
        }
        Ok(wave)
    }
    fn charge(&mut self, t: u64, bytes: u64, bw: usize, e: f64) {
        if bytes == 0 {
            return;
        }
        let full = bytes / bw as u64;
        let rem = bytes % bw as u64;
        if full > 0 {
            self.power.add(t, full, (full * bw as u64) as f64 * e);
        }
        if rem > 0 {
            self.power.add(t + full, 1, rem as f64 * e);
        }
    }
    fn charge_sh(&mut self, t: u64, service: &[u64], energy: f64) {
        for (i, &bytes) in service.iter().enumerate() {
            self.power.add(t + i as u64, 1, bytes as f64 * energy);
        }
    }
    pub fn wave(&mut self, groups: &[Group], bases: &[usize]) -> Result<()> {
        self.validate_groups(groups)?;
        self.wave_prevalidated(groups, bases)
    }
    /// Crate-internal only: caller must validate these exact groups immediately
    /// before numerical execution, without changing hardware or reservations.
    pub(crate) fn wave_prevalidated(&mut self, groups: &[Group], bases: &[usize]) -> Result<()> {
        let h = self.hardware.clone();
        self.profile_cache.prepare(&h)?;
        let b = &h.base;
        let mut states: Vec<_> = groups
            .iter()
            .map(|_| State {
                pc: 0,
                ready: self.stats.cycles,
                run: None,
                blocking: None,
                tokens: HashMap::new(),
            })
            .collect();
        let mut ports: Vec<_> = (0..b.sms).map(|_| Ports::default()).collect();
        let mut engines: Vec<Vec<Vec<bool>>> = (0..b.sms)
            .map(|_| {
                vec![
                    vec![false; 1],
                    vec![false; 1],
                    vec![false; h.reduction_units],
                    vec![false; h.tc_count],
                    vec![false; h.dma_engines],
                ]
            })
            .collect();
        let mut dmas: Vec<Option<Dma>> = vec![];
        let mut dma_queues: Vec<Vec<VecDeque<usize>>> = (0..b.sms)
            .map(|_| (0..h.dma_engines).map(|_| VecDeque::new()).collect())
            .collect();
        let mut txs: HashMap<u64, Tx> = HashMap::new();
        let mut returns: Vec<Return> = vec![];
        let mut t = self.stats.cycles;
        let mut iterations = 0u64;
        loop {
            iterations += 1;
            if iterations > 200_000_000 || t.saturating_sub(self.stats.cycles) > 2_000_000_000 {
                return Err("concurrent wave host/cycle work budget exceeded".into());
            }
            let completions = self.memory.advance_to(t);
            for e in self.memory.drain_energy() {
                self.power.add(e.cycle, e.duration, e.energy_pj);
            }
            self.power.advance(t);
            for c in completions {
                let tx = txs
                    .remove(&c.request.token)
                    .ok_or("unknown memory completion")?;
                returns.push(Return {
                    tx,
                    done: if c.request.write { Some(t) } else { None },
                });
            }
            let mut k = 0;
            while k < returns.len() {
                let ret = &mut returns[k];
                let d = dmas[ret.tx.dma].as_ref().ok_or("missing DMA")?;
                let sm = groups[d.group].sm;
                if let Some(done) = ret.done {
                    if done <= t {
                        let ret = returns.swap_remove(k);
                        self.memory.ack_local(ret.tx.id)?;
                        dmas[ret.tx.dma].as_mut().unwrap().inflight -= 1;
                        continue;
                    }
                } else {
                    let p = &mut ports[sm];
                    let bytes = ret.tx.words.len() as u64 * 4;
                    if d.shared {
                        if p.sh_write <= t {
                            let service = bank_cycles(&ret.tx.words, b.sh_banks, false, 1);
                            p.sh_write = t + service;
                            ret.done = Some(t + service + b.sh_latency());
                            self.charge_sh(
                                t,
                                &bank_activity(&ret.tx.words, b.sh_banks, false, 1),
                                h.shared_energy_per_byte(true),
                            );
                            self.stats.sh_bytes += bytes;
                        }
                    } else if p.write <= t {
                        let service = bytes.div_ceil(b.write_bw() as u64);
                        p.write = t + service;
                        ret.done = Some(t + service + b.rf_latency());
                        self.charge(t, bytes, b.write_bw(), b.rf_energy());
                        self.stats.rf_bytes += bytes;
                    }
                }
                k += 1;
            }
            // Engines advance one bounded address batch per cycle. Sources and return writes share ports with compute.
            // Hardware validation bounds SMs by 32. Fixed scratch avoids four
            // heap allocations on every micro-event without changing timing.
            let mut front_per_sm = [0usize; 32];
            let mut staged = [0usize; 32];
            for d in dmas.iter().flatten() {
                front_per_sm[groups[d.group].sm] += d.front.len();
                staged[groups[d.group].sm] += d.front.iter().filter(|x| x.staged).count();
            }
            let mut blocked_submit = HashSet::new();
            let mut address_issued = [false; 32];
            let mut submitted = [0usize; 32];
            for (di, dma_slot) in dmas.iter_mut().enumerate() {
                let Some(mut d) = dma_slot.take() else {
                    continue;
                };
                let sm = groups[d.group].sm;
                if dma_queues[sm][d.engine].front() != Some(&di) {
                    *dma_slot = Some(d);
                    continue;
                }
                if d.ready <= t {
                    if d.next < d.global.len() && front_per_sm[sm] <= 16 && !address_issued[sm] {
                        address_issued[sm] = true;
                        let end = (d.next + 16).min(d.global.len());
                        let mut batch: Vec<(u64, Vec<usize>)> = vec![];
                        for index in d.next..end {
                            let line = ((d.base + d.global.at(index) * 4) / 64 * 64) as u64;
                            if let Some((_, v)) = batch.iter_mut().find(|(a, _)| *a == line) {
                                v.push(d.local.at(index));
                            } else {
                                batch.push((line, vec![d.local.at(index)]));
                            }
                        }
                        self.power
                            .add(t, 1, (end - d.next) as f64 * 0.5 + batch.len() as f64 * 2.);
                        for (line, words) in batch {
                            let id = self.next_token;
                            self.next_token += 1;
                            d.front.push_back(Tx {
                                id,
                                dma: di,
                                line,
                                words,
                                ready: t,
                                staged: false,
                            });
                            front_per_sm[sm] += 1;
                        }
                        d.next = end;
                        d.ready = t + 1;
                    }
                    while submitted[sm] < 2 {
                        let Some(tx) = d.front.front_mut() else {
                            break;
                        };
                        if tx.ready > t {
                            break;
                        }
                        if d.write && !tx.staged {
                            if staged[sm] >= 2 {
                                break;
                            }
                            let p = &mut ports[sm];
                            let bytes = tx.words.len() as u64 * 4;
                            let service;
                            if d.shared {
                                if p.sh_read > t {
                                    break;
                                }
                                service = bank_cycles(&tx.words, b.sh_banks, true, h.shared_ports);
                                p.sh_read = t + service;
                                tx.ready = t + service + b.sh_latency();
                                self.charge_sh(
                                    t,
                                    &bank_activity(&tx.words, b.sh_banks, true, h.shared_ports),
                                    h.shared_energy_per_byte(false),
                                );
                                self.stats.sh_bytes += bytes;
                            } else {
                                if p.read > t {
                                    break;
                                }
                                service = bytes.div_ceil(b.read_bw() as u64);
                                p.read = t + service;
                                tx.ready = t + service + b.rf_latency();
                                self.charge(t, bytes, b.read_bw(), b.rf_energy());
                                self.stats.rf_bytes += bytes;
                            }
                            tx.staged = true;
                            staged[sm] += 1;
                            break;
                        }
                        let req = Request {
                            sm,
                            group: d.group as u64,
                            token: tx.id,
                            address: tx.line,
                            valid_bytes: (tx.words.len() * 4) as u32,
                            write: d.write,
                        };
                        if !self.memory.try_submit(req)? {
                            blocked_submit.insert(di);
                            break;
                        }
                        submitted[sm] += 1;
                        self.power.add(t, 1, 8.);
                        let tx = d.front.pop_front().unwrap();
                        front_per_sm[sm] -= 1;
                        if tx.staged {
                            staged[sm] -= 1;
                        }
                        d.inflight += 1;
                        txs.insert(tx.id, tx);
                    }
                }
                if d.next == d.global.len() && d.front.is_empty() && d.inflight == 0 {
                    let completed = dma_queues[sm][d.engine].pop_front();
                    debug_assert_eq!(completed, Some(di));
                    engines[sm][4][d.engine] = !dma_queues[sm][d.engine].is_empty();
                    let state = &mut states[d.group];
                    if let Some(token) = d.token {
                        state.tokens.insert(token, true);
                    } else {
                        state.blocking = None;
                        state.pc += 1;
                        state.ready = t;
                    }
                } else {
                    *dma_slot = Some(d);
                }
            }
            // Rotating group priority; issue width one instruction/SM/cycle.
            for off in 0..groups.len() {
                let gi = (off + self.round) % groups.len();
                let g = &groups[gi];
                let state = &mut states[gi];
                if state.ready > t || state.blocking.is_some() {
                    continue;
                }
                while let Some(mut run) = state.run.take() {
                    if run.step == run.profile.steps.len() {
                        engines[g.sm][engine_index(run.profile.engine)][run.engine] = false;
                        state.pc += 1;
                        state.ready = t;
                        break;
                    }
                    let step = &run.profile.steps[run.step];
                    let p = &mut ports[g.sm];
                    let mut progressed = false;
                    match run.phase {
                        0 => {
                            let ready = (if step.rf_read_bytes > 0 { p.read } else { 0 })
                                .max(if step.sh_read_bytes > 0 { p.sh_read } else { 0 })
                                .max(if step.sh_tc_bytes > 0 { p.interface } else { 0 });
                            if (step.rf_read_bytes == 0 || p.read <= t) && ready <= t {
                                let mut delay = 0;
                                if step.rf_read_bytes > 0 {
                                    let s = step.rf_read_bytes.div_ceil(b.read_bw() as u64);
                                    p.read = t + s;
                                    delay = delay.max(s + b.rf_latency());
                                    self.charge(t, step.rf_read_bytes, b.read_bw(), b.rf_energy());
                                    self.stats.rf_bytes += step.rf_read_bytes;
                                }
                                if step.sh_read_bytes > 0 {
                                    let s = step.sh_read_cycles.max(1);
                                    p.sh_read = t + s;
                                    delay = delay.max(s + b.sh_latency());
                                    self.charge_sh(
                                        t,
                                        &step.sh_read_service,
                                        h.shared_energy_per_byte(false),
                                    );
                                    self.stats.sh_bytes += step.sh_read_bytes;
                                }
                                if step.sh_tc_bytes > 0 {
                                    let start = t + step.sh_read_cycles + b.sh_latency();
                                    let s = step.sh_tc_bytes.div_ceil(b.sh_tc_bw as u64);
                                    p.interface = start + s;
                                    self.charge(start, step.sh_tc_bytes, b.sh_tc_bw, 0.2);
                                    delay = delay.max(start + s + 2 - t);
                                }
                                state.ready = t + delay;
                                progressed = true;
                            }
                        }
                        1 => {
                            if step.core_duration > 0 {
                                self.power.add(t, step.core_duration, step.core_energy_pj);
                                state.ready = t + step.core_duration;
                            }
                            progressed = true;
                        }
                        2 => {
                            if (step.rf_write_bytes == 0 || p.write <= t)
                                && (step.sh_write_bytes == 0 || p.sh_write <= t)
                            {
                                let mut delay = 0;
                                if step.rf_write_bytes > 0 {
                                    let s = step.rf_write_bytes.div_ceil(b.write_bw() as u64);
                                    p.write = t + s;
                                    delay = delay.max(s + b.rf_latency());
                                    self.charge(
                                        t,
                                        step.rf_write_bytes,
                                        b.write_bw(),
                                        b.rf_energy(),
                                    );
                                    self.stats.rf_bytes += step.rf_write_bytes;
                                }
                                if step.sh_write_bytes > 0 {
                                    let s = step.sh_write_cycles.max(1);
                                    p.sh_write = t + s;
                                    delay = delay.max(s + b.sh_latency());
                                    self.charge_sh(
                                        t,
                                        &step.sh_write_service,
                                        h.shared_energy_per_byte(true),
                                    );
                                    self.stats.sh_bytes += step.sh_write_bytes;
                                }
                                state.ready = t + delay;
                                progressed = true;
                            }
                        }
                        _ => {
                            run.step += 1;
                            run.phase = 0;
                            state.run = Some(run);
                            continue;
                        }
                    }
                    if progressed {
                        run.phase += 1;
                    }
                    state.run = Some(run);
                    if !progressed || state.ready > t {
                        break;
                    }
                }
                if state.run.is_some() || state.pc == g.commands.len() {
                    continue;
                }
                match &g.commands[state.pc] {
                    Command::Wait { tokens } => {
                        if ports[g.sm].issue <= t
                            && tokens.iter().all(|x| state.tokens.get(x) == Some(&true))
                        {
                            ports[g.sm].issue = t + 1;
                            for x in tokens {
                                state.tokens.remove(x);
                            }
                            state.pc += 1;
                            state.ready = t + 1;
                            self.stats.instructions += 1;
                            self.power.add(t, 1, 8.);
                        }
                    }
                    Command::Run { instruction } | Command::Async { instruction, .. } => {
                        if ports[g.sm].issue > t {
                            continue;
                        }
                        if let Some((tensor, global, local, write, shared)) = dma_info(instruction)
                        {
                            let async_token =
                                if let Command::Async { token, .. } = &g.commands[state.pc] {
                                    Some(*token)
                                } else {
                                    None
                                };
                            let Some(engine) = dma_queues[g.sm]
                                .iter()
                                .enumerate()
                                .filter(|(e, q)| {
                                    q.len() < 1 + h.dma_depth
                                        && (!engines[g.sm][4][*e] || !q.is_empty())
                                })
                                .min_by_key(|(_, q)| q.len())
                                .map(|(e, _)| e)
                            else {
                                continue;
                            };
                            engines[g.sm][4][engine] = true;
                            let di = dmas.iter().position(Option::is_none).unwrap_or(dmas.len());
                            let dma = Dma {
                                group: gi,
                                token: async_token,
                                engine,
                                global,
                                local,
                                base: *bases.get(tensor).ok_or("unknown tensor")?,
                                shared,
                                write,
                                next: 0,
                                ready: t + 7,
                                front: VecDeque::new(),
                                inflight: 0,
                            };
                            if di == dmas.len() {
                                dmas.push(Some(dma));
                            } else {
                                dmas[di] = Some(dma);
                            }
                            dma_queues[g.sm][engine].push_back(di);
                            self.stats.max_descriptors_per_engine = self
                                .stats
                                .max_descriptors_per_engine
                                .max(dma_queues[g.sm][engine].len());
                            if let Some(token) = async_token {
                                state.tokens.insert(token, false);
                                state.pc += 1;
                                state.ready = t + 1;
                            } else {
                                state.blocking = Some(di);
                            }
                            ports[g.sm].issue = t + 1;
                            self.power.add(t, 1, 8.);
                            self.power.add(t + 1, 6, 20.);
                            self.stats.instructions += 1;
                        } else {
                            let ei = engine_index(instruction_engine(instruction, &h));
                            let Some(engine) = engines[g.sm][ei].iter().position(|&x| !x) else {
                                continue;
                            };
                            let profile = self.profile_cache.profile(&h, instruction)?;
                            engines[g.sm][ei][engine] = true;
                            self.stats.physical_fmas += profile.physical_fmas;
                            state.run = Some(Running {
                                profile,
                                step: 0,
                                phase: 0,
                                engine,
                            });
                            state.ready = t + 1;
                            ports[g.sm].issue = t + 1;
                            self.power.add(t, 1, 8.);
                            self.stats.instructions += 1;
                        }
                    }
                }
            }
            self.round = (self.round + 1) % groups.len().max(1);
            self.stats.max_active_dmas = self
                .stats
                .max_active_dmas
                .max(dmas.iter().flatten().count());
            self.stats.max_return_queue = self.stats.max_return_queue.max(returns.len());
            self.stats.max_frontend = self
                .stats
                .max_frontend
                .max(front_per_sm.iter().copied().max().unwrap_or(0));
            if states.iter().enumerate().all(|(i, s)| {
                s.pc == groups[i].commands.len() && s.run.is_none() && s.blocking.is_none()
            }) && dmas.iter().all(Option::is_none)
                && returns.is_empty()
            {
                break;
            }
            // Jump only to a real future event. Completed groups and unresolved
            // token waits never force cycle-by-cycle polling of HBM latency.
            let mut next = u64::MAX;
            for (gi, state) in states.iter().enumerate() {
                if state.blocking.is_some()
                    || (state.pc == groups[gi].commands.len() && state.run.is_none())
                {
                    continue;
                }
                if state.ready > t {
                    next = next.min(state.ready);
                    continue;
                }
                let sm = groups[gi].sm;
                let port = &ports[sm];
                if let Some(run) = &state.run {
                    let step = &run.profile.steps[run.step];
                    let ready = match run.phase {
                        0 => (if step.rf_read_bytes > 0 { port.read } else { 0 })
                            .max(if step.sh_read_bytes > 0 {
                                port.sh_read
                            } else {
                                0
                            })
                            .max(if step.sh_tc_bytes > 0 {
                                port.interface
                            } else {
                                0
                            }),
                        2 => (if step.rf_write_bytes > 0 {
                            port.write
                        } else {
                            0
                        })
                        .max(if step.sh_write_bytes > 0 {
                            port.sh_write
                        } else {
                            0
                        }),
                        _ => t,
                    };
                    next = next.min(ready.max(t + 1));
                } else {
                    match &groups[gi].commands[state.pc] {
                        Command::Wait { tokens } => {
                            if tokens.iter().all(|x| state.tokens.get(x) == Some(&true)) {
                                next = next.min(port.issue.max(t + 1));
                            }
                        }
                        Command::Run { instruction } | Command::Async { instruction, .. } => {
                            let available = if dma_info(instruction).is_some() {
                                dma_queues[sm].iter().enumerate().any(|(e, q)| {
                                    q.len() < 1 + h.dma_depth
                                        && (!engines[sm][4][e] || !q.is_empty())
                                })
                            } else {
                                let ei = engine_index(instruction_engine(instruction, &h));
                                engines[sm][ei].iter().any(|&busy| !busy)
                            };
                            if available {
                                next = next.min(port.issue.max(t + 1));
                            }
                        }
                    }
                }
            }
            for (di, d) in dmas
                .iter()
                .enumerate()
                .filter_map(|(i, d)| d.as_ref().map(|d| (i, d)))
            {
                let sm = groups[d.group].sm;
                if dma_queues[sm][d.engine].front() != Some(&di) {
                    continue;
                }
                if d.next < d.global.len() && front_per_sm[sm] <= 16 {
                    next = next.min(d.ready.max(t + 1));
                }
                if let Some(tx) = d.front.front()
                    && !blocked_submit.contains(&di)
                {
                    let source = if d.write && !tx.staged {
                        if d.shared {
                            ports[sm].sh_read
                        } else {
                            ports[sm].read
                        }
                    } else {
                        0
                    };
                    next = next.min(tx.ready.max(source).max(t + 1));
                }
            }
            for ret in &returns {
                let d = dmas[ret.tx.dma].as_ref().unwrap();
                let sm = groups[d.group].sm;
                let ready = ret.done.unwrap_or(if d.shared {
                    ports[sm].sh_write
                } else {
                    ports[sm].write
                });
                next = next.min(ready.max(t + 1));
            }
            if let Some(e) = self.memory.next_event_time() {
                next = next.min(e.max(t + 1));
            }
            if next == u64::MAX {
                return Err(format!(
                    "deadlocked concurrent wave at {t}; states={:?}; dmas={:?}; memory_pending={}",
                    states
                        .iter()
                        .map(|s| (
                            s.pc,
                            s.ready,
                            s.blocking,
                            s.run.as_ref().map(|r| (r.step, r.phase)),
                            &s.tokens
                        ))
                        .collect::<Vec<_>>(),
                    dmas.iter()
                        .flatten()
                        .map(|d| (d.group, d.engine, d.next, d.front.len(), d.inflight))
                        .collect::<Vec<_>>(),
                    self.memory.pending()
                ));
            }
            t = next;
        }
        self.stats.scheduler_iterations += iterations;
        self.stats.cycles = t + 1;
        self.stats.waves += 1;
        self.stats.groups += groups.len() as u64;
        self.stats.max_resident_groups = self.stats.max_resident_groups.max(
            (0..b.sms)
                .map(|sm| groups.iter().filter(|g| g.sm == sm).count())
                .max()
                .unwrap_or(0),
        );
        self.power.add(t, 1, b.sms as f64 * 8.);
        Ok(())
    }
    pub fn report(&self) -> Report {
        let base = self.hardware.base_power();
        let (short, long) = self.power.finish(self.stats.cycles, base);
        Report {
            model: crate::MODEL,
            area_au: self.hardware.area(),
            base_power_w: base,
            short_power_w: short,
            long_power_w: long,
            power_pass: short <= 34. && long <= 26.,
            energy_j: self.power.dynamic_pj * 1e-12 + base * self.stats.cycles as f64 * 2e-9,
            stats: self.stats.clone(),
            memory: self.memory.stats().clone(),
        }
    }
}
fn bank_cycles(words: &[usize], banks: usize, read: bool, ports: usize) -> u64 {
    let mut counts = vec![0usize; banks];
    let mut seen = HashSet::new();
    for &word in words {
        if !read || seen.insert(word) {
            counts[word % banks] += 1;
        }
    }
    counts
        .into_iter()
        .map(|n| n.div_ceil(ports))
        .max()
        .unwrap_or(0)
        .max(1) as u64
}

fn instruction_engine(i: &Instr, h: &Hardware) -> Engine {
    match i {
        Instr::Mma { .. } | Instr::MmaShared { .. } => Engine::Tc,
        Instr::SharedRead { .. } | Instr::SharedWrite { .. } => Engine::Copy,
        Instr::Reduce { .. } if h.reduction_units > 0 => Engine::Reduction,
        Instr::Vector {
            kind: Op::Exp | Op::Tanh | Op::Rsqrt,
            ..
        } => Engine::Sfu,
        _ => Engine::Vector,
    }
}

fn bank_activity(words: &[usize], banks: usize, read: bool, ports: usize) -> Vec<u64> {
    let mut counts = vec![0usize; banks];
    for (i, &word) in words.iter().enumerate() {
        if !read || !words[..i].contains(&word) {
            counts[word % banks] += 1;
        }
    }
    let cycles = counts.iter().max().unwrap().div_ceil(ports);
    (0..cycles)
        .map(|cycle| {
            counts
                .iter()
                .map(|&n| n.saturating_sub(cycle * ports).min(ports) as u64 * 4)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hardware() -> Hardware {
        let mut h = Hardware::default();
        h.base.sms = 2;
        h.base.sfu = 1;
        h.resident_groups = 2;
        h
    }
    fn group(commands: Vec<Command>) -> Group {
        Group {
            sm: 0,
            rf_kib: 8,
            sh_kib: 0,
            commands,
        }
    }
    fn load(token: u32, local: usize) -> Command {
        Command::Async {
            token,
            instruction: Instr::Load {
                tensor: 0,
                global: View::contiguous(local, 1, 16),
                local: View::contiguous(local, 1, 16),
            },
        }
    }
    fn compute() -> Command {
        Command::Run {
            instruction: Instr::Vector {
                kind: Op::Tanh,
                dst: 256,
                len: 128,
                a: Arg::Imm { value: 0.1 },
                b: Arg::Imm { value: 0. },
            },
        }
    }
    #[test]
    fn profile_memo_on_off_preserves_complete_report() {
        let mut h = hardware();
        h.dma_depth = 2;
        h.base.sh_kib = 64;
        h.base.sh_banks = 8;
        let mut g = group(vec![
            load(1, 0),
            Command::Wait { tokens: vec![1] },
            Command::Run {
                instruction: Instr::Fill {
                    dst: 64,
                    len: 64,
                    value: 1.,
                },
            },
            Command::Run {
                instruction: Instr::Mma {
                    a: 0,
                    b: 64,
                    c: 128,
                    m: 8,
                    n: 8,
                    k: 8,
                },
            },
            Command::Run {
                instruction: Instr::SharedWrite {
                    shared: View::contiguous(0, 1, 64),
                    local: View::contiguous(128, 1, 64),
                },
            },
            Command::Run {
                instruction: Instr::SharedRead {
                    shared: View::contiguous(0, 1, 64),
                    local: View::contiguous(256, 1, 64),
                },
            },
            Command::Run {
                instruction: Instr::Reduce {
                    src: 256,
                    dst: 400,
                    scratch: 512,
                    len: 64,
                    max: false,
                },
            },
            compute(),
        ]);
        g.sh_kib = 4;
        let mut on = Scheduler::new(h.clone(), false).unwrap();
        on.profile_cache.enabled = true;
        let mut off = Scheduler::new(h, false).unwrap();
        off.profile_cache.enabled = false;
        for _ in 0..3 {
            on.wave(std::slice::from_ref(&g), &[0]).unwrap();
            off.wave(std::slice::from_ref(&g), &[0]).unwrap();
        }
        assert!(!on.profile_cache.entries.is_empty());
        assert!(off.profile_cache.entries.is_empty());
        assert_eq!(
            serde_json::to_value(on.report()).unwrap(),
            serde_json::to_value(off.report()).unwrap()
        );
    }
    #[test]
    fn memo_is_bounded_and_bank_addresses_are_in_key() {
        let mut h = hardware();
        h.base.sh_kib = 64;
        h.base.sh_banks = 8;
        let mut memo = ProfileCache::new();
        memo.enabled = true;
        memo.prepare(&h).unwrap();
        let op = |stride| Instr::SharedRead {
            shared: View {
                base: 0,
                rows: 16,
                cols: 1,
                row_stride: stride,
                col_stride: 1,
            },
            local: View::contiguous(0, 1, 16),
        };
        let a = memo.profile(&h, &op(1)).unwrap();
        let b = memo.profile(&h, &op(8)).unwrap();
        assert_ne!(a.sh_read_cycles, b.sh_read_cycles);
        assert!(Arc::ptr_eq(&a, &memo.profile(&h, &op(1)).unwrap()));
        for dst in 0..256 {
            memo.profile(
                &h,
                &Instr::Fill {
                    dst,
                    len: 8192,
                    value: 1.,
                },
            )
            .unwrap();
        }
        assert_eq!(memo.entries.len(), 128);
        assert_eq!(memo.entries.len(), memo.fifo.len());
        assert!(memo.bytes + 16384 <= PROFILE_CACHE_BYTES);
        h.shared_ports = 2;
        memo.prepare(&h).unwrap();
        assert!(memo.entries.is_empty());
    }
    #[test]
    fn overflowing_public_inputs_are_rejected_without_panics() {
        let s = Scheduler::new(hardware(), false).unwrap();
        for instruction in [
            Instr::Mma {
                a: 0,
                b: 0,
                c: 0,
                m: usize::MAX,
                n: 2,
                k: 2,
            },
            Instr::Vector {
                kind: Op::Add,
                dst: 0,
                len: usize::MAX,
                a: Arg::reg(0),
                b: Arg::reg(0),
            },
            Instr::Load {
                tensor: usize::MAX,
                global: View::contiguous(0, 1, 1),
                local: View::contiguous(0, 1, 1),
            },
            Instr::SharedRead {
                shared: View {
                    base: 0,
                    rows: usize::MAX,
                    cols: 2,
                    row_stride: 2,
                    col_stride: 1,
                },
                local: View::contiguous(0, 1, 1),
            },
        ] {
            assert!(
                s.validate_groups(&[group(vec![Command::Run {
                    instruction: instruction.clone()
                }])])
                .is_err()
            );
            assert!(native_groups(&vec![vec![instruction]], &hardware(), false).is_err());
        }
    }
    #[test]
    fn zero_micro_phases_do_not_cost_ticks() {
        let mut s = Scheduler::new(hardware(), false).unwrap();
        s.wave(
            &[group(vec![Command::Run {
                instruction: Instr::Fill {
                    dst: 0,
                    len: 1,
                    value: 0.,
                },
            }])],
            &[],
        )
        .unwrap();
        assert_eq!(s.stats.cycles, 6); // issue + one write-service + RF latency 3 + fence
    }
    #[test]
    fn finished_groups_do_not_poll_hbm_latency() {
        let mut h = hardware();
        h.dma_depth = 1;
        let mut s = Scheduler::new(h, false).unwrap();
        s.wave(
            &[
                group(vec![]),
                group(vec![load(1, 0), Command::Wait { tokens: vec![1] }]),
            ],
            &[0],
        )
        .unwrap();
        assert!(s.stats.cycles > 250);
        assert!(
            s.stats.scheduler_iterations < 30,
            "{} iterations",
            s.stats.scheduler_iterations
        );
    }
    #[test]
    fn descriptor_depth_is_per_engine_and_changes_overlap() {
        let groups = vec![
            group(vec![
                load(1, 0),
                load(2, 16),
                compute(),
                Command::Wait { tokens: vec![1, 2] },
            ]),
            group(vec![
                load(1, 0),
                load(2, 16),
                compute(),
                Command::Wait { tokens: vec![1, 2] },
            ]),
        ];
        let mut h = hardware();
        h.dma_depth = 1;
        let mut shallow = Scheduler::new(h.clone(), false).unwrap();
        shallow.wave(&groups, &[0]).unwrap();
        h.dma_depth = 2;
        let mut deep = Scheduler::new(h, false).unwrap();
        deep.wave(&groups, &[0]).unwrap();
        assert_eq!(shallow.stats.max_descriptors_per_engine, 2);
        assert_eq!(deep.stats.max_descriptors_per_engine, 3);
        assert!(deep.stats.cycles <= shallow.stats.cycles);
        assert_eq!(deep.stats.max_active_dmas, 3);
    }
    #[test]
    fn multiple_tcs_overlap_but_share_rf() {
        let groups = vec![
            group(vec![Command::Run {
                instruction: Instr::Mma {
                    a: 0,
                    b: 512,
                    c: 1024,
                    m: 8,
                    n: 8,
                    k: 64
                }
            }]);
            2
        ];
        let mut h = hardware();
        let mut one = Scheduler::new(h.clone(), false).unwrap();
        one.wave(&groups, &[]).unwrap();
        h.tc_count = 2;
        let mut two = Scheduler::new(h, false).unwrap();
        two.wave(&groups, &[]).unwrap();
        assert!(two.stats.cycles < one.stats.cycles);
        assert!(two.stats.cycles * 2 > one.stats.cycles);
        assert_eq!(two.stats.rf_bytes, one.stats.rf_bytes);
        assert_eq!(two.stats.physical_fmas, one.stats.physical_fmas);
    }
    #[test]
    fn shared_activity_keeps_concentrated_peak() {
        let words: Vec<_> = (0..8).map(|i| i * 16).chain(1..9).collect();
        assert_eq!(
            bank_activity(&words, 16, true, 1),
            vec![36, 4, 4, 4, 4, 4, 4, 4]
        );
        assert_eq!(bank_activity(&words, 16, true, 2), vec![40, 8, 8, 8]);
    }
}

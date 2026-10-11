use crate::{
    hardware::Hardware,
    isa::{Arg, Instr, Op, View, Wave},
    power::Power,
};
use serde::Serialize;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
};

#[derive(Clone, Copy, Debug)]
struct Request {
    sm: usize,
    channel: usize,
    write: bool,
    valid: usize,
    shared: Option<crate::shared::Batch>,
}
#[derive(Clone, Copy, Debug)]
enum Event {
    WriteReady(Request),
    Up(Request),
    Hbm(Request),
    Down(Request),
    Local(Request),
}
#[derive(Clone, Copy, Debug)]
struct Scheduled {
    time: u64,
    order: u64,
    event: Event,
}
impl PartialEq for Scheduled {
    fn eq(&self, o: &Self) -> bool {
        (self.time, self.order) == (o.time, o.order)
    }
}
impl Eq for Scheduled {}
impl PartialOrd for Scheduled {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Scheduled {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (self.time, self.order).cmp(&(o.time, o.order))
    }
}
#[derive(Clone)]
struct Packet {
    req: Request,
    remaining: usize,
}
struct Channel {
    reads: VecDeque<(u64, Request)>,
    writes: VecDeque<(u64, Request)>,
    used: usize,
    free: u64,
    direction: Option<bool>,
    burst: usize,
}
impl Channel {
    fn new() -> Self {
        Self {
            reads: VecDeque::new(),
            writes: VecDeque::new(),
            used: 0,
            free: 0,
            direction: None,
            burst: 0,
        }
    }
}
struct Dma {
    global: View,
    local: View,
    shared: bool,
    base: usize,
    next: usize,
    write: bool,
    front: VecDeque<Request>,
    inflight: usize,
    start: u64,
}
// One division per batch instead of two per element. Preserve first-seen
// transaction order, including row wrapping and broadcast/overlapping views.
#[inline]
fn coalesce_batch(global: View, base: usize, next: usize) -> ([(usize, usize); 16], usize, usize) {
    let count = (global.len() - next).min(16);
    let mut lines = [(usize::MAX, 0); 16];
    let mut row = next / global.cols;
    let mut col = next % global.cols;
    let mut address = global.base + row * global.row_stride + col * global.col_stride;
    let first = (base + address * 4) / 64;
    if count <= global.cols - col
        && (base + (address + (count - 1) * global.col_stride) * 4) / 64 == first
    {
        lines[0] = (first, count);
        return (lines, 1, count);
    }
    let mut used = 0;
    for i in 0..count {
        let line = (base + address * 4) / 64;
        if let Some(j) = (0..used).find(|&j| lines[j].0 == line) {
            lines[j].1 += 1;
        } else {
            lines[used] = (line, 1);
            used += 1;
        }
        if i + 1 < count {
            col += 1;
            if col == global.cols {
                col = 0;
                row += 1;
                address = global.base + row * global.row_stride;
            } else {
                address += global.col_stride;
            }
        }
    }
    (lines, used, count)
}
struct Sm {
    pc: usize,
    ready: u64,
    local_free: u64,
    write_staged: usize,
    dma: Option<Dma>,
}
#[derive(Default, Serialize, Clone, Debug)]
pub struct Stats {
    pub cycles: u64,
    pub instructions: u64,
    pub hbm_reads: u64,
    pub hbm_writes: u64,
    pub valid_dma_bytes: u64,
    pub noc_bytes: u64,
    pub direction_switches: u64,
    pub max_hbm_inflight: usize,
    pub max_frontend: usize,
    pub max_live_events: usize,
    pub max_write_staged: usize,
    pub max_power_events: usize,
    pub hbm_queue_full_sm_cycles: u64,
    pub waves: u64,
    pub tc_physical_fmas: u64,
    pub rf_bytes: u64,
    pub sh_bytes: u64,
    pub sh_bank_stall_cycles: u64,
    pub max_sh_copy_inflight: usize,
    pub sh_tc_bytes: u64,
    pub sh_tc_stall_cycles: u64,
}
pub struct Timing {
    pub cycle_reference: bool,
    pub(crate) hw: Hardware,
    pub stats: Stats,
    pub power: Power,
    channels: Vec<Channel>,
    order: u64,
    rf_energy: f64,
    sh_energy: f64,
}
impl Timing {
    pub fn hardware(&self) -> &Hardware {
        &self.hw
    }
    pub(crate) fn new(hw: Hardware) -> Self {
        Self {
            cycle_reference: false,
            channels: (0..hw.hbm_channels).map(|_| Channel::new()).collect(),
            rf_energy: hw.rf_energy(),
            sh_energy: hw.sh_energy(),
            hw,
            stats: Stats::default(),
            power: Power::default(),
            order: 0,
        }
    }
    fn push(&mut self, events: &mut BinaryHeap<Reverse<Scheduled>>, time: u64, event: Event) {
        self.order += 1;
        events.push(Reverse(Scheduled {
            time,
            order: self.order,
            event,
        }));
        self.stats.max_live_events = self.stats.max_live_events.max(events.len());
    }
    fn byte_energy(&mut self, t: u64, bytes: usize, bw: usize, energy: f64) {
        let full = bytes / bw;
        if full > 0 {
            self.power.add(t, full as u64, (full * bw) as f64 * energy);
        }
        if !bytes.is_multiple_of(bw) {
            self.power
                .add(t + full as u64, 1, (bytes % bw) as f64 * energy);
        }
    }
    fn rf(&mut self, t: u64, bytes: usize, write: bool) -> u64 {
        if bytes == 0 {
            return 0;
        }
        let bw = if write {
            self.hw.write_bw()
        } else {
            self.hw.read_bw()
        };
        let full = bytes / bw;
        let tail = bytes % bw;
        if full > 0 {
            self.power
                .add(t, full as u64, (full * bw) as f64 * self.rf_energy);
        }
        if tail > 0 {
            self.power
                .add(t + full as u64, 1, tail as f64 * self.rf_energy);
        }
        self.stats.rf_bytes += bytes as u64;
        bytes.div_ceil(bw) as u64 + self.hw.rf_latency()
    }
    // Blocking dialect: RF read, functional unit, and RF write are explicit
    // sequential stages. TC streams K inside one physical block, but blocks
    // and instructions do not overlap. This is a versioned conservative model.
    fn shared_service(&mut self, t: u64, batch: crate::shared::Batch) -> u64 {
        for (i, words) in batch.activity().enumerate() {
            self.stats.sh_bytes += words as u64 * 4;
            self.power
                .add(t + i as u64, 1, words as f64 * 4. * self.sh_energy);
        }
        self.stats.sh_bank_stall_cycles += batch.cycles().saturating_sub(1) as u64;
        batch.cycles() as u64 + self.hw.sh_latency()
    }
    fn compute(&mut self, t: u64, op: &Instr) -> u64 {
        let mut end = t + 1;
        match *op {
            Instr::SharedRead { shared, local } | Instr::SharedWrite { shared, local } => {
                let write = matches!(op, Instr::SharedWrite { .. });
                self.power.add(end, 6, 20.);
                end += 6;
                let mut source_free = end;
                let mut dest_free = end;
                // Six fixed memory-pipeline stages plus two DMA service slots.
                // Reserve a slot until destination completion: no unbounded tail.
                let mut slots = [end; 8];
                for start in (0..shared.len()).step_by(16) {
                    let count = (shared.len() - start).min(16);
                    let mut words = [0; 16];
                    for (i, word) in words[..count].iter_mut().enumerate() {
                        *word = shared.at(start + i);
                    }
                    let batch = crate::shared::batch(&words[..count], self.hw.sh_banks, write);
                    let slot = (start / 16) % slots.len();
                    let issue = source_free.max(slots[slot]);
                    let inflight = slots.iter().filter(|&&ready| ready > issue).count() + 1;
                    self.stats.max_sh_copy_inflight = self.stats.max_sh_copy_inflight.max(inflight);
                    self.power.add(issue, 1, count as f64 * 0.5);
                    if write {
                        let mut source = [0; 16];
                        let mut unique = 0;
                        for i in start..start + count {
                            let word = local.at(i);
                            if !source[..unique].contains(&word) {
                                source[unique] = word;
                                unique += 1;
                            }
                        }
                        let source_duration = self.rf(issue, unique * 4, false);
                        source_free = issue + source_duration - 2;
                        let dest_start = dest_free.max(issue + source_duration);
                        end = dest_start + self.shared_service(dest_start, batch);
                        dest_free = dest_start + batch.cycles() as u64;
                    } else {
                        let source_duration = self.shared_service(issue, batch);
                        source_free = issue + batch.cycles() as u64;
                        let dest_start = dest_free.max(issue + source_duration);
                        let dest_duration = self.rf(dest_start, count * 4, true);
                        end = dest_start + dest_duration;
                        dest_free = dest_start + dest_duration - 2;
                    }
                    slots[slot] = end;
                }
            }
            Instr::Fill { len, .. } => {
                end += self.rf(end, len * 4, true);
            }
            Instr::Vector {
                kind, len, a, b, ..
            } => {
                let unary = matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt | Op::Square);
                let sfu = matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt);
                let width = if sfu { self.hw.sfu } else { self.hw.vector };
                let energy = match kind {
                    Op::Exp | Op::Tanh => 12.,
                    Op::Rsqrt => 8.,
                    Op::Fma => 4.,
                    _ => 1.5,
                };
                for start in (0..len).step_by(width) {
                    let count = (len - start).min(width);
                    let bytes = |arg| match arg {
                        Arg::Imm { .. } => 0,
                        Arg::Reg { stride: 0, .. } => 4,
                        Arg::Reg { .. } => count * 4,
                    };
                    end += self.rf(
                        end,
                        bytes(a)
                            + if unary { 0 } else { bytes(b) }
                            + if matches!(kind, Op::Fma) {
                                count * 4
                            } else {
                                0
                            },
                        false,
                    );
                    self.power.add(end, 1, count as f64 * energy);
                    end += if sfu { 12 } else { 4 };
                    end += self.rf(end, count * 4, true);
                }
            }
            Instr::Reduce { len, .. } => {
                let mut n = len;
                if n == 1 {
                    end += self.rf(end, 4, false);
                    end += self.rf(end, 4, true);
                }
                while n > 1 {
                    let output = n.div_ceil(2);
                    for start in (0..output).step_by(self.hw.vector) {
                        let count = (output - start).min(self.hw.vector);
                        let inputs = (n - 2 * start).min(count * 2);
                        end += self.rf(end, inputs * 4, false);
                        self.power.add(end, 1, (inputs / 2) as f64 * 1.5);
                        end += 4;
                        end += self.rf(end, count * 4, true);
                    }
                    n = output;
                }
            }
            Instr::MmaShared { m, n, k, .. } => {
                for i in (0..m).step_by(self.hw.p) {
                    for j in (0..n).step_by(self.hw.q) {
                        let mm = (m - i).min(self.hw.p);
                        let nn = (n - j).min(self.hw.q);
                        end += self.rf(end, mm * nn * 4, false);
                        if cfg!(feature = "explore") {
                            let rb = mm * 4;
                            let sb = nn * 4;
                            let rs = rb.div_ceil(self.hw.read_bw()) as u64;
                            let ss = nn.div_ceil(self.hw.sh_banks) as u64;
                            let bs = sb.div_ceil(self.hw.sh_tc_bw) as u64;
                            let step = rs.max(ss).max(bs);
                            let interface_start = ss + self.hw.sh_latency();
                            let startup = (interface_start + bs + 2).max(rs + self.hw.rf_latency());
                            if step == 1 {
                                self.power.add(
                                    end,
                                    k as u64,
                                    sb as f64 * self.sh_energy * k as f64,
                                );
                                self.power.add(
                                    end,
                                    k as u64,
                                    rb as f64 * self.rf_energy * k as f64,
                                );
                                self.power.add(
                                    end + interface_start,
                                    k as u64,
                                    sb as f64 * 0.2 * k as f64,
                                );
                                self.power.add(
                                    end + startup,
                                    k as u64,
                                    (self.hw.p * self.hw.q * k) as f64 * 3.,
                                );
                            } else {
                                for kk in 0..k {
                                    let t = end + kk as u64 * step;
                                    self.byte_energy(t, sb, self.hw.sh_banks * 4, self.sh_energy);
                                    self.byte_energy(t, rb, self.hw.read_bw(), self.rf_energy);
                                    self.byte_energy(
                                        t + interface_start,
                                        sb,
                                        self.hw.sh_tc_bw,
                                        0.2,
                                    );
                                    self.power.add(
                                        t + startup,
                                        1,
                                        (self.hw.p * self.hw.q) as f64 * 3.,
                                    );
                                }
                            }
                            self.stats.rf_bytes += (rb * k) as u64;
                            self.stats.sh_bytes += (sb * k) as u64;
                            self.stats.sh_tc_bytes += (sb * k) as u64;
                            self.stats.sh_tc_stall_cycles += k as u64 * (step - 1);
                            self.stats.sh_bank_stall_cycles += k as u64 * (ss - 1);
                            self.stats.tc_physical_fmas += (self.hw.p * self.hw.q * k) as u64;
                            end += startup
                                + (k as u64 - 1) * step
                                + 1
                                + (self.hw.p + self.hw.q - 2) as u64;
                            end += self.rf(end, mm * nn * 4, true);
                            continue;
                        }
                        // Contiguous <=16 B words hit distinct banks (bank menu >=16).
                        // A+acc stay in RF. Both source pipelines reserve their data;
                        // the B interface throttles issue, with 6+2 cycles to TC.
                        let step = (nn * 4).div_ceil(self.hw.sh_tc_bw) as u64;
                        let rf_energy = (mm * 4) as f64 * self.rf_energy;
                        let sh_energy = (nn * 4) as f64 * self.sh_energy;
                        let interface_energy = (nn * 4) as f64 * 0.2;
                        let compute_energy = (self.hw.p * self.hw.q) as f64 * 3.;
                        // SH service(1)+latency(6), interface service(step)+latency(2).
                        // A is fetched just in time; no long-lived extra RF queue.
                        if step == 1 {
                            self.power.add(end, k as u64, sh_energy * k as f64);
                            self.power.add(
                                end + 7,
                                k as u64,
                                (rf_energy + interface_energy) * k as f64,
                            );
                            self.power
                                .add(end + 10, k as u64, compute_energy * k as f64);
                        } else {
                            for kk in 0..k {
                                let t = end + kk as u64 * step;
                                self.power.add(t, 1, sh_energy);
                                self.power.add(t + 6 + step, 1, rf_energy);
                                for fragment in 0..step {
                                    let bytes = (nn * 4 - fragment as usize * self.hw.sh_tc_bw)
                                        .min(self.hw.sh_tc_bw);
                                    self.power.add(t + 7 + fragment, 1, bytes as f64 * 0.2);
                                }
                                self.power.add(t + 9 + step, 1, compute_energy);
                            }
                        }
                        self.stats.rf_bytes += (mm * 4 * k) as u64;
                        self.stats.sh_bytes += (nn * 4 * k) as u64;
                        self.stats.sh_tc_bytes += (nn * 4 * k) as u64;
                        self.stats.sh_tc_stall_cycles += k as u64 * (step - 1);
                        self.stats.tc_physical_fmas += (self.hw.p * self.hw.q * k) as u64;
                        end += 10 + k as u64 * step + (self.hw.p + self.hw.q - 2) as u64;
                        end += self.rf(end, mm * nn * 4, true);
                    }
                }
            }
            Instr::Mma { m, n, k, .. } => {
                for i in (0..m).step_by(self.hw.p) {
                    for j in (0..n).step_by(self.hw.q) {
                        let mm = (m - i).min(self.hw.p);
                        let nn = (n - j).min(self.hw.q);
                        end += self.rf(end, mm * nn * 4, false);
                        let read_bytes = (mm + nn) * 4 * k;
                        if cfg!(feature = "explore") {
                            let step = ((mm + nn) * 4).div_ceil(self.hw.read_bw()) as u64;
                            let startup = step + self.hw.rf_latency();
                            if step == 1 {
                                self.power
                                    .add(end, k as u64, read_bytes as f64 * self.rf_energy);
                                self.power.add(
                                    end + startup,
                                    k as u64,
                                    (self.hw.p * self.hw.q * k) as f64 * 3.,
                                );
                            } else {
                                for kk in 0..k {
                                    let t = end + kk as u64 * step;
                                    self.byte_energy(
                                        t,
                                        (mm + nn) * 4,
                                        self.hw.read_bw(),
                                        self.rf_energy,
                                    );
                                    self.power.add(
                                        t + startup,
                                        1,
                                        (self.hw.p * self.hw.q) as f64 * 3.,
                                    );
                                }
                            }
                            self.stats.rf_bytes += read_bytes as u64;
                            self.stats.tc_physical_fmas += (self.hw.p * self.hw.q * k) as u64;
                            end += startup
                                + (k as u64 - 1) * step
                                + 1
                                + (self.hw.p + self.hw.q - 2) as u64;
                            end += self.rf(end, mm * nn * 4, true);
                            continue;
                        }
                        self.power
                            .add(end, k as u64, (read_bytes as f64) * self.rf_energy);
                        self.stats.rf_bytes += read_bytes as u64;
                        end += 2;
                        let fmas = self.hw.p * self.hw.q * k;
                        self.power.add(end, k as u64, fmas as f64 * 3.);
                        self.stats.tc_physical_fmas += fmas as u64;
                        end += (k + self.hw.p + self.hw.q - 2) as u64;
                        end += self.rf(end, mm * nn * 4, true);
                    }
                }
            }
            _ => unreachable!(),
        }
        end
    }
    pub(crate) fn wave(&mut self, wave: &Wave, bases: &[usize]) {
        let mut sms: Vec<Sm> = (0..self.hw.sms)
            .map(|_| Sm {
                pc: 0,
                ready: self.stats.cycles,
                local_free: self.stats.cycles,
                write_staged: 0,
                dma: None,
            })
            .collect();
        let mut events: BinaryHeap<Reverse<Scheduled>> = BinaryHeap::new();
        let mut up: Vec<VecDeque<Packet>> = (0..sms.len()).map(|_| VecDeque::new()).collect();
        let mut down = up.clone();
        let mut up_ready = 0u32;
        let mut down_ready = 0u32;
        let mut rr = [0, 0];
        let mut t = self.stats.cycles;
        loop {
            self.power.advance(t);
            while events.peek().is_some_and(|e| e.0.time <= t) {
                let e = events.pop().unwrap().0;
                match e.event {
                    Event::WriteReady(r) => {
                        up[r.sm].push_back(Packet {
                            req: r,
                            remaining: 72,
                        });
                        up_ready |= 1 << r.sm;
                    }
                    Event::Up(r) => {
                        let ch = &mut self.channels[r.channel];
                        if r.write {
                            ch.writes.push_back((t + 250, r));
                        } else {
                            ch.reads.push_back((t + 250, r));
                        }
                    }
                    Event::Hbm(r) => {
                        down[r.sm].push_back(Packet {
                            req: r,
                            remaining: if r.write { 8 } else { 64 },
                        });
                        down_ready |= 1 << r.sm;
                    }
                    Event::Down(r) => {
                        let duration = if r.write {
                            0
                        } else {
                            if let Some(batch) = r.shared {
                                self.shared_service(t, batch)
                            } else {
                                self.rf(t, r.valid * 4, true)
                            }
                        };
                        self.push(&mut events, t + duration, Event::Local(r));
                    }
                    Event::Local(r) => {
                        self.channels[r.channel].used -= 1;
                        sms[r.sm].dma.as_mut().unwrap().inflight -= 1;
                    }
                }
            }
            for sm in 0..sms.len() {
                let done = sms[sm].dma.as_ref().is_some_and(|d| {
                    d.next == d.global.len()
                        && d.front.is_empty()
                        && d.inflight == 0
                        && t >= d.start
                });
                if done {
                    sms[sm].dma = None;
                    sms[sm].ready = t;
                }
                if sms[sm].dma.is_none() && sms[sm].ready <= t && sms[sm].pc < wave[sm].len() {
                    let instr = &wave[sm][sms[sm].pc];
                    sms[sm].pc += 1;
                    self.stats.instructions += 1;
                    self.power.add(t, 1, 8.);
                    match *instr {
                        Instr::Load {
                            tensor,
                            global,
                            local,
                        }
                        | Instr::Store {
                            tensor,
                            global,
                            local,
                        }
                        | Instr::LoadShared {
                            tensor,
                            global,
                            local,
                        }
                        | Instr::StoreShared {
                            tensor,
                            global,
                            local,
                        } => {
                            let write =
                                matches!(instr, Instr::Store { .. } | Instr::StoreShared { .. });
                            let shared = matches!(
                                instr,
                                Instr::LoadShared { .. } | Instr::StoreShared { .. }
                            );
                            self.power.add(t + 1, 6, 20.);
                            sms[sm].dma = Some(Dma {
                                global,
                                local,
                                shared,
                                base: bases[tensor],
                                next: 0,
                                write,
                                front: VecDeque::new(),
                                inflight: 0,
                                start: t + 7,
                            });
                        }
                        _ => sms[sm].ready = self.compute(t, instr),
                    }
                }
                let mut write_staged = sms[sm].write_staged;
                let mut local_free = sms[sm].local_free;
                if let Some(d) = sms[sm].dma.as_mut() {
                    if t >= d.start && d.next < d.global.len() && d.front.len() <= 16 {
                        let (lines, used, count) = coalesce_batch(d.global, d.base, d.next);
                        for &(line, valid) in &lines[..used] {
                            d.front.push_back(Request {
                                sm,
                                channel: (line / 4) % self.hw.hbm_channels,
                                write: d.write,
                                valid,
                                shared: if d.shared {
                                    let mut words = [0; 16];
                                    let mut used_words = 0;
                                    for i in d.next..d.next + count {
                                        if (d.base + d.global.at(i) * 4) / 64 == line {
                                            words[used_words] = d.local.at(i);
                                            used_words += 1;
                                        }
                                    }
                                    Some(crate::shared::batch(
                                        &words[..used_words],
                                        self.hw.sh_banks,
                                        !d.write,
                                    ))
                                } else {
                                    None
                                },
                            });
                        }
                        self.power.add(t, 1, count as f64 * 0.5 + used as f64 * 2.);
                        d.next += count;
                        self.stats.valid_dma_bytes += (count * 4) as u64;
                        self.stats.max_frontend = self.stats.max_frontend.max(d.front.len());
                    }
                    if t >= d.start {
                        let mut source_bytes = 0;
                        for _ in 0..2 {
                            let Some(&r) = d.front.front() else { break };
                            let ch = &mut self.channels[r.channel];
                            if ch.used == self.hw.hbm_queue
                                || (r.write
                                    && (write_staged == 2
                                        || source_bytes + r.valid * 4 > 64
                                        || (r.shared.is_some() && local_free > t)))
                            {
                                break;
                            }
                            ch.used += 1;
                            d.inflight += 1;
                            d.front.pop_front();
                            if r.write {
                                write_staged += 1;
                                source_bytes += r.valid * 4;
                                let delay = if let Some(batch) = r.shared {
                                    local_free = t + batch.cycles() as u64;
                                    self.shared_service(t, batch)
                                } else {
                                    self.rf(t, r.valid * 4, false)
                                };
                                self.push(&mut events, t + delay, Event::WriteReady(r));
                            } else {
                                up[sm].push_back(Packet {
                                    req: r,
                                    remaining: 8,
                                });
                                up_ready |= 1 << sm;
                            }
                            self.power.add(t, 1, 8.);
                        }
                    }
                }
                sms[sm].write_staged = write_staged;
                sms[sm].local_free = local_free;
                self.stats.max_write_staged = self.stats.max_write_staged.max(write_staged);
            }
            for ch in 0..self.channels.len() {
                let c = &mut self.channels[ch];
                if c.free > t {
                    continue;
                }
                let read = c.reads.front().is_some_and(|r| r.0 <= t);
                let write = c.writes.front().is_some_and(|r| r.0 <= t);
                if !read && !write {
                    continue;
                }
                let dir = match (read, write, c.direction) {
                    (true, true, Some(last)) => {
                        let other = if last {
                            c.reads.front().unwrap().0
                        } else {
                            c.writes.front().unwrap().0
                        };
                        if c.burst >= 8 || t - other >= 256 {
                            !last
                        } else {
                            last
                        }
                    }
                    (true, _, _) => false,
                    _ => true,
                };
                let switched = c.direction.is_some_and(|last| last != dir);
                let (_, req) = if dir {
                    c.writes.pop_front().unwrap()
                } else {
                    c.reads.pop_front().unwrap()
                };
                if c.direction == Some(dir) {
                    c.burst += 1;
                } else {
                    c.burst = 1;
                }
                c.direction = Some(dir);
                c.free = t + if switched { 6 } else { 2 };
                let finish = c.free;
                if switched {
                    self.power.add(t, 4, 200.);
                    self.stats.direction_switches += 1;
                }
                self.power.add(finish - 2, 2, 64. * 150.);
                if dir {
                    self.stats.hbm_writes += 1;
                } else {
                    self.stats.hbm_reads += 1;
                }
                self.push(&mut events, finish, Event::Hbm(req));
            }
            for (direction, rr_start) in rr.iter_mut().enumerate() {
                let queues = if direction == 0 { &mut up } else { &mut down };
                let ready_mask = if direction == 0 {
                    &mut up_ready
                } else {
                    &mut down_ready
                };
                debug_assert_eq!(
                    *ready_mask,
                    queues.iter().enumerate().fold(0u32, |bits, (sm, q)| bits
                        | if q.is_empty() { 0 } else { 1 << sm })
                );
                let mut total = self.hw.global_noc;
                let mut served = false;
                let mut per_sm = [self.hw.sm_noc; 24];
                // Same round-robin order, but only inspect nonempty endpoints.
                let masks = [
                    *ready_mask & (u32::MAX << *rr_start),
                    *ready_mask & ((1u32 << *rr_start) - 1),
                ];
                for mut active in masks {
                    while active != 0 {
                        let sm = active.trailing_zeros() as usize;
                        active &= active - 1;
                        while total > 0 && per_sm[sm] > 0 {
                            let Some(p) = queues[sm].front_mut() else {
                                break;
                            };
                            let n = p.remaining.min(total).min(per_sm[sm]);
                            // Reserve RF reception before completing a return packet.
                            // No unbounded local queue of already-arrived line data.
                            if direction == 1
                                && !p.req.write
                                && n == p.remaining
                                && sms[sm].local_free > t + 7
                            {
                                break;
                            }

                            served = true;
                            p.remaining -= n;
                            total -= n;
                            per_sm[sm] -= n;
                            self.power.add(t, 1, n as f64 * 2.);
                            self.stats.noc_bytes += n as u64;
                            if p.remaining == 0 {
                                let r = queues[sm].pop_front().unwrap().req;
                                if queues[sm].is_empty() {
                                    *ready_mask &= !(1 << sm);
                                }
                                if direction == 1 && !r.write {
                                    sms[sm].local_free =
                                        t + 7 + r.shared.map_or(1, |b| b.cycles() as u64);
                                }
                                if direction == 0 && r.write {
                                    sms[sm].write_staged -= 1;
                                }

                                self.push(
                                    &mut events,
                                    t + 1 + 6,
                                    if direction == 0 {
                                        Event::Up(r)
                                    } else {
                                        Event::Down(r)
                                    },
                                );
                            } else {
                                break;
                            }
                        }
                    }
                }
                if served {
                    *rr_start += 1;
                    if *rr_start == sms.len() {
                        *rr_start = 0;
                    }
                }
            }
            let inflight = self.channels.iter().map(|c| c.used).sum();
            self.stats.max_hbm_inflight = self.stats.max_hbm_inflight.max(inflight);
            if sms
                .iter()
                .enumerate()
                .all(|(i, s)| s.pc == wave[i].len() && s.dma.is_none() && s.ready <= t)
            {
                break;
            }
            // Jump only when no network, request generator, or controller can
            // make progress before the next boundary. No history trace is stored.
            let network = up_ready != 0 || down_ready != 0;
            let mut next = events.peek().map_or(u64::MAX, |e| e.0.time);
            for (i, s) in sms.iter().enumerate() {
                if let Some(d) = &s.dma {
                    if (d.next < d.global.len() && d.front.len() <= 16)
                        || d.front.front().is_some_and(|r| {
                            self.channels[r.channel].used < self.hw.hbm_queue
                                && (!r.write || s.write_staged < 2)
                        })
                    {
                        next = next.min(d.start.max(t + 1));
                    }
                } else if s.pc < wave[i].len() || s.ready > t {
                    next = next.min(s.ready.max(t + 1));
                }
            }
            for c in &self.channels {
                for queue in [&c.reads, &c.writes] {
                    if let Some((ready, _)) = queue.front() {
                        next = next.min((*ready).max(c.free).max(t + 1));
                    }
                }
            }
            let next_time = if network || self.cycle_reference {
                t + 1
            } else {
                next.max(t + 1)
            };
            assert!(next_time < u64::MAX, "timing deadlock");
            self.stats.hbm_queue_full_sm_cycles += sms
                .iter()
                .filter_map(|s| s.dma.as_ref())
                .filter(|d| {
                    d.front
                        .front()
                        .is_some_and(|r| self.channels[r.channel].used == self.hw.hbm_queue)
                })
                .map(|d| next_time.saturating_sub(t.max(d.start)))
                .sum::<u64>();
            t = next_time;
        }
        self.power.add(t, 1, sms.len() as f64 * 8.); // explicit wave fence, one issue per SM
        self.stats.cycles = t + 1;
        self.stats.instructions += sms.len() as u64;
        self.stats.waves += 1;
    }
}

#[cfg(test)]
mod coalescing_tests {
    use super::*;
    #[test]
    fn affine_batches_match_independent_per_element_oracle() {
        // Include cross-row batches, padding, overlaps, reversed row progression
        // relative to the column span, and zero-stride broadcasts.
        for rows in [1, 2, 3, 17] {
            for cols in [1, 3, 8, 15, 16, 17, 32] {
                for row_stride in [0, 1, cols, cols + 7, 64] {
                    for col_stride in [0, 1, 2, 17, 64] {
                        for base in [0, 1, 15, 16, 63] {
                            let view = View {
                                base,
                                rows,
                                cols,
                                row_stride,
                                col_stride,
                            };
                            for next in (0..view.len()).step_by(16) {
                                let (mut oracle, count) =
                                    (Vec::<(usize, usize)>::new(), (view.len() - next).min(16));
                                for i in next..next + count {
                                    let line = (256 + view.at(i) * 4) / 64;
                                    if let Some(entry) =
                                        oracle.iter_mut().find(|entry| entry.0 == line)
                                    {
                                        entry.1 += 1;
                                    } else {
                                        oracle.push((line, 1));
                                    }
                                }
                                let (actual, n, k) = coalesce_batch(view, 256, next);
                                assert_eq!(k, count);
                                assert_eq!(&actual[..n], oracle.as_slice(), "{view:?} at {next}");
                            }
                        }
                    }
                }
            }
        }
        let view = View {
            base: 0,
            rows: 1,
            cols: 1,
            row_stride: usize::MAX,
            col_stride: usize::MAX,
        };
        assert_eq!(coalesce_batch(view, 0, 0).0[0], (0, 1));
    }
}

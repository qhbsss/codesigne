use crate::{
    Result,
    hardware::Hardware,
    isa::{Arg, Instr, Op, View, Wave},
    timing::{Stats, Timing},
};
use serde::Serialize;
use std::io::Write;

/// Bounded metadata for cross-group HBM race checks; never removes execution work.
pub const MAX_WAVE_ACCESS_INTERVALS: usize = 262_144;

#[cfg(feature = "compact")]
#[derive(Default)]
struct AccessRanges {
    buckets: std::collections::BTreeMap<
        (usize, usize, bool, usize),
        std::collections::BTreeMap<usize, usize>,
    >,
    count: usize,
}
#[cfg(feature = "compact")]
impl AccessRanges {
    fn insert(
        &mut self,
        tensor: usize,
        group: usize,
        write: bool,
        stride: usize,
        mut start: usize,
        mut end: usize,
    ) -> Result<()> {
        let stride = stride.max(1);
        let rows = self
            .buckets
            .entry((tensor, group, write, stride))
            .or_default();
        if stride == 1 {
            if let Some((&a, &b)) = rows.range(..=start).next_back()
                && b >= start
            {
                start = a;
                end = end.max(b);
                rows.remove(&a);
                self.count -= 1;
            }
            while let Some((&a, &b)) = rows.range(start..).next() {
                if a > end {
                    break;
                }
                end = end.max(b);
                rows.remove(&a);
                self.count -= 1;
            }
        } else if let Some(existing) = rows.get_mut(&start) {
            // Same start and stride: one progression is a prefix of the other.
            *existing = (*existing).max(end);
            return Ok(());
        }
        if self.count == MAX_WAVE_ACCESS_INTERVALS {
            return Err("wave distinct access intervals exceed 262144 after coalescing".into());
        }
        rows.insert(start, end);
        self.count += 1;
        Ok(())
    }
    fn into_ranges(self) -> Vec<(usize, usize, usize, usize, bool, usize)> {
        let mut out = Vec::with_capacity(self.count);
        for ((tensor, group, write, stride), rows) in self.buckets {
            for (start, end) in rows {
                out.push((tensor, start, end, group, write, stride));
            }
        }
        out
    }
}

pub struct Tensor {
    pub name: String,
    pub data: Vec<f32>,
}
pub struct Machine {
    pub tensors: Vec<Tensor>,
    pub timing: Timing,
    pub functional: bool,
    bases: Vec<usize>,
    allocated: usize,
    peak_allocated: usize,
    execution_valid: bool,
    recorder: Option<Box<dyn Write>>,
    pub k_parallel: usize,
    #[cfg(feature = "concurrent")]
    pub concurrent: Option<crate::v09_schedule::Scheduler>,
}
#[derive(Serialize)]
pub struct Report {
    pub model: &'static str,
    pub functional: bool,
    pub execution_valid: bool,
    pub area_au: f64,
    pub base_power_w: f64,
    pub short_power_w: f64,
    pub long_power_w: f64,
    pub energy_j: f64,
    pub power_pass: bool,
    pub hbm_allocated_bytes: usize,
    pub hbm_peak_allocated_bytes: usize,
    pub stats: Stats,
}
fn check_view(v: View, limit: usize) -> Result<()> {
    if v.rows == 0
        || v.cols == 0
        || v.rows.checked_mul(v.cols).is_none_or(|n| n > 16384)
        || v.end().is_none_or(|n| n > limit)
    {
        return Err("invalid/out-of-bounds view or DMA >64 KiB".into());
    }
    Ok(())
}
fn check_range(base: usize, len: usize, limit: usize) -> Result<()> {
    if len == 0
        || len
            > if cfg!(feature = "explore") {
                8192
            } else {
                4096
            }
        || base.checked_add(len).is_none_or(|end| end > limit)
    {
        return Err("invalid/out-of-bounds RF range".into());
    }
    Ok(())
}
// Called only after complete view/range validation. Unused physical capacity does
// not need a host allocation; every touched location is still reset at each wave/SM.
fn local_extent(ins: &Instr) -> (usize, usize) {
    match *ins {
        Instr::Load { local, .. } | Instr::Store { local, .. } => (local.end().unwrap(), 0),
        Instr::LoadShared { local, .. } | Instr::StoreShared { local, .. } => {
            (0, local.end().unwrap())
        }
        Instr::SharedRead { shared, local } | Instr::SharedWrite { shared, local } => {
            (local.end().unwrap(), shared.end().unwrap())
        }
        Instr::Fill { dst, len, .. } => (dst + len, 0),
        Instr::Vector { dst, len, a, b, .. } => {
            let extent = |a| match a {
                Arg::Imm { .. } => 0,
                Arg::Reg { base, stride } => base + (len - 1) * stride + 1,
            };
            ((dst + len).max(extent(a)).max(extent(b)), 0)
        }
        Instr::Reduce {
            dst,
            src,
            scratch,
            len,
            ..
        } => ((dst + 1).max(src + len).max(scratch + len), 0),
        Instr::Mma { a, b, c, m, n, k } => ((a + m * k).max(b + k * n).max(c + m * n), 0),
        Instr::MmaShared { a, b, c, m, n, k } => ((a + m * k).max(c + m * n), b.end().unwrap()),
    }
}
impl Machine {
    pub fn new(hw: Hardware, functional: bool) -> Result<Self> {
        hw.validate()?;
        Self::storage(hw, functional)
    }
    fn storage(hw: Hardware, functional: bool) -> Result<Self> {
        Ok(Self {
            tensors: vec![],
            timing: Timing::new(hw),
            functional,
            bases: vec![],
            allocated: 0,
            peak_allocated: 0,
            execution_valid: true,
            recorder: None,
            k_parallel: 1,
            #[cfg(feature = "concurrent")]
            concurrent: None,
        })
    }
    #[cfg(feature = "concurrent")]
    pub fn new_concurrent(
        hw: crate::v09_hardware::Hardware,
        functional: bool,
        groups: usize,
        asynchronous: bool,
    ) -> Result<Self> {
        hw.validate()?;
        if groups == 0 || groups > hw.resident_groups {
            return Err("invalid parallel_groups".into());
        }
        let mut m = Self::storage(hw.base.clone(), functional)?;
        m.k_parallel = hw.tc_k_parallel;
        m.timing.hw.sms *= groups;
        m.concurrent = Some(crate::v09_schedule::Scheduler::new(hw, asynchronous)?);
        Ok(m)
    }
    #[cfg(feature = "concurrent")]
    pub fn concurrent_wave(&mut self, groups: Vec<crate::v09_schedule::Group>) -> Result<()> {
        let scheduler = self
            .concurrent
            .as_ref()
            .ok_or("concurrent engine required")?;
        let wave = scheduler.validate_groups(&groups)?;
        self.functional_wave(wave)?;
        let result = self
            .concurrent
            .as_mut()
            .unwrap()
            .wave_prevalidated(&groups, &self.bases);
        if result.is_err() {
            self.execution_valid = false;
        }
        result
    }
    #[cfg(feature = "concurrent")]
    pub fn concurrent_report(&self) -> Result<crate::v09_schedule::Report> {
        if !self.execution_valid {
            return Err("machine invalid after failed execution".into());
        }
        Ok(self
            .concurrent
            .as_ref()
            .ok_or("concurrent engine required")?
            .report())
    }
    fn allocation_range(&self, len: usize) -> Result<(usize, usize)> {
        let bytes = len.checked_mul(4).ok_or("tensor overflow")?;
        let base = self.allocated.next_multiple_of(256);
        let end = base.checked_add(bytes).ok_or("HBM overflow")?;
        if end > 2 * 1024 * 1024 * 1024 {
            return Err("HBM capacity exceeds 2 GiB".into());
        }
        Ok((base, end))
    }
    /// Export a data-independent static program instead of simulating it.
    /// This is an offline generator facility, never enabled by the evaluator.
    pub fn record_to(&mut self, writer: Box<dyn Write>) {
        self.recorder = Some(writer);
    }
    pub fn finish_recording(&mut self) -> Result<()> {
        if let Some(mut writer) = self.recorder.take() {
            writer.flush().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn record(&mut self, record: &crate::submission::Record) -> Result<()> {
        if let Some(writer) = self.recorder.as_mut() {
            serde_json::to_writer(&mut **writer, record).map_err(|e| e.to_string())?;
            writer.write_all(b"\n").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn bind_input(&mut self, source: &str, data: &[f32], len: usize) -> Result<usize> {
        self.allocation_range(len)?;
        if data.len() > len {
            return Err("input exceeds allocation".into());
        }
        self.record(&crate::submission::Record::Input {
            source: source.into(),
            len,
        })?;
        let mut values = vec![f32::NAN; len];
        values[..data.len()].copy_from_slice(data);
        self.alloc_inner(source, values)
    }
    pub fn checkpoint(&mut self, outputs: crate::submission::Checkpoint) -> Result<()> {
        self.record(&crate::submission::Record::Commit { outputs })?;
        if self.recorder.is_none() {
            self.wave(vec![vec![]; self.timing.hw.sms])?;
        }
        Ok(())
    }
    pub fn alloc(&mut self, name: &str, data: Vec<f32>) -> Result<usize> {
        self.record(&crate::submission::Record::Alloc { len: data.len() })?;
        self.alloc_inner(name, data)
    }
    fn alloc_inner(&mut self, name: &str, data: Vec<f32>) -> Result<usize> {
        let (base, end) = self.allocation_range(data.len())?;
        let id = self.tensors.len();
        self.bases.push(base);
        self.allocated = end;
        self.peak_allocated = self.peak_allocated.max(end);
        self.tensors.push(Tensor {
            name: name.into(),
            data,
        });
        Ok(id)
    }
    pub fn empty(&mut self, name: &str, len: usize) -> Result<usize> {
        self.allocation_range(len)?;
        self.alloc(name, vec![f32::NAN; len])
    }
    /// Stack scratch release at a completed global wave barrier. Reuses addresses
    /// only after all DMA notifications and compute have completed.
    pub fn release_last(&mut self, tensor: usize) -> Result<()> {
        if !self.execution_valid {
            return Err("cannot release scratch on invalid machine".into());
        }
        if self.tensors.is_empty() || tensor != self.tensors.len() - 1 {
            return Err("scratch release must follow reverse allocation order".into());
        }
        #[cfg(feature = "concurrent")]
        if let Some(scheduler) = self.concurrent.as_mut() {
            let base = self.bases[tensor] as u64;
            scheduler
                .memory
                .invalidate_range(base, base + self.tensors[tensor].data.len() as u64 * 4)?;
        }
        self.record(&crate::submission::Record::Release { tensor })?;
        self.tensors.pop();
        self.bases.pop();
        self.allocated = self
            .tensors
            .last()
            .zip(self.bases.last())
            .map_or(0, |(t, base)| base + t.data.len() * 4);
        Ok(())
    }
    pub fn reset(&mut self) {
        let hw = self.timing.hw.clone();
        let cycle_reference = self.timing.cycle_reference;
        self.timing = Timing::new(hw);
        self.timing.cycle_reference = cycle_reference;
        self.recorder = None;
        #[cfg(feature = "concurrent")]
        if let Some(old) = self.concurrent.take() {
            self.k_parallel = old.hardware.tc_k_parallel;
            self.concurrent = Some(
                crate::v09_schedule::Scheduler::new(old.hardware, old.async_native)
                    .expect("previously validated hardware"),
            );
        }
        self.tensors.clear();
        self.bases.clear();
        self.allocated = 0;
        self.peak_allocated = 0;
        self.execution_valid = true;
    }
    pub fn wave(&mut self, wave: Wave) -> Result<()> {
        self.wave_inner(wave, true)
    }
    /// Functional validation of independent workgroup streams, without old timing.
    #[cfg(feature = "concurrent")]
    pub fn functional_wave(&mut self, wave: Wave) -> Result<()> {
        if wave.len() > 256 {
            return Err("at most 256 workgroups per wave".into());
        }
        self.wave_inner(wave, false)
    }
    fn wave_inner(&mut self, wave: Wave, timed: bool) -> Result<()> {
        if self.recorder.is_some() {
            return self.record(&crate::submission::Record::Wave { streams: wave });
        }
        if !self.execution_valid {
            return Err("machine invalid after execution error; reset before reuse".into());
        }
        if timed && wave.len() != self.timing.hw.sms {
            return Err("wave must provide one instruction stream per SM".into());
        }
        if wave.iter().map(Vec::len).sum::<usize>() > crate::MAX_WAVE_PRIMITIVES {
            return Err("wave exceeds instruction admission limit".into());
        }
        #[cfg(feature = "concurrent")]
        let concurrent_groups = if timed && let Some(scheduler) = self.concurrent.as_ref() {
            let groups = crate::v09_schedule::native_groups(
                &wave,
                &scheduler.hardware,
                scheduler.async_native,
            )?;
            scheduler.validate_groups(&groups)?;
            Some(groups)
        } else {
            None
        };
        let limit = self.timing.hw.rf_kib * 256;
        // Per-row access intervals are discarded at the wave barrier. No global
        // per-address access history, and no dependence on textual SM order.
        let mut written = vec![false; self.tensors.len()];
        for ins in wave.iter().flatten() {
            if let Instr::Store { tensor, .. } | Instr::StoreShared { tensor, .. } = ins {
                *written.get_mut(*tensor).ok_or("unknown tensor")? = true;
            }
        }
        #[cfg(not(feature = "compact"))]
        let mut ranges = Vec::new();
        #[cfg(feature = "compact")]
        let mut compact_ranges = AccessRanges::default();
        #[cfg(feature = "compact")]
        let needs_race_check = wave.iter().filter(|program| !program.is_empty()).count() > 1;
        for (sm, program) in wave.iter().enumerate() {
            for ins in program {
                match *ins {
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
                        let shared =
                            matches!(ins, Instr::LoadShared { .. } | Instr::StoreShared { .. });
                        let local_limit = if shared {
                            self.timing.hw.sh_kib * 256
                        } else {
                            limit
                        };
                        let tensor_len =
                            self.tensors.get(tensor).ok_or("unknown tensor")?.data.len();
                        check_view(global, tensor_len)?;
                        check_view(local, local_limit)?;
                        if (global.rows, global.cols) != (local.rows, local.cols) {
                            return Err("DMA shape mismatch".into());
                        }
                        let write = matches!(ins, Instr::Store { .. } | Instr::StoreShared { .. });
                        let dest = if write { global } else { local };
                        if dest.col_stride != 1 || dest.row_stride < dest.cols {
                            return Err("DMA destination must have disjoint contiguous rows".into());
                        }
                        if !written[tensor] {
                            continue;
                        }
                        #[cfg(feature = "compact")]
                        if !needs_race_check {
                            continue;
                        }
                        #[cfg(not(feature = "compact"))]
                        if ranges.len() + global.rows > MAX_WAVE_ACCESS_INTERVALS {
                            return Err(
                                "wave access intervals exceed 262144; split at a barrier".into()
                            );
                        }
                        for row in 0..global.rows {
                            let start = global.base + row * global.row_stride;
                            let end = start + (global.cols - 1) * global.col_stride + 1;
                            #[cfg(feature = "compact")]
                            compact_ranges.insert(
                                tensor,
                                sm,
                                write,
                                global.col_stride,
                                start,
                                end,
                            )?;
                            #[cfg(not(feature = "compact"))]
                            ranges.push((tensor, start, end, sm, write, global.col_stride));
                        }
                    }
                    Instr::SharedRead { shared, local } | Instr::SharedWrite { shared, local } => {
                        check_view(shared, self.timing.hw.sh_kib * 256)?;
                        check_view(local, limit)?;
                        if (shared.rows, shared.cols) != (local.rows, local.cols) {
                            return Err("SH/RF shape mismatch".into());
                        }
                        let dest = if matches!(ins, Instr::SharedWrite { .. }) {
                            shared
                        } else {
                            local
                        };
                        if dest.col_stride != 1 || dest.row_stride < dest.cols {
                            return Err(
                                "local destination must have disjoint contiguous rows".into()
                            );
                        }
                    }
                    Instr::Fill { dst, len, value } => {
                        check_range(dst, len, limit)?;
                        if !value.is_finite() {
                            return Err("non-finite immediate".into());
                        }
                    }
                    Instr::Vector { dst, len, a, b, .. } => {
                        check_range(dst, len, limit)?;
                        for arg in [a, b] {
                            match arg {
                                Arg::Reg { base, stride } => {
                                    if base
                                        .checked_add(
                                            (len - 1).checked_mul(stride).ok_or("RF overflow")?,
                                        )
                                        .is_none_or(|end| end >= limit)
                                    {
                                        return Err("RF operand out of bounds".into());
                                    }
                                    if !(base == dst && (stride == 1 || len == 1))
                                        && (0..len).any(|i| {
                                            base + i * stride >= dst
                                                && base + i * stride < dst + len
                                        })
                                    {
                                        return Err(
                                            "vector partial alias; use disjoint RF scratch".into(),
                                        );
                                    }
                                }
                                Arg::Imm { value } => {
                                    if !value.is_finite() {
                                        return Err("non-finite immediate".into());
                                    }
                                }
                            }
                        }
                    }
                    Instr::Reduce {
                        dst,
                        src,
                        len,
                        scratch,
                        ..
                    } => {
                        check_range(dst, 1, limit)?;
                        check_range(src, len, limit)?;
                        check_range(scratch, len, limit)?;
                        if (scratch < src + len && src < scratch + len)
                            || (dst >= scratch && dst < scratch + len)
                            || (dst >= src && dst < src + len)
                        {
                            return Err("reduction source/output/scratch overlap".into());
                        }
                    }
                    Instr::MmaShared { a, b, c, m, n, k } => {
                        if self.timing.hw.p == 0
                            || self.timing.hw.sh_tc_bw == 0
                            || m == 0
                            || n == 0
                            || k == 0
                            || m > 64
                            || n > 64
                            || k > 256
                        {
                            return Err("invalid/disabled shared MMA".into());
                        }
                        check_view(b, self.timing.hw.sh_kib * 256)?;
                        if b.rows != k
                            || b.cols != n
                            || b.col_stride != 1
                            || b.row_stride < n
                            || b.base % 16 != 0
                        {
                            return Err("shared MMA B must be aligned row-major view".into());
                        }
                        for (base, len) in [(a, m * k), (c, m * n)] {
                            if base.checked_add(len).is_none_or(|e| e > limit) {
                                return Err("shared MMA RF overflow".into());
                            }
                        }
                        if a < c + m * n && c < a + m * k {
                            return Err("shared MMA accumulator aliases A".into());
                        }
                    }
                    Instr::Mma { a, b, c, m, n, k } => {
                        if self.timing.hw.p == 0
                            || m == 0
                            || n == 0
                            || k == 0
                            || m > 64
                            || n > 64
                            || k > 256
                        {
                            return Err("invalid MMA shape".into());
                        }
                        for (base, len) in [(a, m * k), (b, k * n), (c, m * n)] {
                            if base.checked_add(len).is_none_or(|end| end > limit) {
                                return Err("MMA RF overflow".into());
                            }
                        }
                        if (a < c + m * n && c < a + m * k) || (b < c + m * n && c < b + k * n) {
                            return Err("MMA accumulator aliases input".into());
                        }
                    }
                }
            }
        }
        #[cfg(feature = "compact")]
        let mut ranges = compact_ranges.into_ranges();
        ranges.sort_unstable();
        for group in ranges.chunk_by(|a, b| a.0 == b.0) {
            if group.iter().any(|r| !r.4 && r.5 > 1) {
                // Only strided reads need exact arithmetic-progression checks.
                // Writes have contiguous rows, merged per SM without word sets.
                let mut unions: Vec<Vec<(usize, usize)>> = vec![vec![]; wave.len()];
                for &(_, start, end, sm, write, _) in group {
                    if write {
                        if let Some(last) = unions[sm].last_mut()
                            && last.1 >= start
                        {
                            last.1 = last.1.max(end);
                        } else {
                            unions[sm].push((start, end));
                        }
                    }
                }
                for &(_, start, end, sm, _, stride) in group {
                    for (other, intervals) in unions.iter().enumerate() {
                        if other == sm {
                            continue;
                        }
                        let lo = intervals.partition_point(|&(_, b)| b <= start);
                        let hi = intervals.partition_point(|&(a, _)| a < end);
                        let candidates = &intervals[lo..hi];
                        let step = stride.max(1);
                        let points = (end - 1 - start) / step + 1;
                        let overlap = if points < candidates.len() {
                            (0..points).any(|i| {
                                let x = start + i * step;
                                let j = candidates.partition_point(|&(_, b)| b <= x);
                                j < candidates.len() && candidates[j].0 <= x
                            })
                        } else {
                            candidates.iter().any(|&(a, b)| {
                                let offset = a.saturating_sub(start).div_ceil(step);
                                // Compare the index before multiplying to avoid overflow.
                                offset < points && start + offset * step < b.min(end)
                            })
                        };
                        if overlap {
                            return Err("cross-SM HBM race: insert a wave barrier".into());
                        }
                    }
                }
            } else {
                let mut reads = vec![0; wave.len()];
                let mut writes = reads.clone();
                for &(_, start, end, sm, write, _) in group {
                    for other in 0..wave.len() {
                        if other != sm && (writes[other] > start || (write && reads[other] > start))
                        {
                            return Err("cross-SM HBM race: insert a wave barrier".into());
                        }
                    }
                    if write {
                        writes[sm] = writes[sm].max(end);
                    } else {
                        reads[sm] = reads[sm].max(end);
                    }
                }
            }
        }
        // Numerical execution is independent of timing. Barrier contract makes
        // SM order irrelevant; RF values cannot cross wave lifetimes.
        if self.functional && wave.iter().any(|p| !p.is_empty()) {
            let mut rf = Vec::new();
            let mut sh = Vec::new();
            for program in wave.iter().filter(|p| !p.is_empty()) {
                let (rf_len, sh_len) = program
                    .iter()
                    .map(local_extent)
                    .fold((0, 0), |(r, s), (rr, ss)| (r.max(rr), s.max(ss)));
                rf.resize(rf_len, f32::NAN);
                sh.resize(sh_len, f32::NAN);
                rf.fill(f32::NAN);
                sh.fill(f32::NAN);
                for ins in program {
                    if let Err(error) = self.execute(ins, &mut rf, &mut sh) {
                        self.execution_valid = false;
                        return Err(error);
                    }
                }
            }
        }
        if timed {
            #[cfg(feature = "concurrent")]
            if let Some(scheduler) = self.concurrent.as_mut() {
                if let Err(e) =
                    scheduler.wave_prevalidated(concurrent_groups.as_ref().unwrap(), &self.bases)
                {
                    self.execution_valid = false;
                    return Err(e);
                }
                return Ok(());
            }
            self.timing.wave(&wave, &self.bases);
        }
        Ok(())
    }
    fn execute(&mut self, ins: &Instr, rf: &mut [f32], sh: &mut [f32]) -> Result<()> {
        fn value(rf: &[f32], arg: Arg, i: usize) -> Result<f32> {
            let x = match arg {
                Arg::Imm { value } => value,
                Arg::Reg { base, stride } => rf[base + i * stride],
            };
            if x.is_finite() {
                Ok(x)
            } else {
                Err("uninitialized or non-finite RF read".into())
            }
        }
        match *ins {
            Instr::Load {
                tensor,
                global,
                local,
            }
            | Instr::LoadShared {
                tensor,
                global,
                local,
            } => {
                let target = if matches!(ins, Instr::LoadShared { .. }) {
                    &mut *sh
                } else {
                    &mut *rf
                };
                for i in 0..global.len() {
                    let x = self.tensors[tensor].data[global.at(i)];
                    if !x.is_finite() {
                        return Err(format!(
                            "uninitialized HBM read: {}[{}]",
                            self.tensors[tensor].name,
                            global.at(i)
                        ));
                    }
                    target[local.at(i)] = x;
                }
            }
            Instr::Store {
                tensor,
                global,
                local,
            }
            | Instr::StoreShared {
                tensor,
                global,
                local,
            } => {
                let source = if matches!(ins, Instr::StoreShared { .. }) {
                    &*sh
                } else {
                    &*rf
                };
                for i in 0..global.len() {
                    let x = source[local.at(i)];
                    if !x.is_finite() {
                        return Err("uninitialized local store".into());
                    }
                    self.tensors[tensor].data[global.at(i)] = x;
                }
            }
            Instr::SharedRead { shared, local } | Instr::SharedWrite { shared, local } => {
                let write = matches!(ins, Instr::SharedWrite { .. });
                for i in 0..shared.len() {
                    let x = if write {
                        rf[local.at(i)]
                    } else {
                        sh[shared.at(i)]
                    };
                    if !x.is_finite() {
                        return Err("uninitialized local copy source".into());
                    }
                    if write {
                        sh[shared.at(i)] = x;
                    } else {
                        rf[local.at(i)] = x;
                    }
                }
            }
            Instr::Fill { dst, len, value } => rf[dst..dst + len].fill(value),
            Instr::Vector {
                kind,
                dst,
                len,
                a,
                b,
            } => {
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let x = value(rf, a, i)?;
                    let unary = matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt | Op::Square);
                    let y = if unary { 0. } else { value(rf, b, i)? };
                    let z = match kind {
                        Op::Add => x + y,
                        Op::Fma => x.mul_add(y, value(rf, Arg::reg(dst), i)?),
                        Op::Sub => x - y,
                        Op::Mul => x * y,
                        Op::Div => x / y,
                        Op::Exp => x.exp(),
                        Op::Tanh => x.tanh(),
                        Op::Rsqrt => 1. / x.sqrt(),
                        Op::Square => x * x,
                    };
                    if !z.is_finite() {
                        return Err("non-finite vector result".into());
                    }
                    out.push(z);
                }
                rf[dst..dst + len].copy_from_slice(&out);
            }
            Instr::Reduce {
                dst,
                src,
                len,
                max,
                scratch,
            } => {
                let mut work = rf[src..src + len].to_vec();
                if work.iter().any(|x| !x.is_finite()) {
                    return Err("uninitialized reduction input".into());
                }
                let mut n = len;
                while n > 1 {
                    for j in 0..n / 2 {
                        work[j] = if max {
                            work[j * 2].max(work[j * 2 + 1])
                        } else {
                            work[j * 2] + work[j * 2 + 1]
                        };
                    }
                    if n % 2 == 1 {
                        work[n / 2] = work[n - 1];
                    }
                    n = n.div_ceil(2);
                    if n > 1 {
                        rf[scratch..scratch + n].copy_from_slice(&work[..n]);
                    }
                }
                if !work[0].is_finite() {
                    return Err("non-finite reduction result".into());
                }
                rf[dst] = work[0];
            }
            Instr::Mma { a, c, m, n, k, .. } | Instr::MmaShared { a, c, m, n, k, .. } => {
                let aa = rf[a..a + m * k].to_vec();
                let bb = match *ins {
                    Instr::Mma { b, .. } => rf[b..b + k * n].to_vec(),
                    Instr::MmaShared { b, .. } => (0..b.len()).map(|i| sh[b.at(i)]).collect(),
                    _ => unreachable!(),
                };
                let cc = &mut rf[c..c + m * n];
                if aa
                    .iter()
                    .chain(&bb)
                    .chain(cc.iter())
                    .any(|x| !x.is_finite())
                {
                    return Err("uninitialized MMA operand".into());
                }
                // Same ascending-K FP32 fused accumulation for both input paths.
                for (crow, arow) in cc.chunks_exact_mut(n).zip(aa.chunks_exact(k)) {
                    if self.k_parallel == 1 {
                        for (&x, brow) in arow.iter().zip(bb.chunks_exact(n)) {
                            for (dst, &y) in crow.iter_mut().zip(brow) {
                                *dst = x.mul_add(y, *dst);
                            }
                        }
                    } else {
                        for start in (0..k).step_by(self.k_parallel) {
                            for (j, dst) in crow.iter_mut().enumerate() {
                                let mut sums = [0f32; 4];
                                for (offset, value) in
                                    sums.iter_mut().take(self.k_parallel).enumerate()
                                {
                                    if start + offset < k {
                                        *value =
                                            arow[start + offset] * bb[(start + offset) * n + j];
                                    }
                                }
                                let mut count = self.k_parallel;
                                while count > 1 {
                                    for x in 0..count / 2 {
                                        sums[x] = sums[2 * x] + sums[2 * x + 1];
                                    }
                                    count /= 2;
                                }
                                *dst += sums[0];
                            }
                        }
                    }
                }
                if cc.iter().any(|x| !x.is_finite()) {
                    return Err("non-finite MMA result".into());
                }
            }
        }
        Ok(())
    }
    #[cfg(feature = "concurrent")]
    pub fn tensor_bases(&self) -> &[usize] {
        &self.bases
    }
    #[cfg(feature = "concurrent")]
    pub fn hbm_usage(&self) -> (usize, usize) {
        (self.allocated, self.peak_allocated)
    }
    pub fn report(&mut self) -> Report {
        let hw = &self.timing.hw;
        let base = hw.base_power();
        let cycles = self.timing.stats.cycles;
        let (short, long) = self.timing.power.finish(cycles, base);
        self.timing.stats.max_power_events = self.timing.power.max_pending;
        Report {
            model: crate::MODEL,
            functional: self.functional,
            execution_valid: self.execution_valid,
            area_au: hw.area(),
            base_power_w: base,
            short_power_w: short,
            long_power_w: long,
            energy_j: self.timing.power.dynamic_pj * 1e-12 + base * cycles as f64 * 2e-9,
            power_pass: self.execution_valid && short <= 34. && long <= 26.,
            hbm_allocated_bytes: self.allocated,
            hbm_peak_allocated_bytes: self.peak_allocated,
            stats: self.timing.stats.clone(),
        }
    }
}

#[cfg(all(test, feature = "compact"))]
mod access_range_tests {
    use super::*;

    #[test]
    fn repeated_stores_and_adjacent_rows_do_not_spend_metadata_budget() {
        let mut ranges = AccessRanges::default();
        for _ in 0..MAX_WAVE_ACCESS_INTERVALS + 1 {
            ranges.insert(0, 0, true, 1, 0, 1).unwrap();
        }
        ranges.insert(0, 0, true, 1, 2, 3).unwrap();
        ranges.insert(0, 0, true, 1, 1, 2).unwrap();
        ranges.insert(0, 1, false, 1, 0, 3).unwrap();
        assert_eq!(ranges.count, 2);
        assert_eq!(
            ranges.into_ranges(),
            vec![(0, 0, 3, 0, true, 1), (0, 0, 3, 1, false, 1)]
        );
    }

    #[test]
    fn coalescing_preserves_exact_addresses_and_group_access_kind() {
        use std::collections::BTreeSet;
        let mut expected = BTreeSet::new();
        let mut ranges = AccessRanges::default();
        let mut state = 71u64;
        for _ in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let start = (state as usize >> 8) % 128;
            let stride = (state as usize >> 16) % 4;
            let len = (state as usize >> 24) % 8 + 1;
            let end = start + (len - 1) * stride + 1;
            let tensor = (state as usize >> 32) % 3;
            let group = (state as usize >> 34) % 3;
            let write = state & 1 == 1;
            ranges
                .insert(tensor, group, write, stride, start, end)
                .unwrap();
            for x in (start..end).step_by(stride.max(1)) {
                expected.insert((tensor, group, write, x));
            }
        }
        let actual: BTreeSet<_> = ranges
            .into_ranges()
            .into_iter()
            .flat_map(|(t, a, b, g, w, s)| (a..b).step_by(s).map(move |x| (t, g, w, x)))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn fragmented_metadata_limit_is_checked_after_merging() {
        let mut ranges = AccessRanges::default();
        for i in 0..MAX_WAVE_ACCESS_INTERVALS {
            ranges.insert(0, 0, true, 1, 2 * i, 2 * i + 1).unwrap();
        }
        assert!(
            ranges
                .insert(
                    0,
                    0,
                    true,
                    1,
                    2 * MAX_WAVE_ACCESS_INTERVALS,
                    2 * MAX_WAVE_ACCESS_INTERVALS + 1
                )
                .unwrap_err()
                .contains("after coalescing")
        );
        ranges
            .insert(0, 0, true, 1, 0, 2 * MAX_WAVE_ACCESS_INTERVALS + 1)
            .unwrap();
        assert_eq!(ranges.count, 1);
    }
}

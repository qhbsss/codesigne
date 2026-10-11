//! Versioned concurrent hardware. All energies are pJ, time is 500 MHz cycles.
//! Compute uses explicit operand-read, arithmetic, and result-write stages.
#[cfg(test)]
use crate::isa::View;
use crate::{
    Result, hardware,
    isa::{Arg, Instr, Op},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Hardware {
    #[serde(flatten)]
    pub base: hardware::Hardware,
    pub tc_count: usize,
    pub tc_k_parallel: usize,
    pub reduction_units: usize,
    pub shared_ports: usize,
    pub dma_depth: usize,
    pub dma_engines: usize,
    pub multicast: bool,
    pub cache_mib: usize,
    pub resident_groups: usize,
}
impl Default for Hardware {
    fn default() -> Self {
        Self {
            base: hardware::Hardware::default(),
            tc_count: 1,
            tc_k_parallel: 1,
            reduction_units: 0,
            shared_ports: 1,
            dma_depth: 0,
            dma_engines: 1,
            multicast: false,
            cache_mib: 0,
            resident_groups: 1,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Engine {
    Vector,
    Sfu,
    Reduction,
    Tc,
    Copy,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct ComputeStep {
    pub sh_read_service: Vec<u64>,
    pub sh_write_service: Vec<u64>,
    pub rf_read_bytes: u64,
    pub rf_write_bytes: u64,
    pub sh_read_bytes: u64,
    pub sh_write_bytes: u64,
    pub sh_read_cycles: u64,
    pub sh_write_cycles: u64,
    pub sh_tc_bytes: u64,
    pub core_duration: u64,
    pub core_energy_pj: f64,
    pub physical_fmas: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct ComputeProfile {
    pub engine: Engine,
    pub steps: Vec<ComputeStep>,
    pub rf_read_bytes: u64,
    pub rf_write_bytes: u64,
    pub sh_read_bytes: u64,
    pub sh_write_bytes: u64,
    pub sh_read_cycles: u64,
    pub sh_write_cycles: u64,
    pub sh_tc_bytes: u64,
    pub core_duration: u64,
    pub core_energy_pj: f64,
    pub physical_fmas: u64,
}
impl ComputeProfile {
    fn new(engine: Engine) -> Self {
        Self {
            engine,
            steps: Vec::new(),
            rf_read_bytes: 0,
            rf_write_bytes: 0,
            sh_read_bytes: 0,
            sh_write_bytes: 0,
            sh_read_cycles: 0,
            sh_write_cycles: 0,
            sh_tc_bytes: 0,
            core_duration: 0,
            core_energy_pj: 0.,
            physical_fmas: 0,
        }
    }
    fn push(&mut self, s: ComputeStep) {
        self.rf_read_bytes += s.rf_read_bytes;
        self.rf_write_bytes += s.rf_write_bytes;
        self.sh_read_bytes += s.sh_read_bytes;
        self.sh_write_bytes += s.sh_write_bytes;
        self.sh_read_cycles += s.sh_read_cycles;
        self.sh_write_cycles += s.sh_write_cycles;
        self.sh_tc_bytes += s.sh_tc_bytes;
        self.core_duration += s.core_duration;
        self.core_energy_pj += s.core_energy_pj;
        self.physical_fmas += s.physical_fmas;
        self.steps.push(s);
    }
}
impl Hardware {
    pub fn validate(&self) -> Result<()> {
        let b = &self.base;
        for (name, value, choices) in [
            ("sms", b.sms, &[2, 4, 8, 12, 16, 20, 24, 28, 32][..]),
            ("rf_kib", b.rf_kib, &[16, 32, 64, 128, 256]),
            ("rf_tier", b.rf_tier, &[1, 2, 3]),
            ("vector", b.vector, &[8, 16, 32, 64, 128]),
            ("hbm_channels", b.hbm_channels, &[1, 2, 4, 8]),
            ("hbm_queue", b.hbm_queue, &[32, 64, 128, 256, 512]),
            ("sm_noc", b.sm_noc, &[16, 32, 64, 128, 256]),
            ("global_noc", b.global_noc, &[64, 128, 256, 512]),
            (
                "sh_kib",
                b.sh_kib,
                &[0, 32, 64, 96, 128, 192, 256, 512, 1024],
            ),
            ("sh_tc_bw", b.sh_tc_bw, &[0, 32, 64, 128, 256]),
        ] {
            if !choices.contains(&value) {
                return Err(format!("invalid v0.9 {name}: {value}"));
            }
        }
        if b.sfu == 0 || b.sfu > b.vector {
            return Err("SFU lanes must lie in 1..=vector".into());
        }
        let dims = [1, 2, 4, 8, 16, 32, 64];
        if !((b.p == 0 && b.q == 0)
            || (dims.contains(&b.p) && dims.contains(&b.q) && (16..=512).contains(&(b.p * b.q))))
        {
            return Err("invalid TC shape".into());
        }
        if (b.sh_kib == 0 && (b.sh_banks != 0 || b.sh_tc_bw != 0))
            || (b.sh_kib > 0
                && (![1, 2, 4, 8, 16, 32, 64].contains(&b.sh_banks) || b.sh_kib < b.sh_banks))
            || (b.p == 0 && b.sh_tc_bw != 0)
        {
            return Err("invalid SH configuration".into());
        }
        for (name, value, choices) in [
            ("tc_count", self.tc_count, &[0, 1, 2, 3, 4, 6, 8][..]),
            ("tc_k_parallel", self.tc_k_parallel, &[1, 2, 4]),
            ("reduction_units", self.reduction_units, &[0, 1, 2]),
            ("shared_ports", self.shared_ports, &[1, 2]),
            ("dma_depth", self.dma_depth, &[0, 1, 2]),
            ("dma_engines", self.dma_engines, &[1, 2, 4]),
            ("cache_mib", self.cache_mib, &[0, 1, 2, 4, 8, 16]),
            ("resident_groups", self.resident_groups, &[1, 2, 4, 8]),
        ] {
            if !choices.contains(&value) {
                return Err(format!("invalid v0.9 {name}: {value}"));
            }
        }
        if (self.tc_count == 0) != (self.base.p == 0 && self.base.q == 0)
            || (self.tc_count == 0 && self.tc_k_parallel != 1)
        {
            return Err(
                "TC count/shape mismatch; disabled TC requires p=q=0 and K parallel=1".into(),
            );
        }
        if self.base.sh_kib == 0 && self.shared_ports != 1 {
            return Err("disabled SH requires shared_ports=1".into());
        }
        if self.area() > 100. {
            return Err(format!("area {:.3} exceeds 100 AU", self.area()));
        }
        Ok(())
    }
    pub fn tc_area_per_engine(&self) -> f64 {
        if self.tc_count == 0 {
            return 0.;
        }
        0.072
            + 0.008
                * ((self.base.p * self.base.q + self.base.p + self.base.q) * self.tc_k_parallel)
                    as f64
    }
    pub fn memory_pending_limit(&self) -> usize {
        128
    }
    pub fn cache_bandwidth(&self) -> usize {
        if self.cache_mib == 0 { 0 } else { 64 }
    }
    pub fn cache_mshrs(&self) -> usize {
        if self.cache_mib == 0 { 0 } else { 256 }
    }
    pub fn cache_latency(&self) -> u64 {
        if self.cache_mib == 0 {
            0
        } else {
            20 + 2 * self.cache_mib.ilog2() as u64
        }
    }
    pub fn cache_energy_per_byte(&self) -> f64 {
        if self.cache_mib == 0 {
            0.
        } else {
            2.5 + 0.15 * self.cache_mib.ilog2() as f64
        }
    }
    pub fn area(&self) -> f64 {
        let b = &self.base;
        let old_tc = if b.p == 0 {
            0.
        } else {
            0.072 + 0.008 * (b.p * b.q + b.p + b.q) as f64
        };
        let extra_sh = (self.shared_ports.saturating_sub(1)) as f64
            * (0.0015 * b.sh_kib as f64 + 0.008 * b.sh_banks as f64);
        // Replace the base's single DMA engine, descriptor and four front slots.
        let dma =
            0.514 + self.dma_engines as f64 * (0.10 + 0.006 * (4 + 2 * self.dma_depth) as f64);
        let per_sm = 0.002 * self.memory_pending_limit() as f64
            + self.tc_count as f64 * self.tc_area_per_engine()
            - old_tc
            + dma
            - (0.514 + 0.10 + 0.006 * 4.)
            + extra_sh
            + self.reduction_units as f64 * (0.10 + 0.012 * b.vector as f64)
            + 0.08 * self.resident_groups.saturating_sub(1) as f64
            + 0.002 * (self.dma_depth * self.resident_groups) as f64
            + if self.multicast { 0.04 } else { 0. };
        let cache = if self.cache_mib == 0 {
            0.
        } else {
            3. * self.cache_mib as f64
                + 0.6
                + 0.012 * self.cache_mshrs() as f64
                + 0.006 * self.cache_bandwidth() as f64
        };
        b.area() + b.sms as f64 * per_sm + cache + if self.multicast { 0.2 } else { 0. }
    }
    pub fn base_power(&self) -> f64 {
        0.025 * self.area() + 0.15 * self.base.hbm_channels as f64
    }
    pub fn shared_energy_per_byte(&self, write: bool) -> f64 {
        self.base.sh_energy()
            * if !write && self.shared_ports == 2 {
                1.15
            } else {
                1.
            }
    }
    /// Count requests in bounded 16-word batches. Read broadcasts are free only
    /// within a batch; distinct words mapped to the same bank serialize.
    #[cfg(test)]
    fn sh_service(&self, view: View, write: bool) -> Result<(u64, u64)> {
        if self.base.sh_banks == 0 {
            return Err("SH instruction requires SH".into());
        }
        if view.rows.checked_mul(view.cols).is_none() || view.end().is_none() || view.len() > 16384
        {
            return Err("invalid or oversized SH view".into());
        }
        let mut cycles = 0;
        let mut words = 0;
        for start in (0..view.len()).step_by(16) {
            let mut addresses = [usize::MAX; 16];
            let mut counts = [0usize; 64];
            for i in 0..(view.len() - start).min(16) {
                let address = view.at(start + i);
                if write || !addresses[..i].contains(&address) {
                    counts[address % self.base.sh_banks] += 1;
                    words += 1;
                }
                addresses[i] = address;
            }
            let ports = if write { 1 } else { self.shared_ports };
            cycles += counts.iter().max().unwrap().div_ceil(ports) as u64;
        }
        Ok((words * 4, cycles))
    }
    pub fn compute_profile(&self, op: &Instr) -> Result<ComputeProfile> {
        let b = &self.base;
        let mut p = ComputeProfile::new(Engine::Vector);
        match *op {
            Instr::Fill { len, .. } => {
                check_len(len)?;
                for start in (0..len).step_by(b.vector) {
                    p.push(ComputeStep {
                        rf_write_bytes: ((len - start).min(b.vector) * 4) as u64,
                        ..Default::default()
                    });
                }
            }
            Instr::Vector {
                kind,
                len,
                a,
                b: arg_b,
                ..
            } => {
                check_len(len)?;
                let unary = matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt | Op::Square);
                let sfu = matches!(kind, Op::Exp | Op::Tanh | Op::Rsqrt);
                p.engine = if sfu { Engine::Sfu } else { Engine::Vector };
                let width = if sfu { b.sfu } else { b.vector };
                for start in (0..len).step_by(width) {
                    let count = (len - start).min(width);
                    let bytes = |arg| match arg {
                        Arg::Imm { .. } => 0,
                        Arg::Reg { stride: 0, .. } => 4,
                        _ => (count * 4) as u64,
                    };
                    p.push(ComputeStep {
                        rf_read_bytes: bytes(a)
                            + if unary { 0 } else { bytes(arg_b) }
                            + if matches!(kind, Op::Fma) {
                                (count * 4) as u64
                            } else {
                                0
                            },
                        rf_write_bytes: (count * 4) as u64,
                        core_duration: if sfu { 12 } else { 4 },
                        core_energy_pj: count as f64
                            * match kind {
                                Op::Exp | Op::Tanh => 12.,
                                Op::Rsqrt => 8.,
                                Op::Fma => 4.,
                                _ => 1.5,
                            },
                        ..Default::default()
                    });
                }
            }
            Instr::Reduce { len, .. } => {
                check_len(len)?;
                let dedicated = self.reduction_units > 0;
                if dedicated {
                    p.engine = Engine::Reduction;
                }
                let mut n = len;
                if n == 1 {
                    p.push(ComputeStep {
                        rf_read_bytes: 4,
                        rf_write_bytes: 4,
                        ..Default::default()
                    });
                }
                while n > 1 {
                    let out = n.div_ceil(2);
                    // Both paths use the same pairwise tree and visible scratch.
                    // A dedicated unit handles one tree layer in 1 cycle per
                    // lane group; the general vector ALU needs 4 cycles.
                    for start in (0..out).step_by(b.vector) {
                        let count = (out - start).min(b.vector);
                        let inputs = (n - 2 * start).min(2 * count);
                        p.push(ComputeStep {
                            rf_read_bytes: (inputs * 4) as u64,
                            rf_write_bytes: (count * 4) as u64,
                            core_duration: if dedicated { 1 } else { 4 },
                            core_energy_pj: (inputs / 2) as f64 * 1.5,
                            ..Default::default()
                        });
                    }
                    n = out;
                }
            }
            Instr::SharedRead { shared, .. } | Instr::SharedWrite { shared, .. } => {
                if shared.rows.checked_mul(shared.cols).is_none()
                    || shared.end().is_none()
                    || shared.len() > 16384
                {
                    return Err("invalid or oversized SH view".into());
                }
                p.engine = Engine::Copy;
                let write = matches!(op, Instr::SharedWrite { .. });
                // Every service group contains at most sixteen words.
                for start in (0..shared.len()).step_by(16) {
                    let count = (shared.len() - start).min(16);
                    let mut words = [0usize; 16];
                    for (i, w) in words[..count].iter_mut().enumerate() {
                        *w = shared.at(start + i);
                    }
                    let service = self.sh_activity(&words[..count], write)?;
                    let bytes = service.iter().sum();
                    let cycles = service.len() as u64;
                    let mut step = ComputeStep {
                        core_duration: if start == 0 { 6 } else { 0 },
                        core_energy_pj: count as f64 * 0.5 + if start == 0 { 20. } else { 0. },
                        ..Default::default()
                    };
                    if write {
                        step.rf_read_bytes = (count * 4) as u64;
                        step.sh_write_bytes = bytes;
                        step.sh_write_cycles = cycles;
                        step.sh_write_service = service;
                    } else {
                        step.sh_read_bytes = bytes;
                        step.sh_read_cycles = cycles;
                        step.sh_read_service = service;
                        step.rf_write_bytes = (count * 4) as u64;
                    }
                    p.push(step);
                }
            }
            Instr::Mma { m, n, k, .. } | Instr::MmaShared { m, n, k, .. } => {
                if self.tc_count == 0 {
                    return Err("MMA requires a TC engine".into());
                }
                for len in [m.checked_mul(k), k.checked_mul(n), m.checked_mul(n)] {
                    if len.ok_or("MMA shape overflow")? > 16384 {
                        return Err("MMA operand exceeds bounded shape".into());
                    }
                }
                let shared = if let Instr::MmaShared { b, .. } = op {
                    Some(*b)
                } else {
                    None
                };
                if shared.is_some() && b.sh_tc_bw == 0 {
                    return Err("MMA.SH requires paid SH-to-TC interface".into());
                }
                p.engine = Engine::Tc;
                for i in (0..m).step_by(b.p) {
                    for j in (0..n).step_by(b.q) {
                        let mm = (m - i).min(b.p);
                        let nn = (n - j).min(b.q);
                        p.push(ComputeStep {
                            rf_read_bytes: (mm * nn * 4) as u64,
                            ..Default::default()
                        });
                        for kk in (0..k).step_by(self.tc_k_parallel) {
                            let active = (k - kk).min(self.tc_k_parallel);
                            let mut step = ComputeStep {
                                rf_read_bytes: (mm * active * 4) as u64,
                                core_duration: 1 + self.tc_k_parallel.ilog2() as u64,
                                physical_fmas: (b.p * b.q * self.tc_k_parallel) as u64,
                                ..Default::default()
                            };
                            if let Some(view) = shared {
                                let mut words = Vec::with_capacity(nn * active);
                                for t in kk..kk + active {
                                    for col in j..j + nn {
                                        words.push(
                                            view.base + t * view.row_stride + col * view.col_stride,
                                        );
                                    }
                                }
                                // Kp groups may need up to 256 distinct B operands;
                                // split bank service into fixed sixteen-word batches.
                                for chunk in words.chunks(16) {
                                    let service = self.sh_activity(chunk, false)?;
                                    step.sh_read_bytes += service.iter().sum::<u64>();
                                    step.sh_read_cycles += service.len() as u64;
                                    step.sh_read_service.extend(service);
                                }
                                step.sh_tc_bytes = step.sh_read_bytes;
                            } else {
                                step.rf_read_bytes += (nn * active * 4) as u64;
                            }
                            step.core_energy_pj = step.physical_fmas as f64 * 3.
                                + (b.p * b.q * self.tc_k_parallel.saturating_sub(1)) as f64 * 1.5;
                            p.push(step);
                        }
                        p.push(ComputeStep {
                            rf_write_bytes: (mm * nn * 4) as u64,
                            core_duration: (b.p + b.q - 2) as u64,
                            ..Default::default()
                        });
                    }
                }
            }
            _ => return Err("DMA requires memory scheduler, not compute_profile".into()),
        }
        Ok(p)
    }
    fn sh_activity(&self, words: &[usize], write: bool) -> Result<Vec<u64>> {
        if self.base.sh_banks == 0 {
            return Err("SH instruction requires SH".into());
        }
        let mut counts = [0usize; 64];
        for (i, &word) in words.iter().enumerate() {
            if write || !words[..i].contains(&word) {
                counts[word % self.base.sh_banks] += 1;
            }
        }
        let ports = if write { 1 } else { self.shared_ports };
        let cycles = counts.iter().max().unwrap().div_ceil(ports);
        Ok((0..cycles)
            .map(|cycle| {
                counts
                    .iter()
                    .map(|&n| n.saturating_sub(cycle * ports).min(ports) as u64 * 4)
                    .sum()
            })
            .collect())
    }
}
fn check_len(n: usize) -> Result<()> {
    if n == 0 || n > 8192 {
        Err("compute operand length must be 1..8192".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extended_hardware_json_roundtrip() {
        let h = Hardware::default();
        let j = serde_json::to_value(&h).unwrap();
        let parsed: Hardware = serde_json::from_value(j.clone()).unwrap();
        let mut unknown = j;
        unknown
            .as_object_mut()
            .unwrap()
            .insert("typo_parameter".into(), serde_json::json!(1));
        assert!(serde_json::from_value::<Hardware>(unknown).is_err());
        parsed.validate().unwrap();
        assert_eq!(parsed.tc_count, h.tc_count);
        assert_eq!(parsed.base.sms, h.base.sms);
    }
    #[test]
    fn micro_steps_have_paid_bounded_operand_storage() {
        let mut h = Hardware::default();
        h.base.p = 1;
        h.base.q = 64;
        h.tc_k_parallel = 4;
        let p = h
            .compute_profile(&Instr::Mma {
                a: 0,
                b: 0,
                c: 0,
                m: 64,
                n: 64,
                k: 128,
            })
            .unwrap();
        for s in &p.steps {
            assert!(s.rf_read_bytes <= (65 * 4 * 4) as u64);
            assert!(s.rf_write_bytes <= 64 * 4);
        }
        assert_eq!(p.physical_fmas, 64 * 64 * 128);
        assert_eq!(p.steps.len(), 64 * (2 + 32));
    }
    #[test]
    fn restored_menus_and_disabled_unit_rules() {
        let mut h = Hardware::default();
        h.base.sms = 2;
        h.base.sfu = 3;
        h.base.sh_kib = 96;
        h.base.sh_banks = 1;
        h.validate().unwrap();
        h.base.p = 0;
        h.base.q = 0;
        h.tc_count = 0;
        h.validate().unwrap();
        h.tc_k_parallel = 2;
        assert!(h.validate().is_err());
        h.tc_k_parallel = 1;
        h.base.sh_tc_bw = 32;
        assert!(h.validate().is_err());
    }
    #[test]
    fn unchanged_hardware_has_exact_base_cost() {
        let h = Hardware::default();
        assert!((h.area() - h.base.area() - h.base.sms as f64 * 0.256).abs() < 1e-12);
        let mut more = h.clone();
        more.tc_count = 2;
        assert!(
            (more.area() - h.area() - h.base.sms as f64 * h.tc_area_per_engine()).abs() < 1e-12
        );
        more = h.clone();
        more.dma_engines = 2;
        assert!((more.area() - h.area() - h.base.sms as f64 * 0.124).abs() < 1e-12);
    }
    #[test]
    fn k_parallel_pays_tail_energy_and_preserves_operand_bytes() {
        let h = Hardware::default();
        let op = Instr::Mma {
            a: 0,
            b: 0,
            c: 0,
            m: 8,
            n: 8,
            k: 5,
        };
        let one = h.compute_profile(&op).unwrap();
        assert_eq!(one.rf_read_bytes, (64 + 40 + 40) * 4);
        assert_eq!(one.rf_write_bytes, 256);
        assert_eq!(one.physical_fmas, 320);
        assert_eq!(one.core_duration, 19);
        let mut k = h.clone();
        k.tc_k_parallel = 4;
        let four = k.compute_profile(&op).unwrap();
        assert_eq!(four.rf_read_bytes, one.rf_read_bytes);
        assert_eq!(four.physical_fmas, 512);
        assert_eq!(four.core_duration, 20);
        assert!(k.area() > h.area());
    }
    #[test]
    fn bank_ports_speed_reads_only() {
        let mut h = Hardware::default();
        h.base.sh_kib = 64;
        h.base.sh_banks = 8;
        let v = View {
            base: 0,
            rows: 16,
            cols: 1,
            row_stride: 8,
            col_stride: 1,
        };
        assert_eq!(h.sh_service(v, false).unwrap(), (64, 16));
        h.shared_ports = 2;
        assert_eq!(h.sh_service(v, false).unwrap(), (64, 8));
        assert_eq!(h.sh_service(v, true).unwrap(), (64, 16));
        let v = View { row_stride: 0, ..v };
        assert_eq!(h.sh_service(v, false).unwrap(), (4, 1));
    }
    #[test]
    fn dedicated_reduction_still_pays_reads() {
        let mut h = Hardware::default();
        let op = Instr::Reduce {
            scratch: 0,
            dst: 0,
            src: 0,
            len: 128,
            max: false,
        };
        let fallback = h.compute_profile(&op).unwrap();
        h.reduction_units = 1;
        let unit = h.compute_profile(&op).unwrap();
        assert_eq!(unit.rf_read_bytes, fallback.rf_read_bytes);
        assert_eq!(unit.rf_write_bytes, fallback.rf_write_bytes);
        assert!(unit.core_duration < fallback.core_duration);
        assert_eq!(unit.core_energy_pj, fallback.core_energy_pj);
    }
}

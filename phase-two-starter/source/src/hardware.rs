use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Hardware {
    pub sms: usize,
    pub rf_kib: usize,
    pub rf_tier: usize,
    #[serde(default)]
    pub sh_kib: usize,
    #[serde(default)]
    pub sh_banks: usize,
    #[serde(default)]
    pub sh_tc_bw: usize,
    pub vector: usize,
    pub sfu: usize,
    pub p: usize,
    pub q: usize,
    pub hbm_channels: usize,
    pub hbm_queue: usize,
    pub sm_noc: usize,
    pub global_noc: usize,
}
impl Default for Hardware {
    fn default() -> Self {
        Self {
            sms: 8,
            rf_kib: 64,
            rf_tier: 2,
            sh_kib: 0,
            sh_banks: 0,
            sh_tc_bw: 0,
            vector: 32,
            sfu: 4,
            p: 8,
            q: 8,
            hbm_channels: 4,
            hbm_queue: 128,
            sm_noc: 64,
            global_noc: 256,
        }
    }
}
impl Hardware {
    pub fn validate(&self) -> Result<()> {
        if cfg!(feature = "explore") {
            return self.validate_explore();
        }
        for (name, value, menu) in [
            ("sms", self.sms, &[4, 8, 16, 24][..]),
            ("rf_kib", self.rf_kib, &[32, 64, 128]),
            ("rf_tier", self.rf_tier, &[1, 2, 3]),
            ("vector", self.vector, &[16, 32, 64]),
            ("sfu", self.sfu, &[2, 4, 8, 16]),
            ("hbm_channels", self.hbm_channels, &[1, 2, 4, 8]),
            ("hbm_queue", self.hbm_queue, &[64, 128, 256]),
            ("sm_noc", self.sm_noc, &[32, 64, 128]),
            ("global_noc", self.global_noc, &[128, 256, 512]),
        ] {
            if !menu.contains(&value) {
                return Err(format!("invalid {name}: {value}"));
            }
        }
        if ![0, 64, 128, 256].contains(&self.sh_kib)
            || (self.sh_kib == 0 && self.sh_banks != 0)
            || (self.sh_kib > 0
                && (![16, 32, 64].contains(&self.sh_banks)
                    || self.sh_kib < self.sh_banks
                    || self.sh_kib > 16 * self.sh_banks))
        {
            return Err("invalid SH capacity/bank configuration".into());
        }
        if ![0, 32, 64, 128].contains(&self.sh_tc_bw) || (self.sh_tc_bw > 0 && self.sh_kib == 0) {
            return Err("invalid SH-to-TC interface".into());
        }
        if ![(8, 8), (8, 16), (16, 16)].contains(&(self.p, self.q)) || self.sfu > self.vector {
            return Err("invalid TC/SFU shape".into());
        }
        if self.area() > 100. {
            return Err(format!("area {:.3} exceeds 100 AU", self.area()));
        }
        Ok(())
    }
    fn validate_explore(&self) -> Result<()> {
        for (name, value, menu) in [
            ("sms", self.sms, &[2, 4, 8, 12, 16, 24, 32][..]),
            ("rf_kib", self.rf_kib, &[16, 32, 64, 128, 256]),
            ("rf_tier", self.rf_tier, &[1, 2, 3]),
            ("vector", self.vector, &[8, 16, 32, 64, 128]),
            ("sfu", self.sfu, &[1, 2, 4, 8, 16, 32]),
            ("hbm_channels", self.hbm_channels, &[1, 2, 4, 8]),
            ("hbm_queue", self.hbm_queue, &[32, 64, 128, 256, 512]),
            ("sm_noc", self.sm_noc, &[16, 32, 64, 128, 256]),
            ("global_noc", self.global_noc, &[64, 128, 256, 512]),
            ("sh_kib", self.sh_kib, &[0, 32, 64, 128, 256, 512, 1024]),
            ("sh_tc_bw", self.sh_tc_bw, &[0, 32, 64, 128, 256]),
        ] {
            if !menu.contains(&value) {
                return Err(format!("invalid explore {name}: {value}"));
            }
        }
        let no_tc = self.p == 0 && self.q == 0;
        let dims = [1, 2, 4, 8, 16, 32, 64];
        if !(no_tc
            || dims.contains(&self.p)
                && dims.contains(&self.q)
                && (16..=512).contains(&(self.p * self.q)))
            || self.sfu > self.vector
        {
            return Err("invalid explore TC/SFU".into());
        }
        if (self.sh_kib == 0 && (self.sh_banks != 0 || self.sh_tc_bw != 0))
            || (self.sh_kib > 0
                && (![8, 16, 32, 64].contains(&self.sh_banks)
                    || self.sh_kib < self.sh_banks
                    || self.sh_kib > 128 * self.sh_banks))
            || (no_tc && self.sh_tc_bw != 0)
        {
            return Err("invalid explore SH configuration".into());
        }
        if self.area() > 100. {
            return Err(format!("area {:.3} exceeds 100 AU", self.area()));
        }
        Ok(())
    }
    pub fn rf_latency(&self) -> u64 {
        if cfg!(feature = "explore") {
            2 + (self.rf_kib / 32).max(1).ilog2() as u64
        } else {
            2
        }
    }
    pub fn sh_latency(&self) -> u64 {
        if cfg!(feature = "explore") && self.sh_banks > 0 {
            6 + (self.sh_kib / self.sh_banks / 4).max(1).ilog2() as u64
        } else {
            6
        }
    }
    fn tc_area(&self) -> f64 {
        if cfg!(feature = "explore") {
            if self.p == 0 && self.q == 0 {
                return 0.;
            }
            // PE cost plus paid edge registers/ports; anchored at the old 8x8.
            0.072 + 0.008 * (self.p * self.q + self.p + self.q) as f64
        } else {
            0.20 + 0.008 * (self.p * self.q) as f64
        }
    }
    pub fn read_bw(&self) -> usize {
        [128, 256, 512][self.rf_tier - 1]
    }
    pub fn write_bw(&self) -> usize {
        [64, 128, 256][self.rf_tier - 1]
    }
    pub fn rf_energy(&self) -> f64 {
        [0.30, 0.42, 0.60][self.rf_tier - 1] * (1. + 0.1 * (self.rf_kib as f64 / 32.).log2())
    }
    pub fn area(&self) -> f64 {
        let sm = 0.45
            + 0.018 * self.vector as f64
            + 0.06 * self.sfu as f64
            + self.tc_area()
            + 0.018 * self.rf_kib as f64 * [1., 1.6, 2.6][self.rf_tier.saturating_sub(1).min(2)]
            + 0.004 * self.sh_kib as f64
            + 0.015 * self.sh_banks as f64
            + if self.sh_tc_bw == 0 {
                0.
            } else {
                0.28 + 0.004 * self.sh_tc_bw as f64
            }
            + 0.514
            + 0.10
            + 0.006 * 4.
            + 0.15 * (self.sm_noc as f64 / 64.).powf(1.3);
        sm * self.sms as f64
            + 0.65 * self.hbm_channels as f64
            + 0.0015 * (self.hbm_channels * self.hbm_queue) as f64
            + 1.
            + 0.035 * self.sms as f64
            + 0.8 * (self.global_noc as f64 / 128.).powf(1.3)
            + 2.
    }
    pub fn sh_energy(&self) -> f64 {
        if self.sh_kib == 0 {
            0.
        } else {
            0.8 * (1. + 0.05 * (self.sh_kib as f64 / 64.).log2())
        }
    }
    pub fn base_power(&self) -> f64 {
        0.025 * self.area() + 0.15 * self.hbm_channels as f64
    }
}

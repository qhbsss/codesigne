// Sliding-window integrals for piecewise-constant integer-cycle activity.
// Release: 65,536 slots. Explore: 262,144 slots accommodate up to 8192
// SFU elements at one lane plus expiration. Memory never grows with runtime.
// Sparse jumps are analytically equivalent; f64 rounding is oracle-tested.
const CAPACITY: usize = if cfg!(feature = "explore") {
    262144
} else {
    65536
};
#[derive(Clone)]
struct Window {
    width: u64,
    changes: Vec<f64>,
    time: u64,
    slope: f64,
    energy: f64,
    peak: f64,
    last_event: u64,
    pending: usize,
    occupied: Vec<u64>,
    sparse: bool,
}
impl Window {
    fn new(width: u64) -> Self {
        Self {
            width,
            changes: vec![0.; CAPACITY],
            time: 0,
            slope: 0.,
            energy: 0.,
            peak: 0.,
            last_event: 0,
            pending: 0,
            occupied: vec![0; CAPACITY / 64],
            sparse: cfg!(feature = "explore")
                && std::env::var("VNEXT_POWER_SPARSE").as_deref() == Ok("1"),
        }
    }
    fn add(&mut self, t: u64, duration: u64, energy: f64) {
        let rate = energy / duration as f64;
        for (when, delta) in [
            (t, rate),
            (t + duration, -rate),
            (t + self.width, -rate),
            (t + duration + self.width, rate),
        ] {
            assert!(
                when >= self.time && when - self.time < CAPACITY as u64,
                "power horizon exceeds bounded ISA lookahead"
            );
            let slot = &mut self.changes[when as usize & (CAPACITY - 1)];
            self.pending -= usize::from(*slot != 0.);
            *slot += delta;
            self.pending += usize::from(*slot != 0.);
            let index = when as usize & (CAPACITY - 1);
            let bit = 1u64 << (index % 64);
            if self.sparse {
                if *slot != 0. {
                    self.occupied[index / 64] |= bit;
                } else {
                    self.occupied[index / 64] &= !bit;
                }
            }
            self.last_event = self.last_event.max(when);
        }
    }
    fn advance(&mut self, to: u64) {
        assert!(to >= self.time);
        let stop = to.min(self.last_event + 1);
        while self.time < stop {
            let index = self.time as usize & (CAPACITY - 1);
            if self.sparse {
                let bits = self.occupied[index / 64] >> (index % 64);
                let gap = if bits == 0 {
                    64 - index % 64
                } else {
                    bits.trailing_zeros() as usize
                };
                let jump = (gap as u64).min(stop - self.time);
                if jump > 0 {
                    self.energy += self.slope * jump as f64;
                    self.peak = self.peak.max(self.energy);
                    self.time += jump;
                    continue;
                }
            }
            if self.sparse {
                self.occupied[index / 64] &= !(1u64 << (index % 64));
            }
            let slot = &mut self.changes[self.time as usize & (CAPACITY - 1)];
            self.slope += *slot;
            self.pending -= usize::from(*slot != 0.);
            *slot = 0.;
            self.energy += self.slope;
            self.peak = self.peak.max(self.energy);
            self.time += 1;
        }
        if to > self.last_event {
            self.slope = 0.;
            self.energy = 0.;
        }
        self.time = to;
    }
}
#[derive(Clone)]
pub struct Power {
    short: Window,
    long: Window,
    pub dynamic_pj: f64,
    pub max_pending: usize,
}
impl Default for Power {
    fn default() -> Self {
        Self {
            short: Window::new(100),
            long: Window::new(10000),
            dynamic_pj: 0.,
            max_pending: 0,
        }
    }
}
impl Power {
    pub fn add(&mut self, t: u64, duration: u64, energy: f64) {
        assert!(duration > 0 && energy.is_finite() && energy >= 0. && t >= self.short.time);
        self.short.add(t, duration, energy);
        self.long.add(t, duration, energy);
        self.dynamic_pj += energy;
        self.max_pending = self.max_pending.max(self.short.pending + self.long.pending);
    }
    pub fn advance(&mut self, t: u64) {
        self.short.advance(t);
        self.long.advance(t);
    }
    pub fn finish(&self, t: u64, base: f64) -> (f64, f64) {
        // Querying a report does not move the machine's clock into its idle tail.
        let mut p = self.clone();
        p.advance(t + 10000);
        (
            base + p.short.peak / 100. / 2000.,
            base + p.long.peak / 10000. / 2000.,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn burst_not_smeared_over_latency() {
        let mut p = Power::default();
        p.add(250, 2, 9600.);
        let (s, l) = p.finish(252, 2.);
        assert!((s - 2.048).abs() < 1e-10);
        assert!((l - 2.00048).abs() < 1e-10);
    }
    #[test]
    fn matches_discrete_oracle() {
        let mut p = Power::default();
        let mut cycles = vec![0.; 11000];
        for i in 0..200 {
            let t = i * 3;
            let d = 1 + i % 7;
            let e = (i % 13 + 1) as f64 * 200.;
            p.add(t as u64, d as u64, e);
            for v in &mut cycles[t..t + d] {
                *v += e / d as f64;
            }
        }
        let (s, l) = p.finish(1000, 0.);
        for (w, actual) in [(100, s), (10000, l)] {
            let expected = cycles
                .windows(w)
                .map(|x| x.iter().sum::<f64>() / w as f64 / 2000.)
                .fold(0., f64::max);
            assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
        }
    }
}

#[cfg(test)]
mod wrap_tests {
    use super::*;
    #[test]
    fn ring_wrap_and_idle_jumps_match_prefix_sum_oracle() {
        let mut p = Power::default();
        let mut a = vec![0.; 220_000];
        for t in (0..200_000).step_by(137) {
            p.advance(t as u64);
            let d = 1 + t % 113;
            let energy = (t % 43 + 1) as f64 * 77.;
            p.add(t as u64, d as u64, energy);
            for v in &mut a[t..t + d] {
                *v += energy / d as f64;
            }
        }
        let (s, l) = p.finish(200_000, 0.);
        let mut prefix = vec![0.; a.len() + 1];
        for (i, x) in a.iter().enumerate() {
            prefix[i + 1] = prefix[i] + x;
        }
        for (w, actual) in [(100, s), (10000, l)] {
            let expected = (w..prefix.len())
                .map(|i| (prefix[i] - prefix[i - w]) / w as f64 / 2000.)
                .fold(0., f64::max);
            assert!((actual - expected).abs() < 1e-8);
        }
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    #[test]
    fn sparse_integrals_match_dense_across_wrap_and_long_instructions() {
        let mut dense = Window::new(10000);
        dense.sparse = false;
        let mut sparse = dense.clone();
        sparse.sparse = true;
        for i in 0..4000u64 {
            let t = i * 137;
            dense.advance(t);
            sparse.advance(t);
            let duration = 1 + (i * 53) % 4000;
            let energy = (i % 79 + 1) as f64 * 0.13;
            dense.add(t, duration, energy);
            sparse.add(t, duration, energy);
        }
        dense.advance(600000);
        sparse.advance(600000);
        assert!((dense.peak - sparse.peak).abs() < 1e-6);
        assert_eq!(dense.pending, sparse.pending);
        assert_eq!(dense.changes, sparse.changes);
    }
}

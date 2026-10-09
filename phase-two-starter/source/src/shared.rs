use std::num::NonZeroU32;

// A partition of <=16 words uses <=31 bits: each service cycle is encoded
// as `active_words` one bits, followed by a zero separator. NonZero gives
// Option<Batch> the same four-byte representation; RF requests use None.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct Batch(NonZeroU32);
impl Batch {
    pub fn cycles(self) -> usize {
        let bits = self.0.get();
        (32 - bits.leading_zeros() - bits.count_ones() + 1) as usize
    }
    pub fn activity(self) -> impl Iterator<Item = u32> {
        let mut bits = self.0.get();
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let active = bits.trailing_ones();
            bits >>= active + 1;
            Some(active)
        })
    }
}
// Same-word reads broadcast within one service batch. Writes never broadcast.
pub(crate) fn batch(words: &[usize], banks: usize, write: bool) -> Batch {
    debug_assert!(!words.is_empty() && words.len() <= 16 && [8, 16, 32, 64].contains(&banks));
    let mut counts = [0usize; 64];
    let mut activity = [0u8; 16];
    let mut cycles = 0;
    for (i, &word) in words.iter().enumerate() {
        if !write && words[..i].contains(&word) {
            continue;
        }
        let bank = word % banks;
        activity[counts[bank]] += 1;
        counts[bank] += 1;
        cycles = cycles.max(counts[bank]);
    }
    let mut bits = 0;
    let mut offset = 0;
    for &n in &activity[..cycles] {
        bits |= ((1u32 << n) - 1) << offset;
        offset += n as u32 + 1;
    }
    Batch(NonZeroU32::new(bits).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    #[test]
    fn bank_service_matches_independent_word_sets() {
        for banks in [8, 16, 32, 64] {
            for seed in 0usize..256 {
                let words: Vec<_> = (0..1 + seed % 16)
                    .map(|i| ((i * i + seed * i * 17) % 97) * (seed % 33))
                    .collect();
                for write in [false, true] {
                    let mut accesses = BTreeMap::<usize, Vec<usize>>::new();
                    for &word in &words {
                        accesses.entry(word % banks).or_default().push(word);
                    }
                    let lengths: Vec<_> = accesses
                        .values()
                        .map(|v| {
                            if write {
                                v.len()
                            } else {
                                v.iter().collect::<BTreeSet<_>>().len()
                            }
                        })
                        .collect();
                    let actual = batch(&words, banks, write);
                    assert_eq!(actual.cycles(), *lengths.iter().max().unwrap());
                    for (cycle, active) in actual.activity().enumerate() {
                        assert_eq!(
                            active as usize,
                            lengths.iter().filter(|&&n| n > cycle).count()
                        );
                    }
                }
            }
        }
    }
}

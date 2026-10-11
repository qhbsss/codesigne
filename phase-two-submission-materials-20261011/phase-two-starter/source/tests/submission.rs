// Frozen v0.6 submission contract; v0.9 has its own audit suite.
#![cfg(not(feature = "explore"))]
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use vnext_sim::{
    baseline::{self, Fixture, FixtureProfile, Policy, Shape},
    hardware::Hardware,
    machine::Machine,
    submission::{self, ExportConfig},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "vnext-submission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn config(shared: bool) -> ExportConfig {
    let mut hw = Hardware::default();
    let mut policy = Policy::square(16);
    policy.head_group = 2;
    policy.persistent_keys = true;
    if shared {
        hw.sh_kib = 64;
        hw.sh_banks = 16;
        hw.sh_tc_bw = 64;
        policy.sh_rows = 2;
        policy.sh_direct = true;
    }
    ExportConfig {
        shape: Shape::smoke(),
        hardware: hw,
        tile: 16,
        seed: 7,
        fixture_profile: FixtureProfile::AttentionStress,
        prefill_policy: Some(policy),
        decode_policy: Some(policy),
    }
}
#[test]
fn exported_program_preserves_every_report_field_and_full_outputs() {
    for shared in [false, true] {
        let config = config(shared);
        let temp = Temp::new();
        submission::export(&config, &temp.0).unwrap();
        submission::pair_headers(&temp.0, false).unwrap();
        assert!(submission::pair_headers(&temp.0, true).is_err());
        for profile in [FixtureProfile::Legacy, FixtureProfile::AttentionStress] {
            let fixture = Fixture::new_profile(config.shape, 19, profile).unwrap();
            for decode in [false, true] {
                let mut m = Machine::new(config.hardware.clone(), true).unwrap();
                let actual =
                    baseline::run_policy(&mut m, &fixture, decode, config.prefill_policy.unwrap())
                        .unwrap();
                let eval = submission::evaluate(
                    &temp.0.join(if decode {
                        "decode.jsonl"
                    } else {
                        "prefill.jsonl"
                    }),
                    &fixture,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(m.report()).unwrap(),
                    serde_json::to_value(eval.report).unwrap()
                );
                assert_eq!(
                    eval.comparison.elements,
                    actual
                        .hidden
                        .iter()
                        .chain(&actual.keys)
                        .chain(&actual.values)
                        .map(Vec::len)
                        .sum::<usize>()
                );
                assert_eq!(
                    eval.step_cycles.iter().sum::<u64>(),
                    m.report().stats.cycles
                );
            }
        }
    }
}
#[test]
fn exported_program_is_independent_of_requested_seed_and_profile() {
    let a = Temp::new();
    let b = Temp::new();
    let mut c = config(false);
    submission::export(&c, &a.0).unwrap();
    c.seed = 987654;
    c.fixture_profile = FixtureProfile::Legacy;
    submission::export(&c, &b.0).unwrap();
    for name in ["prefill.jsonl", "decode.jsonl"] {
        assert_eq!(
            fs::read(a.0.join(name)).unwrap(),
            fs::read(b.0.join(name)).unwrap()
        );
    }
    assert!(submission::export(&c, &a.0).is_err());
}
#[test]
fn empty_program_missing_commits_and_after_commit_records_are_rejected() {
    let dir = Temp::new();
    let c = config(false);
    submission::export(&c, &dir.0).unwrap();
    let fixture = Fixture::new_profile(c.shape, 7, FixtureProfile::AttentionStress).unwrap();
    let path = dir.0.join("prefill.jsonl");
    let valid = fs::read_to_string(&path).unwrap();
    fs::write(&path, valid.lines().next().unwrap()).unwrap();
    assert!(
        submission::evaluate(&path, &fixture)
            .unwrap_err_string()
            .contains("missing")
    );
    fs::write(
        &path,
        format!("{valid}{{\"record\":\"alloc\",\"len\":1}}\n"),
    )
    .unwrap();
    assert!(
        submission::evaluate(&path, &fixture)
            .unwrap_err_string()
            .contains("after final")
    );
}
trait ErrorString {
    fn unwrap_err_string(self) -> String;
}
impl<T> ErrorString for Result<T, String> {
    fn unwrap_err_string(self) -> String {
        match self {
            Ok(_) => panic!("unexpected success"),
            Err(e) => e,
        }
    }
}
#[test]
fn score_has_no_artificial_ceiling() {
    assert_eq!(
        submission::score(submission::REFERENCE_PREFILL, submission::REFERENCE_DECODE).unwrap(),
        1000.
    );
    assert!(
        submission::score(
            submission::REFERENCE_PREFILL / 2,
            submission::REFERENCE_DECODE / 2
        )
        .unwrap()
            > 1999.
    );
    assert!(submission::score(0, 1).is_err());
}

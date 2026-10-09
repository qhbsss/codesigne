// Frozen v0.6 submission contract; v0.9 has its own audit suite.
#![cfg(not(feature = "explore"))]
//! Tests trust boundaries of static submissions independently of the generator.
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use vnext_sim::{
    baseline::{Fixture, FixtureProfile, Shape},
    hardware::Hardware,
    isa::{Instr, View},
    reference,
    submission::{self, Checkpoint, Header, OutputView, Record},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct ProgramFile(PathBuf);
impl ProgramFile {
    fn new(header: &Header, records: &[Record]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "vnext-submission-audit-{}-{}.jsonl",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut text = serde_json::to_string(header).unwrap() + "\n";
        for record in records {
            text.push_str(&serde_json::to_string(record).unwrap());
            text.push('\n');
        }
        fs::write(&path, text).unwrap();
        Self(path)
    }
}
impl Drop for ProgramFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn header(decode: bool) -> Header {
    Header {
        contract: submission::CONTRACT.into(),
        model: vnext_sim::MODEL.into(),
        hardware: Hardware::default(),
        shape: Shape::smoke(),
        decode,
    }
}
fn fixture(seed: u64) -> Fixture {
    Fixture::new_profile(Shape::smoke(), seed, FixtureProfile::AttentionStress).unwrap()
}
fn error(records: &[Record], decode: bool) -> String {
    let file = ProgramFile::new(&header(decode), records);
    submission::evaluate(&file.0, &fixture(7))
        .err()
        .expect("expected rejection")
}
#[test]
fn sources_cannot_be_refilled_realiased_or_bound_before_step() {
    let cases = [
        (
            vec![Record::Input {
                source: "input1".into(),
                len: 32,
            }],
            true,
            "future/past",
        ),
        (
            vec![Record::Input {
                source: "input00".into(),
                len: 224,
            }],
            false,
            "future/past",
        ),
        (
            vec![Record::Input {
                source: "layer00.qb".into(),
                len: 96,
            }],
            false,
            "noncanonical",
        ),
        (
            vec![Record::Input {
                source: "layer0.past_k".into(),
                len: 352,
            }],
            false,
            "unknown/unavailable",
        ),
        (
            vec![
                Record::Input {
                    source: "input0".into(),
                    len: 224,
                },
                Record::Release { tensor: 0 },
                Record::Input {
                    source: "input0".into(),
                    len: 224,
                },
            ],
            false,
            "only be bound once",
        ),
        (
            vec![Record::Input {
                source: "layer0.qb".into(),
                len: 97,
            }],
            false,
            "length mismatch",
        ),
        (vec![Record::Alloc { len: usize::MAX }], false, "allocation"),
    ];
    for (records, decode, expected) in cases {
        let e = error(&records, decode);
        assert!(e.contains(expected), "{e}");
    }
}
#[test]
fn arbitrary_literal_answer_passes_known_fixture_but_fails_unseen_fixture() {
    // Demonstrates why organizers must replay frozen hashes on private seeds.
    // Fixed constants are legal instructions; one public fixture is never a proof.
    let f = fixture(7);
    let expected = reference::run(&f, false);
    let mut values = expected.hidden[0].clone();
    for v in expected.keys.iter().chain(&expected.values) {
        values.extend(v);
    }
    let len = expected.hidden[0].len();
    let output = |base| OutputView {
        tensor: 1,
        base,
        len,
    };
    let mut streams = vec![vec![]; 8];
    for (i, value) in values.iter().copied().enumerate() {
        streams[0].push(Instr::Fill {
            dst: 0,
            len: 1,
            value,
        });
        streams[0].push(Instr::Store {
            tensor: 1,
            global: View::contiguous(i, 1, 1),
            local: View::contiguous(0, 1, 1),
        });
    }
    let records = vec![
        Record::Input {
            source: "input0".into(),
            len,
        },
        Record::Alloc { len: values.len() },
        Record::Wave { streams },
        Record::Commit {
            outputs: Checkpoint {
                hidden: output(0),
                keys: (1..3).map(|i| output(i * len)).collect(),
                values: (3..5).map(|i| output(i * len)).collect(),
            },
        },
    ];
    let file = ProgramFile::new(&header(false), &records);
    assert!(submission::evaluate(&file.0, &f).is_ok());
    assert!(submission::evaluate(&file.0, &fixture(19)).is_err());
}
#[test]
fn missing_commits_and_zero_allocations_fail_closed() {
    assert!(error(&[], false).contains("missing output commits"));
    assert!(error(&[Record::Alloc { len: 0 }], false).contains("invalid allocation"));
}

#![cfg(feature = "concurrent")]
//! Adversarial input tests operate on public static-program entry points.
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use vnext_sim::{
    baseline::{FixtureProfile, Shape},
    v09_hardware::Hardware,
    v09_submission::{self, CONTRACT, Header},
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Program(PathBuf);
impl Drop for Program {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn header() -> Header {
    Header {
        contract: CONTRACT.into(),
        model: vnext_sim::MODEL.into(),
        hardware: Hardware::default(),
        shape: Shape {
            layers: 1,
            d: 8,
            heads: 2,
            f: 16,
            prefill: 3,
            history: 3,
            steps: 2,
        },
        batch: 2,
        decode: true,
    }
}
fn program(h: Value, records: Vec<Value>) -> Program {
    let path = std::env::temp_dir().join(format!(
        "v09-audit-{}-{}.jsonl",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut text = serde_json::to_string(&h).unwrap();
    text.push('\n');
    for r in records {
        text.push_str(&serde_json::to_string(&r).unwrap());
        text.push('\n');
    }
    fs::write(&path, text).unwrap();
    Program(path)
}
fn rejects(h: Value, records: Vec<Value>, expected: &str) {
    let p = program(h, records);
    let result = std::panic::catch_unwind(|| {
        v09_submission::evaluate(&p.0, 29, FixtureProfile::AttentionStress)
    });
    assert!(result.is_ok(), "malformed student input must not panic");
    let error = result
        .unwrap()
        .err()
        .expect("invalid program unexpectedly accepted");
    assert!(
        error.contains(expected),
        "expected {expected:?}, got {error:?}"
    );
}
fn h() -> Value {
    serde_json::to_value(header()).unwrap()
}
fn group(command: Value) -> Value {
    json!({"record":"wave","groups":[{"sm":0,"rf_kib":4,"sh_kib":0,"commands":[command]}]})
}
fn view() -> Value {
    json!({"base":0,"rows":1,"cols":1,"row_stride":1,"col_stride":1})
}
#[test]
fn rejects_wrong_contract_model_and_unknown_header_fields() {
    let mut x = h();
    x["contract"] = json!("vnext-static-v08");
    rejects(x, vec![], "wrong model/contract");
    let mut x = h();
    x["model"] = json!("vnext-explore-v0.8");
    rejects(x, vec![], "wrong model/contract");
    let mut x = h();
    x["free_cache"] = json!(true);
    rejects(x, vec![], "unknown field");
}
#[test]
fn rejects_missing_commits_unknown_records_and_unknown_record_fields() {
    rejects(h(), vec![], "missing commits");
    rejects(
        h(),
        vec![json!({"record":"host_python","code":"print(1)"})],
        if cfg!(feature = "long-waves") {
            "unknown field"
        } else {
            "unknown variant"
        },
    );
    rejects(
        h(),
        vec![json!({"record":"alloc","len":1,"zero_cost":true})],
        "unknown field",
    );
}
#[test]
fn rejects_future_noncanonical_and_duplicate_input_bindings() {
    for source in ["input1", "input00"] {
        rejects(
            h(),
            vec![json!({"record":"input","source":source,"len":16})],
            "future/past input",
        );
    }
    let input = json!({"record":"input","source":"input0","len":16});
    rejects(h(), vec![input.clone(), input], "only once");
}
#[test]
fn rejects_releasing_input_and_capacity_overflow_before_allocation() {
    rejects(
        h(),
        vec![
            json!({"record":"input","source":"input0","len":16}),
            json!({"record":"release","tensor":0}),
        ],
        "cannot release bound input",
    );
    rejects(
        h(),
        vec![json!({"record":"alloc","len":usize::MAX})],
        "allocation",
    );
    rejects(
        h(),
        vec![json!({"record":"alloc","len":536_870_913usize})],
        "HBM capacity",
    );
}
#[test]
fn rejects_unknown_wait_and_unjoined_async_dma() {
    rejects(
        h(),
        vec![group(json!({"command":"wait","tokens":[7]}))],
        "unknown/repeated wait token",
    );
    let mut x = h();
    x["hardware"]["dma_depth"] = json!(1);
    let command = json!({"command":"async","token":0,"instruction":{"op":"load","tensor":0,"global":view(),"local":view()}});
    rejects(x, vec![group(command)], "explicit wait");
}
#[test]
fn adversarial_integer_fields_reject_without_panicking() {
    let bad_tensor = json!({"command":"run","instruction":{"op":"load","tensor":usize::MAX,"global":view(),"local":view()}});
    let p = program(h(), vec![group(bad_tensor)]);
    let outcome = std::panic::catch_unwind(|| {
        v09_submission::evaluate(&p.0, 1, FixtureProfile::AttentionStress)
    });
    assert!(outcome.is_ok());
    assert!(outcome.unwrap().is_err());
    for field in ["m", "n", "k"] {
        let mut command =
            json!({"command":"run","instruction":{"op":"mma","a":0,"b":0,"c":0,"m":1,"n":1,"k":1}});
        command["instruction"][field] = json!(usize::MAX);
        let p = program(h(), vec![group(command)]);
        let outcome = std::panic::catch_unwind(|| {
            v09_submission::evaluate(&p.0, 1, FixtureProfile::AttentionStress)
        });
        assert!(outcome.is_ok(), "field {field} panicked");
        assert!(outcome.unwrap().is_err());
    }
}
#[test]
fn unknown_hardware_cost_bypass_is_rejected_by_serde() {
    let mut x = h();
    x["hardware"]["area_override"] = json!(0);
    rejects(x, vec![], "unknown field");
}

fn small_config(decode: bool, k_parallel: usize) -> v09_submission::Config {
    let mut hardware = Hardware::default();
    hardware.base.sms = 4;
    hardware.base.rf_kib = 64;
    hardware.base.sh_kib = 64;
    hardware.base.sh_banks = 8;
    hardware.base.sh_tc_bw = 64;
    hardware.tc_k_parallel = k_parallel;
    hardware.resident_groups = 2;
    hardware.dma_depth = 1;
    hardware.cache_mib = 1;
    hardware.multicast = true;
    v09_submission::Config {
        shape: header().shape,
        batch: 3,
        decode,
        hardware,
        policy: vnext_sim::baseline::Policy {
            m: 3,
            n: 8,
            k: 8,
            head_group: 2,
            pack_transpose: true,
            persistent_keys: true,
            ..vnext_sim::baseline::Policy::square(8)
        },
        fixture_profile: FixtureProfile::AttentionStress,
        reference_threads: 1,
        parallel_groups: 2,
        async_prefetch: true,
    }
}
#[test]
fn exported_program_matches_native_prefill_decode_across_k_parallel_and_seeds() {
    for decode in [false, true] {
        for kp in [2, 4] {
            let c = small_config(decode, kp);
            let path = std::env::temp_dir().join(format!(
                "v09-static-{}-{}.jsonl",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let program = Program(path);
            v09_submission::export(&c, &program.0).unwrap();
            for seed in [7, 29] {
                let native = v09_submission::check(&c, seed).unwrap();
                let replay =
                    v09_submission::evaluate(&program.0, seed, FixtureProfile::AttentionStress)
                        .unwrap();
                assert_eq!(
                    native["output_sha256"].as_str().unwrap(),
                    replay.output_sha256
                );
                assert_eq!(
                    native["report"]["stats"]["cycles"].as_u64().unwrap(),
                    replay.report.stats.cycles
                );
                for (field, value) in [
                    ("short_power_w", replay.report.short_power_w),
                    ("long_power_w", replay.report.long_power_w),
                    ("energy_j", replay.report.energy_j),
                ] {
                    let a = native["report"][field].as_f64().unwrap();
                    assert!((a - value).abs() < 1e-9, "{field}: {a} vs {value}");
                }
                assert!(replay.comparison.max_scaled <= 1.);
            }
        }
    }
}
#[test]
fn machine_reset_clears_cache_state_but_waves_preserve_it() {
    use vnext_sim::{
        isa::{Instr, View},
        machine::Machine,
        v09_schedule::{Command, Group},
    };
    let c = small_config(false, 1);
    let mut m = Machine::new_concurrent(c.hardware, true, 1, false).unwrap();
    let wave = || {
        vec![Group {
            sm: 0,
            rf_kib: 4,
            sh_kib: 0,
            commands: vec![Command::Run {
                instruction: Instr::Load {
                    tensor: 0,
                    global: View::contiguous(0, 1, 1),
                    local: View::contiguous(0, 1, 1),
                },
            }],
        }]
    };
    m.bind_input("input0", &[1.], 1).unwrap();
    m.concurrent_wave(wave()).unwrap();
    m.concurrent_wave(wave()).unwrap();
    assert_eq!(m.concurrent_report().unwrap().memory.cache_hits, 1);
    m.reset();
    assert_eq!(m.concurrent_report().unwrap().stats.cycles, 0);
    m.bind_input("input0", &[2.], 1).unwrap();
    m.concurrent_wave(wave()).unwrap();
    let report = m.concurrent_report().unwrap();
    assert_eq!(report.memory.cache_hits, 0);
    assert_eq!(report.memory.hbm_reads, 1);
}
struct Suite(PathBuf);
impl Drop for Suite {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
#[test]
fn grade_admission_binds_fixed_workloads_and_one_common_hardware() {
    for mixed_hardware in [false, true] {
        let suite = Suite(std::env::temp_dir().join(format!(
            "v09-grade-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        )));
        fs::create_dir(&suite.0).unwrap();
        for (i, name) in v09_submission::CASES.iter().enumerate() {
            let mut h = header();
            let (shape, batch, decode) = v09_submission::known_shape(name).unwrap();
            h.shape = shape;
            h.batch = batch;
            h.decode = decode;
            if mixed_hardware && i == 1 {
                h.hardware.multicast = true;
            } else if !mixed_hardware && i == 0 {
                h.batch += 1;
            }
            fs::write(
                suite.0.join(format!("{name}.jsonl")),
                format!("{}\n", serde_json::to_string(&h).unwrap()),
            )
            .unwrap();
        }
        let error = v09_submission::grade(&suite.0, 7, FixtureProfile::AttentionStress)
            .err()
            .expect("invalid admission accepted");
        assert!(
            error.contains(if mixed_hardware {
                "identical hardware"
            } else {
                "fixed workload mismatch"
            }),
            "{error}"
        );
    }
}

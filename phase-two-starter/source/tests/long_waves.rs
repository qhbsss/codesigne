#![cfg(feature = "concurrent")]
use vnext_sim::{
    isa::Instr,
    machine::Machine,
    v09_hardware::Hardware,
    v09_schedule::{Command, Group, Scheduler},
};
fn fills(n: usize) -> Group {
    Group {
        sm: 0,
        rf_kib: 1,
        sh_kib: 0,
        commands: (0..n)
            .map(|_| Command::Run {
                instruction: Instr::Fill {
                    dst: 0,
                    len: 1,
                    value: 3.,
                },
            })
            .collect(),
    }
}
#[test]
fn experimental_admission_is_explicit_and_legacy_limit_is_preserved() {
    let h = Hardware::default();
    let mut m = Machine::new_concurrent(h, true, 1, false).unwrap();
    let result = m.concurrent_wave(vec![fills(16_385)]);
    assert_eq!(result.is_ok(), cfg!(feature = "long-waves"));
    if cfg!(feature = "long-waves") {
        let r = m.concurrent_report().unwrap();
        assert_eq!(r.stats.instructions, 16_385);
        assert!(r.stats.cycles > 16_385);
        assert_ne!(vnext_sim::MODEL, "vnext-concurrent-v0.9");
    }
}
#[test]
fn primitive_limit_is_enforced_at_boundary() {
    let s = Scheduler::new(Hardware::default(), false).unwrap();
    assert!(
        s.validate_groups(&[fills(vnext_sim::MAX_WAVE_PRIMITIVES)])
            .is_ok()
    );
    assert!(
        s.validate_groups(&[fills(vnext_sim::MAX_WAVE_PRIMITIVES + 1)])
            .is_err()
    );
}

#[test]
fn record_parser_preserves_strict_field_validation_and_field_order() {
    use vnext_sim::v09_submission::Record;
    for valid in [
        r#"{"len":2,"record":"alloc"}"#,
        r#"{"record":"wave","groups":[]}"#,
        r#"{"record":"input","source":"input0","len":1}"#,
        r#"{"tensor":0,"record":"release"}"#,
    ] {
        let record: Record = serde_json::from_str(valid).unwrap();
        let _: Record = serde_json::from_str(&serde_json::to_string(&record).unwrap()).unwrap();
    }
    for invalid in [
        r#"{"record":"wave","groups":[],"len":0}"#,
        r#"{"record":"wave","groups":null}"#,
        r#"{"record":"alloc","len":1,"len":2}"#,
        r#"{"record":"input","len":1}"#,
        r#"{"record":"alien"}"#,
        r#"{"record":"alloc","len":1,"extra":0}"#,
    ] {
        assert!(
            serde_json::from_str::<Record>(invalid).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn file_limits_are_versioned_without_allocating_the_limit() {
    assert_eq!(
        vnext_sim::submission::MAX_FILE_BYTES,
        if cfg!(feature = "compact") {
            128u64 << 20
        } else if cfg!(feature = "long-waves") {
            4u64 << 30
        } else {
            1u64 << 30
        }
    );
    assert_eq!(
        vnext_sim::submission::MAX_LINE_BYTES,
        if cfg!(feature = "compact") {
            16u64 << 20
        } else if cfg!(feature = "long-waves") {
            64u64 << 20
        } else {
            8u64 << 20
        }
    );
}

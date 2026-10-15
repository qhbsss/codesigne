//! Independent publication-boundary audit. No wall-time assertions.
use vnext_sim::{
    baseline::{Fixture, FixtureProfile, Shape},
    hardware::Hardware,
    isa::{Instr, View},
    machine::Machine,
    reference,
};

#[test]
fn comparator_rejects_replay_missing_kv_and_nonfinite_output() {
    let shape = Shape::smoke();
    let a = Fixture::new_profile(shape, 7, FixtureProfile::AttentionStress).unwrap();
    let b = Fixture::new_profile(shape, 19, FixtureProfile::AttentionStress).unwrap();
    for decode in [false, true] {
        let expected = reference::run(&a, decode);
        let replay = reference::run(&b, decode);
        assert!(reference::compare(&replay, &expected).is_err());
        let mut missing = reference::run(&a, decode);
        missing.keys[0].clear();
        assert!(reference::compare(&missing, &expected).is_err());
        let mut nonfinite = reference::run(&a, decode);
        nonfinite.hidden[0][0] = f32::NAN;
        assert!(reference::compare(&nonfinite, &expected).is_err());
    }
}

#[test]
fn overflowing_addresses_and_products_reject_before_execution() {
    for global in [
        View {
            base: usize::MAX,
            rows: 1,
            cols: 1,
            row_stride: 1,
            col_stride: 1,
        },
        View {
            base: 0,
            rows: usize::MAX,
            cols: 2,
            row_stride: 0,
            col_stride: 0,
        },
        View {
            base: 0,
            rows: 2,
            cols: 2,
            row_stride: usize::MAX,
            col_stride: 1,
        },
    ] {
        let mut machine = Machine::new(Hardware::default(), true).unwrap();
        let tensor = machine.alloc("input", vec![1.; 16]).unwrap();
        let mut wave = vec![vec![]; 8];
        wave[0].push(Instr::Load {
            tensor,
            global,
            local: View::contiguous(0, 1, 1),
        });
        assert!(machine.wave(wave).is_err());
        let report = machine.report();
        assert_eq!(report.stats.cycles, 0);
        assert!(
            report.execution_valid,
            "static errors must not mutate execution"
        );
    }
}

#[test]
fn maximum_bank_conflict_copy_matches_cycle_oracle_and_finite_metadata() {
    let hw = Hardware {
        sms: 4,
        sh_kib: 256,
        sh_banks: 16,
        ..Hardware::default()
    };
    let mut reports = vec![];
    for cycle_reference in [false, true] {
        let mut machine = Machine::new(hw.clone(), true).unwrap();
        machine.timing.cycle_reference = cycle_reference;
        let mut wave = vec![vec![]; 4];
        // 4K distinct words in one bank; RF->SH->RF with 16-way conflict in each batch.
        for base in [0, 4096, 8192, 12288] {
            wave[0].push(Instr::Fill {
                dst: base,
                len: 4096,
                value: 1.,
            });
        }
        // Destination contiguous rows of one element, each 16 words apart.
        let shared = View {
            base: 0,
            rows: 4096,
            cols: 1,
            row_stride: 16,
            col_stride: 1,
        };
        let local = View::contiguous(0, 4096, 1);
        wave[0].push(Instr::SharedWrite { shared, local });
        wave[0].push(Instr::SharedRead { shared, local });
        machine.wave(wave).unwrap();
        let report = machine.report();
        assert_eq!(report.stats.sh_bank_stall_cycles, 2 * 256 * 15);
        assert!(report.stats.max_sh_copy_inflight <= 8);
        assert!(report.stats.max_power_events <= 2 * 65536);
        reports.push(serde_json::to_value(report).unwrap());
    }
    assert_eq!(reports[0], reports[1]);
}

#[test]
fn largest_legal_shared_mma_fits_power_horizon() {
    let hw = Hardware {
        sms: 4,
        rf_kib: 128,
        sh_kib: 128,
        sh_banks: 16,
        sh_tc_bw: 32,
        p: 8,
        q: 16,
        ..Hardware::default()
    };
    let mut machine = Machine::new(hw, true).unwrap();
    let input = machine.alloc("B", vec![1.; 256 * 64]).unwrap();
    let mut wave = vec![vec![]; 4];
    wave[0].push(Instr::Fill {
        dst: 0,
        len: 4096,
        value: 0.,
    });
    for base in [4096, 8192, 12288, 16384] {
        wave[0].push(Instr::Fill {
            dst: base,
            len: 4096,
            value: 1.,
        });
    }
    wave[0].push(Instr::LoadShared {
        tensor: input,
        global: View::contiguous(0, 256, 64),
        local: View::contiguous(0, 256, 64),
    });
    wave[0].push(Instr::MmaShared {
        a: 4096,
        b: View::contiguous(0, 256, 64),
        c: 0,
        m: 64,
        n: 64,
        k: 256,
    });
    let output = machine.empty("C", 4096).unwrap();
    wave[0].push(Instr::Store {
        tensor: output,
        global: View::contiguous(0, 64, 64),
        local: View::contiguous(0, 64, 64),
    });
    machine.wave(wave).unwrap();
    assert!(machine.tensors[output].data.iter().all(|&x| x == 256.));
    assert!(machine.report().execution_valid);
}

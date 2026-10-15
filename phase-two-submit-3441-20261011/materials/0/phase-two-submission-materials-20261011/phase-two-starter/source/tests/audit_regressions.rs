use vnext_sim::{hardware::Hardware, isa::Instr, machine::Machine};

#[test]
fn reset_discards_hbm_and_all_timing_state() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    m.alloc("old_output", vec![123.; 16]).unwrap();
    m.reset();
    assert!(
        m.tensors.is_empty(),
        "reset must not preserve free computed HBM state"
    );
    assert_eq!(m.report().hbm_allocated_bytes, 0);
}
#[test]
fn impossible_empty_allocation_is_rejected_before_allocating() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    assert!(m.empty("overflow", usize::MAX).is_err());
}
#[test]
fn overflowing_reduction_fails_even_if_result_is_not_stored() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 2,
            value: f32::MAX,
        },
        Instr::Reduce {
            dst: 2,
            src: 0,
            len: 2,
            max: false,
            scratch: 3,
        },
    ];
    assert!(m.wave(w).is_err());
}

#[test]
fn failed_execution_cannot_report_pass_or_continue_until_reset() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![Instr::Reduce {
        src: 0,
        dst: 8,
        scratch: 16,
        len: 4,
        max: false,
    }];
    assert!(m.wave(w).is_err());
    assert!(!m.report().execution_valid);
    assert!(!m.report().power_pass);
    assert!(m.wave(vec![vec![]; 8]).is_err());
    m.reset();
    m.wave(vec![vec![]; 8]).unwrap();
    assert!(m.report().execution_valid);
}

// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn matrix_compute_time_and_energy_match_hand_calculation() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 64,
            value: 0.,
        },
        Instr::Fill {
            dst: 64,
            len: 512,
            value: 1.,
        },
        Instr::Fill {
            dst: 576,
            len: 512,
            value: 1.,
        },
        Instr::Mma {
            a: 64,
            b: 576,
            c: 0,
            m: 8,
            n: 8,
            k: 64,
        },
    ];
    m.wave(w).unwrap();
    let r = m.report();
    assert_eq!(r.stats.cycles, 132);
    assert_eq!(r.stats.rf_bytes, 8960);
    assert_eq!(r.stats.tc_physical_fmas, 4096);
    assert!((m.timing.power.dynamic_pj - (8960. * 0.462 + 4096. * 3. + 12. * 8.)).abs() < 1e-8);
}
// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn odd_reduction_time_and_energy_include_scratch_traffic() {
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    let mut w = vec![vec![]; 8];
    w[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 5,
            value: 1.,
        },
        Instr::Reduce {
            src: 0,
            dst: 5,
            len: 5,
            scratch: 6,
            max: false,
        },
    ];
    m.wave(w).unwrap();
    let r = m.report();
    assert_eq!(r.stats.cycles, 36);
    assert_eq!(r.stats.rf_bytes, 84);
    assert!((m.timing.power.dynamic_pj - (84. * 0.462 + 4. * 1.5 + 10. * 8.)).abs() < 1e-8);
}

// Exact v0.6 latency/energy fixture; newer models are tested separately.
#[cfg(not(feature = "explore"))]
#[test]
fn completed_dma_history_does_not_accumulate_live_state() {
    use vnext_sim::isa::View;
    let mut peaks = vec![];
    for count in [100, 4096] {
        let mut m = Machine::new(Hardware::default(), true).unwrap();
        let t = m.alloc("x", vec![1.; 16]).unwrap();
        let mut wave = vec![vec![]; 8];
        wave[0] = vec![
            Instr::Load {
                tensor: t,
                global: View::contiguous(0, 1, 16),
                local: View::contiguous(0, 1, 16)
            };
            count
        ];
        m.wave(wave).unwrap();
        let r = m.report();
        assert_eq!(r.stats.cycles, count as u64 * 276 + 1);
        assert_eq!(r.stats.hbm_reads, count as u64);
        assert_eq!(r.stats.max_hbm_inflight, 1);
        assert!(r.stats.max_live_events <= 2);
        assert!(r.stats.max_power_events < 2000);
        peaks.push((r.short_power_w, r.long_power_w));
    }
    assert!((peaks[0].0 - peaks[1].0).abs() < 1e-7);
    assert!((peaks[0].1 - peaks[1].1).abs() < 1e-7);
}
#[test]
fn longest_sfu_instruction_fits_bounded_power_horizon() {
    use vnext_sim::isa::{Arg, Op};
    let mut m = Machine::new(
        Hardware {
            sfu: 2,
            ..Hardware::default()
        },
        true,
    )
    .unwrap();
    let mut wave = vec![vec![]; 8];
    wave[0] = vec![
        Instr::Fill {
            dst: 0,
            len: 4096,
            value: 0.,
        },
        Instr::Vector {
            kind: Op::Exp,
            dst: 0,
            len: 4096,
            a: Arg::reg(0),
            b: Arg::Imm { value: 0. },
        },
    ];
    m.wave(wave).unwrap();
    let r = m.report();
    assert!(r.execution_valid);
    assert!(r.stats.cycles > 36000);
    assert!(r.stats.max_power_events <= 2 * 65536);
}
#[test]
fn mixed_compute_memory_and_wave_boundaries_match_cycle_steps() {
    use vnext_sim::isa::{Arg, Op, View};
    for seed in 0..36 {
        let h = Hardware {
            sms: 4,
            rf_kib: 32,
            rf_tier: seed % 3 + 1,
            sh_kib: 0,
            sh_banks: 0,
            sh_tc_bw: 0,
            vector: 16,
            sfu: 2,
            p: 8,
            q: 8,
            hbm_channels: 1 << (seed % 4),
            hbm_queue: 64,
            sm_noc: 32 << (seed % 3),
            global_noc: 128 << (seed % 3),
        };
        let mut reports = vec![];
        for cycle in [false, true] {
            let mut m = Machine::new(h.clone(), true).unwrap();
            m.timing.cycle_reference = cycle;
            let a = m.alloc("input", vec![1.; 65536]).unwrap();
            let b = m.empty("output", 4 * 64).unwrap();
            for pass in 0..3 {
                let mut wave = vec![vec![]; 4];
                for (sm, p) in wave.iter_mut().enumerate() {
                    let n = 3 + (sm + seed) % 6;
                    let k = 3 + (seed * 7 + sm) % 19;
                    p.extend([
                        Instr::Fill {
                            dst: 0,
                            len: n * n,
                            value: 0.,
                        },
                        Instr::Load {
                            tensor: a,
                            global: View {
                                base: seed + sm * 256,
                                rows: n,
                                cols: k,
                                row_stride: 256,
                                col_stride: if pass == 0 { 2 } else { 1 },
                            },
                            local: View::contiguous(64, n, k),
                        },
                        Instr::Fill {
                            dst: 512,
                            len: k * n,
                            value: 0.25,
                        },
                        Instr::Mma {
                            a: 64,
                            b: 512,
                            c: 0,
                            m: n,
                            n,
                            k,
                        },
                        Instr::Reduce {
                            dst: 1024,
                            src: 0,
                            len: n * n,
                            scratch: 1088,
                            max: pass == 1,
                        },
                        Instr::Vector {
                            kind: Op::Square,
                            dst: 1024,
                            len: 1,
                            a: Arg::scalar(1024),
                            b: Arg::Imm { value: 0. },
                        },
                        Instr::Store {
                            tensor: b,
                            global: View::contiguous(sm * 64, 1, 1),
                            local: View::contiguous(1024, 1, 1),
                        },
                    ]);
                }
                m.wave(wave).unwrap();
            }
            reports.push(serde_json::to_value(m.report()).unwrap());
        }
        assert_eq!(reports[0], reports[1], "mixed case {seed}");
    }
}

#[test]
fn disjoint_strided_reads_and_writes_on_same_line_are_legal() {
    use vnext_sim::isa::View;
    let mut m = Machine::new(Hardware::default(), true).unwrap();
    let t = m.alloc("interleaved", vec![1.; 8]).unwrap();
    let mut wave = vec![vec![]; 8];
    wave[0] = vec![Instr::Load {
        tensor: t,
        global: View {
            base: 0,
            rows: 1,
            cols: 4,
            row_stride: 0,
            col_stride: 2,
        },
        local: View::contiguous(0, 1, 4),
    }];
    wave[1] = vec![
        Instr::Fill {
            dst: 0,
            len: 4,
            value: 7.,
        },
        Instr::Store {
            tensor: t,
            global: View {
                base: 1,
                rows: 4,
                cols: 1,
                row_stride: 2,
                col_stride: 1,
            },
            local: View::contiguous(0, 4, 1),
        },
    ];
    m.wave(wave).unwrap();
    assert_eq!(m.tensors[t].data, vec![1., 7., 1., 7., 1., 7., 1., 7.]);
}

#[test]
fn strided_hazard_validation_matches_word_set_oracle() {
    use vnext_sim::isa::View;
    for stride in 0..6 {
        for read_base in 0..5 {
            for write_base in 0..20 {
                let mut machine = Machine::new(Hardware::default(), false).unwrap();
                let tensor = machine.alloc("shared", vec![1.; 64]).unwrap();
                let mut wave = vec![vec![]; 8];
                wave[0].push(Instr::Load {
                    tensor,
                    global: View {
                        base: read_base,
                        rows: 2,
                        cols: 3,
                        row_stride: 17,
                        col_stride: stride,
                    },
                    local: View {
                        base: 0,
                        rows: 2,
                        cols: 3,
                        row_stride: 3,
                        col_stride: 1,
                    },
                });
                wave[1].push(Instr::Store {
                    tensor,
                    global: View {
                        base: write_base,
                        rows: 2,
                        cols: 2,
                        row_stride: 5,
                        col_stride: 1,
                    },
                    local: View {
                        base: 0,
                        rows: 2,
                        cols: 2,
                        row_stride: 2,
                        col_stride: 1,
                    },
                });
                let collision = (0..2).any(|r| {
                    (0..3).any(|c| {
                        let address = read_base + r * 17 + c * stride;
                        (0..2).any(|w| {
                            (write_base + w * 5..write_base + w * 5 + 2).contains(&address)
                        })
                    })
                });
                assert_eq!(
                    machine.wave(wave).is_err(),
                    collision,
                    "stride={stride}, read={read_base}, write={write_base}"
                );
            }
        }
    }
}

#[test]
fn noc_active_mask_covers_highest_sm_and_round_robin_wrap() {
    use vnext_sim::isa::View;
    let hw = Hardware {
        sms: 24,
        rf_kib: 32,
        rf_tier: 1,
        vector: 16,
        sfu: 2,
        hbm_channels: 1,
        hbm_queue: 64,
        sm_noc: 32,
        global_noc: 128,
        ..Hardware::default()
    };
    let mut reports = vec![];
    for cycle_reference in [false, true] {
        let mut machine = Machine::new(hw.clone(), true).unwrap();
        machine.timing.cycle_reference = cycle_reference;
        let input = machine.alloc("input", vec![1.; 24 * 512]).unwrap();
        let output = machine.empty("output", 24 * 512).unwrap();
        let mut wave = vec![vec![]; 24];
        for (sm, program) in wave.iter_mut().enumerate() {
            if sm % 3 == 0 || sm == 23 {
                for pass in 0..4 {
                    program.push(Instr::Load {
                        tensor: input,
                        global: View::contiguous(sm * 512 + pass, 1, 257),
                        local: View::contiguous(0, 1, 257),
                    });
                    program.push(Instr::Store {
                        tensor: output,
                        global: View::contiguous(sm * 512 + pass, 1, 257),
                        local: View::contiguous(0, 1, 257),
                    });
                }
            }
        }
        machine.wave(wave).unwrap();
        assert_eq!(machine.tensors[output].data[23 * 512], 1.);
        reports.push(serde_json::to_value(machine.report()).unwrap());
    }
    assert_eq!(reports[0], reports[1]);
}

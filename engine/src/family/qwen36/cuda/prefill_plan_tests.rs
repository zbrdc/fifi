//! Tests of the chunked-prefill plan (CPF-1 v1, decision 0049): the chunk policy and list, the
//! admission rule against 0049's table, and the arena list of one chunk. CPU only. Every
//! number quoted from decision 0049 or from the CPF-7 plan (rev 3, section C.3) is named where
//! it is used. Setup is inline; the shared helpers are small and pinned by the tests below.
use super::{
    common_tests::{dims, plan_of},
    dims::KvMode,
    prefill_plan::{
        Admission, ChunkCost, ChunkSlot as S, Extra, M_LADDER, PART_CHARGE, Phase, Phase::*,
        PrefillStep, SPARE_MIN, chunk_cost, chunk_costs, chunk_list, chunk_tensors, kv_bytes,
        m_pos, state_bytes,
    },
};
use crate::{
    EngError,
    family::project::cuda::WeightLayout,
    kernels::cuda::{
        ArenaTensor, plan_arena,
        prefill::{
            attn_prefill::{
                CPF_ATTN_Q_TILE, CpfAttnGeometry, cpf_attn_part_bytes, cpf_attn_split_plan,
            },
            mmq_size::Q8_1_PAD,
        },
    },
};

const MIB: usize = 1 << 20;
/// The SM count of the RTX 3090; production reads it from the device facts.
const NSM: usize = 82;

fn chunk(p0: usize, rows: usize) -> PrefillStep {
    PrefillStep::Chunk { p0, rows }
}

fn feed(p0: usize) -> PrefillStep {
    PrefillStep::Feed { p0 }
}

fn steps(p0: usize, n_rows: usize, m_sess: usize) -> Vec<PrefillStep> {
    chunk_list(p0, n_rows, m_sess).expect("chunk list")
}

fn error_text(error: EngError) -> String {
    format!("{error:?}")
}

// ---- the policy and the chunk list ---------------------------------------------------------

#[test]
fn m_pos_steps_down_at_3k_and_32k_positions() {
    // 0049 item 9: 2,048 below 3K, 1,024 from 3K to 32K, 512 above (floor 512); K = 1,024.
    let rows = [(0, 2048), (3071, 2048), (3072, 1024), (32767, 1024)];
    for (p0, want) in rows
        .into_iter()
        .chain([(32768, 512), (131_072, 512), (usize::MAX, 512)])
    {
        assert_eq!(m_pos(p0), want, "p0 {p0}");
    }
}

#[test]
fn an_empty_run_has_no_steps_and_one_row_is_fed() {
    assert_eq!(steps(0, 0, 2048), vec![]);
    assert_eq!(steps(5000, 0, 128), vec![]);
    assert_eq!(steps(0, 1, 2048), vec![feed(0)]);
    assert_eq!(steps(7, 1, 512), vec![feed(7)]);
    assert_eq!(steps(0, 2, 2048), vec![chunk(0, 2)]);
}

#[test]
fn a_one_row_tail_goes_to_feed_and_longer_tails_are_chunks() {
    for m in M_LADDER {
        assert_eq!(steps(0, m, m), vec![chunk(0, m)], "{m} rows");
        assert_eq!(
            steps(0, m + 1, m),
            vec![chunk(0, m), feed(m)],
            "{m} + 1 rows"
        );
        assert_eq!(
            steps(0, m + 2, m),
            vec![chunk(0, m), chunk(m, 2)],
            "{m} + 2 rows"
        );
    }
}

#[test]
fn the_plans_prompt_rows_give_its_chunk_counts() {
    // CPF-7 plan C.6: a 1,024-token prompt feeds 1,023 rows as one chunk, a 2,048-token prompt
    // one chunk of 2,047, a 16K prompt 14 chunks (2 x 2,048, then 1,024s), a 128K prompt 222.
    assert_eq!(steps(0, 1023, 2048), vec![chunk(0, 1023)]);
    assert_eq!(steps(0, 2047, 2048), vec![chunk(0, 2047)]);
    let sixteen_k = steps(0, 16_383, 2048);
    assert_eq!(sixteen_k.len(), 14);
    assert_eq!(
        &sixteen_k[..3],
        [chunk(0, 2048), chunk(2048, 2048), chunk(4096, 1024)]
    );
    assert_eq!(sixteen_k.last(), Some(&chunk(15_360, 1023)));
    let big = steps(0, 131_071, 2048);
    let sizes = |rows: usize| {
        big.iter()
            .filter(|s| matches!(s, PrefillStep::Chunk { rows: r, .. } if *r == rows))
            .count()
    };
    assert_eq!(
        (big.len(), sizes(2048), sizes(1024), sizes(512), sizes(511)),
        (222, 2, 28, 191, 1)
    );
}

#[test]
fn m_pos_follows_the_position_of_each_chunk_start() {
    // A prefix import starts the run deep: the first chunk already takes the deep size.
    let deep = steps(32_768, 1500, 2048);
    assert_eq!(
        deep,
        vec![chunk(32_768, 512), chunk(33_280, 512), chunk(33_792, 476)]
    );
    // A chunk that starts just below 3K takes 2,048 rows and ends past the threshold.
    assert_eq!(
        steps(3000, 3000, 2048),
        vec![chunk(3000, 2048), chunk(5048, 952)]
    );
}

#[test]
fn m_sess_bounds_every_chunk() {
    assert!(
        steps(0, 3000, 256)
            .iter()
            .all(|s| matches!(s, PrefillStep::Chunk { rows, .. } if *rows <= 256))
    );
    assert_eq!(steps(0, 3000, 256).len(), 12);
}

#[test]
fn every_list_covers_the_run_in_order_with_only_a_last_feed() {
    for m_sess in M_LADDER {
        for p0 in [0, 1, 3071, 3072, 32_767, 32_768, 100_000] {
            for n_rows in [0, 1, 2, 3, 127, 128, 129, 2047, 2048, 2049, 4097, 10_000] {
                let list = steps(p0, n_rows, m_sess);
                let mut at = p0;
                for (index, step) in list.iter().enumerate() {
                    let context = format!("m_sess {m_sess} p0 {p0} n {n_rows} step {index}");
                    match *step {
                        PrefillStep::Chunk { p0: start, rows } => {
                            assert_eq!(start, at, "{context}");
                            assert!((2..=m_pos(at).min(m_sess)).contains(&rows), "{context}");
                            at += rows;
                        }
                        PrefillStep::Feed { p0: start } => {
                            assert_eq!((start, index + 1), (at, list.len()), "{context}");
                            at += 1;
                        }
                    }
                }
                assert_eq!(at, p0 + n_rows, "m_sess {m_sess} p0 {p0} n {n_rows}");
            }
        }
    }
}

#[test]
fn a_chunk_size_off_the_ladder_or_an_overflowing_run_is_refused() {
    // 0 comes last: a list that accepted it would make no progress.
    for m_sess in [1000, 4096, 100, 1, 0] {
        let error = chunk_list(0, 10, m_sess).expect_err("off the ladder");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        assert!(error_text(error).contains("[2048, 1024, 512, 256, 128]"));
    }
    let error = chunk_list(usize::MAX, 2, 2048).expect_err("p0 + n overflows");
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
}

// ---- admission against 0049's table --------------------------------------------------------

/// 0049 item 9: the basis for KV and the prefill arena on the 3090 (Tiled era).
const BASIS: usize = 2490 * MIB;
/// 0049 item 9: the P-b arena in tenths of a MiB for M = 128 ... 2,048 (EST).
const PB_TENTHS: [(usize, usize); 5] = [
    (128, 260),
    (256, 469),
    (512, 886),
    (1024, 1721),
    (2048, 3391),
];

fn pb_costs() -> Vec<ChunkCost> {
    let arena = |tenths: usize| tenths * MIB / 10;
    PB_TENTHS
        .iter()
        .map(|&(m, tenths)| ChunkCost {
            m,
            arena: arena(tenths),
            part: PART_CHARGE,
        })
        .collect()
}

fn cost_of(costs: &[ChunkCost], m: usize) -> ChunkCost {
    *costs.iter().find(|cost| cost.m == m).expect("a cost for m")
}

fn per_position(mode: KvMode) -> usize {
    kv_bytes(&dims(), mode, 1).expect("bytes per position")
}

/// The largest window `cost.m` passes, with `state(window)` the state term of each window.
fn last_window_with(
    admission: &Admission,
    cost: &ChunkCost,
    state: impl Fn(usize) -> usize,
) -> usize {
    let passes = |window: usize| admission.passes(state(window), cost).expect("sum");
    let (mut low, mut high) = (0usize, 1usize << 24);
    while low < high {
        let mid = (low + high).div_ceil(2);
        if passes(mid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// The largest window `cost.m` passes at `per_position` bytes of KV a position: the state
/// term is the KV alone, 0049's rows (the explicit zero-pool input; production charges
/// [`state_bytes`], the whole resident state).
fn last_window(admission: &Admission, cost: &ChunkCost, per_position: usize) -> usize {
    last_window_with(admission, cost, |window| window * per_position)
}

#[test]
fn the_constants_are_0049s_and_the_kv_bytes_are_the_pools() {
    assert_eq!((SPARE_MIN, PART_CHARGE), (64 * MIB, 64 * MIB));
    assert_eq!(per_position(KvMode::Fp16), 20_480);
    assert_eq!(per_position(KvMode::Int8G32), 10_880);
    let d = dims();
    assert_eq!(
        kv_bytes(&d, KvMode::Fp16, 120_000).unwrap(),
        120_000 * 20_480
    );
    assert_eq!(
        kv_bytes(&d, KvMode::Int8G32, 120_000).unwrap(),
        120_000 * 10_880
    );
}

#[test]
fn the_state_term_is_the_whole_resident_state_not_kv_alone() {
    // Lane C ruling (2026-10-14 00:18, option (b)): ResidentState::allocate allocates the
    // DeltaNet pool, the KV pool and the words together after the budget snapshot, so the
    // state term is StateLayout::bytes(), never the KV alone. Derived from the layout's
    // inputs at Qwen3.6: a DeltaNet layer holds align256(32 * 128 * 128 * 4 B) of recurrent
    // state (2,097,152) and align256(3 * 8,192 * 4 B) of convolution taps (98,304), over 30
    // layers: 65,863,680 B a slot (62.8125 MiB); the words block is 3 * 256 = 768 B; Fp16 KV
    // planes (1,024 B a position a plane) are whole multiples of the 256 B alignment, so the
    // pool and the words are the whole difference there. Int8G32 rounds its 20 scale planes
    // (32 B a position each) up to the alignment: 20 * 160 B more at a window of 100,003.
    let d = dims();
    assert_eq!(
        state_bytes(&d, KvMode::Fp16, 120_000, 1).unwrap(),
        120_000 * 20_480 + 65_863_680 + 768
    );
    assert_eq!(
        state_bytes(&d, KvMode::Int8G32, 120_000, 1).unwrap(),
        120_000 * 10_880 + 65_863_680 + 768
    );
    assert_eq!(
        state_bytes(&d, KvMode::Int8G32, 100_003, 1).unwrap(),
        100_003 * 10_880 + 65_863_680 + 768 + 20 * 160
    );
    assert!(
        state_bytes(&d, KvMode::Fp16, 120_000, 1).unwrap()
            > kv_bytes(&d, KvMode::Fp16, 120_000).unwrap()
    );
    // One more DeltaNet slot costs exactly one more pool.
    assert_eq!(
        state_bytes(&d, KvMode::Fp16, 120_000, 2).unwrap()
            - state_bytes(&d, KvMode::Fp16, 120_000, 1).unwrap(),
        65_863_680
    );
    let error = state_bytes(&d, KvMode::Fp16, 120_000, 0).expect_err("zero slots");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
}

#[test]
fn the_rule_reproduces_0049s_window_ends_from_its_pb_rows() {
    // 0049's rows are KV-only arithmetic: this reproduction passes the KV bytes alone as the
    // state term (an explicit zero DeltaNet pool and words), so 0049's figures reproduce
    // exactly; production charges state_bytes (pool + KV + words, lane C ruling
    // 2026-10-14 00:18, option (b)), which moves the real window ends down.
    let (costs, admission) = (pb_costs(), Admission::new(BASIS));
    let (fp16, kv8) = (per_position(KvMode::Fp16), per_position(KvMode::Int8G32));
    // 0049: "FP16 prefill ends at 119.6K tokens (103.5K at M = 2,048)", to 0.1K.
    assert_eq!(
        last_window(&admission, &cost_of(&costs, 128), fp16) / 100,
        1196
    );
    assert_eq!(
        last_window(&admission, &cost_of(&costs, 2048), fp16) / 100,
        1035
    );
    // 0049: "With 8-bit KV, M = 2,048 holds to about 195K tokens (about 153K with the Q4_K
    // fallback active)": the fallback is the 432 MiB untile scratch.
    let at_2048 = cost_of(&costs, 2048);
    assert!((194_500..=195_500).contains(&last_window(&admission, &at_2048, kv8)));
    let fallback = Admission {
        fallback: 432 * MIB,
        ..Admission::new(BASIS)
    };
    assert!((152_500..=153_500).contains(&last_window(&fallback, &at_2048, kv8)));
}

#[test]
fn admit_picks_the_largest_size_that_passes_at_every_window_end() {
    let (costs, admission) = (pb_costs(), Admission::new(BASIS));
    let fp16 = per_position(KvMode::Fp16);
    let mut ends: Vec<(usize, usize)> = M_LADDER
        .iter()
        .map(|&m| (m, last_window(&admission, &cost_of(&costs, m), fp16)))
        .collect();
    ends.sort();
    assert!(
        ends.windows(2).all(|w| w[0].1 > w[1].1),
        "smaller sizes reach deeper: {ends:?}"
    );
    let admit = |window: usize| admission.admit(window * fp16, &costs, None).expect("admit");
    for (index, &(m, end)) in ends.iter().enumerate() {
        assert_eq!(admit(end), Some(m), "at the end of {m}");
        let next_smaller = index.checked_sub(1).map(|below| ends[below].0);
        assert_eq!(
            admit(end + 1),
            next_smaller,
            "one position past the end of {m}"
        );
    }
}

#[test]
fn finding_13_dropping_the_part_row_flips_exactly_256_at_120k_fp16() {
    let (costs, admission) = (pb_costs(), Admission::new(BASIS));
    let kv = 120_000 * per_position(KvMode::Fp16);
    // 146 MiB are left after the KV (2,490 - 2,343.75 = 146.25).
    assert_eq!((BASIS - kv) / MIB, 146);
    assert_eq!(admission.admit(kv, &costs, None).unwrap(), None);
    assert!(
        costs
            .iter()
            .all(|cost| !admission.passes(kv, cost).unwrap())
    );
    // Without the 64 MiB part row: 256 leaves 146.25 - 46.9 = 99.35 MiB spare and is admitted,
    // 512 leaves 57.65 MiB and stays refused. 128 passes too, so the largest size flips to 256.
    // The part term is max(flat, real) (n3), so dropping the row zeroes both: the P-b rows
    // carry the flat charge as their part figure.
    let dropped_costs: Vec<ChunkCost> = costs
        .iter()
        .map(|cost| ChunkCost { part: 0, ..*cost })
        .collect();
    let dropped = Admission {
        part_charge: 0,
        ..Admission::new(BASIS)
    };
    assert_eq!(dropped.admit(kv, &dropped_costs, None).unwrap(), Some(256));
    let passing: Vec<usize> = M_LADDER
        .iter()
        .rev()
        .copied()
        .filter(|&m| dropped.passes(kv, &cost_of(&dropped_costs, m)).unwrap())
        .collect();
    assert_eq!(passing, vec![128, 256]);
    assert!(!dropped.passes(kv, &cost_of(&dropped_costs, 512)).unwrap());
}

#[test]
fn at_least_64_mib_must_stay_spare_not_just_one_byte() {
    let cost = ChunkCost {
        m: 1024,
        arena: 300 * MIB,
        part: 0,
    };
    let kv = 1000 * MIB;
    let need = kv + cost.arena + PART_CHARGE + SPARE_MIN;
    let passes = |budget: usize| Admission::new(budget).passes(kv, &cost).unwrap();
    assert!(passes(need) && passes(need + 1));
    assert!(!passes(need - 1), "64 MiB - 1 spare");
    assert!(!passes(need - SPARE_MIN + 1), "1 byte spare");
    assert!(!passes(need - SPARE_MIN), "0 bytes spare");
    assert!(!passes(0), "a charge above the budget");
}

/// Whether `admission` passes at the exact boundary where `charge` more device bytes are due.
fn exact_boundary(
    configure: impl Fn(Admission) -> Admission,
    charge: usize,
    m: usize,
) -> [bool; 2] {
    let cost = ChunkCost {
        m,
        arena: 300 * MIB,
        part: 0,
    };
    let kv = 1000 * MIB;
    let need = kv + cost.arena + PART_CHARGE + SPARE_MIN + charge;
    let at = |budget: usize| configure(Admission::new(budget)).passes(kv, &cost).unwrap();
    [at(need), at(need - 1)]
}

#[test]
fn every_device_charge_enters_the_sum_once_and_at_full_size() {
    let decode = plan_of(&dims(), WeightLayout::Native).arena_bytes();
    assert!(
        decode > 0,
        "the decode plan stands in for DeviceArena::bytes()"
    );
    let extra = |fixed, bytes_per_token| Extra {
        name: "x",
        fixed,
        bytes_per_token,
    };
    let rows: [(&str, Admission, usize); 6] = [
        (
            "decode arena",
            Admission {
                decode_arena: decode,
                ..Admission::new(0)
            },
            decode,
        ),
        (
            "fallback",
            Admission {
                fallback: 432 * MIB,
                ..Admission::new(0)
            },
            432 * MIB,
        ),
        (
            "fixed extra",
            Admission {
                device_extras: vec![extra(12_345, 0)],
                ..Admission::new(0)
            },
            12_345,
        ),
        (
            "per-token extra",
            Admission {
                device_extras: vec![extra(0, 8192)],
                ..Admission::new(0)
            },
            8192 * 1024,
        ),
        (
            "two extras",
            Admission {
                device_extras: vec![extra(7, 3), extra(11, 5)],
                ..Admission::new(0)
            },
            7 + 11 + 8 * 1024,
        ),
        (
            "part",
            Admission {
                part_charge: 3 * PART_CHARGE,
                ..Admission::new(0)
            },
            2 * PART_CHARGE,
        ),
    ];
    for (name, row, charge) in rows {
        let configure = |budget: Admission| Admission {
            device_budget: budget.device_budget,
            ..row.clone()
        };
        assert_eq!(
            exact_boundary(configure, charge, 1024),
            [true, false],
            "{name}"
        );
    }
}

#[test]
fn a_per_token_extra_scales_with_the_chunk_size() {
    let extra = Extra {
        name: "partial",
        fixed: 0,
        bytes_per_token: 8192,
    };
    let configure = |admission: Admission| Admission {
        device_extras: vec![extra],
        ..admission
    };
    // 8,192 B per token: 16 MiB at M = 2,048 and 1 MiB at M = 128.
    assert_eq!(exact_boundary(configure, 16 * MIB, 2048), [true, false]);
    assert_eq!(exact_boundary(configure, MIB, 128), [true, false]);
}

#[test]
fn a_lowered_flat_part_charge_still_pays_the_real_part_when_it_is_larger() {
    // n3: the part term of the sum is max(part_charge, cost.part). A later lowering of the
    // flat charge (lane M, A6 item 8) below a chunk's real part still pays the real part,
    // so the lowering cannot undercharge the chunk.
    let cost = ChunkCost {
        m: 1024,
        arena: 300 * MIB,
        part: 3 * MIB,
    };
    let kv = 100 * MIB;
    let lowered = |budget: usize| Admission {
        device_budget: budget,
        part_charge: MIB,
        ..Admission::new(0)
    };
    let need = kv + cost.arena + 3 * MIB + SPARE_MIN;
    assert!(lowered(need).passes(kv, &cost).unwrap());
    assert!(!lowered(need - 1).passes(kv, &cost).unwrap());
}

#[test]
fn pinned_host_rows_have_their_own_budget_and_never_enter_the_device_sum() {
    let host_row = Extra {
        name: "pinned",
        fixed: 0,
        bytes_per_token: 8192,
    };
    let with_host = |host_budget: usize| {
        move |admission: Admission| Admission {
            host_budget,
            host_extras: vec![host_row],
            ..admission
        }
    };
    // The device boundary does not move: 16 MiB of host rows are no device charge.
    assert_eq!(
        exact_boundary(with_host(8192 * 2048), 0, 2048),
        [true, false]
    );
    // The host budget refuses one byte short, and a smaller chunk fits the same budget.
    let cost = ChunkCost {
        m: 2048,
        arena: MIB,
        part: 0,
    };
    let tight = with_host(8192 * 2048 - 1)(Admission::new(100 * MIB * 1024));
    assert!(!tight.passes(0, &cost).unwrap());
    assert!(tight.passes(0, &ChunkCost { m: 1024, ..cost }).unwrap());
}

#[test]
fn a_cap_limits_m_sess_and_absent_means_2048() {
    let (costs, rich) = (pb_costs(), Admission::new(100 * 1024 * MIB));
    assert_eq!(rich.admit(0, &costs, None).unwrap(), Some(2048));
    for cap in M_LADDER {
        assert_eq!(
            rich.admit(0, &costs, Some(cap)).unwrap(),
            Some(cap),
            "cap {cap}"
        );
    }
    for cap in [0, 64, 100, 1000, 4096] {
        let error = rich
            .admit(0, &costs, Some(cap))
            .expect_err("off the ladder");
        assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
        assert!(error_text(error).contains("[2048, 1024, 512, 256, 128]"));
    }
}

#[test]
fn admission_wins_over_a_larger_cap_and_the_cap_over_a_larger_admission() {
    // 114,688 FP16 positions leave 250 MiB: 512 passes (250 - 88.6 - 64 >= 64), 1,024 does not.
    let (costs, admission) = (pb_costs(), Admission::new(BASIS));
    let kv = 114_688 * per_position(KvMode::Fp16);
    assert_eq!(admission.admit(kv, &costs, None).unwrap(), Some(512));
    assert_eq!(admission.admit(kv, &costs, Some(2048)).unwrap(), Some(512));
    assert_eq!(admission.admit(kv, &costs, Some(256)).unwrap(), Some(256));
}

#[test]
fn eight_bit_kv_admits_what_fp16_refuses() {
    let (costs, admission) = (pb_costs(), Admission::new(BASIS));
    let d = dims();
    let kv = |mode| kv_bytes(&d, mode, 150_000).unwrap();
    assert_eq!(
        admission.admit(kv(KvMode::Fp16), &costs, None).unwrap(),
        None
    );
    assert_eq!(
        admission.admit(kv(KvMode::Int8G32), &costs, None).unwrap(),
        Some(2048)
    );
}

#[test]
fn sums_are_checked_not_wrapped_or_saturated() {
    let d = dims();
    // 2^52 positions * 20,480 B = 5 * 2^64, which wraps to 0.
    let error = kv_bytes(&d, KvMode::Fp16, 1 << 52).expect_err("the product wraps");
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    assert!(kv_bytes(&d, KvMode::Int8G32, usize::MAX).is_err());
    let cost = ChunkCost {
        m: 128,
        arena: usize::MAX,
        part: 0,
    };
    assert!(
        Admission::new(usize::MAX).passes(1, &cost).is_err(),
        "kv + arena wraps to 0"
    );
    let small = ChunkCost {
        m: 2,
        arena: 1,
        part: 0,
    };
    let huge = |fixed, bytes_per_token| Extra {
        name: "huge",
        fixed,
        bytes_per_token,
    };
    for extras in [
        vec![huge(0, usize::MAX)],
        vec![huge(usize::MAX, 1)],
        vec![huge(usize::MAX, 0), huge(1, 0)],
    ] {
        let admission = Admission {
            device_extras: extras.clone(),
            ..Admission::new(usize::MAX)
        };
        let error = admission
            .passes(0, &small)
            .expect_err("the extras overflow");
        assert!(error_text(error).contains("huge"), "{extras:?}");
        let host = Admission {
            host_extras: extras,
            ..Admission::new(usize::MAX)
        };
        assert!(host.passes(0, &small).is_err());
    }
}

// ---- the arena list of one chunk -----------------------------------------------------------

/// The plan's table (C.3): per-token bytes at Qwen3.6, the fixed bytes (the constant q8_1 pad
/// of rule M1, the ids and expert-bound capacities, the fixup range at 82 SMs) and the phases.
/// The two `xf` sinks sit outside it: their bytes are not `per-token * m + fixed` (see
/// [`XF_SINKS`]).
#[rustfmt::skip]
const TABLE: [(S, usize, usize, Phase, Phase); 33] = [
    (S::Hidden, 8192, 0, Embed, Combine),
    (S::IdsTok, 4, 0, Embed, Embed),
    (S::X, 8192, 0, NormA, QuantA),
    (S::X2, 8192, 0, NormF, Scatter),
    (S::Q8X, 2304, Q8_1_PAD, QuantA, AProj),
    (S::Q8X2, 2304, Q8_1_PAD, Shared, Shared),
    (S::ProjD, 49_408, 0, DProj, DScan),
    (S::ConvOut, 32_768, 0, DConv, DScan),
    (S::OOut, 16_384, 0, DScan, DQuant),
    (S::Q8OD, 4608, Q8_1_PAD, DQuant, DOut),
    (S::Q8OA, 4608, Q8_1_PAD, AQuant, AOut),
    (S::ProjA, 36_864, 0, AProj, AFlash),
    (S::Q, 16_384, 0, APrep, AFlash),
    (S::AttnOut, 16_384, 0, AFlash, AQuant),
    (S::Delta, 8192, 0, DOut, NormF),
    (S::Logits, 1028, 0, Route, Route),
    (S::TopkId, 32, 0, Route, Combine),
    (S::TopkWeight, 32, 0, Route, Combine),
    (S::SharedGate, 4, 0, Route, Combine),
    (S::GuSh, 4096, 0, Shared, Shared),
    (S::ActSh, 2048, 0, Shared, Shared),
    (S::Q8Ash, 576, Q8_1_PAD, Shared, Shared),
    (S::ShOut, 8192, 0, Shared, Combine),
    (S::IdsG, 32, 0, Ids, Ids),
    (S::IdsSrc1, 32, 0, Ids, Scatter),
    (S::IdsDst, 36, 4 * 127, Ids, Down),
    (S::ExpertBounds, 0, 4 * 257, Ids, Down),
    (S::Q8S, 18_432, Q8_1_PAD, Scatter, GateUp),
    (S::Gu, 32_768, 0, GateUp, Swiglu),
    (S::Act, 16_384, 0, Swiglu, QuantAct),
    (S::Q8Act, 4608, Q8_1_PAD, QuantAct, Down),
    (S::Down, 65_536, 0, Down, Combine),
    (S::Fixup, 0, 4 * NSM * 128 * 128, Embed, Combine),
];

/// The plan's two `xf` sinks, the one slots whose bytes are not `per-token * m + fixed`:
/// `xf_bytes` returns the fixed fragment `65,537 * 16` B (the vendored fragment-order pack,
/// `native.rs`) for any chunk of at most 16 rows, then `m * k * 2` halves. The sink planned
/// at `M_sess` is the max of the two (verdict M1), or a 2-to-16-row chunk of the session
/// would not fit its prefix: at Qwen3.6 4,096 B a row with the 1,048,592 B floor, so the
/// sinks of M = 128 and 256 hold the whole fragment.
const XF_SINKS: [(S, Phase, Phase); 2] = [(S::XfA, NormA, NormA), (S::XfF, NormF, NormF)];

/// Bytes of one `xf` sink planned at `m` rows (the M1 max, pinned above the code).
fn xf_sink_bytes(m: usize) -> usize {
    (4096 * m).max(65_537 * 16)
}

fn tensors_of(m: usize) -> Vec<(S, ArenaTensor)> {
    chunk_tensors(&dims(), m, NSM).expect("chunk tensors")
}

fn only(tensors: &[(S, ArenaTensor)]) -> Vec<ArenaTensor> {
    tensors.iter().map(|&(_, tensor)| tensor).collect()
}

#[test]
fn the_tensor_list_is_the_plans_table_at_every_ladder_size() {
    for m in M_LADDER {
        let tensors = tensors_of(m);
        assert_eq!(tensors.len(), TABLE.len() + XF_SINKS.len(), "m {m}");
        for (slot, per_token, fixed, first, last) in TABLE {
            let found: Vec<&ArenaTensor> = tensors
                .iter()
                .filter(|(s, _)| *s == slot)
                .map(|(_, t)| t)
                .collect();
            assert_eq!(found.len(), 1, "m {m} {slot:?} appears once");
            let tensor = found[0];
            assert_eq!(tensor.bytes, per_token * m + fixed, "m {m} {slot:?} bytes");
            assert_eq!(
                (tensor.first, tensor.last),
                (first as usize, last as usize),
                "m {m} {slot:?}"
            );
        }
        for &(slot, first, last) in XF_SINKS.iter() {
            let tensor = tensors
                .iter()
                .find(|(s, _)| *s == slot)
                .expect("one sink")
                .1;
            assert_eq!(tensor.bytes, xf_sink_bytes(m), "m {m} {slot:?} bytes");
            assert_eq!(
                (tensor.first, tensor.last),
                (first as usize, last as usize),
                "m {m} {slot:?}"
            );
        }
    }
}

#[test]
fn every_smaller_chunk_fits_the_slots_planned_at_m_sess() {
    // M1: the prefix promise of `chunk_tensors`. `xf_bytes` is not monotone in m (the fixed
    // 1,048,592 B fragment of any chunk of at most 16 rows), so the sinks planned at M_sess
    // carry the fragment floor; every other slot grows with m. Each slot of every smaller
    // chunk must then fit its session's slot, at both SM counts.
    let d = dims();
    for &nsm in &[NSM, 20] {
        for &m_sess in M_LADDER.iter().rev() {
            let planned = chunk_tensors(&d, m_sess, nsm).expect("the session's plan");
            let planned_bytes = |slot: S| {
                planned
                    .iter()
                    .find(|(s, _)| *s == slot)
                    .expect("slot")
                    .1
                    .bytes
            };
            for m_prime in 2..=m_sess {
                let smaller = chunk_tensors(&d, m_prime, nsm).expect("a smaller chunk");
                for (slot, tensor) in smaller {
                    assert!(
                        tensor.bytes <= planned_bytes(slot),
                        "nsm {nsm} M_sess {m_sess} m' {m_prime} {slot:?}: {} B exceeds the \
                         session's planned {} B",
                        tensor.bytes,
                        planned_bytes(slot)
                    );
                }
            }
        }
    }
}

/// The largest sum of live bytes at one phase.
fn peak_live(tensors: &[ArenaTensor]) -> usize {
    (0..=Combine as usize)
        .map(|step| {
            tensors
                .iter()
                .filter(|t| (t.first..=t.last).contains(&step))
                .map(|t| t.bytes)
                .sum::<usize>()
        })
        .max()
        .expect("phases")
}

#[test]
fn plan_arena_places_the_list_without_overlap_between_live_tensors() {
    for m in M_LADDER {
        let tensors = only(&tensors_of(m));
        let plan = plan_arena(&tensors).expect("plan");
        let raw: usize = tensors.iter().map(|t| t.bytes).sum();
        for (i, a) in tensors.iter().enumerate() {
            assert_eq!(plan.offsets[i] % 256, 0, "m {m} tensor {i} alignment");
            assert!(
                plan.offsets[i] + a.bytes <= plan.total,
                "m {m} tensor {i} inside the arena"
            );
            for (j, b) in tensors.iter().enumerate().skip(i + 1) {
                let live_together = a.first <= b.last && b.first <= a.last;
                let apart = plan.offsets[i] + a.bytes <= plan.offsets[j]
                    || plan.offsets[j] + b.bytes <= plan.offsets[i];
                assert!(
                    !live_together || apart,
                    "m {m}: tensors {i} and {j} overlap"
                );
            }
        }
        assert!(
            peak_live(&tensors) <= plan.total,
            "m {m}: the arena holds the live set"
        );
        assert!(
            plan.total < raw,
            "m {m}: aliasing saves bytes ({} of {raw})",
            plan.total
        );
        let cost = chunk_cost(&dims(), m, NSM).expect("cost");
        assert_eq!(cost.arena, plan.total, "m {m}: the cost is the plan total");
    }
}

#[test]
fn the_deltanet_and_attention_blocks_share_bytes() {
    for m in M_LADDER {
        let tensors = tensors_of(m);
        let plan = plan_arena(&only(&tensors)).expect("plan");
        let range = |slot: S| {
            let index = tensors.iter().position(|(s, _)| *s == slot).expect("slot");
            (
                plan.offsets[index],
                plan.offsets[index] + tensors[index].1.bytes,
            )
        };
        let (delta, attn) = (range(S::ProjD), range(S::ProjA));
        assert!(
            delta.0 < attn.1 && attn.0 < delta.1,
            "m {m}: {delta:?} and {attn:?}"
        );
    }
}

#[test]
fn the_fixup_range_and_the_part_buffer_follow_the_device_sm_count() {
    let d = dims();
    let fixup = |nsm: usize| {
        let tensors = chunk_tensors(&d, 128, nsm).expect("tensors");
        tensors
            .iter()
            .find(|(slot, _)| *slot == S::Fixup)
            .expect("fixup")
            .1
            .bytes
    };
    assert_eq!(
        (fixup(82), fixup(40)),
        (4 * 82 * 128 * 128, 4 * 40 * 128 * 128)
    );
    // The split plan fills two blocks per SM: 6 splits at 82 SMs, 3 at 40, for 128 rows.
    let part = |nsm: usize| chunk_cost(&d, 128, nsm).expect("cost").part;
    assert_eq!(
        (part(82), part(40)),
        (128 * 2 * 6 * 8 * 258 * 4, 128 * 2 * 3 * 8 * 258 * 4)
    );
}

#[test]
fn the_part_buffer_is_the_sister_plan_and_the_flat_charge_covers_it() {
    // CPF-7 plan C.5 note 5: 16.1 / 16.1 / 32.3 MiB at M = 512 / 1,024 / 2,048 (82 SMs).
    let d = dims();
    let part = |m: usize| chunk_cost(&d, m, NSM).expect("cost").part;
    assert_eq!(
        (part(512), part(1024), part(2048)),
        (16_908_288, 16_908_288, 33_816_576)
    );
    assert_eq!((part(128), part(256)), (12_681_216, 12_681_216));
    assert!(M_LADDER.iter().all(|&m| part(m) < PART_CHARGE));
}

#[test]
fn the_part_cap_covers_every_chunks_own_split_plan() {
    // m1: the driver allocates the part buffer at the flat cap, never at ChunkCost.part,
    // because a smaller chunk launched with its own split plan can need more. The sweep pins
    // the other half of the rule: the flat cap covers every chunk's own plan, at both SM
    // counts (the largest is m' = 2,041..2,048 at one split: 33,816,576 B).
    let d = dims();
    let a = &d.shape.attn;
    let heads = (a.heads_q, a.heads_kv);
    for &nsm in &[NSM, 20] {
        for m in 1..=2048 {
            let geometry = CpfAttnGeometry {
                m,
                n_head: heads.0,
                n_kv_head: heads.1,
                head_dim: a.head_dim,
            };
            let split = cpf_attn_split_plan(geometry, m, CPF_ATTN_Q_TILE, nsm).expect("split");
            let bytes = cpf_attn_part_bytes(geometry, split).expect("part bytes");
            assert!(
                bytes <= PART_CHARGE,
                "nsm {nsm} m {m}: {bytes} over the cap"
            );
        }
    }
    // The ChunkCost.part doc's two counterexamples, from the code's inputs: the base grid is
    // tiles * 2 kv heads with tiles = ceil(m'/8); splits = ceil(164 / base) (164 = 2 * 82
    // SMs); part = the tiles*8 rounded rows * 2 kv heads * splits * 8 heads a group
    // (16 / 2) * (2 + 256) floats * 4 B.
    // m' = 209: 27 tiles, base 54, 4 splits, 216 rows -> 14,266,368, against the 256-row
    // plan's 12,681,216 (32 tiles, base 64, 3 splits).
    // m' = 641: 81 tiles, base 162, 2 splits, 648 rows -> 21,399,552, against the 1,024-row
    // plan's 16,908,288 (128 tiles, base 256, 1 split).
    let part = |m: usize| chunk_cost(&d, m, NSM).expect("cost").part;
    assert_eq!((part(209), part(256)), (14_266_368, 12_681_216));
    assert_eq!((part(641), part(1024)), (21_399_552, 16_908_288));
}

#[test]
fn a_chunk_outside_one_to_2048_rows_or_a_zero_sm_count_is_refused() {
    let d = dims();
    for m in [0, 2049, 4096] {
        let error = chunk_tensors(&d, m, NSM).expect_err("m outside the domain");
        assert!(
            matches!(error, EngError::BadRequest(_)) && error_text(error).contains("chunk cap")
        );
    }
    assert!(chunk_tensors(&d, 1, NSM).is_ok() && chunk_tensors(&d, 2048, NSM).is_ok());
    let error = chunk_tensors(&d, 128, 0).expect_err("a zero SM count");
    assert!(matches!(error, EngError::BadRequest(_)), "{error:?}");
    assert!(
        error_text(error).contains("nsm 0"),
        "chunk_tensors' own refusal, not plan_arena's"
    );
}

#[test]
fn chunk_costs_are_the_ladder_in_order_and_feed_admission() {
    let d = dims();
    let costs = chunk_costs(&d, NSM).expect("costs");
    assert_eq!(costs.iter().map(|c| c.m).collect::<Vec<_>>(), M_LADDER);
    for cost in &costs {
        assert_eq!(*cost, chunk_cost(&d, cost.m, NSM).unwrap());
    }
    // With the computed totals and the whole resident state charged (the pool and the words
    // move every FP16 window end down by about 3.2K positions against the KV-only figure),
    // 100K FP16 positions admit the 2,048 arena and 120K admit none.
    let admission = Admission::new(BASIS);
    let state = |window: usize| state_bytes(&d, KvMode::Fp16, window, 1).expect("state");
    assert_eq!(
        admission.admit(state(100_000), &costs, None).unwrap(),
        Some(2048)
    );
    assert_eq!(admission.admit(state(120_000), &costs, None).unwrap(), None);
}

#[test]
fn fp16_kv_is_capped_near_80k_positions_at_the_2048_row_arena_ceiling() {
    // The ceiling is the M = 2,048 unaliased arena, 777,223,680 B (pinned below at 7,412
    // tenths of a MiB); the charges are the 64 MiB part, the 64 MiB spare and the resident
    // state — the pool 65,863,680 B and the words 768 B on top of 20,480 B a position (Fp16
    // KV planes are whole multiples of the alignment, so the state adds the fixed
    // 65,864,448 B to the per-position figure).
    // The bases are decision 0049's basis, not a measurement: 2,490 MiB (2,610,954,240 B)
    // with the Native restatement at +42 MiB = 2,532 MiB (2,654,994,432 B). The measured
    // Tiled free-after-load is 3,618,963,456 B (knowledge entry cuda-3090-context-budget,
    // /home/dan/fifi/knowledge/), which leaves 2,427 MiB after the 1 GiB headroom — a 63 MiB
    // gap below 0049's basis; no numeric change follows from it here (Q1).
    //   2,490 MiB: (2,610,954,240 - 777,223,680 - 2 * 67,108,864 - 65,864,448) / 20,480
    //             = 1,633,648,384 / 20,480 = 79,767 (79,768 * 20,480 is 256 B past it).
    //   2,532 MiB: (2,654,994,432 - 777,223,680 - 2 * 67,108,864 - 65,864,448) / 20,480
    //             = 1,677,688,576 / 20,480 = 81,918 (remainder 7,936).
    let ceiling: usize = tensors_of(2048).iter().map(|(_, t)| t.bytes).sum();
    assert_eq!(ceiling / (MIB / 10), 7412);
    let cost = ChunkCost {
        m: 2048,
        arena: ceiling,
        part: PART_CHARGE,
    };
    let d = dims();
    let end = |basis_mib: usize| {
        last_window_with(&Admission::new(basis_mib * MIB), &cost, |window| {
            state_bytes(&d, KvMode::Fp16, window, 1).expect("state")
        })
    };
    assert_eq!(end(2490), 79_767);
    assert_eq!(end(2532), 81_918);
}

/// The one report the packet prints (run with `--nocapture`): v1's `plan_arena` totals beside
/// 0049's P-b rows, and the admission tables of both. Only the invariants are gated.
#[test]
fn v1_totals_and_admission_tables_are_reported_beside_0049s_rows() {
    let d = dims();
    let v1 = chunk_costs(&d, NSM).expect("costs");
    let mib = |bytes: usize| bytes as f64 / MIB as f64;
    println!("M      v1 arena  peak live  unaliased  P-b row   v1 part   flat part  (MiB)");
    for (&(m, tenths), cost) in PB_TENTHS.iter().zip(v1.iter().rev()) {
        let tensors = only(&tensors_of(m));
        let (peak, raw) = (
            peak_live(&tensors),
            tensors.iter().map(|t| t.bytes).sum::<usize>(),
        );
        assert!(peak <= cost.arena && cost.arena <= raw, "m {m}");
        println!(
            "{m:<6} {:>8.2} {:>10.2} {:>10.2} {:>8.1} {:>9.2} {:>10.2}",
            mib(cost.arena),
            mib(peak),
            mib(raw),
            tenths as f64 / 10.0,
            mib(cost.part),
            mib(PART_CHARGE)
        );
    }
    let pb = pb_costs();
    let admission = Admission::new(BASIS);
    let modes = [("FP16", KvMode::Fp16), ("KV8", KvMode::Int8G32)];
    println!("last window each M passes (positions), P-b rows | v1 plan (whole state)");
    for m in M_LADDER.iter().rev() {
        let ends = modes.map(|(_, mode)| {
            let per = per_position(mode);
            let state = |window: usize| state_bytes(&d, mode, window, 1).expect("state");
            let pb_end = last_window(&admission, &cost_of(&pb, *m), per);
            let v1_end = last_window_with(&admission, &cost_of(&v1, *m), state);
            (pb_end, v1_end)
        });
        println!(
            "  M {m:>4}: FP16 {} | {}   KV8 {} | {}",
            ends[0].0, ends[0].1, ends[1].0, ends[1].1
        );
    }
    let windows = [
        32_768usize,
        65_536,
        98_304,
        104_000,
        110_000,
        114_000,
        118_000,
        120_000,
        131_072,
        160_000,
        196_000,
        262_144,
    ];
    for (label, mode) in modes {
        println!("{label}: M x window, P-b/v1 (Y passes, - refused)");
        println!("  M    {}", windows.map(|w| format!("{w:>8}")).join(""));
        for &m in M_LADDER.iter().rev() {
            let cells = windows.map(|window| {
                let pb_state = kv_bytes(&d, mode, window).expect("kv");
                let v1_state = state_bytes(&d, mode, window, 1).expect("state");
                let pb = admission.passes(pb_state, &cost_of(&pb, m)).expect("sum");
                let v1 = admission.passes(v1_state, &cost_of(&v1, m)).expect("sum");
                let mark = |ok: bool| if ok { 'Y' } else { '-' };
                format!("{:>6}{}/{}", "", mark(pb), mark(v1))
            });
            println!("  {m:<5}{}", cells.join(""));
        }
        println!(
            "  admit P-b {:?}",
            windows.map(|w| {
                admission
                    .admit(kv_bytes(&d, mode, w).unwrap(), &pb, None)
                    .unwrap()
            })
        );
        println!(
            "  admit v1  {:?}",
            windows.map(|w| {
                admission
                    .admit(state_bytes(&d, mode, w, 1).unwrap(), &v1, None)
                    .unwrap()
            })
        );
    }
}

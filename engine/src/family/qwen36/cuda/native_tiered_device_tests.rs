//! Device tests of the TIERED MoE block of OFF-2 D3, in the isolation split of the native MoE
//! tests: each device phase creates its own context, stream, bank, weights and buffers, has only
//! `?` inside, drops every owner when it returns, and only then does the test assert on host
//! data (a drop while panicking would latch the backend for the rest of a one-process suite).
//!
//! The bank holds every expert of the synthetic layer (16 experts, so 16 slots); a host that maps
//! expert `e` to `slot_of[e]` makes the tiered block a re-indexed copy of the native block and its
//! output has to be BITWISE the native one. Hosts that serve experts on the CPU, faulted routers
//! and capture windows are separate tests below.
use super::moe::tiered::{
    LayerPlan, RouteRead, TieredBlock, TieredLayerHost, TieredScratch, run_moe_block_tiered,
};
use super::moe::{Launch, MoeLayerWeights, StackWeights};
use super::native_device_rig::{DIMS, Sources, device_present, rig_weights, sources};
use super::native_device_rig::{f32_buf, zero_buf};
use super::native_moe_device_tests::{
    BlockRun, block_bufs, moe_bufs, produce, read_back, run_block,
};
use super::tiered_load::tier_kernel;
use super::tiered_step::BankFill;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::qwen36::tiered::cpu_pool::{CpuPool, ExpertJob, MappedRegion, PendingLayer};
use crate::family::qwen36::tiered::host::BankDevice;
use crate::family::qwen36::tiered::locator::ExpertRanges;
use crate::kernels::cuda::resident::moe_combine::{CombineCall, CombineOut, CombineShape};
use crate::kernels::cuda::{
    CapturePlan, CudaContext, DeviceStream, ExpertBank, ExpertBankGeometry,
};
use crate::kernels::expert_cpu::ExpertShape;
use std::sync::Arc;
use std::time::Duration;

/// Bytes of one expert's gate (or up, or down) slice: 256 rows of 256 values, one block each.
const SLICE: usize = DIMS.inter * DIMS.hidden / 256 * 144;

/// Serves every pick from the bank: expert `e` sits in slot `slot_of[e]`; the router-fault id
/// passes through; nothing runs on the CPU.
struct Resident {
    slot_of: Vec<i32>,
}

impl TieredLayerHost for Resident {
    fn begin_layer(&mut self, _layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError> {
        let slot = |id: i32| usize::try_from(id).map_or(id, |e| self.slot_of[e]);
        Ok(LayerPlan {
            slot_ids: route.ids.iter().map(|&id| slot(id)).collect(),
            device_weights: route.weights.clone(),
        })
    }

    fn finish_layer(&mut self, _layer: usize) -> Result<Option<Vec<f32>>, EngError> {
        Ok(None)
    }
}

/// A bank of every expert of the sources, expert `e` in slot `slot_of[e]`.
fn full_bank(
    ctx: &CudaContext,
    stream: &DeviceStream,
    s: &Sources,
    slot_of: &[i32],
) -> Result<ExpertBank, EngError> {
    let geometry = ExpertBankGeometry {
        hidden: DIMS.hidden,
        inter: DIMS.inter,
        top_k: DIMS.top_k,
    };
    let bank = ExpertBank::new(ctx, geometry, DIMS.n_expert)?;
    for (expert, &slot) in slot_of.iter().enumerate() {
        let at = expert * SLICE;
        let (gate, up, down) = (
            &s.gate[at..at + SLICE],
            &s.up[at..at + SLICE],
            &s.down[at..at + SLICE],
        );
        bank.fill(stream, usize::try_from(slot).unwrap_or(0), gate, up, down)?;
    }
    Ok(bank)
}

/// The block's weights with the bank stacks in place of the full stacks.
fn bank_weights<'w>(real: &MoeLayerWeights<'w>, bank: &'w ExpertBank) -> MoeLayerWeights<'w> {
    MoeLayerWeights {
        router: real.router,
        router_shared: real.router_shared,
        gate: StackWeights::Native(bank.gate()),
        up: StackWeights::Native(bank.up()),
        down: StackWeights::Native(bank.down()),
        shared_gate: real.shared_gate,
        shared_up: real.shared_up,
        shared_down: real.shared_down,
    }
}

/// The device phase: the native block, then the tiered block over a full bank with the same
/// sources and `host`; both read back as host data after every owner dropped.
fn native_and_tiered(
    s: &Sources,
    slot_of: &[i32],
    host: &mut dyn TieredLayerHost,
) -> Result<(BlockRun, BlockRun, u32), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let weights = rig_weights(&ctx, s)?;
    let native = run_block(&ctx, &stream, s, WeightLayout::Native, &weights)?;
    let real = weights.moe(WeightLayout::Native)?;
    let bank = full_bank(&ctx, &stream, s, slot_of)?;
    let tiered_weights = bank_weights(&real, &bank);
    let b = block_bufs(&ctx, s, WeightLayout::Native)?;
    let bufs = moe_bufs(&b, WeightLayout::Native)?;
    produce(&ctx, &stream, &b, WeightLayout::Native)?;
    stream.synchronize()?;
    let mut scratch = TieredScratch::new(&ctx, DIMS.hidden, DIMS.top_k)?;
    let mut block = TieredBlock {
        weights: &tiered_weights,
        bufs: &bufs,
        scratch: &mut scratch,
        dims: &DIMS,
    };
    run_moe_block_tiered(
        &Launch {
            ctx: &ctx,
            stream: &stream,
        },
        host,
        0,
        &mut block,
    )?;
    stream.synchronize()?;
    let tiered = read_back(&stream, &b)?;
    let mut status = [0u8; 4];
    b.smalls[4].read_at(&stream, 0, &mut status)?;
    Ok((native, tiered, u32::from_le_bytes(status)))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Packet test (git-db's anchor): with every expert resident the tiered block IS the native
/// block, bit for bit - the strongest check of the wiring (the slot ids and weights buffers, the
/// unchanged nine ops over the bank). The empty CPU list adds nothing observable here: a second combine
/// of a zero partial would be bitwise the same, so the launch itself is not pinned by this test.
#[test]
fn tiered_block_with_everything_resident_equals_the_native_block() -> Result<(), EngError> {
    if !device_present("tiered_block_with_everything_resident_equals_the_native_block") {
        return Ok(());
    }
    let s = sources(0x7125_0001);
    let identity: Vec<i32> = (0..DIMS.n_expert as i32).collect();
    let mut host = Resident {
        slot_of: identity.clone(),
    };
    let (native, tiered, status) = native_and_tiered(&s, &identity, &mut host)?;
    assert_eq!(status, 0, "a clean run sets no status bit");
    assert_ne!(
        bits(&native.resid_out),
        bits(&s.resid),
        "the block did something"
    );
    assert_eq!(tiered.ids, native.ids, "the router is untouched");
    assert_eq!(bits(&tiered.weights), bits(&native.weights));
    assert_eq!(bits(&tiered.expert_gu), bits(&native.expert_gu));
    assert_eq!(bits(&tiered.expert_down), bits(&native.expert_down));
    assert_eq!(bits(&tiered.shared_out), bits(&native.shared_out));
    assert_eq!(bits(&tiered.resid_out), bits(&native.resid_out));
    Ok(())
}

/// The same with the experts in a different order in the bank (expert `e` in slot `(e + 5) % 16`):
/// the result must not depend on where an expert sits.
#[test]
fn a_permuted_bank_gives_the_native_bits_too() -> Result<(), EngError> {
    if !device_present("a_permuted_bank_gives_the_native_bits_too") {
        return Ok(());
    }
    let s = sources(0x7125_0002);
    let permuted: Vec<i32> = (0..DIMS.n_expert as i32)
        .map(|e| (e + 5) % DIMS.n_expert as i32)
        .collect();
    let mut host = Resident {
        slot_of: permuted.clone(),
    };
    let (native, tiered, _) = native_and_tiered(&s, &permuted, &mut host)?;
    assert_eq!(bits(&tiered.expert_gu), bits(&native.expert_gu));
    assert_eq!(bits(&tiered.expert_down), bits(&native.expert_down));
    assert_eq!(bits(&tiered.resid_out), bits(&native.resid_out));
    Ok(())
}

/// The kernel-level fact the dummy-slot rule rests on: a combine with a zero-weight eighth term
/// over a finite row leaves every output bit as the combine of the first seven terms (the same
/// arithmetic order, `+= 0 * finite` adds a signed zero to a nonzero sum). The all-zero edge is
/// pinned too: a zero sum turns a -0.0 residual into some zero, so a residual of -0.0 may become
/// +0.0 when every pick is skipped; nonzero residual elements never change.
fn combine_pair(
    res: &[f32],
    rows: &[f32],
    weights: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let n = res.len();
    let run =
        |ctx: &CudaContext, stream: &DeviceStream, top_k: usize| -> Result<Vec<f32>, EngError> {
            let res_dev = f32_buf(ctx, res)?;
            let down = f32_buf(ctx, &rows[..top_k * n])?;
            let w = f32_buf(ctx, &weights[..top_k])?;
            let out = zero_buf(ctx, n * 4)?;
            ctx.moe_combine(
                stream,
                CombineCall {
                    shape: CombineShape {
                        m: 1,
                        n,
                        top_k,
                        ld_down: n,
                    },
                    res: res_dev.range(0, n * 4)?,
                    down: down.range(0, top_k * n * 4)?,
                    weights: w.range(0, top_k * 4)?,
                    shared: None,
                    out: CombineOut::To(out.range(0, n * 4)?),
                },
            )?;
            stream.synchronize()?;
            let mut bytes = vec![0u8; n * 4];
            out.read_at(stream, 0, &mut bytes)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    Ok((run(&ctx, &stream, 8)?, run(&ctx, &stream, 7)?))
}

#[test]
fn a_zero_weight_term_leaves_the_combine_bit_identical_to_omitting_it() -> Result<(), EngError> {
    if !device_present("a_zero_weight_term_leaves_the_combine_bit_identical_to_omitting_it") {
        return Ok(());
    }
    let n = 256;
    let mut rng = super::native_device_rig::Rng::new(0x7125_0003);
    let mut draw = |count: usize| -> Vec<f32> {
        (0..count)
            .map(|_| (rng.below(2000) as f32 - 1000.0) / 250.0)
            .collect()
    };
    let (res, rows) = (draw(n), draw(8 * n));
    let mut weights = vec![0.1f32, 0.2, 0.05, 0.3, 0.15, 0.1, 0.1, 0.0];
    let (with_zero, omitted) = combine_pair(&res, &rows, &weights)?;
    assert_eq!(
        bits(&with_zero),
        bits(&omitted),
        "a zero-weight term must not change a bit"
    );
    // The edge: every weight zero. Nonzero residual values are kept exactly; zero ones stay zero.
    let mut res_with_negative_zero = res.clone();
    res_with_negative_zero[0] = -0.0;
    weights.iter_mut().for_each(|w| *w = 0.0);
    let (all_zero, _) = combine_pair(&res_with_negative_zero, &rows, &weights)?;
    for (got, want) in all_zero.iter().zip(&res_with_negative_zero).skip(1) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    assert!(
        all_zero[0] == 0.0,
        "a -0.0 residual stays some zero when every term is skipped"
    );
    Ok(())
}

/// A router fault passes through the tiered block: the router writes ids -1 and weights 0 and ORs
/// status bit 1, the plan keeps the -1, the expert rows come out NaN and the combine multiplies
/// NaN by weight 0, so the block output is NaN and the step poisons the token.
#[test]
fn a_faulted_router_row_still_gives_nan_and_poisons() -> Result<(), EngError> {
    if !device_present("a_faulted_router_row_still_gives_nan_and_poisons") {
        return Ok(());
    }
    let mut s = sources(0x7125_0004);
    s.router[0] = f32::NAN;
    let identity: Vec<i32> = (0..DIMS.n_expert as i32).collect();
    let mut host = Resident {
        slot_of: identity.clone(),
    };
    let (_, tiered, status) = native_and_tiered(&s, &identity, &mut host)?;
    assert_eq!(
        tiered.ids,
        vec![-1; DIMS.top_k],
        "the router wrote the fault id"
    );
    assert!(tiered.weights.iter().all(|w| *w == 0.0));
    assert_eq!(status & 1, 1, "the router fault bit is set");
    assert!(
        tiered.resid_out.iter().all(|v| v.is_nan()),
        "NaN rows times weight 0 stay NaN"
    );
    Ok(())
}

/// Serves the chosen experts on the CPU (real `CpuPool`, the load's own `tier_kernel`) from the rig's own expert bytes,
/// the rest from the bank; picks on the CPU read slot 0 with weight 0.0.
struct Mixed {
    slot_of: Vec<i32>,
    on_cpu: Vec<usize>,
    pool: CpuPool,
    region: Arc<dyn MappedRegion>,
    pending: Option<PendingLayer>,
}

impl Mixed {
    fn new(s: &Sources, slot_of: Vec<i32>, on_cpu: Vec<usize>) -> Result<Self, EngError> {
        let shape = ExpertShape {
            hidden: DIMS.hidden,
            inter: DIMS.inter,
        };
        let pool = CpuPool::with_kernel(2, shape, tier_kernel())
            .map_err(|e| backend(&format!("{e:?}")))?;
        let bytes: Vec<u8> = [&s.gate, &s.up, &s.down]
            .iter()
            .flat_map(|stack| stack.iter().copied())
            .collect();
        Ok(Self {
            slot_of,
            on_cpu,
            pool,
            region: Arc::new(bytes),
            pending: None,
        })
    }
}

fn backend(message: &str) -> EngError {
    EngError::Backend {
        operation: "tiered_test_host",
        message: message.to_owned(),
    }
}

impl TieredLayerHost for Mixed {
    fn begin_layer(&mut self, _layer: usize, route: &RouteRead) -> Result<LayerPlan, EngError> {
        let stack = DIMS.n_expert * SLICE;
        let (mut slot_ids, mut device_weights, mut jobs) = (Vec::new(), Vec::new(), Vec::new());
        for (&id, &weight) in route.ids.iter().zip(&route.weights) {
            let expert = usize::try_from(id).ok();
            match expert.filter(|e| self.on_cpu.contains(e)) {
                Some(e) => {
                    slot_ids.push(0);
                    device_weights.push(0.0);
                    jobs.push(ExpertJob {
                        region: Arc::clone(&self.region),
                        gate: e * SLICE..(e + 1) * SLICE,
                        up: stack + e * SLICE..stack + (e + 1) * SLICE,
                        down: 2 * stack + e * SLICE..2 * stack + (e + 1) * SLICE,
                        weight,
                    });
                }
                None => {
                    slot_ids.push(expert.map_or(id, |e| self.slot_of[e]));
                    device_weights.push(weight);
                }
            }
        }
        self.pending = self
            .pool
            .start_layer(jobs, &route.hidden)
            .map_err(|e| backend(&format!("{e:?}")))?;
        Ok(LayerPlan {
            slot_ids,
            device_weights,
        })
    }

    fn finish_layer(&mut self, _layer: usize) -> Result<Option<Vec<f32>>, EngError> {
        let Some(pending) = self.pending.take() else {
            return Ok(None);
        };
        pending
            .finish(Duration::from_secs(60))
            .map(Some)
            .map_err(|e| backend(&format!("{e:?}")))
    }
}

/// The largest difference of the two residual deltas relative to the largest native delta.
fn delta_error(resid: &[f32], native: &[f32], tiered: &[f32]) -> f32 {
    let delta = |out: &[f32]| {
        out.iter()
            .zip(resid)
            .map(|(o, r)| o - r)
            .collect::<Vec<f32>>()
    };
    let (a, b) = (delta(native), delta(tiered));
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    worst
        / a.iter()
            .fold(0.0f32, |m, x| m.max(x.abs()))
            .max(f32::MIN_POSITIVE)
}

/// The rule-C bound for `n_cpu` experts served on the CPU, derived from its terms. Both paths put the
/// normed row and the SwiGLU row on groups of 32 with `d = amax / 127` and `q = round(x / d)` (the CPU
/// in f32 with `x * (127 / amax)`, the device's vendored q8_1 by a division), so the grids agree except
/// on exact ties. They differ in what the dot multiplies by: the device stores the scale `d` and the
/// group sum as f16 (relative half-ulp `2^-11 = 4.9e-4` each), the CPU keeps f32. Per expert three
/// roundings act on the output (the scale of the gate/up input, the scale of the SwiGLU row, the f16
/// sum of the min terms) and add linearly at worst: `u = 3 * 2^-11 = 1.5e-3` of the expert's own scale,
/// which bounds its share of the residual delta too (the shared expert and the other picks only dilute
/// it; the device SwiGLU's fast `__expf` is at 1e-6). The errors of different experts are independent,
/// so they add in quadrature: `u * sqrt(n_cpu)`. The f32 summation order of the partial against the
/// device combine adds `n * 2^-23`, orders of magnitude below. Device numbers (3090, 07c73ff8): 6.67e-4
/// at 4 experts (bound 2.9e-3) and 3.23e-4 at 16 (bound 5.9e-3). The Q8_K grid measured 5.34e-3 and
/// 1.35e-2 (133d5519), above both bounds, which is what pins the kernel choice.
fn rule_c_bound(n_cpu: usize) -> f32 {
    3.0 * 2f32.powi(-11) * (n_cpu as f32).sqrt() + n_cpu as f32 * 2f32.powi(-23)
}

/// The pure pin of the bound: above what the per-32 kernel measured, below what the Q8_K kernel did,
/// at both expert counts of the tests. No device needed.
#[test]
fn the_rule_c_bound_sits_between_the_per_32_and_the_q8_k_measurements() {
    for (n_cpu, per_32, q8_k) in [(4usize, 6.67e-4f32, 5.337e-3f32), (16, 3.23e-4, 1.349e-2)] {
        let bound = rule_c_bound(n_cpu);
        assert!(
            bound > 2.0 * per_32,
            "{n_cpu}: bound {bound} leaves no headroom over {per_32}"
        );
        assert!(
            bound < q8_k,
            "{n_cpu}: bound {bound} would accept the Q8_K kernel ({q8_k})"
        );
    }
}

/// Rule C for CPU-served experts: not bitwise (the CPU keeps the group scales in f32, the GPU's q8_1
/// stores them as f16) but inside [`rule_c_bound`] on the residual delta.
#[test]
fn tiered_block_with_cpu_served_experts_matches_the_native_block_within_rule_c()
-> Result<(), EngError> {
    if !device_present(
        "tiered_block_with_cpu_served_experts_matches_the_native_block_within_rule_c",
    ) {
        return Ok(());
    }
    let s = sources(0x7125_0005);
    let identity: Vec<i32> = (0..DIMS.n_expert as i32).collect();
    let mut host = Mixed::new(&s, identity.clone(), vec![1, 4, 9, 12])?;
    let (native, tiered, status) = native_and_tiered(&s, &identity, &mut host)?;
    assert_eq!(status, 0);
    assert!(tiered.resid_out.iter().all(|v| v.is_finite()));
    let error = delta_error(&s.resid, &native.resid_out, &tiered.resid_out);
    println!(
        "RULE-C MEASURED mixed (4 of 16 experts on the CPU): max|tiered-native| / max|native delta| = {error:e}"
    );
    let bound = rule_c_bound(4);
    assert!(
        error <= bound,
        "relative delta error {error} above the bound {bound}"
    );
    assert_ne!(
        bits(&tiered.resid_out),
        bits(&native.resid_out),
        "the CPU path really ran"
    );
    Ok(())
}

/// Every expert on the CPU: every pick reads slot 0 with weight 0.0, the GPU sum is zero and
/// finite, and the CPU partial carries the whole expert term.
#[test]
fn an_all_skipped_list_is_finite_and_agrees_with_the_native_block_by_rule_c() -> Result<(), EngError>
{
    if !device_present("an_all_skipped_list_is_finite_and_agrees_with_the_native_block_by_rule_c") {
        return Ok(());
    }
    let s = sources(0x7125_0006);
    let identity: Vec<i32> = (0..DIMS.n_expert as i32).collect();
    let mut host = Mixed::new(&s, identity.clone(), (0..DIMS.n_expert).collect())?;
    let (native, tiered, status) = native_and_tiered(&s, &identity, &mut host)?;
    assert_eq!(status, 0);
    assert!(tiered.resid_out.iter().all(|v| v.is_finite()));
    let error = delta_error(&s.resid, &native.resid_out, &tiered.resid_out);
    println!(
        "RULE-C MEASURED all skipped (16 of 16 experts on the CPU): max|tiered-native| / max|native delta| = {error:e}"
    );
    let bound = rule_c_bound(DIMS.n_expert);
    assert!(
        error <= bound,
        "relative delta error {error} above the bound {bound}"
    );
    Ok(())
}

/// The block syncs with the host (event wait, blocking writes), so a capture window refuses it
/// with the window's typed error instead of recording half of it.
fn captured_attempt(s: &Sources) -> Result<String, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let weights = rig_weights(&ctx, s)?;
    let real = weights.moe(WeightLayout::Native)?;
    let identity: Vec<i32> = (0..DIMS.n_expert as i32).collect();
    let bank = full_bank(&ctx, &stream, s, &identity)?;
    let tiered_weights = bank_weights(&real, &bank);
    let b = block_bufs(&ctx, s, WeightLayout::Native)?;
    let bufs = moe_bufs(&b, WeightLayout::Native)?;
    produce(&ctx, &stream, &b, WeightLayout::Native)?;
    stream.synchronize()?;
    let mut scratch = TieredScratch::new(&ctx, DIMS.hidden, DIMS.top_k)?;
    let mut host = Resident { slot_of: identity };
    // One eager run loads every kernel module (a first launch inside a window is refused).
    let mut block = TieredBlock {
        weights: &tiered_weights,
        bufs: &bufs,
        scratch: &mut scratch,
        dims: &DIMS,
    };
    run_moe_block_tiered(
        &Launch {
            ctx: &ctx,
            stream: &stream,
        },
        &mut host,
        0,
        &mut block,
    )?;
    stream.synchronize()?;
    let plan = CapturePlan {
        pins: &[],
        holds: Vec::new(),
    };
    let attempt = ctx.capture(&stream, plan, || {
        run_moe_block_tiered(
            &Launch {
                ctx: &ctx,
                stream: &stream,
            },
            &mut host,
            0,
            &mut block,
        )
    });
    Ok(match attempt {
        Ok(_) => String::from("ACCEPTED"),
        Err(error) => format!("{error:?}"),
    })
}

#[test]
fn tiered_block_inside_a_capture_is_refused_with_a_typed_error() -> Result<(), EngError> {
    if !device_present("tiered_block_inside_a_capture_is_refused_with_a_typed_error") {
        return Ok(());
    }
    let answer = captured_attempt(&sources(0x7125_0007))?;
    assert!(
        answer.contains("not allowed inside a capture window"),
        "{answer}"
    );
    Ok(())
}

/// The production fill path (`BankFill`, the device half of every bank change): the three ranges a
/// locator names are cut from the region and land in the slices of the chosen slot, each in its own
/// stack, and nothing else moves. A swapped range, a wrong slot offset or a missing cut shows as a byte
/// difference; the three refusals (no bank for the layer, a slot out of range, a range outside the
/// region) happen before any byte is written.
#[test]
fn bank_fill_writes_the_slices_the_ranges_name_into_the_slot_and_refuses_the_rest()
-> Result<(), EngError> {
    if !device_present(
        "bank_fill_writes_the_slices_the_ranges_name_into_the_slot_and_refuses_the_rest",
    ) {
        return Ok(());
    }
    let region_bytes: Vec<u8> = (0..4 * SLICE)
        .map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8)
        .collect();
    let ranges = ExpertRanges {
        gate: 5..5 + SLICE,
        up: 5 + SLICE + 3..5 + 2 * SLICE + 3,
        down: 5 + 2 * SLICE + 11..5 + 3 * SLICE + 11,
    };
    let region: Arc<dyn MappedRegion> = Arc::new(region_bytes.clone());
    let slot = 3;
    // The device phase has only `?` inside and hands out host data (the isolation split above).
    let (stacks, refusals) = {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let geometry = ExpertBankGeometry {
            hidden: DIMS.hidden,
            inter: DIMS.inter,
            top_k: DIMS.top_k,
        };
        let banks = [ExpertBank::new(&ctx, geometry, DIMS.n_expert)?];
        let mut fill = BankFill {
            banks: &banks,
            stream: &stream,
        };
        fill.fill_slot(0, slot, &ranges, &region)
            .map_err(|error| backend(&format!("{error:?}")))?;
        let outside = ExpertRanges {
            down: 4 * SLICE..5 * SLICE + 64,
            ..ranges.clone()
        };
        let refusals = [
            fill.fill_slot(1, slot, &ranges, &region).is_err(),
            fill.fill_slot(0, DIMS.n_expert, &ranges, &region).is_err(),
            fill.fill_slot(0, 5, &outside, &region).is_err(),
        ];
        stream.synchronize()?;
        let mut stacks = Vec::new();
        for stack in [banks[0].gate(), banks[0].up(), banks[0].down()] {
            let mut bytes = vec![0u8; DIMS.n_expert * SLICE];
            ctx.download(stack.buffer(), &mut bytes)?;
            stacks.push(bytes);
        }
        (stacks, refusals)
    };
    assert_eq!(
        refusals, [true; 3],
        "no layer, slot out of range, range outside"
    );
    for (name, (stack, range)) in ["gate", "up", "down"].iter().zip(stacks.iter().zip([
        &ranges.gate,
        &ranges.up,
        &ranges.down,
    ])) {
        let mine = slot * SLICE..(slot + 1) * SLICE;
        assert_eq!(
            &stack[mine.clone()],
            &region_bytes[range.clone()],
            "{name} slice of slot {slot}"
        );
        let spilled = stack
            .iter()
            .enumerate()
            .any(|(i, &b)| !mine.contains(&i) && b != 0);
        assert!(
            !spilled,
            "{name}: a byte outside the slot (or of a refused fill) moved"
        );
    }
    Ok(())
}

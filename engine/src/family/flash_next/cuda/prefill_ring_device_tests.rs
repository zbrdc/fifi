//! The prefill ring on the device: against a small frozen bank, the ring plans the non-resident
//! experts, streams them in behind a compute stream via events, and the bytes that land in the
//! ring slots are exactly the store's records. Three layers with a ring of two layers' worth of
//! entries exercises the wrap: layer 2 reuses layer 0's slots after `free_layer`.

use super::bank::{
    BankGeometry, ExpertBankState, HostMemory, HostMode, HostPinPlan, HostTerms, RecordSource,
    ReplaceOnMissLru, device::DeviceBank,
};
use super::prefill_ring::device::PrefillRing;
use crate::{EngError, kernels::cuda::device_present};

/// The record of an expert is a pattern of its layer and number, distinct per byte position
/// (the bank device tests' `Pattern`).
struct Pattern;

impl RecordSource for Pattern {
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError> {
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = (index as u32 * 31 + u32::from(expert) * 7 + layer as u32 * 101) as u8;
        }
        Ok(())
    }
}

fn expected(layer: usize, expert: u16, record: usize) -> Vec<u8> {
    let mut out = vec![0u8; record];
    Pattern.record(layer, expert, &mut out).unwrap();
    out
}

/// 3 layers, 12 experts, 4 slots: experts 0..4 resident per layer, 8 streamed per layer, 24
/// entries through a ring of 16 (two layers' worth). Layers 0 and 1 take slots 0..16; layer 2
/// wraps onto 0..8 after layer 0 frees.
#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn the_ring_streams_every_non_resident_expert_in_layer_order() {
    assert!(
        device_present("Flash-Next prefill ring"),
        "a GPU is required"
    );
    let record = 9216;
    let geometry = BankGeometry::new(3, 12, 2, 4, record).unwrap();
    let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
    let counts = vec![(0..12u64).collect::<Vec<_>>(); 3];
    let rows: Vec<&[i32]> = vec![row.as_slice(); 3];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();

    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    let mut bank = DeviceBank::build(ctx, state, &Pattern, &plan).expect("build the bank");

    let compute = ctx.stream().expect("compute stream");
    let mut ring = PrefillRing::build(ctx, &bank, 16).expect("plan the ring");
    assert_eq!(ring.slot_of(0, 4), Some(0));
    assert_eq!(ring.slot_of(0, 11), Some(7));
    assert_eq!(ring.slot_of(1, 4), Some(8));
    assert_eq!(ring.slot_of(2, 4), Some(0), "layer 2 wraps onto layer 0");
    assert_eq!(
        ring.slot_of(0, 3),
        None,
        "a resident expert is not streamed"
    );
    assert_eq!(ring.slot_of(3, 4), None, "past the last layer");

    // Two layers in flight covers every entry; the wrap onto layer 0's slots blocks until
    // layer 0 frees, which the driver cadence does between layers.
    ring.issue_until(ctx, &mut bank, 16).expect("issue");
    ring.wait_layer(&compute, 0).expect("wait layer 0");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(0, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(0, expert, record), "layer 0 expert {expert}");
    }
    // The wrap: after layer 0 frees, layer 2's copies land in its slots, and layer 1's reads
    // (settled earlier) still see layer 1's records.
    ring.free_layer(&compute, 0).expect("free layer 0");
    ring.issue_until(ctx, &mut bank, 24)
        .expect("issue the wrap");
    ring.wait_layer(&compute, 1).expect("wait layer 1");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(1, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(1, expert, record), "layer 1 expert {expert}");
    }
    ring.free_layer(&compute, 1).expect("free layer 1");
    ring.wait_layer(&compute, 2).expect("wait layer 2");
    compute.synchronize().expect("sync");
    for expert in 4..12u16 {
        let slot = usize::from(ring.slot_of(2, expert).unwrap());
        let mut got = vec![0u8; record];
        ring.slots()
            .read_at(&compute, slot * record, &mut got)
            .expect("read the ring slot");
        assert_eq!(got, expected(2, expert, record), "layer 2 expert {expert}");
    }
}

#[test]
#[ignore = "needs a GPU (under 1 GiB)"]
fn freeing_out_of_order_is_refused() {
    assert!(
        device_present("Flash-Next prefill ring"),
        "a GPU is required"
    );
    let record = 9216;
    let geometry = BankGeometry::new(2, 12, 2, 4, record).unwrap();
    let row: Vec<i32> = (0..12).map(|e| if e < 4 { e } else { -1 }).collect();
    let counts = vec![(0..12u64).collect::<Vec<_>>(); 2];
    let rows: Vec<&[i32]> = vec![row.as_slice(); 2];
    let policy = ReplaceOnMissLru::from_fill(geometry, &rows, &counts, None).unwrap();
    let state = ExpertBankState::with_policy(geometry, HostMode::Exclusive, policy).unwrap();
    let projector = crate::family::project::cuda::CudaProjector::new().expect("projector");
    let ctx = projector.context();
    let plan = HostPinPlan::new(
        &geometry,
        HostMode::Exclusive,
        &HostMemory {
            total: 62 << 30,
            available: 55 << 30,
        },
        &HostTerms::DEFAULT,
    )
    .expect("the host plan");
    let bank = DeviceBank::build(ctx, state, &Pattern, &plan).expect("build the bank");
    let compute = ctx.stream().expect("compute stream");
    let mut ring = PrefillRing::build(ctx, &bank, 16).expect("plan the ring");
    assert!(ring.free_layer(&compute, 1).is_err());
}

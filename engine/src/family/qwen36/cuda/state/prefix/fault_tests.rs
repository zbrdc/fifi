//! A real sticky fault must poison import without publishing or freeing native owners.
use super::super::super::{
    book::Health, common_tests::dims, paging_cow_tests::write_loaded, paging_store::PageId,
};
use super::super::paged_prefix_tests::{options, pages, seeded_state, snapshot_budget};
use super::PrefixState;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, assert_cuda_fault_child_finishes, device_present, test_inject_unmapped_fault,
    },
};

mod terminal;

const CHILD: &str = "family::qwen36::cuda::state::prefix::fault_tests::prefix_import_fault_child";
const READ_CHILD: &str =
    "family::qwen36::cuda::state::prefix::fault_tests::state_copy_read_fault_child";
const WRITE_CHILD: &str =
    "family::qwen36::cuda::state::prefix::fault_tests::state_copy_write_fault_child";

#[test]
fn state_copy_read_fault_does_not_publish_host_data_in_child() {
    if !device_present("state_copy_read_fault_does_not_publish_host_data_in_child") {
        return;
    }
    assert_cuda_fault_child_finishes(READ_CHILD);
    println!("PKV_PREFIX_READ_FAULT_CHILD_PASS real_fault700=true");
}

#[test]
fn state_copy_write_fault_preserves_snapshot_source_in_child() {
    if !device_present("state_copy_write_fault_preserves_snapshot_source_in_child") {
        return;
    }
    assert_cuda_fault_child_finishes(WRITE_CHILD);
    println!("PKV_PREFIX_WRITE_FAULT_CHILD_PASS real_fault700=true");
}

#[test]
fn state_copy_read_fault_child() {
    if std::env::var("INFERENCE_CUDA_FAULT_CHILD").is_ok_and(|v| v == "1") {
        copy_fault(false);
    } else {
        println!("SKIPPED state_copy_read_fault_child: not a fault child");
    }
}

#[test]
fn state_copy_write_fault_child() {
    if std::env::var("INFERENCE_CUDA_FAULT_CHILD").is_ok_and(|v| v == "1") {
        copy_fault(true);
    } else {
        println!("SKIPPED state_copy_write_fault_child: not a fault child");
    }
}

fn copy_fault(upload: bool) {
    let ctx = CudaContext::new().expect("copy fault context");
    let stream = ctx.stream().expect("copy fault stream");
    let (_, limit) = snapshot_budget(&dims(), &options()).expect("copy fault budget");
    let (state, _) = seeded_state(&ctx, &stream, limit, 37).expect("copy fault state");
    let saved = state
        .export_prefix(&ctx, &stream)
        .expect("copy fault saved prefix");
    let PrefixState::Paged(snapshot) = &saved else {
        panic!("paged snapshot");
    };
    let held = pages(&state).expect("store").owned_host_bytes();
    let live = ctx.live_owners();
    let device_bytes = ctx.live_bytes();
    let destroys = ctx.destroy_calls();
    {
        let mut host = [0xa5; 4096];
        let _read_charge = if upload {
            None
        } else {
            Some(
                snapshot
                    .pages
                    .charge_host(host.len())
                    .expect("admit output bytes"),
            )
        };
        assert_eq!(test_inject_unmapped_fault(), Err(700));
        assert!(
            ctx.ensure_usable().is_ok(),
            "copy must map the real unmapped fault"
        );
        let mut store = pages(&state).expect("store");
        let range = state
            .delta
            .range(0, host.len())
            .expect("actual DeltaNet range");
        let copied = if upload {
            store.write_state(
                &ctx,
                &stream,
                &snapshot.delta[0].recurrent[..host.len()],
                range,
            )
        } else {
            store.read_state(&ctx, &stream, range, &mut host)
        };
        let error = copied.expect_err("real CUDA copy/completion failure");
        assert!(matches!(error, EngError::Backend { .. }));
        assert!(format!("{error:?}").contains("700"));
        assert!(ctx.ensure_usable().is_err());
        assert_eq!(
            host, [0xa5; 4096],
            "failed D2H never publishes from staging"
        );
        assert!(
            store.resident(0).is_err(),
            "state transfer does not publish a KV page"
        );
    }
    assert_eq!(pages(&state).expect("store").owned_host_bytes(), held);
    assert!(
        snapshot
            .delta
            .iter()
            .all(|layer| { layer.recurrent.iter().chain(&layer.conv).all(|b| *b == 1) })
    );
    drop(state);
    assert_eq!(ctx.destroy_calls(), destroys);
    assert_eq!(ctx.live_owners(), live);
    assert_eq!(ctx.live_bytes(), device_bytes);
    drop(saved);
    println!("fault child: done");
}

#[test]
fn prefix_import_fault_retains_snapshots_and_native_owners_in_child() {
    if !device_present("prefix_import_fault_retains_snapshots_and_native_owners_in_child") {
        return;
    }
    assert_cuda_fault_child_finishes(CHILD);
    println!("PKV_PREFIX_FAULT_CHILD_PASS real_fault700=true");
}

#[test]
fn prefix_import_fault_child() {
    if !std::env::var("INFERENCE_CUDA_FAULT_CHILD").is_ok_and(|v| v == "1") {
        println!("SKIPPED prefix_import_fault_child: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("fault fixture context");
    let stream = ctx.stream().expect("fault fixture stream");
    let (_, limit) = snapshot_budget(&dims(), &options()).expect("bounded fault fixture");
    let (mut state, _) = seeded_state(&ctx, &stream, limit, 37).expect("real nonzero state");
    let saved = state
        .export_prefix(&ctx, &stream)
        .expect("snapshot before real fault");
    pages(&state)
        .expect("store")
        .load(
            &ctx,
            &stream,
            0,
            PageId {
                full_layer: 0,
                group: 0,
            },
        )
        .expect("occupied slot");
    write_loaded(&ctx, &stream, &pages(&state).expect("store"), 2).expect("real divergent slot");
    let held = pages(&state).expect("store").owned_host_bytes();
    let live = ctx.live_owners();
    let device_bytes = ctx.live_bytes();
    let destroys = ctx.destroy_calls();
    assert_eq!(test_inject_unmapped_fault(), Err(700));
    assert!(
        ctx.ensure_usable().is_ok(),
        "the native fault is still unmapped"
    );
    let error = state
        .import_prefix(&ctx, &stream, &saved)
        .expect_err("real failed completion");
    assert!(matches!(error, EngError::Backend { .. }));
    assert!(format!("{error:?}").contains("700"));
    assert_eq!(state.health(), Health::Poisoned);
    assert_eq!(state.tokens(), 37);
    assert!(ctx.ensure_usable().is_err());
    assert_eq!(pages(&state).expect("store").owned_host_bytes(), held);
    assert!(
        pages(&state).expect("store").resident(0).is_ok(),
        "failed proof keeps occupancy"
    );
    assert!(state.export_prefix(&ctx, &stream).is_err());
    assert_eq!(pages(&state).expect("store").owned_host_bytes(), held);
    let PrefixState::Paged(snapshot) = &saved else {
        panic!("paged snapshot");
    };
    assert_eq!(snapshot.tokens, 37);
    assert!(
        snapshot
            .delta
            .iter()
            .all(|layer| { layer.recurrent.iter().chain(&layer.conv).all(|b| *b == 1) })
    );
    drop(state);
    assert_eq!(
        ctx.destroy_calls(),
        destroys,
        "terminal drop must not call native destroy"
    );
    assert_eq!(
        ctx.live_owners(),
        live,
        "unreleased native allocations remain counted"
    );
    assert_eq!(ctx.live_bytes(), device_bytes);
    drop(saved);
    assert_eq!(ctx.destroy_calls(), destroys);
    println!("fault child: done");
}

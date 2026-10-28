//! Standalone allocation must report an actual poisoned context before host admission.
use crate::{
    EngError,
    family::qwen36::cuda::{common_tests::dims, paging_store::PagedKv},
    kernels::cuda::{
        CudaContext, assert_cuda_fault_child_finishes, device_present, test_inject_unmapped_fault,
    },
};

const CHILD: &str = "family::qwen36::cuda::state::prefix::fault_tests::terminal::paged_allocation_terminal_fault_child";

#[test]
fn paged_allocation_reports_terminal_before_host_admission_in_child() {
    if !device_present("paged_allocation_reports_terminal_before_host_admission_in_child") {
        return;
    }
    assert_cuda_fault_child_finishes(CHILD);
    println!("PKV_TERMINAL_ADMISSION_PASS actual_fault700=true");
}

#[test]
fn paged_allocation_terminal_fault_child() {
    if !std::env::var("INFERENCE_CUDA_FAULT_CHILD").is_ok_and(|v| v == "1") {
        println!("SKIPPED terminal admission: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("terminal admission context");
    let stream = ctx.stream().expect("healthy stream");
    assert_eq!(test_inject_unmapped_fault(), Err(700));
    let fault = stream.synchronize().expect_err("map real fault700");
    assert!(matches!(fault, EngError::Backend { .. }));
    assert!(format!("{fault:?}").contains("700"));
    assert!(ctx.ensure_usable().is_err());
    let before = (ctx.live_owners(), ctx.live_bytes(), ctx.destroy_calls());
    let error = PagedKv::allocate(&ctx, &dims(), 320, 0)
        .err()
        .expect("poisoned context cannot allocate");
    println!("PKV_TERMINAL_ADMISSION_READY actual_fault700=true observed={error:?}");
    assert!(
        matches!(
            error,
            EngError::Backend {
                operation: "execute",
                ..
            }
        ),
        "terminal allocation must precede host quota admission"
    );
    assert_eq!(
        (ctx.live_owners(), ctx.live_bytes(), ctx.destroy_calls()),
        before
    );
    println!("fault child: done");
}

//! Normal upload ownership and real allocation-stage failure; no synthetic failure hook.
use super::{
    CHILD_ENV, CUDA_ERROR_ILLEGAL_ADDRESS, CudaContext, EngError, Ordering, TERMINAL,
    assert_child_finishes, device_present, native, test_inject_unmapped_fault,
};

const CHILD: &str = "kernels::cuda::fault::upload_cleanup::upload_allocation_fault_child";

#[test]
fn healthy_upload_roundtrips_transfers_ownership_and_rejects_empty_input() {
    if !device_present("healthy_upload_roundtrips_transfers_ownership_and_rejects_empty_input") {
        return;
    }
    let ctx = CudaContext::new().expect("healthy upload context");
    let before_bytes = ctx.live_bytes();
    let before = native::allocation_calls();
    assert!(matches!(ctx.upload(&[]), Err(EngError::BadRequest(_))));
    assert_eq!(native::allocation_calls(), before);
    let data: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let device = ctx.upload(&data).expect("healthy real upload");
    let allocated = native::allocation_calls();
    let mut got = vec![0; data.len()];
    ctx.download(&device, &mut got)
        .expect("healthy upload readback");
    drop(device);
    assert_eq!(got, data);
    assert_eq!(allocated, (before.0 + 1, before.1));
    assert_eq!(native::allocation_calls(), (before.0 + 1, before.1 + 1));
    assert_eq!(ctx.live_bytes(), before_bytes);
    println!("PKV_UPLOAD_HEALTHY_PASS roundtrip=true ownership=true");
}

#[test]
fn upload_allocation_failure_preserves_first_stage_and_does_not_free_in_child() {
    if !device_present("upload_allocation_failure_preserves_first_stage_and_does_not_free_in_child")
    {
        return;
    }
    assert_child_finishes(CHILD);
    println!("PKV_UPLOAD_ALLOCATION_FAULT_PASS real_fault700=true");
}

#[test]
fn upload_allocation_fault_child() {
    if !std::env::var(CHILD_ENV).is_ok_and(|v| v == "1") {
        println!("SKIPPED upload_allocation_fault_child: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("upload fault context");
    let existing = ctx.allocate(256).expect("healthy existing owner");
    let before_bytes = ctx.live_bytes();
    let before = native::allocation_calls();
    assert_eq!(
        test_inject_unmapped_fault(),
        Err(CUDA_ERROR_ILLEGAL_ADDRESS)
    );
    assert!(!TERMINAL.load(Ordering::Acquire));
    let error = match ctx.upload(&[1; 4096]) {
        Err(error) => error,
        Ok(_) => panic!("sticky allocation must fail"),
    };
    let EngError::Backend { operation, message } = error else {
        panic!("real sticky fault must be a backend error");
    };
    assert_eq!(
        operation, "cuda allocate",
        "preserve the first failed stage"
    );
    assert!(message.contains("700"));
    assert!(TERMINAL.load(Ordering::Acquire));
    assert_eq!(native::allocation_calls(), (before.0 + 1, before.1));
    drop(existing);
    assert_eq!(native::allocation_calls(), (before.0 + 1, before.1));
    assert_eq!(ctx.live_bytes(), before_bytes);
    println!("fault child: done");
}

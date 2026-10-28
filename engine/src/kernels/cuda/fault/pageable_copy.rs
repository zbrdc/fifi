//! Real sticky faults test owned staging retention, not post-enqueue failure.
use super::{
    CHILD_ENV, CUDA_ERROR_ILLEGAL_ADDRESS, CudaContext, EngError, Ordering, TERMINAL,
    assert_child_finishes, device_present, test_inject_unmapped_fault,
};
use crate::kernels::cuda::native::resident::retained_pageable_bytes;

const READ_CHILD: &str = "kernels::cuda::fault::pageable_copy::pageable_read_fault_child";
const WRITE_CHILD: &str = "kernels::cuda::fault::pageable_copy::pageable_write_fault_child";

#[test]
fn failed_pageable_read_retains_owned_storage_in_child() {
    if !device_present("failed_pageable_read_retains_owned_storage_in_child") {
        return;
    }
    assert_child_finishes(READ_CHILD);
    println!("PKV_PAGEABLE_READ_FAULT_PASS real_sticky700=true");
}

#[test]
fn failed_pageable_write_retains_owned_storage_in_child() {
    if !device_present("failed_pageable_write_retains_owned_storage_in_child") {
        return;
    }
    assert_child_finishes(WRITE_CHILD);
    println!("PKV_PAGEABLE_WRITE_FAULT_PASS real_sticky700=true");
}

fn fault_copy(read: bool) {
    if !std::env::var(CHILD_ENV).is_ok_and(|v| v == "1") {
        println!("SKIPPED pageable fault: not a fault child");
        return;
    }
    let ctx = CudaContext::new().expect("pageable fault context");
    let stream = ctx.stream().expect("healthy stream");
    let buffer = ctx.allocate(4096).expect("healthy real device allocation");
    let source = vec![0x39; 4096];
    buffer
        .write_at(&stream, 0, &source)
        .expect("healthy control upload");
    let before = retained_pageable_bytes();
    let device_before = ctx.live_bytes();
    assert_eq!(
        test_inject_unmapped_fault(),
        Err(CUDA_ERROR_ILLEGAL_ADDRESS)
    );
    assert!(!TERMINAL.load(Ordering::Acquire));
    let mut output = vec![0xA7; 4096];
    let result = if read {
        buffer.read_at(&stream, 0, &mut output)
    } else {
        buffer.write_at(&stream, 0, &source)
    };
    let Err(EngError::Backend { operation, message }) = result else {
        panic!("real sticky copy must be a backend error");
    };
    assert_eq!(operation, "cuda copy", "preserve the first copy failure");
    assert!(message.contains("700"));
    assert!(TERMINAL.load(Ordering::Acquire));
    assert!(
        output.iter().all(|v| *v == 0xA7),
        "failed D2H must not publish staging"
    );
    assert!(source.iter().all(|v| *v == 0x39));
    println!("PKV_PAGEABLE_FAULT_READY read={read} real_sticky700=true");
    assert_eq!(
        retained_pageable_bytes() - before,
        4096,
        "native pageable staging is retained while completion is unknown"
    );
    drop(buffer);
    assert_eq!(
        ctx.live_bytes(),
        device_before,
        "terminal device owner remains charged"
    );
    println!("fault child: done");
}

#[test]
fn pageable_read_fault_child() {
    fault_copy(true);
}

#[test]
fn pageable_write_fault_child() {
    fault_copy(false);
}

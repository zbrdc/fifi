//! The device test of the MMQ facts binding: the first glue FFI child proves the glue archive
//! links and resolves by reading the 3090's own facts through the C entry. The asserted values
//! are the 3090's: sm_86 cubins (the built architecture the facts report, not the device's
//! capability), 82 SMs, 99 KiB of opt-in shared memory per block.

use crate::kernels::cuda::testkit;

#[test]
fn mmq_facts_read_the_3090() {
    if !testkit::device_present("mmq_facts_read_the_3090") {
        return;
    }
    let phase = || -> Result<(i32, usize, usize), crate::EngError> {
        // The fixed API: a context method with the usable check and the dispatch guard, like
        // `device_is_integrated`.
        let ctx = crate::kernels::cuda::CudaContext::new()?;
        let facts = ctx.mmq_device_facts()?;
        Ok((facts.cc, facts.nsm, facts.smpbo))
    };
    let (cc, nsm, smpbo) = phase().expect("the device phase");
    // The hard gate first: every value is in its valid domain (a sentinel or a wrapped
    // negative would fail here) before the equality pins.
    assert!((700..=1000).contains(&cc), "cc {cc} is not an architecture");
    assert!((1..=1024).contains(&nsm), "nsm {nsm} is out of range");
    assert!(
        (32_768..=(1 << 20)).contains(&smpbo),
        "smpbo {smpbo} is out of range"
    );
    assert_eq!(cc, 860, "the 3090 runs the sm_86 cubins");
    assert_eq!(nsm, 82, "the 3090 has 82 SMs");
    assert_eq!(smpbo, 101_376, "the 3090 offers 99 KiB opt-in per block");
    println!("MMQ facts: cc {cc}, nsm {nsm}, smpbo {smpbo}");
}

#[test]
fn mmq_facts_are_queried_once_per_context() {
    // The facts are read from the device once per context and every later call returns the
    // cached value, so a launch does not pay five attribute calls. The cache is per context,
    // not per process: a second context queries again.
    if !testkit::device_present("mmq_facts_are_queried_once_per_context") {
        return;
    }
    let phase = || -> Result<[(super::MmqFacts, usize); 4], crate::EngError> {
        let ctx = crate::kernels::cuda::CudaContext::new()?;
        let first = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        let second = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        let third = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        drop(ctx);
        let next = crate::kernels::cuda::CudaContext::new()?;
        let fresh = (next.mmq_device_facts()?, next.mmq_facts.queries.get());
        drop(next);
        Ok([first, second, third, fresh])
    };
    let [first, second, third, fresh] = phase().expect("the device phase");
    assert_eq!(first.0.cc, 860, "the 3090 runs the sm_86 cubins");
    assert_eq!(first.1, 1, "the first call queries the device");
    assert_eq!(
        (second.0, second.1),
        (first.0, 1),
        "the second call is cached"
    );
    assert_eq!((third.0, third.1), (first.0, 1), "the third call is cached");
    assert_eq!(
        (fresh.0, fresh.1),
        (first.0, 1),
        "a new context queries once itself"
    );
    println!("MMQ facts: three calls, one query; a fresh context one query");
}

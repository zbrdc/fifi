//! Native CoW refusals preserve occupied slots; clean attention must never detach.
use super::{
    common_tests::dims,
    paging_cow_tests::{read_loaded, read_page, write_loaded, write_page},
    paging_plan::PagingPlan,
    paging_store::{PageAttention, PageChunk, PageId, PagedKv},
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, device_present, resident::attn_decode_q8::AttnQ8Shape},
};

fn observe_clean_attention() -> Result<(bool, bool, bool), EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    // No room for a detached group: even a temporary unnecessary copy must fail.
    let limit = plan.backing_bytes + plan.pinned_bytes + (64 << 10);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, limit)?;
    write_page(&ctx, &stream, &mut store, 1)?;
    let base = store.owned_host_bytes();
    let saved = store.snapshot(&ctx, &stream, 37)?;
    let held = store.owned_host_bytes();
    let a = &dimensions.shape.attn;
    let bytes = a.heads_q * a.head_dim * size_of::<f32>();
    let q = ctx.allocate(bytes)?;
    q.write_at(&stream, 0, &vec![0; bytes])?;
    let out = ctx.allocate(bytes)?;
    let mut scratch = ctx.paged_q8_scratch(
        AttnQ8Shape {
            m: 1,
            n_head: a.heads_q,
            n_kv_head: a.heads_kv,
            head_dim: a.head_dim,
            cells: 320,
            ld: a.heads_q * a.head_dim * 2,
        },
        plan.group_cells,
    )?;
    eprintln!("PKV_COW_CLEAN_ATTENTION_REAL_DEVICE_READY");
    let mut unchanged = true;
    for _ in 0..3 {
        store.attend(
            &ctx,
            &stream,
            &mut scratch,
            PageAttention {
                full_layer: 0,
                visible_keys: 37,
                scale: 1.0 / 16.0,
                q: q.range(0, bytes)?,
                proj: None,
                out: out.range(0, bytes)?,
            },
        )?;
        unchanged &= store.owned_host_bytes() == held;
    }
    let mut result = vec![0; bytes];
    out.read_at(&stream, 0, &mut result)?;
    // The all-0x01 page has V codes 1 and fp16 subnormal scales 0x0101.
    let expected = 257.0_f64 / 16_777_216.0;
    let accurate = result
        .as_chunks::<4>()
        .0
        .iter()
        .all(|b| (f64::from(f32::from_le_bytes(*b)) - expected).abs() < 1e-9);
    drop(saved);
    Ok((unchanged, accurate, store.owned_host_bytes() == base))
}

#[test]
fn clean_attention_does_not_detach_a_shared_page_under_tight_quota() {
    if !device_present("clean_attention_does_not_detach_a_shared_page_under_tight_quota") {
        return;
    }
    let (unchanged, accurate, released) =
        observe_clean_attention().expect("shared read-only attention");
    assert!(unchanged && accurate && released);
}

struct Refusals {
    quota: bool,
    foreign_domain: bool,
    wrong_stream: bool,
    range: bool,
    unchanged: bool,
    occupied: bool,
    device_bytes: bool,
    recovered: bool,
    released: bool,
}

fn observe_refusals() -> Result<Refusals, EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let limit = plan.backing_bytes + plan.pinned_bytes + (64 << 10);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let other_stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, limit)?;
    let original = write_page(&ctx, &stream, &mut store, 1)?;
    let base = store.owned_host_bytes();
    let saved = store.snapshot(&ctx, &stream, 37)?;
    let mut other = PagedKv::allocate(&ctx, &dimensions, 320, 8 << 20)?;
    let foreign = other.snapshot(&ctx, &stream, 37)?;
    let page = PageId {
        full_layer: 0,
        group: 0,
    };
    store.load(&ctx, &stream, 0, page)?;
    let changed = write_loaded(&ctx, &stream, &store, 2)?;
    let held = store.owned_host_bytes();
    let queued = (stream.enqueued(), other_stream.enqueued());
    eprintln!("PKV_COW_REFUSALS_REAL_DEVICE_READY");
    let quota = matches!(
        store.writeback(&ctx, &stream, 0),
        Err(EngError::MemoryPlan(_))
    );
    let foreign_domain = matches!(
        store.restore(&stream, &foreign),
        Err(EngError::BadRequest(_))
    );
    let wrong_stream = matches!(
        store.restore(&other_stream, &saved),
        Err(EngError::BadRequest(_))
    );
    let range = matches!(
        store.snapshot(&ctx, &stream, 321),
        Err(EngError::BadRequest(_))
    );
    let occupied = matches!(
        store.load(&ctx, &stream, 0, page),
        Err(EngError::BadRequest(_))
    );
    let unchanged =
        store.owned_host_bytes() == held && (stream.enqueued(), other_stream.enqueued()) == queued;
    let device_bytes = read_loaded(&ctx, &stream, &store)? == changed;
    store.restore(&stream, &saved)?;
    drop(saved);
    let recovered = read_page(&ctx, &stream, &mut store)? == original;
    let released = store.owned_host_bytes() == base;
    Ok(Refusals {
        quota,
        foreign_domain,
        wrong_stream,
        range,
        unchanged,
        occupied,
        device_bytes,
        recovered,
        released,
    })
}

#[test]
fn cow_quota_and_identity_refusals_preserve_slot_bytes_and_allow_restore() {
    if !device_present("cow_quota_and_identity_refusals_preserve_slot_bytes_and_allow_restore") {
        return;
    }
    let x = observe_refusals().expect("real occupied-slot CoW refusals");
    assert!(x.quota && x.foreign_domain && x.wrong_stream && x.range);
    assert!(x.unchanged && x.occupied && x.device_bytes);
    assert!(x.recovered && x.released);
}

fn observe_refused_append() -> Result<(), EngError> {
    let dimensions = dims();
    let plan = PagingPlan::new(&dimensions, 320)?;
    let limit = plan.backing_bytes + plan.pinned_bytes + (64 << 10);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut store = PagedKv::allocate(&ctx, &dimensions, 320, limit)?;
    write_page(&ctx, &stream, &mut store, 1)?;
    let base = store.owned_host_bytes();
    let saved = store.snapshot(&ctx, &stream, 37)?;
    let page = PageId {
        full_layer: 0,
        group: 0,
    };
    let short = ctx.allocate(4)?;
    let full = ctx.allocate(dimensions.shape.attn.heads_kv * dimensions.shape.attn.head_dim * 4)?;
    let mut changed = Vec::new();
    for dirty in [false, true] {
        store.load(&ctx, &stream, 0, page)?;
        if dirty {
            changed = write_loaded(&ctx, &stream, &store, 2)?;
        }
        let before = (stream.enqueued(), store.owned_host_bytes());
        let refusal = store.append_chunk(
            &ctx,
            &stream,
            0,
            PageChunk {
                page,
                position: 0,
                rows: 1,
                k: short.range(0, short.bytes())?,
                v: full.range(0, full.bytes())?,
            },
        );
        assert!(matches!(refusal, Err(EngError::BadRequest(_))));
        assert_eq!((stream.enqueued(), store.owned_host_bytes()), before);
        println!("PKV_APPEND_REFUSAL_READY dirty={dirty} bad_request=true no_enqueue=true");
        let writeback = store.writeback(&ctx, &stream, 0);
        if dirty {
            assert!(
                matches!(writeback, Err(EngError::MemoryPlan(_))),
                "an already dirty shared page must still require CoW admission"
            );
        } else {
            assert!(
                writeback.is_ok(),
                "refused append must preserve clean-slot writeback"
            );
        }
    }
    drop(saved);
    store.writeback(&ctx, &stream, 0)?;
    assert!(read_page(&ctx, &stream, &mut store)? == changed);
    assert_eq!(store.owned_host_bytes(), base);
    println!("PKV_APPEND_REFUSAL_PASS clean=true dirty=true recovery=true");
    Ok(())
}

#[test]
fn refused_append_preserves_clean_and_already_dirty_cow_state() {
    if !device_present("refused_append_preserves_clean_and_already_dirty_cow_state") {
        return;
    }
    observe_refused_append().expect("real tight-quota append refusal");
}

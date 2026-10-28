// Synthetic session routing, not the approved real-VRAM-pressure model acceptance.
mod cache_routes {
    use super::{counts, options};
    use crate::{
        EngError,
        family::{
            project::cuda::WeightLayout,
            qwen36::cuda::{
                book::Health,
                common_tests::dims,
                dims::KvMode,
                paging_pipeline_tests::{floats, inputs, plane, query_and_gate, read, upload},
                paging_plan::PagingPlan,
                paging_store::{
                    PageAttention, PageChunk, PageId, PageToken, PagedKv,
                    device_first::{CachedAttention, DeviceCachePlan},
                },
                state::{PagedStatePlan, ResidentState, StepField},
                state_layout::{KvPool, Plane, StateLayout, Word},
            },
        },
        kernels::cuda::{
            CudaContext, device_present,
            resident::{
                attn_decode::{AttnSpan, AttnSplit},
                attn_decode_q8::{AttnDecodeQ8Call, AttnQ8Shape, attn_q8_part_bytes},
                kv8_append::{Kv8CodeScale, Kv8Data, Kv8Shape},
                step_params::StepParams,
            },
        },
    };

    struct Observed {
        allocation: bool,
        logical: bool,
        outputs: bool,
        words: bool,
        reset: bool,
    }

    fn observe_routes() -> Result<Observed, EngError> {
        const N: usize = 1280;
        let dimensions = dims();
        let mut opts = options(KvMode::Int8G32, WeightLayout::Native);
        opts.context_tokens = N;
        let logical = StateLayout::new(&dimensions, &opts)?;
        let paging = PagingPlan::new(&dimensions, N)?;
        let cache_rows = 2 * paging.group_cells;
        let cache = DeviceCachePlan::new(
            &dimensions,
            N,
            2 * paging.group.total * paging.full_layers,
            32 << 20,
        )?;
        let planned = PagedStatePlan::new_cached(&dimensions, &opts, cache)?;
        let expected_bytes = planned.device_bytes()?;
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let before = counts(&ctx);
        let mut state = ResidentState::allocate_from_plan(&ctx, &stream, &planned)?;
        let allocation = counts(&ctx)[0] - before[0] == expected_bytes;
        let logical = *state.layout() == logical
            && state.cached_store().is_some()
            && state.kv_plane(0, Plane::K).is_err()
            && state.export_state(&stream).is_err();
        let (k, v) = inputs(N);
        let k = upload(&ctx, &stream, &floats(&k))?;
        let v = upload(&ctx, &stream, &floats(&v))?;
        let (q, gate) = query_and_gate();
        let q = upload(&ctx, &stream, &floats(&q))?;
        let gate = upload(&ctx, &stream, &floats(&gate))?;
        let shape = AttnQ8Shape {
            m: 1,
            n_head: 16,
            n_kv_head: 2,
            head_dim: 256,
            cells: N,
            ld: 8192,
        };
        let mut scratch = ctx.paged_q8_scratch(shape, paging.group_cells)?;
        let mut host_scratch = ctx.paged_q8_scratch(shape, paging.group_cells)?;
        let actual = upload(&ctx, &stream, &vec![0; 16 * 256 * 4])?;
        let expected = upload(&ctx, &stream, &vec![0; actual.bytes()])?;
        let native_pool = KvPool::new(&dimensions, KvMode::Int8G32, cache_rows)?;
        let native = upload(&ctx, &stream, &vec![0; native_pool.total])?;
        let part_bytes = attn_q8_part_bytes(1, 2, 256, AttnSplit::Q8.max_splits)?;
        let part = ctx.allocate(part_bytes)?;
        let row_bytes = 2 * 256 * 4;
        ctx.kv8_chunk(
            &stream,
            Kv8Shape {
                rows: cache_rows,
                n_kv_head: 2,
                head_dim: 256,
                cells: cache_rows,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: k.range(0, cache_rows * row_bytes)?,
                v: v.range(0, cache_rows * row_bytes)?,
                k_codes: plane(&native, &native_pool, Plane::K)?,
                k_scales: plane(&native, &native_pool, Plane::KScales)?,
                v_codes: plane(&native, &native_pool, Plane::V)?,
                v_scales: plane(&native, &native_pool, Plane::VScales)?,
            },
            0,
        )?;
        let mut host = PagedKv::allocate(&ctx, &dimensions, N, 32 << 20)?;
        for group in 0..paging.groups {
            let position = group * paging.group_cells;
            let page = PageId {
                full_layer: 0,
                group,
            };
            host.load(&ctx, &stream, 0, page)?;
            host.append_chunk(
                &ctx,
                &stream,
                0,
                PageChunk {
                    page,
                    position,
                    rows: paging.group_cells,
                    k: k.range(position * row_bytes, paging.group_cells * row_bytes)?,
                    v: v.range(position * row_bytes, paging.group_cells * row_bytes)?,
                },
            )?;
            host.writeback(&ctx, &stream, 0)?;
        }
        let mut outputs = true;
        let mut words = true;
        for position in 0..N {
            let params = StepParams::new(1, position as i32)?.to_bytes();
            let block = state.word(Word::StepParams)?;
            block.buffer().write_at(&stream, block.offset(), &params)?;
            let store = state
                .cached_store()
                .ok_or_else(|| EngError::BadRequest("missing cached state".into()))?;
            let mut store = store
                .try_borrow_mut()
                .map_err(|_| EngError::BadRequest("cached state borrowed".into()))?;
            store.append(
                &ctx,
                &stream,
                PageToken {
                    page: PageId {
                        full_layer: 0,
                        group: position / paging.group_cells,
                    },
                    position,
                    params: block,
                    k: k.range(position * row_bytes, row_bytes)?,
                    v: v.range(position * row_bytes, row_bytes)?,
                },
            )?;
            let visible = position + 1;
            if ![1, 512, 513, 1024, N].contains(&visible) {
                continue;
            }
            store.attend(
                &ctx,
                &stream,
                &mut scratch,
                CachedAttention {
                    page: PageAttention {
                        full_layer: 0,
                        visible_keys: visible,
                        scale: 1.0 / 16.0,
                        q: q.range(0, q.bytes())?,
                        proj: Some(gate.range(0, gate.bytes())?),
                        out: actual.range(0, actual.bytes())?,
                    },
                    kv_len: state.step_field(StepField::KvLen)?,
                },
            )?;
            if visible <= cache_rows {
                ctx.attn_decode_q8(
                    &stream,
                    AttnDecodeQ8Call {
                        shape: AttnQ8Shape {
                            cells: cache_rows,
                            ..shape
                        },
                        span: AttnSpan::FULL,
                        split: AttnSplit::Q8,
                        scale: 1.0 / 16.0,
                        q: q.range(0, q.bytes())?,
                        proj: Some(gate.range(0, gate.bytes())?),
                        k_codes: plane(&native, &native_pool, Plane::K)?,
                        k_scales: plane(&native, &native_pool, Plane::KScales)?,
                        v_codes: plane(&native, &native_pool, Plane::V)?,
                        v_scales: plane(&native, &native_pool, Plane::VScales)?,
                        kv_len: state.step_field(StepField::KvLen)?,
                        part: part.range(0, part.bytes())?,
                        out: expected.range(0, expected.bytes())?,
                    },
                )?;
            } else {
                host.attend(
                    &ctx,
                    &stream,
                    &mut host_scratch,
                    PageAttention {
                        full_layer: 0,
                        visible_keys: visible,
                        scale: 1.0 / 16.0,
                        q: q.range(0, q.bytes())?,
                        proj: Some(gate.range(0, gate.bytes())?),
                        out: expected.range(0, expected.bytes())?,
                    },
                )?;
            }
            outputs &= read(&ctx, &stream, actual.range(0, actual.bytes())?)?
                == read(&ctx, &stream, expected.range(0, expected.bytes())?)?;
            words &= read(&ctx, &stream, block)? == params;
        }
        state.poison();
        state.reset(&stream)?;
        let reset = state.health() == Health::Valid
            && state.tokens() == 0
            && read(&ctx, &stream, state.word(Word::StepParams)?)?
                .iter()
                .all(|&byte| byte == 0);
        let params = StepParams::new(1, 0)?.to_bytes();
        let block = state.word(Word::StepParams)?;
        block.buffer().write_at(&stream, block.offset(), &params)?;
        let store = state.cached_store().unwrap();
        let mut store = store.borrow_mut();
        store.append(
            &ctx,
            &stream,
            PageToken {
                page: PageId {
                    full_layer: 0,
                    group: 0,
                },
                position: 0,
                params: block,
                k: v.range(0, row_bytes)?,
                v: k.range(0, row_bytes)?,
            },
        )?;
        store.attend(
            &ctx,
            &stream,
            &mut scratch,
            CachedAttention {
                page: PageAttention {
                    full_layer: 0,
                    visible_keys: 1,
                    scale: 1.0 / 16.0,
                    q: q.range(0, q.bytes())?,
                    proj: Some(gate.range(0, gate.bytes())?),
                    out: actual.range(0, actual.bytes())?,
                },
                kv_len: state.step_field(StepField::KvLen)?,
            },
        )?;
        ctx.kv8_chunk(
            &stream,
            Kv8Shape {
                rows: 1,
                n_kv_head: 2,
                head_dim: 256,
                cells: cache_rows,
                code_scale: Kv8CodeScale::StoredHalf,
            },
            Kv8Data {
                k: v.range(0, row_bytes)?,
                v: k.range(0, row_bytes)?,
                k_codes: plane(&native, &native_pool, Plane::K)?,
                k_scales: plane(&native, &native_pool, Plane::KScales)?,
                v_codes: plane(&native, &native_pool, Plane::V)?,
                v_scales: plane(&native, &native_pool, Plane::VScales)?,
            },
            0,
        )?;
        ctx.attn_decode_q8(
            &stream,
            AttnDecodeQ8Call {
                shape: AttnQ8Shape {
                    cells: cache_rows,
                    ..shape
                },
                span: AttnSpan::FULL,
                split: AttnSplit::Q8,
                scale: 1.0 / 16.0,
                q: q.range(0, q.bytes())?,
                proj: Some(gate.range(0, gate.bytes())?),
                k_codes: plane(&native, &native_pool, Plane::K)?,
                k_scales: plane(&native, &native_pool, Plane::KScales)?,
                v_codes: plane(&native, &native_pool, Plane::V)?,
                v_scales: plane(&native, &native_pool, Plane::VScales)?,
                kv_len: state.step_field(StepField::KvLen)?,
                part: part.range(0, part.bytes())?,
                out: expected.range(0, expected.bytes())?,
            },
        )?;
        let reused = read(&ctx, &stream, actual.range(0, actual.bytes())?)?
            == read(&ctx, &stream, expected.range(0, expected.bytes())?)?;
        eprintln!("post-reset attention with swapped K/V is byte-exact: {reused}");
        Ok(Observed {
            allocation,
            logical,
            outputs,
            words,
            reset: reset && reused,
        })
    }

    #[test]
    #[ignore = "requires the isolated standing GPU lease"]
    fn cache_state_selects_native_then_composed_attention_and_retains_global_words() {
        if !device_present("cache state and native/composed attention routing") {
            return;
        }
        let observed = observe_routes().expect("cache session routing");
        assert!(observed.allocation && observed.logical);
        assert!(
            observed.outputs,
            "routes must be byte-exact to their unchanged native/host-paged controls"
        );
        assert!(observed.words && observed.reset);
        eprintln!(
            "DEVICE_CACHE_MODEL_ROUTES cases=5 logical=1280 cached=512 byte_exact=true reset=true"
        );
    }
}

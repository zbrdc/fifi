//! Single-sequence Flint cache. Absolute positions own visibility; slots only own storage.

use super::config::FULL_LAYERS;
#[cfg(test)]
use super::{factory::context_limit, reference::ChunkWindow};
use crate::EngError;
#[cfg(test)]
use std::ops::Range;

pub(super) const WINDOW: u32 = 513;

/// KV-plane copy accounting (test builds only): every potentially-copying
/// plane op records its site, whether the layer is window-bounded (SWA), the
/// rows it touches, and the prefix position. One recorded row of one plane is
/// 4 heads x 128 x 2 bytes (BF16) = 1 KiB, so sum_rows is KiB moved.
#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
type PlaneCopySample = (&'static str, bool, u32, u32);

#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
fn plane_copy_store() -> &'static std::sync::Mutex<Vec<PlaneCopySample>> {
    use std::sync::Mutex;
    static SAMPLES: std::sync::OnceLock<Mutex<Vec<PlaneCopySample>>> = std::sync::OnceLock::new();
    SAMPLES.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(in crate::family::flint) fn note_plane_copy(
    site: &'static str,
    windowed: bool,
    rows: u32,
    p0: u32,
) {
    if let Ok(mut samples) = plane_copy_store().lock() {
        samples.push((site, windowed, rows, p0));
    }
}

/// Snapshot of [`note_plane_copy`] samples for the qualification reports.
#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(in crate::family::flint) fn plane_copy_samples() -> Vec<PlaneCopySample> {
    plane_copy_store()
        .lock()
        .map(|s| s.clone())
        .unwrap_or_default()
}

/// The model-pinned context ceiling: no `with_limit` may exceed it, whatever the
/// budget would allow (the budget check still refuses below it when memory is short).
pub(super) const CONTEXT_CEILING: u32 = 262_144;

#[derive(Clone, Debug)]
pub(super) struct CacheStep {
    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(super) debug_layer: usize,
    pub(super) capacity: u32,
    #[cfg(test)]
    pub(super) slot: u32,
    #[cfg(test)]
    pub(super) first: Range<u32>,
    #[cfg(test)]
    pub(super) second: Range<u32>,
    pub(super) rope: Option<u32>,
}

impl CacheStep {
    #[cfg(test)]
    pub(super) fn new(layer: usize, pos: u32) -> Result<Self, EngError> {
        Self::with_limit(layer, pos, context_limit())
    }

    /// The context-limit seam: the same step at a raised full-layer capacity. The
    /// limit is a SESSION parameter (the coordinator's 131,072 default), set only
    /// after the factory/budget gates; `new` keeps the 8,192 default unchanged.
    /// The hard ceiling is the model's 262,144.
    pub(super) fn with_limit(layer: usize, pos: u32, limit: u32) -> Result<Self, EngError> {
        if limit < WINDOW {
            return Err(EngError::BadRequest(format!(
                "Flint context limit {limit} is below the SWA window {WINDOW}"
            )));
        }
        if limit > CONTEXT_CEILING {
            return Err(EngError::BadRequest(format!(
                "Flint context limit {limit} exceeds the model ceiling {CONTEXT_CEILING}"
            )));
        }
        if layer >= 49 {
            return Err(EngError::BadRequest("Flint layer is outside 0..49".into()));
        }
        if pos >= limit {
            return Err(EngError::ContextLimit(format!(
                "Flint context exceeds {limit} positions"
            )));
        }
        let full = FULL_LAYERS.contains(&(layer as u64));
        let capacity = if full { limit } else { WINDOW };
        #[cfg(test)]
        let (slot, first, second) = {
            let range = ChunkWindow::new(capacity as usize, pos as usize, 1)?.live_range();
            let start = range.start as u32 % capacity;
            let length = range.len() as u32;
            let first_end = (start + length).min(capacity);
            (
                pos % capacity,
                start..first_end,
                0..length - (first_end - start),
            )
        };
        Ok(Self {
            #[cfg(all(
                test,
                feature = "native-mlx",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            debug_layer: layer,
            capacity,
            #[cfg(test)]
            slot,
            #[cfg(test)]
            first,
            #[cfg(test)]
            second,
            rope: (!full).then_some(pos),
        })
    }
}

/// The plan for one prefill chunk on one plane: the LAST row's step plus the chunk's
/// base position and row count. `p0` rides along because a full-plane `CacheStep`
/// carries no position (RoPE-less), yet the driver must know where the chunk begins.
#[derive(Clone, Debug)]
pub(super) struct ChunkCachePlan {
    pub(super) step: CacheStep,
    pub(super) p0: u32,
    pub(super) rows: u32,
}

impl ChunkCachePlan {
    /// Validates the serving bounds (1..=1024 rows, the whole chunk inside the
    /// limit) and derives the last row's step. The driver builds one plan per
    /// (layer, chunk) and hands it to `Plane::prepare_chunk`.
    pub(super) fn new(layer: usize, p0: u32, rows: u32, limit: u32) -> Result<Self, EngError> {
        if rows == 0 || rows > 1024 {
            return Err(EngError::BadRequest(format!(
                "Flint chunk carries 1..=1024 rows, not {rows}"
            )));
        }
        let last = p0
            .checked_add(rows - 1)
            .ok_or_else(|| EngError::ContextLimit("Flint chunk overflows the limit".into()))?;
        Ok(Self {
            step: CacheStep::with_limit(layer, last, limit)?,
            p0,
            rows,
        })
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::*;
    #[cfg(test)]
    use crate::tensor::Tensor;
    use crate::{
        kernels::MlxKernels,
        tensor::{Dtype, ExecutionTensor},
    };

    pub(in crate::family::flint) struct Plane {
        key: ExecutionTensor,
        value: ExecutionTensor,
    }

    pub(in crate::family::flint) struct State {
        pub(in crate::family::flint) planes: Vec<Option<Plane>>,
        pub(in crate::family::flint) pos: u32,
        pub(in crate::family::flint) valid: bool,
        /// Grown full-attention capacity (CTX32K-GROW): full planes start at
        /// the rounded-up prompt need, not the context limit, and double as
        /// chunks cross it. Ring planes stay at WINDOW and never grow.
        pub(in crate::family::flint) full_capacity: u32,
    }

    impl State {
        pub(in crate::family::flint) fn new() -> Self {
            Self {
                planes: (0..49).map(|_| None).collect(),
                pos: 0,
                valid: true,
                full_capacity: 0,
            }
        }

        /// Grows the full-attention planes so `need_end` rows fit, keeping
        /// short prompts at a short-prompt allocation (CTX32K-GROW). The
        /// capacity rounds up to 4096 rows, doubles on later crossings, and
        /// never passes the context limit; a need beyond the limit is the
        /// usual context error and grows nothing.
        pub(in crate::family::flint) fn ensure_full_capacity(
            &mut self,
            k: &mut MlxKernels,
            need_end: u32,
        ) -> Result<(), EngError> {
            if need_end <= self.full_capacity {
                return Ok(());
            }
            let limit = crate::family::flint::factory::context_limit();
            if need_end > limit {
                return Err(EngError::ContextLimit(format!(
                    "Flint context {} exceeds the limit {}",
                    need_end, limit
                )));
            }
            let target = super::capacity_target(need_end, self.full_capacity, limit);
            for (layer, slot) in self.planes.iter_mut().enumerate() {
                if FULL_LAYERS.contains(&(layer as u64))
                    && let Some(plane) = slot.take()
                {
                    *slot = Some(plane.grow(k, target)?);
                }
            }
            self.full_capacity = target;
            Ok(())
        }

        pub(in crate::family::flint) fn reset(&mut self) {
            // Never reuse stale ring rows, a partial forward, or another request's arrays.
            *self = Self::new();
        }

        /// A frozen view of every plane for the prefix store. Every tensor is
        /// an Rc share, so `kv_append` in the live session cannot donate these
        /// buffers (the extra share keeps the refcount above one) and must
        /// allocate fresh ones instead: the snapshot stays exact for as long
        /// as the store holds it, at no copy cost.
        pub(in crate::family::flint) fn snapshot_prefix(&self) -> PrefixSnapshot {
            PrefixSnapshot {
                planes: share_planes(&self.planes),
                pos: self.pos,
                full_capacity: self.full_capacity,
            }
        }

        /// Installs a snapshot as the live state, sharing again so the store's
        /// entry stays valid after this session continues. The WORKER hands
        /// over an already-reset session (it resets before invoking the
        /// executor restore; Sol wave-2 N-5); `valid` is true because the
        /// snapshot rows are exactly the rows the saved prefill produced.
        pub(in crate::family::flint) fn restore_prefix(&mut self, snapshot: &PrefixSnapshot) {
            self.planes = share_planes(&snapshot.planes);
            self.pos = snapshot.pos;
            self.valid = true;
            self.full_capacity = snapshot.full_capacity;
        }
    }

    /// The saved model state of a Flint prefix: every plane as storage
    /// shares plus the position bookkeeping that resumes growth and ring
    /// wrapping where the saved prefill stopped.
    pub(in crate::family::flint) struct PrefixSnapshot {
        planes: Vec<Option<Plane>>,
        pos: u32,
        full_capacity: u32,
    }

    /// The charged KV bytes of this snapshot for `prefix_report`: the
    /// ALLOCATED plane bytes (shape product x dtype width) of every plane
    /// it holds. Allocation, not `pos`, is what holds memory — a snapshot
    /// shares grown planes, so this is the number the ~2x-peak note in the
    /// design page reasons about.
    fn snapshot_charged_bytes(snapshot: &PrefixSnapshot) -> u64 {
        fn tensor_bytes(t: &crate::tensor::ExecutionTensor) -> u64 {
            t.shape().iter().map(|&d| d as u64).product::<u64>() * t.dtype().byte_width()
        }
        snapshot
            .planes
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|plane| tensor_bytes(&plane.key) + tensor_bytes(&plane.value))
            .sum()
    }

    impl PrefixSnapshot {
        pub(in crate::family::flint) fn charged_bytes(&self) -> u64 {
            snapshot_charged_bytes(self)
        }
    }

    fn share_planes(planes: &[Option<Plane>]) -> Vec<Option<Plane>> {
        planes
            .iter()
            .map(|slot| {
                slot.as_ref().map(|plane| Plane {
                    key: plane.key.share(),
                    value: plane.value.share(),
                })
            })
            .collect()
    }

    impl Plane {
        #[cfg(test)]
        pub(in crate::family::flint) fn snapshot(
            &self,
            k: &mut MlxKernels,
        ) -> Result<(Tensor, Tensor), EngError> {
            Ok((k.readback(&self.key)?, k.readback(&self.value)?))
        }

        fn new(k: &mut MlxKernels, capacity: u32) -> Result<Self, EngError> {
            // CTX32K-GROW: device zeros - no host staging vector, no import
            // copy, so a full-capacity plane costs no host bandwidth.
            let key = k.kv_zeros(capacity)?;
            let value = k.kv_zeros(capacity)?;
            Ok(Self { key, value })
        }

        /// Moves a full-attention plane into a larger zero plane. The old
        /// rows sit entirely below the new capacity, so one head append moves
        /// them; the tail stays zero, exactly as a fresh allocation would be.
        pub(in crate::family::flint) fn grow(
            self,
            k: &mut MlxKernels,
            target: u32,
        ) -> Result<Self, EngError> {
            let mut grown = Self::new(k, target)?;
            grown.key = append_block(k, grown.key, &self.key, 0)?;
            grown.value = append_block(k, grown.value, &self.value, 0)?;
            k.evaluate(&[&grown.key, &grown.value])?;
            Ok(grown)
        }

        /// Append current K/V, then expose the complete chronological causal window.
        #[cfg(test)]
        pub(in crate::family::flint) fn append(
            k: &mut MlxKernels,
            held: &mut Option<Self>,
            plan: &CacheStep,
            pair: (&ExecutionTensor, &ExecutionTensor),
        ) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
            let plane = match held.take() {
                Some(plane) => plane,
                None => Self::new(k, plan.capacity)?,
            };
            let key = k.cast(pair.0, Dtype::Bf16)?;
            let value = k.cast(pair.1, Dtype::Bf16)?;
            let key = k.kv_append(plane.key, &key, &[0, plan.slot, 0])?;
            let value = k.kv_append(plane.value, &value, &[0, plan.slot, 0])?;
            // Complete the append before taking views, so no later write aliases attention input.
            k.evaluate(&[&key, &value])?;
            let keys = visible(k, &key, plan)?;
            let values = visible(k, &value, plan)?;
            *held = Some(Self { key, value });
            Ok((keys, values))
        }

        /// Stage what this chunk's attention will read. Two shapes: the
        /// default stages an immutable chronological union (no cache row is
        /// overwritten here); the ring-view route (full plane, unwrapped,
        /// one-row, chained commit, flag set) APPENDS the chunk into the
        /// plane in place and returns the whole capacity plane — a write
        /// before the reader, not a write-free staging.
        pub(in crate::family::flint) fn prepare_chunk(
            k: &mut MlxKernels,
            held: &mut Option<Self>,
            plan: &ChunkCachePlan,
            k_rows: &ExecutionTensor,
            v_rows: &ExecutionTensor,
            // The commit that will consume the returned guard (the caller
            // owns this decision): only the chained commit can take the
            // ring-view route, because it alone knows the append already
            // happened when the guard reaches it.
            chained: bool,
        ) -> Result<(ExecutionTensor, ExecutionTensor, ChunkGuard), EngError> {
            for rows in [k_rows, v_rows] {
                if rows.shape() != [4, plan.rows, 128]
                    || !matches!(rows.dtype(), Dtype::F32 | Dtype::Bf16)
                {
                    return Err(EngError::BadRequest(
                        "Flint chunk KV must be floating [4,T,128]".into(),
                    ));
                }
            }
            if (plan.p0 > 0 && held.is_none())
                || held.as_ref().is_some_and(|plane| {
                    plane.key.shape() != [4, plan.step.capacity, 128]
                        || plane.value.shape() != [4, plan.step.capacity, 128]
                })
            {
                return Err(EngError::BadRequest(
                    "Flint chunk cache history/capacity differs".into(),
                ));
            }
            let new_keys = k.cast(k_rows, Dtype::Bf16)?;
            let new_values = k.cast(v_rows, Dtype::Bf16)?;
            let plane = match held.take() {
                Some(plane) => plane,
                None => Self::new(k, plan.step.capacity)?,
            };
            // Ring-view route (the Qwen Q36-KV-INTEGRATE shape): a
            // full-attention plane whose chunk does not wrap the ring appends
            // its rows in place FIRST and hands attention the plane;
            // shared_attention passes len = p0+1 separately from the plane's
            // row count, and rows at or beyond len are masked or skipped, so
            // retired ring rows are never read. Safety is MLX functional
            // semantics, not geometry alone: kv_append donates in place only
            // when the handle is unique (a copy otherwise), rows at or beyond
            // len are never read, and the D2 group-boundary roots still
            // evaluate the plane each four layers, so the next step's append
            // finds the handle unique.
            //
            // F3: when FIFI_FLINT_MLX_DECODE_ATTN arms for this process,
            // attention instead gets the PREFIX SLICE [0, len); flag off
            // keeps the whole-plane share bit for bit. The slice is a
            // zero-copy view for the vendor kernel (MLX vector-mode SDPA
            // accepts batch-1, D-stride-1 planes without copying, and the
            // never-wrap invariant makes rows [0, len) the valid
            // chronological history), it turns len into the handed plane's
            // own row count so the existing s <= kv_len guard admits the
            // vendor route, and its bytes match the ring-OFF union arm
            // exactly. The flash kernel WOULD copy that same slice (its
            // metal_kernel API pins ensure_row_contiguous), so the slice
            // arms only with the vendor predicate.
            let capacity = plane.key.shape()[1];
            let routes = crate::family::flint::mlx_forward::ring_view_routes(
                crate::family::flint::mlx_forward::RingViewGeometry {
                    full: plan.step.rope.is_none(),
                    wrapped: plan.p0 + plan.rows > capacity,
                    one_row: plan.rows == 1,
                    chained,
                },
            );
            if routes {
                let key = append_block(k, plane.key, &new_keys, plan.p0)?;
                let value = append_block(k, plane.value, &new_values, plan.p0)?;
                let armed = crate::kernels::mlx::decode_attn_flagged();
                let (keys, values) =
                    match vendor_decode_slice_len(plan.p0, plan.rows, capacity, armed) {
                        Some(len) => (
                            k.slice(&key, &[0, 0, 0], &[4, len, 128])?,
                            k.slice(&value, &[0, 0, 0], &[4, len, 128])?,
                        ),
                        None => (key.share(), value.share()),
                    };
                return Ok((
                    keys,
                    values,
                    ChunkGuard {
                        plane: Plane { key, value },
                        new_keys,
                        new_values,
                        plan: plan.clone(),
                        prepended: true,
                    },
                ));
            }
            let keys = chunk_union(k, &plane.key, &new_keys, plan)?;
            let values = chunk_union(k, &plane.value, &new_values, plan)?;
            Ok((
                keys,
                values,
                ChunkGuard {
                    plane,
                    new_keys,
                    new_values,
                    plan: plan.clone(),
                    prepended: false,
                },
            ))
        }

        /// The tensors a group-boundary evaluate must root: the appended key
        /// and value planes (D2 async chain).
        pub(in crate::family::flint) fn append_roots(&self) -> [&ExecutionTensor; 2] {
            [&self.key, &self.value]
        }

        /// Complete every query consumer before the first bulk cache write.
        pub(in crate::family::flint) fn commit_chunk(
            k: &mut MlxKernels,
            held: &mut Option<Self>,
            guard: ChunkGuard,
            consumed: &ExecutionTensor,
        ) -> Result<(), EngError> {
            if held.is_some() {
                return Err(EngError::BadRequest(
                    "Flint chunk guard has no vacant cache slot".into(),
                ));
            }
            if guard.prepended {
                return Err(EngError::BadRequest(
                    "ring-view chunk met the fenced commit".into(),
                ));
            }
            k.evaluate(&[consumed])?;
            let key = append_block(k, guard.plane.key, &guard.new_keys, guard.plan.p0)?;
            let value = append_block(k, guard.plane.value, &guard.new_values, guard.plan.p0)?;
            k.evaluate(&[&key, &value])?;
            *held = Some(Self { key, value });
            Ok(())
        }

        /// D2 async chain (design scrch/7c82644c/flint-async-chain/DESIGN.md):
        /// the one-row commit with no host-ordered fences. Instead of the
        /// two evaluates above, each append carries a graph edge on this
        /// layer's union views (`depended_update`), so MLX orders the in-place
        /// plane write after every pending reader inside the graph; the
        /// forward loop roots the accumulated graph at each group boundary.
        /// The donate semantics of `kv_append` are unchanged.
        pub(in crate::family::flint) fn commit_chunk_chained(
            k: &mut MlxKernels,
            held: &mut Option<Self>,
            guard: ChunkGuard,
            union_keys: &ExecutionTensor,
            union_values: &ExecutionTensor,
        ) -> Result<(), EngError> {
            if held.is_some() {
                return Err(EngError::BadRequest(
                    "Flint chunk guard has no vacant cache slot".into(),
                ));
            }
            if guard.prepended {
                // The ring-view path appended inside `prepare_chunk`; the
                // depended-update ordering duty does not apply because the
                // append precedes every reader of this step's view (MLX
                // functional semantics), so only re-arm the held plane.
                *held = Some(guard.plane);
                return Ok(());
            }
            let new_keys = depended_update(k, &guard.new_keys, union_keys)?;
            let new_values = depended_update(k, &guard.new_values, union_values)?;
            let key = append_block(k, guard.plane.key, &new_keys, guard.plan.p0)?;
            let value = append_block(k, guard.plane.value, &new_values, guard.plan.p0)?;
            *held = Some(Self { key, value });
            Ok(())
        }
    }

    #[cfg(test)]
    fn visible(
        k: &mut MlxKernels,
        plane: &ExecutionTensor,
        plan: &CacheStep,
    ) -> Result<ExecutionTensor, EngError> {
        let first = k.slice(plane, &[0, plan.first.start, 0], &[4, plan.first.end, 128])?;
        if plan.second.is_empty() {
            return Ok(first);
        }
        let second = k.slice(plane, &[0, 0, 0], &[4, plan.second.end, 128])?;
        k.concatenate(&[&first, &second], 1)
    }

    // Retain the immutable union until its consumer completes before ring writes.
    pub(in crate::family::flint) struct ChunkGuard {
        plane: Plane,
        new_keys: ExecutionTensor,
        new_values: ExecutionTensor,
        plan: ChunkCachePlan,
        /// The ring-view path appended this chunk into the plane inside
        /// `prepare_chunk` (before the reader); the chained commit then only
        /// re-arms the held plane instead of appending again.
        prepended: bool,
    }

    fn chunk_union(
        k: &mut MlxKernels,
        old: &ExecutionTensor,
        new: &ExecutionTensor,
        plan: &ChunkCachePlan,
    ) -> Result<ExecutionTensor, EngError> {
        let origin = if plan.step.rope.is_some() {
            plan.p0.saturating_sub(WINDOW - 1)
        } else {
            0
        };
        let rows = plan.p0 - origin;
        if rows == 0 {
            return Ok(new.share());
        }
        let start = origin % plan.step.capacity;
        let first = rows.min(plan.step.capacity - start);
        let head = k.slice(old, &[0, start, 0], &[4, start + first, 128])?;
        let history = if first == rows {
            head
        } else {
            let tail = k.slice(old, &[0, 0, 0], &[4, rows - first, 128])?;
            k.concatenate(&[&head, &tail], 1)?
        };
        #[cfg(all(
            test,
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        note_plane_copy("union_head_tail", plan.step.rope.is_some(), rows, plan.p0);
        #[cfg(all(
            test,
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        note_plane_copy(
            "union_concat",
            plan.step.rope.is_some(),
            rows + plan.rows,
            plan.p0,
        );
        k.concatenate(&[&history, new], 1)
    }

    /// Graph-ordered dependency for the chained commit: route the update rows
    /// through a concatenate that also names a real one-row slice of `reader`
    /// (the union view this layer's attention reads), then slice the update
    /// rows back out. The pass-through concat+slice pair copies the update
    /// bytes verbatim (bitwise identity), but every consumer of the result —
    /// the plane append — now waits for the reader's graph, which is what the
    /// two host-ordered fences in `commit_chunk` guaranteed. The adapter's
    /// slice validator rejects empty ranges, so the edge must name a real
    /// row; the union always has at least the update's own rows, so `end >= 1`.
    fn depended_update(
        k: &mut MlxKernels,
        rows: &ExecutionTensor,
        reader: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let total = rows.shape()[1];
        let end = reader.shape()[1];
        let last = k.slice(reader, &[0, end - 1, 0], &[4, end, 128])?;
        // window flag unknown here; the update rows are small either way.
        #[cfg(all(
            test,
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        crate::family::flint::mlx_state::note_plane_copy("depended_update", false, total + 1, 0);
        let joined = k.concatenate(&[rows, &last], 1)?;
        k.slice(&joined, &[0, 0, 0], &[4, total, 128])
    }

    fn append_block(
        k: &mut MlxKernels,
        target: ExecutionTensor,
        rows: &ExecutionTensor,
        start: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let capacity = target.shape()[1];
        let total = rows.shape()[1];
        let count = total.min(capacity);
        let skip = total - count;
        let slot = (start + skip) % capacity;
        let first = count.min(capacity - slot);
        let head = k.slice(rows, &[0, skip, 0], &[4, skip + first, 128])?;
        #[cfg(all(
            test,
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        crate::family::flint::mlx_state::note_plane_copy("kv_append_head", false, first, start);
        let target = k.kv_append(target, &head, &[0, slot, 0])?;
        if first == count {
            return Ok(target);
        }
        let tail = k.slice(rows, &[0, skip + first, 0], &[4, total, 128])?;
        #[cfg(all(
            test,
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        crate::family::flint::mlx_state::note_plane_copy(
            "kv_append_tail",
            false,
            count - first,
            start,
        );
        k.kv_append(target, &tail, &[0, 0, 0])
    }
}

/// Capacity that satisfies `need_end` rows: the need rounded up to 4096
/// rows, at least a doubling of `current`, capped at `limit`. Callers must
/// reject `need_end > limit` before use; `need_end <= current` never asks.
pub(in crate::family::flint) fn capacity_target(need_end: u32, current: u32, limit: u32) -> u32 {
    let rounded = need_end.div_ceil(4096) * 4096;
    rounded.max(current.max(1) * 2).min(limit)
}

/// F3: the prefix-slice row count vendor decode reads instead of the whole
/// ring-view plane - the visible length INCLUDING the row just appended
/// (p0 + rows) - or `None` to keep the whole-plane share. `None` also when
/// the plane is exactly full (the slice would be the plane itself) and when
/// the vendor flag does not arm for this process (flash must never meet the
/// strided slice: it copies what the vendor kernel takes as a view). Pure
/// so tests can pin the bounds without MLX arrays.
pub(in crate::family::flint) fn vendor_decode_slice_len(
    p0: u32,
    rows: u32,
    capacity: u32,
    armed: bool,
) -> Option<u32> {
    debug_assert!(
        p0 + rows <= capacity,
        "the ring-view arm only serves unwrapped chunks"
    );
    (armed && p0 + rows < capacity).then_some(p0 + rows)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) use native::{Plane, PrefixSnapshot, State};

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[path = "mlx_state_tests.rs"]
mod native_tests;

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[path = "mlx_chunk_cache_tests.rs"]
mod chunk_native_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_target_rounds_to_4096_doubles_then_caps() {
        // Rounding: any need up to 4096 allocates exactly 4096 rows.
        assert_eq!(capacity_target(1, 0, 32768), 4096);
        assert_eq!(capacity_target(4096, 0, 32768), 4096);
        // 4097 rounds to 8192.
        assert_eq!(capacity_target(4097, 0, 32768), 8192);
        // Doubling: an existing 8192 never grows to less than 16384.
        assert_eq!(capacity_target(8193, 8192, 32768), 16384);
        assert_eq!(capacity_target(8193, 8192, 8192), 8192); // capped
        // A far need caps at the limit instead of jumping past it.
        assert_eq!(capacity_target(30000, 16384, 32768), 32768);
        assert_eq!(capacity_target(32768, 0, 32768), 32768);
        // Rollback limit: growth lands exactly on 8192.
        assert_eq!(capacity_target(5000, 0, 8192), 8192);
    }

    #[test]
    fn vendor_decode_slice_len_bounds() {
        // The slice end INCLUDES the row just appended (p0 + rows).
        assert_eq!(vendor_decode_slice_len(4095, 1, 8192, true), Some(4096));
        // First step past the growth boundary still slices (4097 < 8192).
        assert_eq!(vendor_decode_slice_len(4096, 1, 8192, true), Some(4097));
        // Exactly full plane: the slice would be the plane itself.
        assert_eq!(vendor_decode_slice_len(8191, 1, 8192, true), None);
        // Flag off: byte-for-byte whole plane.
        assert_eq!(vendor_decode_slice_len(4095, 1, 8192, false), None);
    }

    #[test]
    fn local_cache_wraps_513_cells_and_keeps_current_token() {
        for (pos, slot, first, second) in [
            (0, 0, 0..1, 0..0),
            (512, 512, 0..513, 0..0),
            (513, 0, 1..513, 0..1),
            (514, 1, 2..513, 0..2),
            (1025, 512, 0..513, 0..0),
            (1026, 0, 1..513, 0..1),
        ] {
            let plan = CacheStep::new(2, pos).unwrap();
            assert_eq!(plan.capacity, 513);
            assert_eq!(plan.slot, slot);
            assert_eq!(plan.first, first);
            assert_eq!(plan.second, second);
            assert_eq!(plan.rope, Some(pos));
        }
    }

    #[test]
    fn full_layers_have_no_rope_and_keep_the_whole_prefix() {
        let limit = context_limit();
        for layer in [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48] {
            let plan = CacheStep::new(layer, limit - 1).unwrap();
            assert_eq!(plan.capacity, limit);
            assert_eq!(plan.slot, limit - 1);
            assert_eq!(plan.first, 0..limit);
            assert!(plan.second.is_empty());
            assert_eq!(plan.rope, None);
        }
        assert!(matches!(
            CacheStep::new(2, limit),
            Err(EngError::ContextLimit(_))
        ));
        assert!(matches!(
            CacheStep::new(49, 0),
            Err(EngError::BadRequest(_))
        ));
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    #[test]
    fn with_limit_matches_the_default_limit() {
        let a = CacheStep::new(0, 4_000).unwrap();
        let b = CacheStep::with_limit(0, 4_000, context_limit()).unwrap();
        assert_eq!(a.capacity, b.capacity);
        assert_eq!(a.slot, b.slot);
        assert_eq!(a.first, b.first);
        assert_eq!(a.second, b.second);
        assert_eq!(a.rope, b.rope);
    }

    #[test]
    fn a_raised_limit_raises_only_the_full_layers() {
        let full = CacheStep::with_limit(0, 100_000, 131_072).unwrap();
        assert_eq!(full.capacity, 131_072);
        let swa_layer = (0..49)
            .find(|l| !FULL_LAYERS.contains(&(*l as u64)))
            .unwrap();
        let swa = CacheStep::with_limit(swa_layer, 100_000, 131_072).unwrap();
        assert_eq!(swa.capacity, WINDOW);
        assert!(swa.rope.is_some());
        assert!(full.rope.is_none());
    }

    #[test]
    fn the_ceiling_and_floor_refuse() {
        assert!(CacheStep::with_limit(0, 0, 262_145).is_err());
        assert!(CacheStep::with_limit(0, 0, 512).is_err());
        assert!(CacheStep::with_limit(0, 131_072, 131_072).is_err());
        assert!(CacheStep::with_limit(49, 0, 131_072).is_err());
    }

    #[test]
    fn a_chunk_plan_derives_its_last_row_step() {
        let plan = ChunkCachePlan::new(0, 4_000, 512, 8_192).unwrap();
        assert_eq!(plan.p0, 4_000);
        assert_eq!(plan.rows, 512);
        assert_eq!(plan.step.capacity, 8_192, "the full layer's capacity");
        let swa_layer = (0..49)
            .find(|l| !FULL_LAYERS.contains(&(*l as u64)))
            .unwrap();
        let plan = ChunkCachePlan::new(swa_layer, 4_000, 512, 8_192).unwrap();
        assert_eq!(plan.step.capacity, WINDOW, "the SWA layer keeps the ring");
    }

    #[test]
    fn a_chunk_plan_refuses_bad_bounds() {
        assert!(ChunkCachePlan::new(0, 0, 0, 8_192).is_err(), "zero rows");
        assert!(
            ChunkCachePlan::new(0, 0, 1_025, 8_192).is_err(),
            "beyond 1024"
        );
        assert!(
            ChunkCachePlan::new(0, 8_000, 512, 8_192).is_err(),
            "the chunk would pass the limit"
        );
    }
}

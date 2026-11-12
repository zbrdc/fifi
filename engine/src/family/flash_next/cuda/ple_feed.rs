//! The PLE feed: the host step that turns a token's n-gram rows and the residual stream into the
//! stream update that layer `ple_layer` adds before its attention read.
//!
//! WHAT: [`NgramRows`] takes the pack's n-gram file (opened by `FlashPackFiles` through the
//! local-storage guard) and reads and decodes global rows (16 per token, 102 bytes each, random
//! positions in a 30 GiB file on local NVMe). [`PleFeed`] holds the PLE vectors from the pack (the
//! norms and the convolution, which the device does not hold) and a [`PleProjection`] for the two
//! projections, and runs the existing host step (`ple::step`) over them, so the numbers are the
//! host reference's. The update is `branches * hidden` F32 values (40 KiB); the device copy is
//! `stream_io`.
//! WHY three calls ([`PleFeed::plan`], [`PleFeed::gather`], [`PleFeed::update`]): the rows depend
//! on the token history only, not on the residual stream, so a caller may gather on another thread
//! while the device runs the layers before `ple_layer`; `gather` takes `&self` and no lock. Only
//! `update` needs the stream (the query side of the gate) and mutates the convolution ring.
//!
//! The pack carries no `weight_scale` for the EXL3 n-gram table (the scale is inside each row's
//! decode), so the step runs with scale 1. Hashing is `ple::plan_rows`, unchanged.
//!
//! Host cost: the key projection is `width x ple_embed` (10,240 x 2,560) F32 and the value
//! projection `hidden x ple_embed`, about 33 M multiply-adds per token on `threads` host threads.

use crate::{
    EngError,
    family::{
        exl3_pack::Root,
        flash_next::{
            ple::{self, PleShape, PleState, PleWeights, RowPlan},
            stored::{host_f32, stored_bytes},
        },
        project::Projector,
    },
    tensor::Dtype,
    weights::{
        StreamedCheckpoint,
        exl3::ngram::{NgramAux, NgramDecoder, ROW_DIM},
        exl3::ngram_file::NgramFile,
    },
};

/// The scale of the n-gram rows in the step. The EXL3 table has none (see the module doc).
const WEIGHT_SCALE: f32 = 1.0;

// The batch route projects the packed start embeddings unscaled; `ple::step` scales its own
// rows by `WEIGHT_SCALE`. They agree only while the constant is one.
const _: () = assert!(WEIGHT_SCALE == 1.0);

fn le_i64(bytes: &[u8], what: &str) -> Result<Vec<i64>, EngError> {
    let (words, rest) = bytes.as_chunks::<8>();
    if !rest.is_empty() {
        return Err(EngError::Checkpoint(format!(
            "n-gram {what} is not a whole number of I64 values"
        )));
    }
    Ok(words.iter().map(|w| i64::from_le_bytes(*w)).collect())
}

/// The n-gram file with its decoder and hash tensors: global row to values. `Sync`, so the rows of
/// the next token can be gathered on another thread while the card works (asserted below).
pub(in crate::family) struct NgramRows {
    file: NgramFile,
    decoder: NgramDecoder,
    multipliers: Vec<i64>,
    vocab: Vec<i64>,
    offsets: Vec<i64>,
}

impl NgramRows {
    /// Takes the table `FlashPackFiles::open` opened through the guard (the key and layout were
    /// checked there) and reads its small hash tensors.
    pub(in crate::family) fn new(file: NgramFile) -> Result<Self, EngError> {
        let offsets = le_i64(&file.read_aux(NgramAux::HeadOffsets)?, "head offsets")?;
        let vocab = le_i64(
            &file.read_aux(NgramAux::HeadVocabSizes)?,
            "head vocabulary sizes",
        )?;
        let multipliers = le_i64(
            &file.read_aux(NgramAux::LayerMultipliers)?,
            "layer multipliers",
        )?;
        let bias: Vec<u16> = file
            .read_aux(NgramAux::HeadBias)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        let decoder = NgramDecoder::new(file.table(), &offsets, &vocab, &bias)?;
        Ok(Self {
            file,
            decoder,
            multipliers,
            vocab,
            offsets,
        })
    }

    /// Rows in the table, the single virtual table the step addresses.
    fn rows(&self) -> u64 {
        self.file.table().rows()
    }

    /// Reads and decodes the rows of `plan`, in plan order. Takes `&self` and no lock, so it may
    /// run on another thread while the device works: the rows depend on the token history only.
    pub(in crate::family) fn gather(&self, plan: &RowPlan) -> Result<Gathered, EngError> {
        let rows = plan
            .rows
            .iter()
            .map(|&(_, row)| {
                let row = u64::from(row);
                Ok((row, self.row(row)?))
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        Ok(Gathered { rows })
    }

    /// Reads and decodes global row `row`.
    pub(in crate::family) fn row(&self, row: u64) -> Result<Vec<f32>, EngError> {
        let mut packed = vec![0u8; self.file.table().row_bytes()];
        self.file.read_row(row, &mut packed)?;
        let mut values = [0f32; ROW_DIM];
        self.decoder.decode_row(row, &packed, &mut values)?;
        Ok(values.to_vec())
    }
}

/// The keys and the value the PLE block projects from a token's gathered embedding.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::family) struct PleKeyValue {
    /// `branches * hidden` values (`key_proj`).
    pub(in crate::family) keys: Vec<f32>,
    /// `hidden` values (`value_proj`).
    pub(in crate::family) value: Vec<f32>,
}

/// The two projections of the PLE block, `key_proj` (`width x ple_embed`) and `value_proj`
/// (`hidden x ple_embed`), applied to one embedding of `ple_embed` values. The step needs nothing
/// else from the weights, so a route plugs in here: [`HostProjection`] is the F32 reference; the
/// device route runs the same two matvecs on the card (one launch pair, one 44 KiB download).
///
/// Not `Send`: the step runs on one thread, and the device route holds the step's `Rc` handles to
/// the CUDA context and stream, which cannot move between threads.
pub(in crate::family) trait PleProjection {
    fn project(&mut self, embedding: &[f32]) -> Result<PleKeyValue, EngError>;
}

const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<NgramRows>();
};

/// A dense row-major `[rows, cols]` F32 matrix on the host.
pub(in crate::family) struct HostMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl HostMatrix {
    pub(in crate::family) fn new(
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    ) -> Result<Self, EngError> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(EngError::Checkpoint(format!(
                "PLE projection of {} values is not {rows} x {cols}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// `self * x`, rows split across `threads` (the row split of `qwen36::host_cpu`), each row
    /// summed in order.
    fn matvec(&self, x: &[f32], threads: usize) -> Result<Vec<f32>, EngError> {
        if x.len() != self.cols {
            return Err(EngError::BadRequest(format!(
                "PLE projection input has {} values, expected {}",
                x.len(),
                self.cols
            )));
        }
        let cols = self.cols;
        let mut out = vec![0.0f32; self.rows];
        let chunk = self.rows.div_ceil(threads.max(1)).max(1);
        std::thread::scope(|scope| {
            for (index, part) in out.chunks_mut(chunk).enumerate() {
                let data = &self.data;
                scope.spawn(move || {
                    for (offset, slot) in part.iter_mut().enumerate() {
                        let row = (index * chunk + offset) * cols;
                        *slot = data[row..row + cols]
                            .iter()
                            .zip(x)
                            .map(|(w, v)| w * v)
                            .sum();
                    }
                });
            }
        });
        Ok(out)
    }
}

/// The host route: both projections as F32 matvecs on `threads` threads.
pub(in crate::family) struct HostProjection {
    key: HostMatrix,
    value: HostMatrix,
    threads: usize,
}

impl HostProjection {
    pub(in crate::family) fn new(key: HostMatrix, value: HostMatrix, threads: usize) -> Self {
        Self {
            key,
            value,
            threads,
        }
    }

    /// The thread count the chunk driver thresholds its small-chunk fallback against.
    pub(in crate::family) fn threads(&self) -> usize {
        self.threads
    }

    /// One weight row dotted against four embeddings with the four folds interleaved. Each
    /// sum still adds its products in index order starting from `-0.0` — the identity `Sum for
    /// f32` folds from, which [`HostMatrix::matvec`]'s `.sum()` therefore uses — so the values
    /// are bit-identical, an all-`-0.0` row included; only the interleaving differs, and the
    /// fold order cannot see it. Interleaving breaks the dependent-add chain that otherwise
    /// pins one dot to one add per weight load.
    fn row_dots(weights: &[f32], quad: [&[f32]; 4], embed: usize) -> [f32; 4] {
        let mut sums = [-0.0f32; 4];
        for column in 0..embed {
            let weight = weights[column];
            sums[0] += weight * quad[0][column];
            sums[1] += weight * quad[1][column];
            sums[2] += weight * quad[2][column];
            sums[3] += weight * quad[3][column];
        }
        sums
    }

    /// Tokens a blocked product computes against one weight row before the next weight row
    /// streams in. A block's embeddings stay in cache while the 131 MB of weights sweep once
    /// per block; bigger blocks mean fewer sweeps and more residency.
    const BLOCK: usize = 32;

    /// Tokens one [`Self::row_dots`] call interleaves. Four independent folds give the
    /// scheduler four adds per weight load instead of one dependent chain.
    const UNROLL: usize = 4;

    /// The projections of a whole chunk in one product: `rows` embeddings of `key.cols`
    /// values each, flattened. The token runs split over the projection threads (whole rows
    /// per thread, so no two threads share an output element); inside a run the tokens go
    /// [`Self::BLOCK`] at a time and each weight row serves the whole block while hot, so the
    /// weights stream once per block instead of once per token. Every output element folds its
    /// dot exactly as [`HostMatrix::matvec`] does — the same sequential `sum` over the same
    /// products — so the pairs are bit-identical to per-token [`PleProjection::project`] calls.
    pub(in crate::family) fn project_chunk(
        &self,
        embeddings: &[f32],
        rows: usize,
    ) -> Result<Vec<PleKeyValue>, EngError> {
        let embed = self.key.cols;
        if self.value.cols != embed {
            return Err(EngError::Checkpoint(format!(
                "PLE projections disagree on the embedding: key {embed} values, value {}",
                self.value.cols
            )));
        }
        if embeddings.len() != rows * embed {
            return Err(EngError::BadRequest(format!(
                "PLE chunk projection input has {} values, expected {rows} rows of {embed}",
                embeddings.len()
            )));
        }
        let mut out: Vec<PleKeyValue> = (0..rows)
            .map(|_| PleKeyValue {
                keys: vec![0.0; self.key.rows],
                value: vec![0.0; self.value.rows],
            })
            .collect();
        let run = rows.div_ceil(self.threads.max(1)).max(1);
        std::thread::scope(|scope| {
            for (index, part) in out.chunks_mut(run).enumerate() {
                let (key, value) = (&self.key, &self.value);
                let first = index * run;
                scope.spawn(move || {
                    let embedding = |token: usize| &embeddings[token * embed..(token + 1) * embed];
                    let last = rows - 1;
                    let quad_at = |base: usize| {
                        [
                            embedding(base.min(last)),
                            embedding((base + 1).min(last)),
                            embedding((base + 2).min(last)),
                            embedding((base + 3).min(last)),
                        ]
                    };
                    let mut start = 0;
                    while start < part.len() {
                        let end = (start + Self::BLOCK).min(part.len());
                        let block = &mut part[start..end];
                        for row in 0..key.rows {
                            let weights = &key.data[row * embed..(row + 1) * embed];
                            for (group, slots) in block.chunks_mut(Self::UNROLL).enumerate() {
                                let sums = Self::row_dots(
                                    weights,
                                    quad_at(first + start + group * Self::UNROLL),
                                    embed,
                                );
                                for (sum, pair) in sums.iter().zip(slots.iter_mut()) {
                                    pair.keys[row] = *sum;
                                }
                            }
                        }
                        for row in 0..value.rows {
                            let weights = &value.data[row * embed..(row + 1) * embed];
                            for (group, slots) in block.chunks_mut(Self::UNROLL).enumerate() {
                                let sums = Self::row_dots(
                                    weights,
                                    quad_at(first + start + group * Self::UNROLL),
                                    embed,
                                );
                                for (sum, pair) in sums.iter().zip(slots.iter_mut()) {
                                    pair.value[row] = *sum;
                                }
                            }
                        }
                        start = end;
                    }
                });
            }
        });
        Ok(out)
    }
}

impl PleProjection for HostProjection {
    fn project(&mut self, embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        Ok(PleKeyValue {
            keys: self.key.matvec(embedding, self.threads)?,
            value: self.value.matvec(embedding, self.threads)?,
        })
    }
}

/// What the host step's `Projector` calls address: the two projections and the n-gram table.
pub(in crate::family) enum FeedWeight {
    Key,
    Value,
    /// The table the step's `rows` calls address; the rows themselves are in [`Gathered`].
    Table,
}

/// The decoded rows of one token, keyed by global row.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::family) struct Gathered {
    rows: Vec<(u64, Vec<f32>)>,
}

impl Gathered {
    pub(in crate::family) fn len(&self) -> usize {
        self.rows.len()
    }

    /// The decoded values of global row `row`, if it was gathered.
    pub(in crate::family) fn get(&self, row: u64) -> Option<&[f32]> {
        self.rows
            .iter()
            .find(|(global, _)| *global == row)
            .map(|(_, values)| values.as_slice())
    }
}

/// The projector the host step runs on: the key and value calls go to the [`PleProjection`] (one
/// `project` call serves both, the step asks for the keys first and the value second on the same
/// embedding) and table rows are served from a [`Gathered`].
struct FeedProjector<'g> {
    gathered: &'g Gathered,
    projection: &'g mut dyn PleProjection,
    value: Option<Vec<f32>>,
}

impl Projector for FeedProjector<'_> {
    type Weight = FeedWeight;

    fn linear(&mut self, weight: &FeedWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        match weight {
            FeedWeight::Key => {
                let both = self.projection.project(x)?;
                self.value = Some(both.value);
                Ok(both.keys)
            }
            FeedWeight::Value => match self.value.take() {
                Some(value) => Ok(value),
                None => Ok(self.projection.project(x)?.value),
            },
            FeedWeight::Table => Err(EngError::BadRequest("linear needs a projection".into())),
        }
    }

    fn expert_linear(&mut self, _: &FeedWeight, _: u32, _: &[f32]) -> Result<Vec<f32>, EngError> {
        Err(EngError::BadRequest(
            "the PLE feed has no expert stack".into(),
        ))
    }

    fn rows(&mut self, table: &FeedWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        if !matches!(table, FeedWeight::Table) {
            return Err(EngError::BadRequest("rows needs the n-gram table".into()));
        }
        let mut out = Vec::with_capacity(rows.len() * ROW_DIM);
        for &row in rows {
            let values = self.gathered.get(u64::from(row)).ok_or_else(|| {
                EngError::BadRequest(format!("n-gram row {row} was not gathered"))
            })?;
            out.extend_from_slice(values);
        }
        Ok(out)
    }
}

/// The PLE vectors of the pack, on the host: the three norms and the convolution, as stored
/// (`ple::step` applies `1 + w` itself).
pub(in crate::family) struct PleVectors {
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    conv: Vec<f32>,
}

impl PleVectors {
    /// Reads the vectors of the PLE block of `layer` from the pack, and the host route over its
    /// two projections on `threads` threads. Widths come from `shape`.
    pub(in crate::family) fn read(
        source: &StreamedCheckpoint,
        layer: usize,
        shape: &PleShape,
        threads: usize,
    ) -> Result<(Self, HostProjection), EngError> {
        let dims = shape.dims();
        let (hidden, branches) = (dims.hidden as usize, dims.branches as usize);
        let width = hidden * branches;
        let embed = shape.embed();
        let at = |suffix: &str| Root::Layers.name(layer, &format!("ple.{suffix}"));
        let matrix = |suffix: &str, rows: usize| {
            HostMatrix::new(rows, embed, host_f32(source, &at(suffix), rows * embed)?)
        };
        let projection = HostProjection::new(
            matrix("key_proj.weight", width)?,
            matrix("value_proj.weight", hidden)?,
            threads,
        );
        let vectors = Self {
            norm_key: host_f32(source, &at("norm_key.weight"), width)?,
            norm_query: host_f32(source, &at("norm_query.weight"), width)?,
            norm_conv: host_f32(source, &at("norm_conv.weight"), width)?,
            conv: host_f32(
                source,
                &at("conv1d.weight"),
                width * dims.conv_kernel as usize,
            )?,
        };
        Ok((vectors, projection))
    }

    /// The two projections as stored (dtype and raw bytes, no conversion): the device route
    /// uploads them to the card as the pack holds them, F16 or BF16.
    pub(in crate::family) fn raw_projections(
        source: &StreamedCheckpoint,
        layer: usize,
        shape: &PleShape,
    ) -> Result<(Dtype, Vec<u8>, Vec<u8>), EngError> {
        let dims = shape.dims();
        let width = (dims.hidden * dims.branches) as usize;
        let embed = shape.embed();
        let at = |suffix: &str| Root::Layers.name(layer, &format!("ple.{suffix}"));
        let raw = |suffix: &str, rows: usize| stored_bytes(source, &at(suffix), rows * embed);
        let (key_dtype, key) = raw("key_proj.weight", width)?;
        let (value_dtype, value) = raw("value_proj.weight", dims.hidden as usize)?;
        if key_dtype != value_dtype {
            return Err(EngError::Checkpoint(format!(
                "the PLE projections are stored as {key_dtype:?} and {value_dtype:?}"
            )));
        }
        Ok((key_dtype, key, value))
    }
}

/// The weights `ple::step` and `ple::plan_rows` read, borrowed field by field so that the
/// projection can be borrowed mutably beside them.
fn step_weights<'a>(
    rows: &'a NgramRows,
    vectors: &'a PleVectors,
    table: &'a [FeedWeight; 1],
    table_rows: &'a [u64; 1],
) -> PleWeights<'a, FeedWeight> {
    PleWeights {
        multipliers: &rows.multipliers,
        vocab: &rows.vocab,
        offsets: &rows.offsets,
        shards: table,
        shard_rows: table_rows,
        weight_scale: WEIGHT_SCALE,
        key: &FeedWeight::Key,
        value: &FeedWeight::Value,
        norm_key: &vectors.norm_key,
        norm_query: &vectors.norm_query,
        norm_conv: &vectors.norm_conv,
        conv: &vectors.conv,
    }
}

/// The host PLE step with its state.
pub(in crate::family) struct PleFeed {
    shape: PleShape,
    rows: NgramRows,
    vectors: PleVectors,
    projection: Box<dyn PleProjection>,
    /// The device route's hand-off: when set, [`PleFeed::set_fixed`] installs a precomputed
    /// key/value pair and `ple_update` returns it instead of running the projection.
    fixed: Option<std::rc::Rc<FixedProjection>>,
    table: [FeedWeight; 1],
    table_rows: [u64; 1],
    state: PleState,
}

/// A [`PleProjection`] that returns pairs set from outside: the device route computes the
/// chunk's projections on the card and hands each row's pair over before the host step runs.
pub(in crate::family) struct FixedProjection {
    pair: std::cell::RefCell<Option<PleKeyValue>>,
}

impl FixedProjection {
    pub(in crate::family) fn new() -> Self {
        Self {
            pair: std::cell::RefCell::new(None),
        }
    }

    /// Installs the pair the next `project` returns; the last write wins.
    pub(in crate::family) fn set(&self, pair: PleKeyValue) {
        *self.pair.borrow_mut() = Some(pair);
    }

    /// Takes the installed pair (the `&self` form: the shared handle delegates here, because an
    /// `Rc` alias is the design, not an error).
    fn take(&self) -> Option<PleKeyValue> {
        self.pair.borrow_mut().take()
    }
}

impl PleProjection for FixedProjection {
    fn project(&mut self, _embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        self.take().ok_or_else(|| EngError::Backend {
            operation: "flash_next_ple_device",
            message: "the device PLE pair was not set for this row".to_string(),
        })
    }
}

/// The boxed half of a [`FixedProjection`] the feed owns: delegating so the feed's `Box<dyn>` and
/// the driver's `Rc` see the same cell.
struct SharedFixed(std::rc::Rc<FixedProjection>);

impl PleProjection for SharedFixed {
    fn project(&mut self, _embedding: &[f32]) -> Result<PleKeyValue, EngError> {
        self.0.take().ok_or_else(|| EngError::Backend {
            operation: "flash_next_ple_device",
            message: "the device PLE pair was not set for this row".to_string(),
        })
    }
}

impl PleFeed {
    pub(in crate::family) fn new(
        shape: PleShape,
        rows: NgramRows,
        vectors: PleVectors,
        projection: Box<dyn PleProjection>,
    ) -> Self {
        let table_rows = [rows.rows()];
        Self {
            state: PleState::new(&shape),
            shape,
            rows,
            vectors,
            projection,
            fixed: None,
            table: [FeedWeight::Table],
            table_rows,
        }
    }

    /// The n-gram table, which a caller may read from another thread (`NgramRows` is `Sync`).
    pub(in crate::family) fn rows(&self) -> &NgramRows {
        &self.rows
    }

    /// A feed whose projections come from [`FixedProjection::set`]: the device route installs
    /// each row's pair there, and the host step consumes it unchanged.
    pub(in crate::family) fn with_fixed(
        shape: PleShape,
        rows: NgramRows,
        vectors: PleVectors,
        fixed: std::rc::Rc<FixedProjection>,
    ) -> Self {
        let projection = Box::new(SharedFixed(std::rc::Rc::clone(&fixed)));
        let table_rows = [rows.rows()];
        Self {
            state: PleState::new(&shape),
            shape,
            rows,
            vectors,
            projection,
            fixed: Some(fixed),
            table: [FeedWeight::Table],
            table_rows,
        }
    }

    /// Whether [`Self::with_fixed`] built this feed.
    pub(in crate::family) fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// Installs the pair the next update of this row consumes. Only a fixed feed takes pairs.
    pub(in crate::family) fn set_fixed(&self, pair: PleKeyValue) -> Result<(), EngError> {
        self.fixed.as_ref().map_or_else(
            || {
                Err(EngError::BadRequest(
                    "PLE pairs can only be set on a fixed-projection feed".to_string(),
                ))
            },
            |fixed| {
                fixed.set(pair);
                Ok(())
            },
        )
    }

    fn weights(&self) -> PleWeights<'_, FeedWeight> {
        step_weights(&self.rows, &self.vectors, &self.table, &self.table_rows)
    }

    /// The rows `token` needs after the tokens already fed.
    pub(in crate::family) fn plan(&self, token: u32) -> Result<RowPlan, EngError> {
        ple::plan_rows(&self.shape, &self.weights(), self.state.history(), token)
    }

    /// The plan for `token` as if `prefix` tokens had been fed after the current history: row
    /// `i` of a chunk plans against the history it will have when its turn comes. The rows
    /// depend on the history only, never on the residual stream, so the plans stay valid while
    /// the device runs.
    pub(in crate::family) fn plan_after(
        &self,
        prefix: &[u32],
        token: u32,
    ) -> Result<RowPlan, EngError> {
        let mut history = self.state.history().to_vec();
        history.extend_from_slice(prefix);
        //  reads the pair from a history of at most two; a longer slice would read
        // the end-of-sequence marker for the second token and pick a different row set.
        let keep_from = history.len().saturating_sub(2);
        ple::plan_rows(&self.shape, &self.weights(), &history[keep_from..], token)
    }

    /// Reads and decodes the rows of `plan` (see [`NgramRows::gather`]).
    pub(in crate::family) fn gather(&self, plan: &RowPlan) -> Result<Gathered, EngError> {
        self.rows.gather(plan)
    }

    /// The PLE embedding of `token` as the chunk feeds assemble it: planned against the
    /// history it will have when its turn comes (the `prefix` tokens before it in the
    /// chunk), gathered, and the gathered rows concatenated in plan order. Returns the
    /// gathered rows too — `pack_feeds::begin_chunk` keeps them pending for the update
    /// pass. The per-token step projects the same values (its own `plan` + `gather`, the
    /// rows unscaled while `WEIGHT_SCALE` is one); the chunk-embeddings test pins the two
    /// assemblies together bit for bit.
    pub(in crate::family) fn chunk_embedding(
        &self,
        prefix: &[u32],
        token: u32,
    ) -> Result<(Vec<f32>, Gathered), EngError> {
        let plan = self.plan_after(prefix, token)?;
        let rows = self.gather(&plan)?;
        let mut embedding = Vec::new();
        for &(_, local) in &plan.rows {
            let row = rows.get(u64::from(local)).ok_or_else(|| {
                EngError::BadRequest(format!("n-gram row {local} was not gathered"))
            })?;
            embedding.extend_from_slice(row);
        }
        Ok((embedding, rows))
    }

    /// The residual update of `token`: `stream` is the residual stream (`branches * hidden`) at the
    /// PLE layer's input, `gathered` the rows of [`Self::plan`] for the same token. Advances the
    /// history and the convolution ring. On an error the state is unchanged.
    pub(in crate::family) fn update(
        &mut self,
        token: u32,
        gathered: &Gathered,
        stream: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let mut next = self.state.clone();
        let profile = crate::family::flash_next::cuda::driver::chunk::prof::on();
        let mut t = std::time::Instant::now();
        let out = {
            let mut projector = FeedProjector {
                gathered,
                projection: self.projection.as_mut(),
                value: None,
            };
            let weights = step_weights(&self.rows, &self.vectors, &self.table, &self.table_rows);
            if profile {
                crate::family::flash_next::cuda::driver::chunk::prof::add(14, t.elapsed());
                t = std::time::Instant::now();
            }
            ple::step(
                &mut projector,
                &self.shape,
                &weights,
                &mut next,
                token,
                stream,
            )?
        };
        if profile {
            crate::family::flash_next::cuda::driver::chunk::prof::add(15, t.elapsed());
        }
        self.state = next;
        Ok(out)
    }

    /// Forgets the history and zeroes the ring (a new sequence).
    pub(in crate::family) fn reset(&mut self) {
        self.state = PleState::new(&self.shape);
    }

    /// The host half of a resume snapshot: the token history and the host ring mirror.
    pub(in crate::family) fn state_parts(&self) -> (&[u32], &[f32]) {
        (self.state.history(), self.state.ring_values())
    }

    /// Installs a saved host half, as a prefix restore brings the feed back to a boundary.
    pub(in crate::family) fn resume(
        &mut self,
        history: &[u32],
        ring: &[f32],
    ) -> Result<(), EngError> {
        self.state.resume(history, ring)
    }
}

#[cfg(test)]
mod tests;

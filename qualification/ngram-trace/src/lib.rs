use std::collections::HashSet;

pub const PLE_HEADS: usize = 16;
pub const PLE_HEADS_PER_NGRAM: usize = 8;
pub const PLE_POSITION_MULTIPLIERS: usize = 3;
pub const PAGE_BYTES_4K: u64 = 4 * 1024;
pub const PAGE_BYTES_16K: u64 = 16 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleConfig {
    pub eos_token_id: u32,
    /// One multiplier per token position, shared by every head.
    pub position_multipliers: [u64; PLE_POSITION_MULTIPLIERS],
    pub head_vocab_sizes: [u64; PLE_HEADS],
    pub head_offsets: [u64; PLE_HEADS],
    pub table_rows: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    EmptyHistory,
    EmptyTable,
    InvalidHeadRange { head: usize },
    RowOutOfBounds { row: u64, table_rows: u64 },
    InvalidRowBytes,
    AddressOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCounts {
    pub pages_4k: usize,
    pub pages_16k: usize,
}

fn validate_config(config: &PleConfig) -> Result<(), TraceError> {
    if config.table_rows == 0 {
        return Err(TraceError::EmptyTable);
    }
    let reference_row_limit = i32::MAX as u64;
    for head in 0..PLE_HEADS {
        let vocab_size = config.head_vocab_sizes[head];
        let Some(end) = config.head_offsets[head].checked_add(vocab_size) else {
            return Err(TraceError::InvalidHeadRange { head });
        };
        if vocab_size == 0 || end > config.table_rows || end > reference_row_limit {
            return Err(TraceError::InvalidHeadRange { head });
        }
    }
    Ok(())
}

/// Computes all 16 head rows for the newest token in chronological `history`.
///
/// Head indices 0..8 use two tokens. Indices 8..16 use three tokens. Missing
/// predecessors and tokens before the latest predecessor EOS are replaced with
/// EOS. A current EOS token does not erase its own preceding context.
pub fn row_ids(config: &PleConfig, history: &[u32]) -> Result<[u64; PLE_HEADS], TraceError> {
    validate_config(config)?;
    if history.is_empty() {
        return Err(TraceError::EmptyHistory);
    }
    Ok(row_ids_validated(config, history))
}

/// Computes one row vector for each token in `tokens`, in input order.
pub fn trace_tokens(
    config: &PleConfig,
    tokens: &[u32],
) -> Result<Vec<[u64; PLE_HEADS]>, TraceError> {
    validate_config(config)?;
    let mut history = Vec::with_capacity(tokens.len());
    let mut trace = Vec::with_capacity(tokens.len());
    for &token in tokens {
        history.push(token);
        trace.push(row_ids_validated(config, &history));
    }
    Ok(trace)
}

fn row_ids_validated(config: &PleConfig, history: &[u32]) -> [u64; PLE_HEADS] {
    // Both public callers ensure `history` contains a current token.
    let mut context = [config.eos_token_id; PLE_POSITION_MULTIPLIERS];
    context[0] = history[history.len() - 1];
    let mut cut = false;
    for (distance, slot) in context.iter_mut().enumerate().skip(1) {
        let token = history
            .len()
            .checked_sub(distance + 1)
            .map(|index| history[index])
            .unwrap_or(config.eos_token_id);
        cut |= token == config.eos_token_id;
        *slot = if cut { config.eos_token_id } else { token };
    }

    let mut rows = [0; PLE_HEADS];
    for (head, row) in rows.iter_mut().enumerate() {
        let order = 2 + head / PLE_HEADS_PER_NGRAM;
        let mixed = context[..order]
            .iter()
            .zip(config.position_multipliers.iter())
            .fold(0_u64, |hash, (&token, &multiplier)| {
                hash ^ u64::from(token).wrapping_mul(multiplier)
            });
        *row = config.head_offsets[head] + mixed % config.head_vocab_sizes[head];
    }
    rows
}

/// Counts distinct file pages touched by the supplied table rows.
///
/// `tensor_offset` is the absolute file offset of row zero. `row_bytes` and
/// `table_rows` must come from checked tensor geometry. Each row contributes
/// every page intersecting its half-open byte range, including boundary
/// crossings. These are simulations, not physical-I/O measurements.
pub fn page_counts(
    row_ids: &[u64],
    tensor_offset: u64,
    row_bytes: u64,
    table_rows: u64,
) -> Result<PageCounts, TraceError> {
    if table_rows == 0 {
        return Err(TraceError::EmptyTable);
    }
    if row_bytes == 0 {
        return Err(TraceError::InvalidRowBytes);
    }
    let pages_4k = distinct_pages(row_ids, tensor_offset, row_bytes, table_rows, PAGE_BYTES_4K)?;
    let pages_16k = distinct_pages(
        row_ids,
        tensor_offset,
        row_bytes,
        table_rows,
        PAGE_BYTES_16K,
    )?;
    Ok(PageCounts {
        pages_4k,
        pages_16k,
    })
}

fn distinct_pages(
    row_ids: &[u64],
    tensor_offset: u64,
    row_bytes: u64,
    table_rows: u64,
    page_bytes: u64,
) -> Result<usize, TraceError> {
    let mut pages = HashSet::new();
    for &row in row_ids {
        if row >= table_rows {
            return Err(TraceError::RowOutOfBounds { row, table_rows });
        }
        let start = row
            .checked_mul(row_bytes)
            .and_then(|bytes| tensor_offset.checked_add(bytes))
            .ok_or(TraceError::AddressOverflow)?;
        let end = start
            .checked_add(row_bytes)
            .ok_or(TraceError::AddressOverflow)?;
        let last_byte = end.checked_sub(1).ok_or(TraceError::AddressOverflow)?;
        let first_page = start / page_bytes;
        let last_page = last_byte / page_bytes;
        for page in first_page..=last_page {
            pages.insert(page);
        }
    }
    Ok(pages.len())
}

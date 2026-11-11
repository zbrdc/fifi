//! The job-table machinery of an EXL3 routed pass, shared by the families that run chunked
//! prefill over a bank of expert records: the plan (a counting sort of the chunk's pairs by
//! member, cut into jobs of at most [`EXL3_EXPERT_TILE_ROWS`] rows), the one-write upload of
//! the row lists and the job table, and the construction of the job-table launch of one routed
//! projection.
//!
//! The member id a job carries is the caller's: the expert index of a whole-bank launch, or the
//! slot id of a banked launch whose weights come from a slot map — the launch only walks the
//! members of [`Exl3Members`] at the fixed strides that object carries. The orchestration of a
//! block (which projections run, the taps, the fallbacks) stays with the family; this module
//! holds the arithmetic two families must not duplicate.
//!
//! Pure plan functions plus launch construction over borrowed device ranges; nothing here
//! allocates device memory, enqueues or observes health.

use crate::{
    EngError,
    family::{
        project::cuda::{Exl3JobsIo, exl3::Exl3Members},
        qwen36::cuda::moe::{Launch, MoeDims},
    },
    kernels::cuda::{DeviceRange, resident::exl3::Exl3Job},
};

const I32: usize = size_of::<i32>();

use crate::family::project::exl3_jobs_plan::EXL3_EXPERT_TILE_ROWS;

/// The sorted plan of a chunk's `m * top_k` pairs: the counting sort by member and the two row
/// lists the launches read.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::family) struct PairPlan {
    /// Pairs of each member.
    pub(in crate::family) counts: Vec<usize>,
    /// For each sorted position, the row of the chunk it reads.
    pub(in crate::family) order: Vec<i32>,
    /// For each sorted position, its pair `row * top_k + rank`: the place its output lands in.
    pub(in crate::family) scatter: Vec<i32>,
}

/// Counting sort of the pairs by member: within a member the rows keep their chunk order. An id
/// outside `0..n_member` is a fault mark (-1) or a corrupt word, and refuses the chunk.
pub(in crate::family) fn plan_pairs(
    picks: &[i32],
    n_member: usize,
    top_k: usize,
) -> Result<PairPlan, EngError> {
    let mut counts = vec![0usize; n_member];
    for &pick in picks {
        let member = usize::try_from(pick).ok().filter(|&e| e < n_member);
        let Some(member) = member else {
            return Err(EngError::ModelOutput(
                "EXL3 moe chunk: the router picked no valid expert (non-finite logits)".into(),
            ));
        };
        counts[member] += 1;
    }
    let mut next = Vec::with_capacity(n_member);
    let mut at = 0;
    for &count in &counts {
        next.push(at);
        at += count;
    }
    let (mut order, mut scatter) = (vec![0i32; picks.len()], vec![0i32; picks.len()]);
    for (pair, &pick) in picks.iter().enumerate() {
        let slot = &mut next[pick as usize];
        order[*slot] = i32::try_from(pair / top_k).map_err(|_| too_many())?;
        scatter[*slot] = i32::try_from(pair).map_err(|_| too_many())?;
        *slot += 1;
    }
    Ok(PairPlan {
        counts,
        order,
        scatter,
    })
}

fn too_many() -> EngError {
    EngError::BadRequest("EXL3 moe chunk: the pair count exceeds i32".into())
}

/// Reads the picks of the chunk back, which waits for the stream.
pub(in crate::family) fn read_ids(
    launch: &Launch<'_>,
    ids: DeviceRange<'_>,
    dims: &MoeDims,
) -> Result<Vec<i32>, EngError> {
    let pairs = dims.rows * dims.top_k;
    let mut raw = vec![0u8; pairs * I32];
    ids.buffer()
        .read_at(launch.stream, ids.offset(), &mut raw)?;
    Ok(raw
        .as_chunks::<I32>()
        .0
        .iter()
        .map(|bytes| i32::from_le_bytes(*bytes))
        .collect())
}

/// The jobs of a routed pass: each member with rows, in member order, cut into tiles of at most
/// [`EXL3_EXPERT_TILE_ROWS`] rows. The rows of consecutive jobs are back to back, which is the
/// layout of the sorted order of the lists.
pub(in crate::family) fn plan_jobs(counts: &[usize]) -> Vec<Exl3Job> {
    let mut jobs = Vec::new();
    let mut row0 = 0usize;
    for (member, &count) in counts.iter().enumerate() {
        let mut left = count;
        while left > 0 {
            let rows = left.min(EXL3_EXPERT_TILE_ROWS);
            jobs.push(Exl3Job {
                member: member as u32,
                row0: row0 as u32,
                rows: rows as u32,
            });
            row0 += rows;
            left -= rows;
        }
    }
    jobs
}

/// Uploads the two row lists and the job table in one write.
pub(in crate::family) fn write_lists(
    launch: &Launch<'_>,
    perm: DeviceRange<'_>,
    (plan, jobs): (&PairPlan, &[Exl3Job]),
) -> Result<(), EngError> {
    let mut bytes: Vec<u8> = plan
        .order
        .iter()
        .chain(&plan.scatter)
        .flat_map(|index| index.to_le_bytes())
        .collect();
    bytes.extend(Exl3Job::table(jobs));
    perm.buffer().write_at(launch.stream, perm.offset(), &bytes)
}

/// The row lists of one routed projection (`exl3.h`).
#[derive(Clone, Copy)]
pub(in crate::family) struct Lists<'b> {
    /// The list the input rows are read through (`None` is the table order).
    pub(in crate::family) input: Option<DeviceRange<'b>>,
    /// The list the output rows are written through (`None` is the table order).
    pub(in crate::family) output: Option<DeviceRange<'b>>,
    /// Distinct input rows the kernel reads (`x`'s row count).
    pub(in crate::family) x_rows: usize,
    /// Rows the output takes over the whole call.
    pub(in crate::family) out_rows: usize,
}

/// The job table as the launch takes it.
#[derive(Clone, Copy)]
pub(in crate::family) struct Table<'a, 'b> {
    pub(in crate::family) jobs: &'a [Exl3Job],
    pub(in crate::family) range: DeviceRange<'b>,
    pub(in crate::family) rows: usize,
}

/// The rotated-row and partial-sum ranges one routed projection works in.
#[derive(Clone, Copy)]
pub(in crate::family) struct Scratch<'b> {
    pub(in crate::family) xh: DeviceRange<'b>,
    pub(in crate::family) z: DeviceRange<'b>,
}

/// One routed projection over every job: one launch whatever the number of members. `flag` is
/// the sticky fault word the projection ORs its faults into.
#[allow(clippy::too_many_arguments)] // every launch argument, handed to the kernel call
pub(in crate::family) fn project_jobs(
    launch: &Launch<'_>,
    flag: DeviceRange<'_>,
    scratch: Scratch<'_>,
    table: Table<'_, '_>,
    members: &Exl3Members<'_>,
    x: DeviceRange<'_>,
    out: (DeviceRange<'_>, usize, usize),
    lists: Lists<'_>,
) -> Result<(), EngError> {
    let call = jobs_call(flag, scratch, table, members, x, out, lists)?;
    launch.ctx.exl3_jobs(launch.stream, call)
}

/// The job-table call of one routed projection, over the ranges of `scratch`.
#[allow(clippy::too_many_arguments)] // every launch argument, handed to the kernel call
pub(in crate::family) fn jobs_call<'b>(
    flag: DeviceRange<'b>,
    scratch: Scratch<'b>,
    table: Table<'_, 'b>,
    members: &Exl3Members<'b>,
    x: DeviceRange<'b>,
    // The output face, its row stride and its first column.
    out: (DeviceRange<'b>, usize, usize),
    lists: Lists<'b>,
) -> Result<crate::kernels::cuda::resident::exl3::Exl3JobsCall<'b>, EngError> {
    let (out, out_ld, out_col) = out;
    let io = Exl3JobsIo {
        x,
        xh: scratch.xh,
        z: scratch.z,
        out,
        flag,
        jobs: table.range,
        jobs_count: table.jobs.len(),
        max_rows: table
            .jobs
            .iter()
            .map(|job| job.rows as usize)
            .max()
            .unwrap_or(1),
        total_rows: table.rows,
        out_ld,
        out_col,
    };
    let mut call = members.jobs_call(&io);
    call.in_rows = lists.input;
    call.out_rows = lists.output;
    call.x_rows = lists.x_rows;
    call.out_total_rows = lists.out_rows;
    Ok(call)
}

/// Where the pair rotation puts its scratch inside the partial-sum range of `range_bytes` bytes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(in crate::family) struct PairLayout {
    /// Bytes of partial sums one projection needs (`pairs * inter` values), at the head.
    pub(in crate::family) z_bytes: usize,
    /// Offset of the second rotated-row range: `z_bytes` rounded up to [`PAIR_ALIGN`].
    pub(in crate::family) tail: usize,
    /// Bytes of that range (`pairs * hidden` halves).
    pub(in crate::family) xh_bytes: usize,
}

/// Alignment of the second rotated-row range, in bytes.
pub(in crate::family) const PAIR_ALIGN: usize = 256;

/// The layout of the pair's scratch for `pairs` rows of a projection with `inter` outputs over
/// `hidden` inputs, or `None` when the second rotated-row range does not fit behind the partial
/// sums in `range_bytes` bytes (which is the case when `inter` is more than half of `hidden`
/// for the usual `pairs * max(hidden, inter)` values of F32) or the sizes overflow.
pub(in crate::family) fn pair_layout(
    (pairs, inter, hidden): (usize, usize, usize),
    range_bytes: usize,
) -> Option<PairLayout> {
    const F16: usize = 2;
    const F32: usize = 4;
    let z_bytes = pairs.checked_mul(inter)?.checked_mul(F32)?;
    let tail = z_bytes.checked_next_multiple_of(PAIR_ALIGN)?;
    let xh_bytes = pairs.checked_mul(hidden)?.checked_mul(F16)?;
    (tail.checked_add(xh_bytes)? <= range_bytes).then_some(PairLayout {
        z_bytes,
        tail,
        xh_bytes,
    })
}

/// The scratch ranges of the gate and up projections when they rotate their input rows together:
/// the first projection keeps the usual rotated-row range and the head of the partial-sum range,
/// the second takes its rotated rows from the unused tail of the partial-sum range
/// ([`pair_layout`]), so the pair needs no buffer of its own. `None` when the tail is too small.
pub(in crate::family) fn paired_scratch<'b>(
    xh: DeviceRange<'b>,
    z: DeviceRange<'b>,
    dims: &MoeDims,
) -> Result<Option<(Scratch<'b>, Scratch<'b>)>, EngError> {
    let pairs = dims.rows * dims.top_k;
    let Some(layout) = pair_layout((pairs, dims.inter, dims.hidden), z.bytes()) else {
        return Ok(None);
    };
    let z_head = window(z, 0, layout.z_bytes)?;
    let first = Scratch { xh, z: z_head };
    let second = Scratch {
        xh: window(z, layout.tail, layout.xh_bytes)?,
        z: z_head,
    };
    Ok(Some((first, second)))
}

/// The sub-range of `range` at `at` bytes of `len` bytes.
fn window(range: DeviceRange<'_>, at: usize, len: usize) -> Result<DeviceRange<'_>, EngError> {
    if at + len <= range.bytes() {
        range.buffer().range(range.offset() + at, len)
    } else {
        Err(EngError::BadRequest(
            "EXL3 moe chunk: the scratch range is too small".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::project::exl3_jobs_plan::exl3_max_jobs;

    #[test]
    fn the_sort_groups_pairs_by_member_and_keeps_chunk_order_within_one() {
        // Three rows, two picks each, over four members.
        let picks = [2, 0, 2, 3, 0, 2];
        let plan = plan_pairs(&picks, 4, 2).expect("plan");
        assert_eq!(plan.counts, [2, 0, 3, 1]);
        // Member 0: pairs 1 and 4 (rows 0 and 2); member 2: pairs 0, 2 and 5 (rows 0, 1, 2);
        // member 3: pair 3 (row 1).
        assert_eq!(plan.order, [0, 2, 0, 1, 2, 1]);
        // Each sorted position lands in the pair it came from.
        assert_eq!(plan.scatter, [1, 4, 0, 2, 5, 3]);
        for (place, &pair) in plan.scatter.iter().enumerate() {
            assert_eq!(
                plan.order[place] as usize,
                pair as usize / 2,
                "place {place}"
            );
            assert_eq!(
                picks[pair as usize],
                [0, 0, 2, 2, 2, 3][place],
                "place {place}"
            );
        }
        // The scatter is a permutation of the pairs.
        let mut seen = plan.scatter.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..6).collect::<Vec<i32>>());
    }

    /// The job rows of `counts`: `(member, row0, rows)`.
    fn jobs_of(counts: &[usize]) -> Vec<(u32, u32, u32)> {
        plan_jobs(counts)
            .iter()
            .map(|job| (job.member, job.row0, job.rows))
            .collect()
    }

    /// A member's rows are cut at the tile height and the cut is exact at the edges of the
    /// kernel's narrowest CTA: 1, 8, 9, 32, 63 and 64 rows are one job, 65 rows are a full tile
    /// and a one-row tail, and the last short tile of a larger member keeps the remainder.
    /// Mutation: cutting at `EXL3_EXPERT_TILE_ROWS + 1` (or `- 1`) rows fails the 65 and 64 rows.
    #[test]
    fn jobs_cut_at_the_tile_height_and_keep_the_short_tail() {
        assert_eq!(
            EXL3_EXPERT_TILE_ROWS, 64,
            "the literals below are written for 64-row tiles"
        );
        for rows in [1usize, 8, 9, 32, 63, 64] {
            assert_eq!(jobs_of(&[rows]), [(0, 0, rows as u32)], "{rows} rows");
        }
        assert_eq!(jobs_of(&[65]), [(0, 0, 64), (0, 64, 1)]);
        assert_eq!(jobs_of(&[129]), [(0, 0, 64), (0, 64, 64), (0, 128, 1)]);
        assert_eq!(jobs_of(&[150]), [(0, 0, 64), (0, 64, 64), (0, 128, 22)]);
    }

    /// Members keep their order, a member without rows has no job (a zero-row job is outside
    /// the kernel's limits), and the first row of a job is the first sorted row of its member.
    /// Mutation: pushing a job for an empty member fails the empty-member row; restarting `row0`
    /// at zero for each member fails the boundaries.
    #[test]
    fn jobs_follow_member_order_skip_empty_members_and_start_at_the_member_boundary() {
        let counts = [0, 5, 64, 65, 0, 150, 1];
        assert_eq!(
            jobs_of(&counts),
            [
                (1, 0, 5),
                (2, 5, 64),
                (3, 69, 64),
                (3, 133, 1),
                (5, 134, 64),
                (5, 198, 64),
                (5, 262, 22),
                (6, 284, 1),
            ]
        );
        assert!(plan_jobs(&[0, 0, 0]).is_empty());
        assert_eq!(jobs_of(&[0, 0, 7]), [(2, 0, 7)]);
    }

    /// The jobs of any counts tile the sorted rows exactly once: each job lies inside the rows,
    /// the jobs are back to back and in order (so no two jobs share an input row, a scratch row
    /// or an output row), and each is within the kernel's limits.
    /// Mutation: dropping the `row0 += rows` advance, or letting a tail job repeat its tile,
    /// breaks the back-to-back check.
    #[test]
    fn jobs_tile_the_sorted_rows_exactly_once_for_any_counts() {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = move |limit: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % limit
        };
        for round in 0..200 {
            let members = 1 + next(256) as usize;
            let counts: Vec<usize> = (0..members)
                .map(|_| match next(4) {
                    0 => 0,
                    1 => 1 + next(8) as usize,
                    2 => 1 + next(300) as usize,
                    _ => EXL3_EXPERT_TILE_ROWS * (1 + next(3) as usize),
                })
                .collect();
            let total: usize = counts.iter().sum();
            let jobs = plan_jobs(&counts);
            let mut at = 0usize;
            for job in &jobs {
                assert_eq!(job.row0 as usize, at, "round {round}: back to back");
                assert!(
                    (1..=EXL3_EXPERT_TILE_ROWS as u32).contains(&job.rows),
                    "round {round}"
                );
                assert!((job.member as usize) < members, "round {round}");
                at += job.rows as usize;
            }
            assert_eq!(at, total, "round {round}: every row once");
            // One job run per member with rows, in order, and its jobs cover its count.
            for (member, &count) in counts.iter().enumerate() {
                let covered: u32 = jobs
                    .iter()
                    .filter(|job| job.member as usize == member)
                    .map(|job| job.rows)
                    .sum();
                assert_eq!(covered as usize, count, "round {round}: member {member}");
            }
            assert!(
                jobs.windows(2).all(|w| w[0].member <= w[1].member),
                "round {round}"
            );
        }
    }

    /// The table never outgrows the room the plan gives it, nor the kernel's job limit, at the
    /// worst shapes of the largest chunk: every pair on its own member, every member one tile
    /// and a row, and all pairs on a few members.
    /// Mutation: cutting tiles at half the height doubles the jobs and fails the bound.
    #[test]
    fn the_job_count_stays_within_the_planned_table_and_the_kernel_limit() {
        let (members, pairs) = (256usize, 2048 * 8);
        let shapes: [Vec<usize>; 4] = [
            vec![pairs / members; members],
            {
                // Members of one tile and a row each, the last one short.
                let mut counts =
                    vec![EXL3_EXPERT_TILE_ROWS + 1; pairs / (EXL3_EXPERT_TILE_ROWS + 1)];
                counts.push(pairs - counts.iter().sum::<usize>());
                counts
            },
            {
                let mut counts = vec![0; members];
                counts[3] = pairs;
                counts
            },
            {
                let mut counts = vec![1; members];
                counts[0] = pairs - (members - 1);
                counts
            },
        ];
        for counts in shapes {
            assert_eq!(counts.iter().sum::<usize>(), pairs);
            let jobs = plan_jobs(&counts).len();
            assert!(jobs <= exl3_max_jobs(members, pairs), "{jobs} jobs");
            assert!(jobs <= 4096, "{jobs} jobs");
        }
    }

    #[test]
    fn a_pick_outside_the_members_refuses_the_chunk() {
        for bad in [-1, 4, i32::MAX] {
            let refused = plan_pairs(&[0, bad], 4, 2).expect_err("a refusal");
            assert!(matches!(refused, EngError::ModelOutput(_)), "{bad}");
        }
    }

    #[test]
    fn a_full_chunk_sorts_every_pair_exactly_once() {
        let (rows, top_k, members) = (2048, 8, 256);
        let picks: Vec<i32> = (0..rows * top_k)
            .map(|p| (((p * 2_654_435_761usize) >> 7) % members) as i32)
            .collect();
        let plan = plan_pairs(&picks, members, top_k).expect("plan");
        assert_eq!(plan.counts.iter().sum::<usize>(), rows * top_k);
        let mut seen = vec![false; rows * top_k];
        for &pair in &plan.scatter {
            assert!(!std::mem::replace(&mut seen[pair as usize], true));
        }
    }

    #[test]
    fn the_pair_layout_puts_the_second_rotated_rows_behind_the_partial_sums() {
        // Qwen3.6 shape: hidden 2048, inter 512, 16,384 pairs, partial sums of pairs * 2048 F32.
        let range = 16_384 * 2048 * 4;
        let layout = pair_layout((16_384, 512, 2048), range).expect("fits");
        assert_eq!(layout.z_bytes, 16_384 * 512 * 4);
        assert_eq!(layout.tail, layout.z_bytes, "already a multiple of 256");
        assert_eq!(layout.xh_bytes, 16_384 * 2048 * 2);
        assert!(layout.tail + layout.xh_bytes <= range);
        // The tail rounds up to 256 bytes.
        let layout = pair_layout((3, 100, 64), 1 << 20).expect("fits");
        assert_eq!(
            (layout.z_bytes, layout.tail, layout.xh_bytes),
            (1200, 1280, 384)
        );
    }

    #[test]
    fn the_pair_layout_fits_exactly_and_refuses_one_byte_less() {
        let need = 1280 + 384;
        assert!(pair_layout((3, 100, 64), need).is_some());
        assert_eq!(pair_layout((3, 100, 64), need - 1), None);
        assert_eq!(pair_layout((3, 100, 64), 0), None);
    }

    #[test]
    fn the_pair_layout_refuses_an_inter_above_half_of_hidden() {
        // The partial-sum range is pairs * max(hidden, inter) F32: the head takes 4 * inter and
        // the second rotated rows 2 * hidden of the 4 * hidden bytes a pair row owns.
        for (hidden, inter, expect) in [
            (2048, 1024, true),
            (2048, 1025, false),
            (2048, 2048, false),
            (512, 2048, false),
            (4096, 2048, true),
        ] {
            let pairs = 64;
            let range = pairs * hidden.max(inter) * 4;
            assert_eq!(
                pair_layout((pairs, inter, hidden), range).is_some(),
                expect,
                "hidden {hidden} inter {inter}"
            );
        }
    }

    #[test]
    fn the_pair_layout_refuses_sizes_that_overflow() {
        assert_eq!(pair_layout((usize::MAX, 2, 2), usize::MAX), None);
        assert_eq!(pair_layout((2, usize::MAX, 2), usize::MAX), None);
        assert_eq!(pair_layout((2, 2, usize::MAX), usize::MAX), None);
    }
}

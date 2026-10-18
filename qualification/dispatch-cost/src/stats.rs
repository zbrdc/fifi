use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct Summary {
    pub samples: usize,
    pub median_us: f64,
    pub p95_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
}

/// Median and nearest-rank 95th percentile of microsecond samples.
pub fn summarize(samples: &[f64]) -> Summary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    let median = if count == 0 {
        0.0
    } else if count % 2 == 1 {
        sorted[count / 2]
    } else {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    };
    let rank = ((count as f64) * 0.95).ceil() as usize;
    let p95 = if count == 0 {
        0.0
    } else {
        sorted[rank.clamp(1, count) - 1]
    };
    Summary {
        samples: count,
        median_us: median,
        p95_us: p95,
        min_us: sorted.first().copied().unwrap_or(0.0),
        max_us: sorted.last().copied().unwrap_or(0.0),
        mean_us: if count == 0 {
            0.0
        } else {
            sorted.iter().sum::<f64>() / count as f64
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{Summary, summarize};

    fn assert_summary(actual: Summary, expected: Summary) {
        assert_eq!(actual.samples, expected.samples);
        assert_eq!(actual.median_us, expected.median_us);
        assert_eq!(actual.p95_us, expected.p95_us);
        assert_eq!(actual.min_us, expected.min_us);
        assert_eq!(actual.max_us, expected.max_us);
        assert_eq!(actual.mean_us, expected.mean_us);
    }

    #[test]
    fn empty_input_returns_zero_summary() {
        assert_summary(
            summarize(&[]),
            Summary {
                samples: 0,
                median_us: 0.0,
                p95_us: 0.0,
                min_us: 0.0,
                max_us: 0.0,
                mean_us: 0.0,
            },
        );
    }

    #[test]
    fn single_sample_populates_every_statistic() {
        assert_summary(
            summarize(&[7.0]),
            Summary {
                samples: 1,
                median_us: 7.0,
                p95_us: 7.0,
                min_us: 7.0,
                max_us: 7.0,
                mean_us: 7.0,
            },
        );
    }

    #[test]
    fn odd_sample_count_uses_middle_value_as_median() {
        assert_summary(
            summarize(&[9.0, 1.0, 5.0]),
            Summary {
                samples: 3,
                median_us: 5.0,
                p95_us: 9.0,
                min_us: 1.0,
                max_us: 9.0,
                mean_us: 5.0,
            },
        );
    }

    #[test]
    fn even_sample_count_averages_middle_values_for_median() {
        assert_summary(
            summarize(&[8.0, 2.0, 6.0, 4.0]),
            Summary {
                samples: 4,
                median_us: 5.0,
                p95_us: 8.0,
                min_us: 2.0,
                max_us: 8.0,
                mean_us: 5.0,
            },
        );
    }

    #[test]
    fn nearest_rank_p95_uses_rank_nineteen_for_twenty_samples() {
        assert_summary(
            summarize(&[
                20.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0,
            ]),
            Summary {
                samples: 20,
                median_us: 10.5,
                p95_us: 19.0,
                min_us: 1.0,
                max_us: 20.0,
                mean_us: 10.5,
            },
        );
    }

    #[test]
    fn duplicate_values_are_retained_in_summary() {
        assert_summary(
            summarize(&[2.0, 2.0, 6.0, 6.0]),
            Summary {
                samples: 4,
                median_us: 4.0,
                p95_us: 6.0,
                min_us: 2.0,
                max_us: 6.0,
                mean_us: 4.0,
            },
        );
    }

    #[test]
    fn negative_values_are_sorted_and_summarized() {
        assert_summary(
            summarize(&[-4.0, -1.0, -7.0]),
            Summary {
                samples: 3,
                median_us: -4.0,
                p95_us: -1.0,
                min_us: -7.0,
                max_us: -1.0,
                mean_us: -4.0,
            },
        );
    }
}

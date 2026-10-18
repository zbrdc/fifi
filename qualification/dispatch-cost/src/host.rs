/// Load-average parsing for `sysctl -n vm.loadavg`.
///
/// The command prints three averages, usually inside braces. Each position keeps its place.
/// A malformed or absent word leaves that position unknown, so a later average never shifts
/// into an earlier field.
/// A non-finite word is also unknown. A gate must not read `nan` or `inf` as a load.
pub(crate) fn parse_load_averages(text: &str) -> [Option<f64>; 3] {
    let text = text.trim();
    let text = text.strip_prefix('{').unwrap_or(text);
    let text = text.strip_suffix('}').unwrap_or(text);
    let mut words = text.split_whitespace();
    let parse = |word: Option<&str>| {
        word.and_then(|word| word.parse::<f64>().ok())
            .filter(|value| value.is_finite())
    };
    [
        parse(words.next()),
        parse(words.next()),
        parse(words.next()),
    ]
}

#[cfg(test)]
mod tests {
    use super::parse_load_averages;

    #[test]
    fn brace_delimited_averages_keep_their_positions() {
        assert_eq!(
            parse_load_averages("{ 1.0 0.5 0.25 }"),
            [Some(1.0), Some(0.5), Some(0.25)]
        );
        assert_eq!(
            parse_load_averages("1.0 0.5 0.25"),
            [Some(1.0), Some(0.5), Some(0.25)]
        );
        assert_eq!(
            parse_load_averages("{ 0 0 0 }"),
            [Some(0.0), Some(0.0), Some(0.0)]
        );
    }

    #[test]
    fn malformed_first_average_does_not_shift_positions() {
        assert_eq!(
            parse_load_averages("{ bad 1.0 0.5 }"),
            [None, Some(1.0), Some(0.5)]
        );
    }

    #[test]
    fn malformed_middle_and_missing_averages_remain_unknown() {
        assert_eq!(
            parse_load_averages("{ 1.0 bad 0.5 }"),
            [Some(1.0), None, Some(0.5)]
        );
        assert_eq!(
            parse_load_averages("{ 1.0 0.5 }"),
            [Some(1.0), Some(0.5), None]
        );
        assert_eq!(parse_load_averages(""), [None, None, None]);
    }

    #[test]
    fn malformed_third_average_and_one_word_input_remain_unknown() {
        assert_eq!(
            parse_load_averages("{ 1.0 0.5 bad }"),
            [Some(1.0), Some(0.5), None]
        );
        assert_eq!(parse_load_averages("{ 1.0 }"), [Some(1.0), None, None]);
    }

    #[test]
    fn non_finite_first_average_is_unknown() {
        assert_eq!(
            parse_load_averages("{ nan 1.0 0.5 }"),
            [None, Some(1.0), Some(0.5)]
        );
        assert_eq!(
            parse_load_averages("{ inf 1.0 0.5 }"),
            [None, Some(1.0), Some(0.5)]
        );
        assert_eq!(
            parse_load_averages("{ -inf 1.0 0.5 }"),
            [None, Some(1.0), Some(0.5)]
        );
    }

    #[test]
    fn non_finite_middle_average_is_unknown() {
        assert_eq!(
            parse_load_averages("{ 1.0 NaN 0.5 }"),
            [Some(1.0), None, Some(0.5)]
        );
        assert_eq!(
            parse_load_averages("{ 1.0 +inf 0.5 }"),
            [Some(1.0), None, Some(0.5)]
        );
        assert_eq!(
            parse_load_averages("{ 1.0 -infinity 0.5 }"),
            [Some(1.0), None, Some(0.5)]
        );
    }

    #[test]
    fn non_finite_last_average_is_unknown() {
        assert_eq!(
            parse_load_averages("{ 1.0 0.5 nan }"),
            [Some(1.0), Some(0.5), None]
        );
        assert_eq!(
            parse_load_averages("{ 1.0 0.5 INF }"),
            [Some(1.0), Some(0.5), None]
        );
    }

    #[test]
    fn overflowed_average_is_unknown() {
        assert_eq!(
            parse_load_averages("{ 1e400 1.0 0.5 }"),
            [None, Some(1.0), Some(0.5)]
        );
    }
}

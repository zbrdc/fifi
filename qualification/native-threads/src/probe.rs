pub fn swap_used_mib(report: &str) -> Result<f64, String> {
    let used = report
        .split("used = ")
        .nth(1)
        .ok_or_else(|| format!("swap report has no used field: {report:?}"))?;
    let token = used
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("swap report has an empty used field: {report:?}"))?;
    let number = token
        .strip_suffix('M')
        .ok_or_else(|| format!("swap used value lacks the M unit: {token:?}"))?;
    number
        .parse::<f64>()
        .map_err(|_| format!("swap used value is not numeric: {token:?}"))
}

#[cfg(test)]
mod tests {
    use super::swap_used_mib;

    #[test]
    fn parses_used_mib_from_the_macos_report() {
        let report = "total = 2048.00M  used = 12.50M  free = 2035.50M\n";
        assert_eq!(swap_used_mib(report), Ok(12.5));
    }

    #[test]
    fn rejects_a_report_without_a_used_field() {
        let report = "total = 2048.00M  free = 2035.50M\n";
        assert!(swap_used_mib(report).is_err());
    }

    #[test]
    fn rejects_a_non_numeric_used_value() {
        let report = "total = 2048.00M  used = abcM  free = 2035.50M\n";
        assert!(swap_used_mib(report).is_err());
    }

    #[test]
    fn rejects_a_used_value_without_the_m_unit() {
        let report = "total = 2048.00M  used = 12.50  free = 2035.50M\n";
        assert!(swap_used_mib(report).is_err());
    }

    #[test]
    fn rejects_an_empty_report() {
        assert!(swap_used_mib("").is_err());
    }
}

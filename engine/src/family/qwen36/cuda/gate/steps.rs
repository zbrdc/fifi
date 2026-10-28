//! Which token is fed to each of the 128 decode steps (teacher forcing) and the logit of the reference token.
//! Pure host code (U13).

/// Number of decode steps per run; row `j` is the logit row after the prompt and `j` generated tokens.
pub(super) const N_STEPS: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Mode {
    Free,
    Forced,
}

/// Input of decode step `j`. Step 0 is fed the last prompt token and yields row 0. After that a free run feeds its own
/// previous argmax and a forced run feeds the reference token of the previous row.
pub(super) fn input_for(
    mode: Mode,
    j: usize,
    last_prompt: u32,
    forced: &[u32],
    prev_argmax: u32,
) -> Result<u32, &'static str> {
    if j == 0 {
        return Ok(last_prompt);
    }
    match mode {
        Mode::Free => Ok(prev_argmax),
        Mode::Forced => forced
            .get(j - 1)
            .copied()
            .ok_or("forced stream shorter than the run"),
    }
}

/// The engine's own logit for the reference token of row `j` (None when the token is outside the row).
pub(super) fn logit_of(row: &[f32], token: u32) -> Option<f32> {
    row.get(token as usize).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_zero_is_fed_the_last_prompt_token_in_both_modes() {
        assert_eq!(input_for(Mode::Free, 0, 77, &[1, 2], 9), Ok(77));
        assert_eq!(input_for(Mode::Forced, 0, 77, &[1, 2], 9), Ok(77));
    }

    #[test]
    fn a_free_run_feeds_its_own_argmax() {
        assert_eq!(input_for(Mode::Free, 5, 77, &[1, 2, 3, 4, 5], 42), Ok(42));
    }

    #[test]
    fn a_forced_run_feeds_the_reference_token_of_the_previous_row() {
        let forced = [10, 20, 30];
        assert_eq!(input_for(Mode::Forced, 1, 77, &forced, 999), Ok(10));
        assert_eq!(input_for(Mode::Forced, 3, 77, &forced, 999), Ok(30));
    }

    #[test]
    fn a_forced_stream_that_is_too_short_is_an_error() {
        assert!(input_for(Mode::Forced, 4, 77, &[10, 20, 30], 0).is_err());
    }

    #[test]
    fn the_reference_token_logit_is_read_at_its_own_index() {
        assert_eq!(N_STEPS, 128);
        assert_eq!(logit_of(&[0.5, 1.5, 2.5], 2), Some(2.5));
        assert_eq!(logit_of(&[0.5], 9), None);
    }
}

pub const CALLER_COUNT: u8 = 2;
pub const JOBS_PER_CALLER: u32 = 5000;
pub const TOTAL_JOBS: u32 = CALLER_COUNT as u32 * JOBS_PER_CALLER;

pub const FIXTURE_A: u8 = 0;
pub const FIXTURE_B: u8 = 1;

pub const OUTPUT_DTYPE: &str = "F32";
pub const OUTPUT_SHAPE: [u32; 2] = [2, 2];
pub const FIXTURE_A_OUTPUT: [f32; 4] = [-288.0, -1216.0, -8.0, -12.0];
pub const FIXTURE_B_OUTPUT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

pub fn fixture_for(caller: u8, job: u32) -> u8 {
    ((caller as u32 + job) % 2) as u8
}

pub fn expected_output(fixture: u8) -> Output {
    let values = if fixture == FIXTURE_A {
        FIXTURE_A_OUTPUT
    } else {
        FIXTURE_B_OUTPUT
    };
    Output {
        dtype: OUTPUT_DTYPE,
        shape: OUTPUT_SHAPE,
        values,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub caller: u8,
    pub job: u32,
    pub fixture: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    pub dtype: &'static str,
    pub shape: [u32; 2],
    pub values: [f32; 4],
}

impl Output {
    pub fn same_bytes(&self, other: &Output) -> bool {
        self.dtype == other.dtype
            && self.shape == other.shape
            && self.values.map(f32::to_bits) == other.values.map(f32::to_bits)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Response {
    pub request: Request,
    pub output: Output,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_selection_follows_the_identifier_sum_modulo_two() {
        assert_eq!(fixture_for(0, 0), FIXTURE_A);
        assert_eq!(fixture_for(0, 1), FIXTURE_B);
        assert_eq!(fixture_for(1, 0), FIXTURE_B);
        assert_eq!(fixture_for(1, 1), FIXTURE_A);
        assert_eq!(fixture_for(1, 4999), FIXTURE_A);
    }

    #[test]
    fn fixture_outputs_match_the_pinned_literals() {
        assert_eq!(expected_output(FIXTURE_A).values, FIXTURE_A_OUTPUT);
        assert_eq!(expected_output(FIXTURE_B).values, FIXTURE_B_OUTPUT);
        for fixture in [FIXTURE_A, FIXTURE_B] {
            let output = expected_output(fixture);
            assert_eq!(output.dtype, OUTPUT_DTYPE);
            assert_eq!(output.shape, OUTPUT_SHAPE);
        }
    }

    #[test]
    fn byte_equality_distinguishes_signed_zero() {
        let expected = expected_output(FIXTURE_B);
        let neg_zero = Output {
            values: [0.0, -0.0, 0.0, 0.0],
            ..expected
        };
        assert!(!expected.same_bytes(&neg_zero));
        assert!(expected.same_bytes(&expected));
        assert!(!expected.same_bytes(&expected_output(FIXTURE_A)));
    }
}

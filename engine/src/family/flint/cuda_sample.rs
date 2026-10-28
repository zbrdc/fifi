//! Generic device selection; stochastic draws retain the shared sampler's RNG/order.

use crate::{
    EngError,
    family::sampler::Sampler,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceStream,
        resident::{
            shortlist::{self, ShortlistCall},
            step_params::{
                ARGMAX_PART_BYTES, ArgmaxCall, RESULT_PAIR_BYTES, ResultPair, STEP_PARAMS_BYTES,
                StepParams,
            },
        },
    },
};

type Result<T> = std::result::Result<T, EngError>;

pub(super) struct SamplingScratch {
    params: DeviceBuffer,
    part: DeviceBuffer,
    result: DeviceBuffer,
    shortlist: DeviceBuffer,
}

impl SamplingScratch {
    pub(super) fn new(ctx: &CudaContext) -> Result<Self> {
        Ok(Self {
            params: ctx.allocate(STEP_PARAMS_BYTES)?,
            part: ctx.allocate(ARGMAX_PART_BYTES)?,
            result: ctx.allocate(RESULT_PAIR_BYTES)?,
            shortlist: ctx.allocate(shortlist::SCRATCH_BYTES)?,
        })
    }

    pub(super) fn sample(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        head: &DeviceBuffer,
        logits: &[f32],
        sampler: &mut Sampler,
        vocab: usize,
    ) -> Result<u32> {
        let visible = logits
            .get(..vocab)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("Flint sampler vocabulary exceeds the head or is empty"))?;
        if sampler.is_greedy() {
            return self.greedy(ctx, stream, head, vocab);
        }
        if (1..=shortlist::K as u32).contains(&sampler.params().top_k()) {
            ctx.shortlist(
                stream,
                ShortlistCall {
                    logits: head.range(0, vocab * 4)?,
                    scratch: self.shortlist.range(0, self.shortlist.bytes())?,
                    n: vocab,
                },
            )?;
            let mut bytes = vec![0; shortlist::RESULT_BYTES];
            self.shortlist
                .read_at(stream, shortlist::RESULT_OFFSET, &mut bytes)?;
            return sampler.sample_shortlist(&candidates(&bytes, vocab)?, vocab);
        }
        sampler.sample(visible)
    }

    fn greedy(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        head: &DeviceBuffer,
        vocab: usize,
    ) -> Result<u32> {
        self.params
            .write_at(stream, 0, &StepParams::new(0, 0)?.to_bytes())?;
        ctx.argmax(
            stream,
            ArgmaxCall {
                n: vocab,
                logits: head.range(0, vocab * 4)?,
                part: self.part.range(0, self.part.bytes())?,
                params: self.params.range(0, self.params.bytes())?,
                result: Some(self.result.range(0, RESULT_PAIR_BYTES)?),
            },
        )?;
        let mut bytes = [0; RESULT_PAIR_BYTES];
        self.result.read_at(stream, 0, &mut bytes)?;
        accepted_pair(ResultPair::from_bytes(bytes), vocab)
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Backend {
        operation: "sample",
        message: message.into(),
    }
}

fn accepted_pair(pair: ResultPair, vocab: usize) -> Result<u32> {
    if pair.status.0 != 0 {
        return Err(invalid("device argmax returned nonzero status"));
    }
    let token = u32::try_from(pair.into_token()?)
        .map_err(|_| invalid("device argmax returned a negative token"))?;
    if token as usize >= vocab {
        return Err(invalid(
            "device argmax token exceeds the visible vocabulary",
        ));
    }
    Ok(token)
}

fn candidates(bytes: &[u8], vocab: usize) -> Result<Vec<(u32, f32)>> {
    if bytes.len() != shortlist::RESULT_BYTES
        || bytes[shortlist::K * 8..shortlist::K * 8 + 4] != [0; 4]
    {
        return Err(invalid("device shortlist certificate is invalid"));
    }
    Ok(bytes[..vocab.min(shortlist::K) * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|entry| {
            (
                u32::from_le_bytes(entry[4..8].try_into().expect("candidate id word")),
                f32::from_le_bytes(entry[..4].try_into().expect("candidate logit word")),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::resident::step_params::StepStatus;

    #[test]
    fn argmax_acceptance_checks_all_status_bits_and_visible_vocabulary() {
        assert_eq!(
            accepted_pair(
                ResultPair {
                    token: 2,
                    status: StepStatus(0)
                },
                3
            )
            .unwrap(),
            2
        );
        for (token, status) in [(2, 8), (2, 2), (3, 0), (-1, 0)] {
            assert!(
                accepted_pair(
                    ResultPair {
                        token,
                        status: StepStatus(status)
                    },
                    3
                )
                .is_err()
            );
        }
    }

    #[test]
    fn shortlist_certificate_is_checked_before_shared_sampler_draw() {
        let mut bytes = vec![0; shortlist::RESULT_BYTES];
        assert_eq!(candidates(&bytes, 2).unwrap().len(), 2);
        bytes[shortlist::K * 8] = 1;
        assert!(candidates(&bytes, 2).is_err());
        assert!(candidates(&[], 2).is_err());
    }
}

//! Test-only reduction: log-softmax over the full row, with exactly five entries read back.
use crate::EngError;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream, require_fit};
use crate::kernels::cuda::native::resident::top_logits;

impl CudaContext {
    pub(crate) fn top_logits(
        &self,
        stream: &DeviceStream,
        row: DeviceRange<'_>,
        n: usize,
    ) -> Result<[(u32, f32); 5], EngError> {
        self.ensure_usable()?;
        if !(5..=4_194_304).contains(&n) {
            return Err(EngError::BadRequest(
                "top-five trace requires 5 to 4194304 logits".to_owned(),
            ));
        }
        require_fit("logits", &row, n * 4, 4)?;
        let x = self.raw_range(&row)?;
        let out = self.upload(&[0u8; 44])?;
        let range = out.range(0, 44)?;
        let raw = self.raw_range(&range)?;
        self.launch_on(stream, |s| top_logits::launch(s, raw, x, n))?;
        stream.synchronize()?;
        let mut bytes = [0u8; 44];
        self.download(&out, &mut bytes)?;
        if bytes[40..] != [0; 4] {
            return Err(EngError::BadRequest(
                "top-five trace found non-finite logits".to_owned(),
            ));
        }
        let mut top = [(0, 0.0); 5];
        for (i, entry) in top.iter_mut().enumerate() {
            let id = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("id word"));
            let lp = f32::from_le_bytes(
                bytes[20 + i * 4..24 + i * 4]
                    .try_into()
                    .expect("logprob word"),
            );
            if id as usize >= n || !lp.is_finite() {
                return Err(EngError::BadRequest("invalid top-five trace".to_owned()));
            }
            *entry = (id, lp);
        }
        Ok(top)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn top_logits_uses_the_complete_row_and_breaks_ties_by_id() {
        use crate::kernels::cuda::{CudaContext, testkit};
        if !testkit::device_present("top five qualification") {
            return;
        }
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        let row = testkit::upload_f32(&ctx, &[0.0; 32]);
        let top = ctx
            .top_logits(&stream, row.range(0, 128).expect("row"), 32)
            .expect("top five");
        for (rank, (id, logprob)) in top.iter().enumerate() {
            assert_eq!(*id, rank as u32);
            assert!((logprob + 32.0f32.ln()).abs() < 1e-6);
        }
        assert!(
            ctx.top_logits(&stream, row.range(0, 128).expect("row"), 4)
                .is_err()
        );
        let mut values = [0.0; 32];
        values[31] = 2.0;
        values[28] = 1.0;
        values[29] = 1.0;
        values[1] = -1.0;
        let varied = testkit::upload_f32(&ctx, &values);
        let top = ctx
            .top_logits(&stream, varied.range(0, 128).expect("row"), 32)
            .expect("top five");
        assert_eq!(top.map(|entry| entry.0), [31, 28, 29, 0, 2]);
        let normalizer = (2.0f64.exp() + 2.0 * 1.0f64.exp() + (-1.0f64).exp() + 28.0).ln();
        for (id, logprob) in top {
            assert!(
                (f64::from(logprob) - (f64::from(values[id as usize]) - normalizer)).abs() < 1e-6
            );
        }
        let bad = testkit::upload_f32(&ctx, &[f32::NAN; 32]);
        assert!(
            ctx.top_logits(&stream, bad.range(0, 128).expect("row"), 32)
                .is_err()
        );
    }
}

# Pinned publisher embedding rounding fixture

Model: RecursiveSI/flint-int4, revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
Publisher: Rec `e53663b262af4cb1ca498e596ca8b1a2d79e0800`, MLX 0.32.0, mlx-lm 0.31.3.
These are only embedding rows 0 and 150000, in that order.
The fixture retains INT8 codes, BF16 group64 scales/additive biases, and the captured publisher CPU dequantization output.
The golden output is a lossless F32 container for the publisher BF16 result.
The CPU dequantizer rounds the product to BF16, then rounds the sum to BF16.
The CPU fixture is a negative witness for the GPU policy. Its bytes remain unchanged.
It does not qualify GPU attention, routed activation fusion, or full-model parity.

SHA256:
- embedding-weight.bin: `8b3d9c263ad7852fa0c4faeff2be6bee2f9c6b41930ef3e144df855ae1232cdd`
- embedding-scales.bin: `5fd6bb86b89dbfd0537703e64b3cfd88c9cbb12ec9faeb743322de9672f2ac5e`
- embedding-biases.bin: `39d4fedcb07ae54e6e1916a44c1c21ecc9dde1732ad0bc3807d38629186192c8`
- embedding-publisher-dequant.f32.bin: `41e6982610cd7051458a79d1cb039c98040e7203c612df11b54f5d4cfb0de849`

## GPU fixture

The GPU fixture contains rows 198 and 2, in that order.
The publisher ran these two frozen token IDs on its GPU on 2026-10-22.
An observation wrapper saved the actual input to layer 0. The wrapper delegated every original operation.
The saved BF16 values use a lossless F32 container.
All 4096 elements match F32 multiply-add with one final BF16 rounding.
Only 1411 elements match the CPU policy with two BF16 rounds.
The production test requires exact equality with the GPU values. No tolerance applies.

The model revision, publisher commit, and package versions are the same as the CPU fixture.
This fixture does not qualify attention, routed activation fusion, or full-model parity.

SHA256:
- embedding-gpu-weight.bin: `746c20bf96843c85d0c8e820a429331b6b0be856d483964e7d1cc5cebc464266`
- embedding-gpu-scales.bin: `e4be2f1ae2e4151115ae264bed3f9f829bfd0f65141f4683e2d17918e64c3db3`
- embedding-gpu-biases.bin: `9f71b2ddb3cfc0c2ee59532e702785202830293f17608912e3ef05658d2f30dd`
- embedding-gpu-publisher.f32.bin: `f8d5a92abc274c93b00bffd9f6cdfa54df38c86f2a3589fb0921b42e25fb6f98`

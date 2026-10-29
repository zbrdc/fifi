# Flint embedding rounding fixtures

Model: `RecursiveSI/flint-int4`, revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
Reference outputs: the model publisher's own source code at commit `e53663b262af4cb1ca498e596ca8b1a2d79e0800`, run with MLX 0.32.0 and mlx-lm 0.31.3.

The embedding is stored as INT8 codes with BF16 scales and additive biases, one pair per group of 64 values.
The two fixtures below hold the stored bytes of two embedding rows each, and the publisher's dequantized output for those rows.
The BF16 outputs are stored losslessly as F32.

The publisher's GPU path computes each value with an F32 multiply-add and rounds once to BF16.
Its CPU dequantizer instead rounds the product to BF16 and then rounds the sum to BF16.
The engine follows the GPU path.

## CPU fixture

Embedding rows 0 and 150000, in that order, dequantized by the publisher code on the CPU.
All 4096 elements match the two-rounding CPU rule.
The test `publisher_cpu_fixture_is_not_the_gpu_rounding_policy` (`engine/src/family/flint/precision.rs`) checks that the engine output differs from this fixture, which shows that the engine does not use the CPU rule.
It does not qualify GPU attention, routed activation fusion, or full-model parity.

SHA256:
- embedding-weight.bin: `8b3d9c263ad7852fa0c4faeff2be6bee2f9c6b41930ef3e144df855ae1232cdd`
- embedding-scales.bin: `5fd6bb86b89dbfd0537703e64b3cfd88c9cbb12ec9faeb743322de9672f2ac5e`
- embedding-biases.bin: `39d4fedcb07ae54e6e1916a44c1c21ecc9dde1732ad0bc3807d38629186192c8`
- embedding-publisher-dequant.f32.bin: `41e6982610cd7051458a79d1cb039c98040e7203c612df11b54f5d4cfb0de849`

## GPU fixture

Embedding rows 198 and 2, in that order, from the publisher code running on the GPU.
A wrapper recorded the actual input to layer 0 and passed every original operation through unchanged.
All 4096 elements match an F32 multiply-add with one final BF16 rounding.
Only 1411 elements match the two-rounding CPU rule.
The test `embedding_matches_publisher_gpu_values_exactly` requires exact equality with these values. No tolerance applies.
This fixture does not qualify attention, routed activation fusion, or full-model parity.

SHA256:
- embedding-gpu-weight.bin: `746c20bf96843c85d0c8e820a429331b6b0be856d483964e7d1cc5cebc464266`
- embedding-gpu-scales.bin: `e4be2f1ae2e4151115ae264bed3f9f829bfd0f65141f4683e2d17918e64c3db3`
- embedding-gpu-biases.bin: `9f71b2ddb3cfc0c2ee59532e702785202830293f17608912e3ef05658d2f30dd`
- embedding-gpu-publisher.f32.bin: `f8d5a92abc274c93b00bffd9f6cdfa54df38c86f2a3589fb0921b42e25fb6f98`

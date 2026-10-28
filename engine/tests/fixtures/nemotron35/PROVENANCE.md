# Nemotron text configuration fixture

This is an extracted configuration fixture from `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at revision `8a115b9d966adce8c059a7bcf47f49a1e57957d8`.

The source `config.json` SHA-256 is `35189a8038d1b1f934e58cba50eb7e97b42a4da569424e35b9fd1048058bc7ea` (8,668,620 bytes). `INVENTORY.json` records the pinned file under the format-1 artifact-worker scratch directory. The fixture retains root `model_type`, `architectures`, `tie_word_embeddings`, and the complete `llm_config`. It omits vision, quantization, and unrelated wrapper metadata.

The fixture is only an architecture-configuration test input. It does not qualify tensor payloads, model loading, inference, vision, MTP quantization, memory use, quality, or throughput.

## Text execution scope

The text descriptor preserves all 88 blocks in `llm_config.layers_block_type` order.
It has 40 Mamba blocks, 40 MoE blocks, and 8 attention blocks.
Embedding and residual policy values come from `llm_config`, not wrapper defaults.

The parser accepts MTP, vision, and audio metadata without enabling their execution.
The descriptor records their configuration paths in `excluded_members()`.
`validate_execution_member()` and `key_plan()` reject access outside text execution with a named-member error.
The fixture retains MTP metadata. Tests add small vision and audio objects to exercise their scope checks.

## Attention rotation

The pinned source and this fixture contain no `rope` or `theta` members in `llm_config`.
The descriptor explicitly selects `NemotronRotation::None`. It does not supply missing RoPE defaults.

The coordinator authorized this selection from the local `transformers 5.10.0.dev0` reference.
The file is `/home/dan/.local/lib/python3.13/site-packages/transformers/models/nemotron_h/modeling_nemotron_h.py`.
Its SHA-256 is `859ff78fae71fa3f8857e4ea0620c43aebea624e310e8422c124c507131398d2`.
`NemotronHAttention.forward` at lines 851–893 projects Q, K, and V without rotary application.
`apply_rotary_pos_emb` has no callers in this file. Its definition and class decorator do not apply rotation.
This is source evidence, not numerical qualification of Rust attention.

## Key plan

The naming authority is the pinned `model.safetensors.index.json` at the same checkpoint revision.
Its SHA-256 is `c368acdbcfe882236af1ec79df4a1fb6127e6ed9f742d59c01c21fe96ddc2741`.
The plan uses `language_model.backbone.embeddings`, `norm_f`, and `language_model.lm_head`.
All three block types use `language_model.backbone.layers.{i}.mixer`.
The plan does not accept guessed `embed_tokens`, `self_attn`, or `moe` namespace aliases.

The real-index test checks 164,989 text keys, including the three `lm_head` tensors.
The format inventory classifies those three tensors as ambiguous, so its text count is 164,986.
Each Mamba and attention block has 13 keys. Each MoE block has 4,109 keys.
The test checks every block's classes and all 512 routed expert indices per MoE block.

Set `NEM35_COMPLETE_INDEX` to the pinned index path to enable the real-index test.
The test reads the index directly and checks its SHA-256 before classifying keys.
Set `NEM35_COMPLETE_CONFIG` to the pinned complete configuration to exercise the original wrapper too.
The default tests use only the extracted fixture and require no network or model payloads.

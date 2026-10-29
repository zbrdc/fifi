# Gemma 4 Humanizer test data

The files here belong to the Hugging Face model `jialinyyzz/humanizer` at revision `2e57b6528a4070c33f3e09ca24e3325a84272f68`, a Gemma 4 fine-tune.
At that revision the model repository is licensed under Apache-2.0; see `THIRD_PARTY_NOTICES.md` at the repository root.
No file here contains model weights.

- `humanizer-config.json`: the unmodified `config.json` of the model:
  https://huggingface.co/jialinyyzz/humanizer/blob/2e57b6528a4070c33f3e09ca24e3325a84272f68/config.json
  SHA-256 `a9f6a5679ff252dce0845866c58992520171113e1e3489631ca8e2e7f47f7bb4` (4,298 bytes).
  The configuration tests in `engine/src/family/gemma4/tests.rs` read it.
- `prompt-cases.json`: five prompt cases for the model's completion prompt.
  The instruction and separator come from the model README.
  Each case records the draft text, the full prompt, its SHA-256, and the token ids from the model's tokenizer with a literal `<bos>` prefix.
  `engine/src/family/gemma4/prompt.rs` checks its prompt builder against these cases.
- `parity-prompt.json`: the prompt `The capital of France is` as token ids (BOS id `2` first), with the SHA-256 of the model's `tokenizer.json`.
  The ignored test `traces_first_position_top_five` in `engine/src/family/gemma4/resident/tests.rs` feeds these ids to a real checkpoint named by `GEMMA4_Q8_FILE`.
- `kernels/`: CUDA kernel reference fixtures. See `kernels/README.md`.

These files support configuration, prompt, and kernel checks only. They do not load weights or enable generation by themselves.

# Flint tokenizer fixture

Unmodified publisher metadata from
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint/tree/2a247be0c0aec895ccb0ac7ca92229ec6df37288),
revision `2a247be0c0aec895ccb0ac7ca92229ec6df37288` (Apache-2.0;
[publisher license](https://huggingface.co/RecursiveSI/flint/blob/2a247be0c0aec895ccb0ac7ca92229ec6df37288/LICENSE)).
The same four assets are published by
[RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0).
The repository keeps the small metadata and golden rows, but not the full `tokenizer.json`.
The seven full-tokenizer tests are ignored by default.
Set `FLINT_MODEL_DIR` to the pinned publisher model directory and run those tests with `--ignored`.
`TokenizerBundle::load` checks all four SHA256 pins before use, including tokenizer hash
`a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68`.
Missing or modified assets fail the opted-in tests. No download or reduced tokenizer fallback occurs.

`goldens.json` records Python Jinja2 rendered bytes and Python Hugging Face
`tokenizers` IDs, library versions, asset hashes, messages, tools, and the
generation-prompt flag. It covers code/Unicode, developer messages, reasoning
replay, tool definitions and assistant/tool history. Both generation modes are
recorded; `enable_thinking` does not alter this publisher template.

To reproduce a reference row, use the recorded versions, load the unmodified
`chat_template.jinja` in `jinja2.Environment(trim_blocks=True,
lstrip_blocks=True)`, replace its `tojson` filter with
`lambda value, ensure_ascii=True: json.dumps(value, ensure_ascii=ensure_ascii)`,
and call `template.render(messages=case['messages'], tools=case['tools'],
add_generation_prompt=reference['generation_prompt'])`. This matches the
Transformers Python-JSON filter, not Jinja's HTML-escaping built-in. Encode the
render with `tokenizers.Tokenizer.from_file('tokenizer.json').encode(rendered,
add_special_tokens=False).ids`. No automatic BOS/EOS is used.

Grammar selection is conservative: only the exact pinned template hash enables
Flint's argument grammar and developer/reasoning replay. Other templates retain
the existing Qwen/default request-rendering and parsing behavior. A future
publisher template revision requires explicit review of this identity binding.

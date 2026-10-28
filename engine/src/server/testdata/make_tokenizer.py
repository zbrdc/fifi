import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# GPT-2 byte to unicode table: byte b is the character table[b], and the vocabulary id of that character is b.
printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
codes = printable[:]
extra = 0
for byte in range(256):
    if byte not in printable:
        printable.append(byte)
        codes.append(256 + extra)
        extra += 1
table = {byte: chr(code) for byte, code in zip(printable, codes)}
vocab = {table[byte]: byte for byte in range(256)}


def special(identifier, content):
    return {"id": identifier, "content": content, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": False, "special": True}


tokenizer = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [special(256, "<|im_end|>"), special(257, "<|im_start|>")],
    "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
              "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
              "ignore_merges": False, "vocab": vocab, "merges": []},
}
(out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=1) + "\n")
(out / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>"}\n')
(out / "generation_config.json").write_text('{"eos_token_id": 256}\n')
# The fixture template opens a think block only when the request sets enable_thinking to true.
# The pinned template of the target model also opens one by default.
(out / "chat_template.jinja").write_text(
    "{% for m in messages %}<|im_start|>{{ m.role }}{{ '\\n' }}{{ m.content }}<|im_end|>{{ '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant{{ '\\n' }}"
    "{% if enable_thinking is defined and enable_thinking is true %}<think>{{ '\\n' }}{% endif %}"
    "{% endif %}\n")

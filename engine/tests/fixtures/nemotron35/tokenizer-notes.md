# Nemotron 3.5 tokenizer fixture notes

The real bundle pins `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and `generation_config.json` to revision `8a115b9d` of `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.
The source files are in `/media/dan/nvme/fifi/nem35-model/NVFP4` and are not copied into this fixture directory.

The portable test builds a small BPE tokenizer in Rust with 16 vocabulary entries, including `<|im_start|>` and `<|im_end|>`.
Its fixed fixture hashes exercise the same pinned `TokenizerBundle::load` path as the Nemotron constructor.
The minimal template loops over message content and checks rendering, encoding, and decoding without model artifacts.

The real `generation_config.json` is present and readable. It declares EOS ids `[2, 11]`.
`tokenizer_config.json` declares `<|im_end|>` as the EOS token, which resolves to id `11`.

# gguf-header

Real-header check for the Flash-Next IQ3_XXS GGUF shards (item 147, slice G1).

The command reads only header prefixes through `GgufHeader::parse`. It never reads tensor payloads.
It needs the true full file length of each shard, because the prefixes are shorter than the files.

Environment variables:

- `GGUF_SHARD1`, `GGUF_SHARD2`: header prefix files.
- `GGUF_LEN1`, `GGUF_LEN2`: full shard lengths in bytes.
- `GGUF_EVIDENCE`: the recorded `iq3_xxs-tensor-bytes.json`.
- `GGUF_REPORT`: optional output file for the JSON report. The default is standard output.
- `GGUF_HEADER_LIMIT`: optional header byte limit. The default is 64 MiB.

Run `cargo run --release` here. The command prints a JSON report and exits 1 on any mismatch.
It checks tensor counts, split metadata, the one n-gram tensor, per-class and per-type byte totals, and the absence of lookup tables.
The report lists every numeric metadata array. A numeric array outside the known list fails the check.

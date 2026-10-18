# CUDA-RD stage 1 EAGER PRE-RUN, not a stage gate 2026-10-11

Identities (from the raw files; all 12 files agree):

- capture_requested: eager
- debug: False
- driver: 580.159.03
- engine_commit: 44ebc72d08a541830c3884e735ab5137987ef5ba
- exe_sha256: dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- gguf_sha256: 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7
- gpu: NVIDIA GeForce RTX 3090
- kv_mode: fp16
- mem_free_after_load: 3721265152
- mem_free_before: 24441126912
- mem_total: 25292898304
- opt_level: 3
- pid: 958893
- profile: release
- started_unix_ns: 1791698361538109199
- a5-gated-steps.json sha256 18d329a693be4983c73caddba470e5643943b0f014b68aa43670669bf726d49a (the tracked copy, checked equal to the frozen 18d329a6... before evaluation)
- reference inputs verified against the hashes frozen in the gated file: 15 committed, 11 frozen but not committed (references of other builds)
- sha256 a5-gated-steps.json: 18d329a693be4983c73caddba470e5643943b0f014b68aa43670669bf726d49a
- sha256 a5_eval.py: d750c88d6f960d0ab24f4ed7defbb4f20804badbfd6c4a794073fa1f7da10374
- sha256 a5_gated_steps.py: 8cae5dcecef97bb51fa98074c9da93eedb1ab9cbcb9cb2cbc35e09db707d63e1
- sha256 a5_stats.py: 154638dc8acc85bdd6b78d2a9528a92c3677e2a53e6712171d724158c1d5fa88
- sha256 assemble_report.py: 352b9daf48824f7b7a99f7375661c71d9249b364d3e6fd395415b31934141471
- sha256 binary cuda-rd-run: dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- sha256 check_deps.py: 0fe0e99833bbf6245ac36d17168ff47901fd06e32d8d75c426cf4cdc2816bb37
- sha256 check_gguf.py: 3e4116ecf10dc182ae1e9605c43ad0e4114dfd5a2ac414b21be8f66b856f758e
- sha256 check_release_build.py: 5483c9da90de579f516d7d568db7bd065eebcc28292e8b071168f3a86139dce4
- sha256 clockcheck.py: 3f2f3a87c3814eec938763deaa7e5a7baf61d494a54b08edaa9f6a96b5cfcccf
- sha256 data A1.json: 270c2cc8eea0f8464748a9d3335eb2180b5c1840a44338c7c53b418c2a2fe1c8
- sha256 data A1.watch.jsonl: 3a7c999800e6bd18a95fcf795609e9e431ec807c9033af1ec9c9fde152d152aa
- sha256 data A2.json: 8566e95228160a305039957f35079410fb25ea3698a993e2f3281c237608715b
- sha256 data A2.watch.jsonl: 3e11c32ec0fccae5dbb7d7c5f69abb5cd0b49fd9670dd477ef0226ab9ebda0d2
- sha256 data B1.json: a37f3ecd0968d9126a7fa38320d5e7158dfe1c25e308d3b4bbd177e0be53159c
- sha256 data B1.watch.jsonl: 96a464dc10492b8a69827e6e96b9c1ed28c98838bb4141db466b5e7219be1c10
- sha256 data B2.json: 73856821768fcea43980da13568f0c72f97e2fcdaee2e31888fe816d1332fb09
- sha256 data B2.watch.jsonl: 3c573c757567513a0368aaddfe0769c9cb00189e97de0eaf5258df259425a361
- sha256 data B3.json: 614e6205b8dc215b55624b00bd7116a707e6706479d13603aaee99f6689c1b05
- sha256 data B3.watch.jsonl: 76d2b9178209ba2f41629c818be92842aa82010894caf8e11409c03cda5c9c3c
- sha256 data B4.json: c0872dccb85db88aa680f8edb85cca908cf45b7282df5b17e0b91e69ec6ce4da
- sha256 data B4.watch.jsonl: dc3525932ef155840b7cca58b67c32ab675e6c8625c81c6c971b9f18f8a440c4
- sha256 data B5.json: 95827167eb78adf1ec0470a159dc769eb5e78f8cd3ddb1f5522c261d4c50c876
- sha256 data B5.watch.jsonl: c22d5028463e85dc9d12034585d3fb33e76276fec7e454648ca84e9be1fca46c
- sha256 data B6.json: d56a98a52a74d0d901af7379c906a3720f82283e58e0731110785d7cdcaa4421
- sha256 data B6.watch.jsonl: 7745d343bdfad86c99e2848223f7a8d2f3a4ccb45fef00c2ac41c81371c2b3d9
- sha256 data C1.json: d02251a54afae57746b786bf971cfd6eb95ff5f8307a93292799bf0173f41075
- sha256 data C1.watch.jsonl: 80740786d8d5277e62304fc991ee713b9815297a94c53530df261a1602145b4c
- sha256 data C2.json: 26921ac5449ab0e328ea00ff0e66638f28dbf35ad4389651806e9670ef48d30e
- sha256 data C2.watch.jsonl: 515f740ebf47df015cd64f3c66ca148cd9b14e11668d4bc28f0778bd73be2cde
- sha256 data C3.json: 68a18a5c04efd9c9a5463682c1256f578555432d7ea0271a69e682aaf350a9e1
- sha256 data C3.watch.jsonl: f3599f305371aa8013d790c9df45fddef7774159f37155d0feaf760a2c8f4b29
- sha256 data C4.json: 902949570c8334cfd89eb743f04dc4c8b5f3d6e273246f0eab5f892eb8179d30
- sha256 data C4.watch.jsonl: 9ebd34909b30858306c8fdeeb5b0ff348b37c697d70efe559d34fff5a041af5e
- sha256 data C5.json: aa8faec30a9c1612871aaa7efa1d43bd4495f0fbb7eb46cdce0c3dfb5225c60f
- sha256 data C5.watch.jsonl: 9bf80b9331e6de13a91b2af5e618fde7b634187f80fb6c93bf5ff67cf0dfebd0
- sha256 data E1.json: 2675627a25d3ee53a6487b4207943212e952223348847e90cb6a16e8a862839a
- sha256 data E1.watch.jsonl: 7cde99ad3c266717e8584ed0112b400b7ad0cd031f3ae0db9e2fda75a040e109
- sha256 data E2.json: e09059fbc456f272918c44a7035899bcb787bc303ffc81449c30fe162baee687
- sha256 data E2.watch.jsonl: 851df060fbc043b85364e507c6277de09ccf14158df0b9499806dde2699df4df
- sha256 data wrapper-log: 93c4fad73db361e72efd77913e9a47ce9360976d8405486f8c5778095de365f5
- sha256 find_test_exes.py: 4c8b8bf3f92dec49284d3bab9188fcdff738d3198593140b655ee8011ca0ce2b
- sha256 fixtures.py: 72996526ef4ad1337f4470b87bd0f285ae1760bd82d6c52151a9bac79ef476d8
- sha256 fixtures_files.py: 517e00348736226346ecd978d289a4bf83d167aa4a9f3c7b5c265ca7e501f1e9
- sha256 gated set a5-gated-steps.json: 18d329a693be4983c73caddba470e5643943b0f014b68aa43670669bf726d49a
- sha256 hold.sh: 4740d3ea8658925343d022f9a515fea76d5061e0fa79b514770dab37c6d38469
- sha256 idle_check.py: e66255d22506f23f20f059998d19344ba82893ac45113cff634906cd36c4ee6e
- sha256 input cpu/clean/P1/run.json: ebeea96eda5ac684164802eebce8295b306cd031d8ab5067cad4f38744b6ce82
- sha256 input cpu/clean/P2/run.json: a2613f61748665bc80248678c1ef93e2a384971d072520cbb0aafd6ba9384fec
- sha256 input cpu/clean/P3/run.json: 737c070030f9b0efc89782711630941cd083a1500a2b3052656d885a3bf871f0
- sha256 input cpu/clean/P4/run.json: f3973c8bc10a203a02c1e693e78edddba4e4b1776722790151c6910756b9a990
- sha256 input cpu/clean/P5/run.json: b459693b507361a4c7c1debdf61d1da44ab59ca40bf0684176d86e19b8f8f07a
- sha256 input prompts/P1.ids: 845538446910825551943eecb1ce7f92fc6339cd00e40f618f806d0a77fe0940
- sha256 input prompts/P1.ids.json: a9b7d55565855fdf48f44e9fec3ef5a997e77616e8c6601f91855258c5c88ff9
- sha256 input prompts/P2.ids: 142861a1b653d474b12294072e06003b7c6b752c140e9c90d9dde4d22601a8f4
- sha256 input prompts/P2.ids.json: 01a22600c931765e70063054907173f574e2bfe30864723775903287e0366198
- sha256 input prompts/P3.ids: cb12a6e3d4bae67f709e10d599d880a015ee9eed8efd9f1a3a196a9a0cba60b1
- sha256 input prompts/P3.ids.json: 556f0d9c70ff18b14d535cd3d3c479e02c84a5d9f413e0a0760ddbfa8b909564
- sha256 input prompts/P4.ids: 4eb070f7a94c4c8b5d33194535260eb17dc6ebcdbe1a6ab4cf28398f59e93a2f
- sha256 input prompts/P4.ids.json: d80e3084d63bf1f0955687d73887259d1c371458f9db5e4797c36d2d4fff73ef
- sha256 input prompts/P5.ids: 92d74ce2b62f52445e49084cdac3899907f48226c3a74801fd915029c04a17f6
- sha256 input prompts/P5.ids.json: 8589556ee1e4dff410f4a3d16b0a266be3324fa9b8603714bf62cdec1be8f7f2
- sha256 lockcheck.py: 210d4356f3374d15143d00ac77fd0c289e40c1c8644cb875169040b588cd2ab0
- sha256 public_items.py: 2b587bc08a5cb75e354cf4ba661096f8436ac7f55d69f834eb2557a2d4948ae6
- sha256 rawcheck.py: dc9fa50f4a889702cb17fcc154d48bac2a76f9f5f3c05ff4188dce9e0d489670
- sha256 rd2e_shapes.json: 1fcc440e4e06bbfc33232db19b7114cd349a6dac23df22db31e0c514ee009d63
- sha256 release-check json: 5ac777d25d2224a146a18ace2c041bf4937e96abe8e609eafaf66a0e77e9b4a3
- sha256 relprobe.cu: 438fd66917ead4e4cd56f252f8a6708a1705a2133730fd203aa625e308493384
- sha256 report_case.py: cb9da9877025466e4393f565581afc99da99624dcbc435d4a50f8de537190ccf
- sha256 report_text.py: 8a4d29f2175e02c9e05f7ec7fdb2594bba63406dfc7a9cdccc39ea77f8541d8a
- sha256 rstokens.py: 30ff9c922fc6205c5492f675025faedc152f83a1804450e539283cdc1d136e22
- sha256 run_selftests.py: bd1242e523f9ae8022b2c43934eee44704718356cbfad121bca15618b4750563
- sha256 run_stage1.sh: 66eac93e14c32fdef7fd10da809096b32746cfed0e3776f0ddc31388957d51f7
- sha256 selftest_attacks.py: 63ca9cac178232ce144069ba3a1ed4c86862601ace89b95e01f636a063f0f7a4
- sha256 selftest_clock.py: 27b444fcd27a2583e325d228debd99323a9f818b89fab57c7de8afefb3dd8c4b
- sha256 selftest_deps.py: 186943f4409aaf77cb991fe189cdba1344c64256a5ded2887f765ff2d723ae20
- sha256 selftest_lock.py: 1941e18e4beecab54befa95bba3448ca76cc24698397a1ff8aca02b24e62b333
- sha256 selftest_loop.py: f7f57f210aa053d53f418dc25a19951111ecc3b78ebc3824716638a0115c52b7
- sha256 selftest_public_items.py: 3aa1b6106c120ddbe5952fc26d6e7042f192965ebfc5dc1646d635eacda3801a
- sha256 selftest_raw.py: 5c7c0533ef731a275bdad00a920f4b1ef3b3a661a46ed16171b7f31718cd88cf
- sha256 selftest_release_check.sh: 7065406fb4543e31aa943b37722e55fc44df928003296c22e0acad8e0b6a038a
- sha256 selftest_render.py: 57618b575ccfc9464c483cd8efb2197d86f010a23989cc5d65eb5f797e0fb5b2
- sha256 selftest_report.py: 1985c37da6ba01c3d23c0811d51f84adce6f94e49cd94b032ceda87ca2add068
- sha256 selftest_rule.py: e8d26bff39519dcf474190661e02f487a5415e27ee1692d71658090e806ac011
- sha256 selftest_tokens.py: f04f39e791cfe598769040f869017091ad3e8f1d479e637b2e921c4abd50df1c
- sha256 selftest_tools.py: 0d2ebd80914abff5fd1980e528494f4b4c152a526a803f717af920ab6a68528e
- sha256 selftest_wrapper.py: a0b9f4ad180f716336928a302756d8c5426f65d02f443f4013a4dae963af1637
- sha256 wrapperlog.py: f906e8436e1f2aa7ab341d72085b38d11ed28fbfe5e5e07a890b9f2dc780279c
- capture outcome of every run (RD-9d `CaptureOutcome`): Eager (capture_requested = eager and captured = false on every run). This record reports the first token, the A5 parity and the eager tok/s as INFORMATION; it can never satisfy stage 1, and the captured gate run follows as a separate later row once the slim RD-11a has merged.
- exit status of this evaluator: 1
- timing rules (REPLY-72 Q8, Q9; lane C): the graph is instantiated and the prompt pass has synchronized before the timed region; the card is sampled every 0.5 s DURING the timed region only; any sample below 1395 MHz, or a busy (>= 50 percent) median below 1600 MHz (judged only with at least 10 busy samples), would have voided the run (a voided run is repeated, never counted); the first timed step is reported beside the median step and flagged when over 2x, never dropped

## Table 1: exactness and speed (EAGER PRE-RUN, not a stage gate)

| prompt | prompt tokens | gen | gated | A5 fails | excluded (share, information only, REPLY-41/REPLY-69) | max logit dev | max abs lse diff vs CPU | first divergence vs llama.cpp CPU (free run, info) | margin there | tok/s (P1 median of 5, others of 3) | top-5 set agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 1024 | 128 | 122 | 8 | 6/128 (4.69 percent) | 1.0227 | 0.9091 | 41 | 0.0198 | 27.04 | 99/128 |
| P2 | 133 | 128 | 122 | 15 | 6/128 (4.69 percent) | 0.7165 | 0.7165 | 57 | 0.1128 | 27.13 | 102/128 |
| P3 | 532 | 128 | 116 | 7 | 12/128 (9.38 percent) | 1.2183 | 1.2183 | none | - | 27.13 | 97/128 |
| P4 | 69 | 128 | 125 | 18 | 3/128 (2.34 percent) | 3.8374 | 3.8374 | none | - | 27.14 | 105/128 |
| P5 | 612 | 128 | 122 | 9 | 6/128 (4.69 percent) | 1.1241 | 1.1239 | none | - | 26.94 | 104/128 |

Timed region: the 128 decode steps (device step, token read-back, copy of the 993 KB row into a preallocated host buffer); host top-32, hash and log-sum-exp run after the clock stops. Decode steps only, prompt tokens excluded.
ms/token breakdown (P1, median run): replay 36.71 + readback 0.27 + host 0.00 = 36.99 against 1000/tok/s = 36.99 (error 0.0 percent, limit 3). Host analysis outside the clock: 2.20 ms/token. Host-blocking syncs per token: 1 and host writes per token: 1 (the start-block write) on every P1 run (both self-reported by the step counters, lane C Q-C6). Run spread (max-min)/median: 0.5 percent. Largest phase: replay.

First timed step (P1 median run): 36.87 ms beside the median step 37.05 ms; runs whose first step is over 2x the median: none (included in the timing, never dropped).

Clocks during the timed region (total and busy sample counts, minimum and median of the busy samples per run; floor 1395 MHz, busy median at least 1600 MHz):

| call | prompt | samples | busy samples | min MHz | busy median MHz | median rule | voided |
|---|---|---|---|---|---|---|---|
| A1 | P1 | 9 | 9 | 1950 | 1950 | not evaluated (9 busy samples, fewer than 10) | no |
| A1 | P3 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| A2 | P2 | 9 | 9 | 1695 | 1695 | not evaluated (9 busy samples, fewer than 10) | no |
| A2 | P4 | 9 | 9 | 1695 | 1695 | not evaluated (9 busy samples, fewer than 10) | no |
| A2 | P5 | 8 | 8 | 1935 | 1935 | not evaluated (8 busy samples, fewer than 10) | no |
| B1 | P2 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| B1 | P3 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| B1 | P4 | 8 | 8 | 1920 | 1920 | not evaluated (8 busy samples, fewer than 10) | no |
| B2 | P2 | 9 | 9 | 1695 | 1695 | not evaluated (9 busy samples, fewer than 10) | no |
| B2 | P3 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| B2 | P4 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| B3 | P2 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| B3 | P3 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| B3 | P4 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| B4 | P5 | 9 | 9 | 1695 | 1695 | not evaluated (9 busy samples, fewer than 10) | no |
| B5 | P5 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| B6 | P5 | 8 | 8 | 1920 | 1920 | not evaluated (8 busy samples, fewer than 10) | no |
| C1 | P1 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| C2 | P1 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| C3 | P1 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| C4 | P1 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| C5 | P1 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| E1 | P1 | 8 | 8 | 1920 | 1920 | not evaluated (8 busy samples, fewer than 10) | no |
| E1 | P3 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |
| E2 | P2 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| E2 | P4 | 9 | 9 | 1935 | 1935 | not evaluated (9 busy samples, fewer than 10) | no |
| E2 | P5 | 9 | 9 | 1920 | 1920 | not evaluated (9 busy samples, fewer than 10) | no |

## Table 2: determinism and capture (EAGER PRE-RUN, not a stage gate)

| check | runs | result |
|---|---|---|
| repeated runs identical (token ids and row hashes), P1 | 5 | pass |
| repeated runs identical (token ids and row hashes), P2 | 3 | pass |
| repeated runs identical (token ids and row hashes), P3 | 3 | pass |
| repeated runs identical (token ids and row hashes), P4 | 3 | pass |
| repeated runs identical (token ids and row hashes), P5 | 3 | pass |
| forced and free row hashes equal up to the first divergence | 5 prompts | pass |
| captured step equals uncaptured launch (row hash, all 128 rows) (trivial in a pre-run) | 5 prompts | pass |
| every graph-mode run was really captured and replayed | all gate runs | n/a (pre-run) |

## Table 3: gate (EAGER PRE-RUN, not a stage gate)

| stage | gate tok/s | measured (median of 5, P1) | exactness (A5, every gated step) | verdict |
|---|---|---|---|---|
| 1 | 18 | 27.04 | FAIL | EAGER PRE-RUN, not a stage gate: tables FAIL (information only, never a stage 1 result) |

Misses: A5 P1: 8 gated steps fail; A5 P2: 15 gated steps fail; A5 P3: 7 gated steps fail; A5 P4: 18 gated steps fail; A5 P5: 9 gated steps fail; wrapper test (RD-2e): no passing line for expert_gate_up 512x2048 (needs `(repeats=1000)`); wrapper test (RD-2e): no passing line for expert_down 2048x512 (needs `(repeats=1000)`)

## Hold records (appended by script from the call logs)

- call-A1.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.14", "pstate": "P8", "t": 1791698361.4592018, "temperature.gpu": "32", "utilization.gpu": "0", "why_not": []}
- call-A1.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-A1.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "149.93", "pstate": "P0", "t": 1791698455.543215, "temperature.gpu": "59", "utilization.gpu": "13"}
- call-A1.log: HOLD seconds 94
- call-A2.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.67", "pstate": "P8", "t": 1791698470.485968, "temperature.gpu": "52", "utilization.gpu": "0", "why_not": []}
- call-A2.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-A2.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "157.85", "pstate": "P0", "t": 1791698563.9517264, "temperature.gpu": "63", "utilization.gpu": "99"}
- call-A2.log: HOLD seconds 93
- call-B1.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.81", "pstate": "P8", "t": 1791699043.8368382, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-B1.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B1.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "154.72", "pstate": "P2", "t": 1791699102.0674462, "temperature.gpu": "63", "utilization.gpu": "0"}
- call-B1.log: HOLD seconds 59
- call-B1.log: HOLD pre {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "154.78", "pstate": "P2", "t": 1791699102.1983626, "temperature.gpu": "63", "utilization.gpu": "0", "why_not": []}
- call-B1.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B1.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "158.77", "pstate": "P0", "t": 1791699160.0405595, "temperature.gpu": "64", "utilization.gpu": "99"}
- call-B1.log: HOLD seconds 58
- call-B2.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.94", "pstate": "P8", "t": 1791699185.2250836, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-B2.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B2.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "154.22", "pstate": "P2", "t": 1791699243.3001769, "temperature.gpu": "63", "utilization.gpu": "6"}
- call-B2.log: HOLD seconds 58
- call-B3.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.85", "pstate": "P8", "t": 1791699268.4835174, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-B3.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B3.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "154.56", "pstate": "P0", "t": 1791699326.2977333, "temperature.gpu": "63", "utilization.gpu": "99"}
- call-B3.log: HOLD seconds 58
- call-B4.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.01", "pstate": "P8", "t": 1791699351.4756699, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-B4.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B4.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1695", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "153.31", "pstate": "P2", "t": 1791699394.8979785, "temperature.gpu": "62", "utilization.gpu": "93"}
- call-B4.log: HOLD seconds 43
- call-B5.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.14", "pstate": "P8", "t": 1791699420.077832, "temperature.gpu": "52", "utilization.gpu": "0", "why_not": []}
- call-B5.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B5.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "153.63", "pstate": "P0", "t": 1791699463.920614, "temperature.gpu": "62", "utilization.gpu": "99"}
- call-B5.log: HOLD seconds 43
- call-B6.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.25", "pstate": "P8", "t": 1791699489.0894377, "temperature.gpu": "52", "utilization.gpu": "0", "why_not": []}
- call-B6.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-B6.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "153.33", "pstate": "P2", "t": 1791699532.443393, "temperature.gpu": "63", "utilization.gpu": "99"}
- call-B6.log: HOLD seconds 43
- call-C1.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.60", "pstate": "P8", "t": 1791698597.0558527, "temperature.gpu": "49", "utilization.gpu": "0", "why_not": []}
- call-C1.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-C1.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "165.13", "pstate": "P0", "t": 1791698675.9534056, "temperature.gpu": "63", "utilization.gpu": "6"}
- call-C1.log: HOLD seconds 79
- call-C2.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.73", "pstate": "P8", "t": 1791698688.3991296, "temperature.gpu": "55", "utilization.gpu": "0", "why_not": []}
- call-C2.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-C2.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "171.01", "pstate": "P2", "t": 1791698754.7351344, "temperature.gpu": "64", "utilization.gpu": "12"}
- call-C2.log: HOLD seconds 66
- call-C3.log: HOLD pre {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "156.96", "pstate": "P2", "t": 1791698754.8923113, "temperature.gpu": "64", "utilization.gpu": "12", "why_not": []}
- call-C3.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-C3.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "159.80", "pstate": "P2", "t": 1791698812.6191785, "temperature.gpu": "64", "utilization.gpu": "96"}
- call-C3.log: HOLD seconds 58
- call-C4.log: HOLD pre {"apps": [], "clocks.mem": "9501", "clocks.sm": "1935", "driver_version": "580.159.03", "idle": false, "memory.used": "531", "power.draw": "155.81", "pstate": "P2", "t": 1791698812.7765477, "temperature.gpu": "64", "utilization.gpu": "96", "why_not": ["utilization above 15"]}
- call-C4.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.01", "pstate": "P8", "t": 1791698874.0654936, "temperature.gpu": "46", "utilization.gpu": "0", "why_not": []}
- call-C4.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-C4.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "153.17", "pstate": "P0", "t": 1791698935.1765826, "temperature.gpu": "62", "utilization.gpu": "0"}
- call-C4.log: HOLD seconds 61
- call-C5.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.97", "pstate": "P8", "t": 1791698960.3536136, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-C5.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-C5.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "159.06", "pstate": "P2", "t": 1791699018.6604114, "temperature.gpu": "64", "utilization.gpu": "0"}
- call-C5.log: HOLD seconds 58
- call-E1.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "33.66", "pstate": "P8", "t": 1791699594.4367344, "temperature.gpu": "49", "utilization.gpu": "0", "why_not": []}
- call-E1.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-E1.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "155.50", "pstate": "P2", "t": 1791699677.6755013, "temperature.gpu": "64", "utilization.gpu": "0"}
- call-E1.log: HOLD seconds 83
- call-E2.log: HOLD pre {"apps": [], "clocks.mem": "405", "clocks.sm": "210", "driver_version": "580.159.03", "idle": true, "memory.used": "531", "power.draw": "34.03", "pstate": "P8", "t": 1791699702.8563738, "temperature.gpu": "51", "utilization.gpu": "0", "why_not": []}
- call-E2.log: HOLD exe_sha256 dd7f8a5d22c13edb83a6cc1f6776faca381d51fbc3787d391960bf91459c0614
- call-E2.log: HOLD post {"apps": [], "clocks.mem": "9501", "clocks.sm": "1920", "driver_version": "580.159.03", "memory.used": "531", "power.draw": "167.37", "pstate": "P2", "t": 1791699763.974764, "temperature.gpu": "64", "utilization.gpu": "99"}
- call-E2.log: HOLD seconds 61

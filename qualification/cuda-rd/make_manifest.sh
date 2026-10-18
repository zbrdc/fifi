#!/usr/bin/env bash
# Writes MANIFEST-SCRIPTS.txt: the sha256 of every file that decides or produces the verdict. Files that do not exist yet are skipped,
# so each commit of RD-10a regenerates the manifest of what it has; the last one (sub-packet 16) lists all of them. Run after the last edit.
set -eu
cd "$(dirname "$0")"
: > MANIFEST-SCRIPTS.txt
for f in a5_eval.py rawcheck.py clockcheck.py wrapperlog.py report_text.py assemble_report.py check_release_build.py check_deps.py idle_check.py hold.sh \
         run_stage1.sh rstokens.py public_items.py find_test_exes.py check_gguf.py fixtures.py fixtures_files.py report_case.py lockcheck.py \
         run_selftests.py selftest_rule.py selftest_raw.py selftest_clock.py selftest_wrapper.py selftest_render.py selftest_report.py selftest_attacks.py \
         selftest_tools.py selftest_lock.py selftest_loop.py selftest_tokens.py selftest_deps.py selftest_public_items.py selftest_release_check.sh relprobe.cu \
         rd2e_shapes.json a5-gated-steps.json a5_stats.py a5_gated_steps.py; do
  if [ -f "$f" ]; then sha256sum "$f" | sed 's/  / /' >> MANIFEST-SCRIPTS.txt; fi
done
wc -l MANIFEST-SCRIPTS.txt

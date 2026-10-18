#!/usr/bin/env bash
# Mutation check of selftest-cuda.sh. Each mutation breaks one refusal of the kubdan CUDA serve
# side or of its launcher mode in a scratch copy of this directory; the self-test must fail on
# every copy. A mutation that the self-test does not kill is a gap in the test.
# Usage: mutate-q36-cuda.sh [scratch dir]    Exit status: 0 when every mutation is killed.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOP=${1:-$(mktemp -d)}
mkdir -p "$TOP"
SURVIVED=0
# mutate <name> <file> <sed program>: copy, patch, check that the patch changed the file, run the
# self-test (the original one) against the copy.
mutate() {
  # MUTATE_ONLY (optional): an extended regular expression; only the mutations whose name matches run.
  if [ -n "${MUTATE_ONLY:-}" ] && ! [[ $1 =~ $MUTATE_ONLY ]]; then return; fi
  local name=$1 file=$2 prog=$3 dir=$TOP/$1/agent-acceptance
  mkdir -p "$TOP/$name"; rm -rf "$dir"; mkdir -p "$dir"; cp -r "$HERE"/. "$dir"/; rm -f "$dir/selftest.sh"  # selftest.sh holds the test key literal
  if [ -L "$dir" ] || [ "$dir/kubdan-acc-a.sh" -ef "$HERE/kubdan-acc-a.sh" ]; then echo "ERROR $name: the copy aliases the original"; exit 2; fi
  sed -e "$prog" "$HERE/$file" > "$dir/$file"
  if cmp -s "$HERE/$file" "$dir/$file"; then echo "ERROR $name: the patch changed nothing"; SURVIVED=$((SURVIVED + 1)); return; fi
  bash "$HERE/selftest-cuda.sh" "$dir" "$TOP/$name/run" > "$TOP/$name/out.log" 2>&1
  local status=$?
  if [ "$status" = 0 ]; then
    echo "SURVIVED $name"; SURVIVED=$((SURVIVED + 1))
  elif ! grep -q '^cuda selftest: .* failed' "$TOP/$name/out.log"; then
    echo "ERROR $name: the self-test did not run to its summary"; SURVIVED=$((SURVIVED + 1))
  else
    echo "killed   $name ($(grep -c '^FAIL' "$TOP/$name/out.log") failed checks)"
  fi
}
mutate no-baton-check run-acc-a.sh 's/^  require_baton$/  true/'
mutate no-ssh-check run-acc-a.sh 's/ || check_kubdan_ssh$/ || true/'
mutate no-18791-refusal driver-lib.sh '/port 18791 is the Mac port/d'
mutate no-key-argv-refusal kubdan-acc-a.sh 's/ || \[ \$# -gt 3 \]//'
mutate no-key-stdin-refusal kubdan-acc-a.sh '/the first line of stdin must be the bearer key/d'
mutate no-env-key-refusal kubdan-acc-a.sh '/ENGINESRV_API_KEY is set in the environment/d'
mutate pins-always-pass kubdan-acc-a.sh 's/^  return \$bad$/  return 0/'
mutate no-gpu-memory-check kubdan-acc-a.sh 's/^if \[ "\$GPU_USED" -ge "\$GPU_FREE_LIMIT_MIB" \]; then/  if false; then/'
mutate no-compute-apps-check kubdan-acc-a.sh 's/^if \[ -n "\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>\/dev\/null)" \]; then/  if false; then/'
mutate no-local-path-check kubdan-acc-a.sh '/is not under the local NVMe root/d'
mutate no-nvcc-check kubdan-acc-a.sh 's/^if \[ -z "\$NVCC" \] || \[ ! -x "\$NVCC" \]; then/  if false; then/'
mutate no-size-pin kubdan-acc-a.sh 's/^  if \[ "\$size" != "\$Q36_CUDA_GGUF_SIZE" \]; then/  if false; then/'
mutate no-port-busy-refusal kubdan-acc-a.sh 's/; exit 70$/; exit 0/'
mutate no-kubdan-18791-refusal kubdan-acc-a.sh '/port 18791 is the Mac port/d'
mutate no-disk-check kubdan-acc-a.sh 's/^  if \[ -z "\$avail" \] || \[ "\$avail" -lt "\$MIN_FREE_MB" \]; then/  if false; then/'
mutate no-prebuilt-root-check kubdan-acc-a.sh '/is not under the prebuilt root/d'
mutate no-prebuilt-commit-check kubdan-acc-a.sh 's/\[ -z "\$WT_COMMIT" \] || \[ "\$WT_COMMIT" != "\$BIN_COMMIT" \]/false/'
mutate builds-despite-prebuilt kubdan-acc-a.sh ':a;N;$!ba;s/if \[ -n "\$ENGINESRV_BIN" \]; then\n  say "using the prebuilt/if false; then\n  say "using the prebuilt/'
mutate no-image-pin kubdan-acc-a.sh 's/\[ "\$ENGINESRV_IMAGE" != "\$PINNED_IMAGE" \]/false/'
mutate no-image-needs-binary kubdan-acc-a.sh 's/^if \[ -n "\$ENGINESRV_IMAGE" \] \&\& \[ -z "\$ENGINESRV_BIN" \]; then/if false; then/'
mutate container-without-host-network kubdan-acc-a.sh 's/ --network host//'
mutate container-without-gpus kubdan-acc-a.sh 's/ --gpus all//'
mutate container-pulls kubdan-acc-a.sh 's/ --pull never//'
mutate container-key-in-argv kubdan-acc-a.sh 's/ -e ENGINESRV_API_KEY -v/ -e ENGINESRV_API_KEY=$ENGINESRV_API_KEY -v/'
mutate container-lifetime-ignored kubdan-acc-a.sh 's/timeout -s KILL \$SERVER_LIFETIME/timeout -s KILL 1740/'
mutate no-wrapper-in-remote-command driver-lib.sh 's/\${KUBDAN_SERVE_WRAP:+\$KUBDAN_SERVE_WRAP }//'
mutate no-binary-in-remote-command driver-lib.sh 's/bin="ENGINESRV_BIN=\$ENGINESRV_BIN "/bin=""/'
mutate capture-as-integer kubdan-acc-a.sh 's/"capture":\$CAPTURE_JSON/"capture":$CAPTURE/'
echo
if [ "$SURVIVED" = 0 ]; then echo "every mutation was killed"; exit 0; fi
echo "$SURVIVED mutation(s) survived"; exit 1

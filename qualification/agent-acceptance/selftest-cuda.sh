#!/usr/bin/env bash
# CUDA part of the self-test (sourced by selftest.sh, which passes its own directory and scratch;
# it also runs alone). It rehearses the kubdan serve script and the q36cuda launcher mode on this
# host with a fake nvidia-smi, a fake cargo, a fake nvcc, and fake_server.py: no kubdan, no Mac,
# no lock, no model, no container. mutate-q36-cuda.sh runs this script against mutated copies of
# the scripts; every refusal below is one mutation.
# Usage: selftest-cuda.sh [agent-acceptance dir under test] [scratch dir]
# Exit status (alone): 0 when every check passes, 1 otherwise. It tests the scripts, not the engine.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AHD=${1:-$HERE}
TOP=${2:-$(mktemp -d)}
if [ -z "${FAILS:-}" ]; then
  FAILS=0
  ok() { echo "ok   $*"; }
  bad() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
  expect_status() {
    if [ "$1" = "$2" ]; then ok "$3 (status $2)"; else bad "$3: want status $1, got $2"; fi
  }
fi
# The default key is derived, not a shared literal, so copies of this directory that other
# mutation scripts make never hold the key that selftest.sh scans the scratch for.
TEST_KEY=${TEST_KEY:-cuda-selftest-$$}
mkdir -p "$TOP"
export DOCKER=${DOCKER:-$AHD/fake_docker.py}
export SANDBOX_IMAGE=${SANDBOX_IMAGE:-fake@sha256:0000000000000000000000000000000000000000000000000000000000000000}
export STOP_TIMEOUT=${STOP_TIMEOUT:-60}
export SESSION_STOP_WAIT=${SESSION_STOP_WAIT:-30}
# The fake servers of this part bind ports above 18800 only.
PORT_C=${SELFTEST_PORT_C:-18801}
PORT_D=${SELFTEST_PORT_D:-18802}

# The scripts find the client files beside themselves and the repository two levels above. That
# layout is rebuilt around the directory under test (selftest.sh does the same for its own dir).
REPO=$TOP/repo
mkdir -p "$REPO/qualification" "$REPO/engine"
ln -sfn "$AHD" "$REPO/qualification/agent-acceptance"
# The launcher is invoked through the symlinked path, so the repository it computes for itself is
# $REPO (selftest.sh does the same for its own dir).
AH=$REPO/qualification/agent-acceptance

for f in kubdan-acc-a.sh run-acc-a.sh driver-lib.sh selftest-cuda.sh mutate-q36-cuda.sh; do
  if bash -n "$AHD/$f"; then ok "cuda syntax $f"; else bad "cuda syntax $f"; fi
done

# --- the fakes: an idle RTX 3090, a cargo that only makes the binary, an nvcc, a dead ssh ------
KB=$TOP/cuda-bin; mkdir -p "$KB"
cat > "$KB/nvidia-smi" <<'SMI'
#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps*)
    if [ -n "${KUBDAN_SMI_APPS:-}" ]; then echo "$KUBDAN_SMI_APPS"; fi
    exit 0;;
  *--query-gpu=*nounits*)
    echo "NVIDIA GeForce RTX 3090, ${KUBDAN_SMI_MEM:-245}, 24576, 41, 1500, 575.51.02"; exit 0;;
  *--query-gpu=*)
    echo "NVIDIA GeForce RTX 3090, ${KUBDAN_SMI_MEM:-245} MiB, 24576 MiB, 41 C, 1500 MHz, 575.51.02"; exit 0;;
esac
echo "kubdan fake nvidia-smi"; exit 0
SMI
cat > "$KB/cargo" <<'CARGO'
#!/usr/bin/env bash
# A fake cargo: the rehearsal build only needs the binary path to exist.
mkdir -p "$CARGO_TARGET_DIR/release"
printf '#!/bin/sh\nsleep 5\n' > "$CARGO_TARGET_DIR/release/enginesrv"
chmod +x "$CARGO_TARGET_DIR/release/enginesrv"
echo "kubdan fake cargo build"
CARGO
cat > "$KB/nvcc" <<'NVCC'
#!/usr/bin/env bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release 12.2, V11.7.99"
echo "Build cuda_12.2.r12.2.compiler.0"
NVCC
chmod +x "$KB/nvidia-smi" "$KB/cargo" "$KB/nvcc"
# A second fake bin without nvidia-smi (so the real one of a host without a driver answers: a 71)
# and a third without nvcc (no CUDA toolkit anywhere: a 73).
KB2=$TOP/cuda-bin-nosmi; KB3=$TOP/cuda-bin-nonvcc; mkdir -p "$KB2" "$KB3"
ln "$KB/cargo" "$KB2/cargo"; ln "$KB/nvcc" "$KB2/nvcc"; chmod +x "$KB2/cargo" "$KB2/nvcc"
ln "$KB/nvidia-smi" "$KB3/nvidia-smi"; ln "$KB/cargo" "$KB3/cargo"; chmod +x "$KB3/nvidia-smi" "$KB3/cargo"
# A fake ssh and flock: any q36cuda run that reaches them in a refusal test leaves a marker.
CFB=$TOP/cuda-fakebin; mkdir -p "$CFB"
for tool in ssh scp flock; do
  printf '#!/bin/sh\ntouch "%s/%s-called"\nexit 1\n' "$TOP" "cuda-$tool" > "$CFB/$tool"; chmod +x "$CFB/$tool"
done

# --- the fixture model dir, its pins, and the baton --------------------------------------------
CK=$TOP/cuda-ck; mkdir -p "$CK"
for f in tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do echo "scratch $f" > "$CK/$f"; done
echo "scratch gguf, padded to a size the rehearsal pins" > "$CK/Qwen3.6-35B-A3B-Q4_K_M.gguf"
head -c 4096 /dev/zero >> "$CK/Qwen3.6-35B-A3B-Q4_K_M.gguf"
(cd "$CK" && sha256sum Qwen3.6-35B-A3B-Q4_K_M.gguf tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json) > "$TOP/cuda-pins.test"
FSIZE=$(wc -c < "$CK/Qwen3.6-35B-A3B-Q4_K_M.gguf" | tr -d ' ')
BATON=$TOP/gpu-baton.txt
printf 'baton git-db granted the 3090 at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BATON"

# The pinned values of the real run (the gate's frozen identity; the four tokenizer hashes that
# engine/src/family/qwen36/settings.rs pins).
PIN_FILE=$AHD/q36-cuda-pins.sha256
for want in 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7:Qwen3.6-35B-A3B-Q4_K_M.gguf \
            87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4:tokenizer.json \
            e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182:tokenizer_config.json \
            e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259:chat_template.jinja \
            e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e:generation_config.json; do
  if grep -qx "${want%%:*}  ${want#*:}" "$PIN_FILE"; then ok "the CUDA pin for ${want#*:} is the gate's hash"; else bad "CUDA pin ${want#*:}"; fi
done
if [ "$(wc -l < "$PIN_FILE")" = 5 ]; then ok "the CUDA pin file has exactly five lines"; else bad "CUDA pin file length"; fi
if grep -q '^Q36_CUDA_PINNED_DIR=/media/dan/nvme/models/qwen3.6-35b-a3b$' "$AHD/kubdan-acc-a.sh" \
   && grep -q '^Q36_CUDA_GGUF_SIZE=20419565568$' "$AHD/kubdan-acc-a.sh"; then ok "the pinned CUDA model dir and GGUF size are constants of the script"; else bad "pinned CUDA constants"; fi

# --- direct serve tests (kubdan-acc-a.sh in rehearsal) ------------------------------------------
# KSERVE <label> <scratch> [env assignments...]: a rehearsal serve against the fixture model.
# The fixed assignments come first, so a per-test assignment (later) overrides one of them.
KSERVE() {
  local label=$1 ms=$2; shift 2
  printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C \
    KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP KUBDAN_MIN_FREE_MB=1 \
    Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
    STOP_TIMEOUT=10 SERVER_CMD=true "$@" bash "$AHD/kubdan-acc-a.sh" serve "$ms" "$REPO" > "$TOP/$label.log" 2>&1
}
KSERVE cuda-serve "$TOP/k-serve"
expect_status 81 $? "a kubdan rehearsal passes the preflight and the pins and reaches the server start (81: SERVER_CMD=true)"
if python3 -I - "$TOP/k-serve/evidence/config.json" "$CK" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
(engine,) = config["engines"]
assert (engine["id"], engine["family"], engine["model"]) == ("q36cuda", "qwen3_6_35b_a3b", "fifi-q36"), engine
assert engine["qwen36"] == {"capture": False, "context_limit": 16384}, engine["qwen36"]
assert engine["qwen36"]["capture"] is False, "capture is a JSON boolean"
assert engine["model_dir"] == engine["tokenizer_dir"] == sys.argv[2]
assert config["api_key_env"] == "ENGINESRV_API_KEY", config
text = open(sys.argv[1]).read()
for key in ("cache_limit", "session_cap", "prefill_", "prefix_reuse"):
    assert key not in text, key
PY
then ok "the CUDA config has capture 0, context_limit 16384, model fifi-q36, and no MLX-only key"; else bad "CUDA config"; fi
KSERVE cuda-capture "$TOP/k-capture" CAPTURE=1 CONTEXT_LIMIT=8192
if python3 -I - "$TOP/k-capture/evidence/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
(engine,) = config["engines"]
assert engine["qwen36"] == {"capture": True, "context_limit": 8192}, engine["qwen36"]
assert engine["qwen36"]["capture"] is True, "capture is a JSON boolean"
PY
then ok "CAPTURE=1 writes the JSON boolean true and CONTEXT_LIMIT=8192"; else bad "CUDA capture-on config"; fi
if grep -q '^engine q36cuda$' "$TOP/k-serve/evidence/identity.txt" && grep -q '^model_id fifi-q36$' "$TOP/k-serve/evidence/identity.txt" \
   && grep -q '^capture 0$' "$TOP/k-serve/evidence/identity.txt" && grep -q '^context_limit 16384$' "$TOP/k-serve/evidence/identity.txt" \
   && [ "$(grep -c '^artifact_file ' "$TOP/k-serve/evidence/identity.txt")" = 5 ] \
   && grep -qx "artifact_size Qwen3.6-35B-A3B-Q4_K_M.gguf $FSIZE" "$TOP/k-serve/evidence/identity.txt" \
   && grep -q '^nvcc ' "$TOP/k-serve/evidence/identity.txt" && grep -q '^cuda_path ' "$TOP/k-serve/evidence/identity.txt" \
   && grep -q '^binary_sha256 ' "$TOP/k-serve/evidence/identity.txt"; then ok "the CUDA identity records the engine, capture, five artifact hashes, the size, nvcc, and the binary"; else bad "CUDA identity"; fi
for f in system-before.txt config.json; do
  if [ -s "$TOP/k-serve/evidence/$f" ]; then ok "kubdan evidence $f (a SERVER_CMD=true run stops at 81 before the samplers)"; else bad "kubdan evidence $f is missing or empty"; fi
done

# --- a prebuilt binary (ENGINESRV_BIN): no build inside a time-limited GPU hold -------------------
# The worktree is a real git repository here, because the script compares its commit with the
# commit file beside the binary. The fake cargo of KB3X leaves a marker and fails, so a run that
# builds anyway cannot pass.
BINREPO=$TOP/binrepo; PRE=$TOP/prebuilt
git init -q "$BINREPO" && git -C "$BINREPO" -c user.name=t -c user.email=t@t commit -q --allow-empty -m rehearsal
BINCOMMIT=$(git -C "$BINREPO" rev-parse HEAD)
mkdir -p "$PRE" "$TOP/other-prebuilt"
printf '#!/bin/sh\nsleep 5\n' > "$PRE/enginesrv"; chmod +x "$PRE/enginesrv"
cp "$PRE/enginesrv" "$TOP/other-prebuilt/enginesrv"
KB3X=$TOP/cuda-bin-nocargo; mkdir -p "$KB3X"
ln "$KB/nvidia-smi" "$KB3X/nvidia-smi"; ln "$KB/nvcc" "$KB3X/nvcc"
printf '#!/bin/sh\ntouch "%s/cargo-called"\nexit 1\n' "$TOP" > "$KB3X/cargo"; chmod +x "$KB3X/cargo"
KBIN() { # <label> <scratch> <binary> [env assignments...]
  local label=$1 ms=$2 bin=$3; shift 3
  printf 'k\n' | env PATH="$KB3X:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C \
    KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP KUBDAN_PREBUILT_ROOT=$PRE KUBDAN_MIN_FREE_MB=1 \
    Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
    STOP_TIMEOUT=10 SERVER_CMD=true ENGINESRV_BIN=$bin "$@" bash "$AHD/kubdan-acc-a.sh" serve "$ms" "$BINREPO" > "$TOP/$label.log" 2>&1
}
KBIN cuda-bin-root "$TOP/k-binroot" "$TOP/other-prebuilt/enginesrv"
expect_status 75 $? "a prebuilt binary outside the prebuilt root stops the run (75)"
KBIN cuda-bin-dots "$TOP/k-bindots" "$PRE/../other-prebuilt/enginesrv"
expect_status 75 $? "a prebuilt binary path with a parent reference stops the run (75)"
KBIN cuda-bin-missing "$TOP/k-binmissing" "$PRE/absent"
expect_status 75 $? "a prebuilt binary that is not an executable file stops the run (75)"
printf '%s\n' "0000000000000000000000000000000000000000" > "$PRE/enginesrv.commit"
KBIN cuda-bin-commit "$TOP/k-bincommit" "$PRE/enginesrv"
expect_status 76 $? "a prebuilt binary built from another commit stops the run (76)"
if [ -e "$TOP/k-bincommit/server.pid" ]; then bad "a commit mismatch started a server"; else ok "a commit mismatch starts no server"; fi
printf '%s\n' "$BINCOMMIT" > "$PRE/enginesrv.commit"
rm -f "$TOP/cargo-called"
KBIN cuda-bin-ok "$TOP/k-binok" "$PRE/enginesrv"
expect_status 81 $? "a prebuilt binary of the right commit reaches the server start (81: SERVER_CMD=true)"
if [ -e "$TOP/cargo-called" ]; then bad "cargo ran although a prebuilt binary was named"; else ok "a prebuilt binary skips the build"; fi
if grep -qx 'binary_prebuilt yes' "$TOP/k-binok/evidence/identity.txt" && grep -qx "binary_commit $BINCOMMIT" "$TOP/k-binok/evidence/identity.txt" \
   && grep -q '^binary_sha256 ' "$TOP/k-binok/evidence/identity.txt"; then ok "the identity records the prebuilt binary, its commit and its hash"; else bad "prebuilt identity"; fi
# A build without a named binary records that it built (the fake cargo of KB makes the binary).
if grep -qx 'binary_prebuilt no' "$TOP/k-serve/evidence/identity.txt"; then ok "a build records binary_prebuilt no"; else bad "binary_prebuilt no"; fi
# The remote command of the launcher: the hold wrapper stands right before bash, after the environment.
# --- the container mode (ENGINESRV_IMAGE): the server runs in the pinned build image ---------------
# A fake docker records its arguments and the key variable, then idles; no /ready follows, so the
# run ends with 83 after the short ready wait, and the recorded command is what the test judges.
KDK=$TOP/cuda-bin-docker; mkdir -p "$KDK"
ln "$KB/nvidia-smi" "$KDK/nvidia-smi"; ln "$KB/nvcc" "$KDK/nvcc"; ln "$KB3X/cargo" "$KDK/cargo"
cat > "$KDK/docker" <<'DOCKER'
#!/usr/bin/env bash
case "$1" in
  image) echo "sha256:rehearsal-image-id"; exit 0;;
  run) printf '%s\n' "$*" > "$DOCKER_ARGS_FILE"; printf '%s\n' "${ENGINESRV_API_KEY:-unset}" > "$DOCKER_KEY_FILE"; exec sleep 20;;
esac
exit 0
DOCKER
chmod +x "$KDK/docker"
KDOCK() { # <label> <scratch> [env assignments...]
  local label=$1 ms=$2; shift 2
  printf 'secret-key-9137\n' | env PATH="$KDK:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C \
    KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP KUBDAN_PREBUILT_ROOT=$PRE KUBDAN_MIN_FREE_MB=1 \
    Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
    STOP_TIMEOUT=10 READY_TIMEOUT=3 SERVER_CMD=true DOCKER_ARGS_FILE=$TOP/$label.args DOCKER_KEY_FILE=$TOP/$label.key \
    "$@" bash "$AHD/kubdan-acc-a.sh" serve "$ms" "$BINREPO" > "$TOP/$label.log" 2>&1
}
KDOCK cuda-img-nobin "$TOP/k-imgnobin" ENGINESRV_IMAGE=inference-cuda-build:cuda12.2-ubuntu22.04
expect_status 75 $? "an image without a prebuilt binary stops the run (75)"
KDOCK cuda-img "$TOP/k-img" ENGINESRV_IMAGE=inference-cuda-build:cuda12.2-ubuntu22.04 ENGINESRV_BIN=$PRE/enginesrv SERVER_LIFETIME=1700
expect_status 83 $? "a container server that never answers /ready ends with 83"
WANTRUN="run --pull never --rm --name acca-k-img --network host --gpus all --cpus 8 --memory 16g --memory-swap 16g -e ENGINESRV_API_KEY -v /media/dan/nvme:/media/dan/nvme inference-cuda-build:cuda12.2-ubuntu22.04 timeout -s KILL 1700 $PRE/enginesrv --config $TOP/k-img/config.json"
if [ "$(cat "$TOP/cuda-img.args" 2>/dev/null)" = "$WANTRUN" ]; then ok "the container command is the pinned form"; else bad "container command: $(cat "$TOP/cuda-img.args" 2>/dev/null)"; fi
if [ "$(cat "$TOP/cuda-img.key" 2>/dev/null)" = secret-key-9137 ] && ! grep -q secret-key-9137 "$TOP/cuda-img.args" "$TOP/k-img/evidence/identity.txt" "$TOP/cuda-img.log"; then ok "the key reaches the docker client by environment only"; else bad "key handling in the container mode"; fi
if grep -qx 'server_image inference-cuda-build:cuda12.2-ubuntu22.04' "$TOP/k-img/evidence/identity.txt" && grep -qx 'server_image_id sha256:rehearsal-image-id' "$TOP/k-img/evidence/identity.txt" \
   && grep -qx 'server_container acca-k-img' "$TOP/k-img/evidence/identity.txt" && grep -qx 'server_lifetime 1700' "$TOP/k-img/evidence/identity.txt"; then ok "the identity records the image, its id, the container and the lifetime"; else bad "container identity"; fi
# Outside a rehearsal only the pinned build image is accepted; the pinned one passes this check
# and stops later, at the NVMe roots, with another text.
printf 'k\n' | env PATH="$KDK:$PATH" ENGINE=q36cuda PORT=$PORT_C ENGINESRV_IMAGE=other:1 \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-realimg" "$REPO" > "$TOP/k-realimg.log" 2>&1
expect_status 75 $? "a real run refuses an image other than the pinned build image (75)"
if grep -q 'is not the pinned build image inference-cuda-build:cuda12.2-ubuntu22.04' "$TOP/k-realimg.log"; then ok "the image refusal names the pinned image"; else bad "image refusal message"; fi
printf 'k\n' | env PATH="$KDK:$PATH" ENGINE=q36cuda PORT=$PORT_C ENGINESRV_IMAGE=inference-cuda-build:cuda12.2-ubuntu22.04 \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-realimg2" "$REPO" > "$TOP/k-realimg2.log" 2>&1
expect_status 75 $? "the pinned image passes the image check and stops at the NVMe root check (75)"
if grep -q 'local NVMe root' "$TOP/k-realimg2.log" && ! grep -q 'pinned build image' "$TOP/k-realimg2.log"; then ok "the pinned image is not refused as an image"; else bad "pinned image acceptance"; fi
KDOCK cuda-life "$TOP/k-life" SERVER_LIFETIME=abc
expect_status 64 $? "a non-numeric SERVER_LIFETIME stops the run (64)"

# The function is read in a subshell, so that the sourced library cannot change this script.
serve_command() { ( . "$AHD/driver-lib.sh"; MW=/w MS=/s PORT=18792 ENGINE=q36cuda CAPTURE=1 CONTEXT_LIMIT=8192 READY_TIMEOUT=1800 STOP_TIMEOUT=900; kubdan_serve_command ); }
WANT="PORT=18792 ENGINE=q36cuda CAPTURE=1 CONTEXT_LIMIT=8192 READY_TIMEOUT=1800 STOP_TIMEOUT=900 ENGINESRV_BIN=/p/enginesrv ENGINESRV_IMAGE=img:1 SERVER_LIFETIME=1700 /h/gpu_run.sh 1800 bash /w/qualification/agent-acceptance/kubdan-acc-a.sh serve /s /w"
GOT=$(KUBDAN_SERVE_WRAP="/h/gpu_run.sh 1800" ENGINESRV_BIN=/p/enginesrv ENGINESRV_IMAGE=img:1 SERVER_LIFETIME=1700 serve_command)
if [ "$GOT" = "$WANT" ]; then ok "the remote serve command carries the wrapper and the binary"; else bad "remote serve command: $GOT"; fi
WANT="PORT=18792 ENGINE=q36cuda CAPTURE=1 CONTEXT_LIMIT=8192 READY_TIMEOUT=1800 STOP_TIMEOUT=900 bash /w/qualification/agent-acceptance/kubdan-acc-a.sh serve /s /w"
GOT=$(unset KUBDAN_SERVE_WRAP ENGINESRV_BIN ENGINESRV_IMAGE SERVER_LIFETIME; serve_command)
if [ "$GOT" = "$WANT" ]; then ok "without wrapper and binary the remote serve command is unchanged"; else bad "plain remote serve command: $GOT"; fi

printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-argv" "$REPO" a-bearer-key > "$TOP/k-argv.log" 2>&1
expect_status 64 $? "kubdan-acc-a refuses a key on the command line"
printf '\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-nokey" "$REPO" > "$TOP/k-nokey.log" 2>&1
expect_status 64 $? "kubdan-acc-a needs a key on stdin"
printf 'k\n' | env ENGINESRV_API_KEY=preset PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-envkey" "$REPO" > "$TOP/k-envkey.log" 2>&1
expect_status 64 $? "kubdan-acc-a refuses a key that is already in the environment"
printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=8788 bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-8788" "$REPO" > "$TOP/k-8788.log" 2>&1
expect_status 64 $? "kubdan-acc-a refuses port 8788"
printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=18791 bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-18791" "$REPO" > "$TOP/k-18791.log" 2>&1
expect_status 64 $? "kubdan-acc-a refuses the Mac port 18791"
printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C CAPTURE=2 bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-capbad" "$REPO" > "$TOP/k-capbad.log" 2>&1
expect_status 64 $? "kubdan-acc-a refuses CAPTURE other than 0 or 1"

# A busy GPU (a fake nvidia-smi earlier in PATH): memory above the limit, then a compute process.
KSERVE cuda-busy "$TOP/k-busy" KUBDAN_SMI_MEM=4096
expect_status 71 $? "a GPU with memory.used above 2,048 MiB stops the run (71)"
KSERVE cuda-apps "$TOP/k-apps" KUBDAN_SMI_APPS=4321
expect_status 71 $? "a GPU with a compute process stops the run (71)"
printf 'k\n' | env PATH="$KB2:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-smi" "$REPO" > "$TOP/k-smi.log" 2>&1
expect_status 71 $? "an nvidia-smi that reports no usable GPU stops the run (71)"
# nvcc missing (a fake bin without it and no CUDA toolkit on this host).
printf 'k\n' | env PATH="$KB3:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP \
  KUBDAN_MIN_FREE_MB=1 Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-nvcc" "$REPO" > "$TOP/k-nvcc.log" 2>&1
expect_status 73 $? "a missing nvcc stops the run (73)"
if grep -q 'set CUDA_PATH to the CUDA 12 toolkit' "$TOP/k-nvcc.log"; then ok "the nvcc refusal names the remedy"; else bad "nvcc refusal message"; fi
# A non-local build path: a worktree outside the NVMe root.
mkdir -p "$TOP.nonlocal-tree"
printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_C KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP \
  KUBDAN_MIN_FREE_MB=1 Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-path" "$TOP.nonlocal-tree" > "$TOP/k-path.log" 2>&1
expect_status 75 $? "a worktree outside the local NVMe root stops the run (75)"
# The free disk of the root.
KSERVE cuda-disk "$TOP/k-disk" KUBDAN_MIN_FREE_MB=999999999
expect_status 74 $? "a root with less free disk than the build needs stops the run (74)"
# A busy port: another listener (not enginesrv) gives 70.
python3 -I -c "import socket, time; s = socket.socket(); s.bind(('127.0.0.1', $PORT_D)); s.listen(); time.sleep(120)" &
KLISTENER=$!
sleep 1
printf 'k\n' | env PATH="$KB:$PATH" REHEARSE=1 ENGINE=q36cuda PORT=$PORT_D KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP \
  KUBDAN_MIN_FREE_MB=1 Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-busyport" "$REPO" > "$TOP/k-busyport.log" 2>&1
expect_status 70 $? "a busy port stops the kubdan serve script (70)"
kill "$KLISTENER" 2>/dev/null; wait "$KLISTENER" 2>/dev/null
# A pin mismatch on a fixture: a tampered tokenizer file and a wrong declared size.
echo "tampered" >> "$CK/tokenizer.json"
KSERVE cuda-pin "$TOP/k-pin"
expect_status 88 $? "a changed CUDA model file stops the run (88)"
if grep -q 'Q36 CUDA PIN MISMATCH: tokenizer.json' "$TOP/cuda-pin.log"; then ok "the mismatch names the changed file"; else bad "CUDA mismatch message"; fi
if [ -e "$TOP/k-pin/server.pid" ]; then bad "a pin mismatch started a server"; else ok "a pin mismatch starts no server"; fi
echo "scratch tokenizer.json" > "$CK/tokenizer.json"
(cd "$CK" && sha256sum Qwen3.6-35B-A3B-Q4_K_M.gguf tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json) > "$TOP/cuda-pins.test"
KSERVE cuda-size "$TOP/k-size" Q36_CUDA_SIZE_OVERRIDE=$((FSIZE + 1))
expect_status 88 $? "a GGUF whose size differs from the pin stops the run (88)"
if grep -q 'Q36 CUDA PIN MISMATCH: Qwen3.6-35B-A3B-Q4_K_M.gguf has size' "$TOP/cuda-size.log"; then ok "the size mismatch names the GGUF"; else bad "size mismatch message"; fi
# Outside a rehearsal the overrides are ignored: the pinned roots and dir are constants.
printf 'k\n' | env PATH="$KB:$PATH" ENGINE=q36cuda PORT=$PORT_C KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP \
  Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
  bash "$AHD/kubdan-acc-a.sh" serve "$TOP/k-real" "$REPO" > "$TOP/k-real.log" 2>&1
expect_status 75 $? "outside a rehearsal the NVMe-root overrides are ignored"
if grep -q 'not under the local NVMe root /media/dan/nvme/fifi/wrktr/runs' "$TOP/k-real.log" \
   && ! grep -q "$CK" "$TOP/k-real.log"; then ok "a real run checks the pinned NVMe roots"; else bad "real-run roots"; fi

# --- the launcher mode --------------------------------------------------------------------------
# Without a baton the run refuses (66) and starts nothing.
env -u GPU_BATON ACC_TEST_KEY=$TEST_KEY PORT=$PORT_C SANDBOX_IMAGE=$SANDBOX_IMAGE DOCKER=$DOCKER \
  bash "$AH/run-acc-a.sh" "$TOP/c-nobaton" --rehearse --engine q36cuda > "$TOP/c-nobaton.log" 2>&1
expect_status 66 $? "the q36cuda launcher refuses without GPU_BATON (66)"
if [ -e "$TOP/c-nobaton" ]; then bad "the run made files without a baton"; else ok "the run makes no files without a baton"; fi
# Without ssh the run refuses (64) before the lock, the build, and any model call.
env GPU_BATON=$BATON PATH="$CFB:$PATH" SANDBOX_IMAGE=$SANDBOX_IMAGE DOCKER=$DOCKER \
  bash "$AH/run-acc-a.sh" "$TOP/c-nossh" --engine q36cuda > "$TOP/c-nossh.log" 2>&1
expect_status 64 $? "the q36cuda launcher refuses when ssh to kubdan does not work (64)"
if [ -e "$TOP/c-nossh" ] || [ -e "$TOP/cuda-flock-called" ]; then bad "the run reached the lock or made files without ssh"; else ok "the ssh refusal comes before the lock and any file"; fi
env GPU_BATON=$BATON ACC_TEST_KEY=$TEST_KEY PORT=8788 SANDBOX_IMAGE=$SANDBOX_IMAGE DOCKER=$DOCKER \
  bash "$AH/run-acc-a.sh" "$TOP/c-8788" --rehearse --engine q36cuda > "$TOP/c-8788.log" 2>&1
expect_status 64 $? "the q36cuda launcher refuses port 8788"
env GPU_BATON=$BATON ACC_TEST_KEY=$TEST_KEY PORT=18791 SANDBOX_IMAGE=$SANDBOX_IMAGE DOCKER=$DOCKER \
  bash "$AH/run-acc-a.sh" "$TOP/c-18791" --rehearse --engine q36cuda > "$TOP/c-18791.log" 2>&1
expect_status 64 $? "the q36cuda launcher refuses the Mac port 18791"

# The whole chain in a rehearsal: launcher -> kubdan-acc-a.sh (fake nvidia-smi, fake cargo) ->
# fake_server.py -> acc_a.py here, with the ps of this host sampled for the key.
run_cuda() { # <label> <out> [env assignments...] (a per-test assignment overrides a fixed one)
  local label=$1 out=$2; shift 2
  env GPU_BATON=$BATON ACC_TEST_KEY=$TEST_KEY PORT=$PORT_C PATH="$KB:$PATH" FAKE_READY_DELAY=1 \
    KUBDAN_SCRCH_ROOT=$TOP KUBDAN_WRKTR_ROOT=$TOP KUBDAN_MIN_FREE_MB=1 \
    Q36_CUDA_DIR_OVERRIDE=$CK Q36_CUDA_PINS_OVERRIDE=$TOP/cuda-pins.test Q36_CUDA_SIZE_OVERRIDE=$FSIZE \
    "$@" bash "$AH/run-acc-a.sh" "$out" --rehearse --engine q36cuda > "$TOP/$label.log" 2>&1
}
run_cuda c-good "$TOP/c-good" &
CUDA_RUN=$!
: > "$TOP/cuda-ps.log"
while kill -0 "$CUDA_RUN" 2>/dev/null; do ps -eo args >> "$TOP/cuda-ps.log"; sleep 1; done
wait "$CUDA_RUN"; expect_status 0 $? "the q36cuda rehearsal passes the whole chain"
if grep -qF -f - "$TOP/cuda-ps.log" <<< "$TEST_KEY"; then bad "the bearer key appears in the argv of a process of the CUDA run"; else ok "the bearer key is in no argv of the CUDA run ($(wc -l < "$TOP/cuda-ps.log") ps lines)"; fi
for f in acc-a/timings.tsv acc-a/transcript.jsonl acc-a/result.json acc-a/models.json evidence/identity.txt \
         evidence/system-before.txt evidence/system-after.txt evidence/rss.tsv evidence/vram.tsv \
         evidence/server.log evidence/summary.txt evidence/config.json gpu-baton.txt launcher.txt serve.out; do
  if [ -s "$TOP/c-good/$f" ]; then ok "cuda evidence $f"; else bad "cuda evidence $f is missing or empty"; fi
done
if cmp -s "$BATON" "$TOP/c-good/gpu-baton.txt"; then ok "the evidence holds a copy of the baton"; else bad "baton copy"; fi
if awk -F', *' 'NR == 1 { exit (NF >= 4 ? 0 : 1) }' "$TOP/c-good/evidence/vram.tsv"; then ok "vram.tsv samples memory, temperature, and clocks"; else bad "vram.tsv"; fi
if grep -q '^vram_peak_mib ' "$TOP/c-good/evidence/summary.txt" && grep -q '^capture 0$' "$TOP/c-good/evidence/summary.txt"; then ok "the kubdan summary records the vram peak and the capture mode"; else bad "kubdan summary"; fi
if grep -q '^engine q36cuda$' "$TOP/c-good/launcher.txt" && grep -q '^model fifi-q36$' "$TOP/c-good/launcher.txt" \
   && grep -q '^capture 0$' "$TOP/c-good/launcher.txt" && grep -q '^request_timeout 1800$' "$TOP/c-good/launcher.txt"; then ok "launcher.txt records the engine, model, capture 0, and the 1,800 s request timeout"; else bad "cuda launcher.txt"; fi
if python3 -I - "$TOP/c-good/acc-a/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["verdict"] == "pass", r
assert (r["engine"], r["model"]) == ("q36cuda", "fifi-q36"), r
assert r["timeout_seconds"] == 1800.0, r
PY
then ok "result.json records pass, engine q36cuda, model fifi-q36, and timeout_seconds 1800"; else bad "cuda result.json"; fi
if [ "$(grep -c '^artifact_file ' "$TOP/c-good/evidence/identity.txt")" = 5 ] && grep -q '^capture 0$' "$TOP/c-good/evidence/identity.txt" \
   && grep -q '^ready_seconds ' "$TOP/c-good/evidence/identity.txt" && grep -q '"ok": true' "$TOP/c-good/acc-a/result.json"; then ok "the fetched kubdan identity and the server-log post-check are in the evidence"; else bad "cuda fetched evidence"; fi
if (cd "$TOP/c-good" && sha256sum -c SHA256SUMS --quiet) && (cd "$TOP/c-good/evidence" && sha256sum -c SHA256SUMS --quiet); then ok "the cuda SHA256SUMS verifies (top and inner)"; else bad "cuda SHA256SUMS"; fi
if grep -rqF -f - "$TOP/c-good" <<< "$TEST_KEY"; then bad "the key is in the cuda evidence"; else ok "the key is not in the cuda evidence"; fi
# CAPTURE=1 is a separate run with its own record; ACC_REQUEST_TIMEOUT overrides the client timeout.
run_cuda c-capture "$TOP/c-capture" CAPTURE=1 ACC_REQUEST_TIMEOUT=123
expect_status 0 $? "the q36cuda rehearsal with CAPTURE=1 passes as a separate run"
if grep -q '^capture 1$' "$TOP/c-capture/launcher.txt" && grep -q '^capture 1$' "$TOP/c-capture/evidence/identity.txt" \
   && grep -q '^request_timeout 123$' "$TOP/c-capture/launcher.txt" \
   && grep -q '"capture":true' "$TOP/c-capture/evidence/config.json" \
   && python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["timeout_seconds"]==123.0' "$TOP/c-capture/acc-a/result.json"; then ok "the capture-on run records capture 1 and its 123 s timeout everywhere"; else bad "capture-on records"; fi
if pgrep -f "fake_server.py --port $PORT_C|fake_server.py --port $PORT_D|kubdan-acc-a.sh serve $TOP" > /dev/null; then bad "cuda: a server or session is still running"; else ok "cuda: no server or session remains"; fi

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "cuda selftest scratch: $TOP"
  if [ "$FAILS" = 0 ]; then echo "cuda selftest: all checks passed"; exit 0; fi
  echo "cuda selftest: $FAILS check(s) failed"; exit 1
fi

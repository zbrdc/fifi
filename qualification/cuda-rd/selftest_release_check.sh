#!/usr/bin/env bash
# CPU only: proves check_release_build.py on REAL executables built by the CUDA 12.2 nvcc of the build image.
# -G -O0 must be refused (exit 2), -O3 -lineinfo and -O3 must be accepted. Usage: selftest_release_check.sh SCRATCH_DIR
set -u
S="$1"; mkdir -p "$S/relprobe"
F="$(cd "$(dirname "$0")" && pwd)"
cp "$F/relprobe.cu" "$S/relprobe/k.cu"
docker run --rm -v /media/dan/nvme:/media/dan/nvme inference-cuda-build:cuda12.2-ubuntu22.04 bash -c "
  umask 000; cd '$S/relprobe' &&
  nvcc -arch=sm_86 -G -O0 k.cu -o exe_G && nvcc -arch=sm_86 -O3 -lineinfo k.cu -o exe_LI && nvcc -arch=sm_86 -O3 k.cu -o exe_O3" >"$S/relprobe/build.log" 2>&1 || { echo "FAIL build"; tail -5 "$S/relprobe/build.log"; exit 1; }
rc=0
python3 -I "$F/check_release_build.py" "$S/relprobe/exe_G" >"$S/relprobe/G.out" 2>&1; g=$?
python3 -I "$F/check_release_build.py" "$S/relprobe/exe_LI" >"$S/relprobe/LI.out" 2>&1; li=$?
python3 -I "$F/check_release_build.py" "$S/relprobe/exe_O3" >"$S/relprobe/O3.out" 2>&1; o3=$?
python3 -I "$F/check_release_build.py" "$S/relprobe/missing" >"$S/relprobe/M.out" 2>&1; mi=$?
python3 -I "$F/check_release_build.py" "$S/relprobe/exe_O3" --out "$S/relprobe/O3.json" >/dev/null 2>&1
python3 -I -c "
import json, sys
sys.path.insert(0, '$F')
import assemble_report as ar
r = json.load(open('$S/relprobe/O3.json'))
p = ar.release_problems(r, r['exe_sha256'])
print('assembler accepts the release JSON of the real -O3 exe (version string end to end):', p == [], repr(r['cuobjdump_version']))
sys.exit(0 if p == [] else 1)" || rc=1
echo "exe -G -O0: exit $g (want 2)   exe -O3 -lineinfo: exit $li (want 0)   exe -O3: exit $o3 (want 0)   missing exe: exit $mi (want 2, no traceback)"
grep -q Traceback "$S/relprobe/M.out" && { echo "FAIL traceback"; rc=1; }
grep -o '"misses": \[[^]]*\]' "$S/relprobe/G.out"
[ "$g" = 2 ] && [ "$li" = 0 ] && [ "$o3" = 0 ] && [ "$mi" = 2 ] && [ "$rc" = 0 ] && echo "RELEASE-CHECK SELFTEST OK" || { echo "RELEASE-CHECK SELFTEST FAILED"; exit 1; }

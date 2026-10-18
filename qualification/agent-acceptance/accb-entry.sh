#!/usr/bin/env bash
# ACC-B container entry (design 4.4, steps 1 to 7). This script is PID 1 of the sandbox: it is
# mounted read-only, its hash is recorded in the evidence, and it runs as uid 65534 with
# --network none (design 4.3). The bearer key arrives as the first stdin line, never as an
# argument or a docker -e option. The exit status is the pi status, 64 for an empty key, 65 when
# the bridge is not up, or 137 when the timeout killed pi. The launcher takes the status from
# Docker only: every line this container prints, including the ACCB-EXPORT line, is data.
set -u
# Step 1: read one stdin line into ENGINESRV_API_KEY and export it. An empty line stops with 64.
IFS= read -r ENGINESRV_API_KEY
if [ -z "$ENGINESRV_API_KEY" ]; then exit 64; fi
export ENGINESRV_API_KEY
# Step 2: the scratch agent directory and the task fixture.
cp -a /in/agent "$HOME/agent"
cp -a /fixture/. /work
# Step 3: start the bridge (127.0.0.1:18791 <-> /ipc/fifi.sock) and wait up to 10 s for the port.
# The bridge probes the relay socket at startup and exits when it cannot connect, so a missing
# forward shows up here as status 65. DISPLAY is not set: pi loads its X11 module only with it.
/opt/node/bin/node /opt/accb/bridge.js &
BRIDGE_PID=$!
status=65
n=0
while [ "$n" -lt 20 ]; do
  if (exec 3<>/dev/tcp/127.0.0.1/18791) 2>/dev/null; then exec 3>&- 3<&-; status=0; break; fi
  kill -0 "$BRIDGE_PID" 2>/dev/null || break
  sleep 0.5
  n=$((n + 1))
done
if [ "$status" != 0 ]; then kill -KILL "$BRIDGE_PID" 2>/dev/null; exit 65; fi
# Step 4: the pi run itself (the task and the system prompt are the constants of the ACC-B run).
# timeout -s KILL enforces PI_TIMEOUT inside the container, whatever the model does (design C9).
SYSTEM_PROMPT="You are a coding agent. Use the tools to do the task. Keep replies short. Work in the current directory."
TASK="Create hello.py that prints the sum of 2 and 3, then run it."
cd /work && timeout -s KILL "$PI_TIMEOUT" /opt/node/bin/node \
  /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js \
  --offline -nc -ne -ns -np -na --no-session --mode json -p --thinking off \
  --model "fifi/flash" --system-prompt "$SYSTEM_PROMPT" --tools write,bash "$TASK" < /dev/null
status=$?
# Step 5: end every other process of the container. This entry is PID 1 only inside the
# container, whose /proc holds nothing but the container's own processes (its own PID
# namespace, uid 65534, --network none), so the sweep ends each of them by pid, never with a
# signal to every process at once (project rule of 2026-10-11). On a host debug run ($$ is not
# 1) nothing is swept: only the bridge this entry started is ended, by its recorded pid.
if [ "$$" = 1 ]; then
  for proc in /proc/[0-9]*; do
    pid=${proc#/proc/}
    [ "$pid" = "$$" ] || kill -KILL "$pid" 2>/dev/null
  done
else
  kill -KILL "$BRIDGE_PID" 2>/dev/null
fi
# Step 6: the export (capped by EXPORT_MAX) on stdout, the peaks on stderr (diagnostic only).
# The container stdout is a fifo that the runtime holds with O_NONBLOCK, so one big printf of
# the multi-MiB export line fails with EAGAIN; the redirection through /proc/self/fd/1 reopens
# the fifo as a blocking descriptor, so the write waits for the launcher reader instead.
b64=$(tar -C /work -cf - . | head -c "$EXPORT_MAX" | base64 -w 0)
printf 'ACCB-EXPORT %s\n' "$b64" > /proc/self/fd/1
printf 'ACCB-PEAK memory.peak=%s pids.peak=%s\n' \
  "$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo '?')" \
  "$(cat /sys/fs/cgroup/pids.peak 2>/dev/null || echo '?')" >&2
# Step 7: the container exit status is the pi status, 64, 65, or 137.
exit "$status"

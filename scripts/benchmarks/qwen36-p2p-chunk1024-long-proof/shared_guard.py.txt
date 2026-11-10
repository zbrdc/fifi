#!/usr/bin/env python3
"""Stop only our GPU validation if another workload enters its test window."""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT=Path('/home/ben/fifi-scratch/e4ecf0370956/publish-prefix-evidence')
PAIR={'GPU-704fef60-5280-9b88-fa3d-d11135caf576', 'GPU-282e266a-24ac-6fa8-a16d-4281e52b4c0a'}
OTHER='GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c'
BINARIES={
 '/home/ben/fifi-scratch/5d97e11c1ca1/p2p-sliced-async-events/target/release/deps/inference_engine-d032985c3e37fde1',
 '/home/ben/fifi-scratch/5d97e11c1ca1/p2p-local-kv-gather/target/release/deps/inference_engine-d032985c3e37fde1',
 '/home/ben/fifi-scratch/c41d98f68b55/p2p-chunk-prefill/target/release/deps/inference_engine-d032985c3e37fde1',
 '/home/ben/fifi-scratch/e4ecf0370956/publish-prefix-evidence/target/release/deps/inference_engine-d032985c3e37fde1',
}
BUILD={'cargo','cargo-clippy','rustc','nvcc','cicc','ptxas','cc1plus'}
OTHER_ASSISTANTS={146543:617113, 664377:3530809}
SHARED_LEASE=Path('/home/ben/fifi-qwen36/gpu.lock')


def interrupted(signum, _frame):
 raise SystemExit(128+signum)


def lease_available(path=SHARED_LEASE):
 """Probe the established shared lease without retaining it or truncating the file."""
 fd=os.open(path,os.O_RDWR|os.O_CLOEXEC)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fcntl.flock(fd,fcntl.LOCK_UN)
  return True
 finally:os.close(fd)


def processes():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                              '--format=csv,noheader,nounits'],text=True,timeout=5)
 return [(r[0].strip(),int(r[1])) for r in csv.reader(io.StringIO(out))]


def builders():
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   name=p.joinpath('comm').read_text().strip()
   if name in BUILD:result.append((int(p.name),name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result


def assistant_children():
 """Observe, but never signal, the other sessions' direct children."""
 live=set()
 for pid,start in OTHER_ASSISTANTS.items():
  owner=Path('/proc')/str(pid)
  try:
   fields=owner.joinpath('stat').read_text().rsplit(')',1)[1].split()
   if int(fields[19])!=start or owner.joinpath('comm').read_text().strip()!='claude':
    raise RuntimeError(f'other assistant {pid} identity changed')
   live.add(pid)
  except FileNotFoundError:
   continue
 result=set()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
   owner=int(fields[1])
   if owner not in live:continue
   name=p.joinpath('comm').read_text().strip()
   args=p.joinpath('cmdline').read_bytes().split(b'\0')
   service=(name=='uv' or name.startswith('npm exec') or name=='pyright-langser'
            or (name=='python3' and any(a.endswith(b'/searxng-mcp.py') for a in args)))
   result.add((owner,int(p.name),int(fields[19]),name,service))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 return result


def assistant_activity(initial,current):
 return sorted((owner,pid,name) for owner,pid,start,name,service in current
               if not service or (owner,pid,start,name,service) not in initial)


def self_test():
 initial={(100,1,10,'uv',True),(200,2,20,'python3',True)}
 assert not assistant_activity(initial,initial)
 assert assistant_activity(initial,initial|{(100,3,30,'bash',False)})==[(100,3,'bash')]
 assert assistant_activity(initial,initial|{(200,4,40,'uv',True)})==[(200,4,'uv')]
 assert assistant_activity(initial,{(100,1,11,'uv',True),(200,2,20,'python3',True)})==[(100,1,'uv')]
 assert assistant_activity(set(),{(200,5,50,'bash',False)})==[(200,5,'bash')]
 with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
  lock=Path(directory)/'gpu.lock'
  lock.touch()
  assert lease_available(lock)
  with lock.open('r+') as held:
   fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
   assert not lease_available(lock)
   fcntl.flock(held,fcntl.LOCK_UN)
  assert lease_available(lock)
 actual=assistant_children()
 print(f'Other-session guard self-test passed; {len(assistant_activity(actual,actual))} active child jobs observed.')


def unit_pid(unit):
 return int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True,timeout=5).strip())


def unexpected(records,main,secondary,initial_main):
 result=[]
 for gpu,pid in records:
  if gpu==OTHER and pid==secondary:continue
  if gpu in PAIR and pid in (main,initial_main) and pid:continue
  try:exe=os.readlink(f'/proc/{pid}/exe')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if gpu in PAIR and exe in BINARIES:continue
  result.append((gpu,pid,exe))
 return result


def restore_if_needed(start_hash,config_hash):
 state=subprocess.check_output(['systemctl','--user','show','fifi-qwen36-engine.service',
                                '-p','ActiveState','--value'],text=True,timeout=5).strip()
 if state!='inactive':return
 start=Path('/home/ben/fifi-qwen36/start.sh')
 config=Path('/home/ben/fifi-qwen36/qwen36-p2p-256k.json')
 if hashlib.sha256(start.read_bytes()).hexdigest()!=start_hash:return
 if hashlib.sha256(config.read_bytes()).hexdigest()!=config_hash:return
 if any(gpu in PAIR for gpu,_ in processes()):return
 with socket.socket() as s:
  s.settimeout(1)
  if s.connect_ex(('127.0.0.1',18791))==0:return
 subprocess.run(['systemctl','--user','start','--no-block','fifi-qwen36-engine.service'],check=True,timeout=10)
 print('Guard restored the unchanged main service.',flush=True)


def main():
 if not lease_available():
  print('The shared GPU lease is held; no GPU window reserved.',flush=True)
  return 75
 initial_children=assistant_children()
 active_children=assistant_activity(initial_children,initial_children)
 if active_children:
  print(f'Other assistant has active child work: {active_children}; no GPU window reserved.',flush=True)
  return 75
 secondary=unit_pid('fifi-qwen36-2080ti.service')
 assert secondary, 'the independent 2080 Ti service is not running'
 initial_main=unit_pid('fifi-qwen36-engine.service')
 assert initial_main, 'the main service must be running before validation'
 start_hash=hashlib.sha256(Path('/home/ben/fifi-qwen36/start.sh').read_bytes()).hexdigest()
 config_hash=hashlib.sha256(Path('/home/ben/fifi-qwen36/qwen36-p2p-256k.json').read_bytes()).hexdigest()
 run_id=f'{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}'
 run=ROOT/'logs'/f'peer-guard-{run_id}.jsonl'
 env=os.environ.copy();env['FIFI_PEER_RUN_ID']=run_id
 child=subprocess.Popen(['bash',str(ROOT/'run-peer-validation.sh')],start_new_session=True,env=env)
 fault=None
 try:
  with run.open('x') as out:
   while child.poll() is None:
    try:
     main_pid=unit_pid('fifi-qwen36-engine.service')
     gpu=processes()
     util=int(subprocess.check_output(['nvidia-smi','-i',OTHER,'--query-gpu=utilization.gpu',
              '--format=csv,noheader,nounits'],text=True,timeout=5).strip())
     data={'time':time.time(),'unexpected':unexpected(gpu,main_pid,secondary,initial_main),
           'builders':builders(),'assistant_activity':assistant_activity(initial_children,assistant_children()),
           'secondary_utilization':util,'processes':gpu}
     out.write(json.dumps(data)+'\n');out.flush()
     if data['unexpected'] or data['builders'] or data['assistant_activity'] or util:
      fault=data;print('Competing work detected; cancelling only our GPU tests.',flush=True)
      try:os.killpg(child.pid,signal.SIGTERM)
      except ProcessLookupError:pass
      break
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
     fault={'monitor_error':repr(error)};out.write(json.dumps(fault)+'\n');out.flush()
     try:os.killpg(child.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     break
    time.sleep(1)
   try:status=child.wait(timeout=60)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);status=child.wait()
   if status==0 and fault is None:
    try:
     main_pid=unit_pid('fifi-qwen36-engine.service');gpu=processes()
     util=int(subprocess.check_output(['nvidia-smi','-i',OTHER,'--query-gpu=utilization.gpu',
              '--format=csv,noheader,nounits'],text=True,timeout=5).strip())
     post={'unexpected':unexpected(gpu,main_pid,secondary,initial_main),
           'pair_extras':[(device,pid) for device,pid in gpu
                          if device in PAIR and pid not in (main_pid,initial_main)],
           'builders':builders(),'assistant_activity':assistant_activity(initial_children,assistant_children()),
           'secondary_utilization':util}
     out.write(json.dumps({'postcheck':post})+'\n')
     if any(post.values()):fault={'postcheck':post}
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
     fault={'postcheck_error':repr(error)}
   out.write(json.dumps({'exit':status,'fault':fault})+'\n')
  print(f'GPU guard receipt: {run}',flush=True)
  if status==0 and fault is None:
   directory=ROOT/'logs'/f'peer-validation-{run_id}'
   subprocess.run([sys.executable,str(ROOT/'peer-evidence.py'),'create',str(directory),str(run)],check=True)
   print(f'Complete peer evidence: {directory}',flush=True)
  return 75 if fault else status
 finally:
  if child.poll() is None:
   try:os.killpg(child.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   child.wait(timeout=60)
  try:restore_if_needed(start_hash,config_hash)
  except (OSError,ValueError,subprocess.SubprocessError) as error:
   print(f'Guard could not restore the main service: {error}',flush=True)


if __name__=='__main__':
 if sys.argv[1:]==['--self-test']:
  self_test()
  raise SystemExit(0)
 signal.signal(signal.SIGTERM,interrupted)
 signal.signal(signal.SIGINT,interrupted)
 raise SystemExit(main())

"""Bounded execution, restartable caches, and two retries per failed branch."""
import json,time,subprocess,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
START=1790949979.;EXPERIMENT_END=START+7.25*3600;FINAL_END=START+9*3600

def run(arguments,label,deadline=EXPERIMENT_END):
    if time.time()>=deadline:return False
    state={'label':label,'arguments':list(map(str,arguments)),'deadline':deadline}
    (OUT/'active_job.json').write_text(json.dumps(state,indent=2)+'\n')
    with (OUT/(label+'.log')).open('a') as log:
        for attempt in range(1,4):
            child=subprocess.Popen([ROOT/'venv/bin/python',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(attempt=attempt,pid=child.pid,status='running');(OUT/'active_job.json').write_text(json.dumps(state,indent=2)+'\n')
            print('JOB',label,'attempt',attempt,'pid',child.pid,flush=True)
            while child.poll() is None:
                if time.time()>=deadline:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    state['status']='time_limit';(OUT/'active_job.json').write_text(json.dumps(state,indent=2)+'\n');return False
                time.sleep(2)
            if child.returncode==0:
                state['status']='complete';(OUT/'active_job.json').write_text(json.dumps(state,indent=2)+'\n');return True
            print('RETRY',label,'exit',child.returncode,flush=True)
            if time.time()>=deadline:return False
    state['status']='failed_after_three_attempts';(OUT/'active_job.json').write_text(json.dumps(state,indent=2)+'\n');return False

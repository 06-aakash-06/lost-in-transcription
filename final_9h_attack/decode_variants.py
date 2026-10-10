"""Two meaningful decode alternatives, with a strict one-hour wall limit."""
import argparse,time,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
OUT=HERE/'results';sys.path.insert(0,str(ROOT))

def checkpoint(fold,coefficient):
    if coefficient in (0,1):return ROOT/f'overnight_attack/runs/large_v3_{fold}/epoch_1_{"half" if coefficient==0 else "full"}'
    return HERE/f'models/soup_{coefficient}_{fold}'

def run(coefficient,deadline=None):
    started=time.time();end=min(started+3600,deadline or float('inf'))
    (OUT/'decode_clock.json').write_text(json.dumps({'start':started,'deadline':end},indent=2)+'\n')
    # Both-fold greedy auto first; Indonesian beam3 is the only beam variant.
    for language,beams in [('auto',1),('indonesian',3)]:
        for fold in ['fold_B','fold_A']:
            label=f'decode_{language}_beam{beams}_{fold}';output=OUT/f'predictions/decoding/{language}_beam{beams}/{fold}_valid.json';output.parent.mkdir(parents=True,exist_ok=True)
            args=[ROOT/'overnight_attack/evaluate.py','--manifest',ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv','--checkpoint',checkpoint(fold,coefficient),'--base',ROOT/'overnight_attack/models/large_v3_base','--dtype','bfloat16','--language',language,'--beams',str(beams),'--batch-size','4','--output',output]
            if time.time()>=end:return
            with (OUT/(label+'.log')).open('a') as log:
                for attempt in range(1,4):
                    child=subprocess.Popen([ROOT/'venv/bin/python',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    print('DECODE',label,'attempt',attempt,'pid',child.pid,flush=True)
                    while child.poll() is None:
                        if time.time()>=end:
                            child.terminate()
                            try:child.wait(timeout=15)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                            print('DECODE TIME LIMIT; partial cache retained',flush=True);return
                        time.sleep(2)
                    if child.returncode==0:break
                    print('DECODE RETRY',label,'exit',child.returncode,flush=True)
                    if time.time()>=end:return

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coefficient',type=float,required=True);p.add_argument('--deadline',type=float);a=p.parse_args();run(a.coefficient,a.deadline)

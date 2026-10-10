"""Bounded checkpoint selection; prior predictions remain untouched."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from overnight_attack.soup import average
from overnight_attack.evidence import score,aggregate
from phase4_stronger_anchor.audit import aligned
import pandas as pd
START=1790949979.;PHASE_END=START+90*60
BASE=ROOT/'overnight_attack/models/large_v3_base'
OUT=HERE/'results';OUT.mkdir(parents=True,exist_ok=True)

def prediction(fold,coefficient,dataset):
    name='epoch_1_half' if coefficient==0 else 'epoch_1_full'
    if coefficient in (0,1):
        return ROOT/f'overnight_attack/results/predictions/large_v3_{fold}/{name}/{dataset}.csv'
    return OUT/f'predictions/soup_{coefficient}/{fold}/{dataset}.csv'

def summary():
    records={}
    for coefficient in [0,.25,.5,.75,1]:
        row={}
        for fold in ['fold_A','fold_B']:
            for dataset in [f'{fold}_valid','jember_holdout']:
                path=prediction(fold,coefficient,dataset)
                if not path.exists():continue
                manifest=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{dataset}.tsv',sep='\t',keep_default_na=False)
                row[fold if dataset.endswith('_valid') else fold+'_jember']=score(manifest.text.tolist(),aligned(path,manifest))
        if all(f in row for f in ['fold_A','fold_B']):
            row['aggregate']=aggregate([row['fold_A'],row['fold_B']]);row['worst_fold']=max(row[f]['wer'] for f in ['fold_A','fold_B'])
        if all(f+'_jember' in row for f in ['fold_A','fold_B']):row['jember']=aggregate([row[f+'_jember'] for f in ['fold_A','fold_B']])
        records[str(coefficient)]=row
    (OUT/'checkpoint_comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    print('CHECKPOINT SUMMARY',json.dumps(records),flush=True)

def run(args,label):
    if time.time()>=PHASE_END:return False
    if '--output' in args:
        output=Path(args[args.index('--output')+1]);cache=output.with_suffix('.cache.json')
        batch=json.loads(cache.read_text())['provenance']['batch_size'] if cache.exists() else 12
        args=[*args,'--batch-size',str(batch)]
    log=OUT/(label+'.log')
    with log.open('a') as stream:
        for attempt in range(1,4):
            child=subprocess.Popen([str(ROOT/'venv/bin/python'),*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            print('RUN',label,'attempt',attempt,'pid',child.pid,flush=True)
            while child.poll() is None:
                if time.time()>=PHASE_END:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    print('PHASE TIME LIMIT; cache retained',label,flush=True);return False
                time.sleep(2)
            if child.returncode==0:return True
            print('RETRY',label,'exit',child.returncode,flush=True)
            if time.time()>=PHASE_END:return False
    return False

if __name__=='__main__':
    (OUT/'clock.json').write_text(json.dumps({'start_unix':START,'phase1_end_unix':PHASE_END,'experiments_end_unix':START+7.25*3600,'final_end_unix':START+9*3600},indent=2)+'\n')
    summary()
    for coefficient in [.75,.5,.25]:
        for fold in ['fold_A','fold_B']:
            run_dir=ROOT/f'overnight_attack/runs/large_v3_{fold}'
            adapter=HERE/f'models/soup_{coefficient}_{fold}'
            average(run_dir/'epoch_1_half',run_dir/'epoch_1_full',adapter,1-coefficient)
            for dataset in [f'{fold}_valid','jember_holdout']:
                output=prediction(fold,coefficient,dataset).with_suffix('.json')
                if not run([ROOT/'overnight_attack/evaluate.py','--manifest',ROOT/f'phase1_whisper_anchor/data/{dataset}.tsv','--checkpoint',adapter,'--base',BASE,'--dtype','bfloat16','--output',output],f'soup_{coefficient}_{fold}_{dataset}'):summary();sys.exit(0)
                summary()
    run([ROOT/'overnight_attack/evaluate.py','--manifest',ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv','--checkpoint',ROOT/'overnight_attack/runs/final_large_v3/epoch_1_half','--base',BASE,'--dtype','bfloat16','--output',OUT/'predictions/final_half_jember.json'],'final_half_jember')
    summary()

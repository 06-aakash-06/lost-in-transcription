"""Serial MPS queue: confidence, cross-fold gate, and final-model OOD checks."""
import sys,json,time,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from final_9h_attack.jobs import run,EXPERIMENT_END
from overnight_attack.soup import average
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def execute(wait_pid):
    while wait_pid:
        state=subprocess.run(['ps','-o','stat=','-p',str(wait_pid)],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith('Z'):break
        time.sleep(2)
    status={}
    status['confidence']=run([HERE/'confidence.py','--coefficient','1'],'confidence_full',min(time.time()+1800,EXPERIMENT_END))
    if status['confidence']:
        status['edit_gate']=run([HERE/'edit_gate.py','--coefficient','1'],'edit_gate_full',min(time.time()+600,EXPERIMENT_END))
    config=OUT/'turbo_diagnostic_config.json'
    config.write_text(json.dumps({'family':'whisper','language':'indonesian','beams':1,'cuda_dtype':'float32','mps_dtype':'float32','cpu_dtype':'float32'})+'\n')
    manifest=ROOT/'phase1_whisper_anchor/data/jember_holdout.tsv'
    for name,model in [('turbo_full',ROOT/'phase3_protected_fusion/models/whisper_full'),('turbo_half',ROOT/'phase1_whisper_anchor/models/final_merged')]:
        status[name]=run([HERE/'evaluate_model.py','--model',model,'--config',config,'--manifest',manifest,'--output',OUT/f'predictions/final_ood/{name}.json','--batch-size','4'],f'final_ood_{name}',min(time.time()+2400,EXPERIMENT_END))
    final=ROOT/'overnight_attack/runs/final_large_v3'
    for name,adapter in [('large_half',final/'epoch_1_half'),('large_soup075',HERE/'models/final_soup_0.75')]:
        if name=='large_soup075':average(final/'epoch_1_half',final/'epoch_1_full',adapter,.25)
        status[name]=run([ROOT/'overnight_attack/evaluate.py','--manifest',manifest,'--checkpoint',adapter,'--base',ROOT/'overnight_attack/models/large_v3_base','--dtype','bfloat16','--language','indonesian','--beams','1','--batch-size','8','--output',OUT/f'predictions/final_ood/{name}.json'],f'final_ood_{name}',min(time.time()+1800,EXPERIMENT_END))
    # Finish only the missing remainder from the bounded checkpoint grid.
    partial=OUT/'predictions/soup_0.25/fold_B/jember_holdout.cache.json'
    batch=json.loads(partial.read_text())['provenance']['batch_size'] if partial.exists() else 8
    status['soup025_jember_B']=run([ROOT/'overnight_attack/evaluate.py','--manifest',manifest,'--checkpoint',HERE/'models/soup_0.25_fold_B','--base',ROOT/'overnight_attack/models/large_v3_base','--dtype','bfloat16','--language','indonesian','--beams','1','--batch-size',str(batch),'--output',OUT/'predictions/soup_0.25/fold_B/jember_holdout.json'],'finish_soup025_jember_B',min(time.time()+900,EXPERIMENT_END))
    (OUT/'phase3_done.json').write_text(json.dumps({'status':status,'completed_at':time.time()},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args();execute(a.wait_pid)

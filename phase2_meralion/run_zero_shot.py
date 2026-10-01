from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase2_meralion.evaluate import evaluate
from phase2_meralion.analyze import analyze,aggregate
HERE=Path(__file__).resolve().parent
metrics=[]
for fold in ('fold_A','fold_B'):
    output=HERE/f'results/zero_shot_{fold}.csv'
    evaluate(ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv',output)
    metrics.append(analyze(fold,output,HERE/f'results/zero_shot_pair_{fold}.json'))
result=aggregate(metrics)
result['passes_training_gate']=result['oracle_gain']>=.01 or metrics[0]['oracle_gain']>=.015
(HERE/'results/zero_shot_aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
print('ZERO SHOT GATE',json.dumps(result,indent=2),flush=True)

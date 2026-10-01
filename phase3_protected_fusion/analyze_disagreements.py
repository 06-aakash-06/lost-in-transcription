import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase3_protected_fusion.data import load
from phase3_protected_fusion.runtime.fusion import blocks,repetition
from phase1_whisper_anchor.training.metrics import counts
import pandas as pd
HERE=Path(__file__).resolve().parent

def errors(ref,text):return sum(counts(ref,text)[1:])
def main():
    rows=[]
    for fold in ('fold_A','fold_B'):
        for item in load(fold).itertuples():
            a,b=item.anchor.split(),item.meralion.split();base=errors(item.text,item.anchor)
            for block in blocks(a,b):
                s,e,u,v=block.start,block.end,block.other_start,block.other_end
                kind='substitution' if s<e and u<v else 'insertion' if u<v else 'deletion'
                candidate=' '.join(a[:s]+b[u:v]+a[e:]);gain=base-errors(item.text,candidate)
                rows.append({'fold':fold,'clip_id':item.clip_id,'operation':kind,'anchor_span':' '.join(a[s:e]),'meralion_span':' '.join(b[u:v]),'anchor_start':s,'anchor_end':e,'meralion_start':u,'meralion_end':v,'span_size':max(e-s,v-u),'left_context':block.left,'right_context':block.right,'error_gain':gain,'meralion_repeat_fraction':repetition(item.meralion)})
    f=pd.DataFrame(rows);f.to_csv(HERE/'results/disagreements.csv',index=False)
    summary=[]
    for (fold,kind),part in f.groupby(['fold','operation']):
        summary.append({'fold':fold,'operation':kind,'regions':len(part),'helps':int(sum(part.error_gain>0)),'damages':int(sum(part.error_gain<0)),'neutral':int(sum(part.error_gain==0)),'net_error_gain':int(part.error_gain.sum())})
    (HERE/'results/disagreement_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()

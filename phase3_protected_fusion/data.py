"""Immutable aligned OOF inputs; exact Phase 1 competition scorer."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def load(fold):
    manifest=pd.read_csv(ROOT/f'phase1_whisper_anchor/data/{fold}_valid.tsv',sep='\t',keep_default_na=False).set_index('clip_id')
    paths={'anchor':ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_half/valid.csv','meralion':ROOT/f'phase2_meralion/results/pred_epoch_1_half_{fold}.csv','ratio4':ROOT/f'phase1_whisper_anchor/runs/indonesian_x4_{fold}/epoch_1_full/valid.csv','auto':ROOT/f'phase1_whisper_anchor/runs/auto_x3_{fold}/epoch_1_half/valid.csv'}
    paths['full']=ROOT/f'phase1_whisper_anchor/runs/indonesian_x3_{fold}/epoch_1_full/valid.csv'
    for name,path in paths.items():
        f=pd.read_csv(path,keep_default_na=False).set_index('clip_id')
        assert f.index.is_unique and set(f.index)==set(manifest.index),(name,'alignment')
        f=f.loc[manifest.index]
        assert f.reference.tolist()==manifest.text.tolist(),(name,'reference')
        manifest[name]=f.transcript
    return manifest.reset_index()

"""Persist a factual handoff report from completed runs only."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def report(verdict,reason='',selected=None):
    def read(name):
        p=HERE/'results'/name
        return json.loads(p.read_text()) if p.exists() else None
    zero=[read(f'zero_shot_pair_fold_{f}.json') for f in ('A','B')]
    adapted=[read(f'adapted_{selected}_fold_{f}.json') if selected else None for f in ('A','B')]
    agg=read('adapted_aggregate.json') or read('zero_shot_aggregate.json')
    lines=['# Phase 2 MERaLiON report','',f'## Verdict: {verdict}','',reason,'','## Official WER','', '| System | Fold A | Fold B | Aggregate |','| --- | ---: | ---: | ---: |']
    def row(name,values,key,aggregate=None):
        vals=[f'{v[key]["wer"]:.6f}' if v else 'not run' for v in values]
        lines.append('| '+name+' | '+' | '.join(vals)+' | '+(f'{aggregate[key]["wer"]:.6f}' if aggregate else 'not run')+' |')
    zagg=read('zero_shot_aggregate.json');aagg=read('adapted_aggregate.json')
    row('Whisper Phase 1',zero,'whisper',zagg)
    row('MERaLiON zero-shot',zero,'meralion',zagg)
    row('Zero-shot pair oracle',zero,'oracle',zagg)
    row('Zero-shot simple ROVER',zero,'fusion',zagg)
    if selected:
        for name,key in [('Adapted MERaLiON','meralion'),('Adapted pair oracle','oracle'),('Adapted simple ROVER','fusion')]:row(name,adapted,key,aagg)
        if all(adapted):
            regressions=[f'Fold {fold}: +{value["fusion"]["wer"]-value["whisper"]["wer"]:.6f}' for fold,value in zip(('A','B'),adapted) if value['fusion']['wer']>value['whisper']['wer']]
            if regressions:lines+=['','Simple fusion regresses versus Whisper ('+'; '.join(regressions)+'). The oracle supports Phase 3 experiments, but this diagnostic fusion is not ready for deployment.']
    lines+=['','The per-clip oracle uses references to choose the lower-error transcript. It is an upper bound on transcript selection, not an executable fusion strategy. Simple ROVER is a fixed equal-weight diagnostic, with Whisper substitution tie priority and conservative blanks; it is not a Phase 3 submission. Raw Phase 1 OOF predictions are read without modification.','', '## Artifacts','', '- Zero-shot and adapted predictions/analyses: `results/`.','- Decoder adapters and training metrics: `runs/`.','- Frozen speech feature cache: `cache/`; never packaged.','- Existing Phase 1 manifests/scorer/text/audio utilities are reused by import.','', '## Recipe','', 'One seed (1337), ratio 3, one epoch, r16/alpha32/dropout0.05 decoder LoRA on q/k/v/o/gate/up/down projections; frozen encoder and audio projector; AdamW 1e-4, weight decay0.01, warmup8%, clip1.0, microbatch1, accumulation16, gradient checkpointing. MPS BF16 training and enabled PyTorch CPU operator fallback. Cached frozen speech representations and checkpointed answer-only LM-head loss reduce memory without changing the causal objective.']
    if selected:lines+=['',f'Selected checkpoint: `{selected}`.']
    recovery=read('training_recovery.json')
    if recovery:lines+=['','## Training recovery','',json.dumps(recovery,indent=2)]
    merged=HERE/'models/final_merged/merge_validation.json'
    if merged.exists():
        size=sum(p.stat().st_size for p in merged.parent.rglob('*') if p.is_file())
        lines+=['', '## Final model','', f'`models/final_merged/` ({size:,} bytes)', '', 'Merge validation: `'+json.dumps(json.loads(merged.read_text()))+'`.']
        offline=read('offline_validation.json');tokenizer=read('tokenizer_validation.json')
        if offline:lines+=['','Offline standalone load validation: `'+json.dumps(offline)+'`.']
        if tokenizer:lines+=['','Processor validation: `'+json.dumps(tokenizer)+'`. The saved-tokenizer regex warning is retained without changing the trained tokenizer; its regex and all checked token IDs match the base.']
        for run in ('fold_A','fold_B','final'):
            p=HERE/'runs'/run/selected/'train_metrics.json' if selected else None
            if p and p.exists():lines+=['',f'{run} training: `'+json.dumps(json.loads(p.read_text()))+'`.']
    lines+=['','## Runtime','', 'Local generation uses MPS; CUDA is preferred automatically when available. CUDA/A100 runtime is not measurable on this machine. Per-run elapsed times are in evaluation JSON files. All weights and custom model/processor code are local; no competition archive is built in Phase 2.','', '## License','', 'Public model source and downloaded licence are recorded in `MODEL_NOTICES.md`; preserve the licence and attribution in Phase 3 packaging.']
    if selected:
        timing=[read(f'pred_{selected}_fold_{fold}.json') for fold in ('A','B')]
        if all(timing):
            seconds=sum(v['elapsed_this_invocation_s'] for v in timing)
            clips=sum(v['clips'] for v in timing)
            cost={'device':'mps','batch_size':1,'clips':clips,'elapsed_seconds':seconds,'clips_per_second':clips/seconds,'projected_2118_clip_seconds_mps':seconds/clips*2118,'cuda_a100_measured':False}
            (HERE/'results/inference_cost.json').write_text(json.dumps(cost,indent=2)+'\n')
            lines+=['','Measured adapted OOF generation cost (includes loading and equivalence checks): `'+json.dumps(cost)+'`. The MPS projection is not an A100 runtime estimate. Phase 3 must benchmark CUDA batching before committing its final archive.']
    (HERE/'PHASE2_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':report('IN PROGRESS','Fresh zero-shot gate is running; expensive training has not begun.')

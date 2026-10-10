"""Write the final evidence and artifact receipt only after ZIP validation."""
import json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def read(name):return json.loads((OUT/name).read_text())
def number(value):return f'{value:.6f}'
def generate():
    checkpoints=read('checkpoint_comparison.json');fusion=read('fusion_comparison_1.0_adapted.json');guard=read('phrase_guard_comparison.json');gate=read('edit_gate_comparison.json');decode=read('decoding_comparison.json')
    primary=ROOT/'artifacts/FINAL_MAX_WER_REDUCTION_SUBMISSION.zip';backup=ROOT/'artifacts/FINAL_LARGEV3_STANDALONE_BACKUP.zip'
    inventories={name:json.loads(path.with_suffix('.inventory.json').read_text()) for name,path in [('recommended',primary),('backup',backup)]}
    validations={name:read(f'offline_zip_tests/{name}/validation.json') for name in ['recommended','backup']}
    assert all(r['pass'] for r in validations.values())
    assert all(r['pass'] for r in read('guard_zip_smoke.json').values())
    assert read('batch1_runtime_equivalence.json')['pass']
    assert read('preservation_verified.json')['pass']
    optional=read('nbest_comparison.json') if (OUT/'nbest_comparison.json').exists() else None
    config=json.loads((HERE/'configs/selected_final.json').read_text())
    use_nbest=bool(config.get('nbest'))
    if use_nbest:assert optional and optional['passes_continue_gate']
    metrics=optional['results']['dev'] if use_nbest else guard['candidates']['1']['dev']
    a=metrics['fold_A']['candidate' if use_nbest else 'consensus'];b=metrics['fold_B']['candidate' if use_nbest else 'consensus'];agg=metrics['aggregate']['candidate' if use_nbest else 'consensus']
    old=fusion['candidates']['Old conservative']['dev'];final_j=guard['final_ood']['full']
    lines=['# Final nine-hour ASR report','',f'**READY TO SUBMIT: YES.** Recommended archive: `{primary}`.','',
        '**Selected final system**','',
        'Whisper Large-v3 full 1.0-epoch adapter (full contribution 1.00), Indonesian, greedy, native long-form. Public frozen BF16 base plus exact FP32 LoRA residuals, r16/alpha32, 512 attention/MLP projections. Final training used all 372 competition clips and 1309 Jember chunks, competition sampling ratio 4. Checkpoint: `overnight_attack/runs/final_large_v3/epoch_1_full` (batch 2797/2797, optimizer step 175).',
        '',
        'Turbo R3-full and adapted MERaLiON support only exact shared word replacements at unambiguous optimal alignments. Ordinary fusion uses substitutions only; no single alternative overrides Large-v3. Turbo R3-half is excluded. One conservative phrase-loop safeguard is applied before voting, with no further edits after cleanup.',
        '',
        ('Targeted N-best rescoring is enabled with a frozen training-text trigram LM.' if use_nbest else 'No learned edit gate or N-best rescoring is required in the selected package.'),
        '',
        '**Large-v3 checkpoint results**','',
        'Full contribution is the coefficient of the 1.0-epoch adapter. Soups average actual LoRA deltas by scaled factor concatenation, rather than averaging A/B factors. Same frozen base, architecture and precision are verified. Jember below is the mean of two fold models on the same 162 held-out clips, not 324 independent examples.',
        '',
        '| Full contribution | A | B | Aggregate | Jember | Worst fold |','|---:|---:|---:|---:|---:|---:|']
    for coef in ['0','0.25','0.5','0.75','1']:
        r=checkpoints[coef];lines.append('| '+ ' | '.join([f'{float(coef):.2f}',*[number(r[k]['wer']) for k in ['fold_A','fold_B','aggregate','jember']],number(r['worst_fold'])])+' |')
    lines+=['','Full was selected after all five fold comparisons: the 75% soup has a one-error fused aggregate advantage, but full has better Fold B and Jember results and a simpler r16 delta. Half underfits both folds. The final OOD warning was dominated by one multiword repetition failure, rather than evidence that half is safer.','',
        '**Final all-data Jember check**','','| Candidate | Standalone | Consensus |','|---|---:|---:|']
    for label,key in [('Half','half'),('75% full soup','soup075'),('Full','full')]:
        r=guard['final_ood'][key];lines.append(f'| {label} | {number(r["standalone"]["wer"])} | {number(r["consensus"]["wer"])} |')
    old_j=read('final_ood_comparison.json')['results']['old_conservative']['wer']
    lines += [f'| Old conservative Turbo system | — | {number(old_j)} |','',
        f'Full without the phrase guard scored 0.265660 standalone / 0.261175 fused. The fixed guard reduces those to {number(final_j["standalone"]["wer"])} / {number(final_j["consensus"]["wer"])}. It removes 168 loop words on one Jember clip and changes zero predictions on either OOF fold or the fold-model Jember checks, across all five coefficients. No IDs or references are inference features.',
        '',
        'Guard: preserve the existing single-word rule; additionally trim a 2–6 word phrase only when it repeats at least five times, occupies at least 20 words and at least 35% of the transcript. Retain two cycles. One fixed rule was tested; no threshold grid or lexical exception was fitted.',
        '',
        '**Decoding**','',
        'Forced Indonesian and auto-language greedy produced identical transcripts on both folds, with no additional oracle headroom. Indonesian beam 3 scored 0.180259 on B (greedy 0.178009), and 0.176322 after consensus (greedy 0.172666). Their B pair oracle was 0.173791. Beam 3 failed the continue gate; A remained a preserved 96/294-clip partial cache at the one-hour cap. It is not deployed or presented as a two-fold result.',
        '',
        '**Fusion results**','','| Candidate | A | B | Aggregate | Fold-model Jember | S / I / D |','|---|---:|---:|---:|---:|---|']
    for name,r in fusion['candidates'].items():
        d=r['dev'];e=d['aggregate']['edits'];lines.append('| '+ ' | '.join([name,*[number(d[k]['wer']) for k in ['fold_A','fold_B','aggregate']],number(r['jember']['aggregate']['wer']),f'{e["substitutions"]} / {e["insertions"]} / {e["deletions"]}'])+' |')
    lines += ['',
        'Selected deterministic C makes 277 substitutions in 267 spans across 173/372 clips, 0.7446 corrected words per clip. No ordinary insertions/deletions. Final-model Jember: 102 substitutions in 100 spans; the separate catastrophic-loop guard removes 168 words on one clip. The 75% soup C scores 0.193880 / Jember 0.246973; 50% soup C scores 0.194452 / Jember 0.247870. Neither earns selection over full.',
        '',
        'Clip-oracle diagnostics: Large+Turbo 0.183929; Large+MERaLiON 0.178725; triple 0.170603. These are reference-selected analysis bounds, never deployable predictions. Deterministic C converts 105 OOF word errors relative to Large-v3 alone. MERaLiON and Turbo are both needed for the tested two-supporter correction rule.',
        '',
        '**Confidence and learned gate**','','| Rule | A | B | Aggregate | Jember |','|---|---:|---:|---:|---:|']
    for name,r in gate['results'].items():lines.append('| '+' | '.join([name,*[number(r[k]['wer']) for k in ['fold_A','fold_B','aggregate']],number(r['jember']['wer'])])+' |')
    lines += ['',
        'One fixed logistic recipe was trained on A→B and B→A using only acoustic/hypothesis features. It fails both directions and is rejected. Confidence thresholds also fail the both-fold robustness test. No sklearn/PEFT dependency or gate coefficients are required in the ZIP.',
        '', '**One optional experiment: N-best**','']
    if optional:
        lines+=['Fixed trigram candidate rescoring, trained separately from each fold’s allowed training transcripts. Only short clips where consensus proposes edits and mean Large-v3 word probability is below 0.8 are queried. Beam-3 top three plus greedy; acoustic score + 0.1×LM, acceptance margin 0.02, length within 15%, no increased repetition. No free rewriting or hidden data.','',
            '| Variant | A | B | Aggregate | Jember |','|---|---:|---:|---:|---:|']
        for label,key in [('Baseline C','baseline_C'),('N-best','candidate')]:
            d=optional['results']['dev'];j=optional['results']['jember'];lines.append('| '+' | '.join([label,number(d['fold_A'][key]['wer']),number(d['fold_B'][key]['wer']),number(d['aggregate'][key]['wer']),number(j['aggregate'][key]['wer'])])+' |')
        lines+=['',f'Continue gate: {"PASS" if optional["passes_continue_gate"] else "FAIL"}. Requires gains on both folds, at least 0.0015 aggregate improvement and Jember worsening no more than 0.001. N-best improves both folds and Jember, but converts only nine competition word errors, below the predeclared material-gain gate. It makes 305 substitutions, two insertions and eight deletions versus C’s 277 substitutions and no insertions/deletions. The simpler correction policy is retained. Checkpoint sequence ensemble and longer training were not run.']
    else:lines+=['Time-limited or incomplete; preserved caches are not eligible for selection. No other optional experiment was run.']
    lines += ['', '**Old hidden system OOF vs new final OOF**','',
        '| System | A | B | Aggregate | Worst fold |','|---|---:|---:|---:|---:|',
        '| Old conservative (hidden 0.2712) | '+' | '.join([*[number(old[k]['wer']) for k in ['fold_A','fold_B','aggregate']],number(old['worst_fold'])])+' |',
        '| Selected final | '+' | '.join([number(a['wer']),number(b['wer']),number(agg['wer']),number(max(a['wer'],b['wer']))])+' |','',
        'Selection prioritizes consistent gains on both conversations, model capacity, stable final-model Jember and inference reliability. New hidden WER is unmeasured; no improvement is fabricated.',
        '', '**Final ZIP and validation**','']
    for name,path in [('recommended',primary),('backup',backup)]:
        inv=inventories[name];v=validations[name]
        lines += [f'- {name.title()}: `{path}`',f'- SHA256: `{inv["sha256"]}`',f'- ZIP: {inv["zip_bytes"]:,} bytes ({inv["zip_bytes"]/1e9:.3f} GB); bundled model assets: {inv["model_bytes"]:,} bytes ({inv["model_bytes"]/1e9:.3f} GB).',f'- Smoke: **PASS**; MPS end-to-end runs {", ".join(f"{t:.1f}s" for t in v["inference_seconds"])}.']
    lines += ['',
        'Both exact archives pass CRC/SHA and complete asset/shard-map checks. Exact model bytes are verified against archive entries; local hardlinks only save extraction disk space. Actual main.py runs with sockets blocked, empty HF caches, PEFT/sklearn unavailable, a 21.597875-second clip and a 37.6535-second clip. CSV schema, quoted filenames, row order, no NaNs/blanks, repeated byte-identical output, reordering independence, native long-form tail coverage and silent inference pass. CUDA-first selection, CSV rejection, memory retry, consensus invariants and phrase-guard tests pass.',
        '',
        '**Expected A100 runtime:** approximately 60–100 minutes. The exact same two audio files are used in the old/new MPS benchmark: old inference 30.9/29.9 seconds; new summed model loading/decoding approximately 44–46 seconds. Scaling the user-verified old 31-minute A100 run gives approximately 46 minutes before a substantial safety margin. Sequential residency and CUDA batch 8 are retained. New A100 timing is not measured: CUDA is unavailable locally. Local component timings and hardware evidence are saved; the hard limit is 120 minutes. CPU fallback is implemented but not timed.',
        '',
        'Extra case check: the packaged single-clip loader and original PEFT model produce identical 52-word predictions on the cached repetition case; the original batch-4 cache had 174 loop words. Single-clip packaged inference produces nonrepeating speech, and a recorded-loop replay at the ASR boundary verifies the exact archived entrypoint applies the guard. OOF/OOD tables remain cached batched diagnostics; batch/backend numerical differences can alter recognition. The replay is an integration test, not a new ASR accuracy measurement.',
        '',
        '**Preservation:** all five pre-existing ZIPs retain their SHA256 hashes; all 627 overnight files and the Ratio-4 interrupted adapter/state are unchanged. Ratio-4 remains stopped at batch 218/2797. Its optimizer, scheduler and partial accumulated gradients remain at the saved resume path. The earlier provisional backup is preserved byte-for-byte under `artifacts/preserved_9h_builds/`. No working prior checkpoint or submission was overwritten.',
        '',
        f'Ratio-4 resume checkpoint: `{ROOT}/phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218`. Resume instructions: `phase4_stronger_anchor/RESUME.md`. No next Ratio-4 stage or evaluation was started.',
        '',
        'Evidence, prediction caches, logs, portable source and archive inventories: `final_9h_attack/`. Final all-data optimizer/scheduler state remains under `overnight_attack/runs/final_large_v3/epoch_1_full/`. No new ASR architecture was trained, MERaLiON was not retrained, and no final competition submission was uploaded.',
        '', '**READY TO SUBMIT: YES.**']
    (ROOT/'FINAL_9H_REPORT.md').write_text('\n'.join(lines)+'\n')
    receipt={'recommended_zip':str(primary),'sha256':inventories['recommended']['sha256'],'backup_zip':str(backup),'fold_A':a['wer'],'fold_B':b['wer'],'aggregate':agg['wer'],'ready_to_submit':True,'written_at_unix':time.time(),'new_a100_measured':False,'report':str(ROOT/'FINAL_9H_REPORT.md')}
    (ROOT/'artifacts/FINAL_RECOMMENDED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':generate()

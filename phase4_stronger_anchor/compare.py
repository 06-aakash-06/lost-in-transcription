"""Official cached-text comparisons; references are used only for scoring."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase1_whisper_anchor.training.metrics import counts
from phase3_protected_fusion.runtime.fusion import fuse, collapse_single_runs, repetition, alignment
from phase4_stronger_anchor.audit import aligned
HERE = ROOT / 'phase4_stronger_anchor'; OUT = HERE / 'results'
FOLDS = ['fold_A', 'fold_B']
CHECKPOINTS = ['epoch_1_half', 'epoch_1_full', 'epoch_2_half', 'epoch_2_full']

@lru_cache(maxsize=100000)
def errors(ref, hyp): return counts(ref, hyp)

def corpus(refs, hyps):
    assert len(refs) == len(hyps)
    n = s = d = i = 0
    for ref, hyp in zip(refs, hyps):
        a, b, c, e = errors(ref, hyp); n += a; s += b; d += c; i += e
    return {'wer': (s+d+i)/n, 'words': n, 'substitutions': s, 'deletions': d, 'insertions': i}

def sumstats(stats):
    total = {key: sum(s[key] for s in stats) for key in ['words', 'substitutions', 'deletions', 'insertions']}
    total['wer'] = sum(total[k] for k in ['substitutions', 'deletions', 'insertions']) / total['words']
    return total

def manifest(fold=None):
    name = f'{fold}_valid' if fold else 'jember_holdout'
    return pd.read_csv(ROOT / f'phase1_whisper_anchor/data/{name}.tsv', sep='\t', keep_default_na=False)

def prediction(run, checkpoint, name, language, frame):
    path = OUT / 'predictions' / run / checkpoint / f'{name}_{language}.csv'
    return aligned(path, frame)

def old(fold, checkpoint, frame, dataset='valid', language='indonesian', ratio=3):
    return aligned(ROOT / f'phase1_whisper_anchor/runs/{language}_x{ratio}_{fold}/{checkpoint}/{dataset}.csv', frame)

def save(path, data): path.write_text(json.dumps(data, indent=2)+'\n')

def checkpoint_comparison():
    comparison = {}; baseline = {}; old_jember = []
    for fold in FOLDS:
        frame = manifest(fold)
        baseline[fold] = corpus(frame.text.tolist(), list(map(collapse_single_runs, old(fold, 'epoch_1_full', frame))))
        held = manifest(); old_jember.append(corpus(held.text.tolist(), list(map(collapse_single_runs, old(fold, 'epoch_1_full', held, 'jember')))))
    baseline['aggregate'] = sumstats([baseline[f] for f in FOLDS]); baseline['jember'] = sumstats(old_jember)
    for checkpoint in CHECKPOINTS:
        record = {}; raw_stats = []; safe_stats = []; held_stats = []
        for fold in FOLDS:
            frame = manifest(fold); run = f'indonesian_x4_two_epochs_{fold}'
            raw = prediction(run, checkpoint, f'{fold}_valid', 'indonesian', frame)
            record[fold] = {'raw': corpus(frame.text.tolist(), raw), 'safe': corpus(frame.text.tolist(), list(map(collapse_single_runs, raw)))}
            raw_stats.append(record[fold]['raw']); safe_stats.append(record[fold]['safe'])
            held = manifest(); hp = prediction(run, checkpoint, 'jember_holdout', 'indonesian', held)
            held_stats.append(corpus(held.text.tolist(), list(map(collapse_single_runs, hp))))
        record['aggregate_raw'] = sumstats(raw_stats); record['aggregate_safe'] = sumstats(safe_stats); record['jember_fold_models'] = sumstats(held_stats)
        final = json.loads((OUT / 'predictions/final_indonesian_x4_two_epochs' / checkpoint / 'jember_holdout_indonesian.json').read_text())
        record['jember_final_model'] = final['safe']
        # This guard allows a small Fold A sacrifice like the user's motivating
        # R4 result, but rejects an obvious conversation or domain collapse.
        record['passes_sanity_guard'] = all(record[f]['safe']['wer'] <= baseline[f]['wer']+.005 for f in FOLDS) and record['jember_fold_models']['wer'] <= baseline['jember']['wer']+.02
        comparison[checkpoint] = record
    candidates = [c for c in CHECKPOINTS if comparison[c]['passes_sanity_guard']] or CHECKPOINTS
    selected = min(candidates, key=lambda c: (comparison[c]['aggregate_safe']['wer'], max(comparison[c][f]['safe']['wer'] for f in FOLDS)))
    data = {'selected_checkpoint': selected, 'old_R3_full_safe': baseline, 'checkpoints': comparison,
        'selection_note': 'Same two-epoch schedule in final and both conversation-disjoint folds. Competition labels only score OOF, never final weights.',
        'new_anchor_aggregate_gain': baseline['aggregate']['wer']-comparison[selected]['aggregate_safe']['wer']}
    save(OUT / 'checkpoint_selection.json', data)
    print(json.dumps(data, indent=2), flush=True)

def oracle(refs, systems):
    selected = []
    for row in zip(refs, *systems):
        ref, *options = row
        selected.append(min(options, key=lambda h: sum(errors(ref, h)[1:])))
    return corpus(refs, selected)

def decode_select(primary, alternate):
    # One fixed reference-free loop rescue. Ordinary disagreement keeps the
    # primary transcript; no per-clip language classifier is introduced.
    if repetition(primary) >= .35 and repetition(alternate) == 0 and 4 <= len(alternate.split()) < .8*len(primary.split()):
        return collapse_single_runs(alternate)
    return collapse_single_runs(primary)

def edit_count(a, b):
    return sum(not equal for _, _, equal in alignment(a.split(), b.split()))

def fusion_comparison():
    selection = json.loads((OUT / 'checkpoint_selection.json').read_text()); checkpoint = selection['selected_checkpoint']
    frames = {}; systems = {}; hold_systems = {}; refs = {}; heldrefs = manifest().text.tolist()
    for fold in FOLDS:
        frame = manifest(fold); frames[fold] = frame; refs[fold] = frame.text.tolist()
        sys = {}; hs = {}; held = manifest(); run = f'indonesian_x4_two_epochs_{fold}'
        for language in ['indonesian', 'javanese', 'auto']:
            raw = prediction(run, checkpoint, f'{fold}_valid', language, frame)
            sys[language+'_raw'] = raw; sys[language] = list(map(collapse_single_runs, raw))
            hs[language] = list(map(collapse_single_runs, prediction(run, checkpoint, 'jember_holdout', language, held)))
        for name, cp in [('R3_half','epoch_1_half'), ('R3_full','epoch_1_full')]:
            sys[name] = list(map(collapse_single_runs, old(fold, cp, frame)))
            hs[name] = list(map(collapse_single_runs, old(fold, cp, held, 'jember')))
        sys['MERaLiON'] = aligned(ROOT / f'phase2_meralion/results/pred_epoch_1_half_{fold}.csv', frame)
        cached = json.loads((ROOT / 'phase3_protected_fusion/results/conservative_final/jember_meralion_predictions.json').read_text())
        assert set(cached) == set(held.clip_id)
        assert all(cached[r.clip_id]['reference'] == r.text for r in held.itertuples())
        hs['MERaLiON'] = [cached[c]['transcript'] for c in held.clip_id]
        sys['old_conservative'] = aligned(ROOT / f'phase3_protected_fusion/results/conservative_final/dev_B_{fold}.csv', frame)
        hs['old_conservative'] = aligned(ROOT / f'phase3_protected_fusion/results/conservative_final/jember_B_{fold}.csv', held)
        systems[fold] = sys; hold_systems[fold] = hs
    def score(key, pool=systems, hold=False):
        values = {f: corpus(heldrefs if hold else refs[f], pool[f][key]) for f in FOLDS}
        values['aggregate'] = sumstats([values[f] for f in FOLDS]); return values
    individual = {k: score(k) for k in systems['fold_A']}
    jember = {k: score(k, hold_systems, True) for k in hold_systems['fold_A']}
    modes = ['indonesian', 'javanese', 'auto']
    eligible_modes = [m for m in modes if all(individual[m][f]['wer'] <= individual['indonesian'][f]['wer']+.003 for f in FOLDS) and jember[m]['aggregate']['wer'] <= jember['indonesian']['aggregate']['wer']+.015]
    mode = min(eligible_modes, key=lambda m: individual[m]['aggregate']['wer'])
    decode_pairs = {}
    for a, b in itertools.combinations(modes, 2):
        stats = {}; selections = {}
        for fold in FOLDS:
            stats[fold] = oracle(refs[fold], [systems[fold][a], systems[fold][b]])
            selected = [decode_select(x, y) for x, y in zip(systems[fold][a+'_raw'], systems[fold][b+'_raw'])]
            selections[fold] = corpus(refs[fold], selected)
        stats['aggregate'] = sumstats([stats[f] for f in FOLDS]); selections['aggregate'] = sumstats([selections[f] for f in FOLDS])
        decode_pairs[a+'+'+b] = {'clip_oracle': stats, 'conservative_primary_loop_rescue': selections}
    voters = ['R3_half', 'R3_full'] + [m for m in modes if m != mode]
    trials = []; predictions = {}; held_predictions = {}
    baseline = individual[mode]
    for voter in voters:
        for span, context in [(1,1), (2,1), (3,1), (2,2), (3,2)]:
            cfg = {'mode':'support','max_span':span,'context':context,'collapse_anchor':True,
                'operations':['substitution'],'strict_substitutions':True}
            name = f'{mode}_{voter}_span{span}_context{context}'
            stats = {}; heldstats = {}; changed = {}; preds = {}; hp = {}
            for fold in FOLDS:
                primary = systems[fold][mode+'_raw']
                pred = [fuse(a, m, cfg, third=t) for a,m,t in zip(primary, systems[fold]['MERaLiON'], systems[fold][voter])]
                # Label-free fusion is invariant to row order.
                reversed_pred = [fuse(a,m,cfg,third=t) for a,m,t in reversed(list(zip(primary, systems[fold]['MERaLiON'], systems[fold][voter])))]
                assert pred == reversed_pred[::-1]
                stats[fold] = corpus(refs[fold], pred); preds[fold] = pred
                changed[fold] = sum(edit_count(a,b) for a,b in zip(systems[fold][mode],pred))
                h = hold_systems[fold]
                hp[fold] = [fuse(a,m,cfg,third=t) for a,m,t in zip(h[mode],h['MERaLiON'],h[voter])]
                heldstats[fold] = corpus(heldrefs,hp[fold])
            stats['aggregate'] = sumstats([stats[f] for f in FOLDS]); heldstats['aggregate'] = sumstats([heldstats[f] for f in FOLDS])
            robust = all(stats[f]['wer'] <= baseline[f]['wer'] for f in FOLDS) and heldstats['aggregate']['wer'] <= jember[mode]['aggregate']['wer']+.003
            trials.append({'name':name,'config':cfg,'voter':voter,'metrics':stats,'jember':heldstats,'word_edits':changed,'passes_guard':robust})
            predictions[name] = preds; held_predictions[name] = hp
    eligible = [r for r in trials if r['passes_guard']]
    best = min(eligible or trials, key=lambda r:(r['metrics']['aggregate']['wer'],sum(r['word_edits'].values())))
    # Select a rule on A and apply B, then reverse; report it separately from
    # the final selected rule's descriptive OOF metric.
    loco = {}
    for heldout, training in [('fold_A','fold_B'), ('fold_B','fold_A')]:
        selected_trial = min(trials, key=lambda r:(r['metrics'][training]['wer'],r['word_edits'][training]))
        loco[heldout] = {'selected_on':training,'rule':selected_trial['name'],'metrics':selected_trial['metrics'][heldout]}
    loco['aggregate'] = sumstats([loco[f]['metrics'] for f in FOLDS])
    all_oracles = {}
    for voter in voters:
        pairs = {}; triples = {}
        for fold in FOLDS:
            pairs[fold] = oracle(refs[fold], [systems[fold][mode], systems[fold][voter]])
            triples[fold] = oracle(refs[fold], [systems[fold][mode], systems[fold]['MERaLiON'], systems[fold][voter]])
        pairs['aggregate'] = sumstats([pairs[f] for f in FOLDS]); triples['aggregate'] = sumstats([triples[f] for f in FOLDS])
        all_oracles[voter] = {'anchor_voter_pair':pairs,'anchor_meralion_voter_triple':triples}
    meralion_pair = {f: oracle(refs[f],[systems[f][mode],systems[f]['MERaLiON']]) for f in FOLDS}
    meralion_pair['aggregate'] = sumstats([meralion_pair[f] for f in FOLDS]); all_oracles['MERaLiON_pair'] = meralion_pair
    old_fusion = individual['old_conservative']
    deploy = best['passes_guard'] and best['metrics']['aggregate']['wer'] < old_fusion['aggregate']['wer'] and all(best['metrics'][f]['wer'] <= old_fusion[f]['wer']+.003 for f in FOLDS)
    data = {'checkpoint':checkpoint,'anchor_language':mode,'individual':individual,'jember_fold_models':jember,
        'same_checkpoint_decode_pairs':decode_pairs,'ensemble_oracles':all_oracles,'selected_fusion':best,'all_fusion_trials':trials,
        'conversation_cross_selected_rules':loco,'deploy_new_fusion':deploy,
        'R3_continuation_decision':'Exact optimizer continuation is unsupported by existing R3 artifacts. R4 and decode evidence are evaluated first; a weights-only restart is not an exact continuation.',
        'notes':['Oracles choose entire clip hypotheses using references and are analysis-only lower bounds.',
                 'Jember fusion uses cached final adapted MERaLiON; it is a domain sanity diagnostic, not OOF across all three systems.',
                 'Final deployment choice compares descriptive OOF against the previously successful conservative submission.']}
    save(OUT / 'fusion_selection.json',data)
    for fold in FOLDS:
        frame=frames[fold]
        pd.DataFrame({'clip_id':frame.clip_id,'reference':frame.text,'transcript':predictions[best['name']][fold]}).to_csv(OUT/f'selected_new_fusion_{fold}.csv',index=False)
    print(json.dumps({'checkpoint':checkpoint,'mode':mode,'individual':individual,'best_fusion':best,'deploy_new_fusion':deploy,'cross_selected':loco},indent=2),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',choices=['checkpoints','fusion'],required=True);a=p.parse_args()
    if a.step=='checkpoints':checkpoint_comparison()
    else:fusion_comparison()

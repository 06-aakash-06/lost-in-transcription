#!/usr/bin/env python3
"""Replay cached dev hypotheses under labelled and all-mixed routing."""

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))
from ensemble_runtime import (  # noqa: E402
    Hypothesis, _align_token_sequences, alignment_tokens, edit_distance, mean_pairwise_disagreement,
    select_consensus,
)


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def predictions(*filenames):
    result = {}
    for filename in filenames:
        path = ROOT / "experiments" / filename
        if path.exists():
            result.update({row["clip_id"]: row["transcript"] for row in rows(path)})
    return result


def errors(ref, pred):
    return edit_distance(alignment_tokens(ref), alignment_tokens(pred))


def score(dev, output):
    selected = [r for r in dev if r["clip_id"] in output]
    return sum(errors(r["text"], output[r["clip_id"]]) for r in selected) / sum(len(alignment_tokens(r["text"])) for r in selected)


def main():
    dev = rows(ROOT / "processed/v2/dev.tsv")
    config = json.loads((ROOT / "submission_src/ensemble_config.json").read_text())
    calibration = json.loads((ROOT / "submission_src/calibration.json").read_text())
    candidates = {
        "qwen_id": predictions("dev_qwen_id_full.csv", *(("dev_qwen_norm_full.csv",) if not os.environ.get("LIT_USE_RAW_QWEN") else ())),
        "meralion_sea": predictions("dev_meralion_full.csv"),
        "whisper_turbo_jw": predictions("dev_whisper_turbo_jw_current_full.csv", "dev_whisper_turbo_jw_blind_missing.csv"),
        "whisper_large_id": predictions("dev_whisper_large_id_ind.csv"),
        "buzz_javanese": predictions("dev_buzz_jv_full.csv", "dev_buzz_javanese_blind_missing.csv"),
        "buzz_indonesian": predictions("dev_buzz_id_ind.csv"),
    }
    specs = {x["name"]: x for x in config["decodes"] + config["adaptive_decodes"]}

    def available(row, blind):
        clip = row["clip_id"]
        names = ["qwen_id", "meralion_sea"]
        if blind or row["language"] != "ind":
            names.append("whisper_turbo_jw")
        else:
            names.append("whisper_large_id")
        if any(clip not in candidates[name] for name in names):
            return None
        hypotheses = [Hypothesis(name=n, text=candidates[n][clip], backend="cached", prior=specs[n]["prior"]) for n in names]
        if mean_pairwise_disagreement(hypotheses) >= config["adaptive_trigger"]["min_disagreement"] and (blind or row["language"] != "ind") and clip in candidates["buzz_javanese"]:
            hypotheses.append(Hypothesis(name="buzz_javanese", text=candidates["buzz_javanese"][clip], backend="cached", prior=1.5))
        return hypotheses

    def fuse(row, blind, cal=calibration):
        hypotheses = available(row, blind)
        if hypotheses is None:
            return None
        return select_consensus(hypotheses, calibration=cal, language="javind" if blind else row["language"], duration_s=float(row["duration_s"]))[0]

    labelled = {r["clip_id"]: value for r in dev if (value := fuse(r, False)) is not None}
    blind = {r["clip_id"]: value for r in dev if (value := fuse(r, True)) is not None}
    print("coverage", {"labelled": len(labelled), "blind": len(blind), "candidate_counts": {k: len(v) for k,v in candidates.items()}})
    common = [r for r in dev if r["clip_id"] in labelled and r["clip_id"] in blind]
    for label, subset in [("all_common",common),("javind",[r for r in common if r["language"]=="javind"]),("ind",[r for r in common if r["language"]=="ind"])]:
        if subset:
            a,b=score(subset,labelled),score(subset,blind)
            print("route",label,len(subset),round(a,6),round(b,6),round(b-a,6))
    all_blind = [r for r in dev if r["clip_id"] in blind]
    print("individual and oracle: name, coverage, WER on coverage")
    for name, pred in candidates.items():
        subset=[r for r in dev if r["clip_id"] in pred]
        if subset:
            print("individual", name,len(subset),round(score(subset,pred),6),
                  {language: round(score([r for r in subset if r["language"] == language],pred),6)
                   for language in ("javind","ind") if any(r["language"] == language for r in subset)})
    print("adaptive_triggered",sum(any(h.name == "buzz_javanese" for h in available(r, True)) for r in all_blind),"of",len(all_blind))
    for label, subset in [("all",all_blind),("javind",[r for r in all_blind if r["language"]=="javind"]),("ind",[r for r in all_blind if r["language"]=="ind"])]:
        if not subset: continue
        oracle={r["clip_id"]:min((candidates[n][r["clip_id"]] for n in ("qwen_id","meralion_sea","whisper_turbo_jw","buzz_javanese") if r["clip_id"] in candidates[n]),key=lambda p:errors(r["text"],p)) for r in subset}
        print("oracle",label,len(subset),round(score(subset,oracle),6),"rover",round(score(subset,blind),6))
    no_buzz = {}
    medoid = {}
    triggered_oracle = {}
    medoid_cal = dict(calibration)
    medoid_cal["selection_by_language"] = {"javind": "weighted_medoid"}
    for r in all_blind:
        clip = r["clip_id"]
        hyps = available(r, True)
        no_buzz[clip] = select_consensus([h for h in hyps if h.name != "buzz_javanese"], calibration=calibration, language="javind")[0]
        medoid[clip] = select_consensus(hyps, calibration=medoid_cal, language="javind")[0]
        triggered_oracle[clip] = min((h.text for h in hyps), key=lambda p: errors(r["text"],p))
    print("ablation", "no_buzz_rover",round(score(all_blind,no_buzz),6),"medoid",round(score(all_blind,medoid),6),"triggered_oracle",round(score(all_blind,triggered_oracle),6))
    ind_rows = [r for r in all_blind if r["language"] == "ind" and r["clip_id"] in candidates["buzz_indonesian"]]
    if ind_rows:
        with_id = {r["clip_id"]: min(
            [h.text for h in available(r, True)] + [candidates["buzz_indonesian"][r["clip_id"]]],
            key=lambda p: errors(r["text"], p),
        ) for r in ind_rows}
        print("buzz_id_oracle_ind",len(ind_rows),round(score(ind_rows,triggered_oracle),6),round(score(ind_rows,with_id),6))
    for convo in sorted({r["convo_id"] for r in all_blind}):
        subset = [r for r in all_blind if r["convo_id"] == convo]
        print("ablation_convo",convo,len(subset),"current",round(score(subset,blind),6),"no_buzz",round(score(subset,no_buzz),6))
    failure=Counter()
    for r in all_blind:
        clip=r["clip_id"]; ref=r["text"]; hyps=available(r,True)
        rover_error=errors(ref,blind[clip]); best=min(errors(ref,candidates[h.name][clip]) for h in hyps)
        failure["rover_worse_than_best"]+=rover_error>best
        failure["rover_better_than_best"]+=rover_error<best
        if any(h.name=="buzz_javanese" for h in hyps):
            buzz=errors(ref,candidates["buzz_javanese"][clip])
            if buzz < min(errors(ref,candidates[n][clip]) for n in ("qwen_id","meralion_sea","whisper_turbo_jw")):
                failure["buzz_best"]+=1
                failure["buzz_best_rover_worse"]+=rover_error>buzz
    print("failures",dict(failure))
    confusions=Counter()
    for r in all_blind:
        for ref_token,pred_token in _align_token_sequences(alignment_tokens(r["text"]),alignment_tokens(blind[r["clip_id"]])):
            if ref_token and pred_token and ref_token != pred_token:
                confusions[(ref_token,pred_token)]+=1
    print("top_rover_substitutions",confusions.most_common(15))
    # One small orthography ablation suggested by the largest substitution.
    gak_rule = {clip: re.sub(r"\bnggak\b", "gak", value) for clip, value in blind.items()}
    print("gak_rule_all", round(score(all_blind, gak_rule),6))
    for convo in sorted({r["convo_id"] for r in all_blind}):
        subset = [r for r in all_blind if r["convo_id"] == convo]
        print("gak_rule_convo", convo, round(score(subset,blind),6), round(score(subset,gak_rule),6))
    for a,b in (("qwen_id","meralion_sea"),("qwen_id","whisper_turbo_jw"),("qwen_id","buzz_javanese"),("meralion_sea","whisper_turbo_jw"),("meralion_sea","buzz_javanese"),("whisper_turbo_jw","buzz_javanese")):
        pairs=[r for r in dev if r["clip_id"] in candidates[a] and r["clip_id"] in candidates[b]]
        worddist=sum(errors(candidates[a][r["clip_id"]],candidates[b][r["clip_id"]]) for r in pairs)
        words=sum(max(len(alignment_tokens(candidates[a][r["clip_id"]])),len(alignment_tokens(candidates[b][r["clip_id"]]))) for r in pairs)
        print("pair",a,b,len(pairs),round(worddist/words,5))

    # Fixed small grid; each test conversation is unseen during parameter choice.
    variants=[]
    for buzz in (0.25,0.5,1.0,1.5,2.0):
        for blank in (1.0,1.25,1.5):
            cal=json.loads(json.dumps(calibration))
            cal["language_multipliers"]["javind"]={"buzz_javanese":buzz}
            cal["rover"]["blank_factor"]=blank
            outputs={r["clip_id"]:value for r in all_blind if (value:=fuse(r,True,cal)) is not None}
            variants.append((buzz,blank,outputs))
    convos=sorted({r["convo_id"] for r in all_blind})
    held_errors=held_words=0
    for held in convos:
        train=[r for r in all_blind if r["convo_id"]!=held]
        test=[r for r in all_blind if r["convo_id"]==held]
        best=min(variants,key=lambda v:score(train,v[2]))
        err=sum(errors(r["text"],best[2][r["clip_id"]]) for r in test)
        words=sum(len(alignment_tokens(r["text"])) for r in test)
        held_errors+=err;held_words+=words
        print("loco",held,"train",len(train),"test",len(test),"chosen",best[:2],"train_wer",round(score(train,best[2]),6),"test_wer",round(err/words,6),"current_test",round(score(test,blind),6))
    print("loco_weighted",round(held_errors/held_words,6),"full_dev_tuned",round(min(score(all_blind,v[2]) for v in variants),6))


if __name__ == "__main__":
    main()

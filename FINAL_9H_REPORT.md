# Final nine-hour ASR report

**READY TO SUBMIT: YES.** Recommended archive: `artifacts/FINAL_MAX_WER_REDUCTION_SUBMISSION.zip`.

**Selected final system**

Whisper Large-v3 full 1.0-epoch adapter (full contribution 1.00), Indonesian, greedy, native long-form. Public frozen BF16 base plus exact FP32 LoRA residuals, r16/alpha32, 512 attention/MLP projections. Final training used all 372 competition clips and 1309 Jember chunks, competition sampling ratio 4. Checkpoint: `overnight_attack/runs/final_large_v3/epoch_1_full` (batch 2797/2797, optimizer step 175).

Turbo R3-full and adapted MERaLiON support only exact shared word replacements at unambiguous optimal alignments. Ordinary fusion uses substitutions only; no single alternative overrides Large-v3. Turbo R3-half is excluded. One conservative phrase-loop safeguard is applied before voting, with no further edits after cleanup.

No learned edit gate or N-best rescoring is required in the selected package.

**Large-v3 checkpoint results**

Full contribution is the coefficient of the 1.0-epoch adapter. Soups average actual LoRA deltas by scaled factor concatenation, rather than averaging A/B factors. Same frozen base, architecture and precision are verified. Jember below is the mean of two fold models on the same 162 held-out clips, not 324 independent examples.

| Full contribution | A | B | Aggregate | Jember | Worst fold |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.216886 | 0.190101 | 0.211438 | 0.268351 | 0.216886 |
| 0.25 | 0.211286 | 0.184758 | 0.205891 | 0.259755 | 0.211286 |
| 0.50 | 0.205471 | 0.180540 | 0.200400 | 0.253476 | 0.205471 |
| 0.75 | 0.204824 | 0.179415 | 0.199657 | 0.252429 | 0.204824 |
| 1.00 | 0.205542 | 0.178009 | 0.199943 | 0.250635 | 0.205542 |

Full was selected after all five fold comparisons: the 75% soup has a one-error fused aggregate advantage, but full has better Fold B and Jember results and a simpler r16 delta. Half underfits both folds. The final OOD warning was dominated by one multiword repetition failure, rather than evidence that half is safer.

**Final all-data Jember check**

| Candidate | Standalone | Consensus |
|---|---:|---:|
| Half | 0.268501 | 0.257737 |
| 75% full soup | 0.247720 | 0.242488 |
| Full | 0.246524 | 0.242039 |
| Old conservative Turbo system | — | 0.242189 |

Full without the phrase guard scored 0.265660 standalone / 0.261175 fused. The fixed guard reduces those to 0.246524 / 0.242039. It removes 168 loop words on one Jember clip and changes zero predictions on either OOF fold or the fold-model Jember checks, across all five coefficients. No IDs or references are inference features.

Guard: preserve the existing single-word rule; additionally trim a 2–6 word phrase only when it repeats at least five times, occupies at least 20 words and at least 35% of the transcript. Retain two cycles. One fixed rule was tested; no threshold grid or lexical exception was fitted.

**Decoding**

Forced Indonesian and auto-language greedy produced identical transcripts on both folds, with no additional oracle headroom. Indonesian beam 3 scored 0.180259 on B (greedy 0.178009), and 0.176322 after consensus (greedy 0.172666). Their B pair oracle was 0.173791. Beam 3 failed the continue gate; A remained a preserved 96/294-clip partial cache at the one-hour cap. It is not deployed or presented as a two-fold result.

**Fusion results**

| Candidate | A | B | Aggregate | Fold-model Jember | S / I / D |
|---|---:|---:|---:|---:|---|
| Large standalone | 0.205542 | 0.178009 | 0.199943 | 0.250635 | 0 / 0 / 0 |
| Old conservative | 0.209778 | 0.185321 | 0.204804 | 0.244356 | 134 / 0 / 0 |
| A Turbo+MER strict | 0.200732 | 0.174916 | 0.195482 | 0.248393 | 228 / 0 / 0 |
| A Turbo+MER short indels | 0.200804 | 0.176884 | 0.195939 | 0.248318 | 210 / 14 / 19 |
| B Turbo half+full strict | 0.205686 | 0.178571 | 0.200172 | 0.249813 | 603 / 0 / 0 |
| C unambiguous word majority | 0.201307 | 0.177447 | 0.196454 | 0.248318 | 183 / 0 / 0 |
| C unambiguous boundary majority | 0.199368 | 0.172666 | 0.193938 | 0.246001 | 277 / 0 / 0 |

Selected deterministic C makes 277 substitutions in 267 spans across 173/372 clips, 0.7446 corrected words per clip. No ordinary insertions/deletions. Final-model Jember: 102 substitutions in 100 spans; the separate catastrophic-loop guard removes 168 words on one clip. The 75% soup C scores 0.193880 / Jember 0.246973; 50% soup C scores 0.194452 / Jember 0.247870. Neither earns selection over full.

Clip-oracle diagnostics: Large+Turbo 0.183929; Large+MERaLiON 0.178725; triple 0.170603. These are reference-selected analysis bounds, never deployable predictions. Deterministic C converts 105 OOF word errors relative to Large-v3 alone. MERaLiON and Turbo are both needed for the tested two-supporter correction rule.

**Confidence and learned gate**

| Rule | A | B | Aggregate | Jember |
|---|---:|---:|---:|---:|
| C consensus | 0.199368 | 0.172666 | 0.193938 | 0.246001 |
| D low confidence 0.6 | 0.204035 | 0.178009 | 0.198742 | 0.247944 |
| D low confidence 0.8 | 0.201091 | 0.175197 | 0.195825 | 0.247421 |
| D low confidence 0.95 | 0.199727 | 0.172385 | 0.194166 | 0.246748 |
| Cross-fold logistic | 0.200876 | 0.176040 | 0.195825 | 0.247571 |

One fixed logistic recipe was trained on A→B and B→A using only acoustic/hypothesis features. It fails both directions and is rejected. Confidence thresholds also fail the both-fold robustness test. No sklearn/PEFT dependency or gate coefficients are required in the ZIP.

**One optional experiment: N-best**

Fixed trigram candidate rescoring, trained separately from each fold’s allowed training transcripts. Only short clips where consensus proposes edits and mean Large-v3 word probability is below 0.8 are queried. Beam-3 top three plus greedy; acoustic score + 0.1×LM, acceptance margin 0.02, length within 15%, no increased repetition. No free rewriting or hidden data.

| Variant | A | B | Aggregate | Jember |
|---|---:|---:|---:|---:|
| Baseline C | 0.199368 | 0.172666 | 0.193938 | 0.246001 |
| N-best | 0.198937 | 0.171822 | 0.193423 | 0.244506 |

Continue gate: FAIL. Requires gains on both folds, at least 0.0015 aggregate improvement and Jember worsening no more than 0.001. N-best improves both folds and Jember, but converts only nine competition word errors, below the predeclared material-gain gate. It makes 305 substitutions, two insertions and eight deletions versus C’s 277 substitutions and no insertions/deletions. The simpler correction policy is retained. Checkpoint sequence ensemble and longer training were not run.

**Old hidden system OOF vs new final OOF**

| System | A | B | Aggregate | Worst fold |
|---|---:|---:|---:|---:|
| Old conservative (hidden 0.2712) | 0.209778 | 0.185321 | 0.204804 | 0.209778 |
| Selected final | 0.199368 | 0.172666 | 0.193938 | 0.199368 |

Selection prioritizes consistent gains on both conversations, model capacity, stable final-model Jember and inference reliability. New hidden WER is unmeasured; no improvement is fabricated.

**Final ZIP and validation**

- Recommended: `artifacts/FINAL_MAX_WER_REDUCTION_SUBMISSION.zip`
- SHA256: `93f3d6e026057830d2be8e1438dfa26425088392d62b426670bc88549e538ed8`
- ZIP: 13,089,612,505 bytes (13.090 GB); bundled model assets: 13,098,426,152 bytes (13.098 GB).
- Smoke: **PASS**; MPS end-to-end runs 51.3s, 49.8s.
- Backup: `artifacts/FINAL_LARGEV3_STANDALONE_BACKUP.zip`
- SHA256: `b7c2929ecfe3add896269f4f58753ad1bbc63d402b63db1ae47cee380a8ed97c`
- ZIP: 3,203,958,580 bytes (3.204 GB); bundled model assets: 3,208,279,250 bytes (3.208 GB).
- Smoke: **PASS**; MPS end-to-end runs 20.6s, 21.0s.

Both exact archives pass CRC/SHA and complete asset/shard-map checks. Exact model bytes are verified against archive entries; local hardlinks only save extraction disk space. Actual main.py runs with sockets blocked, empty HF caches, PEFT/sklearn unavailable, a 21.597875-second clip and a 37.6535-second clip. CSV schema, quoted filenames, row order, no NaNs/blanks, repeated byte-identical output, reordering independence, native long-form tail coverage and silent inference pass. CUDA-first selection, CSV rejection, memory retry, consensus invariants and phrase-guard tests pass.

**Expected A100 runtime:** approximately 60–100 minutes. The exact same two audio files are used in the old/new MPS benchmark: old inference 30.9/29.9 seconds; new summed model loading/decoding approximately 44–46 seconds. Scaling the user-verified old 31-minute A100 run gives approximately 46 minutes before a substantial safety margin. Sequential residency and CUDA batch 8 are retained. New A100 timing is not measured: CUDA is unavailable locally. Local component timings and hardware evidence are saved; the hard limit is 120 minutes. CPU fallback is implemented but not timed.

Extra case check: the packaged single-clip loader and original PEFT model produce identical 52-word predictions on the cached repetition case; the original batch-4 cache had 174 loop words. Single-clip packaged inference produces nonrepeating speech, and a recorded-loop replay at the ASR boundary verifies the exact archived entrypoint applies the guard. OOF/OOD tables remain cached batched diagnostics; batch/backend numerical differences can alter recognition. The replay is an integration test, not a new ASR accuracy measurement.

**Preservation:** all five pre-existing ZIPs retain their SHA256 hashes; all 627 overnight files and the Ratio-4 interrupted adapter/state are unchanged. Ratio-4 remains stopped at batch 218/2797. Its optimizer, scheduler and partial accumulated gradients remain at the saved resume path. The earlier provisional backup is preserved byte-for-byte under `artifacts/preserved_9h_builds/`. No working prior checkpoint or submission was overwritten.

Ratio-4 resume checkpoint: `phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218`. Resume instructions: `phase4_stronger_anchor/RESUME.md`. No next Ratio-4 stage or evaluation was started.

Evidence, prediction caches, logs, portable source and archive inventories: `final_9h_attack/`. Final all-data optimizer/scheduler state remains under `overnight_attack/runs/final_large_v3/epoch_1_full/`. No new ASR architecture was trained, MERaLiON was not retrained, and no final competition submission was uploaded.

**READY TO SUBMIT: YES.**

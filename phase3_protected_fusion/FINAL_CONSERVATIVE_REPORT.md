# Final conservative submission

Selected: Variant B. Exact MERaLiON and R3-half agreement, strict word-count-preserving substitutions of up to three words, at least one unchanged word on both sides. Existing catastrophic repetition rescue remains enabled.

| System | Fold A | Fold B | Aggregate |
|---|---:|---:|---:|
| R3-full | 0.225860 | 0.188976 | 0.218359 |
| R3-full-safe | 0.213296 | 0.188976 | 0.208350 |
| A | 0.208342 | 0.185321 | 0.203660 |
| B | 0.209778 | 0.185321 | 0.204804 |
| C_0.8 | 0.210281 | 0.187289 | 0.205605 |

The raw R3-full baseline includes three looping clips. Its shared-safe baseline is shown separately. Applied correction counts exclude common repetition rescue (196 deleted repeated words).

| Variant / fold | Correction spans | Word edits | S | I | D |
|---|---:|---:|---:|---:|---:|
| A / fold_A | 124 | 144 | 114 | 15 | 15 |
| A / fold_B | 42 | 51 | 43 | 6 | 2 |
| A / aggregate | 166 | 195 | 157 | 21 | 17 |
| B / fold_A | 95 | 98 | 98 | 0 | 0 |
| B / fold_B | 34 | 36 | 36 | 0 | 0 |
| B / aggregate | 129 | 134 | 134 | 0 | 0 |
| C_0.8 / fold_A | 76 | 89 | 68 | 10 | 11 |
| C_0.8 / fold_B | 28 | 33 | 29 | 3 | 1 |
| C_0.8 / aggregate | 104 | 122 | 97 | 13 | 12 |

C geometric confidence thresholds .45, .65, .80 were the entire bounded search; .80 was frozen using dev only. Missing confidence, including >30s clips, cannot approve C corrections.

## Jember sanity

162 session-disjoint holdout clips; average of two saved Whisper fold checkpoints, paired with the same final adapted MERaLiON predictions. No Jember parameter selection.

| System | WER |
|---|---:|
| R3-full | 0.245552 |
| R3-half | 0.275004 |
| MERaLiON | 0.391837 |
| A | 0.244655 |
| B | 0.244356 |
| C_0.8 | 0.244207 |

B improves both OOF folds over the shared-safe anchor, matches A on Fold B, gives up only .001144 aggregate WER, uses 61 fewer word edits and no fusion indels, and shows no Jember collapse. C regresses both folds relative to A.

No new ASR model training, no dev prediction regeneration, and no changes to model weights or MERaLiON tokenizer. Earlier ZIPs remain intact.

Archive: `artifacts/FINAL_SUBMISSION_CONSERVATIVE.zip`

SHA256: `6001974161873a9d1cd1313cd15960aa319717f0b9d98226f9aeb11f4cbad838`

Smoke PASS: exact extracted ZIP, two separate inference runs with byte-identical CSV output, network blocked, all HF caches empty, one normal clip and one 37.6535s clip, final eight words of long clip retained, correct schema/order/quoting, no NaNs or empty transcripts. Both runs used MPS; CUDA-first selection is unit-tested, but no CUDA container was executed locally. All 14 unit tests passed. ZIP CRC and SHA256 verified.

READY TO SUBMIT: YES

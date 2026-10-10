# Overnight report

Completed Large-v3 two-fold training/validation and final all-data 0.5/1.0 checkpoints. No submission ZIP was built.

## LARGE-V3
MPS passed: BF16, full encoder/decoder LoRA, real finite-loss/gradient smoke, approximately 6 GB driver memory. All fold training and decoding completed on their first attempts.
Recipe: `openai/whisper-large-v3`, Indonesian, Ratio-4, r16/alpha32/dropout0.05, microbatch1/accum16, gradient checkpointing, AdamW LR3e-4, one weighted epoch, seed1337. Existing reference-style pipeline and official WER scorer; exact conversation-disjoint folds.
Public base cached once, pinned revision `06f233fe06e710322aca913c1bc4249a0d71fce1`; SHA-256 verified. License compatibility verified against the [HF model card](https://huggingface.co/openai/whisper-large-v3) and competition brief section13; evidence in `results/license_verification.json`.

| Checkpoint | Fold A | Fold B | Aggregate | Jember mean |
|---|---:|---:|---:|---:|
| epoch_1_half | 0.216886 | 0.190101 | 0.211438 | 0.268351 |
| epoch_1_full | 0.205542 | 0.178009 | 0.199943 | 0.250635 |
| Current conservative fusion | 0.209778 | 0.185321 | 0.204804 | — |

Selected full: worst fold 0.205542. Beats the current conservative fusion on both folds.
Full pair oracle: Turbo R3-full 0.183929; MERaLiON 0.178725; triple 0.170603. These reference-selected whole-clip bounds are analysis only.
Jember mean is 0.250635 versus Turbo 0.245552. OOD improvement is not established.
Fold checkpoints: `overnight_attack/runs/large_v3_fold_A` and `overnight_attack/runs/large_v3_fold_B` (each contains half/full adapters and optimizer/scheduler/RNG/sampler state).

## FINAL LARGE-V3
Complete: all 372 competition clips +1309 Jember chunks, 2797 weighted microbatches, 175 optimizer steps. The 10-hour cutoff preserved batch2676 with four accumulated gradients; the remaining121 batches resumed successfully.
0.5 checkpoint: `overnight_attack/runs/final_large_v3/epoch_1_half`.
1.0 checkpoint/latest usable state: `overnight_attack/runs/final_large_v3/epoch_1_full`.
Standalone model: `overnight_attack/models/final_large_v3_full`.
Merge verified on CPU FP32: max logit difference 0.00002766, identical sample token IDs. Frozen BF16 training-base values preserved before FP32 merging. A100 runtime remains unmeasured.
Final all-data Jember WER: 0.265660 on 162 session-disjoint clips. This OOD regression needs attention before final system selection.
Resume instructions: [RESUME.md](<overnight_attack/RESUME.md>).

## RATIO-4 FINAL
Fallback not needed. Original Turbo partial checkpoint at epoch1/batch218 is preserved; its resume guide is `phase4_stronger_anchor/RESUME.md`.

## MODEL SOUP
Not tested; the window went to the successful capacity branch and final training. Compatible delta-soup code and equivalence/precision guards are saved.

## ADAPTIVE DECODING
Not tested; fallback not needed.

## N-BEST
Not tested; fallback not needed.

## BEST NEW DIRECTION
Whisper Large-v3, Indonesian Ratio-4, full encoder/decoder r16 LoRA, epoch1_full. Stronger two-fold capacity candidate with substantial oracle contribution; retain the working conservative submission while checking OOD behavior.

## NEXT ACTION
Validate one fixed protected Large-v3 + Turbo R3-full + MERaLiON system against the current conservative fusion on both cached OOF folds and Jember.

All adapters, resumable states, prediction caches, audit results, logs and code are retained. SHA-256 inventory: `SHA256SUMS.txt`. Existing working ZIPs and final checkpoints are preserved.

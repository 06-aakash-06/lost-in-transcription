# Phase 3 final leaderboard system

## Verified OOF results

Official Phase 1 scorer/normalizer; 294 Fold A clips and 78 Fold B clips. Unique clip IDs, exact ID sets, manifest order and reference strings match across all hypotheses. No predictions were regenerated. Confidence extraction teacher-forced only saved model hypotheses against held-out audio using the corresponding held-out Whisper adapter.

| System | Fold A | Fold B | Aggregate |
|---|---:|---:|---:|
| Whisper ratio 3, half epoch | 0.223634 | 0.204162 | 0.219674 |
| MERaLiON adapted | 0.226578 | 0.220754 | 0.225393 |
| Previous simple fusion | 0.218465 | 0.215129 | 0.217787 |
| Candidate A: protected two-model | **0.215665** | **0.203881** | **0.213269** |
| Candidate B: protected + full-epoch ratio-3 voter | **0.212434** | **0.197132** | **0.209322** |
| Ratio-4 hybrid, OOF analysis only | 0.209347 | 0.196288 | 0.206691 |
| Auto hybrid, OOF analysis only | 0.213942 | 0.203600 | 0.211839 |

| Best-per-clip oracle | Fold A | Fold B | Aggregate |
|---|---:|---:|---:|
| Whisper + MERaLiON | 0.189245 | 0.171822 | 0.185702 |
| + full-epoch ratio-3 | 0.180846 | 0.163948 | 0.177409 |
| + ratio-4 | 0.180487 | 0.162823 | 0.176894 |
| + auto | 0.186158 | 0.167042 | 0.182271 |

Oracle is reference-assisted diagnostic headroom, never a deployment rule. Ratio-4 lacks an already-trained full-data final checkpoint. No new training was done; its OOF advantage is not packaged. Auto is weaker than the available full-epoch voter. No hidden metadata routing, language classifier, pseudo-labels or test-set statistics are used.

## Selected corrections

Candidate A: at most two-word substitutions, geometric anchor word probability below .65, at least one unchanged word on both sides. No insertion/deletion rules. Candidate B adds exact support from the full-epoch Whisper voter for substitutions/insertions/deletions of at most three words, retaining the same confidence rule for at most two-word substitutions. Both preserve anchor casing/surface outside replacements.

MERaLiON hypotheses with clear loops covering at least 20% of words are vetoed. Anchor runs of five or more identical words collapse to two, independently validated on both folds. No broad repeat collapse or full-transcript MER replacement is deployed.

| Candidate | Fold | Changed clips | Improved clips | Damaged clips | Net errors removed vs anchor |
|---|---|---:|---:|---:|---:|
| A | A | 201 | 99 | 43 | 111 |
| A | B | 43 | 16 | 15 | 1 |
| B | A | 236 | 117 | 55 | 156 |
| B | B | 56 | 30 | 16 | 25 |

Candidate A's Fold B improvement is one word and should not be treated as a robust statistical gain. Candidate B improves both folds over A and reduces aggregate WER by .003946, enough to justify one additional pass through the existing Whisper model. Larger unsupported spans regress Fold B and are rejected. The search consisted of 46 readable configurations, saved with exact S/D/I counts in results/fusion_*.csv; no hundreds of microscopic threshold trials.

## Models and inference

- Anchor: unchanged Phase 1 `models/final_merged`, Indonesian ratio 3, half epoch.
- Corrective: unchanged Phase 2 `models/final_merged`, selected decoder-LoRA half epoch.
- Candidate B voter: `phase3_protected_fusion/models/whisper_full`, merged from the already-trained `phase1_whisper_anchor/runs/final_indonesian_x3/epoch_1_full` adapter. No training. Merge validation: identical sample token IDs, maximum absolute logit difference 0.00002670 on MPS.
- Native Whisper long form and MERaLiON native multi-chunk processor; confidence unavailable for >30s clips rather than scoring a truncated window.
- CUDA-first, BF16, batch 8; MPS batch 1, Whisper FP32 / MERaLiON FP16. Sequential model residency. Greedy decoding. Inference only, local merged weights, offline flags before model imports.

## Runtime

User-measured Phase 1 A100 inference: 278.2 seconds / 2118 clips. Candidate A adds one teacher-forced encoder/decoder confidence pass and MERaLiON. Candidate B adds a second Whisper inference pass.

Conservative planning projection: **A about 35–65 minutes; B about 40–70 minutes** for 2118 clips at CUDA batch 8. This budgets MERaLiON at 5–12 times the measured Whisper wall time, confidence at up to one extra Whisper pass, and model loading/IO overhead. This is an engineering allowance, not an A100 measurement or a proven upper bound. No CUDA profiling is possible on this Mac. Both designs target substantial margin below two hours; actual competition execution remains the runtime gate.

Exact ZIP local MPS smoke (2 clips, one 37.6535s, one 21.5979s): A subprocess 38.84s (entrypoint 30.7s); B subprocess 45.98s (entrypoint 38.7s). These are smoke measurements, not a full MPS or A100 projection. The long prediction includes the reference's final eight words, demonstrating no 30-second tail truncation in this fixture.

## ZIPs and verification

Candidate A:
`artifacts/phase3_protected_fusion_submission.zip` (9,885,665,334 bytes)

SHA256:
`187e17861ba77730cba39c16c69db1fc34babfdcabad41bb1a622b19ee746408`

Candidate B:
`artifacts/phase3_three_model_submission.zip` (13,122,558,495 bytes)

SHA256:
`b866c65efe4583330bd9555e8c4033aa4b80cb3b87bad31b6a9fe60a1c72fce6`

Both exact ZIPs: CRC passed; root main.py; complete standalone model/processor/custom code/notices; extracted-code/config match current source; inference ran with socket connections blocked and empty HF cache; columns/count/order/quoting/nonempty strings/no NaNs passed. The requested row order was deliberately reversed. No PEFT loading or retraining is needed. Mixed-length MERaLiON batch-2 predictions match separate predictions exactly on MPS. 11 unit tests passed. Candidate B independent rebuild has identical SHA256. No archives have been uploaded by this implementation.

Phase 1 ZIP SHA256 before and after:
`6255239690081b884be4454b2e0eef0d187d08e89c73c8c98cefe90bdf0fec38`

## Remaining deployment gate

Official offline CUDA/Python 3.12 Docker execution is unavailable: this Mac has no NVIDIA GPU or running Docker daemon. Local smoke uses Python 3.14.7; official runtime Transformers 4.57.6 matches local. Existing MERaLiON tokenizer conventions were preserved, as validated in Phase 2. All required weights/assets are bundled.

## Submission recommendation

**Submission #2: Candidate B**, the deployable system with the best measured aggregate and both-fold improvement. **Submission #3: reserve Candidate A** until Candidate B's hidden score/runtime is known. Do not spend the last submission automatically. Preserve Submission #1 and its ZIP.

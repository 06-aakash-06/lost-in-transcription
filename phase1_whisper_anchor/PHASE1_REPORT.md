# Phase 1 report

## Repository findings

Reused the repository's prepared Jember chunks, session-disjoint holdout, competition dev clips, conservative spelling mappings, official scorer normalization, and long-audio utilities where applicable. The legacy ensemble remains separate; deployment risks are recorded in [LEGACY_DEPLOYMENT_AUDIT.md](LEGACY_DEPLOYMENT_AUDIT.md). Phase 1 uses one Whisper Turbo model and does not route by metadata.

## Legacy deployment findings

Confirmed risks: the canonical 18 GB archive is stale against the current runtime/config, can route from test language metadata, may truncate clips over 30 seconds, and reads `submission_format.csv` before `test_metadata.csv`. Several similarly named archives exist. The old path can write empty consensus transcripts. These are risks, not a proven cause of the hidden 0.4009 WER.

## Data

| Data | Rows |
| --- | ---: |
| Competition dev | 372 |
| Conversation 5 | 294 |
| Conversation 2 | 78 |
| Prepared Jember total | 1,471 |
| Jember train-only | 1,309 |
| Jember session holdout | 162 |
| Fold A train / validate | 1,386 / 294 |
| Fold B train / validate | 1,602 / 78 |
| Final train, unique rows | 1,681 |

The fold manifests are reproducible. Fold A trains on conversation 2 plus Jember and validates on conversation 5; Fold B reverses the conversations. Jember holdout sessions are disjoint from Jember training sessions. Twenty-nine dev clips exceed 30 seconds (maximum 37.6535 s). Inference uses native Whisper long-form for these clips. Final sampler draws 2,425 microbatches at competition weight 3.

## Experiments

All WER values use the competition scorer. Fold A is conversation 5; Fold B is conversation 2. One epoch was run with half-epoch and full-epoch checkpoints. Jember is the mean of the two fold-model holdout WERs.

| Run | Language | Data ratio | Epoch/checkpoint | Fold A | Fold B | Worst | Aggregate | Jember |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Indonesian x3 | Indonesian | 3 | 1 / half | 0.2236 | 0.2042 | 0.2236 | 0.2197 | 0.2750 |
| Indonesian x3 | Indonesian | 3 | 1 / full | 0.2259 | 0.1890 | 0.2259 | 0.2184 | 0.2456 |
| Javanese x3 | Javanese | 3 | 1 / half | 0.2305 | 0.2053 | 0.2305 | 0.2253 | 0.2778 |
| Javanese x3 | Javanese | 3 | 1 / full | 0.2709 | 0.1842 | 0.2709 | 0.2533 | 0.2442 |
| Auto x3 | Auto | 3 | 1 / half | 0.2325 | 0.1943 | 0.2325 | 0.2247 | 0.2922 |
| Auto x3 | Auto | 3 | 1 / full | 0.2328 | 0.1873 | 0.2328 | 0.2235 | 0.2450 |
| Indonesian x2 | Indonesian | 2 | 1 / half | 0.2423 | 0.1943 | 0.2423 | 0.2325 | 0.2980 |
| Indonesian x2 | Indonesian | 2 | 1 / full | 0.2285 | 0.1862 | 0.2285 | 0.2199 | 0.2621 |
| Indonesian x4 | Indonesian | 4 | 1 / half | 0.2453 | 0.1895 | 0.2453 | 0.2340 | 0.2975 |
| Indonesian x4 | Indonesian | 4 | 1 / full | 0.2249 | 0.1808 | 0.2249 | 0.2159 | 0.2600 |

Selection minimizes worst conversation fold, then aggregate WER, then Jember. Indonesian x3 half is the selected checkpoint (worst fold 0.2236). Indonesian x4 full has a 0.0012 worse worst-fold WER but better aggregate WER; that difference is within the requested caution range, so the conversation-protective selection rule is retained.

The selected raw predictions had Fold A WER 0.22363 and Fold B WER 0.20416. The five-identical-token safety collapse changed Fold A to 0.22277 and left Fold B at 0.20416; it is enabled for catastrophic runs only. Ordinary repeats remain unchanged.

## Selected configuration

Whisper `openai/whisper-large-v3-turbo`, Indonesian task token, competition sampler weight 3, seed 1337, AdamW at 3e-4, weight decay 0.01, gradient clip 1.0, warmup 8%, effective batch 16 with microbatch 1 and gradient accumulation, one epoch with half/full saves, gradient checkpointing, SpecAugment, mild gain and 0.9/1.0/1.1 speed augmentation. LoRA: r32, alpha64, dropout0.05, targets q/k/v/out projections and fc1/fc2 across encoder/decoder. Local MPS training used FP32 and enabled PyTorch's MPS fallback. There are 27,852,800 trainable of 836,730,880 total parameters (3.3288%).

## Final model

The final model was trained on all 372 competition clips plus 1,309 Jember training clips. Selected final training checkpoint: `runs/final_indonesian_x3/epoch_1_half`. Merged standalone model: `models/final_merged` (about 3.0 GB). On MPS, adapter and merged model produced identical sample token IDs; maximum logit difference was 7.86e-05.

## Runtime

The exact submission ZIP was run on MPS against two dev clips in 5.4 seconds, with Python socket access blocked and Hugging Face offline mode enabled. The run checked CSV columns, row count/order, and non-NaN transcript strings. Native long-form Whisper handles inputs over 30 seconds. CUDA/A100 throughput was not measurable on this Mac; the final entrypoint selects CUDA automatically when available and falls back to MPS/CPU locally.

## Submission

Deterministic archive: `artifacts/phase1_whisper_anchor_submission.zip`; SHA256: `6255239690081b884be4454b2e0eef0d187d08e89c73c8c98cefe90bdf0fec38`. The adjacent sidecar has the same digest. ZIP root contains `main.py`, merged weights, processor files, config, local code, and notices. The archive was CRC-checked and the exact ZIP passed local network-blocked inference. It contains no adapter or training checkpoint and performs inference only.

## Remaining blockers

CUDA/A100 execution and the competition's two-hour runtime remain unverified because this development machine has no NVIDIA GPU. Local MPS training, merge, package, and offline ZIP inference completed.

## Recommended Phase 2 input

Use `models/final_merged` as the single-model Whisper anchor, with its Fold A/Fold B predictions and official WER metrics as the Phase 2 comparison baseline.

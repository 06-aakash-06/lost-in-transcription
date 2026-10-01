# Phase 2 MERaLiON report

## Verdict: PROCEED TO PHASE 3

Both adapted folds provide greater than 0.03 absolute pair-oracle gain. Out-of-fold predictions and the final merged standalone model are ready. Simple fusion regresses Fold B and must be improved before deployment. Phase 1 and Submission #1 were preserved.

## Official WER

| System | Fold A | Fold B | Aggregate |
| --- | ---: | ---: | ---: |
| Whisper Phase 1 | 0.223634 | 0.204162 | 0.219674 |
| MERaLiON zero-shot | 0.287458 | 0.229190 | 0.275608 |
| Zero-shot pair oracle | 0.191902 | 0.174634 | 0.188390 |
| Zero-shot simple ROVER | 0.218537 | 0.219910 | 0.218816 |
| Adapted MERaLiON | 0.226578 | 0.220754 | 0.225393 |
| Adapted pair oracle | 0.189245 | 0.171822 | 0.185702 |
| Adapted simple ROVER | 0.218465 | 0.215129 | 0.217787 |

Simple fusion regresses versus Whisper (Fold B: +0.010967). The oracle supports Phase 3 experiments, but this diagnostic fusion is not ready for deployment.

The per-clip oracle uses references to choose the lower-error transcript. It is an upper bound on transcript selection, not an executable fusion strategy. Simple ROVER is a fixed equal-weight diagnostic, with Whisper substitution tie priority and conservative blanks; it is not a Phase 3 submission. Raw Phase 1 OOF predictions are read without modification.

## Artifacts

- Zero-shot and adapted predictions/analyses: `results/`.
- Decoder adapters and training metrics: `runs/`.
- Frozen speech feature cache: `cache/`; never packaged.
- Existing Phase 1 manifests/scorer/text/audio utilities are reused by import.

## Recipe

One seed (1337), ratio 3, one epoch, r16/alpha32/dropout0.05 decoder LoRA on q/k/v/o/gate/up/down projections; frozen encoder and audio projector; AdamW 1e-4, weight decay0.01, warmup8%, clip1.0, microbatch1, accumulation16, gradient checkpointing. MPS BF16 training and enabled PyTorch CPU operator fallback. Cached frozen speech representations and checkpointed answer-only LM-head loss reduce memory without changing the causal objective.

Selected checkpoint: `epoch_1_half`.

## Training recovery

{
  "fold_A_full_attempt": "stopped at batch 945 with nonfinite loss",
  "retained_checkpoint": "runs/fold_A/epoch_1_half",
  "retained_batches": 784,
  "adapter_weights_finite": true,
  "recovery": "Evaluate saved half checkpoint first; if useful, train matched half-epoch B/final. Future nonfinite MPS examples retain accumulated gradients and retry only decoder computation on CPU FP32.",
  "fallback_test": "tiny Gemma2 on MPS, forced nonfinite loss, CPU retry, restored MPS parameters/gradients"
}

## Final model

`models/final_merged/` (6,648,927,422 bytes)

Merge validation: `{"samples": 2, "device": "mps", "dtype": "torch.float16", "same_transcripts": true, "adapter": "runs/final/epoch_1_half", "model_load_seconds": 10.579266666998592, "adapter_inference_seconds": 12.966009708001366, "merged_inference_seconds": 10.466574500002025, "sample_audio_seconds": 51.731375}`.

Offline standalone load validation: `{"empty_hf_cache": true, "socket_connections_blocked": true, "clips": 2, "same_transcripts_as_previous_saved_model_load": true, "device": "mps"}`.

Processor validation: `{"training_texts_checked": 1681, "same_token_ids": true, "same_chat_template": true, "tokenizer_json_structural_differences": ["padding"], "regex_changed": false, "standalone_weight_keys": 781, "contains_lora_weights": false}`. The saved-tokenizer regex warning is retained without changing the trained tokenizer; its regex and all checked token IDs match the base.

fold_A training: `{"batches": 784, "optimizer_steps": 49, "mean_loss": 1.1513588773656864, "elapsed_s": 3244.8460237910003}`.

fold_B training: `{"batches": 1104, "optimizer_steps": 69, "mean_loss": 1.0036610333329958, "elapsed_s": 4307.250866917, "cpu_decoder_retries": 1}`.

final training: `{"batches": 1216, "optimizer_steps": 76, "mean_loss": 0.9670287444758671, "elapsed_s": 4474.726439417, "cpu_decoder_retries": 1}`.

## Runtime

Local generation uses MPS; CUDA is preferred automatically when available. CUDA/A100 runtime is not measurable on this machine. Per-run elapsed times are in evaluation JSON files. All weights and custom model/processor code are local; no competition archive is built in Phase 2.

## License

Public model source and downloaded licence are recorded in `MODEL_NOTICES.md`; preserve the licence and attribution in Phase 3 packaging.

Measured adapted OOF generation cost (includes loading and equivalence checks): `{"device": "mps", "batch_size": 1, "clips": 372, "elapsed_seconds": 1983.2589107080003, "clips_per_second": 0.18757006359154607, "projected_2118_clip_seconds_mps": 11291.78057225684, "cuda_a100_measured": false}`. The MPS projection is not an A100 runtime estimate. Phase 3 built and locally tested both candidate archives; A100 runtime is still unmeasured. See `../phase3_protected_fusion/PHASE3_REPORT.md`.

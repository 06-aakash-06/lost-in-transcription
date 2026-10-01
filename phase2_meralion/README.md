# Phase 2: MERaLiON decoder LoRA

Run from the repository root with the existing venv:

```sh
./venv/bin/python phase2_meralion/run_zero_shot.py
./venv/bin/python phase2_meralion/run_phase2.py
```

The first command runs fresh zero-shot competition-fold predictions and the
complementarity gate. The second runs Fold A first, stops if it is not useful,
then Fold B, selects half/full by pair oracle, and trains/merges all-data LoRA
only if both gates pass. There is no model sweep or submission packaging.
Phase 1 is read-only. All scoring uses its official scorer normalizer.

The base checkpoint is `submission_src/models/meralion3_asr`. Training freezes
the entire speech side and applies LoRA solely to the Gemma2Model inside the
text decoder. This preserves the bundled model's original embedding accessor.
Frozen speech embeddings are cached with a checkpoint fingerprint. Chunked
checkpointed supervised-answer loss matches standard Gemma2 causal loss.
No new text conventions, metadata routing, or hidden test data are used.

CPU operator fallback is enabled before importing PyTorch. Device preference
is CUDA, then MPS, then CPU. MPS training uses BF16 (verified on this Mac),
microbatch 1 and gradient accumulation 16. Inference uses FP16 on MPS and
BF16 on CUDA. Local custom Transformers code is loaded explicitly offline.

Phase 2 raw comparisons use greedy decoding with 440 new tokens, repetition
penalty 1.05 and no n-gram blocking, consistently for zero-shot and adapted
models. The older ensemble used a 256-token cap and six-gram blocking; its
cached predictions are kept as preliminary diagnostics only. Fresh zero-shot
Fold A exposes three severe decoder loops. The oracle gate is robust to these
failures because it chooses the lower-error system per clip using references.
Phase 3 evaluated bounded repetition safeguards and protected fusion; see
`../phase3_protected_fusion/PHASE3_REPORT.md` for the selected rules and
measured out-of-fold results.

Eager attention is retained on both MPS and CUDA to preserve Gemma2 attention
softcapping. CUDA still uses the A100 automatically. Any faster attention
backend must preserve softcapping and pass a prediction check in Phase 3.

## Interrupted Fold A recovery

The initial full-epoch attempt stopped at batch 945 with a nonfinite loss.
The saved half checkpoint (batch 784) has finite weights. Its two-clip
adapter/merged generation check passed. The deadline recovery command is:

```sh
./venv/bin/python phase2_meralion/run_phase2.py --recover-half
```

This waits for the already-running half-checkpoint Fold A evaluation, gates
on its oracle gain, then trains matched half-epoch B and final checkpoints.
Each uses the original one-epoch sampler/scheduler and stops at the saved
half checkpoint. Future nonfinite MPS loss/gradients trigger one CPU FP32
decoder recomputation of the same example, preserving accumulated gradients.
No example is silently skipped. This fallback has a passing MPS unit test.

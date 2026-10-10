# Resume final Large-v3 training

Run from this existing checkout, preserving the cached base and saved run configuration:

```sh
cd "<repository-root>"
./venv/bin/python overnight_attack/train_large.py \
  --config overnight_attack/configs/large_v3_bf16_full.json \
  --manifest phase1_whisper_anchor/data/final_train.tsv \
  --output overnight_attack/runs/final_large_v3 \
  --resume "$(cat overnight_attack/runs/final_large_v3/latest_checkpoint.txt)"
```

The checkpoint restores the adapter, AdamW optimizer, learning-rate scheduler,
weighted sampler position, Python/NumPy/CPU/MPS random states, and any accumulated
microbatch gradients. The recipe ends at 1.0 epoch. Check `train_metrics.json` in
the checkpoint directory for the exact completed batch and optimizer step.

Wait for the current controller and training worker to exit before running this
command. Current job status is in `results/overnight_state.json`; evidence and
checkpoint paths are in `OVERNIGHT_REPORT.md`.

The previous Ratio-4 Turbo run has its own preserved resume guide at
`../phase4_stronger_anchor/RESUME.md`.

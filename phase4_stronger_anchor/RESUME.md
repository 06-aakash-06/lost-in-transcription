# Stopped Ratio-4 run

Stopped at the user's request on October 2, 2026. The trainer and pipeline
have exited. No evaluation or subsequent Ratio-4 stage was started.

- Completed: epoch 1, batch 218 of 2,797 (0.0779406507 epoch).
- Resume begins at batch 219.
- Completed optimizer/scheduler updates: 13.
- Accumulated gradients: 10 of 16 microbatches, 464 tensors, saved intact.
- Mean training loss: 1.1707413616.

Checkpoint:

`runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218`

`training_state.pt` contains optimizer/scheduler states, accumulated gradients,
exact sampled indices and position, Python/NumPy/Torch/MPS RNG states, and
training loss/timing. Adapter weights and processor/tokenizer files are beside
it. `train.py` restores the accumulated gradients before continuing.

To resume **only this current training run** later, from the project root:

```bash
./venv/bin/python phase4_stronger_anchor/train.py \
  --config phase4_stronger_anchor/configs/ratio4_two_epochs.json \
  --manifest phase1_whisper_anchor/data/final_train.tsv \
  --output phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs \
  --resume phase4_stronger_anchor/runs/final_indonesian_x4_two_epochs/interrupted_epoch_1_batch_000218
```

This command has been documented but has **not** been executed. Do not restart
`run_pipeline.py` unless evaluation and subsequent stages are authorized again.

Audit results are in `results/EXISTING_EVIDENCE.md`, `existing_evidence.csv`,
and `existing_evidence.json`. Training, pipeline, recovery, and test logs are
preserved in `results/`. No new prediction cache was created because evaluation
never began; existing Phase 1/2/3 caches remain intact. Their hash inventory is
`results/EXISTING_PREDICTION_CACHE_SHA256SUMS.txt`.

`PRESERVED_SHA256SUMS.txt` and `PRESERVATION_MANIFEST.json` cover the preserved
Phase 4 code, configuration, audit, logs, and checkpoint files. The checkpoint
received structural read-only verification, including optimizer state count,
scheduler position, batch cursor, sampler length, and accumulated gradients.
No resumed-training or inference verification was performed after the stop.

Recovery used LLDB to install a checkpoint hook at the next completed
microbatch after suspending the process. `stop_remote.py` records that one-time
recovery procedure and its loaded-code line number; it is not a general resume
command.

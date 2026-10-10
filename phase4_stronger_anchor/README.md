# Stronger Whisper anchor experiment

Requested October 2, 2026: train final Indonesian Ratio-4 Whisper Turbo LoRA
on all 372 competition clips and 1,309 Jember chunks for up to two epochs,
evaluate non-leaky checkpoints and decode modes, and rebuild protected fusion.

The existing models and ZIPs remain intact. `train.py` reuses Phase 1 audio,
augmentation, target text, broad LoRA modules, FP32 MPS, gradient checkpointing,
and optimizer settings. The linear warmup/decay schedule spans two epochs.
Half saves occur at the next optimizer boundary: the all-data run has 2,797
microbatches per epoch and its half save is at batch 1,408 (0.503397 epoch).
The second half save is therefore 1.503397 epochs. Full saves are at 1 and 2.
Each save includes adapter weights and full optimizer/scheduler/sampler/RNG
state. Training does not stop at half epoch.

`audit.py` recovers all ten existing configurations and their raw/safe
competition and Jember results without decoding. `run_pipeline.py` waits
for the final Ratio-4 run, scores all four checkpoints on Jember, trains the
same recipe with each competition conversation withheld, then evaluates
checkpoints and forced Indonesian/Javanese/automatic decode modes. Final
all-data weights are never presented as competition OOF predictions.

`compare.py` records raw and repetition-safe WER, clip-hypothesis pair/triple
oracles, same-checkpoint loop rescue, protected strict-substitution fusion,
and conversation-cross-selected fusion rules. Labels only score predictions;
they never enter decoding, language choice per clip, or fusion.

Monitor `results/pipeline_status.json`, `results/pipeline.log`, and the
individual job logs. Evaluations cache their exact adapter/manifest hashes.
No caches, clips, reference transcripts, or training states belong in a
competition archive.

Old Ratio-3 runs have no optimizer/scheduler state. Any extension would be a
new weights-only warm-start experiment, not an exact continuation. It is
lower priority than Ratio-4 and requires the new diagnostics to justify it.

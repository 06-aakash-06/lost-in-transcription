# Unattended model work

Public `openai/whisper-large-v3` Transformers weights are pinned and hash-verified.
The older `submission_src/models/whisper_large_v3` is CTranslate2 format.

The selected real MPS smoke uses BF16, full encoder/decoder broad-projection
r16/alpha32/dropout0.05 LoRA, microbatch1/accum16, Ratio-4, Indonesian, the
existing audio/reference-style/augmentation pipeline, and non-reentrant
checkpointing. Both optimizer updates and nonzero LoRA weight changes passed.

`run_overnight.py` keeps one GPU job resident, retries failed jobs twice,
resumes periodic optimizer/RNG checkpoints, evaluates half/full immediately,
and applies the stated conversation-disjoint WER/oracle/OOD gates. It writes
`OVERNIGHT_REPORT.md` and `results/overnight_state.json` throughout the run.
Its maximum compute window is ten hours from 2026-10-02 03:11:45 UTC.

Large-v3 final all-data training only starts after two-fold evidence passes.
If that branch fails, the saved Ratio-4 Turbo state is resumed into a new
output directory, followed by small mathematically compatible delta soups,
acoustic-confidence-selected alternative Whisper decodes, and optional
training-text-only bigram candidate rescoring. No MERaLiON training or ZIP
packaging occurs here.

Every trainer saves optimizer/scheduler/sampler/RNG state every 256 batches
and at half/full checkpoints. SIGTERM/SIGINT request a completed-microbatch
checkpoint, including any accumulated gradients. Old submissions, merged
checkpoints, prediction files, and the preserved batch218 checkpoint are
never overwritten.

Final all-data checkpoints must not be scored as competition OOF. Their
permitted local labelled diagnostic is the session-disjoint Jember holdout.
Oracle results use references only as analysis lower bounds. Model/candidate
selection code never uses references during decoding, confidence extraction,
language choice per clip, or protected edits.

# Development log

This is a concise record of the project stages over roughly three weeks. Dates
below come from the existing Git commit history and dated project artifacts;
this log does not backdate or manufacture commits.

## 2026-09-10 — project and preprocessing foundation

- Initialized this GitHub-linked repository and documented the initial data
  preparation and ASR experiments.
- Began the notebook-based competition analysis and Jember preprocessing work.

## 2026-09-18 — first offline ensemble

- Added an offline competition entrypoint and packaged a multi-model system
  using Qwen, MERaLiON, Whisper, Buzz, and ROVER.
- Added local ensemble scoring and package-building utilities.

## 2026-09-19 — inference fixes

- Fixed the pyctcdecode beam-search call and a CUDA FP16 CTC input mismatch.

## 2026-09-21 — runtime work

- Added batched inference and timeout-oriented submission work; retained the
  faculty summary as a historical description of the earlier pipeline.

## 2026-09-27 to 2026-09-30 — investigate the leaderboard gap

- Replayed the old system on labelled development audio without using test
  metadata to route predictions.
- Local WER was approximately 0.2058, while the prior hidden leaderboard
  result was 0.4009. Metadata-aware replay was approximately 0.2062, so the
  routing metadata did not explain the gap. The cause remains undetermined.
- Saved the diagnosis and an architecture handoff rather than claiming an
  unsupported explanation.

## 2026-09-30 to 2026-10-01 — Phase 1: Whisper anchor

- Prepared 372 labelled competition clips and Jember train/holdout manifests.
- Used strict leave-one-conversation-out folds, the official scorer, and a
  conservative reference-style training target.
- Trained Whisper large-v3-turbo with r32/alpha64 LoRA on Apple MPS. Indonesian
  conditioning and competition sampling ratio 3 were selected; ratios 2 and 4
  were compared. The selected OOF WER was 0.2236 / 0.2042 / 0.2197.
- Merged the adapter, validated offline MPS inference, and built Submission #1.
  Its reported hidden leaderboard WER was 0.2862.

## 2026-10-01 — Phase 2: MERaLiON complementarity

- Reused saved out-of-fold predictions and the official scorer for the
  zero-shot gate. Decoder-only MERaLiON-3-3B LoRA was trained only after the
  Whisper/MERaLiON pair oracle showed useful headroom.
- Adapted MERaLiON scored 0.2266 / 0.2208 / 0.2254. The pair oracle was
  0.1892 / 0.1718 / 0.1857; simple fusion reached only 0.2185 / 0.2151 /
  0.2178. These results motivated protected corrections instead of blind
  voting.

## 2026-10-01 — Phase 3: protected fusion

- Aligned Whisper and MERaLiON word sequences while preserving Whisper surface
  text by default. Tested bounded substitutions, acoustic-confidence rules,
  repetition guards, and existing Whisper third hypotheses on both folds.
- The best two-model candidate scored 0.2157 / 0.2039 / 0.2133. An existing
  full-epoch Whisper checkpoint provided exact correction support and reached
  0.2124 / 0.1971 / 0.2093 without new training.
- Built and locally tested both offline archives. CUDA/A100 execution remains
  the outstanding deployment check.

## Evidence boundaries

- OOF scores are labelled development results, not hidden-test estimates.
- The 0.2862 and 0.4009 values are previously reported hidden leaderboard
  results for different uploaded systems.
- CUDA runtime projections were not measured on the MPS development machine.

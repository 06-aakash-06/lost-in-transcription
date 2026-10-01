# Model and data artifact inventory

Source code, training recipes, inference code, tests, license notices, and
aggregate reports are tracked. Multi-gigabyte weights and submission archives
are kept locally and are not committed to ordinary GitHub Git. Several shard
files exceed GitHub's normal single-file limit, and archives are many gigabytes.

| Local artifact | Origin / contents | Published status |
|---|---|---|
| `submission_src/models/` | Legacy Qwen, MERaLiON, Whisper, and Buzz weights | Local only; original model cards and notices stay in the repository |
| `phase1_whisper_anchor/models/` and `runs/` | Whisper base, merged checkpoint, and LoRA runs | Local only; public base checkpoint is documented in Phase 1 files |
| `phase2_meralion/models/` and `runs/` | MERaLiON merged checkpoint and decoder LoRA runs | Local only; upstream public license is retained in `phase2_meralion/` |
| `phase3_protected_fusion/models/` | Existing full-epoch Whisper third voter | Local only; adapter merge command and validation are in the Phase 3 report |
| `artifacts/phase1_whisper_anchor_submission.zip` | Previously tested Phase 1 archive | Local only; SHA256 recorded in the Phase 1 report |
| `artifacts/phase3_protected_fusion_submission.zip` | Candidate A, Whisper + MERaLiON | Local only; SHA256 recorded in the Phase 3 report |
| `artifacts/phase3_three_model_submission.zip` | Candidate B, with full-epoch Whisper support | Local only; SHA256 recorded in the Phase 3 report |

Competition audio, Jember corpus audio/transcripts, generated fold manifests,
clip-level reference/prediction CSVs, and hidden submission data are excluded.
The scripts expect the corresponding licensed/local data and checkpoints to be
prepared before running training or rebuilding a full submission archive.

Public upstream model IDs are recorded in the phase license files and model
notices. Final adapted weights are not hosted by this repository.

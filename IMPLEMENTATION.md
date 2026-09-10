# Reproducible EDA and preprocessing

The repository now has a versioned, scriptable preprocessing path in
`src/lostintranscription_preproc.py`. It deliberately stops before model
training. The existing `processed/v1` artifact is left untouched; the hardened
default writes `processed/v2`.

## Run

From the repository root:

```bash
./venv/bin/python scripts/build_preprocessing.py --version v2 --clean
./venv/bin/python scripts/validate_preprocessing.py --version v2
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

`--clean` only permits deleting a path of the form `processed/v*`. Omit it to
reuse an output directory, although a clean rebuild is recommended when
changing preprocessing code. `--no-plots` skips the optional PNG summaries.

## What v2 guarantees

- Jember audio is decoded at its true duration, resampled to 16 kHz mono, and
  written as FLAC/PCM_16. MP3 container/header duration is used for audit only.
- Invalid timestamps, blank labels, starts beyond decoded audio, and post-clamp
  stubs are recorded in stage accounting and excluded.
- Jember training text is normalized conservatively: scholarly combining marks
  are removed; curly punctuation and ellipses are made deterministic; casing,
  hyphens, apostrophes, and attached ellipses are preserved. An em dash becomes
  whitespace because the official scorer treats it as a delimiter; mapping it
  to a hyphen would invent a scored token.
- The official scorer normalizer is mirrored separately in
  `official_normalize_text`. It is used for diagnostics and word counts, never
  to rewrite the dev references or training targets broadly.
- Adjacent Jember rows are merged losslessly to at most 29 seconds. The plain
  merge is the shipped variant; a turn-aware boundary candidate is emitted for
  an ablation. Source rows are partitioned exactly once and retained in JSON
  provenance fields.
- Jember chunks receive 150 ms boundary context, then conservative edge-only
  energy trimming. Interior audio is never removed. A one-scalar RMS gain aims
  at -23 dBFS with a -1 dBFS peak ceiling; no compressor or limiter is baked
  into the data.
- The hard gate only removes empty/degenerate examples, very low-SNR examples,
  extreme silence, and clips outside the 2–30 second training range. Flags and
  bounded `quality_weight` hints are emitted but not silently applied.
- The split is session-disjoint and deterministic (`seed=1337`). Separate
  `train_jember_trainonly.tsv` and `heldout_jember.tsv` files reduce accidental
  validation leakage; `train_jember.tsv` retains the split column for auditing.
- Dev references are copied verbatim. They are never used for lexical mapping,
  target selection, or preprocessing decisions.

## Outputs

Each version contains:

- `train_jember.tsv`, `train_jember_trainonly.tsv`, `heldout_jember.tsv`
- `dev.tsv`
- `audio/jember/*.flac` and `audio/dev/*.flac`
- `meta/` inventories, row-integrity records, both merge-boundary variants,
  and the all-chunks prefilter manifest
- `eda/` duration, boundary, merge, quality, vocabulary/OOV, Unicode,
  target-group audio summaries, audio-only domain features, and optional PNGs
- `manifest.json` with configuration, source metadata, stage accounting,
  output counts, split membership, and validation notes

## Deliberately deferred

The attached ideas about target-likeness weighting, lexical correction, MBR or
N-best selection, and routing are useful later, but they are not hidden inside
this preprocessing run. Audio-only domain features and auditable quality
fields are ready for those experiments. Dev transcript-derived choices must
remain validation-only; any future selection or router should be fit on the
session-disjoint heldout split or an external training source, then confirmed
on dev once.

The next phase can compare plain vs turn-aware segmentation, equal vs bounded
quality sampling, and preprocessing ablations against a fixed baseline without
re-decoding or re-auditing the corpus.

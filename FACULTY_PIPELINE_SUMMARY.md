# Lost in Transcription: Preprocessing and ASR Pipeline

> **Historical Phase 0/legacy summary (2026-09-21).** This document describes
> the earlier multi-model approach and its preprocessing experiments. The
> selected final competition candidates are now documented in the root
> [README](README.md) and `phase3_protected_fusion/PHASE3_REPORT.md`.

## Two-minute summary

This project builds an offline Automatic Speech Recognition system for natural
Indonesian–Javanese code-switched speech. The main challenge is that the
training and evaluation data are linguistically different: the Jember corpus
is mostly Javanese conversational speech, while the target development data is
mostly Indonesian–Javanese mixed speech with everyday spelling.

Our solution therefore combines:

1. Conservative, evidence-based preprocessing of the training corpus.
2. Several complementary local ASR models.
3. Language-aware routing.
4. Word-level ensemble voting using the competition's WER rules.

The final system is fully offline: models are bundled locally, no test clip is
used to train another clip, and no network request is made during inference.

## 1. Data observations that determined the design

| Observation | Measured implication | Design response |
|---|---:|---|
| Jember contains 6,679 annotated rows, with 41 invalid timing rows | Some metadata cannot be used safely | Validate timestamps against decoded audio duration |
| Raw Jember segments have a median length of about 5 seconds | The target clips have a median length of 26.2 seconds | Merge adjacent segments into chunks of at most 29 seconds |
| Jember uses scholarly diacritics such as `ê`, `è`, and `ě` | The target mostly uses ordinary spelling; spelling differences become WER substitutions | Remove combining marks from training transcripts |
| Jember audio has a much wider loudness range than the target data | The acoustic model sees inconsistent signal levels | Measure quality and use controlled RMS normalization where it helps |
| Target speech is code-switched | A single forced-language model can suppress words from the other language | Use Indonesian and Javanese specialists together |

## 2. Training-data preprocessing

The preprocessing pipeline prepares the Jember corpus for future model
training and provides a leakage-safe validation split. It is separate from the
final test-time inference pipeline.

### Step 1: Transcript normalization

- Convert curly quotes and Unicode ellipses to deterministic ASCII forms.
- Remove combining marks to map forms such as `těrus` to `terus`.
- Preserve casing, hyphens, apostrophes, and word-attached ellipses because
  these can affect the competition score.
- Do not broadly lowercase or rewrite the official development references.
- Keep a separate implementation of the official scorer normalizer for local
  WER measurement and transcript alignment.

**Why:** The most frequent orthographic mismatch was not an acoustic problem;
it was the spelling convention. Diacritic normalization reduced development
set OOV exposure from about 21.4% to about 18.4%.

### Step 2: Audio and row integrity

- Decode audio to its true duration rather than trusting compressed-file header
  duration.
- Resample to 16 kHz, convert to mono, and store FLAC/PCM-16.
- Remove non-positive or unusable rows and clamp annotations that extend beyond
  the decoded audio.
- Keep source-row IDs in the manifest so every output chunk is auditable.

**Why:** MP3 headers can over-report duration. Using them for slicing can create
  transcript/audio misalignment and train the model on artificial errors.

### Step 3: Duration matching through lossless merging

- Merge adjacent, contiguous Jember annotations in the same session.
- Limit each merged training chunk to 29 seconds, leaving a small margin below
  the model's approximately 30-second input window.
- Concatenate the corresponding transcripts in order.

**Result:** 6,638 usable rows become approximately 1,472 long-form chunks.
The median duration changes from about 5 seconds to 26.0 seconds, matching the
target median of 26.2 seconds. This is a regrouping of existing labelled audio,
not synthetic data creation.

### Step 4: Boundary and loudness handling

- Add up to 150 ms of neighbouring context around Jember boundaries.
- Apply conservative edge-only energy trimming; interior pauses are never
  removed.
- Apply one scalar RMS gain toward -23 dBFS, with a -1 dBFS peak ceiling.
- Do not use compression or aggressive denoising.

**Why:** Jember timestamps are quantized to whole seconds, so words may be cut
at the edges. Edge-only processing can reduce boundary damage without deleting
natural conversational pauses. RMS normalization reduced the Jember p5–p95
loudness spread from roughly 21.1 dB to 4.1 dB, close to the target spread.

### Step 5: Quality gate and split

- Drop only degenerate examples: no detected speech, extremely low SNR,
  excessive silence, or duration outside the 2–30 second training range.
- Keep quality flags such as low SNR, off-mic speech, and possible clipping for
  later ablation studies.
- Split by session, not by individual clip, using a fixed seed. This prevents
  neighbouring clips from leaking across train and held-out sets.

The final prepared training manifest contains about 1,470 usable chunks. The
official development references remain unchanged and are used only for
validation.

## 3. Final offline inference pipeline

```text
Test manifest and audio
        ↓
Validate filenames and preserve clip identity
        ↓
Resample each clip to 16 kHz mono
        ↓
Apply the model-specific audio profile
        ↓
Route models using the language label
        ↓
Generate independent transcripts
        ↓
Align words using the official scorer normalization
        ↓
ROVER voting or weighted-medoid fallback
        ↓
submission/submission.csv
```

Each clip is processed independently. Predictions are aligned by stable clip
identity; no prediction from one test clip is used to predict another.

## 4. Why each model is used

| Model | Route | Audio profile | Role and justification |
|---|---|---|---|
| Qwen3-ASR-1.7B | `ind`, `javind` | RMS-normalized | Strong independent Indonesian-capable candidate; its held-out result was better with controlled normalization. |
| MERaLiON-3-3B-ASR | `ind`, `javind` | Raw waveform | Southeast-Asian ASR model with a different architecture and language prior; adds complementary errors and handles mixed speech. |
| Whisper large-v3-turbo | `jav`, `javind` | Raw waveform | Javanese-forced specialist path; useful when Javanese content is important. |
| BuzzASR Javanese | `jav`, `javind` | Raw waveform | Javanese-specific specialist; provides a second independent Javanese hypothesis. It is down-weighted on mixed speech because it can over-predict Javanese. |
| Whisper large-v3 | `ind` only | Raw waveform | General Indonesian anchor and guard-rail for Indonesian-only clips. It is not forced into the mixed-language route because that can suppress Javanese words. |

The models are deliberately not given identical preprocessing. Controlled
experiments showed that Qwen benefited from RMS normalization, while Whisper,
Whisper Turbo, and MERaLiON performed best or tied on raw audio. This is why the
runtime contains `qwen_norm` and `model_raw` profiles.

## 5. Ensemble decision

The models use different tokenizers, so logits cannot be averaged directly.
Instead, their transcript words are aligned using the competition's exact
normalization.

- For `javind` and `ind`, the final runtime uses calibrated word-level ROVER.
- ROVER selects the weighted majority word at each aligned position.
- A blank factor of 1.25 makes the system conservative and suppresses words
  supported by only one model, reducing insertions.
- For routes where ROVER is not selected, the weighted medoid chooses the whole
  transcript with the lowest weighted edit distance to the other hypotheses.
- BuzzASR receives a lower multiplier on mixed speech and a higher multiplier on
  Javanese-only routing.

This is calibrated using labelled development/held-out predictions only. No
hidden test transcripts are used.

## 6. Measured result

On the local development proxy, the final calibrated ensemble achieved:

| Evaluation slice | WER |
|---|---:|
| Overall, 372 clips | **0.2053** |
| `javind`, 352 clips | **0.2101** |
| `ind`, 20 clips | **0.1138** |

These are local validation results, not the hidden leaderboard score. The
development set contains no Javanese-only clips, so `jav` performance cannot be
measured locally.

## 7. What we deliberately did not do

- No aggressive interior VAD or silence removal.
- No unvalidated noise injection, speed perturbation, SpecAugment, or duplicate
  test-time augmentation in the submission.
- No blanket lowercasing, hyphen removal, or apostrophe removal.
- No transcript deduplication, because repeated backchannels are real speech.
- No test pseudo-labeling or cross-test adaptation.

These choices keep the system reproducible and avoid adding transformations
that were not shown to improve held-out WER.

## 8. Main conclusion

The central contribution is not simply using a larger model. It is matching the
data preparation and decoding strategy to the actual task: target-compatible
orthography, target-compatible duration, conservative acoustic processing,
language-aware specialist routing, and an ensemble that is optimized for the
official WER calculation.

The final submission is packaged as an offline executable with local model
weights, a ZIP-root `main.py`, and the required output schema:
`audio_filename,transcript`.

### Supporting project files

- [Preprocess.md](Preprocess.md) — detailed preprocessing evidence and ablations.
- [Findings.md](Findings.md) — dataset audit and measured risks.
- [submission_src/ensemble_config.json](submission_src/ensemble_config.json) — model routing and preprocessing profiles.
- [submission_src/calibration.json](submission_src/calibration.json) — calibrated ensemble settings.
- [submission_src/ensemble_runtime.py](submission_src/ensemble_runtime.py) — offline inference and consensus logic.

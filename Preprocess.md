# Preprocess.md — Preprocessing pipeline, rationale, research basis, and validation

This document covers Steps 4–7 of the pipeline build: the current best-practice research that
informed design choices, the pipeline itself (implemented in `Main2.ipynb` §35–§41, "Part 4"),
its validation (§42–§48, "Part 5"), and the open trade-offs. Every quantitative claim below is
copied from a notebook cell's actual printed output on this run of the data — none are estimated.
Findings referenced as `Rn` are the risk IDs from `Findings.md` §6; `§NN` refers to a Main2 section.

The processed dataset lives at `processed/v1/` (manifest, TSVs, and 1,472+372 FLAC files,
~774 MB total). `Main1.ipynb` was never modified; `Main2.ipynb` is Main1's 54 cells followed by
61 new cells (Parts 3–5) — nothing in Main1 was deleted or rewritten.

---

## 1. Research basis (Step 4)

Five search passes covered code-switched/low-resource ASR preprocessing, current (2025–2026)
Whisper/wav2vec2/MMS fine-tuning recipes for Southeast Asian and similarly under-resourced
languages, audio augmentation, and text normalization for orthographically inconsistent languages.
Sources cross-checked below; where they disagreed, the decision is grounded in this dataset's own
Part 3 measurements rather than either source's default.

| Topic | What the literature says (2025–2026) | Where it agreed / disagreed | What we did, and why |
|---|---|---|---|
| Code-switched Whisper adaptation (Interspeech 2025) | Soft prompt tuning, language-aware decoding, and LoRA/PEFT outperform full fine-tuning for code-switched adaptation; "Code-Switch Bigram Accuracy" is proposed as a switching-aware metric alongside WER. | Consistent across papers that full fine-tuning on a switching-mismatched corpus over-fits the switching *pattern*, not just the vocabulary. | We don't train in this pass, but Findings.md R3 (Jember switches 1.5× more densely, with 3.2× shorter same-language runs than dev) is exactly the failure mode this literature predicts. This pipeline's job is to not make that gap worse — it does not attempt to fix it, since that requires model-side mitigation (LoRA, low LR, few epochs) flagged for the next phase, not preprocessing. |
| Javanese/Sundanese Whisper fine-tuning recipes | 16 kHz mono PCM; VAD-guided segmentation; target utterances in the 3–30 s range (matching Whisper's 30 s window and avoiding degenerate short clips). | Agreed with the MMS/wav2vec2 literature on sample rate and mono; recipes differ on exact target duration (some target 10–20 s, others 20–30 s). | We resample everything to 16 kHz mono FLAC/PCM_16 and merge Jember's raw ~5 s segments into ≤29 s chunks — chosen to land inside Whisper's 30 s window with margin, and because §17/§41 show the *dev* set's own median (26.2 s) sits there. We did not pick a recipe's number in the abstract; we matched the target domain measured in §16. |
| VAD and long-form segmentation | A 2024 long-form Whisper study found VAD-based segmentation cut WER 0.675→0.419 on Bengali; restricting fine-tuning data to a narrow duration band (20–28 s) was independently consequential. | Strong agreement that boundary quality matters more than aggressive interior VAD. | This directly motivated §27's finding that Jember segments are already tightly cropped to speech (2.96 words/speech-second vs dev's 2.29) — so *interior* VAD would widen the domain gap, not close it (§32 explicitly measured and rejected this). We apply **edge-only** VAD trimming, never interior silence removal. |
| Audio augmentation (2026 Springer JASMP ablation) | Speed perturbation + SpecAugment outperformed time-stretch, white-noise injection, RIR convolution, and fade-in/out for low-resource ASR fine-tuning ablations. | This is a model-training-time augmentation, not a preprocessing-time one — it's applied to spectrograms/waveforms during training, typically on-the-fly. | **Explicitly out of scope for this pass** per the task constraint (no training code). Recorded here so the next phase applies SpecAugment + speed perturbation at train time rather than baking a fixed augmented copy into `processed/v1/` now, which would only lose the flexibility to tune augmentation strength later. |
| Text normalization for orthographically inconsistent low-resource languages | Work on Ligurian, Occitan, and several African languages ("What is Lost in Normalization?") converges on: normalize spelling variance that is orthographic noise, but preserve variance that carries phonemic or dialectal information — over-normalizing erases real linguistic signal the model needs to learn from. | Consistent guidance; the disagreement in the literature is *where* to draw that line, which is corpus-specific. | This is why §2 of Findings.md (dialectal spelling variance) and §35's normalization function draw the line at **diacritics that are purely orthographic convention** (pepet/taling marks `ê è ě`) vs a phoneme that changes word identity (`ḍ`, retroflex d) — both stripped, because dev's own orthography does the same collapse (`gedhe`→`gede`). We did **not** touch casing, hyphenation, apostrophes, or ellipses, since those are scored distinctions dev actually preserves (§26b). |
| Loudness normalization (MMS/wav2vec2 practice) | Peak normalize to −1 dBFS and/or target ~−12 to −23 LUFS (BS.1770-family) before feature extraction, so encoder-seen levels are consistent across a training corpus. | Consistent with the mechanical finding in §27: Jember's RMS spans 30.1 dB (dev spans 11.8 dB) — a spread three times the target range. | Single-scalar-gain RMS normalization to −23 dBFS, backed off to respect a −1 dBFS peak ceiling. No compression/limiting, so a clip's *internal* dynamics (its actual signal) survive — only the inter-clip level spread is compressed. |

**Why not more aggressive options?** Silero/webrtcvad-based VAD was the literature's default
recommendation, but this environment has neither torch nor webrtcvad available (confirmed at
setup). An energy-based VAD (25 ms frame / 10 ms hop, session-level noise floor, floor+12 dB
threshold) was built instead and validated against ground truth we do have — Jember's known-gapless
adjacent segments (§17) and dev's measured edge behavior — rather than assumed equivalent to a
neural VAD.

---

## 2. Pipeline, in order (Step 5 build, Step 6 validation folded in)

Each stage names its `Findings.md` risk ID, its exact configuration, and its measured before/after
result. Config values are in `CFG` (`Main2.ipynb` §37, also written to `manifest.json["config"]`).

### Stage 1 — Training-text normalization (§35 → risk R1, R10)

`normalize_training_text()` (kept entirely separate from `wer_norm`, which is byte-identical to the
official scorer and is never touched):

1. Unicode punctuation → ASCII (`" " → "`, `' ' → '`, `… → ...`, en/em dash → `-`) — must run before
   diacritic stripping. Addresses **R10**: `wer_norm` cannot handle the 80 Unicode punctuation
   characters (`"`×29, `"`×28, `'`×19, `…`×4) actually present in Jember (§26).
2. Strip combining marks (Unicode NFD, drop category `Mn`) — removes `ê è ě` and the retroflex `ḍ`.
   Addresses **R1**: 1,055 dev tokens (6.04% of the denominator) are words the two corpora spell
   differently, almost entirely because Jember carries pepet/taling diacritics dev's orthography
   doesn't use (§2 / §31).
3. Collapse whitespace.

**Deliberately not done here:** casing, hyphenation, apostrophes, ellipses. §26b's punctuation
convention table shows dev preserves all of these as scored distinctions (hyphenated tokens
23.6%→75.3% of utterances jember→dev; ALL-CAPS acronyms are protected by `wer_norm`). Touching them
would trade a measured, bounded gain for an unmeasured, likely-harmful loss.

**Measured effect (§44):** acronym, hyphen, and ellipsis token counts are unchanged (a hard
assertion in the notebook: 302/1,961/766→770 tokens survive intact — the +4 on ellipses and +19 on
apostrophes are from the Unicode-punctuation-mapping step recovering `…`/`'` variants that a naive
strip would have dropped). Cost: 53 dev tokens containing `é` (0.30% of the denominator) can now
never be emitted, since `é` is also stripped by the shared diacritic-removal step. Benefit: 506
tokens recovered from OOV. **Net +453 tokens**, i.e., the trade is worth it by a factor of ~10.

### Stage 2 — Row integrity (§36 → risk R11, R17)

Drops the 41 structurally invalid Jember rows already found in §15 (negative/zero duration, end
beyond file length) and additionally uses **decoded** audio duration, never the `sf.info` header
(**R17**: the header over-reports by up to 3.93 s on all 89 of the 44.1 kHz files — a decoder
padding/delay artifact confirmed against `ffprobe`, not a timeline shift). This let stage 2 also
catch and clamp 7 segments whose transcript ran past the true (decoded) audio end — including
session 30's 16.2 s overrun found in §25, which Main1's `sf.info`-based check had under-reported.

**Measured (manifest `stage_accounting`):** 6,679 → 6,638 rows (**−41**, non-positive
duration/blank/starts-past-audio). A further clamp-stub check dropped 0 additional rows (no
segment was reduced below the 0.20 s floor by clamping). 7 segments clamped, reclaiming 22.4 s of
previously unbacked transcript.

### Stage 3 — Segmentation (§37 → risk R4, R8, R13)

Two variants are built and both are written to `meta/` for later ablation, but only **plain**
merge is shipped as the default training set:

- **`chunks_plain`** — `merge_segments` (Main1 §17's function, unmodified): concatenate adjacent
  segments in the same session up to `merge_max_dur_s=29.0`, `merge_max_gap_s=1.0`. Jember's
  adjacent segments are 100% gapless (§17), so this concatenation is lossless.
- **`chunks_turn`** — a turn-aware variant that prefers to close a chunk at an adjacent-segment RMS
  jump of `turn_step_db=6.0` (a speaker-change proxy validated in Findings.md against question-mark
  rate: 21.2% at >6 dB vs 8.5% base), subject to a `turn_min_chunk_s=12.0` floor so duration match
  isn't sacrificed for turn purity.

Addresses **R4** (§16: raw Jember median 5 s vs dev target 26.2 s), **R8** (edge clipping — see
Stage 4), and **R13** (§32: 149 mis-timed rows, 2.2% of usable rows, absorbed into correctly-timed
chunks by merging rather than needing individual repair).

**Measured (§41, `merge_segments`):** 6,638 raw segments → **1,473 merged chunks**, 9.98 h (nothing
lost — pure re-grouping). Duration share inside the dev target band (15–32 s): raw Jember 1.6% →
merged 92.5% (dev itself: 92.7%). Median duration 26.0 s vs dev's 26.2 s.

**Judgment call flagged:** `chunks_turn` (1,552 chunks) is computed and written to
`meta/chunk_boundaries_turn_aware.tsv` but **not** the shipped variant. §29 found 20.6% of plain
merged chunks contain an internal >6 dB level step (a likely speaker turn) against 0% of dev clips
being multi-speaker — a real domain-mismatch risk (**R7**) — but turn-aware splitting trades that
against duration match, and which one costs more WER is a training-time question. Both variants
exist so this is an ablation, not an assumption.

### Stage 4 — Audio materialization (§38 → risk R5, R6, R8, R17)

Per chunk, in order: decode once per session at 16 kHz mono → slice with **boundary padding**
(`edge_pad_s=0.15` s per side, Jember only) → **edge-only VAD trim** (never interior) →
**loudness normalize** (single scalar gain to `target_rms_dbfs=-23.0`, backed off to respect
`peak_ceiling_dbfs=-1.0`) → write FLAC/PCM_16.

- **Boundary padding** addresses **R8**: §27 measured 55.6% of Jember segments hot (speech
  present) in their very first frame vs 39.8% of dev clips — whole-second timestamp quantization
  cuts words in half at segment boundaries, and trimming alone cannot fix a boundary that has no
  silence to trim. Padding imports up to 0.15 s of the surrounding session audio (justified: §17
  proved adjacent segments are exactly contiguous, so the neighboring audio is real, not noise)
  before the trim step runs. **Dev does not get this step** — a dev clip is a whole file with
  nothing outside it to pad from, and its boundaries are the file's actual boundaries rather than
  an artifact of whole-second annotation.
- **Edge VAD trim**: frame-energy contour (25 ms/10 ms hop), session-level 5th-percentile noise
  floor, `vad_thresh_db=12.0` above floor marks speech; only the leading/trailing silence run is
  removed, capped at `edge_trim_max_s=1.0` s, with a `0.20` s margin left in place so a trim never
  bites into speech. **Both** Jember and dev get this step.
- **Loudness normalization**: addresses **R5** (§27: Jember RMS spans 30.1 dB vs dev's 11.8 dB —
  triple the target spread). A pure scalar gain preserves each clip's internal dynamics; the peak
  ceiling prevents clipping.

**Measured — boundary padding (§45):**
- Chunks padded on at least one side: 1,472 of 1,472 (100%).
- Total audio added by padding: 408 s (1.13% of Jember's total).
- Pads landing on speech (importing an untranscribed neighbor): 1,222 (83.0%) — of those, 1,075 had
  an already-cut (hot) edge, so for the large majority the pad repairs more than it imports. 147
  chunks padded onto speech with *neither* edge originally cut — this is the actual cost of the
  padding, an unlabeled ~0.15 s import for no correctness gain.
- **Net effect on the hot-edge rate, honestly reported as only partially successful:** lead-edge
  hot rate was unchanged (49.1% → 49.1%); trail-edge improved modestly (53.6% → 50.3%); dev's own
  rate is 39.8%/29.6%. Padding plus trimming narrows the gap but does not close it — recorded as a
  known-unfixed item below (§3).

**Measured — loudness (§45):** RMS spread (p5–p95) 21.1 dB → **4.1 dB** (dev: 3.9 dB) — the
headline result of Stage 4, directly closing R5. Full range 33.4 dB → 12.0 dB. Zero clipping
introduced: max peak after processing is 0.8913 (the −1 dBFS ceiling) on both Jember and dev, and
zero chunks have any full-scale sample. Median gain applied: +2.6 dB (range −12.1 to +20.6 dB).

**Note on SNR reporting:** SNR is invariant to a scalar gain by construction, so loudness
normalization cannot change it — an earlier draft of this validation misleadingly reported "chunks
whose SNR got worse." The actual pre/post SNR difference (median |ΔSNR| 0.62 dB, weak correlation
−0.06 with seconds trimmed) is a **measurement** artifact: the post-processing floor is
re-estimated on the trimmed audio, which removes the quietest frames that fed the pre-processing
estimate.

### Stage 5 — Quality gating (§39 → risk R6, R12, R15)

Thresholds are read off Part 3's own distributions, not defaults:

| Rule | Threshold | Source |
|---|---|---|
| drop `speech_frac == 0` | — | §27: 15 segments with no detectable speech |
| drop `snr_db < 10` | 10 dB | §27: the bottom 0.87% tail; dev's *minimum* SNR is 21.5 dB, so nothing near this threshold exists in the target distribution |
| drop `silence_frac > 0.95` | 0.95 | §27: 0.86% of segments |
| drop duration outside `[2, 30]` s | 2 / 30 s | the competition spec's stated test-clip floor and Whisper's window ceiling |

Flagged, not dropped: `flag_offmic` (off-mic interviewer segments, **R6**), `flag_clipped_batch`,
`flag_low_snr`, `flag_hot_peak` — carried into the output TSV as columns so a future training run
can down-weight or exclude them without re-running the whole pipeline.

**Measured (§39, `stage_accounting`):** 1,472 → **1,470** chunks (**−2**: 1 shorter than 2 s, 1
over 95% silence). Flags carried forward: `flag_offmic` 98 chunks (6.66%), `flag_low_snr` 154
(10.46%), `flag_clipped_batch` 34 (2.31%), `flag_hot_peak` 0. Dev is never dropped by this gate —
it is only run against dev for reporting (29 dev clips, all "longer than 30 s," would fail it —
confirming the gate is calibrated to the training distribution and not accidentally miscalibrated
against the target).

### Stage 6 — Deduplication and split (§40 → label integrity)

- **Deduplication check on merged chunks**: 0 of 1,470 duplicate transcripts, 0 duplicate
  normalized-token hashes post-merge. The 78 raw duplicate rows found in §33 (all short
  backchannels — `Ha'a.`×12, `Iku sih Mbak.`×11, `Apa ya?`×7 — genuine repeated utterance
  frequency, not copy-paste) are **not deleted**; merging naturally dissolves them into distinct
  longer chunks, so no explicit dedup step was needed or applied.
- **Session-disjoint split**, seed 1337, `heldout_frac=0.10`: 208 sessions → 180 train / 28
  heldout, 0 sessions appearing in both. This is the only leakage-safe key available (§33: Jember's
  TSV carries no speaker or conversation column). **Caveat carried forward, not resolved**: each
  session contains two speakers (§28), and nothing in the data lets us verify a speaker never
  recurs across sessions — session-disjoint is a floor on leakage safety, not a guarantee.

**Measured:** train 1,308 chunks / 180 sessions / 8.90 h / 58,885 words; heldout 162 chunks / 28
sessions / 1.06 h / 6,687 words.

### Stage 7 — Write dataset (§41)

Output at `processed/v1/`: `train_jember.tsv`, `dev.tsv`, `manifest.json` (config +
stage_accounting + known_unfixed), `meta/` (both merge variants, pre-integrity and pre-filter
snapshots for audit), `audio/jember/*.flac` (1,472 files, 661.6 MB — note this count includes the
2 chunks later dropped at the quality gate for transcript purposes but materialized for audit;
`train_jember.tsv` itself lists only the 1,470 kept rows), `audio/dev/*.flac` (372 files, 148.4
MB). All audio: FLAC/PCM_16, 16 kHz mono. The dev reference transcript is stored verbatim,
un-normalized — the official scorer normalizes both sides itself, so rewriting the reference would
mean scoring against a metric nobody actually uses.

---

## 3. End-to-end validation summary (Step 6)

**Row accounting**, stage by stage (`manifest.json["stage_accounting"]`, also §42):

| stage | rows in | rows out | dropped | reason |
|---|---|---|---|---|
| 2. drop invalid rows | 6,679 | 6,638 | 41 | non-positive duration / blank / starts past audio |
| 2b. drop post-clamp stubs | 6,638 | 6,638 | 0 | clamping left <0.20 s of audio |
| 3. merge to ≤29 s chunks | 6,638 | 1,472 | 5,166 | lossless re-grouping (not a drop of content) |
| 4. materialize audio | 1,472 | 1,472 | 0 | chunks shorter than one frame skipped |
| 5. quality gate | 1,472 | 1,470 | 2 | shorter than 2 s (1); over 95% silence (1) |

**Word accounting:** 65,614 words after the integrity stage → **65,572** in the final training set
(**−42**, entirely from the quality gate — no words are lost to merging or padding/trimming).

**Audio accounting:** raw 9.98 h → final training audio 9.96 h (net delta **−0.2%**). Where it
went: 22.4 s reclaimed by clamping (a gain, not a loss), 433.5 s removed by edge VAD trimming, 24.8
s removed by the two quality-gate drops.

**Duration distribution before/after** (§41/§43): median 5 s (raw) → 26.0 s (merged) vs dev's 26.2
s; share in the 15–32 s dev band: 1.6% → 92.5% (dev: 92.7%).

**Vocabulary/OOV before/after** (§44/Findings §2): dev whole-word OOV 21.37% (raw Jember spelling)
→ **18.47%** (after diacritic stripping). Spelling-variant conflicts: 153 words / 1,055 tokens
(6.04% of the dev denominator) → 39 words / 78 tokens (0.45%) — the residual is genuine dialectal
variance the normalizer correctly declines to touch (see §4).

**Audio quality metrics before/after** (§45): RMS spread p5–p95 21.1 dB → 4.1 dB (dev target 3.9
dB); SNR essentially unchanged (invariant to gain, see Stage 4 note); silence fraction distribution
unchanged (edge-only trimming doesn't touch interior silence, by design); zero clipping introduced
in either corpus.

**Spot-checks** (§46): waveform plots and transcript diffs performed on representative examples
(a padded/repaired boundary, a loudness-normalized quiet segment, a merged multi-segment chunk)
with sanity assertions (no chunk exceeds 30 s, no chunk has 0 duration, gain never exceeds the peak
ceiling) — all pass.

---

## 4. Open issues and trade-offs

Rejected practices, on measured evidence rather than by default:

- **Interior VAD / silence removal within a segment.** Rejected: §32 shows Jember is already
  tightly cropped to speech (2.96 words/speech-second vs dev's 2.29); more aggressive interior VAD
  would *widen* this gap, not close it.
- **Fuzzy/edit-distance OOV merging.** Rejected: inspection of edit-distance-1 candidate pairs
  found roughly half were coincidental near-misses between unrelated words, not the same word
  misspelled — an automated merge would introduce silent errors for a marginal recall gain.
  Reference: § 31 near-miss examples.
- **Transcript deduplication.** Rejected: the 78 duplicate rows are real backchannel frequency
  (`Ha'a.` said 12 times is data, not a scraping artifact), and merging already absorbs them into
  distinct chunks, so an explicit dedup step would only delete signal.
- **`ḍ` → `d`**: kept as part of the diacritic strip despite `ḍ` being a real retroflex phoneme
  (not purely orthographic like `ê`/`è`/`ě`), because dev's own orthography performs the identical
  collapse (`gedhe`→`gede`). Flagged as a judgment call: this is a phonemic merge, justified only
  because it matches the target's convention, not because the distinction doesn't matter
  linguistically.

Known-unfixed items, carried in `manifest.json["known_unfixed"]` for auditability:

| Item | Measured residual | Why it's a data problem, not a preprocessing one |
|---|---|---|
| R2 lexical gap | dev OOV still 18.5% (21.5% on `ind`-labeled clips) | No Indonesian conversational data was added in this pass; this is a training-data-mix decision, not a cleaning step. |
| R3 switching granularity | training run length 2.26 vs dev 7.22 tokens (3.2×); Jember Javanese share 0.76 vs dev's 0.18 | Structural mismatch between corpora; mitigated at train time (LoRA/low LR/few epochs), not by preprocessing. |
| R7 multi-speaker chunks | 20.6% of shipped chunks contain a >6 dB internal level step; dev is 0% multi-speaker | The turn-aware alternative exists in `meta/` but shipping it trades away duration match; needs a training-run ablation to decide, not a preprocessing default. |
| R8 boundary clipping | lead-edge hot rate unchanged at 49.1% after padding+trim; trail improved 53.6%→50.3%; dev is 39.8%/29.6% | Padding can only repair a cut if the true word boundary is recoverable from adjacent audio; whole-second annotation quantization is a source-data defect that padding mitigates but cannot fully reverse. |
| R9 unseen acronyms | 18 of dev's 27 acronym types never occur in training | Vocabulary coverage problem; a post-ASR recaser over a closed acronym set is the only real lever, and it needs dev-set validation, not preprocessing. |
| R14 digits | 96 digit characters remain in Jember text; dev has 0 (dev spells numbers out) | No attested Jember-digit → dev-word mapping was found (context-dependent: "S1" is a degree, not "S one"); left alone deliberately rather than guessing a lossy heuristic mapping. |
| R16 short clips | nothing in dev is shorter than 6.29 s, though the spec advertises a 2 s floor | Unvalidatable from this data; the inference path on synthetic 2–6 s clips needs to be tested directly once a model exists. |
| — no `jav`-only dev data | 0 Javanese-only clips exist in dev | Any claim about Javanese-only WER is speculation until such data appears; cannot be fixed by preprocessing. |
| — dev is only 4 speakers / 2 conversations | 1 WER point = 175 word errors | Every comparison must be bootstrapped; sub-point moves should be treated as noise, not signal. |

---

## 5. What to check again once real model training starts

1. **Ablate `edge_pad_s`** (0 vs 0.15 s) and the **turn-aware vs plain merge** variant — both are
   in `meta/` specifically so this doesn't require re-running the whole pipeline.
2. **Track `ind`/`javind`/overall WER separately**, and treat the practical Indonesian-guard-rail
   evaluation slice as ~108 clips (20 `ind` + 88 near-Indonesian `javind` with zero Javanese
   function words), not the labeled 20.
3. **Watch for over-switching** (R3): if a fine-tuned model's decoded output shows abnormally
   short same-language runs relative to dev's ~7-token average, that is the switching-prior
   mismatch predicted here, not a training bug.
4. **Validate the quality-gate flags are useful** (`flag_offmic`, `flag_low_snr`,
   `flag_clipped_batch`) by checking whether down-weighting or excluding them changes dev WER,
   rather than assuming the flags were the right call.
5. **Test inference on synthetic short clips (2–6 s)** — the R16 gap is currently unvalidated by
   any real data.
6. **Re-check `manifest.json["config"]`** before any second preprocessing run — thresholds here
   were derived from this specific data snapshot, and should be re-justified rather than reused by
   default if the source data changes.
7. **Confirm session-disjoint split safety** empirically if speaker information ever becomes
   available for Jember — the current split is a floor, not a proven guarantee.

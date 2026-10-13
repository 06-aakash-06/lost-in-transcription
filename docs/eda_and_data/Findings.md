# Findings — Deep EDA Audit, Jember + Official Dev Set

**Scope.** This document extends `COMPETITION_ANALYSIS.md` (written from `Main1.ipynb`, Parts 1–2).
It does **not** repeat Main1's findings; it records what a second, deeper audit pass measured that
Main1 did not. Where this document restates a Main1 number it is because the audit *changed* it.

Every figure below was computed in this run from the files on disk. Reproduced in `Main2.ipynb`
Part 3. Measurement method is stated wherever the method materially affects the number.

**What Main1 already covered (not repeated here):** segment-duration distribution; the
scorer-compatible normaliser `wer_norm` and a Levenshtein WER; dev-set language/speaker/convo
counts; the mixed 44.1/48 kHz discovery; the 41 unusable Jember rows; whole-second timestamp
quantisation; the contiguity proof and ≤29 s merge simulation; Jember↔dev vocabulary overlap and
diacritic-stripping OOV gain; reduplication / acronym / ellipsis frequency ratios; the 30 s Whisper
window count.

**What Main1 did not cover, and this audit adds:** every audio-quality dimension (SNR, clipping,
loudness, silence, per-speaker variation) — Main1 never decoded a single sample; the exact raw
character and markup inventory; code-switching *granularity*; per-language-label OOV; character
n-gram overlap; spelling-variance clustering; segment-level audio integrity; the speaker-turn
structure problem created by merging; and three concrete defects in the current tooling.

---

## 1. Dataset overview — confirmed vs. spec

### 1.1 Format consistency (checked on **every** file, not a sample)

| | Jember | Dev |
|---|---|---|
| Files | 208 MP3 sessions | 372 MP3 clips |
| Channels | 1 (208/208) | 1 (372/372) |
| Codec | MPEG Layer III (208/208) | MPEG Layer III (372/372) |
| Sample rate | **48 kHz ×119, 44.1 kHz ×89** | 48 kHz ×372 |
| Total audio (decoded) | 10.01 h | 2.56 h |
| Unreadable | 0 | 0 |

Both TSVs are clean UTF-8, no BOM, LF-only line endings, zero bare CR bytes, zero embedded
newlines. (The `file(1)` command reports "CRLF, CR line terminators" for the Jember TSV — that is a
`file` heuristic misfire, not real; a byte count confirms 0 CR bytes.)

The mixed sample rate is confirmed. Note it is **not** a 50/50 split and is not random: the 44.1 kHz
files are a contiguous block of the session numbering, i.e. two recording batches.

### 1.2 ⚠ New: `soundfile` reports the wrong duration for all 89 of the 44.1 kHz files

`sf.info(...).frames / samplerate` over-reports duration for **89/89** of the 44.1 kHz MP3s and for
**0/119** of the 48 kHz ones.

| | ratio decoded ÷ header | max absolute error |
|---|---|---|
| 44.1 kHz files | 0.99499 (mean), 0.9934–0.9953 | **3.92 s** (session 1: header 835.03 s, true 831.11 s) |
| 48 kHz files | 0.99987 | 0.035 s |

`ffprobe` and `librosa.load` agree with each other and disagree with `sf.info`; the header count is
the wrong one (MP3 decoder-delay/padding frames counted as audio). This is a **0.5 % systematic
over-count**, not a timeline shift — segment timestamps still line up with the audio.

**Consequence:** Main1's "end beyond file length" integrity check ran against inflated durations and
therefore *under*-reports overrun. Re-run against true decoded duration:

- Sessions annotated past the end of their audio: **4** (>0.5 s), **3** (>2 s).
- Worst: **session 30**, annotation runs to 761.0 s against 744.84 s of audio → **16.2 s of
  transcript with no audio behind it** (session 45: 2.80 s; session 12: 2.22 s; session 153: 0.80 s).
- Rows individually overrunning: 4.
- Annotation coverage against true duration: **99.7 %** (9.98 h annotated of 10.01 h).

Session 30 is a real defect, not rounding: 16 s is ~3 median segments of pure hallucination-fuel.

### 1.3 Row integrity (unchanged from Main1, re-confirmed)

6,679 rows → 3 negative-duration, 38 zero-duration, 0 blank/NaN transcripts → **6,638 usable**.
All 41 bad rows carry real text (they are annotation timing slips, not empty rows): e.g. row 89,
session 1, `0:05:52–0:05:52`, "Polusi.".

Duration is quantised to whole seconds; the modal values are 3 s (981), 4 s (932), 5 s (815).

### 1.4 Dev structure — the split is two conversations, and they are not interchangeable

| convo_id | speakers | clips | `ind` | `javind` |
|---|---|---|---|---|
| 5 | 1, 6 | 294 | 0 | 294 |
| 2 | 5, 7 | 78 | 20 | 58 |

Every `ind` clip lives in **convo 2**, from speakers 5 and 7 only. Confirmed: `jav` = 0 clips.

Words per speaker: spk1 7,373 (52.7/clip), spk6 6,565 (42.6), spk7 1,783 (40.5), spk5 1,774 (52.2).
Speakers 1 and 6 supply **80 %** of the WER denominator.

---

## 2. Language and vocabulary findings

### 2.1 OOV broken out by dev language label (new)

Token coverage of dev by the Jember vocabulary, computed under `wer_norm` tokenisation:

| dev label | clips | tokens | types | type-overlap | token-cov raw | **after diacritic strip** | + lowercase |
|---|---|---|---|---|---|---|---|
| `javind` | 352 | 16,613 | 3,170 | 42.0 % | 78.80 % | **81.82 %** | 82.09 % |
| `ind` | 20 | 860 | 346 | 60.4 % | 75.35 % | **78.02 %** | 78.26 % |
| all | 372 | 17,473 | 3,273 | 41.3 % | 78.63 % | **81.63 %** | 81.90 % |

**Jember covers the Indonesian-only clips *worse* than the code-mixed ones** — 21.98 % vs 18.18 %
OOV after stripping. The naive expectation (Jember is Javanese, so it should struggle on `javind`
and struggle *more* on `ind`) holds, and quantifies the matrix-language ceiling per label.

Top `ind` OOV: `PR, ngerjain, soalnya, bilang, maksudnya, yaudah, 'kan, mesti, disuruh, belajarnya,
kayaknya, KKM, besoknya, nilainya, gokil`. These are colloquial Jakarta-Indonesian morphology
(`-in` verbs, `-nya`) and school acronyms — a register Jember simply does not contain.

Reverse direction: **29.1 % of Jember tokens** have a (diacritic-stripped) form that never appears
anywhere in dev. Nearly a third of the training signal is lexically off-target.

### 2.2 Character n-gram overlap — the gap is lexical, not sub-lexical (new)

Dev n-gram *token* coverage by Jember, over diacritic-stripped lowercase word forms with `<` `>`
boundary markers:

| n | dev token coverage | type overlap | Jensen–Shannon divergence |
|---|---|---|---|
| 2 | **99.91 %** | 94.0 % | 0.046 |
| 3 | **98.89 %** | 86.9 % | 0.148 |
| 4 | 94.23 % | 74.2 % | 0.306 |
| 5 | 86.02 % | 60.2 % | 0.437 |

This is the single most useful reframing in the audit. **Whole-word OOV is 18.4 %, but character
3-gram coverage is 98.9 %.** The two corpora share almost all of their sub-word phonotactics and
differ in how those pieces are assembled into words.

Practical reading: a **subword/BPE model (Whisper, MMS) has near-complete sub-lexical coverage of
the target** and the 18.4 % OOV is a *language-model* problem, not an acoustic-units problem.
Corollary — do not let a character-level or grapheme-CTC framing panic anyone into thinking the
acoustic front end needs Indonesian data; it needs Indonesian *decoder prior*.

Supporting evidence: of the top 400 dev OOV types, **149 (713 tokens)** are within edit distance 1
of some Jember word. Genuine near-misses include `senang`/`seneng`, `dapat`/`dapet`,
`gede`/`gedhe`, `ekstrakulikuler`/`ekstrakurikuler`, `hilang`/`ilang`, `lihat`/`liwat`.

⚠ **Do not act on the edit-distance-1 list mechanically.** The same procedure also produces
`suka`→`suk`, `PR`→`pe`, `Solo`→`soto`, `yo`→`o`, `kui`→`ku`. Roughly half the matches are
coincidences between short unrelated words. Any fuzzy-merge rule here would inject errors.

### 2.3 Spelling variance *within* each corpus (new)

Grouping surface forms by key = lowercase(strip-diacritics(word)):

| | keys with >1 surface form | tokens affected |
|---|---|---|
| Jember | 433 | 25,404 (**38.72 %**) |
| Dev | 52 | 1,461 (**8.36 %**) |

The Jember 38.7 % figure is dominated by benign sentence-initial casing (`ya`/`Ya`, `iku`/`Iku`),
but it also contains real orthographic instability:

- `terus` → `těrus` ×756 **and** `têrus` ×4 (caron vs circumflex on the same word)
- `nek` → `nèk` ×483, `nêk` ×2, `nek` ×1 (three spellings)
- `lek` → `lèk` ×812, `lek` ×2 · `ndek` → `ndèk` ×781, `ndek` ×2

Dev's own variance is smaller but real and is **not** casing: `ke`/`ké`, `kene`/`kéné`,
`soale`/`soalé`, `biasane`/`biasané`, `e`/`é`, `museum`/`Museum` (6 vs 5 — a genuine coin-flip on
a common noun). Dev is internally inconsistent about the acute accent on ~54 characters total.

### 2.4 Cross-corpus spelling conflict (new, quantified)

**153 shared keys where Jember's dominant spelling differs from dev's**, covering **1,055 dev
tokens = 6.04 % of the WER denominator.** Ranked:

```
dev 'terus'(222)   <- jember 'těrus'(756)     dev 'banget'(139) <- jember 'bangêt'(208)
dev 'nek'(73)      <- jember 'nèk'(483)       dev 'sekolah'(40) <- jember 'sêkolah'(75)
dev 'pernah'(38)   <- jember 'pêrnah'(37)     dev 'enggak'(34)  <- jember 'ênggak'(40)
dev 'neng'(24)     <- jember 'nèng'(452)      dev 'seneng'(21)  <- jember 'sênêng'(112)
dev 'sering'(16)   <- jember 'sêring'(78)     dev 'meneh'(11)   <- jember 'mênèh'(51)
dev 'kabeh'(9)     <- jember 'kabèh'(100)     dev 'kerja'(9)    <- jember 'kêrja'(120)
```

Every one of these is fixed by diacritic stripping. This is the *direct* WER accounting for
Mismatch 1: **6.04 % of reference tokens are words the two corpora spell differently**, and 1,055
guaranteed substitutions is **6.0 WER points** if the model reproduces Jember orthography.

### 2.5 ⚠ New diacritics Main1 did not report

Main1 / `COMPETITION_ANALYSIS.md` lists circumflex ×999, grave ×356, caron ×46 for Jember. Raw
character counts over the un-normalised text find more:

| Jember | count | Dev | count |
|---|---|---|---|
| `ê` U+00EA | 5,370 | `é` U+00E9 | 54 |
| `è` U+00E8 | 4,826 | — | — |
| `ě` U+011B | 819 | | |
| **`ḍ` U+1E0D (dot below)** | **17** | | |
| `é` U+00E9 | 1 | | |

`ḍ` is a **retroflex d**, a real Javanese phoneme (`dh` in everyday spelling). NFD-stripping maps
`ḍ`→`d`, which is the correct target for dev orthography (dev writes `gedhe` as `gede`, `dhewe` as
`dewe`). Harmless, but it should be a deliberate decision, not an accident.

Jember also carries one stray `é` — Jember's own annotators leaked dev-style orthography once.

### 2.6 Code-switching granularity — a mismatch beyond matrix language (new)

Main1 established *which* language is the matrix. It did not measure *how* the switching happens.
Tagging tokens against a 130-word Javanese and 100-word Indonesian function-word lexicon
(untaggable tokens dropped), per utterance:

| | Javanese share | switch rate | **mean same-language run** |
|---|---|---|---|
| Jember | 0.758 | 0.274 | **2.26 tokens** |
| Dev (all) | 0.178 | 0.178 | **7.22 tokens** |
| Dev `javind` | 0.184 | 0.182 | 7.12 |
| Dev `ind` | 0.062 | 0.103 | 8.90 |

**Jember switches 1.5× more densely than dev by switch rate, and its runs are 3.2× shorter**
(2.26 vs 7.22 tokens). Jember is intra-clausal insertion — Javanese and Indonesian words alternate
every ~2 tokens. Dev is clause-level alternation — a speaker runs ~7 tokens in one language, then
flips.

This is a second, independent axis of domain gap. A model fine-tuned on Jember learns a
*high-frequency* switching prior and will over-switch on dev, substituting Javanese function words
into Indonesian clauses that never switched. It is not fixable by data cleaning; it is a reason to
keep fine-tuning gentle (LoRA / low LR / few epochs) and to blend Indonesian data.

Distribution of the Javanese share within `javind` is extremely wide: mean 0.184, std 0.199,
min 0.000, p10 0.000, median 0.121, p90 0.497, max 0.900. **88 of 352 `javind` clips (25.0 %)
contain zero Javanese function words at all.**

**⇒ The `language` label is conversation-level, not clip-level.** The practical `ind`-like
evaluation slice is not 20 clips, it is ~108 (20 `ind` + 88 near-Indonesian `javind`). That is a
usable guard-rail; the 20-clip `ind` slice alone is not.

### 2.7 Digits (new)

Jember contains 96 digit characters (`0`×18, `1`×26, `2`×28 …), inside acronyms (`S1`, `S2`, `M2`,
`G30S`, `3D`, `H1`) and a couple of bare numbers. **Dev contains zero digit characters.** Dev spells
every number out. A Jember-trained model that emits `2` where dev writes `dua` scores a
substitution. Small (0.15 % of Jember tokens) but free to handle.

---

## 3. Audio quality findings

**Method.** Every file decoded once to 16 kHz mono; frame energy on 25 ms / 10 ms frames.
Noise floor = 5th percentile frame dBFS **computed at session level** (for Jember segments) or
clip level (for dev). Speech frames = frames above floor + 12 dB. SNR = p90(frame dB) − floor.

> **Methodological note, and a trap worth recording.** A first pass computed the noise floor
> *per segment*. That produced 44 Jember segments with `silence_frac == 1.0` — apparently silent
> segments carrying full transcripts. They are not silent. When a short segment is wall-to-wall
> speech, the 10th-percentile frame *is speech*, so nothing clears floor+10 dB and the segment
> measures as 100 % silence. Per-segment adaptive thresholding is invalid on segments shorter than
> the pause structure it is trying to detect. All numbers below use session-level floors.

### 3.1 Dev clips

| metric | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| SNR (dB) | 21.5 | 33.9 | 48.2 | 76.7 | 79.9 |
| noise floor (dBFS) | −96.0 | −94.4 | −69.0 | −50.2 | −38.7 |
| RMS (dBFS) | −28.9 | −27.3 | −22.3 | −19.1 | −17.1 |
| silence fraction | 0.05 | 0.07 | 0.18 | 0.33 | 0.50 |
| leading silence (s) | 0.00 | 0.00 | 0.14 | 0.63 | 1.50 |
| trailing silence (s) | 0.00 | 0.00 | 0.15 | 0.74 | 1.89 |

- **Zero clips below 20 dB SNR.** 7 below 25 dB, 14 below 30 dB. Dev audio is clean.
- **Zero meaningful clipping.** 62 clips touch |x| ≥ 0.999, but no clip has more than 0.1 % of
  samples there; the worst is 0.069 %. 60 clips have decoded peak > 1.0 — MP3 inter-sample
  overshoot, not recorded clipping. **Do not filter dev-like audio on peak.**
- Zero DC offset anywhere (max |mean| = 0.001).
- Leading silence > 0.5 s: **44 clips (11.8 %)**; > 1 s: 4. Trailing > 0.5 s: 58; > 1 s: 10.
  Total edge silence 162 s of 9,222 s = **1.76 %**. Dev is already tightly cropped.
- Clip loudness spread is only **11.8 dB** end to end (IQR −25.7 to −20.6 dBFS).

Worst-SNR clips are all speaker 1, convo 5 — a cluster, not scattered:
`1eab7054cf85481eadb87dbc088ab15b.mp3` (16.3 dB by the per-clip metric, 21.5 dB by the final
metric), `f6a9cb66816849a2b3799f382eaf13fa.mp3`, `2abec9cc7a094912a2f0f8cea5ac0747.mp3`,
`dbe5aed9727e4930b51f470e6c030c33.mp3`. Highest peaks: `f09d639c48ca4dbdaab211a56cb10a82.mp3`
(peak 1.424, 0.069 % of samples at full scale).

### 3.2 Per-speaker / per-conversation variation in dev (new)

| convo | speaker | n | mean dur | SNR | floor | RMS | silence | lead sil | trail sil |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 5 | 34 | 27.4 s | 56.6 | −71.7 | −19.3 | 0.15 | 0.44 | 0.14 |
| 2 | 7 | 44 | 21.2 s | 66.8 | −84.8 | −22.5 | 0.14 | 0.29 | 0.19 |
| 5 | 1 | 140 | 26.9 s | 45.5 | −62.0 | −20.8 | 0.17 | 0.32 | 0.26 |
| 5 | 6 | 154 | 23.4 s | 47.4 | −69.4 | **−25.9** | 0.21 | **0.01** | 0.26 |

Four distinct recording conditions. Speaker 6 is **5.1 dB quieter** than speaker 1 and has
essentially **zero leading silence** (0.01 s mean vs 0.32 s) — a different recording app or a
different hard-crop. Speaker 7's noise floor is 22.8 dB below speaker 1's.

Within-speaker consistency is *very* high — RMS p90−p10 is only 2.0–3.9 dB per speaker. **Dev clips
are studio-consistent within a speaker and vary between speakers.** The hidden test set is drawn
from other speakers, so between-speaker variation of this magnitude (±5 dB level, ±20 dB floor) is
exactly what the model must be robust to.

### 3.3 Jember sessions

| metric | min | p5 | median | p95 | max |
|---|---|---|---|---|---|
| SNR (dB) | 14.6 | 19.7 | 36.0 | 71.3 | 84.3 |
| RMS (dBFS) | **−42.5** | −35.6 | −25.8 | −16.5 | **−12.4** |
| silence fraction | 0.06 | 0.10 | 0.31 | 0.61 | 0.83 |

- **Loudness spread is 30.1 dB** (−42.5 to −12.4 dBFS) against dev's 11.8 dB. This is the largest
  audio-domain gap in the corpora and Main1 never saw it.
- **14 sessions below 20 dB SNR**, 36 below 25 dB. Worst: session 28 (14.6 dB), 26, 100, 101, 29,
  35, 30, 64.
- **Real clipping, in a contiguous batch:** sessions **194, 195, 196, 197, 198** each have >0.1 % of
  samples at full scale — session 196 at **1.27 %**, 197 at 1.03 %, 195 at 0.79 %. Their RMS is
  −12.4 to −14.8 dBFS, i.e. recorded ~10 dB hotter than the corpus. Five consecutive session IDs =
  one recording session with the gain set wrong. These are the only files in either corpus where
  clipping is a real distortion rather than MP3 overshoot.

### 3.4 Jember segments

Over 6,638 usable segments, using session-level floors:

| metric | p5 | median | p95 |
|---|---|---|---|
| SNR (dB) | 17.7 | 32.0 | 68.9 |
| silence fraction | 0.03 | 0.33 | 0.70 |
| leading silence (s) | 0.00 | 0.00 | 0.81 |
| trailing silence (s) | 0.00 | 0.00 | 1.14 |

- SNR < 10 dB: **58 segments (0.87 %)**. SNR < 15 dB: 172 (2.59 %).
- silence > 80 %: 171 (2.58 %). > 95 %: 57 (0.86 %). `speech_frac == 0`: 15.
- **Boundary-clipped onsets: 55.6 % of segments have speech in their very first frame** (vs 39.8 %
  of dev clips); 57.2 % have speech in the last frame (dev 30.1 %); **35.4 % are hot at both
  edges** (dev ~12 %). Combined with the ±0.5 s timestamp quantisation this is the mechanism behind
  Main1's Mismatch 4, now measured directly on the waveform rather than inferred from timestamps.
- Segment loudness p5–p95 spread: **22.1 dB**.

### 3.5 ⚠ New: Jember has an off-mic second speaker

**136 segments (2.05 %, 1,039 words = 1.58 % of Jember text)** measure > 85 % silence against
their own session's floor while carrying a full transcript. These are not annotation errors:

- **58.1 % of them end in a question mark, against a corpus baseline of 8.3 %** — a 7× enrichment.
- Their mean RMS is **−42.0 dBFS vs −29.2 dBFS** for the rest of their own sessions: ~13 dB down.
- Independent confirmation: of segments >10 dB below their session median, **51.7 %** end in `?`.

These are the **interviewer's questions, recorded far from the microphone.** Jember is elicited
dialogue; the interviewee is on-mic and the interviewer is not. Example, session 91:

```
0:00:14-0:00:19  Lèk kuthane, kutha Jember.                                    (on-mic, answer)
0:00:19-0:00:21  Saiki kuliah ndèk êndi?              RMS -48.2 dBFS  <-- off-mic question
0:00:21-0:00:27  Aku kuliahe ndèk Universitas ...                              (on-mic, answer)
0:00:27-0:00:28  Jurusan apa?                         RMS -49.8 dBFS  <-- off-mic question
0:00:28-0:00:32  Lèk jurusanku dhewe iki, ...                                  (on-mic, answer)
0:00:32-0:00:36  Alasane mbiyèn kok sampeyan ...      RMS -48.7 dBFS  <-- off-mic question
```

Within-session segment loudness spread (p90−p10) is a **median 5.1 dB, max 13.6 dB** across the 163
sessions with ≥10 segments — against **2.0–3.9 dB** for a dev speaker. The worst sessions
(176: 13.6 dB, 177: 12.8, 70: 12.7, 91: 11.6, 92: 10.9) are the interview-format ones.

### 3.6 ⚠ New: merging Jember segments creates multi-speaker clips; dev clips are single-speaker

Each dev clip carries exactly one `speaker` label — dev is a set of **single-speaker voice notes**.
Jember sessions are two-speaker dialogues, and the ≤29 s merge crosses speaker turns.

Using an adjacent-segment RMS jump as a turn proxy (validated: at >6 dB jump, the preceding segment
ends in `?` 21.2 % of the time vs an 8.5 % base rate; at >10 dB, 29.3 %):

| | value |
|---|---|
| adjacent-segment RMS jump, median / p90 / p99 | 1.78 / 5.57 / 12.20 dB |
| jumps > 6 dB | 560 of 6,430 boundaries (8.7 %) |
| **merged chunks containing an internal >6 dB jump** | **303 of 1,473 (20.6 %)** |
| merged chunks with an internal RMS spread > 8 dB | 17.2 % |
| merged chunks containing a mid-chunk `?` | 25.2 % |

**Roughly one in four merged 26 s training chunks is a two-speaker exchange with a level step in
the middle, while 100 % of dev clips are one speaker at a stable level.** Merging remains the right
call — the duration match it buys is worth far more — but this is a real, previously unrecorded
cost of it, and it is partly avoidable by making the merge turn-aware.

### 3.7 Duration vs. the stated 2–40 s test range

Dev: min **6.29 s**, max **37.81 s**. Zero clips < 2 s, zero > 40 s, 37 clips > 30 s. Only one clip
is a >3σ duration outlier. The stated 2–40 s range is a superset of what dev actually contains —
the low end (2–6 s) is **entirely unrepresented in dev**, so nothing local validates behaviour on
short test clips.

Shortest dev clips: `6adf01d8685c492cac9fb8b7c49d80c1.mp3` (6.29 s, 11 words),
`569bbba53a544d6b8f0c0ec21c1ea46f.mp3` (10.12 s, 16 words).

---

## 4. Transcript convention findings

### 4.1 The bracketed tags in the spec do not exist. What does exist is different.

Exhaustive regex sweep for `[...]`, `(...)`, `<...>`, `{...}` across all 7,051 transcripts:

**Zero occurrences of any bracket type in either corpus. Zero unbalanced brackets.**
No `[laugh]`, no `(???)`, no `[???]`. Main1 noted this; the audit confirms it exhaustively, which
means the `BRACKET_TAG_RE` and `UNINTEL_RE` branches of `wer_norm` are dead code on our data.

The markup that *does* exist is a raw-character inventory:

| char | Jember | Dev | note |
|---|---|---|---|
| `,` | 10,936 | 1,935 | scorer removes |
| `.` | 8,524 | 2,140 | scorer removes except in ellipses |
| `-` | 1,965 | 592 | reduplication / clitics — **scorer keeps** |
| `?` | 644 | 64 | scorer removes |
| `'` U+0027 | 250 | 56 | `Ha'a`, `'kan` — **scorer keeps** |
| `"` U+0022 | 206 | 37 | scorer removes |
| `!` | 29 | 7 | scorer removes |
| **`“` U+201C** | **29** | 0 | ⚠ **scorer does NOT remove** |
| **`”` U+201D** | **28** | 0 | ⚠ **scorer does NOT remove** |
| **`’` U+2019** | **19** | 0 | ⚠ **scorer does NOT remove** |
| **`…` U+2026** | **4** | 0 | ⚠ not matched by the `\.{3,}` ellipsis logic |
| `%` `;` `:` `+` | 2 / 1 / 1 / 1 | 0 / 0 / 1 / 0 | |

### 4.2 ⚠ Defect: `wer_norm` does not handle the Unicode punctuation Jember contains

The scorer's removal set is `, ? ¿ ¡ ! " ; :` — all ASCII. Jember contains **80 curly
quotes/apostrophes and 4 U+2026 ellipses** that survive normalisation and stay glued to their
tokens (`“ngono` ≠ `ngono`). Dev contains **none** of these characters, so this is purely a
training-text contamination issue — it cannot help, and it silently poisons ~80 training tokens
plus 4 that the disfluency-preserving ellipsis logic will mis-handle.

Fix belongs in the *training-text* normaliser (map `“ ” → "`, `’ → '`, `… → ...` **before**
scorer normalisation), never in the scoring path — the scoring path must stay bit-identical to the
official scorer.

### 4.3 Casing conventions

| | Jember | Dev |
|---|---|---|
| whitespace tokens | 65,743 | 17,495 |
| ALL-CAPS (≥2 letters) | 302 (0.46 %) | 259 (**1.48 %**) |
| Initial-capital | 8,404 (12.78 %) | 1,524 (8.71 %) |
| lowercase-initial | 56,538 (86.00 %) | 15,574 (89.02 %) |
| utterances starting with a capital | 99.2 % | 99.5 % |
| utterances ending in `.` | 89.1 % | 89.0 % |

Both corpora use the same convention — sentence-case, terminal period, capitalised proper nouns and
acronyms. The scorer lowercases only sentence-initial letters, so this convention is *compatible*
between corpora. The difference is **density**: dev uses 3.2× more ALL-CAPS tokens.

Jember ALL-CAPS types (64) skew institutional-Javanese-East-Java: `UMJ, UNMUH, DPRD, PKI, G30S,
CPNS, TKI, WHV, LKSA, RPL, MOBA, FYP, GPT, AI, FOMO`. It also contains **`KARDIMAN`** — a personal
name in full caps, which is an annotation inconsistency, and `HPHP`/`SMASMA`/`PRPR` artefacts that
appear only when hyphens are stripped from reduplicated acronyms (`HP-HP`).

Dev ALL-CAPS types (27): `BB, BBM, CDMA, FYI, GSM, HP, IPA, IPS, KKM, OSIS, PMR, PR, QWERTY, ROHIS,
SD, SHU, SM, SMA, SMP, SMS, TK, TV, UASBN, UN`. Shared with Jember: only `HP, IPA, IPS, SD, SMA,
SMP, TK, TV`.

### 4.4 Punctuation and sentence structure — the biggest *register* difference

| | Jember | Dev | ratio |
|---|---|---|---|
| utterances with a comma | 71.5 % | **97.6 %** | 1.4× |
| with `...` | 10.7 % | **33.6 %** | 3.1× |
| with `-` | 23.4 % | **75.3 %** | 3.2× |
| with `'` | 3.5 % | 9.7 % | 2.8× |
| with `?` | 9.3 % | 12.9 % | 1.4× |
| **containing >1 sentence-final mark** | **14.8 %** | **92.2 %** | **6.2×** |

The multi-sentence figure is the one that matters and it is a direct artefact of segmentation:
a 5 s Jember segment is one clause; a 26 s dev clip is a paragraph. **After merging to 26 s this
gap should close on its own** — a testable prediction, verified in `Preprocess.md` §Validation.

### 4.5 Reduplication and hyphenation

| | Jember | Dev |
|---|---|---|
| hyphenated tokens | 1,962 (2.98 %) | 589 (3.37 %) |
| exact reduplication `X-X` | 1,439 | 458 |
| other hyphenation | 523 | 131 |

Per-token rates are nearly identical (2.98 % vs 3.37 %) — Main1's "3.8× more reduplication" is a
per-*utterance* rate, inflated by the 5× duration gap. Per token there is no reduplication gap.

What differs is the stems: Jember `wong-wong`(120), `arèk-arèk`(36), `rêsik-rêsik`(12) vs dev
`jalan-jalan`(17), `orang-orang`(11), `teman-teman`(7). And the non-exact forms are a live
productive morphology in both: `nge-gym`, `nge-game`, `kanca-kancaku`, `woh-wohan`,
`booming-booming-nya`, `SMA-ku`, `A-nya`, `sashimi-ne`, `deg-degan`, `dipikir-pikir`.

**The hyphen is a scored character** — `jalan-jalan` and `jalan jalan` are different tokenisations
with different WER. Never split on hyphens, never strip them.

### 4.6 Disfluency

| | Jember | Dev |
|---|---|---|
| word-attached ellipsis tokens | 768 | 227 |
| **standalone ellipsis** | **1** | **20** |
| top forms | `e...`(498), `E...`(136), `Em...`(26), `a...`(17) | `e...`(71), `ya...`(12), `yang...`(8) |

Jember's disfluency is concentrated in a hesitation vocable (`e...` + `E...` = 634 of 768, 82.6 %).
Dev's is spread across real words trailing off (`yang...`, `namanya...`, `kidul...`, `enggak...`) —
**dev marks mid-word abandonment, Jember marks filled pauses.** Different phenomena under the same
notation, both scored tokens.

The 20 standalone ellipses in dev are correctly dropped by `wer_norm`; the word-attached ones are
correctly kept. Verified against the implementation.

---

## 5. Label integrity, leakage and duplication

- **Jember has no speaker or conversation IDs.** The TSV is `Audio file name, start, end, text`.
  Session file is the only leakage-safe split key — confirmed. §3.5 shows each session contains
  **two** speakers, so a session-disjoint split is speaker-disjoint only if no speaker recurs
  across sessions, which is unverifiable. Treat session-disjoint as a floor, not a guarantee.
- **Dev cannot be split at all.** 4 speakers, 2 conversations, and every `ind` clip sits in one of
  them. Any dev sub-split is a 1–2 speaker sample. Use dev whole, and bootstrap.
- **Duplicate transcripts, Jember: 78 duplicate rows across 27 distinct texts.** All short
  backchannels — `Ha'a.`(12), `Iku sih Mbak.`(11), `Apa ya?`(7), `Těrus.`(5), `Ngono.`(5),
  `He'em.`(5). Longest duplicated string is 25 characters (`Tanggane duwe apa, panas.`). These are
  genuine repeated utterances,
  not copy-paste errors. **Do not deduplicate them** — they are real backchannel frequency, and
  after merging they are absorbed into distinct longer chunks anyway.
- **Zero duplicate audio, zero duplicate transcripts, zero duplicate filenames in dev.**
  Four (duration, RMS) fingerprint collisions in dev at 2-decimal precision; inspected, all are
  coincidences between distinct clips, not duplicates.
- Jember (duration, RMS, text) triples: 0 collisions.
- Sessions referenced but missing: none. Audio never referenced: none.

### 5.1 Words-per-second as a segmentation-quality proxy

| | mean | median | p1 | p99 | max |
|---|---|---|---|---|---|
| Dev (wall clock) | 1.89 | 1.86 | 1.43 | 2.58 | 3.63 |
| Jember (wall clock) | 1.99 | 2.00 | 0.50 | 5.00 | **10.00** |
| Dev (per *speech* second) | — | 2.30 | | | |
| Jember (per *speech* second) | — | **2.96** | | | |

Wall-clock rates look similar; **speech-adjusted rates differ by 29 %.** Jember segments are
annotator-cropped tight to the speech, dev clips retain natural pauses. Implication: aggressive VAD
trimming of Jember would push it *further* from dev, not closer. Trim edges only.

Jember's tail is quantisation, not fast talking: the 10.0 w/s record is session 35, `0:00:00–0:00:01`
— 10 words annotated inside a 1-second window that is really ~4 s of audio rounded to whole
seconds. Flagged outliers:

- **wps > 4 on a ≤2 s segment: 80 rows** — near-certainly under-timed boundaries.
- **wps < 0.5: 53 rows**, of which **28 also measure >60 % silence** — over-timed boundaries,
  a short utterance annotated across a long pause.
- **silence > 60 % and wps > 3 simultaneously: 21 rows** — internally inconsistent, worst class.
  (Union of all three bad-segmentation classes: **149 rows, 2.2 % of usable rows.**)

All three classes are absorbed by merging (a 1 s under-timed segment merges into a 26 s chunk whose
total duration is correct), which is a third independent argument for the merge.

---

## 6. Risk list — what costs WER if not handled

Ordered by measured exposure. "Exposure" = share of the 17,473-word dev denominator at risk.

| # | Risk | Measured exposure | Mechanism |
|---|---|---|---|
| R1 | Jember diacritics reproduced at inference | **6.04 %** of dev tokens are words the corpora spell differently (1,055 tokens ≈ **6.0 WER pts**); 14.3 % of Jember tokens carry a mark | Straight substitutions |
| R2 | Indonesian matrix not represented in training | **18.4 %** OOV after stripping; 21.98 % on `ind`; 29.1 % of Jember tokens never occur in dev | Decoder prior drifts Javanese |
| R3 | Over-switching learned from Jember | Jember run length is **3.2×** shorter (2.26 vs 7.22 tokens); switch rate 1.5× higher | Function-word substitutions mid-clause |
| R4 | Duration prior (5 s train vs 26 s test) | 1.6 % vs 92.7 % in the 15–32 s band | Early EOS → deletions; repetition loops → insertions |
| R5 | Loudness domain gap | Jember spans **30.1 dB** RMS, dev **11.8 dB**; Jember segments p5–p95 = 22.1 dB | Encoder sees levels it was not adapted to |
| R6 | Off-mic interviewer segments in training | **136 segments / 1,039 words (1.58 %)**, 13 dB below their session; 58.1 % are questions | Teaches the model to transcribe near-inaudible audio → hallucination on quiet test passages |
| R7 | Merged chunks are multi-speaker; dev is not | **20.6 %** of merged chunks contain a >6 dB internal level step; **100 %** of dev clips are single-speaker | Turn-taking prior the target never exercises |
| R8 | Boundary-clipped word edges | **55.6 %** of Jember segments start hot, 57.2 % end hot, **35.4 %** both (dev: 39.8 / 30.1) | Truncated onsets/codas → edge hallucination |
| R9 | Casing / acronyms | dev ALL-CAPS density **1.48 %** vs Jember 0.46 %; only 8 of dev's 27 acronym types occur in Jember | Scorer protects acronyms; `smp` for `SMP` is a full substitution |
| R10 | Unicode punctuation in training text | **80** curly quotes/apostrophes + **4** `…` survive `wer_norm` | Glued to tokens → guaranteed substitutions in training targets |
| R11 | Session 30 transcript beyond its audio | **16.2 s** of text with no audio | Trains the model to invent text at chunk ends |
| R12 | Clipped recording batch | sessions **194–198**, up to **1.27 %** of samples at full scale | Distorted spectra, ~10 dB hot |
| R13 | Bad Jember segmentation | 80 rows wps>4 on ≤2 s; 53 rows wps<0.5 (28 also >60 % silent); 43 rows both | Text/audio misalignment |
| R14 | Digits in training, none in test | 96 digit chars in Jember, **0** in dev | `2` vs `dua` substitutions |
| R15 | Near-silent training segments | 57 segments >95 % silent, 15 with `speech_frac == 0`, 58 below 10 dB SNR | Text with no acoustic support → hallucination |
| R16 | Short test clips unvalidated | dev min **6.29 s**; the spec advertises **2 s** | Nothing local tests 2–6 s behaviour |
| R17 | `sf.info` duration bug on 44.1 kHz files | 89/89 files, up to 3.92 s over-reported | Silent off-by-N in any slicing that trusts the header |

---

## 7. Recommendations — what preprocessing must address, ranked

Ranked by (measured exposure × confidence) ÷ cost. **1–5 are near-free and should land before any
training run.** 6–9 are cheap but need an ablation to confirm. 10 is out of preprocessing's scope.

| # | Action | Addresses | Expected effect | Confidence |
|---|---|---|---|---|
| 1 | **Strip diacritics from all training text** (NFD → drop Mn), including `ě` and `ḍ` | R1 | 1,055 dev tokens stop being systematic substitutions; OOV 21.4 % → 18.4 % | **High — measured** |
| 2 | **Merge to ≤29 s chunks** | R4, R8, R13 | median 5 s → 26.0 s vs dev 26.2 s; in-band 1.6 % → 92.5 % | **High — measured** |
| 3 | **Fix the training-text normaliser first**: `“ ” → "`, `’ → '`, `… → ...`, before scorer normalisation. Leave the scoring path untouched. | R10 | 84 poisoned tokens removed; costs nothing | **High** |
| 4 | **Resample to 16 kHz mono** with a real decoder (`librosa`/`ffmpeg`), never trusting `sf.info` frame counts | R17 | Handles the 44.1/48 kHz split correctly | **High** |
| 5 | **Drop the 41 invalid rows; truncate every segment to true decoded duration** (fixes session 30's 16.2 s and 3 others) | R11 | Removes 16 s of unbacked text | **High** |
| 6 | **Peak-normalise each training chunk to a common target** (e.g. −1 dBFS peak with a −23 dBFS RMS target, no compression) | R5, R6 | Collapses Jember's 30.1 dB spread toward dev's 11.8 dB; brings off-mic segments into range | Medium-high |
| 7 | **Edge-only VAD trim, padded** — trim leading/trailing non-speech to a fixed 0.15–0.25 s margin, matching dev's median 0.14 s / 0.15 s. **Do not touch internal pauses.** | R4, R8 | Jember speech-adjusted 2.96 w/s already exceeds dev's 2.30; internal trimming would widen the gap | **Medium-high — direction is measured** |
| 8 | **Quality-flag, do not blanket-drop.** Emit per-chunk SNR, RMS, clipping, silence. Drop only the hard-degenerate cases (`speech_frac == 0`: 15 rows; SNR < 10 dB: 58 rows; >95 % silence: 57 rows). Flag the rest (194–198 clipping; 136 off-mic segments) for ablation. | R6, R12, R15 | ≤0.5 % of rows dropped; the model keeps hard-but-real acoustic diversity | Medium — thresholds justified, effect not |
| 9 | **Turn-aware merge variant** — prefer breaking a chunk at boundaries with a >6 dB level step. Keep both variants and ablate. | R7 | Would cut the 20.6 % multi-speaker rate; unknown cost in chunk-length fit | **Low — hypothesis, must be measured** |
| 10 | **Blend Indonesian conversational data.** No preprocessing step fixes R2 or R3. | R2, R3 | Largest remaining lever; out of scope for this pass | High that it matters, unknown magnitude |

### Explicitly rejected, with reasons

- **Do not lowercase.** dev 1.48 % ALL-CAPS + protected acronyms; blanket lowercasing forfeits
  ~3.3 WER points before the model runs (Main1 §21).
- **Do not strip hyphens or apostrophes.** Both are scored characters; 3.37 % of dev tokens are
  hyphenated and `Ha'a` / `'kan` are real tokens.
- **Do not strip disfluency markup.** `e...` is a scored token — 227 word-attached ellipses in dev.
- **Do not fuzzy-merge OOV words by edit distance.** §2.2: about half of the edit-distance-1
  matches (`suka`→`suk`, `Solo`→`soto`) are coincidences that would inject new errors.
- **Do not deduplicate Jember's 78 repeated backchannels.** They are real utterance frequency.
- **Do not filter on peak amplitude.** 60 of 372 *dev* clips exceed peak 1.0 from MP3 overshoot; a
  peak filter would reject the target distribution itself.
- **Do not apply internal-silence VAD.** §5.1 — Jember is already 29 % denser than dev in
  words-per-speech-second. Trimming internal pauses moves training *away* from the target.

### Open judgment calls flagged for a human

1. **Recommendation 9 (turn-aware merge) is a hypothesis, not a measurement.** It trades a known
   good (26 s duration match) against a suspected bad (20.6 % multi-speaker chunks). Both variants
   are built; which one ships needs a training run.
2. **Recommendation 6 (loudness normalisation)** could plausibly *hurt* by removing the level
   diversity that makes a model robust. The argument for it is that Jember's spread is 2.5× dev's
   and includes 13 dB-down off-mic speech no test clip resembles. Built as a switch, defaulted on.
3. **`ḍ` → `d`** is a phonemic merge, not just a diacritic strip. It is right for this target
   (dev writes `gede` not `gedhe`) but it is a genuine linguistic distinction being discarded.
   17 characters, so the stakes are tiny; recorded because it should be a decision, not an accident.
4. **The 136 off-mic segments are kept and flagged, not dropped.** They are 1.58 % of Jember text
   and they are also the only quiet-speech training material in the corpus. Dropping them is
   defensible; the ablation is cheap.

---

*Measured in this run against the files on disk. Reproduced end to end in `Main2.ipynb` Part 3.
Where this document and `COMPETITION_ANALYSIS.md` disagree, the disagreement is marked ⚠ and the
reason is given.*

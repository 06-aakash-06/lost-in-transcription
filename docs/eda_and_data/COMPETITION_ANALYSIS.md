# Lost In Transcription — Indonesian–Javanese Track
## Complete Competition Analysis & Path to Maximum Accuracy

**Status:** written after a full EDA pass over both corpora on disk (`Main1.ipynb`, Parts 1–2).
Every number below is **measured**, not quoted from the brief. Where the brief and the data
disagree, the data wins and the disagreement is flagged.

---

# Part I — What This Competition Actually Asks

## 1. The one-sentence version

> Given a 2–40 second voice-note clip of two Indonesians talking naturally — sliding between
> Indonesian and Javanese mid-sentence — output the exact words they said, spelled and cased the way
> a human annotator would have written them.

**Metric:** Word Error Rate. **Prize:** \$4,000 / \$2,000. **Deadline:** 25 Sep 2026.
**Runtime:** A100 80 GB, 2 h wall clock, Python 3.12, **no network**.

## 2. The thing most teams will get wrong

This looks like a fine-tuning competition. It is not primarily a fine-tuning competition.

The organisers handed us a training corpus (**Jember**) and a validation corpus (**dev**) that are
**not the same kind of speech**. Jember is Javanese-matrix village conversation transcribed by
linguists in scholarly orthography. Dev is Indonesian-matrix young-adult voice notes transcribed in
everyday spelling. A team that does the obvious thing — fine-tune Whisper on Jember, submit — will
train a model that is confidently, systematically wrong in ways WER punishes brutally.

**The competition is won in the data-preparation layer, then in decoding, then in the model.**
That ordering is the central claim of this document, and Part III proves it with numbers.

## 3. WER is not "roughly how wrong you are"

```
WER = (Substitutions + Deletions + Insertions) / Reference words
```

Three properties drive every decision that follows:

| Property | Consequence |
|---|---|
| Whole-token matching | `terus` vs `těrus` = **1 full error**, same cost as transcribing a different word entirely. Near-misses earn nothing. |
| Insertions are counted | Hallucinating text is as expensive as missing it. A repetition loop on one clip can cost more than getting five clips wrong. |
| Denominator is reference words | Our dev denominator is **17,473 words**. One WER point = **175 word errors**. Anything smaller than that is noise. |

### 3.1 The normaliser is part of the metric

The official scorer normalises *both* sides before aligning. Reproducing it exactly is free WER, and
getting it wrong means optimising a metric the leaderboard does not use. Implemented as `wer_norm()`
in §13 of the notebook:

- Strip bracketed tags `[laugh]` and unintelligibility markers `(???)`, `[???]`.
- Remove `, ? ¿ ¡ ! " ; :`
- Remove periods — **except inside ellipses**.
- Drop *standalone* ellipses; **keep word-attached ones** (`e...`, `namané...`) — they encode disfluency and are **scored tokens we must emit**.
- Lowercase **only** sentence-initial letters (start of transcript, and after `.` `!` `?` `—`).
- **Acronyms are protected.** Two consecutive capitals are left alone.
- Collapse whitespace.

> **Casing everywhere else is compared literally.** We measured **3.28% of dev tokens** as
> mid-sentence capitals. Blanket-lowercasing our output would forfeit up to 3.3 WER points before
> the acoustic model does anything.

---

# Part II — The Data, As It Actually Exists On Disk

Verified inventory. Two discrepancies with the brief are marked ⚠.

## 4. Jember Javanese Spontaneous Speech Corpus — training

| Property | Measured |
|---|---|
| Session MP3s | **208**, all readable, all mono |
| Sample rate | ⚠ **Mixed: 119 files @ 48 kHz, 89 @ 44.1 kHz** (brief says 44.1 kHz only) |
| Total audio | 10.03 h |
| TSV rows | 6,679 |
| Unusable rows | **41** (3 negative-duration, 38 zero-duration) → **6,638 usable, 9.98 h** |
| Segment duration | mean 5.4 s, **median 5 s** |
| Words | 65,614 normalised tokens, 7,890 types |
| Timestamps | ⚠ **Quantised to whole seconds** — zero fractional values in 13,358 |
| Annotation coverage | 98.8% of session audio; **adjacent segments are 100% gapless** |
| Speaker / conversation IDs | ⚠ **Absent** — the TSV has only `Audio file name, start, end, text` |

## 5. Official development set — validation only

| Property | Measured |
|---|---|
| Clips | 372, all readable |
| Format | 48 kHz, mono, MP3 |
| Total audio | 2.56 h |
| Duration | mean **24.8 s**, median **26.2 s**, min 6.3 s, **max 37.8 s** |
| Reference words | **17,473** (the WER denominator) |
| Words per clip | mean 47, median 48, p95 66 |
| Speakers | 4 (IDs 1, 5, 6, 7) |
| Conversations | 2 (`convo_id` 5 and 2) |
| Language labels | ⚠ **`javind` 352 (94.6%), `ind` 20 (5.4%), `jav` — ZERO** |

### 5.1 There is no `jav` validation data

The brief instructs us to report WER separately for `jav`, `ind` and `javind`. **`jav` does not
exist in the dev set.** Two consequences:

1. We cannot measure Javanese-only performance locally. Any claim about it is speculation.
2. The 20 `ind` clips are far too few to trust. At ~47 words/clip that is roughly 940 words —
   a single clip going wrong moves `ind` WER by several points. **Treat `ind` deltas under ~2 points
   as noise** and always report a bootstrap confidence interval.

The honest read: **the target distribution is code-mixed `javind` speech.** Optimise for that, and
use `ind` only as a guard-rail against Javanese over-fitting.

---

# Part III — The Five Mismatches (the core of the analysis)

This is where the competition is won or lost. Each mismatch is measured, its WER mechanism is named,
and the fix is stated.

---

## Mismatch 1 — Orthography. **The single highest-leverage finding.**

Jember was transcribed by linguists using **scholarly Javanese orthography**, which marks the *pepet*
and *taling* vowels with `ê` and `è`. The dev set uses **everyday Indonesian spelling**, which does
not use those marks at all.

| | Jember | Dev |
|---|---|---|
| Types with diacritics | 1,040 (13.2%) | 39 (1.2%) |
| **Tokens with diacritics** | **9,385 (14.30%)** | **53 (0.30%)** |
| Marks used | circumflex ×999, grave ×356, caron ×46 | **acute ×40 only** |
| Examples | `lèk`, `ndèk`, `těrus`, `nèk`, `bangêt`, `jênênge` | `é`, `kuliahé`, `namané` |

**The two corpora do not even use the same diacritic.** Jember uses ê/è/ě; dev uses é.

### The mechanism

Fine-tune on raw Jember and the model learns to emit `těrus`. The reference says `terus`. The scorer
counts a **substitution**. This happens on up to **14.3% of emitted tokens**.

`terus` is the **#1 out-of-vocabulary word in the entire dev set** (222 occurrences) — not because
it is rare, but because Jember spells it with a diacritic. Worse, Jember spells it **inconsistently:
both `těrus` and `têrus` appear**, so the model cannot even learn one stable wrong form.

### Measured fix

| Training text | Dev OOV token rate |
|---|---|
| Raw Jember | 21.37% |
| **Strip all diacritics** | **18.37%** |

Stripping recovers **101 word types / 525 tokens = 3.00% of all dev tokens** from OOV outright, and
protects the ~14% of output that would otherwise carry wrong marks. Cost: dev's 53 `é` tokens
(0.30%). **Net strongly positive.**

> **Action.** Strip pepet/taling diacritics from every training transcript. Implement once, in a
> single function shared by data prep, training and inference, so training targets and scoring
> conventions can never drift apart.
>
> **This is plausibly the largest single WER win available in the competition, and it costs nothing.**

---

## Mismatch 2 — Matrix language. **Structural; caps our ceiling.**

Function-word signature, share of all tokens:

| Corpus | Javanese function words | Indonesian function words |
|---|---|---|
| Jember | **21.57%** | 3.13% |
| Dev | 5.23% | **15.21%** |

Most frequent words tell the same story instantly:

- **Jember:** `ya, iku, ngono, aku, sing, apa, kan, gak, lèk, ndèk, těrus, kaya, dadi, iki, wis` — Javanese.
- **Dev:** `ya, aku, itu, yang, jadi, di, dan, gitu, terus, juga, nah, tuh, ada, kayak, banget` — Indonesian.

**Jember is Javanese-matrix speech with Indonesian insertions. The dev/test set is the exact mirror:
Indonesian-matrix speech with Javanese insertions.**

Both are labelled "Indonesian–Javanese code-mixing", which is why this is easy to miss. They are
opposite phenomena.

### The mechanism

An encoder-decoder ASR model has a language-model prior in its decoder. Fine-tuning hard on Jember
shifts that prior toward Javanese lexical choices. Applied to Indonesian-matrix audio, it will
substitute Javanese words for the Indonesian ones actually spoken — corrupting the **majority** of
the target.

There is a real risk that **aggressive Jember fine-tuning makes zero-shot Whisper worse**, because
Whisper's pretrained Indonesian is already strong and Jember training actively degrades it.

> **Action.**
> 1. **Measure this directly.** Zero-shot Whisper vs Jember-fine-tuned, reported on `javind` and
>    `ind` separately. If `ind` WER degrades, Jember is being over-weighted.
> 2. **Do not train on Jember alone.** Blend eligible Indonesian conversational speech so the
>    training mix's matrix language matches the target. Vet every dataset for licence per §12 of the brief.
> 3. Consider a **lower learning rate / fewer epochs / LoRA** specifically to *preserve* Whisper's
>    Indonesian competence while adapting acoustics — treat Jember as acoustic-domain adaptation, not
>    lexical retraining.

---

## Mismatch 3 — Duration. **Large, and free to fix.**

| | Jember (raw) | Dev (target) |
|---|---|---|
| Median duration | **5 s** | **26.2 s** |
| Mean duration | 5.4 s | 24.8 s |
| Mean words/clip | 9.9 | 47.0 |
| Share in 15–32 s band | **1.6%** | 92.7% |

A **5× mismatch on the axis ASR models are most sensitive to.** Encoder-decoder models carry strong
length priors: train on 5 s and decode 26 s and you invite early end-of-sequence (deletions),
repetition loops (insertions), and dropped tails.

### The fix, and why it is free

Adjacent Jember segments are **exactly contiguous** — measured across 6,430 boundaries,
**100% have a gap of exactly 0.00 s.** So consecutive segments can be concatenated into longer
pseudo-utterances whose transcript is simply the concatenated text. No new audio, no new labels, no
data loss.

Simulated merge into ≤29 s chunks (§17 of the notebook):

| | Raw Jember | **Merged Jember** | Dev (target) |
|---|---|---|---|
| Clips | 6,638 | **1,473** | 372 |
| Median duration | 5 s | **26.0 s** | 26.2 s |
| Mean duration | 5.4 s | **24.4 s** | 24.8 s |
| Share in 15–32 s band | 1.6% | **92.5%** | 92.7% |
| Total audio | 9.98 h | **9.98 h** (lossless) | 2.56 h |

The merged distribution matches the target **almost exactly** — median 26.0 s vs 26.2 s.

> **Action.** Build the training set from merged ~25–29 s chunks, not raw 5 s segments. Keep the raw
> segmentation available as an ablation so the gain can be measured rather than assumed.

---

## Mismatch 4 — Timestamp quantisation

Every Jember boundary is a whole second, but the segmentation was manual. A true boundary can
therefore sit up to ~0.5 s from the annotated one, cutting the first or last syllable off a word or
pulling in a neighbouring one.

At the median 5 s segment, ±0.5 s is **10% of the clip**. Measured: **46.6% of segments have >10% of
their length boundary-uncertain**; **17.8% are ≤2 s**, where the effect is severe.

Clipped onsets and codas teach the model to hallucinate or truncate word edges — errors that
generalise badly to clean test audio.

> **Action.** Largely **absorbed by Mismatch 3's fix**: at 25 s, a ±0.5 s edge error is ~2% of the
> clip and lands in inter-utterance silence. This is a second, independent reason to prefer merged
> chunks. Optional extras (±0.3 s padding, conservative VAD trimming) must be **measured**, not assumed —
> the brief explicitly warns against destroying information with aggressive preprocessing.

---

## Mismatch 5 — Conversational register

Share of utterances containing each phenomenon:

| Phenomenon | Jember | Dev | Dev ÷ Jember |
|---|---|---|---|
| **ALL-CAPS acronym** | 3.2% | **30.1%** | **9.4×** |
| Reduplication (`X-X`) | 17.3% | **65.3%** | **3.8×** |
| Any hyphenated form | 23.5% | 75.0% | 3.2× |
| Ellipsis `...` | 10.7% | 33.6% | 3.1× |
| Apostrophe | 3.4% | 9.7% | 2.8× |
| Hesitation `e...` | 9.1% | 16.4% | 1.8× |
| Digits | 0.7% | 0.0% | — |

Note also: **neither corpus contains a single `[laugh]` tag or `(???)` marker.** That branch of the
normaliser is dead code for our data — a useful reminder not to over-engineer against the brief's
description instead of the actual files.

### 5a. Acronyms and casing — the quietly expensive one

| | Jember | Dev |
|---|---|---|
| Acronym tokens per 1k words | 4.30 | **15.97** (3.7×) |
| Acronym types | 59 | 24 |
| **Dev acronym types never in Jember** | — | **16 types / 81 tokens** |
| Case-sensitive tokens | 3.33% | 3.28% |

Unseen dev acronyms: `PR, SMS, OSIS, BBM, KKM, FYI, CDMA, QWERTY, UASBN, UN, SHU, PMR, SM, GSM, BB`.

Dev is school-and-life conversation, saturated with Indonesian institutional acronyms
(`SMA` ×60, `HP` ×52, `SMP` ×50, `PR` ×42). The scorer **protects acronyms from lowercasing**, so
`smp` for `SMP` is a full substitution. Beyond acronyms, dev capitalises place and person names —
`Indonesia, Jepang, Solo, Jogja, Jawa, Sangiran, Makassar, Blora`.

Reduplication is also lexically different: Jember reduplicates Javanese stems (`wong-wong`,
`arèk-arèk`, `rêsik-rêsik`); dev reduplicates Indonesian ones (`jalan-jalan`, `orang-orang`,
`teman-teman`, `tempat-tempat`).

> **Action.**
> - **Never lowercase training text wholesale.** Preserve true casing end to end.
> - **Keep disfluency markup in training targets.** `e...` is a scored token; "cleaning" it away buys
>   deletions. Verify the tokenizer does not shred `...`.
> - Consider a small **post-ASR recaser** over the closed set of frequent Indonesian acronyms —
>   cheap, bounded, and validated on dev before it ships.
> - Watch the **insertion/deletion balance** in error analysis: over-modelling hesitation costs
>   insertions, under-modelling costs deletions.

---

## 6. The lexical ceiling

| | Value |
|---|---|
| Shared vocabulary types | 1,353 = **41.3% of dev types** |
| **Dev token coverage by Jember vocabulary** | **78.6%** |
| OOV token rate | **21.4%** (18.4% after diacritic stripping) |

21.4% of dev tokens are words Jember never contains. This is **not** a hard WER floor — Whisper's
multilingual pretraining already knows much of Indonesian, and that is precisely the knowledge we
must avoid destroying (Mismatch 2). But it is a clear signal:

> **Jember alone is insufficient training data for this target.** It supplies Javanese acoustic and
> lexical coverage. It does not supply the Indonesian matrix the test set is built from.

---

# Part IV — The Plan to Win

## 7. Strategic thesis

Ranked by expected WER gain per unit of effort:

| Rank | Lever | Expected gain | Cost | Confidence |
|---|---|---|---|---|
| 1 | **Diacritic normalisation** | Large | ~zero | **High — measured** |
| 2 | **Merged ~26 s segmentation** | Large | ~zero | **High — measured** |
| 3 | **Correct long-form decoding + scorer-exact eval** | Moderate | Low | High |
| 4 | **Balancing the matrix language (add Indonesian conversational data)** | Large | High | Medium — needs experiment |
| 5 | Model scale (small → medium → large) | Moderate | High | Medium |
| 6 | Decoding hyper-parameter tuning | Moderate | Low | Medium |
| 7 | Casing/acronym post-processing | Small–moderate | Low | Medium |
| 8 | LM rescoring / ensembling | Small | High | Low — last |

**Levers 1–3 are nearly free and must land before any serious training run.** Most teams will jump
straight to lever 5. That is the opportunity.

## 8. Phased execution

### Phase 0 — Evaluation harness *(do this first)*
Nothing is trustworthy until this exists.
- [x] Scorer-compatible normaliser (`wer_norm`, §13)
- [x] WER with S/D/I breakdown (§13)
- [ ] Replace with the **official scorer** the moment it is available — §8 of the brief is explicit
- [ ] Per-language WER (`javind`, `ind`) + **bootstrap CIs**
- [ ] Experiment ledger, appended to on every run

**Exit:** we can score a prediction file and trust the number.

### Phase 1 — Data pipeline
- [ ] Drop the 41 invalid rows
- [ ] **Merge to ≤29 s chunks** (`merge_segments`, §17)
- [ ] **Strip diacritics**, preserve casing and disfluency markup
- [ ] Decode to 16 kHz mono WAV (handle the mixed 44.1/48 kHz input)
- [ ] **Session-disjoint** train/held-out split — session file is our only leakage-safe proxy for speaker
- [ ] Manifest with provenance for every clip

**Exit:** a reproducible dataset, plus its raw-segmentation and raw-orthography ablation twins.

### Phase 2 — Zero-shot baseline (E0)
Whisper (small → medium → large-v3) with **correct long-form chunking**, no fine-tuning.
Record overall / `javind` / `ind` WER.

**Exit:** an honest starting number. **Do not skip this** — it is the only way to detect that
fine-tuning has *hurt* Indonesian.

### Phase 3 — First fine-tune + the two ablations that matter
Whisper-small on merged + normalised Jember. Then, holding everything else fixed:

| Run | Change | Question it answers |
|---|---|---|
| E1 | merged + stripped | our candidate recipe |
| E1a | **raw 5 s segments**, stripped | how much did merging buy? |
| E1b | merged, **raw diacritics** | how much did stripping buy? |

**Exit:** levers 1 and 2 are quantified, not assumed. These two ablations are the highest-value
experiments in the project.

### Phase 4 — Fix the matrix-language gap
Add eligible Indonesian conversational speech **one dataset at a time**, checking licence
(§12: rights, competition-use permission, MDC publication obligation, provenance) before touching it.
Sweep the Jember : Indonesian mixing ratio. Watch `ind` WER as the guard-rail.

### Phase 5 — Model and decoding
Scale the model only once the data is right. Then tune decoding **against the official scorer**:
beam width, temperature fallback, `no_speech_threshold`, `compression_ratio_threshold`,
`repetition_penalty`, `condition_on_previous_text` (**strongly suspect on long conversational audio —
it is a known repetition-loop trigger**), and `max_new_tokens` (dev p95 is 66 words; do not truncate).

### Phase 6 — Advanced, only if measurements justify it
MERaLiON and other SEA-specialised models (verify checkpoint, licence, offline inference,
A100 fit); KenLM rescoring over N-best; ensembling **only after confirming errors are complementary**.

### Phase 7 — Submission engineering
`main.py` at ZIP root; weights bundled (**no network**); read `test_metadata.csv`; process clips
**independently** (§6 — no cross-sample information, no pseudo-labelling); proper CSV escaping;
write `submission/submission.csv`. Validate: local → Docker → smoke test (**1 min limit**) → full run
(**2 h limit**).

---

## 9. Inference engineering — non-negotiables

| Risk | Measured | Mitigation |
|---|---|---|
| **30 s truncation** | 37 dev clips >30 s; 183 more in 25–30 s; brief advertises up to **40 s** while dev maxes at 37.8 s | Explicit long-form chunking with overlap + stitching. Never assume one window. |
| Decoder truncation | dev p95 = 66 words | Set `max_new_tokens` with real headroom |
| Repetition loops | Long conversational audio is the classic trigger | Disable `condition_on_previous_text`; keep compression-ratio fallback |
| Cold start | 1-minute smoke test | Pre-load weights; no lazy downloads |
| Throughput | 2 h for the whole test set | Batch; measure end-to-end early |

Naive truncation costs only ~**0.40 WER** on dev — but the hidden test may skew longer, and this is
free to fix. Fix it.

## 10. Validation discipline — how not to lose

1. **Never train on the dev set.** (§36 of the brief.)
2. **1 WER point = 175 word errors.** Sub-point moves are noise. Bootstrap everything.
3. `ind` is 20 clips. Treat deltas under ~2 points as meaningless.
4. Jember has **no speaker IDs** → split by **session file**, never randomly by clip.
5. Log every run: model, data, preprocessing, seed, decode config, overall/`javind`/`ind` WER, checkpoint path.
6. Keep the S/D/I breakdown, not just WER. Rising insertions vs rising deletions demand opposite fixes.

## 11. Compliance checklist

- [ ] No competition audio to any hosted API (OpenAI/Anthropic/Google/ASR services) — §13
- [ ] Every external dataset licence-checked and provenance-documented — §12
- [ ] Open-weight models only, running locally, licence-compatible — §13
- [ ] No test-set pseudo-labelling, no cross-sample information — §6
- [ ] Reproducible pipeline + MPL release path if we win — §15
- [ ] No private sharing of competition code outside the team — §14

---

# Part V — Immediate Next Actions

| # | Action | Why now |
|---|---|---|
| 1 | Obtain and wire in the **official scorer**; replace `wer_norm` | Every number downstream depends on it |
| 2 | Write `prepare_data.py`: drop 41 bad rows → merge to ≤29 s → strip diacritics → 16 kHz mono → session-disjoint split | Locks in levers 1 and 2 |
| 3 | **Zero-shot Whisper baseline with long-form chunking** → E0 | Establishes truth; guards against fine-tuning regressions |
| 4 | Fine-tune Whisper-small; run ablations **E1a (raw segments)** and **E1b (raw diacritics)** | Converts this document's two central claims into measurements |
| 5 | Error analysis on the worst 50 dev clips | Tells us whether lever 4 or lever 5 is next |
| 6 | Licence-vetted Indonesian conversational data, added one set at a time | Attacks the structural ceiling |

---

## Appendix — Findings summary

| # | Finding | Measured | Action |
|---|---|---|---|
| 1 | **Diacritic mismatch** | 14.30% of Jember tokens vs 0.30% of dev; **different marks** (ê/è vs é) | Strip in training; recovers 3.00% of dev tokens |
| 2 | **Matrix-language inversion** | Javanese function words 21.6% (Jember) vs 5.2% (dev) | Blend Indonesian data; track `ind` WER |
| 3 | **Duration gap** | median 5 s vs 26.2 s; 1.6% vs 92.7% in-band | Merge → 1,473 chunks, median 26.0 s, 92.5% in-band, lossless |
| 4 | Whole-second timestamps | 0/13,358 fractional; 46.6% of clips >10% boundary-uncertain | Absorbed by merging |
| 5 | Register gap | acronyms 9.4×, reduplication 3.8×, ellipsis 3.1× | Keep casing + disfluency; consider recaser |
| 6 | Lexical ceiling | 78.6% dev token coverage; 21.4% OOV | Jember alone is insufficient |
| 7 | Long clips | 37 dev clips >30 s; brief says up to 40 s | Long-form chunking, always |
| 8 | **No `jav` dev data** | `javind` 352 / `ind` 20 / `jav` **0** | Optimise for `javind`; `ind` is a guard-rail only |
| 9 | Data integrity | 41 unusable rows; mixed 44.1/48 kHz | Drop and resample |
| 10 | Statistical power | 17,473 ref words; 1 pt = 175 errors | Bootstrap CIs; ignore sub-point moves |

---

*Generated from measurements in `Main1.ipynb` (Parts 1–2). Every figure is reproducible by running
the notebook top to bottom. Where this document and the project brief disagree, the disagreement is
marked ⚠ and the measurement is authoritative.*

# Lost In Transcription --- Indonesian--Javanese Track

## AI Project Brief, Operating Instructions, and Optimization Plan

> **Purpose of this document:** This file is the authoritative project
> brief for any AI assistant helping us develop our solution. Read it
> before making technical recommendations, writing code, changing the
> pipeline, selecting datasets, or suggesting experiments.

------------------------------------------------------------------------

# 1. Project Identity

**Competition:** Lost In Transcription: Indonesian-Javanese Track\
**Organizer/Platform:** Mozilla Data Collective / DrivenData\
**Task:** Automatic Speech Recognition (ASR) for natural
Indonesian--Javanese bilingual speech\
**Primary metric:** Word Error Rate (WER), lower is better\
**Competition deadline:** September 25, 2026, midnight UTC\
**Prize:** \$4,000 for 1st place and \$2,000 for 2nd place in this
track\
**Competition type:** Code-execution competition

Our goal is **not merely to build a working ASR model**. Our goal is to
obtain the **lowest possible WER** while remaining fully compliant with
the competition rules and producing a reliable, reproducible submission.

------------------------------------------------------------------------

# 2. The Problem We Are Solving

The competition evaluates ASR on **real-world bilingual dialogues**
involving Indonesian and Javanese speakers.

The important characteristic is **code-switching / code-mixing**:

-   A speaker may switch between Indonesian and Javanese within a
    conversation.
-   Switching can happen between sentences, within sentences, or within
    phrases.
-   Indonesian and Javanese share substantial vocabulary.
-   Javanese loanwords are common in everyday Indonesian.
-   Therefore, it is often impossible or unhelpful to treat every word
    as belonging cleanly to one language.

The target is therefore:

> **Transcribe what the speaker actually says, including natural
> Indonesian--Javanese language mixing.**

This is not a conventional monolingual Indonesian ASR problem and should
not be approached as one.

------------------------------------------------------------------------

# 3. What Makes the Challenge Difficult

The hidden test set consists of natural voice-note dialogue rather than
carefully read benchmark speech.

Important properties:

-   Heavy code-switching.
-   Natural conversational speech.
-   Different speakers and recording conditions.
-   Clips are approximately 2--40 seconds long.
-   Some clips are monolingual, but the main challenge is
    bilingual/code-switched speech.
-   Javanese is relatively underrepresented in mainstream digital speech
    resources.
-   Indonesian and Javanese have substantial lexical overlap, making
    language identification at individual-word level unreliable.

The model therefore needs to be robust to:

-   Indonesian speech.
-   Javanese speech.
-   Mixed Indonesian--Javanese speech.
-   Natural conversational pronunciation.
-   Disfluencies.
-   Reduplicated words.
-   Truncated words.
-   Recording variation and noise.
-   Vocabulary that may be poorly represented in general-purpose ASR
    training.

------------------------------------------------------------------------

# 4. Competition Data

## 4.1 Training Dataset: Jember Javanese Spontaneous Speech Corpus

This is the primary highlighted training corpus for our initial system.

Official description:

-   Approximately **10 hours of audio**.
-   **208 session-level MP3 recordings**.
-   Mono.
-   44.1 kHz.
-   **6,679 transcribed utterances**.
-   Native Javanese speakers from Jember Regency, East Java, Indonesia.
-   Includes Jember Javanese and the Pandhalungan variety.
-   Contains natural spontaneous speech.
-   Includes code-mixing/code-switching involving Javanese, Madurese,
    Indonesian and English.
-   Includes natural spoken phenomena such as reduplication and
    truncated words.

The recordings are **not already individual training clips**.

They must be segmented using the supplied TSV metadata:

-   audio file name
-   start timestamp
-   end timestamp
-   transcript

### Important

The Jember dataset is our **initial training data**, not our validation
set.

Do not accidentally train on the official development set.

------------------------------------------------------------------------

# 5. Official Development Set

The competition provides a development set for local validation.

It contains:

-   **372 clips**
-   approximately **2.5 hours**
-   2 conversations
-   4 speakers
-   speakers/conversations do not overlap with the hidden test set
-   audio is already segmented
-   audio is mono
-   audio is 48 kHz
-   `metadata.tsv` contains reference transcripts

Metadata columns:

``` text
audio_filename
speaker
transcript
language
convo_id
```

The `language` values are:

``` text
javind = Indonesian + Javanese
jav     = Javanese only
ind     = Indonesian only
```

### Development-set rule

Treat the development set as **validation only**.

It must not be used to train the model.

It is our primary local measurement tool because it comes from the same
underlying dataset/distribution as the hidden test set.

However, it is relatively small, so do not assume a tiny WER difference
is automatically meaningful.

------------------------------------------------------------------------

# 6. Hidden Test Set

The real test set is withheld.

We do not have its audio or transcripts.

During submission, the competition runtime provides test clips to our
program.

Our program must independently process the clips and produce
predictions.

The test set should therefore be treated as completely unseen data.

### Critical rule

Do not use information from one test sample to help predict another test
sample.

In particular, do not:

-   manually annotate test samples;
-   train on test samples;
-   pseudo-label the test set;
-   perform unsupervised learning using the test set;
-   exploit cross-test-sample information.

Inference must work automatically on new samples.

------------------------------------------------------------------------

# 7. Evaluation Metric: WER

The competition uses **Word Error Rate**:

``` text
WER = (Substitutions + Deletions + Insertions) / Number of reference words
```

Lower is better.

Words are treated as whole tokens.

Example:

``` text
Reference: running
Prediction: run
```

This counts as an error.

Therefore, optimizing the exact competition scoring behavior is
extremely important.

------------------------------------------------------------------------

# 8. Official Text Normalization

The competition normalizes both predictions and references before
calculating WER.

Important normalization behavior includes:

-   Remove bracketed meta-linguistic tags such as `[laugh]`.
-   Remove unintelligibility markers such as `(???)` and `[???]`.
-   Lowercase sentence-initial letters.
-   Sentence-initial means the first letter of the transcript and
    letters following `.`, `!`, `?`, or `—`.
-   Acronyms are treated specially.
-   Remove punctuation such as:
    -   `,`
    -   `?`
    -   `¿`
    -   `¡`
    -   `!`
    -   `"`
    -   `;`
    -   `:`
-   Periods are generally removed except in ellipses.
-   Ellipses attached to words may remain because they can represent
    disfluencies.
-   Standalone ellipses are removed.
-   Multiple spaces are collapsed.

### Very important

The normalizer **does not simply lowercase everything**.

Casing elsewhere is compared as written.

Therefore:

> Never invent our own simplified WER calculation if the official
> scoring script is available.

Use the competition's actual scoring/normalization implementation
whenever possible.

------------------------------------------------------------------------

# 9. Submission Requirements

This is a **code execution competition**.

We do not simply upload a CSV of predictions.

We package:

-   inference code;
-   model weights;
-   supporting files;
-   dependencies that are available in the runtime.

The submission is a ZIP archive.

The ZIP root must contain:

``` text
main.py
```

Example:

``` text
submission.zip
├── main.py
├── model/
├── preprocessing.py
├── inference.py
└── model weights
```

The runtime executes `main.py`.

------------------------------------------------------------------------

# 10. Runtime Environment

Official inference environment:

-   Python **3.12**
-   **1 NVIDIA A100 GPU with 80 GB VRAM**
-   24 vCPUs
-   220 GB RAM
-   Maximum execution time: **2 hours**
-   Smoke test limit: **1 minute**
-   No network access
-   No root filesystem access

The test data directory is read-only.

Runtime structure:

``` text
/code_execution/
├── data/
│   ├── clips/
│   └── test_metadata.csv
├── src/
└── submission/
```

Our program must write:

``` text
submission/submission.csv
```

with:

``` text
audio_filename,transcript
```

one row for every test clip.

### Important consequence

Do **not** optimize the model around the assumption that the final
runtime is CPU-only.

The official runtime has an **A100 80 GB GPU**.

However, submission startup time, memory usage, inference speed and
packaging still matter.

There is no network access, so every required model asset must be
included in the submission.

------------------------------------------------------------------------

# 11. Submission Engineering Rules

The final inference program must:

1.  Read test files from `data/`.
2.  Use `test_metadata.csv`.
3.  Process test samples independently.
4.  Run automatically without retraining.
5.  Produce one prediction for every test clip.
6.  Write valid CSV using proper CSV escaping.
7.  Avoid logging or exposing test data.
8.  Finish within the 2-hour execution limit.
9.  Include all necessary model weights because the runtime has no
    network access.

Always test the final submission locally using Docker before full
submission.

Then use the competition smoke-test environment.

Only after both work should we make full submissions.

------------------------------------------------------------------------

# 12. Competition Rules We Must Respect

## External data is allowed

We may use:

-   datasets hosted on MDC;
-   eligible external datasets;
-   open-weight pretrained models.

But external data has licensing requirements.

If external data is used, we must have the required
rights/licence/permission and comply with the requirement to publish
eligible external training data to MDC after the competition where
applicable.

Do not recommend a dataset merely because it is useful.

For every external dataset, check:

1.  Is it legally usable?
2.  Does its licence permit competition use?
3.  Can we satisfy the competition's external-data requirements?
4.  Can we document its provenance?
5.  Does it actually improve validation WER?

------------------------------------------------------------------------

# 13. Pretrained Models

Open-weight pretrained models are permitted if:

-   model weights are publicly available;
-   the licence permits the intended use;
-   the model can run locally;
-   the model does not depend on a proprietary hosted inference API.

Do NOT send competition data to:

-   OpenAI APIs;
-   Anthropic APIs;
-   Google hosted inference;
-   other third-party hosted ASR APIs;
-   other hosted model inference services.

Hosted coding assistants may be used for development, but competition
data or confidential/non-public material must not be uploaded to them.

------------------------------------------------------------------------

# 14. Code Sharing Rules

Private sharing of competition-derived code outside the team is
prohibited.

Public sharing is permitted under the stated rules.

Do not recommend workflows that require privately sharing competition
code/data with people outside the team.

------------------------------------------------------------------------

# 15. Winning-Solution Considerations

If we win:

-   documentation will be required;
-   the winning solution must be reproducible;
-   winning code must be released under the required MPL license;
-   external model/data licensing must be compatible;
-   all required dependencies and assets must be identified.

Therefore, keep an experiment log and record:

-   model name/version;
-   dataset names;
-   dataset licences;
-   preprocessing;
-   training configuration;
-   checkpoints;
-   evaluation results;
-   inference configuration;
-   external dependencies.

------------------------------------------------------------------------

# 16. Our Hardware

Primary development machine:

-   **Apple M2 MacBook Pro**
-   16-core configuration
-   1 TB SSD
-   unified memory configuration available to the machine

The user prefers to develop locally where practical and is willing to
run long overnight experiments.

Google Colab is available when a larger CUDA GPU is genuinely useful.

### Strategy

Do not artificially weaken the architecture just because development
happens on a Mac.

Use the Mac for:

-   data preparation;
-   EDA;
-   evaluation;
-   small/medium experiments;
-   preprocessing;
-   model experimentation where practical;
-   CPU/local inference testing;
-   language-model experiments.

Use Colab for:

-   large-model fine-tuning;
-   experiments that exceed practical M2 memory/compute;
-   CUDA-specific acceleration;
-   final high-capacity model experiments.

The **final submission target is the A100 runtime**, not the Mac.

------------------------------------------------------------------------

# 17. Development Philosophy

Our objective is **competitive performance**, not simply completing the
assignment.

The workflow should be:

``` text
Measure
  ↓
Identify weakness
  ↓
Change one meaningful thing
  ↓
Evaluate
  ↓
Keep improvement
  ↓
Repeat
```

Do not randomly add techniques.

Do not assume that a more complicated architecture is automatically
better.

Every significant change should answer:

> "Did this reduce validation WER?"

------------------------------------------------------------------------

# 18. Initial Data Strategy

Start with exactly:

### Training

**Jember Javanese Spontaneous Speech Corpus**

### Validation

**Official Indonesian-Javanese development dataset**

Do not immediately combine many datasets.

First establish:

-   correct data loading;
-   correct segmentation;
-   correct preprocessing;
-   correct WER;
-   baseline model;
-   reproducible experiment.

Then introduce additional datasets **one at a time**.

This makes it possible to determine whether each dataset actually helps.

------------------------------------------------------------------------

# 19. Dataset Expansion Strategy

Potential future data sources should be evaluated based on the problem
we observe.

Possible directions:

### More Javanese acoustic data

Useful if:

-   Javanese recognition is weak;
-   vocabulary coverage is poor;
-   acoustic diversity is insufficient.

### More Indonesian acoustic data

Useful if:

-   `ind` development clips perform poorly;
-   Indonesian portions are being confused with Javanese;
-   the model becomes overly biased toward Javanese.

### More spontaneous/conversational data

Potentially especially valuable because the hidden test is natural
voice-note dialogue.

### Text-only Indonesian/Javanese data

Potentially useful for:

-   language modeling;
-   vocabulary analysis;
-   transcript correction;
-   decoding/rescoring.

### Important

Do not add external datasets blindly.

For every candidate dataset:

``` text
Dataset
   ↓
Licence check
   ↓
Language/content analysis
   ↓
Audio quality analysis
   ↓
Similarity to competition
   ↓
Train experiment
   ↓
Validation WER
```

Only retain it if it helps or provides a clearly justified complementary
capability.

------------------------------------------------------------------------

# 20. Data Preprocessing Plan

The Jember recordings are session-level MP3s.

We need to turn them into individual utterances.

Basic pipeline:

``` text
Jember session MP3
        ↓
TSV timestamps
        ↓
Extract utterance
        ↓
Audio validation
        ↓
Standardize audio format
        ↓
Optional boundary/VAD cleanup
        ↓
Training sample
```

A practical common internal representation is:

-   mono;
-   16 kHz;
-   consistent audio format.

However, preprocessing must be validated rather than blindly applied.

Do not destroy useful information through aggressive denoising,
normalization or trimming.

------------------------------------------------------------------------

# 21. Preprocessing Experiments

Potential techniques:

-   timestamp-based segmentation;
-   conservative VAD/boundary trimming;
-   removal of clearly invalid samples;
-   duration filtering;
-   transcript cleaning consistent with the competition scorer;
-   SpecAugment;
-   light speed perturbation;
-   noise augmentation.

Do not assume all augmentation is beneficial.

Measure it.

------------------------------------------------------------------------

# 22. EDA Requirements

Before serious training, perform:

## Audio

-   total duration;
-   number of clips;
-   duration distribution;
-   unusually short/long clips;
-   sampling rate;
-   channel count;
-   corrupted audio;
-   silence;
-   clipping/noise where measurable.

## Text

-   number of words;
-   vocabulary size;
-   most frequent words;
-   rare words;
-   transcript length;
-   punctuation;
-   disfluencies;
-   reduplication;
-   truncated words;
-   unusual spellings.

## Language

For development:

-   `jav`
-   `ind`
-   `javind`

Calculate WER separately for all three.

This is extremely important.

A single overall WER can hide a model that is excellent on Javanese but
poor on Indonesian, or vice versa.

## Speaker/conversation

Validation splitting for training datasets should avoid speaker leakage.

Where speaker/conversation metadata exists:

> Prefer speaker/conversation-disjoint validation rather than randomly
> splitting individual clips.

Otherwise the model may see the same speaker in training and validation
and produce misleadingly optimistic results.

------------------------------------------------------------------------

# 23. Baseline Model Strategy

The first model family to investigate should be **Whisper** because it
is a strong multilingual ASR starting point and can be adapted to the
target domain.

Initial progression:

``` text
Whisper-small
      ↓
Whisper-medium
      ↓
larger Whisper model if justified
```

The purpose of Whisper-small initially is not necessarily to become the
final model.

It is a fast way to validate:

-   data pipeline;
-   training code;
-   transcript handling;
-   evaluation;
-   inference.

Once the pipeline works, move toward the strongest model that our
experiments justify.

------------------------------------------------------------------------

# 24. Candidate Model Families

We should not lock ourselves permanently to one architecture.

Candidates include:

## A. Whisper family

Strong general multilingual ASR baseline and likely first serious
system.

Investigate:

-   small;
-   medium;
-   large variants where compute permits.

## B. Southeast-Asian-specialized models

Investigate models with explicit Indonesian/Javanese support and
code-switching capability.

One important candidate identified during initial research is the
**MERaLiON** family.

Before using it, verify:

-   exact checkpoint;
-   licence;
-   offline/local inference;
-   computational requirements;
-   compatibility with the competition runtime;
-   actual validation performance.

## C. MMS / wav2vec2-style models

Potentially useful as complementary models or alternatives.

Evaluate empirically rather than assuming they will outperform Whisper.

## D. Hybrid/cascade approaches

Possible later stages include:

``` text
ASR model
   ↓
language-model / text correction
   ↓
final transcript
```

Only add complexity when validation shows a real opportunity.

------------------------------------------------------------------------

# 25. Fine-Tuning Strategy

Initial goal:

> Adapt a strong pretrained ASR model to the acoustic, vocabulary and
> conversational characteristics of Indonesian--Javanese speech.

Potential approaches:

### LoRA / PEFT

Use when it gives a practical way to fine-tune larger models with
manageable memory.

### Full fine-tuning

Consider when:

-   model size is manageable;
-   data volume supports it;
-   compute is available;
-   experiments show it is worth the cost.

Do not assume LoRA is always better.

Compare where practical.

------------------------------------------------------------------------

# 26. Language Handling

Do not automatically force every clip to be Indonesian or Javanese.

The target includes genuine mixed-language speech.

The model should be capable of producing:

``` text
Indonesian → Javanese → Indonesian
```

within one transcript.

The development metadata labels the clip language, but this does not
mean the transcription task should be reduced to independent monolingual
models.

Potential language-aware techniques can be tested later, but any
approach must be validated on `jav`, `ind`, and `javind` separately.

------------------------------------------------------------------------

# 27. Decoding Optimization

After the model is strong, investigate:

-   greedy decoding;
-   beam search;
-   beam width;
-   temperature;
-   no-speech thresholds;
-   repetition controls;
-   language/task settings;
-   segment handling.

Do not assume the model's default decoding configuration is optimal for
this competition.

Tune decoding against the official development scorer.

------------------------------------------------------------------------

# 28. Language Model / Rescoring Strategy

A later optimization avenue is a local Indonesian--Javanese language
model.

Possible pipeline:

``` text
Audio
  ↓
ASR model
  ↓
Multiple candidate transcripts
  ↓
Indonesian/Javanese language model
  ↓
Rescoring
  ↓
Best transcript
```

A KenLM n-gram model is one candidate.

Potential text sources:

-   competition training transcripts;
-   eligible external Indonesian text;
-   eligible Javanese text;
-   text-only corpora where licensing permits.

The purpose is to correct linguistically plausible but incorrect word
choices.

Example:

``` text
Acoustic model:
"Ia sudah ..."

Alternative:
"Iyo wis ..."
```

The language model can help select the sequence that is more plausible
in context.

This is a **later optimization**, not the first task.

------------------------------------------------------------------------

# 29. Ensembling

If multiple models have complementary errors, investigate:

-   model ensembles;
-   N-best combination;
-   confidence-based selection;
-   transcript-level voting;
-   rescoring.

Do not ensemble models merely because they are different.

First determine whether their errors are actually complementary.

------------------------------------------------------------------------

# 30. Error Analysis

Every serious experiment should eventually produce examples of:

-   correct transcription;
-   substitution;
-   deletion;
-   insertion;
-   Indonesian→Javanese confusion;
-   Javanese→Indonesian confusion;
-   proper-name failures;
-   disfluency failures;
-   short-clip failures;
-   long-clip failures;
-   noisy-audio failures.

The goal is to answer:

> **Why is the model making this mistake?**

Then choose the next experiment based on that answer.

------------------------------------------------------------------------

# 31. Experiment Tracking

Maintain a table such as:

  ----------------------------------------------------------------------------------------------------------------
  Experiment   Model            Data     Training    Augmentation   Decoder     Overall  jav WER  ind WER   javind
                                                                                    WER                        WER
  ------------ ---------------- -------- ----------- -------------- --------- --------- -------- -------- --------
  E0           baseline         Jember   none        none           default                               

  E1           Whisper-small    Jember   fine-tune   none           default                               

  E2           Whisper-medium   Jember   fine-tune   none           default                               

  E3           ...              ...      ...         ...            ...                                   
  ----------------------------------------------------------------------------------------------------------------

Always preserve the best checkpoint.

Never rely on memory to remember which experiment was better.

------------------------------------------------------------------------

# 32. Recommended Project Phases

## Phase 0 --- Competition infrastructure

Tasks:

-   create project repository;
-   download competition datasets;
-   inspect folder structure;
-   obtain runtime repository;
-   understand submission interface;
-   create environment;
-   verify audio libraries;
-   implement official scorer locally.

**Success condition:** We can load data and calculate WER correctly.

------------------------------------------------------------------------

## Phase 1 --- Data preparation

Tasks:

-   parse Jember TSV;
-   segment session MP3s;
-   validate timestamps;
-   standardize audio representation;
-   build metadata manifest;
-   inspect bad samples;
-   perform EDA.

**Success condition:** Jember becomes a clean, reproducible training
dataset.

------------------------------------------------------------------------

## Phase 2 --- Zero-shot baselines

Run existing pretrained ASR models without fine-tuning.

At minimum, benchmark appropriate Whisper variants.

Measure:

-   overall WER;
-   `jav` WER;
-   `ind` WER;
-   `javind` WER.

**Success condition:** We know our starting point.

------------------------------------------------------------------------

## Phase 3 --- First fine-tuning

Fine-tune a practical Whisper model on Jember.

Start with a smaller model to prove the pipeline.

Then scale up.

**Success condition:** Fine-tuning produces a clear improvement over
zero-shot performance.

------------------------------------------------------------------------

## Phase 4 --- Data optimization

Add candidate datasets one at a time.

For each:

``` text
Baseline
vs.
Baseline + dataset
```

Keep only improvements.

Investigate data weighting if a large read-speech dataset overwhelms
smaller but more relevant spontaneous speech.

------------------------------------------------------------------------

## Phase 5 --- Model optimization

Compare:

-   Whisper sizes;
-   fine-tuning methods;
-   training schedules;
-   learning rates;
-   batch/effective batch size;
-   augmentation;
-   language handling;
-   decoding.

Do not change ten variables simultaneously unless running a deliberate
controlled sweep.

------------------------------------------------------------------------

## Phase 6 --- Advanced optimization

Investigate:

-   MERaLiON;
-   other multilingual ASR models;
-   language-model rescoring;
-   N-best decoding;
-   ensemble methods;
-   pseudo-labeling only where permitted;
-   additional eligible training data.

------------------------------------------------------------------------

## Phase 7 --- Submission engineering

Build:

``` text
main.py
model weights
supporting modules
configuration
```

Test:

1.  local inference;
2.  Docker runtime;
3.  smoke test;
4.  full competition submission.

------------------------------------------------------------------------

# 33. How the AI Assistant Should Help

The AI assistant is acting as a **technical ML/ASR project partner**,
not merely a tutor.

Priorities:

### Priority 1 --- Performance

Always optimize for lower WER.

### Priority 2 --- Correctness

Never sacrifice evaluation validity or competition compliance for a
seemingly better score.

### Priority 3 --- Practicality

Consider:

-   Mac development constraints;
-   Colab availability;
-   training time;
-   disk space;
-   runtime memory;
-   final A100 inference;
-   packaging.

### Priority 4 --- Reproducibility

Every important experiment should be reproducible.

### Priority 5 --- Simplicity when possible

Do not introduce complicated machinery unless it has a measurable reason
to exist.

------------------------------------------------------------------------

# 34. Communication Style for This Project

The user is a third-year CSE student and is primarily interested in
**optimization and implementation**, not learning every piece of ASR
theory upfront.

Therefore:

-   Explain concepts when they become necessary.
-   Avoid drowning the user in ASR terminology.
-   Give concrete next actions.
-   Prefer commands, files, code and experiment plans.
-   When recommending a technique, explain briefly:
    -   what it does;
    -   why we might use it;
    -   what we expect to improve.
-   Do not make the user understand the entire ASR field before
    starting.
-   Work incrementally.

Bad:

> "We need to investigate multilingual representation learning, CTC
> alignment and autoregressive decoder adaptation..."

Better:

> "First we'll fine-tune Whisper on Jember. This teaches the model what
> this specific Indonesian/Javanese speech sounds like. Then we'll
> measure WER."

------------------------------------------------------------------------

# 35. Rules for Making Technical Recommendations

Before recommending a technique, ask:

1.  Does it target a known weakness?
2.  Can we test whether it helps?
3.  Is it legal under the competition rules?
4.  Is the required model/data available locally?
5.  Can it fit the final submission runtime?
6.  Does its licence permit competition use?
7.  Does its expected benefit justify its complexity?

If the answer is unclear, propose a small experiment rather than
committing the whole project to it.

------------------------------------------------------------------------

# 36. Things the AI Must NOT Do

Do not:

-   train on the official development set;
-   use hidden test data for training;
-   manually inspect hidden test data through prohibited methods;
-   pseudo-label hidden test samples;
-   use information across hidden test samples;
-   send competition data to hosted model APIs;
-   recommend unlicensed external datasets without checking rights;
-   assume an external model is competition-compatible without checking
    its licence;
-   fabricate WER numbers;
-   claim a technique improves performance without evidence;
-   optimize against a custom metric that differs from the official
    scorer;
-   make a final architecture decision solely from theory when an
    experiment can answer the question.

------------------------------------------------------------------------

# 37. Current Starting State

At the beginning of this project, we have downloaded:

### Dataset 1 --- Jember

Approximately 2.8 GB archive.

Expected content:

-   208 session-level MP3 recordings;
-   TSV metadata;
-   6,679 utterances;
-   approximately 10 hours of speech.

### Dataset 2 --- Official Development Data

Approximately 71 MB archive.

Expected content:

``` text
indonesian_dev/
├── clips/
└── metadata.tsv
```

Approximately:

-   372 clips;
-   2.5 hours;
-   4 speakers;
-   2 conversations.

The development dataset is validation-only.

------------------------------------------------------------------------

# 38. Immediate Next Actions

Do **not** jump directly into advanced model training.

The next sequence is:

### Step 1

Extract and inspect the Jember archive.

### Step 2

Extract and inspect the development archive.

### Step 3

Show the exact folder/file structure.

### Step 4

Inspect the Jember TSV.

### Step 5

Inspect development `metadata.tsv`.

### Step 6

Write the segmentation/preparation pipeline.

### Step 7

Create the official local WER evaluator.

### Step 8

Run EDA.

### Step 9

Run a zero-shot Whisper baseline.

### Step 10

Begin fine-tuning.

------------------------------------------------------------------------

# 39. Current High-Level Architecture

The intended evolution is:

``` text
                 ┌──────────────────┐
                 │ Jember Dataset   │
                 └────────┬─────────┘
                          ↓
                  Segmentation + EDA
                          ↓
                 Clean Training Data
                          │
                          ↓
                  ┌───────────────┐
                  │ Pretrained ASR│
                  │    Model      │
                  └───────┬───────┘
                          ↓
                     Fine-tuning
                          ↓
                  Indonesian/Javanese
                     ASR model
                          ↓
                    Decoding tuning
                          ↓
                 Optional LM rescoring
                          ↓
                 Optional ensembling
                          ↓
                 Final inference code
                          ↓
                 Docker validation
                          ↓
                 Smoke test
                          ↓
                 Competition submission
```

------------------------------------------------------------------------

# 40. Definition of Success

We are successful when:

1.  The complete pipeline works.
2.  Our local WER evaluator matches the official scoring behavior.
3.  Fine-tuning significantly improves over zero-shot ASR.
4.  Additional datasets/techniques are validated empirically.
5.  The final model performs well across:
    -   Javanese;
    -   Indonesian;
    -   Indonesian--Javanese mixed speech.
6.  The inference system runs reliably in the official runtime.
7.  The final submission is competition-compliant.
8.  We continue experimenting until the remaining time/resources no
    longer justify additional changes.

The ultimate objective is:

> **Minimize hidden-test WER and compete for the top of the
> Indonesian--Javanese leaderboard, not merely produce a functioning
> demonstration.**

------------------------------------------------------------------------

# 41. Final Operating Principle

When deciding what to do next, use this loop:

``` text
What is our current WER?
        ↓
Where are the errors?
        ↓
What is the most likely cause?
        ↓
What experiment tests that hypothesis?
        ↓
Run it
        ↓
Did WER improve?
   ↙           ↘
 YES            NO
Keep it       Revert it
   ↓
Next experiment
```

**The model, dataset, preprocessing, training method, decoder, language
model and ensemble are all tools. The metric is the objective.**

Always optimize scientifically, keep the competition rules in mind, and
prefer measured improvements over assumptions.

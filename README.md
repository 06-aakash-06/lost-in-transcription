# Lost in Transcription: Indonesian–Javanese ASR

This repository records the progression from a multi-model ASR submission to
conversation-disjoint adaptation experiments and a simpler protected-fusion
candidate for DrivenData's *Lost in Transcription* competition.

## Start here: the full project story

Read the [case study](LOST_IN_TRANSCRIPTION_CASE_STUDY.md) for the complete
project arc. It explains the problem and execution constraints, the data and
preprocessing pipeline, the initial model exploration, why the broad ensemble
was not reliable enough, and how the project pivoted to an adapted Whisper
anchor with bounded corrections.

The five primary model families explored were **Whisper, Qwen, XLS-R,
MERaLiON, and BuzzASR**. The earliest weighted-medoid system used Whisper,
Qwen, and XLS-R; MERaLiON and BuzzASR entered later ensemble experiments. The
case study distinguishes these exploration stages from the final submitted
system and the later Large-v3 research candidate.

Useful case-study sections:

- [Problem and execution constraints](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#2-problem-and-execution-constraints)
- [Data and domain mismatch](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#3-data-and-domain-mismatch)
- [Preprocessing pipeline](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#5-preprocessing-pipeline)
- [Development chronology](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#6-development-chronology)
- [Initial models and ensemble experiments](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#7-models-and-initial-ensemble-experiments)
- [Why the broad ensemble failed and why I pivoted](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#8-why-the-broad-ensemble-could-fail)
- [Whisper Turbo pivot and protected fusion](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#9-pivot-to-a-whisper-turbo-anchor)
- [Later Large-v3 research and final evidence](LOST_IN_TRANSCRIPTION_CASE_STUDY.md#13-large-v3-follow-on-research)

## What changed

The first competition pipeline combined Qwen, MERaLiON, Whisper, and Buzz with
ROVER. Its local conversation-disjoint dev replay was around 0.2058 WER, while
the uploaded system scored 0.4009 on the hidden leaderboard. That gap was not
explained by test-language metadata: metadata-blind replay was 0.2058 versus
about 0.2062 when metadata was used. The hidden-set cause remains unknown. See
[`DEV_ROUTING_DIAGNOSIS.md`](DEV_ROUTING_DIAGNOSIS.md) and
[`ARCHITECTURE_HANDOFF.md`](ARCHITECTURE_HANDOFF.md).

The next work isolated a single adapted Whisper model, then measured whether
MERaLiON could correct its errors. Simple ROVER did not make useful use of
MERaLiON, so the final candidate uses Whisper as the anchor and only applies
bounded, confidence-aware corrections. An existing full-epoch Whisper
checkpoint supports corrections in the three-model candidate.

| System | Fold A WER | Fold B WER | Aggregate WER | Evidence |
|---|---:|---:|---:|---|
| Earlier ensemble local replay | — | — | 0.2058 | Conversation-disjoint dev replay; hidden submission scored 0.4009 |
| Phase 1 Whisper LoRA | 0.2236 | 0.2042 | 0.2197 | Strict leave-one-conversation-out; Submission #1 hidden WER 0.2862 |
| Phase 2 Whisper + MERaLiON, simple fusion | 0.2185 | 0.2151 | 0.2178 | Out-of-fold simple fusion |
| Earlier Phase 3 protected 2-model fusion | 0.2157 | 0.2039 | 0.2133 | Earlier out-of-fold candidate |
| Earlier Phase 3 fusion + full-epoch Whisper support | 0.2124 | 0.1971 | 0.2093 | Earlier out-of-fold candidate |
| Final conservative Turbo/MERaLiON fusion | 0.2098 | 0.1853 | 0.2048 | Later strict-substitution selection |
| Large-v3 research candidate | **0.1994** | **0.1727** | **0.1939** | Later local OOF; no supplied leaderboard result |

The conversation folds are small and share only two source conversations. These
numbers guide model selection; they do not predict hidden WER exactly. Phase
reports include configurations, selection rules, and deployment limitations.

## Work log

[`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) follows the dated repository commits
and phase artifacts from the initial September work through Phases 1–3.

## Phase implementations

- [`phase1_whisper_anchor/`](phase1_whisper_anchor/README.md): audio/text
  preparation, leave-one-conversation-out manifests, Whisper Turbo LoRA,
  official scoring, MPS inference, and the original single-model submission.
- [`phase2_meralion/`](phase2_meralion/README.md): zero-shot complementarity
  gate, decoder-only LoRA, OOF comparison, and MERaLiON license notices.
- [`phase3_protected_fusion/`](phase3_protected_fusion/README.md): protected
  alignment, confidence extraction, fusion search, inference-only candidates,
  and reproducible ZIP builders.

## Reproducibility and assets

The submission builders create deterministic offline ZIPs with a root `main.py`.
The candidate hashes and local validation results are recorded in
`phase3_protected_fusion/PHASE3_REPORT.md`. Buildable sources and reports are
tracked; model checkpoints, training runs, clip-level transcripts, competition
data, and multi-gigabyte ZIPs are excluded. They exceed ordinary GitHub Git
file/repository limits or are not ours to redistribute. See
[`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) for the artifact inventory and
licensing notes.

No hidden-test transcripts, labels, routing statistics, or pseudo-labels are
included. The supplied CUDA log documents a successful run of the conservative
Turbo/MERaLiON submission in about 31 minutes. The later Large-v3 research
candidate was validated locally on MPS; a new A100 run and leaderboard result
for that candidate are not available.

# Lost in Transcription: Indonesian–Javanese ASR

This repository records the progression from a multi-model ASR submission to
conversation-disjoint adaptation experiments and a simpler protected-fusion
candidate for DrivenData's *Lost in Transcription* competition.

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
| Phase 3 protected 2-model fusion | 0.2157 | 0.2039 | 0.2133 | Out-of-fold |
| Phase 3 protected fusion + full-epoch Whisper support | **0.2124** | **0.1971** | **0.2093** | Best measured deployable OOF candidate |

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
included. The official CUDA/Python 3.12 container run remains unverified on the
local MPS-only machine.

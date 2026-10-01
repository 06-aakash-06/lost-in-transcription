# Bundled model and runtime provenance

This archive is an offline inference submission. It does not send competition
audio to a hosted model or make network requests at inference time.

| Bundle component | Source | Declared license | Intended use here |
| --- | --- | --- | --- |
| `models/whisper_large_v3` | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT | Indonesian-forced Whisper large-v3 candidate |
| `models/whisper_large_v3_turbo` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT | Batched Indonesian/Javanese prompt specialists |
| `models/buzzasr_javanese` | [BuzzASR/javanese](https://huggingface.co/BuzzASR/javanese) | MIT | Javanese specialist for `jav` and a calibrated vote on `javind`; prompt is baked into the checkpoint |
| `models/qwen3_asr_1_7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache-2.0 | Independent local ASR candidate; Indonesian is a supported language |
| `models/meralion3_asr` | [MERaLiON/MERaLiON-3-3B-ASR](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) | MERaLiON-3 Public Licence | Southeast-Asian ASR candidate; used on `ind` and `javind` with its bundled offline chat prompt |

The model repositories and their model cards are included where provided by the
authors. The mixed-language model card documents its external speech/text
training sources and the supplied language model. If the competition organizer
requires post-competition publication of external training data or additional
attribution, that obligation remains with the submission owner.

The runtime uses the competition-provided local packages (`faster-whisper`,
`ctranslate2`, `qwen-asr`, `torch`, and `transformers`).
Only the local bundled checkpoints are referenced by `ensemble_config.json`.

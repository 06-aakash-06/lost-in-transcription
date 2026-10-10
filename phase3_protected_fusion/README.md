# Phase 3 protected final ASR

Current last-submission candidate: `artifacts/FINAL_SUBMISSION_CONSERVATIVE.zip`, with R3-full as anchor and exact MERaLiON/R3-half support for strict substitutions only. OOF WER: 0.209778 / 0.185321 / 0.204804. The existing catastrophic repetition safeguard is retained. See [FINAL_CONSERVATIVE_REPORT.md](FINAL_CONSERVATIVE_REPORT.md) for all three variants, correction counts, and the Jember sanity check. Earlier archives are historical and unchanged.

No new training. Phase 1/2 checkpoints and out-of-fold predictions are retained. Candidate A is a two-model protected fusion; Candidate B adds the already-trained full-epoch Indonesian ratio-3 Whisper as a support voter. Candidate B is recommended for Submission #2.

## Build

```bash
./venv/bin/python phase3_protected_fusion/submission/build_submission.py
./venv/bin/python phase3_protected_fusion/submission/build_submission.py --config phase3_protected_fusion/configs/candidate_B.json --output artifacts/phase3_three_model_submission.zip
```

Both archives contain root `main.py`, merged standalone checkpoints, local custom MERaLiON processor/model code, tokenizers and notices. PEFT, training, datasets and network access are unnecessary at submission time. The original Phase 1 ZIP is untouched.

## Inference

CUDA is preferred, then MPS, then CPU. CUDA uses batch 8 and BF16; local MPS uses microbatch 1, FP32 Whisper and FP16 MERaLiON. Models are loaded sequentially to bound peak memory. Decode float32 mono 16 kHz; Indonesian greedy Whisper; native Whisper long form for audio longer than 30 seconds; MERaLiON's native processor includes all audio chunks. No metadata language routing or test adaptation.

The anchor's predicted text is teacher-forced against that clip's audio to obtain acoustic subtoken log probabilities, averaged into whitespace-word log probabilities (geometric probability). No references are used. Confidence for clips longer than 30 seconds or over the decoder token limit is unavailable; those regions cannot trigger confidence corrections. Third-voter support remains possible on long clips.

Candidate A changes a disagreement of at most two words only with one agreeing word on both sides and anchor geometric confidence below .65. Insertions/deletions are disabled. Candidate B additionally permits MERaLiON corrections of at most three words when the full-epoch Whisper predicts exactly the same replacement over the same aligned anchor span. Existing anchor surface form/casing is retained outside approved replacements. MERaLiON loops are vetoed. Only runs of at least five identical anchor words collapse to two; ordinary repetition remains.

## Validate

```bash
./venv/bin/python -m unittest discover -s phase3_protected_fusion/tests -v
./venv/bin/python phase3_protected_fusion/finalize_metrics.py
./venv/bin/python phase3_protected_fusion/submission/test_zip_offline.py --zip artifacts/phase3_protected_fusion_submission.zip --label candidate_A
./venv/bin/python phase3_protected_fusion/submission/test_zip_offline.py --zip artifacts/phase3_three_model_submission.zip --label candidate_B
```

ZIP tests use the exact extracted archive, an empty Hugging Face cache and blocked Python socket connections; test CSV quoting and reversed row order; include a natural 37.6535-second dev clip. Deployment logs contain no audio filenames, hypotheses or per-clip acoustic/language diagnostics. Test metadata can be supplied with `LIT_DATA_DIR` and output with `LIT_SUBMISSION_PATH`; competition defaults are `/code_execution/data` and `/code_execution/submission/submission.csv`.

## Environment

The inspected official runtime lock at commit `3087188064b854edb72115fa8602d5b392bd7d44` has Transformers 4.57.6, Torch 2.11.0+cu130, tokenizers 0.22.2, librosa 1.0.0 and soundfile 0.14.0. Transformers matches this local environment. Local smoke uses Python 3.14.7 / MPS; the official CUDA/Python 3.12 container could not be executed because this Mac has no NVIDIA GPU or running Docker daemon. [Official runtime lock](https://github.com/drivendataorg/lost-in-transcription-runtime/blob/3087188064b854edb72115fa8602d5b392bd7d44/runtime/uv.lock).

Saved MERaLiON tokenizer conventions are preserved. Do not apply the advertised Mistral regex fix: Phase 2 verified tokenizer equivalence with all 1681 training references.

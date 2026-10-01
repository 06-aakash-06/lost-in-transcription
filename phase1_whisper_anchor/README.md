# Phase 1 Whisper anchor

Independent single-model Whisper large-v3-turbo + r32 LoRA pipeline. Existing ensemble files remain untouched.

## Data and evaluation

Run from the repository root:

```bash
./venv/bin/python phase1_whisper_anchor/training/prepare_splits.py
./venv/bin/python -m unittest discover -s phase1_whisper_anchor/tests -v
```

Fold A trains on conversation 2 plus Jember train and validates on conversation 5. Fold B reverses the conversations. `jember_holdout.tsv` retains all 28 held-out sessions. Dev references remain verbatim; the conservative Jember target conversion only maps curly quotes, Unicode ellipsis/dashes, and whitespace. Existing prepared Jember spellings are otherwise preserved. The official scorer normalizer is copied from [DrivenData runtime commit 3087188](https://github.com/drivendataorg/lost-in-transcription-runtime/blob/3087188064b854edb72115fa8602d5b392bd7d44/score.py) into `training/official_score.py` and is used only for metrics.

## Local MPS training and CUDA submission inference

On the development Mac, use the existing `./venv/bin/python` environment with Apple MPS. The trainer selects CUDA, then MPS, then CPU; MPS uses fp32, microbatch 1, gradient accumulation to effective batch 16, non-reentrant checkpointing, and PyTorch's MPS fallback for unsupported operations. CUDA is selected automatically only on the competition server. The merged model is packaged without PEFT or training code and inference runs offline.

```bash
./venv/bin/python phase1_whisper_anchor/training/run_full_phase1.py
./venv/bin/python phase1_whisper_anchor/training/benchmark.py --model phase1_whisper_anchor/models/final_merged --batch-size 8
```

The orchestrator evaluates both checkpoints on both folds and Jember holdout for each ratio-3 language, selects the best ratio-3 language, evaluates ratios 2 and 4 for that language, selects the best configuration, trains the final model, merges it, builds the ZIP, and tests that ZIP locally on MPS. Results are saved to `results/language_token.csv`. Selection minimizes worst-fold WER, then aggregate WER and Jember holdout. Differences below roughly 0.005–0.01 WER should be treated cautiously. The final run uses all 372 dev clips and Jember train, not the held-out Jember sessions.

LoRA targets are validated against encoder and decoder modules before training. Effective batch 16 uses a one-sample MPS microbatch with gradient accumulation. Checkpoints are saved at half and full epochs. The model is not initialized from any old ensemble candidate except the public Whisper Turbo base checkpoint.

Twenty-nine labeled dev clips are longer than the 30-second training context. Each is split at the quietest 100 ms point within four seconds of its midpoint; reference words are divided by relative time, and one half is sampled each time the clip appears. This approximate alignment is a training limitation. Evaluation and submission always transcribe the entire clip.

## Submission

`finalize.py` merges the chosen adapter into `models/final_merged`, compares logits and generated token IDs on a dev audio sample, and invokes the deterministic ZIP builder. To test the exact ZIP on the local MPS Mac:

```bash
./venv/bin/python phase1_whisper_anchor/submission/test_zip_local.py --zip artifacts/phase1_whisper_anchor_submission.zip --count 2
```

To rebuild later:

```bash
python phase1_whisper_anchor/submission/build_submission.py --model phase1_whisper_anchor/models/final_merged
```

The output is `artifacts/phase1_whisper_anchor_submission.zip` with root `main.py`; its SHA256 is written beside it. The runner reads only `/code_execution/data/test_metadata.csv` and `/code_execution/data/clips/`, batches independent clips, writes `/code_execution/submission/submission.csv`, and uses no network. Native Transformers long-form is the default for clips beyond 27 seconds. Explicit 27-second windows with 2-second overlap remain available with `--long-mode overlap`. On the 29 natural dev clips longer than 30 seconds, the unadapted base model scored 0.216 with native long-form versus 0.257 with overlapping windows; compare again after adaptation.

For the [official runtime](https://github.com/drivendataorg/lost-in-transcription-runtime), run `python phase1_whisper_anchor/submission/test_official_runtime.py --runtime /path/to/lost-in-transcription-runtime`. It verifies the builder's SHA256, copies the exact ZIP and all dev audio into that runtime checkout, runs Docker with `--gpus all --network none`, checks CSV order/count, invokes the official `score.py`, and verifies ZIP identity again. The current Mac has no running Docker daemon or NVIDIA GPU, so that gate remains open.

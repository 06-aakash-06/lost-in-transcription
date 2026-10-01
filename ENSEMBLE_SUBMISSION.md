# Submission ensemble

The current submission is an offline, transcript-level ensemble for the
Indonesian-Javanese track. Each test clip is decoded independently by three
local models and the selected transcript is the weighted medoid under the
competition scorer's exact token normalization.

## Bundled candidates

1. `faster-whisper-large-v3`, forced to Indonesian. This is the general ASR
   anchor and is stored in CTranslate2 format.
2. `Qwen3-ASR-1.7B`, forced to Indonesian. This is the strongest independent
   candidate in the local held-out experiment.
3. `multilingual-xls_r_300m-LARGE-5gram`, covering Indonesian, Javanese, and
   Sundanese. The final runtime uses its bundled KenLM 5-gram decoder.

The models are loaded once and reused. The inference loop never reads one test
clip while predicting another, performs no test pseudo-labeling, and makes no
network request.

## Held-out evidence

Measured on the 162-clip session-disjoint Jember held-out split:

| Candidate/selection | WER |
| --- | ---: |
| Whisper large-v3, forced `id` | 0.7237 |
| Qwen3-ASR-1.7B, forced Indonesian | 0.6611 |
| Mixed XLS-R, local greedy diagnostic | 0.7708 |
| Calibrated weighted medoid | **0.6597** |

The XLS-R diagnostic score is greedy because the local Python 3.14/macOS
environment cannot build the competition runtime's KenLM binding. The official
runtime includes `pyctcdecode==0.5.0` and `kenlm==0.3.0`; therefore the bundled
configuration enables the model's supplied 5-gram decoder during submission.
The calibration priors were fitted only on the held-out table and are stored in
`submission_src/calibration.json`.

## Build and validate

The model directories are intentionally ignored by Git but are included by the
packer. After the weights are present, run:

```bash
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
./venv/bin/python scripts/pack_submission.py --output submission.zip
```

The packer verifies that `main.py` is at the ZIP root. The archive also includes
`MODEL_LICENSES.md` with the model sources and declared licenses.

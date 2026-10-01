# Lost in Transcription ASR: Architecture Handoff

**Purpose:** give another LLM enough project-specific detail to propose and test changes that lower competition WER. Inspect the repository before editing; several source files and archives already contain uncommitted changes.

## Objective and constraints

- Task: Indonesian–Javanese conversational/code-switched ASR.
- Latest reported leaderboard result: **WER 0.4009, rank 85**.
- Historical local dev ensemble: about **0.2053**. The latest metadata-blind replay is about **0.2058**.
- Target: move hidden leaderboard WER toward **0.20 or below**.
- Submission: offline archive, about 19 GB; 2,118 test clips; Python 3.12; one A100 80 GB, 24 vCPUs, 220 GB RAM; hard runtime limit two hours.
- No network at inference. Do not train on hidden test data or tune repeatedly on the same two dev conversations and report that score as independent evidence.

## Current inference path

```text
test manifest and independent audio clips
  → 16 kHz mono decode
  → model-specific profile
  → Qwen + MERaLiON + Whisper Turbo-JW
  → trigger Buzz-JV when core disagreement ≥ 0.20
  → official-normalized word alignment and ROVER vote
  → submission CSV
```

The current working-tree config sets `route_language: javind`. This pins the Indonesian–Javanese track route regardless of per-row language values, based on the competition clarification that test `language` may describe the track rather than each clip. `submission_metadata_blind_candidate.zip` contains that routing guard. The prior `submission.zip` was left untouched; check which archive was actually submitted before comparing behavior.

The current source loads every configured backend at engine startup. Therefore `whisper_large_id` is loaded even when the pinned mixed route never uses it. In the mixed route, three core models run on every clip and Buzz-JV runs adaptively. Whisper large-v3 is configured only for the `ind` route, which is not selected under `route_language: javind`.

## Models and decode settings

| Candidate | Bundled checkpoint / backend | Prompt and decode | Role and observed dev contribution |
| --- | --- | --- | --- |
| Qwen | Qwen3-ASR-1.7B, `qwen_asr` | Forced Indonesian (`id`); 256 max new tokens; normalized input; batch 16 | Independent generalist. Individual mixed-route dev WER ≈0.251. Its Indonesian bias on mixed speech is a concern to test, not a proven cause. |
| MERaLiON | MERaLiON-3-3B-ASR, custom backend | No forced language; bundled chat prompt; no-repeat 6-gram; repetition penalty 1.05; 256 max new tokens; batch 8 | Best single generalist in the latest test-like dev replay: WER ≈0.231. |
| Whisper Turbo | OpenAI Whisper large-v3 Turbo, Transformers backend | Forced Javanese (`jw`); beam 5; 256 max new tokens; fixed 30-second padding; short-form path | Mixed-language complementary candidate; individual dev WER ≈0.253. |
| Buzz-JV | BuzzASR/Javanese, Transformers Whisper backend | No per-clip language override; beam 1; no-repeat 3-gram; repetition penalty 1.2; 256 max new tokens | Adaptive Javanese specialist. Weak alone (WER ≈0.426), but removing it raised ROVER WER from ≈0.2058 to ≈0.2102. Triggered on 198/372 dev clips. |
| Whisper large-v3 | Systran faster-whisper large-v3 | Forced Indonesian; beam 5; temperature 0; VAD off; no previous-text conditioning | Configured for `ind` rows only. Dev WER ≈0.182 on 20 `ind` examples; not run by the pinned `javind` route. |
| Buzz-ID | Not in submission config | Cached only on the 20 `ind` dev clips | Dev WER ≈0.353; did not improve their oracle. Do not add based on current evidence. |

The archive also contains `whisper_large_v3`, despite it being inactive on the pinned mixed route. Bundled checkpoint sizes in the working tree were approximately: Qwen 4.4 GB, MERaLiON 6.2 GB, Whisper Turbo 1.5 GB, Buzz-JV 2.9 GB, Whisper large-v3 2.9 GB. All are loaded once and kept resident; verify GPU memory/startup behavior on the actual A100 image.

## Audio and text preprocessing

Training-data preparation is distinct from test-time audio preprocessing.

| Stage | Current behavior |
| --- | --- |
| Shared audio decode | `librosa.load`, mono, resample to 16 kHz, float32. No denoising or compression. |
| `model_raw` profile | No edge trimming and no loudness normalization. Used by MERaLiON, Whisper Turbo, and Buzz-JV. |
| `qwen_norm` profile | No edge trimming. Apply one RMS gain toward **−23 dBFS**, capped at **−1 dBFS peak**. Used by Qwen. |
| Transcript cleanup before output | Collapse whitespace and remove control characters; preserve wording and casing. |
| Fusion alignment | `official_normalize_text` mirrors the competition normalizer for word alignment. It handles punctuation/parenthetical markup and lowers sentence initials, but does not lowercase all text. The selected transcript is emitted for the official scorer to normalize. |

Training preparation uses the Jember Javanese corpus: validate decoded audio duration and annotations, normalize Unicode/spelling conservatively, convert audio to mono 16 kHz, merge adjacent segments into chunks no longer than about 29 seconds, use limited boundary context and edge-only trimming, and split by session. It produced about 1,470 prepared chunks overall. The current train-only manifest has **1,309 examples** (1,310 TSV lines including the header), and the session-disjoint Jember holdout has 162 examples. The spelling normalization reduced measured dev OOV exposure from about 21.4% to 18.5%. **The bundled ASR checkpoints have not been fine-tuned on this corpus.**

## Routing, adaptive decode, and fusion

- Mixed route core specs: Qwen + MERaLiON + Whisper Turbo-JW.
- Adaptive trigger: mean pairwise normalized token edit distance **≥0.20** or duration **≥999 s**. The duration threshold effectively disables duration triggering for normal clips; disagreement drives Buzz calls.
- At most one adaptive candidate is added. In the current mixed route that is Buzz-JV.
- Calibration chooses word-level **ROVER** for `javind`; blank factor **1.25**.
- ROVER candidate order: MERaLiON, Buzz-JV if present, Whisper Turbo-JW, Qwen. `whisper_large_id` is listed in the order but excluded unless present and allowed by candidate selection.
- Configured prior dictionary sets all candidates to 1.0. `language_multipliers.javind.buzz_javanese = 0.5`; thus Buzz's effective mixed-route weight is 0.5. `use_quality` is false.
- ROVER aligns tokens and votes per word, including blank votes. If ROVER cannot run, the selector falls back to weighted medoid.
- The `ind` calibration branch (Qwen + MERaLiON + Whisper large-v3) remains in the file, but the pinned track route makes it inactive.

## Data and measured evidence

Dev has **372 clips from two conversations/four speakers**: 352 `javind`, 20 `ind`, and **zero `jav`**. Conversation sizes are 294 and 78 clips. The latest replay used the same mixed route for all rows:

| Measurement | WER |
| --- | ---: |
| Metadata-aware replay | 0.2062 |
| All clips through mixed route | **0.2058** |
| Qwen / MERaLiON / Turbo-JW individual | 0.2512 / 0.2312 / 0.2527 |
| Buzz-JV individual | 0.4261 |
| Current adaptive ROVER | **0.2058** |
| ROVER with Buzz removed | 0.2102 |
| Per-clip oracle among available candidates | 0.1964 |

The oracle is only about 0.0094 absolute WER below ROVER, so transcript selection alone appears to have limited headroom. It is an oracle diagnostic, not an attainable inference method. ROVER was worse than the best available candidate on 153 clips and better on 98. Buzz was individually best on only 6 clips, yet its removal worsened both conversation subsets.

A small LOCO fusion grid (Buzz weight and ROVER blank factor) selected the same setting in both directions. Held-out WER was 0.1780 on conversation 2 and 0.2129 on conversation 5; weighted aggregate 0.2058. This validates only that small grid. Earlier preprocessing, model, and fusion work has reused this tiny dev set, so broad claims of generalization are not warranted.

**Qwen cache caveat:** normalized Qwen predictions cover 282/372 clips. The other 90 use the earlier raw-audio Qwen cache because local Metal/CPU inference stalled. The all-raw replay scored 0.205719 versus 0.205776 with the mixed cache, a small observed sensitivity; the normalized-Qwen result is still not complete. Details are in `DEV_ROUTING_DIAGNOSIS.md`.

The metadata-blind test did not explain the leaderboard gap. The actual hidden test manifest/audio, its distribution, and the evaluator runtime log were not available for diagnosis. The dev set cannot establish pure-Javanese performance or which language's words dominate mixed-clip errors.

## Constraints and next experiments for the reviewing LLM

Use these as priorities, not assumptions:

1. **Check what archive and manifest were actually evaluated.** Confirm test `language` values, verify that the pinned mixed route is in the archive, and compare the evaluator log to the local entry point. Metadata-aware dev routing changed WER negligibly, so do not assume it explains 0.4009.
2. **Improve candidate hypotheses.** Oracle WER is near current ROVER. Test Qwen forced-ID versus automatic/unforced decoding and Whisper language prompting on conversation-disjoint data before changing the ensemble. Do not add more models unless oracle complementarity and runtime justify their archive size.
3. **Consider domain adaptation.** A small Whisper Turbo LoRA is a candidate, but Jember is Javanese-heavy and target dev is Indonesian-led. Keep the current holdout and dev conversations out of training; evaluate by conversation and seek permitted, licensed conversational code-switch data. Do not claim an improvement from training-only loss.
4. **Only test LID/segmentation with representative labeled validation.** Soft language evidence is a possible later fusion signal. There is no windowed LID experiment yet, and hard waveform splitting has not been tested; boundary deletion/duplication risks remain.
5. **Profile on the actual A100 runtime.** No valid post-change A100 estimate exists. To meet two hours, 2,118 clips must average below **3.40 seconds per clip including startup**. Measure load time, each model's throughput, Buzz trigger rate, peak VRAM, and full-run completion before recommending a submission.
6. **Keep the official WER path exact.** Do not add global spelling rewrites from this dev set: the top `nggak`/`gak` rule improved one conversation but hurt the other and worsened aggregate WER slightly.

Do not use dev language labels to choose predictions. Report metadata-blind overall and language subsets, conversation-disjoint results, per-model WER, oracle WER, runtime, and archive size for every candidate. Do not claim the 0.20 target is achieved without a leaderboard measurement.

## Key files

- `submission_src/ensemble_config.json` — active model profiles, prompts, batching, route, and Buzz trigger.
- `submission_src/calibration.json` — ROVER candidate order, effective weights, and route-specific settings.
- `submission_src/ensemble_runtime.py` — model backends, routing, batching, normalizer, ROVER, and output writer.
- `submission_src/audio_preprocess.py` — inference resampling and Qwen loudness profile.
- `processed/v2/train_jember_trainonly.tsv`, `processed/v2/heldout_jember.tsv`, `processed/v2/dev.tsv` — data manifests.
- `DEV_ROUTING_DIAGNOSIS.md` — metadata-blind replay and ablation results.
- `submission_src/MODEL_LICENSES.md` — model sources and declared licenses.

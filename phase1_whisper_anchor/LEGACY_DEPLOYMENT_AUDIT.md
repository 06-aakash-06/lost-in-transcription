# Legacy deployment audit

This is an engineering risk audit, not a diagnosis of the hidden WER.

- `submission.zip` (18 GB) is stale relative to `submission_src/ensemble_config.json` and `ensemble_runtime.py`. It lacks the current `route_language: javind` pin. The metadata-blind candidate has the updated files, but is a different ZIP. The local replay near 0.2058 did not establish which ZIP produced the 0.4009 leaderboard score.
- In the older ZIP, `route_language` is unset. `_route_language()` then reads the manifest's `language` field and can route test rows differently from dev rows. This is a deployment risk; the metadata-blind replay changed dev WER only about 0.0004, so it does not explain the leaderboard gap by itself.
- The bundled Whisper Turbo config has `long_form: false`. Its ordinary batch path sets `truncation=True`; clips longer than the 30-second feature window can lose their tail. Twenty-nine dev clips exceed 30 seconds, and hidden clips can reach about 40 seconds. The new Phase 1 runner always windows long audio or explicitly invokes native long-form mode.
- The old bundle includes the expected model asset directories and root `main.py`; the old main writes the expected output path and columns. Model load errors are raised, not silently converted to empty strings. `write_submission()` can serialize an empty consensus as an empty transcript, so all-empty candidate output remains a possible failure mode.
- The old main reads `submission_format.csv` before `test_metadata.csv`. That can differ from the requested input manifest if both exist with different order or contents. The Phase 1 runner reads `test_metadata.csv` exclusively and asserts row order and count.
- Four similar large ZIPs exist locally. Archive identity and source identity must be recorded before upload. The Phase 1 builder writes a SHA256 sidecar and requires the exact built ZIP for container testing.

No claim is made that any one risk caused the hidden score. The hidden evaluation logs and exact uploaded archive identity are unavailable here.

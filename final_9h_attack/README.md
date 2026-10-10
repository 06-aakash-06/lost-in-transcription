# Final nine-hour ASR attack

This isolated directory preserves all earlier checkpoints, archives, caches and logs. Start: 2026-10-02 14:06:19 UTC. Experiments stop by 21:21:19 UTC; packaging/validation ends by 23:06:19 UTC. These are relative budgets supplied by the user, not a claim about the public competition deadline.

`phase1.py` tests exact half/full LoRA-delta interpolation at full contributions 0.25/0.50/0.75, with pure half/full caches reused. `decode_variants.py` bounds auto-language greedy and Indonesian beam3 to one hour. `compare_fusion.py` compares a fixed set of strict support and unambiguous word-majority rules. `confidence.py` and `edit_gate.py` provide only acoustic/hypothesis-derived inference features, with opposite-conversation gate tests. `evaluate_merged.py` verifies the merged inference precision rather than assuming the adapter cache represents the packaged model. `evaluate_model.py` evaluates the exact standalone runtime engine. `jobs.py` preserves partial caches and retries failed jobs twice.

`main.py` and `runtime/` are standalone, offline, CUDA-first inference. No test identifiers/transcripts are logged, no cross-test adaptation occurs, and every long clip uses native long-form decoding. `build.py` includes only configured model families. `validate_zip.py` verifies all ZIP bytes/hashes/CRC and runs the exact entrypoint offline with sockets blocked, empty HF caches, quoted filenames, reverse row order, a normal clip, and a genuine >30-second clip. Large model hardlinks in the local extracted test directory save disk space; every model byte is present inside the submission ZIP and independently verified against its archive entry.

Final recommendation and measured results will be in `../FINAL_9H_REPORT.md`.

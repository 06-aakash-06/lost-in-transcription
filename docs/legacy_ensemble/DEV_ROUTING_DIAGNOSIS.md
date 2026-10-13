# Metadata-blind dev diagnosis

Run: `./venv/bin/python scripts/diagnose_dev_routing.py`

The 372 dev clips contain 352 `javind`, 20 `ind`, and no `jav`. They come from two conversations (294 and 78 clips). Every prediction in the blind replay uses the mixed-track route; true language is used only to split results afterward.

**Cache limit:** all 372 clips have Qwen, MERaLiON, Whisper-Turbo-JW, and Buzz-JV predictions. The normalized-Qwen cache covers 282 clips; 90 use the earlier raw-Qwen cache because local Metal and CPU inference stalled on a remaining clip. Replaying all 372 with raw Qwen changes blind WER from 0.205776 to 0.205719. These are close approximations, not an exact full normalized-Qwen run.

| Evaluation | Overall WER | `javind` | `ind` |
| --- | ---: | ---: | ---: |
| Metadata-aware cached replay | 0.206177 | 0.210900 | 0.114983 |
| All clips through mixed route | **0.205776** | 0.210900 | 0.106852 |
| Oracle best available clip hypothesis | 0.196397 | 0.201516 | 0.097561 |

| Test-like candidate | Dev WER | Coverage |
| --- | ---: | ---: |
| Qwen | 0.251244 | 372 |
| MERaLiON | 0.231170 | 372 |
| Whisper Turbo JW | 0.252731 | 372 |
| Buzz JV | 0.426137 | 372 |
| Whisper large-v3 ID | 0.182346 | 20 `ind` only; not in mixed route |
| Buzz ID | 0.353078 | 20 `ind` only; not in mixed route |
| Current adaptive ROVER | **0.205776** | 372 |
| ROVER without Buzz | 0.210237 | 372 |
| Weighted medoid | 0.222248 | 372 |

Buzz was triggered on 198/372 clips. It had the lowest individual edit count on only 6, and ROVER was worse on 4 of those. ROVER was worse than the best available candidate on 153 clips, better on 98. Its net gain over MERaLiON is 0.025394 WER, while the remaining gap to the clip oracle is 0.009379. Buzz ID did not lower the 20-clip Indonesian oracle.

Pairwise model disagreements (normalized token edit distance): Qwen/MERaLiON 0.224, Qwen/Whisper 0.250, MERaLiON/Whisper 0.243, and Buzz versus each generalist 0.424–0.457.

Leave-one-conversation-out fusion grid (Buzz weight 0.25–2.0, blank factor 1.0–1.5) chose the same setting, `(0.25, 1.25)`, in both directions. Held-out WER was 0.178009 on conversation 2 and 0.212865 on conversation 5; weighted aggregate 0.205776. This tests only that small new grid, not the earlier model and preprocessing choices repeatedly tuned on these conversations.

The most common ROVER substitutions include `gak→nggak` (77), `'kan→kan` (32), `enggak→nggak` (25), and `dadi→jadi` (23). Rewriting `nggak` to `gak` reduced WER on conversation 2 but increased it on conversation 5 and slightly increased overall WER (0.205834), so no text rule was added.

## Decision

Per-clip metadata does **not** explain the 0.4009 leaderboard WER: hiding it changed replay WER by -0.0004. The hidden-test distribution, annotation style, and actual test manifest are unavailable here, so the cause of the gap cannot be established from dev alone. Mixed clips are harder than `ind` clips, but the dev set has no pure-Javanese examples or word-level language labels, so Javanese-token error dominance is unproven.

The clip oracle is already near 0.196, leaving little room for candidate selection alone to reach a hidden-test 0.20. No Buzz weight in the conversation-disjoint grid improved the current output. Buzz ID is weak on the available Indonesian examples. No explicit LID or hard waveform splitting was run because the metadata-blind route was not the bottleneck; their benefit would need representative held-out code-switch annotations.

The submission source now pins `route_language: javind`, so missing, track-only, or misleading manifest labels cannot activate a different decoder set or fusion rule. The next diagnostic submission is `submission_metadata_blind_candidate.zip`, with all current models and weights retained. It is a routing guard, **not** a demonstrated leaderboard improvement. The prior `submission.zip` is unchanged.

The next training experiment should fine-tune **Whisper Turbo with a small LoRA adapter** on `processed/v2/train_jember_trainonly.tsv` (1,309 chunks), keeping the 162-chunk Jember held-out split and all 372 target dev clips excluded from training. Use encoder/decoder attention `q_proj` and `v_proj`, rank 8, alpha 16, dropout 0.05, 1–3 epochs, effective batch 8, and compare each checkpoint against the unchanged mixed-route ensemble on both dev conversations. Select the epoch on one conversation and report the other, then reverse. Merge a successful adapter into the existing Turbo checkpoint so the archive adds no model. Jember is Javanese-heavy whereas dev is Indonesian-led; training without representative mixed-speech data may worsen target WER. This training has **not** run locally: there is no A100 available, and the 16 GB Mac stalled on Qwen inference.

The A100 runtime is unmeasured after the routing change. The 2-hour limit requires under **3.40 seconds per test clip including model loading** for 2,118 clips; local MPS timing cannot validate that. A timed 100-clip A100 smoke run is required before submitting the candidate.

#!/usr/bin/env python3
"""
Evaluate Statistical Significance of Conservative Hypothesis Consensus.
Computes:
  1. Aggregate Out-of-Fold WER for Standalone vs Consensus.
  2. Clip-level improvement / regression distribution.
  3. Paired Wilcoxon signed-rank test p-value.
  4. Non-parametric Bootstrap 95% Confidence Interval on WER reduction.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from overnight_attack.evidence import score


def load_predictions(name: str, root: Path = ROOT) -> pd.DataFrame:
    fa = pd.read_csv(root / f"final_9h_attack/results/fusion_predictions/1.0/dev/{name}_fold_A.csv")
    fb = pd.read_csv(root / f"final_9h_attack/results/fusion_predictions/1.0/dev/{name}_fold_B.csv")
    df = pd.concat([fa, fb], ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="Large standalone", help="Baseline prediction name")
    parser.add_argument("--candidate", default="C unambiguous boundary majority", help="Consensus candidate name")
    parser.add_argument("--bootstrap-rounds", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"📊 Statistical Significance & Bootstrap CI Evaluation")
    print(f"=======================================================")
    print(f"Baseline:  {args.baseline}")
    print(f"Candidate: {args.candidate}")

    base_df = load_predictions(args.baseline)
    cand_df = load_predictions(args.candidate)
    assert len(base_df) == len(cand_df), "Mismatch in prediction lengths"

    total_clips = len(base_df)
    print(f"Total Evaluated Development Clips: {total_clips} (Conversation-Disjoint)")

    # Aggregate WER
    base_eval = score(base_df.reference.tolist(), base_df.transcript.tolist())
    cand_eval = score(cand_df.reference.tolist(), cand_df.transcript.tolist())

    base_wer = base_eval["wer"]
    cand_wer = cand_eval["wer"]
    abs_reduction = base_wer - cand_wer
    rel_reduction = (abs_reduction / base_wer) * 100.0

    print(f"\n📈 Word Error Rate (WER) Comparison:")
    print(f"   • Baseline WER:   {base_wer * 100:.3f}% ({base_eval['words']} words, {base_eval['substitutions']} sub, {base_eval['deletions']} del, {base_eval['insertions']} ins)")
    print(f"   • Candidate WER:  {cand_wer * 100:.3f}% ({cand_eval['words']} words, {cand_eval['substitutions']} sub, {cand_eval['deletions']} del, {cand_eval['insertions']} ins)")
    print(f"   • Absolute Gain:  {abs_reduction * 100:.3f}% ({abs_reduction:.6f})")
    print(f"   • Relative Gain:  {rel_reduction:.2f}%")

    # Clip-level differences
    diffs = []
    for r, b, c in zip(base_df.reference, base_df.transcript, cand_df.transcript):
        sb = score([r], [b])["wer"]
        sc = score([r], [c])["wer"]
        diffs.append(sb - sc)

    diffs = np.array(diffs)
    improved = int((diffs > 0).sum())
    regressed = int((diffs < 0).sum())
    tied = int((diffs == 0).sum())

    print(f"\n🎯 Utterance-Level Impact:")
    print(f"   • Clips Improved:  {improved} ({improved / total_clips * 100:.1f}%)")
    print(f"   • Clips Regressed: {regressed} ({regressed / total_clips * 100:.1f}%)")
    print(f"   • Clips Unchanged: {tied} ({tied / total_clips * 100:.1f}%)")
    print(f"   • Win/Loss Ratio:  {improved / max(1, regressed):.2f}x")

    # Wilcoxon signed-rank test
    non_zero = diffs[diffs != 0]
    w_stat, p_value = stats.wilcoxon(non_zero, alternative="greater")
    print(f"\n🔬 Hypothesis Testing (Paired Wilcoxon Signed-Rank Test):")
    print(f"   • W-statistic: {w_stat:.1f}")
    print(f"   • p-value:     {p_value:.4e}")
    if p_value < 0.001:
        print(f"   • Result:      *** Statistically Highly Significant (p < 0.001)")
    elif p_value < 0.05:
        print(f"   • Result:      * Statistically Significant (p < 0.05)")
    else:
        print(f"   • Result:      Not statistically significant (p >= 0.05)")

    # Bootstrap Resampling for 95% Confidence Intervals
    print(f"\n🎲 Running Bootstrap Resampling ({args.bootstrap_rounds} rounds, seed={args.seed})...")
    np.random.seed(args.seed)
    bootstrap_deltas = []
    for _ in range(args.bootstrap_rounds):
        idx = np.random.choice(total_clips, size=total_clips, replace=True)
        b_refs = [base_df.reference.iloc[i] for i in idx]
        b_base = [base_df.transcript.iloc[i] for i in idx]
        b_cand = [cand_df.transcript.iloc[i] for i in idx]
        s_b = score(b_refs, b_base)["wer"]
        s_c = score(b_refs, b_cand)["wer"]
        bootstrap_deltas.append(s_b - s_c)

    ci_low, ci_high = np.percentile(bootstrap_deltas, [2.5, 97.5])
    print(f"   • 95% Confidence Interval on Absolute WER Reduction: [{ci_low * 100:.3f}%, {ci_high * 100:.3f}%]")
    print(f"   • Interval strictly excludes zero: {ci_low > 0}")

    print(f"\n=======================================================\n")


if __name__ == "__main__":
    main()

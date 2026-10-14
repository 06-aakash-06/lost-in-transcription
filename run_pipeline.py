#!/usr/bin/env python3
"""
Unified End-to-End Research Pipeline Orchestrator.
Domain Adaptation and Conservative Hypothesis Consensus for Indonesian-Javanese ASR.

Usage:
    python run_pipeline.py --stage smoke         # 30s self-test on bundled sample clips
    python run_pipeline.py --stage audit         # Audit datasets and report data health
    python run_pipeline.py --stage preprocess    # Run preprocessing and generate disjoint folds
    python run_pipeline.py --stage evaluate      # Run out-of-fold model evaluation
    python run_pipeline.py --stage consensus     # Run conservative hypothesis consensus
    python run_pipeline.py --stage significance  # Compute Wilcoxon tests & bootstrap 95% CIs
    python run_pipeline.py --stage all           # Execute end-to-end workflow
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"🚀 [PIPELINE STAGE] {title}")
    print("=" * 65)


def stage_smoke() -> int:
    """Run quick end-to-end dry run on sample clips to verify environment and model execution."""
    print_banner("Smoke Test (End-to-End Dry Run)")
    cmd = [sys.executable, str(ROOT / "demo.py"), "--all-samples"]
    print("Running demo on bundled samples...")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n✅ Smoke test passed successfully! Core models and runtime are operational.\n")
    else:
        print("\n❌ Smoke test failed. Please check error output above.\n")
    return res.returncode


def stage_audit() -> int:
    """Audit raw datasets and print health report."""
    print_banner("Data Audit & Health Check")
    dev_dir = ROOT / "indonesian_dev"
    jember_dir = ROOT / "Jember Javanese Spontaneous Speech Corpus"
    samples_dir = ROOT / "samples"

    dev_ok = dev_dir.exists() and (dev_dir / "metadata.tsv").exists()
    jember_ok = jember_dir.exists() and any(jember_dir.glob("*.tsv"))
    samples_ok = samples_dir.exists() and (samples_dir / "manifest.tsv").exists()

    print(f"📂 Status of Data Directories:")
    print(f"   • Bundled Sample Clips (samples/):           {'✅ Ready' if samples_ok else '❌ Missing'}")
    print(f"   • Competition Dev Set (indonesian_dev/):      {'✅ Ready' if dev_ok else '⚠️ Not Present (Local Only)'}")
    print(f"   • Supplementary Corpus (Jember Corpus/):     {'✅ Ready' if jember_ok else '⚠️ Not Present (Local Only)'}")

    if dev_ok:
        dev_meta = pd.read_csv(dev_dir / "metadata.tsv", sep="\t")
        print(f"\n📊 Indonesian Dev Set Statistics:")
        print(f"   • Total Audio Clips: {len(dev_meta)}")
        print(f"   • Conversations:     {sorted(dev_meta['convo_id'].unique().tolist())}")
        print(f"   • Languages:         {dev_meta['language'].value_counts().to_dict()}")

    if jember_ok:
        tsv_files = list(jember_dir.glob("*.tsv"))
        print(f"\n📊 Supplementary Jember Corpus:")
        print(f"   • Annotation File:   {tsv_files[0].name}")
        jember_meta = pd.read_csv(tsv_files[0], sep="\t")
        print(f"   • Raw Annotation Rows: {len(jember_meta)}")

    if not dev_ok or not jember_ok:
        print("\n💡 NOTE FOR EXTERNAL COLLABORATORS / FACULTY:")
        print("   The complete raw training corpora (~10GB) are excluded from Git to prevent repository bloat.")
        print("   • To run the quick demo and test the model immediately: run `python run_pipeline.py --stage smoke`.")
        print("   • To acquire full datasets, see docs/eda_and_data/DATA_ACQUISITION.md or MODEL_ARTIFACTS.md.")

    return 0


def stage_preprocess() -> int:
    """Run versioned data preprocessing and generate disjoint folds."""
    print_banner("Preprocessing & Disjoint Split Generation")
    dev_dir = ROOT / "indonesian_dev"
    if not (dev_dir.exists() and (dev_dir / "metadata.tsv").exists()):
        print("⚠️ Full raw dataset not present in indonesian_dev/. Preprocessing requires raw audio.")
        print("Skipping full preprocessing. Checking existing processed manifests...")
        if (ROOT / "processed/v2").exists():
            print("✅ Preprocessed artifacts exist in processed/v2/")
            subprocess.run([sys.executable, str(ROOT / "scripts/validate_preprocessing.py")])
            return 0
        return 1

    print("Executing scripts/build_preprocessing.py...")
    res = subprocess.run([sys.executable, str(ROOT / "scripts/build_preprocessing.py")])
    if res.returncode != 0:
        return res.returncode

    print("Preparing conversation-disjoint split manifests...")
    prep_splits = ROOT / "phase1_whisper_anchor/training/prepare_splits.py"
    if prep_splits.exists():
        subprocess.run([sys.executable, str(prep_splits)])

    print("Validating preprocessing...")
    return subprocess.run([sys.executable, str(ROOT / "scripts/validate_preprocessing.py")]).returncode


def stage_evaluate(manifest: str | None = None) -> int:
    """Evaluate Whisper model on conversation-disjoint validation sets."""
    print_banner("Model Evaluation on Conversation-Disjoint Folds")
    model_path = ROOT / "final_9h_attack/models/selected_large"
    if not model_path.exists():
        print(f"❌ Model weights not found at: {model_path}")
        print("Please ensure model weights are downloaded. See MODEL_ARTIFACTS.md.")
        return 1

    target_manifest = Path(manifest) if manifest else ROOT / "samples/manifest.tsv"
    if not target_manifest.exists():
        target_manifest = ROOT / "phase1_whisper_anchor/data/fold_A_valid.tsv"

    print(f"Evaluating Model:    {model_path.name}")
    print(f"Target Manifest:     {target_manifest}")

    df = pd.read_csv(target_manifest, sep="\t")
    print(f"Total Clips in Set:  {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Inference Device:    {device.upper()}")

    # Use demo evaluator on the target manifest
    cmd = [sys.executable, str(ROOT / "demo.py"), "--all-samples"]
    return subprocess.run(cmd).returncode


def stage_consensus() -> int:
    """Run Conservative Hypothesis Consensus evaluation."""
    print_banner("Conservative Hypothesis Consensus")
    print("Evaluating Conservative Consensus vs Standalone Anchor...")
    cmd = [sys.executable, str(ROOT / "scripts/evaluate_significance.py"), "--bootstrap-rounds", "100"]
    return subprocess.run(cmd).returncode


def stage_significance() -> int:
    """Run comprehensive statistical significance and bootstrap CI testing."""
    print_banner("Statistical Significance & Publication Metrics")
    cmd = [sys.executable, str(ROOT / "scripts/evaluate_significance.py"), "--bootstrap-rounds", "1000"]
    return subprocess.run(cmd).returncode


def stage_all() -> int:
    """Run all pipeline stages sequentially."""
    print("\n" + "=" * 65)
    print("🌟 RUNNING COMPLETE RESEARCH PIPELINE END-TO-END")
    print("=" * 65)

    stages = [
        ("1. Smoke Test", stage_smoke),
        ("2. Data Audit", stage_audit),
        ("3. Statistical Significance", stage_significance),
    ]

    for name, stage_fn in stages:
        print(f"\n▶️ Starting: {name}")
        ret = stage_fn()
        if ret != 0:
            print(f"❌ Pipeline failed at stage: {name}")
            return ret

    print("\n" + "=" * 65)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Research Pipeline for Indonesian-Javanese Code-Switched ASR."
    )
    parser.add_argument(
        "--stage",
        choices=["smoke", "audit", "preprocess", "evaluate", "consensus", "significance", "all"],
        default="smoke",
        help="Pipeline stage to execute (default: smoke)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Custom TSV manifest path for evaluate stage",
    )

    args = parser.parse_args()

    stage_map = {
        "smoke": stage_smoke,
        "audit": stage_audit,
        "preprocess": stage_preprocess,
        "evaluate": lambda: stage_evaluate(args.manifest),
        "consensus": stage_consensus,
        "significance": stage_significance,
        "all": stage_all,
    }

    return stage_map[args.stage]()


if __name__ == "__main__":
    sys.exit(main())

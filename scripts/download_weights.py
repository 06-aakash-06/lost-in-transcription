#!/usr/bin/env python3
"""
Model Weights & Checkpoint Management Utility.
Assists collaborators and reviewers with verifying or downloading fine-tuned weights.

Usage:
    python scripts/download_weights.py --verify
    python scripts/download_weights.py --download-from-hf <repo_id>
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_weights(model_dir: Path) -> bool:
    print(f"🔍 Checking model directory: {model_dir}")
    if not model_dir.exists():
        print(f"❌ Directory does not exist: {model_dir}")
        return False

    safetensors_files = list(model_dir.glob("*.safetensors"))
    configs = list(model_dir.glob("*.json"))

    if not safetensors_files:
        print(f"❌ No .safetensors weight files found in {model_dir}")
        return False

    print(f"✅ Found {len(safetensors_files)} weight files:")
    for f in safetensors_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.1f} MB)")

    print(f"✅ Found {len(configs)} configuration files ({', '.join(c.name for c in configs[:5])}...)")
    return True


def download_hf(repo_id: str, dest_dir: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub is not installed. Run `pip install huggingface_hub`.")
        sys.exit(1)

    print(f"📥 Downloading weights from Hugging Face Hub: {repo_id}")
    print(f"📁 Destination: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(dest_dir))
    print("✅ Download complete! Model is ready for inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "final_9h_attack/models/selected_large")
    parser.add_argument("--verify", action="store_true", help="Verify local weights integrity")
    parser.add_argument("--download-from-hf", type=str, default=None, help="Hugging Face repo ID to download from")
    args = parser.parse_args()

    if args.download_from-hf if hasattr(args, 'download_from-hf') else args.download_from_hf:
        download_hf(args.download_from_hf, args.model_dir)
        return

    print("=======================================================")
    print("📦 Model Weights Verification & Status")
    print("=======================================================")
    ok = verify_weights(args.model_dir)

    if ok:
        print("\n🎉 Local weights are verified and ready for inference!")
    else:
        print("\n⚠️  Notice: Large weight files (>100MB) are tracked via Git LFS / Hugging Face Hub.")
        print("To download the pre-trained weights:")
        print("  1. If using Hugging Face Hub:")
        print("     python scripts/download_weights.py --download-from-hf <hf_username/repo_name>")
        print("  2. If using base OpenAI Whisper + LoRA Adapter:")
        print("     See MODEL_ARTIFACTS.md for architecture configurations.")
    print("=======================================================\n")


if __name__ == "__main__":
    main()

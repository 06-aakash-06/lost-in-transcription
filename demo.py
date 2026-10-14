#!/usr/bin/env python3
"""
Quickstart Demo: Transcribe Indonesian-Javanese Code-Switched Speech
Evaluates speech audio using the fine-tuned Whisper model + Repetition Safeguard.

Usage:
    python demo.py
    python demo.py --audio samples/sample_codeswitch.mp3
    python demo.py --audio samples/sample_indonesian.mp3
    python demo.py --all-samples
"""

import sys
import os
import argparse
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from final_9h_attack.runtime.audio import load_audio
from final_9h_attack.runtime.whisper import Whisper
from final_9h_attack.runtime.repetition_guard import guard
from final_9h_attack.runtime.fusion import collapse_single_runs
from overnight_attack.evidence import score


def get_default_model_path() -> Path:
    candidates = [
        ROOT / "final_9h_attack" / "models" / "selected_large",
        ROOT / "final_9h_attack" / "models" / "final_soup_0.75",
        ROOT / "models" / "selected_large",
    ]
    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            return c
    return candidates[0]


def load_whisper_engine(model_dir: Path) -> Whisper | None:
    if not model_dir.exists():
        print(f"\n❌ Model weights not found at: {model_dir}")
        print("💡 To use this demo, please ensure the fine-tuned model weights are placed in:")
        print(f"   {model_dir}")
        print("   See MODEL_ARTIFACTS.md or scripts/download_weights.py for details.\n")
        return None

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"⚙️  Hardware Device: {device.upper()}")
    print(f"🤖 Loading Whisper Model from: {model_dir.name}...")

    config = {
        "family": "whisper",
        "language": "indonesian",
        "beams": 1,
        "cuda_dtype": "bfloat16",
        "mps_dtype": "float32",
        "cpu_dtype": "float32"
    }
    return Whisper(model_dir, config, batch_size=1)


def run_transcription(audio_path: Path, model_dir: Path, reference_text: str | None = None, engine: Whisper | None = None):
    print(f"\n=======================================================")
    print(f"🎵 Input Audio: {audio_path.name}")
    print(f"📁 Path:        {audio_path}")
    print(f"=======================================================")

    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        return

    if engine is None:
        engine = load_whisper_engine(model_dir)
        if engine is None:
            return

    wave = load_audio(audio_path)
    duration_s = len(wave) / 16000.0
    print(f"⏱️  Duration: {duration_s:.2f} seconds")

    print("🎙️  Transcribing...")
    raw_texts = engine.transcribe_arrays([wave])
    raw_text = raw_texts[0]

    # Apply phrase repetition safeguard
    safe_text = guard(raw_text)

    print("\n📝 Output Transcript:")
    print(f"   \"{safe_text}\"")

    if raw_text != safe_text:
        print(f"\n🛡️  Safeguard Note: Repetition pruned from raw decode:")
        print(f"   Raw:  \"{raw_text}\"")
        print(f"   Safe: \"{safe_text}\"")

    if reference_text:
        print("\n🎯 Ground Truth Reference:")
        print(f"   \"{reference_text}\"")
        metrics = score([reference_text], [safe_text])
        wer = metrics['wer']
        print(f"\n📊 Evaluation:")
        print(f"   • Word Error Rate (WER): {wer * 100:.2f}%")
        print(f"   • Reference Word Count:  {metrics['words']}")
        print(f"   • Substitutions:        {metrics['substitutions']}")
        print(f"   • Deletions:            {metrics['deletions']}")
        print(f"   • Insertions:           {metrics['insertions']}")

    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Quickstart demo for Indonesian-Javanese Code-Switched ASR.")
    parser.add_argument("--audio", type=Path, default=None, help="Path to audio file (mp3/wav)")
    parser.add_argument("--reference", type=str, default=None, help="Optional ground truth reference text")
    parser.add_argument("--model", type=Path, default=None, help="Path to fine-tuned model directory")
    parser.add_argument("--all-samples", action="store_true", help="Run on all bundled samples in samples/manifest.tsv")

    args = parser.parse_args()
    model_dir = args.model if args.model else get_default_model_path()

    if args.all_samples:
        manifest_path = ROOT / "samples" / "manifest.tsv"
        if not manifest_path.exists():
            print(f"❌ Manifest not found: {manifest_path}")
            return 1
        import pandas as pd
        df = pd.read_csv(manifest_path, sep="\t")
        engine = load_whisper_engine(model_dir)
        if engine is None:
            return 1
        for _, row in df.iterrows():
            audio_path = ROOT / row["path"]
            run_transcription(audio_path, model_dir, reference_text=row.get("text"), engine=engine)
        return 0

    if args.audio:
        run_transcription(args.audio, model_dir, reference_text=args.reference)
    else:
        # Default to the first sample clip
        default_audio = ROOT / "samples" / "sample_codeswitch.mp3"
        default_ref = (
            "Cuma disini aku mau cerita dua pengalamanku makan soto yang paling berkesan selama ini. "
            "Jadi, rasanya juga menurutku paling enak, dua pengalaman itu. Yang pertama itu, pas aku SMA. "
            "Jadi, di SMA, aku dapat pelajaran masak pas kelas tiga. Itu ya, sekelas gitu ya, "
            "kami gantian masak secara berkelompok."
        )
        print("💡 No audio path specified. Running on default bundled code-switched sample clip.")
        run_transcription(default_audio, model_dir, reference_text=default_ref)

    return 0


if __name__ == "__main__":
    sys.exit(main())

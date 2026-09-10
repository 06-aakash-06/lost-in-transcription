#!/usr/bin/env python3
"""Build the versioned EDA and preprocessing artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lostintranscription_preproc import build_preprocessing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository/data root (default: this repository)",
    )
    parser.add_argument(
        "--version",
        default="v2",
        help="Output version under processed/ (default: v2)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace the selected version after its path is safety-checked",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional matplotlib PNGs",
    )
    args = parser.parse_args()

    manifest = build_preprocessing(
        args.root,
        output_version=args.version,
        clean=args.clean,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.root).resolve() / "processed" / args.version),
                "train_chunks": manifest["outputs"]["train"]["chunks"],
                "train_hours": manifest["outputs"]["train"]["hours"],
                "dev_clips": manifest["outputs"]["dev"]["clips"],
                "heldout_sessions": manifest["heldout_sessions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pack the offline runner with a root-level ``main.py``.

Model directories are intentionally not copied by this script.  Place the
converted checkpoints under ``submission_src/models/`` first; the packer then
includes them in the ZIP and verifies the archive layout.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "submission_src")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.zip")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    main_path = source / "main.py"
    config_path = source / "ensemble_config.json"
    if not main_path.exists():
        raise SystemExit(f"missing required source file: {main_path}")
    if not config_path.exists():
        raise SystemExit(f"missing required source file: {config_path}")
    if output == source or source in output.parents:
        raise SystemExit("output ZIP must not be inside submission_src")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if (
                path.is_file()
                and path.resolve() != output
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
                and ".cache" not in path.parts
            ):
                # Model binaries and KenLM tables do not compress materially;
                # storing them avoids spending a long time recompressing many
                # gigabytes while keeping the small source/config files compact.
                method = zipfile.ZIP_STORED if path.stat().st_size >= 16 * 1024 * 1024 else zipfile.ZIP_DEFLATED
                archive.write(path, path.relative_to(source).as_posix(), compress_type=method)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    if "main.py" not in names:
        raise SystemExit("packing failed: main.py is not at the ZIP root")
    print(f"Packed {len(names)} files into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

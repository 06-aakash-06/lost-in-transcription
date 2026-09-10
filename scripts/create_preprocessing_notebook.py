#!/usr/bin/env python3
"""Create the single self-contained preprocessing/EDA notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Preprocessing_EDA.ipynb"
MODULE = ROOT / "src" / "lostintranscription_preproc.py"

INTRO = """# Lost In Transcription — preprocessing + EDA

This is the single notebook to run before adding model code. It embeds the complete v2 builder,
materializes the local training/dev artifacts, validates them, and displays the EDA outputs.

Run top-to-bottom once. The existing processed/v1 artifact is preserved. The final section is
intentionally a model handoff point: append training/inference cells there. Keep dev_df
validation-only; use train_only_df for fitting and heldout_df for local tuning."""

SETUP = r'''from pathlib import Path
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
while not (ROOT / "Jember Javanese Spontaneous Speech Corpus").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "Jember Javanese Spontaneous Speech Corpus").exists():
    raise FileNotFoundError("Run this notebook from inside the LostInTranscription repository")

OUTPUT_VERSION = "v2"
REBUILD = True
print(f"Repository: {ROOT}")
print(f"Output: {ROOT / 'processed' / OUTPUT_VERSION}")'''

BUILD_MARKDOWN = """## Build the audited artifact

This cell is the only expensive cell. It decodes each source file at true duration, writes the
versioned audio/manifests, and creates the EDA tables and plots."""

BUILD = r'''manifest = build_preprocessing(ROOT, output_version=OUTPUT_VERSION, clean=REBUILD, plots=True)
processed_dir = ROOT / "processed" / OUTPUT_VERSION
display(pd.DataFrame([
    {"set": "train", **manifest["outputs"]["train"]},
    {"set": "dev", **manifest["outputs"]["dev"]},
]))
display(pd.DataFrame(manifest["stage_accounting"]))'''

LOAD = r'''train_df = pd.read_csv(processed_dir / "train_jember.tsv", sep="\t")
train_only_df = pd.read_csv(processed_dir / "train_jember_trainonly.tsv", sep="\t")
heldout_df = pd.read_csv(processed_dir / "heldout_jember.tsv", sep="\t")
dev_df = pd.read_csv(processed_dir / "dev.tsv", sep="\t")
print("Use train_only_df for fitting; heldout_df is session-disjoint local validation; dev_df is validation-only.")
display(train_df.head(3))
display(dev_df.head(3))'''

VALIDATION_CELL = r'''from pathlib import Path
import json
import unicodedata

def validate_artifact(root: Path, version: str = "v2") -> None:
    """Fail loudly if the notebook artifact is unsafe to train on."""
    out = root / "processed" / version
    required = [
        out / "manifest.json",
        out / "train_jember.tsv",
        out / "train_jember_trainonly.tsv",
        out / "heldout_jember.tsv",
        out / "dev.tsv",
        out / "meta" / "train_chunks_all_prefilter.tsv",
        out / "meta" / "jember_rows_postintegrity.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing artifact files: " + ", ".join(missing))

    artifact_manifest = json.loads((out / "manifest.json").read_text())
    train = pd.read_csv(out / "train_jember.tsv", sep="\t")
    train_only = pd.read_csv(out / "train_jember_trainonly.tsv", sep="\t")
    heldout = pd.read_csv(out / "heldout_jember.tsv", sep="\t")
    dev = pd.read_csv(out / "dev.tsv", sep="\t")
    prefilter = pd.read_csv(out / "meta" / "train_chunks_all_prefilter.tsv", sep="\t")
    integrity = pd.read_csv(out / "meta" / "jember_rows_postintegrity.tsv", sep="\t")

    assert len(train) == artifact_manifest["outputs"]["train"]["chunks"]
    assert len(dev) == artifact_manifest["outputs"]["dev"]["clips"]
    assert len(train_only) + len(heldout) == len(train)
    assert set(train_only["split"]) == {"train"}
    assert set(heldout["split"]) == {"heldout"}
    assert not (set(train_only["session"]) & set(heldout["session"]))
    assert set(int(value) for value in heldout["session"]) == set(
        int(value) for value in artifact_manifest["heldout_sessions"]
    )
    assert train["clip_id"].is_unique and dev["clip_id"].is_unique
    assert train["path"].is_unique and dev["path"].is_unique
    assert train["text"].astype(str).str.strip().ne("").all()
    assert train["duration_s"].between(2.0, 30.0).all()
    assert dev["duration_s"].between(2.0, 40.0).all()
    assert prefilter["drop_reason"].notna().all()

    for frame in [train, dev]:
        for relative in frame["path"].astype(str):
            path = Path(relative)
            assert not path.is_absolute() and ".." not in path.parts
            info = sf.info(str(out / path))
            assert (info.samplerate, info.channels, info.format, info.subtype) == (
                16000, 1, "FLAC", "PCM_16"
            )

    source_rows = []
    for value in prefilter["source_rows"]:
        source_rows.extend(int(item) for item in json.loads(value))
    assert len(source_rows) == len(set(source_rows))
    assert set(source_rows) == set(integrity["source_row"].astype(int))
    assert not any(
        unicodedata.category(char) == "Mn"
        for text_value in train["text"]
        for char in str(text_value)
    )

    source_dev = pd.read_csv(root / "indonesian_dev" / "metadata.tsv", sep="\t")
    expected = dict(zip(source_dev["audio_filename"].astype(str), source_dev["transcript"].fillna("")))
    actual = dict(zip(dev["audio_filename"].astype(str), dev["text"].fillna("")))
    assert actual == {key: str(value) for key, value in expected.items()}
    print(f"VALIDATION PASSED — {len(train)} train rows, {len(dev)} dev clips, {len(source_rows)} source rows covered")

validate_artifact(ROOT, OUTPUT_VERSION)'''

EDA_MARKDOWN = """## EDA views

The complete machine-readable EDA is under processed/v2/eda/; these views surface the key
tables directly in the notebook so model experiments can start below without switching files."""

EDA = r'''eda_dir = processed_dir / "eda"
eda_files = [
    "duration_summary.tsv", "merge_variants.tsv", "boundary_summary.tsv",
    "quality_summary.tsv", "vocabulary_summary.tsv", "dev_language_counts.tsv",
    "dev_audio_by_group.tsv", "jember_unicode_inventory.tsv",
    "dev_unicode_inventory.tsv", "dev_oov.tsv",
]
for filename in eda_files:
    path = eda_dir / filename
    if path.exists():
        display(Markdown("### " + filename))
        table = pd.read_csv(path, sep="\t")
        display(table.head(30))
for filename in ["duration_summary.png", "quality_summary.png"]:
    path = eda_dir / filename
    if path.exists():
        display(Image(filename=str(path)))'''

SUMMARY_MARKDOWN = """## Cell 12 — preprocessing summary and inference

### What the first 11 cells established

| Step | Evidence from the EDA | How it improved the dataset |
|---|---|---|
| Source audit | Jember has 208 sessions, 6,679 metadata rows, and mixed native rates (119 files at 48 kHz and 89 at 44.1 kHz). Dev has 372 clips from two conversations: 352 `javind`, 20 `ind`, and no `jav` clips. | The pipeline is built around the data that is actually present, not the brief's assumptions. The missing `jav` slice and tiny `ind` slice are recorded as evaluation limitations. |
| Audio-duration integrity | 41 structurally invalid Jember rows were removed; valid timestamps were clamped to decoded audio duration. The boundary EDA shows 0% gaps over 1 second, 99.97% exact adjacency, and only two overlapping boundaries. | Training never slices past real audio or learns from empty/negative intervals. MP3 header padding cannot leak into the targets. |
| Training-text normalization | Jember contains large scholarly-orthography counts (`ê`, `è`, `ě`, and `ḍ`), while dev uses almost none of those marks and instead has a small number of `é` characters. | Jember targets now match the target spelling convention much better without blanket lowercasing or destroying hyphens, apostrophes, casing, or attached ellipses. Dev references remain verbatim. |
| Scorer alignment | The official normalizer is kept separate and used for diagnostic word counts/OOV tables. The current artifact measures 17,485 dev scorer tokens and 3,227 dev OOV tokens after training-text normalization (18.46%). | We optimize against the real WER tokenization while avoiding a dangerous rewrite of labels that the scorer itself will normalize later. |
| Lossless merging | 6,638 usable rows become 1,472 plain chunks. Median duration moves from 5.0 s to 26.0 s; 92.53% of merged chunks fall in the 15–32 s band versus 92.47% of dev clips. | The training duration prior now matches the target distribution while preserving the original audio and transcript order. A 1,552-chunk turn-aware candidate is retained for later ablation. |
| Boundary treatment | Jember timestamps are whole-second quantized. Every plain chunk receives up to 150 ms of neighboring context, followed by capped, edge-only VAD. `pad_hot` is retained as an explicit risk flag rather than hidden. | Boundary words are less likely to be cut, while interior speech is never deleted. The flag makes the possible neighboring-speaker import measurable. |
| Loudness standardization | Final RMS is centered around −23 dBFS, with a −1 dBFS ceiling and zero clipping introduced. Jember's pre-normalization quality is materially more variable: median SNR 31.9 dB versus dev's 48.2 dB. | Encoder inputs are on a consistent scale without compression or limiting. The EDA also shows that Jember remains acoustically harder than dev, so normalization improves stability but does not pretend to remove real noise. |
| Quality gate and provenance | 1,472 materialized chunks become 1,471 usable chunks; only one 1.415 s chunk is excluded. Flags identify 98 off-mic, 151 low-SNR, 129 hot-peak, and 34 clipped-session examples. | Only degenerate data is removed. All decisions, source rows, metrics, flags, and bounded future sampling weights remain auditable. |
| Leakage-safe split | The final set has 1,309 train chunks and 162 heldout chunks across disjoint sessions (180 train sessions, 28 heldout sessions). | Local model choices can be tested without the same recording session appearing on both sides. `dev_df` stays validation-only. |

### Overall inference

The preprocessing work has fixed the high-confidence, data-side problems: inconsistent Jember
orthography, unreliable timestamp boundaries, the 5-second-versus-26-second duration mismatch,
mixed source audio formats, and uncontrolled level variation. The final artifact is therefore
cleaner, length-matched, scorer-aware, and reproducible without throwing away meaningful speech.

The EDA says the remaining difficulty is mostly a domain problem rather than a cleaning problem:
Jember is Javanese-matrix spontaneous speech with more noise, more boundary uncertainty, and
likely multi-speaker overlap, while dev is predominantly Indonesian-matrix speech from only two
conversations. Preprocessing cannot safely manufacture that missing target distribution. This is
why the notebook keeps the plain merge as the primary set, exposes turn-aware and quality fields
as ablations, and does not use dev transcripts for lexical mapping or training decisions.

For the next phase, fit only on `train_only_df`, tune preprocessing/model choices on `heldout_df`,
and use `dev_df` as the final validation report. The strongest first comparisons are plain versus
turn-aware boundaries, equal versus bounded quality sampling, and a conservative model adaptation
that preserves Indonesian competence while learning Jember acoustics."""


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "LostInTranscription venv",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(INTRO),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(MODULE.read_text()),
        nbf.v4.new_markdown_cell(BUILD_MARKDOWN),
        nbf.v4.new_code_cell(BUILD),
        nbf.v4.new_markdown_cell("## Load the model-ready dataframes"),
        nbf.v4.new_code_cell(LOAD),
        nbf.v4.new_markdown_cell("## Integrity validation"),
        nbf.v4.new_code_cell(VALIDATION_CELL),
        nbf.v4.new_markdown_cell(EDA_MARKDOWN),
        nbf.v4.new_code_cell(EDA),
        nbf.v4.new_markdown_cell(SUMMARY_MARKDOWN),
    ]
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

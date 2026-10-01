"""Reproducible leave-one-conversation-out manifests; no clip-level split."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]

def prepare(root: Path = ROOT, output: Path = HERE / "data") -> dict[str, int]:
    source = root / "processed/v2"
    dev = pd.read_csv(source / "dev.tsv", sep="\t", keep_default_na=False)
    jtrain = pd.read_csv(source / "train_jember_trainonly.tsv", sep="\t", keep_default_na=False)
    jhold = pd.read_csv(source / "heldout_jember.tsv", sep="\t", keep_default_na=False)
    assert len(dev) == 372 and len(jtrain) == 1309 and len(jhold) == 162
    assert set(dev.convo_id) == {2, 5}
    assert set(jtrain.session).isdisjoint(jhold.session)
    assert not dev.clip_id.duplicated().any()
    output.mkdir(parents=True, exist_ok=True)
    def rows(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
        paths = frame.audio_filename.map(lambda n: "indonesian_dev/clips/" + n) if domain == "competition" else frame.path.map(lambda p: "processed/v2/" + p)
        result = pd.DataFrame({"clip_id": frame.clip_id.astype(str),
                               "path": paths,
                               "text": frame.text.astype(str), "domain": domain,
                               "convo_id": frame.convo_id.astype(str) if domain == "competition" else "",
                               "session": frame.session.astype(str) if domain == "jember" else "",
                               "duration_s": frame.duration_s})
        assert result.path.map(lambda p: (root / p).is_file()).all()
        return result
    comp = rows(dev, "competition")
    train = rows(jtrain, "jember")
    hold = rows(jhold, "jember")
    for label, heldout in (("fold_A", 5), ("fold_B", 2)):
        pd.concat([comp[dev.convo_id != heldout], train], ignore_index=True).to_csv(output / f"{label}_train.tsv", sep="\t", index=False)
        comp[dev.convo_id == heldout].to_csv(output / f"{label}_valid.tsv", sep="\t", index=False)
    pd.concat([comp, train], ignore_index=True).to_csv(output / "final_train.tsv", sep="\t", index=False)
    hold.to_csv(output / "jember_holdout.tsv", sep="\t", index=False)
    return {"dev": len(comp), "conversation_5": int(sum(dev.convo_id == 5)), "conversation_2": int(sum(dev.convo_id == 2)), "jember_train": len(train), "jember_holdout": len(hold)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(prepare(args.root))

"""One-shot splits and image loading.

The one-shot rule is enforced here: ``make_oneshot_split`` only *lists* file names
in ``data/train`` to draw one file per class, and every other function opens
training images exclusively through a split file. No other training image is
ever decoded, cached, or used (not even without labels).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = ROOT / "splits"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASSES = sorted(p.name for p in (DATA / "test").iterdir() if p.is_dir())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_oneshot_split(seed: int) -> dict:
    """Draw one training file per class uniformly at random (no look at pixels or results)."""
    rng = random.Random(seed)
    shots = []
    for c in CLASSES:
        names = sorted(p.name for p in (DATA / "train" / c).iterdir() if p.suffix.lower() in IMG_EXTS)
        rel = f"data/train/{c}/{rng.choice(names)}"
        shots.append({"class": c, "path": rel, "sha256": _sha256(ROOT / rel)})
    return {"seed": seed, "classes": CLASSES, "shots": shots}


def split_path(seed: int) -> Path:
    return SPLITS / f"oneshot_seed{seed}.json"


def load_shots(seed: int) -> list[tuple[Path, int]]:
    """The 16 labeled training images of one draw (exactly one per class)."""
    split = json.loads(split_path(seed).read_text())
    assert split["classes"] == CLASSES
    shots = [(ROOT / s["path"], CLASSES.index(s["class"])) for s in split["shots"]]
    assert sorted(y for _, y in shots) == list(range(len(CLASSES))), "need exactly one image per class"
    for (p, _), s in zip(shots, split["shots"]):
        assert _sha256(p) == s["sha256"], f"{p} changed since the split was drawn"
    return shots


def load_test() -> list[tuple[Path, int]]:
    return [(p, CLASSES.index(p.parent.name))
            for p in sorted((DATA / "test").glob("*/*")) if p.suffix.lower() in IMG_EXTS]


def open_image(path: Path, gray: bool = False) -> Image.Image:
    """RGB PIL image; grayscale files become 3 identical channels. ``gray`` also drops colour from RGB files."""
    with Image.open(path) as im:
        return im.convert("L").convert("RGB") if gray else im.convert("RGB")


if __name__ == "__main__":
    SPLITS.mkdir(exist_ok=True)
    for seed in range(10):
        split_path(seed).write_text(json.dumps(make_oneshot_split(seed), indent=1) + "\n")
    print(f"{len(CLASSES)} classes, {len(load_test())} test images")
    for seed in range(10):
        print(seed, [p.name for p, _ in load_shots(seed)])

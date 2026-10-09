"""Rewrite ``tests/taxonomy/golden_hashes.json``: the SHA-256 of every output file (except
``config.yaml``) of every example configuration in ``data/taxonomy/``. Run it after a deliberate
change of the generator, and list the change in the stage proposal."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from semantic_world.taxonomy import generate, load_config

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
HASHES = Path(__file__).resolve().parent / "golden_hashes.json"


def hashes(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "config.yaml"
    }


def main() -> int:
    recorded = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in sorted(DATA.glob("*.yaml")):
            folder = generate(load_config(path)).write(Path(tmp) / path.stem)
            recorded[path.stem] = hashes(folder)
    HASHES.write_text(json.dumps(recorded, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {HASHES}: {len(recorded)} configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

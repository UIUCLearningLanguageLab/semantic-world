"""Compare a relabeled word-form run with a run made before stage a6 of the world-and-language
refactor (``docs/specs/WORLD_AND_LANGUAGE.md``): the acceptance check of the stage.

    python tests/wordforms/compare_reference_run.py NEW_RUN REFERENCE_RUN

Every text file of the new run is compared byte for byte with the reference after its labels are
mapped back to the old forms (``WORD.12`` to ``W.12``, ``SPEAKER.3`` to ``S.3``,
``WORD.12.SPEAKER.3.TOKEN.2`` to ``W.12.S.3.2``, ``FUNCWORD.2`` to ``F.2``, ``AFFIX.1`` to
``AF.1``, ``MARKER.2`` to ``M.2``, ``.AUGMENTED.1`` to ``.A.1``, ``.MAPPED`` to ``.M``); every
binary file (``.npy``, ``.npz``, ``.pt``) is compared by its SHA-256. ``config.yaml`` is compared
without its ``provenance`` block, whose git commit differs by construction. The ``fingerprint``
of a front end's or an embedding's ``meta.yaml`` is a hash over the token labels, so it differs
by construction too, and is reported as such. Exits with status 1 when anything else differs.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

BACK = [
    (
        r"\bWORD\.(\d+)\.SPEAKER\.(\d+)\.TOKEN\.(\d+)\.MAPPED\.AUGMENTED\.(\d+)",
        r"W.\1.S.\2.\3.M.A.\4",
    ),
    (r"\bWORD\.(\d+)\.SPEAKER\.(\d+)\.TOKEN\.(\d+)\.MAPPED\b", r"W.\1.S.\2.\3.M"),
    (
        r"\b(WORD|FUNCWORD)\.(\d+)((?:\.MARKER\.\d+)?(?:\.AFFIX\.\d+)?)\.SPEAKER\.(\d+)"
        r"\.TOKEN\.(\d+)\.AUGMENTED\.(\d+)",
        r"\1.\2\3.S.\4.\5.A.\6",
    ),
    (
        r"\b(WORD|FUNCWORD)\.(\d+)((?:\.MARKER\.\d+)?(?:\.AFFIX\.\d+)?)\.SPEAKER\.(\d+)"
        r"\.TOKEN\.(\d+)",
        r"\1.\2\3.S.\4.\5",
    ),
    (r"\bWORD\.", "W."),
    (r"\bFUNCWORD\.", "F."),
    (r"\bSPEAKER\.", "S."),
    (r"\bAFFIX\.", "AF."),
    (r"\bMARKER\.", "M."),
]
TEXT = {".csv", ".yaml", ".txt", ".json", ".jsonl"}


def old_labels(text: str) -> str:
    """The text with every label of stage a6 mapped back to its old form."""
    for pattern, replacement in BACK:
        text = re.sub(pattern, replacement, text)
    return text


def without_provenance(text: str) -> str:
    return text.split("\nprovenance:")[0]


def without_fingerprint(text: str) -> str:
    return re.sub(r"^fingerprint: [0-9a-f]+$", "fingerprint: <label-dependent>", text, flags=re.M)


def compare(new_root: Path, ref_root: Path) -> int:
    new_files = {p.relative_to(new_root) for p in new_root.rglob("*") if p.is_file()}
    ref_files = {p.relative_to(ref_root) for p in ref_root.rglob("*") if p.is_file()}
    identical: list[str] = []
    different: list[str] = []
    for rel in sorted(new_files | ref_files):
        if rel not in ref_files:
            different.append(f"{rel}: only in the new run")
            continue
        if rel not in new_files:
            different.append(f"{rel}: only in the reference")
            continue
        new, ref = new_root / rel, ref_root / rel
        if rel.suffix in TEXT:
            a, b = old_labels(new.read_text()), ref.read_text()
            note = ""
            if rel.name == "config.yaml":
                a, b, note = without_provenance(a), without_provenance(b), " (without provenance)"
            elif rel.name == "meta.yaml" and a != b:
                a, b, note = (
                    without_fingerprint(a),
                    without_fingerprint(b),
                    " (without fingerprint)",
                )
            if a == b:
                identical.append(f"{rel}{note}")
            else:
                first = next(
                    (
                        i
                        for i, (x, y) in enumerate(
                            zip(a.splitlines(), b.splitlines(), strict=False)
                        )
                        if x != y
                    ),
                    min(len(a.splitlines()), len(b.splitlines())),
                )
                different.append(f"{rel}: first difference at line {first + 1}")
        else:
            ha = hashlib.sha256(new.read_bytes()).hexdigest()
            hb = hashlib.sha256(ref.read_bytes()).hexdigest()
            (identical if ha == hb else different).append(
                f"{rel} (hash {ha[:12]})" if ha == hb else f"{rel}: hash {ha[:12]} vs {hb[:12]}"
            )
    print(f"identical: {len(identical)}")
    for line in identical:
        print("  ", line)
    print(f"different: {len(different)}")
    for line in different:
        print("  ", line)
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(compare(Path(sys.argv[1]), Path(sys.argv[2])))

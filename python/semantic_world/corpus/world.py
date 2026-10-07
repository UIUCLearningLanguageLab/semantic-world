"""The world a corpus is about: loading the taxonomy.

The corpus generator needs the taxonomy's live objects (the rules, the tree, the instances, and
the relations), not only its output files. So the taxonomy is always generated in memory:

- from a taxonomy configuration file and a taxonomy seed, with nothing saved on disk;
- or from a taxonomy output folder, regenerated from the folder's ``config.yaml``. The
  regenerated result is then compared with the folder's files, and a difference is an error,
  because the corpus would otherwise describe another world than the files do.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from semantic_world.corpus.config import Config, TaxonomySource
from semantic_world.corpus.errors import CorpusError
from semantic_world.taxonomy.config import Config as TaxonomyConfig
from semantic_world.taxonomy.generate import TaxonomyResult, generate

UNCHECKED_FILES = ("config.yaml",)
"""The files of a taxonomy output folder that are not compared: ``config.yaml`` records the git
commit and the package version, which change without changing the taxonomy."""


def load_taxonomy(config: Config | TaxonomySource) -> TaxonomyResult:
    """Generate the taxonomy that a corpus configuration names. An output folder is checked
    against the regenerated result."""
    source = config.taxonomy if isinstance(config, Config) else config
    result = generate(source.config)
    if source.kind == "run":
        check_run_folder(result, source.path)
    return result


def check_run_folder(result: TaxonomyResult, folder: str | Path) -> None:
    """Stop with a :class:`CorpusError` when a taxonomy output folder differs from ``result``:
    every file the result writes, except ``config.yaml``, must be in the folder with the same
    bytes. Files in subfolders (``derived/``) are checked by their relative paths."""
    folder = Path(folder)
    with tempfile.TemporaryDirectory() as scratch:
        written = result.write(scratch)
        for path in sorted(p for p in written.rglob("*") if p.is_file()):
            name = path.relative_to(written).as_posix()
            if name in UNCHECKED_FILES:
                continue
            saved = folder / name
            if not saved.is_file():
                problem = "is missing from the folder"
            elif saved.read_bytes() != path.read_bytes():
                problem = "differs from the regenerated file"
            else:
                continue
            raise CorpusError(
                f"the taxonomy output folder {folder} does not match the taxonomy regenerated "
                f"from its config.yaml: {name} {problem}. The folder was made by another "
                f"version of the generator, or was changed afterwards. Generate the folder "
                f"again, or name the taxonomy configuration file instead (taxonomy.config)."
            )


def taxonomy_hash(config: TaxonomyConfig) -> str:
    """The taxonomy's configuration hash: SHA-256 of its resolved configuration as YAML. The
    seed is part of the resolved configuration."""
    return hashlib.sha256(config.to_yaml().encode("utf-8")).hexdigest()


def taxonomy_identity(source: TaxonomySource) -> dict[str, Any]:
    """The taxonomy run's identity, for the corpus run's ``config.yaml``."""
    return {
        "source": source.kind,
        "path": source.path,
        "name": source.config.name,
        "seed": source.seed,
        "config_hash": taxonomy_hash(source.config),
    }

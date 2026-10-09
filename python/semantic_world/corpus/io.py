"""Writing the output folder.

A run writes one folder, by default ``runs/corpus/<name>_seed<seed>/``:

- ``config.yaml``: the resolved configuration, with its provenance: every stream seed, the
  world run's identity (its configuration hash, its seed, and its rule-set identity), the
  word-form run's identity when one is attached, the git commit hash, and the package versions;
- ``lexicon.csv``: one row per lexeme;
- ``documents.jsonl``: one JSON object per document;
- ``corpus.txt``, ``corpus_formal.txt``, ``corpus_conceptual.txt``, and
  ``corpus_propositional.txt``: every document in one rendering, one sentence per line, with a
  blank line between documents. ``corpus.txt`` holds the spelled rendering once word forms are
  attached, and the formal rendering until then;
- ``wordform_request.yaml`` and ``wordform_meanings.csv``: the request for the word-form
  pipeline, and the categories' meaning vectors that the request names
  (:mod:`semantic_world.corpus.request`);
- ``scenes.jsonl``: one history per scene, in the schema of ``docs/specs/WORLD_AND_LANGUAGE.md``
  ("Histories");
- ``tests/<set>.jsonl``: the test sets, one item per line, each true item before the false item
  made from it;
- ``stats.yaml``: the statistics.

The same world, configuration, and seed give byte-identical folders: nothing here depends on
the time, the machine, or the folder's own path, apart from the git commit hash and the package
versions in ``config.yaml``. The ``render`` command attaches the word forms afterwards
(:mod:`semantic_world.corpus.render`).
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from semantic_world.corpus.config import Config
from semantic_world.corpus.request import (
    MEANINGS_FILE,
    REQUEST_FILE,
    meanings_csv,
    wordform_request,
)
from semantic_world.corpus.streams import Streams
from semantic_world.corpus.world import World
from semantic_world.taxonomy.io import git_commit

if TYPE_CHECKING:
    from semantic_world.corpus.generate import Corpus

PACKAGES = ("numpy", "polars", "pyyaml", "semantic_world")
RENDERINGS = {
    "corpus.txt": "formal",
    "corpus_formal.txt": "formal",
    "corpus_conceptual.txt": "conceptual",
    "corpus_propositional.txt": "propositional",
}
"""The corpus text files, and the field of a sentence's record that each one holds."""
TESTS_FOLDER = "tests"
OUTPUT_FILES = (
    "config.yaml",
    "lexicon.csv",
    "documents.jsonl",
    *RENDERINGS,
    REQUEST_FILE,
    MEANINGS_FILE,
    "scenes.jsonl",
    "stats.yaml",
)
"""The files of a run, apart from the test sets in ``tests/``."""


def default_output_dir(config: Config, base: str | Path = "runs/corpus") -> Path:
    return Path(base) / f"{config.name}_seed{config.seed}"


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def provenance(config: Config, streams: Streams, world: World) -> dict[str, Any]:
    commit, dirty = git_commit()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "packages": package_versions(),
        "stream_seeds": streams.seeds(),
        "world": world.identity(),
        "wordforms": None,
    }


def _yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)


def _json_lines(records: Any) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def corpus_text(documents: list[dict[str, Any]], field: str) -> str:
    """Every document in one rendering: one sentence per line, and a blank line between
    documents. ``documents`` are the records of ``documents.jsonl``."""
    blocks = [
        "".join(sentence[field] + "\n" for sentence in document["sentences"])
        for document in documents
    ]
    return "\n".join(blocks)


def write_corpus(corpus: Corpus, path: str | Path | None = None) -> Path:
    """Write the output folder of a corpus, and return its path. The test sets of an earlier
    run in the same folder are replaced."""
    config = corpus.config
    folder = Path(path) if path is not None else default_output_dir(config)
    folder.mkdir(parents=True, exist_ok=True)

    config_data = config.resolved()
    config_data["provenance"] = provenance(config, corpus.planner.streams, corpus.planner.world)
    (folder / "config.yaml").write_text(_yaml(config_data), encoding="utf-8")
    corpus.planner.lexicon.frame().write_csv(folder / "lexicon.csv", null_value="")
    records = [document.to_json() for document in corpus.documents]
    (folder / "documents.jsonl").write_text(_json_lines(records), encoding="utf-8")
    for name, field in RENDERINGS.items():
        (folder / name).write_text(corpus_text(records, field), encoding="utf-8")
    (folder / REQUEST_FILE).write_text(_yaml(wordform_request(corpus)), encoding="utf-8")
    (folder / MEANINGS_FILE).write_text(meanings_csv(corpus.planner.world), encoding="utf-8")
    scenes = (scene.to_json() for scene in corpus.planner.scenes)
    (folder / "scenes.jsonl").write_text(_json_lines(scenes), encoding="utf-8")
    (folder / "stats.yaml").write_text(_yaml(corpus.stats), encoding="utf-8")

    tests = folder / TESTS_FOLDER
    tests.mkdir(exist_ok=True)
    for old in tests.glob("*.jsonl"):
        old.unlink()
    for test_set in corpus.test_sets:
        items = (item.to_json() for item in test_set.items)
        (tests / f"{test_set.name}.jsonl").write_text(_json_lines(items), encoding="utf-8")
    return folder

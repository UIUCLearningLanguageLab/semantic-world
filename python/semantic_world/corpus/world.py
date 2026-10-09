"""The world a corpus is about: defining the world run, and the corpus's view of it.

The corpus generator needs the world's live objects (the definition, the able tables, the
entities, the taxonomy's tree and rules for the fixed test), not only its output files. So the
world is always defined in memory, without its statistics episodes:

- from a world configuration file and a seed, with nothing saved on disk;
- or from a world run folder, regenerated from the folder's ``config.yaml``. The regenerated
  world is then checked against the folder's files (the rule-set identity, ``entities.csv``,
  and the derived manifest), and a difference is an error, because the corpus would otherwise
  describe another world than the files do. ``config.yaml`` itself is not compared: it records
  the git commit and the dirty flag.

**Labels.** In stage a5a the taxonomy still carries the old labels (``C1.3``, ``I1.3.2``,
``IS.4``, ``HAS.12``, ``SC.2``, ``CAN.3``, ``V1.2``), while the world package writes the labels
of ``docs/specs/WORLD_AND_LANGUAGE.md`` ("Labels"). This module is the one place in the corpus
where the two vocabularies meet: :class:`World` gives every other corpus module what it needs in
the new labels, and translates at the call for the two things that still run on the taxonomy's
own machinery, the fixed test and the rule terms. Stage a5b removes the translation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.common.boolean import minimal_dnf, settings_array
from semantic_world.corpus.config import Config, WorldSource
from semantic_world.corpus.errors import CorpusError
from semantic_world.taxonomy.fixed import fixed_by_rule
from semantic_world.taxonomy.io import result_frames
from semantic_world.taxonomy.rules import CONE_ENUMERATION_LIMIT, Threshold, evaluate_feature
from semantic_world.taxonomy.similarity import similarity_matrix
from semantic_world.taxonomy.tree import Category, Role
from semantic_world.world.definition import (
    DEFINITION_FILE,
    RuntimeDefinition,
    entities_frame,
    load_definition,
)
from semantic_world.world.derived import DERIVED_DIR, read_manifest
from semantic_world.world.episodes import Relatedness
from semantic_world.world.generate import WorldResult, define
from semantic_world.world.labels import translate, translate_column, untranslate
from semantic_world.world.runtime import able_table

THING = "THING"
"""The generic head noun's concept, and the subject of a rule statement."""
ONE_PLACE_PREFIX = "EVENTTYPE1."
TWO_PLACE_PREFIX = "EVENTTYPE2."
PATIENT_CAPACITY_PREFIX = "CANBE."
SCALAR_PREFIX = "SCALARDIM."
PROPERTY_PREFIX = "PROPERTY."
PART_PREFIX = "PART."
CATEGORY_PREFIX = "CATEGORY."
SCALAR_POLES = ("HIGH", "LOW")

EXACT = "exact"
"""The fixed test, by enumerating the cone: exact."""
LOCAL = "local"
"""The fixed test, by the local test of the taxonomy generator: never wrongly fixed, but it can
miss a fixed feature."""

MEANINGS_TABLE = "categories_generative.csv"
"""The taxonomy table that holds the categories' meaning vectors."""


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_world(config: Config | WorldSource) -> World:
    """Define the world that a corpus configuration names, in memory, and give the corpus's
    view of it. A run folder is checked against the regenerated world."""
    source = config.world if isinstance(config, Config) else config
    result = define(source.config, episode_stats=False)
    if source.kind == "run":
        check_run_folder(result, source.path)
    return World(result, source)


def check_run_folder(result: WorldResult, folder: str | Path) -> None:
    """Stop with a :class:`CorpusError` when a world run folder differs from ``result``: the
    rule-set identity of ``definition.json``, the bytes of ``entities.csv``, and the derived
    manifest must be those of the regenerated world."""
    folder = Path(folder)

    def problem(what: str) -> CorpusError:
        return CorpusError(
            f"the world run folder {folder} does not match the world regenerated from its "
            f"config.yaml: {what}. The folder was made by another version of the generator, or "
            f"was changed afterwards. Generate the folder again, or name the world "
            f"configuration file instead (world.config)."
        )

    try:
        definition = load_definition(folder)
    except Exception as error:  # noqa: BLE001 - any failure to load is a mismatch
        raise problem(f"{DEFINITION_FILE} cannot be loaded ({error})") from None
    if definition.rule_set_id != result.rule_set_id:
        raise problem(
            f"the rule set of {DEFINITION_FILE} is {definition.rule_set_id}, and the "
            f"regenerated world's is {result.rule_set_id}"
        )
    expected = entities_frame(result.definition).write_csv(None).encode("utf-8")
    entities = folder / "entities.csv"
    if not entities.is_file():
        raise problem("entities.csv is missing from the folder")
    if entities.read_bytes() != expected:
        raise problem("entities.csv differs from the regenerated file")
    manifest = {name: result.rule_set_id for name in result.derived_frames()}
    try:
        found = read_manifest(folder / DERIVED_DIR)
    except Exception as error:  # noqa: BLE001
        raise problem(f"the derived manifest cannot be read ({error})") from None
    if found != manifest:
        raise problem("the derived manifest differs from the regenerated world's")


def world_hash(config: Any) -> str:
    """The world's configuration hash: SHA-256 of its resolved configuration as YAML. The seed
    and the resolved taxonomy configuration are part of it."""
    text = yaml.safe_dump(config.resolved(), sort_keys=False, default_flow_style=None)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def world_identity(source: WorldSource, rule_set_id: str) -> dict[str, Any]:
    """The world run's identity, for the corpus run's ``config.yaml``."""
    return {
        "source": source.kind,
        "path": source.path,
        "name": source.config.name,
        "seed": source.seed,
        "config_hash": world_hash(source.config),
        "rule_set_id": rule_set_id,
    }


# ---------------------------------------------------------------------------------------------
# The corpus's view of a world
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryInfo:
    label: str
    level: int
    parent: str | None
    leaf: bool
    path: tuple[str, ...]
    """The categories from the top down to this one, this one included."""
    children: tuple[str, ...]

    @property
    def ancestors(self) -> tuple[str, ...]:
        """The strict ancestors, from the top."""
        return self.path[:-1]


@dataclass(frozen=True)
class EventTypeInfo:
    label: str
    arity: int
    level: int
    parent: str | None
    category: bool
    """True for a category of two-place event types, which has no events of its own."""
    path: tuple[str, ...]
    """The event types from the top of the tree down to this one, this one included."""


@dataclass(frozen=True)
class RuleTerm:
    """One term of the minimal DNF of a rule: a sufficient condition for its output."""

    output: str
    kind: str
    """``is``, ``has``, or ``can``: the kind of the output's predicate."""
    number: int
    """The number of the term, from 1."""
    literals: tuple[tuple[str, bool], ...]
    """The feature labels and their values, in input order."""
    threshold: bool
    """Whether the term reads a scalar threshold, which no pole adjective states."""


class World:
    """The corpus's view of a world run, in the labels of the specification."""

    def __init__(self, result: WorldResult, source: WorldSource | None = None) -> None:
        self.result = result
        self.source = source
        taxonomy = result.taxonomy
        self._taxonomy = taxonomy
        self.definition: RuntimeDefinition = result.definition.runtime()
        self.rule_set_id = result.rule_set_id

        # Categories and entities.
        self.categories: tuple[str, ...] = tuple(
            translate(c.label) for c in taxonomy.tree.categories
        )
        infos: dict[str, CategoryInfo] = {}
        children: dict[str, list[str]] = {label: [] for label in self.categories}
        for category in taxonomy.tree.categories:
            label = translate(category.label)
            parent = None if category.parent is None else translate(category.parent.label)
            if parent is not None:
                children[parent].append(label)
            path = tuple(translate(a.label) for a in reversed(category.ancestors())) + (label,)
            infos[label] = CategoryInfo(label, category.level, parent, category.is_leaf, path, ())
        self.category: dict[str, CategoryInfo] = {
            label: dataclasses.replace(info, children=tuple(children[label]))
            for label, info in infos.items()
        }
        self.leaves: tuple[str, ...] = tuple(c for c in self.categories if self.category[c].leaf)
        self.depth = taxonomy.tree.depth
        self.instances: tuple[str, ...] = tuple(self.definition.entity_labels)
        self.instance_index = {label: i for i, label in enumerate(self.instances)}
        self.instance_leaf: tuple[str, ...] = tuple(self.definition.leaves)
        self.paths: tuple[tuple[str, ...], ...] = tuple(
            self.category[leaf].path for leaf in self.instance_leaf
        )
        self.count = len(self.instances)
        every = np.arange(self.count)
        self._below: dict[str, np.ndarray] = {THING: every}
        for label in self.categories:
            self._below[label] = np.flatnonzero(
                np.array([label in path for path in self.paths], dtype=bool)
            )

        # Static features, scalars, and poles. The one-place event types stand in the feature
        # table too, as the taxonomy's CAN features: their column is the capacity to be their
        # agent, which the able table also gives.
        features = taxonomy.features
        self.features: dict[str, tuple[str, ...]] = {
            kind: tuple(translate(f.label) for f in features.of_type(kind))
            for kind in ("is", "has", "can")
        }
        self.feature_kind: dict[str, str] = {
            label: kind for kind, labels in self.features.items() for label in labels
        }
        self.free: frozenset[str] = frozenset(
            translate(f.label) for f in features.features if f.free
        )
        self.position: dict[str, int] = {translate(f.label): f.position for f in features.features}
        self.values: np.ndarray = taxonomy.instances.values
        """Every instance's static features, one-place capacities included, by ``position``."""
        self.scalars: tuple[str, ...] = tuple(translate(s) for s in features.scalar_labels)
        self.scalar_values: np.ndarray = taxonomy.instances.scalars
        self.poles: tuple[str, ...] = tuple(
            f"{scalar}.{pole}" for scalar in self.scalars for pole in SCALAR_POLES
        )

        # Event types: the one-place ones, and the two-place tree.
        self.event_types: dict[str, EventTypeInfo] = {}
        for et in result.event_types.event_types:
            path = (et.label,)
            parent = et.parent
            while parent is not None:
                path = (parent,) + path
                parent = result.event_types.event_type(parent).parent
            self.event_types[et.label] = EventTypeInfo(
                et.label, et.arity, et.level, et.parent, et.category, path
            )
        self.unary: tuple[str, ...] = tuple(
            label for label, et in self.event_types.items() if et.arity == 1
        )
        self.binary: tuple[str, ...] = tuple(
            label for label, et in self.event_types.items() if et.arity == 2
        )
        """The two-place event types in tree order, categories included."""
        self.binary_leaves: tuple[str, ...] = tuple(
            label for label in self.binary if not self.event_types[label].category
        )
        self.event_depth: int | None = max(
            (et.level for et in self.event_types.values() if et.arity == 2), default=None
        )
        self._able: dict[str, np.ndarray] | None = None

        # Patient capacities: every two-place leaf event type's, from the taxonomy's patient
        # projections (intensional: some possible agent).
        self.patient_capacities: tuple[str, ...] = ()
        self._capacity: dict[str, np.ndarray] = {}
        projections = taxonomy.projections
        if projections is not None:
            for i, verb in enumerate(projections.verb_labels):
                label = f"{PATIENT_CAPACITY_PREFIX}{translate(verb)}"
                self._capacity[label] = projections.patient[:, i].astype(bool)
            self.patient_capacities = tuple(self._capacity)

        # Relatedness: thematic relatedness and taxonomic similarity over the leaves.
        self.relatedness = Relatedness.from_taxonomy(taxonomy, self.definition)
        assert self.relatedness.leaves == self.leaves
        self.entity_leaf: np.ndarray = self.relatedness.entity_leaf
        """Each entity's leaf, as an index into ``leaves``."""
        self.thematic: np.ndarray = self.relatedness.thematic
        leaf_rows = np.array(
            [taxonomy.tree.categories.index(leaf) for leaf in taxonomy.tree.leaves], dtype=np.intp
        )
        analysis = taxonomy.config.analysis
        start = 0 if analysis.similarity_features == "all" else taxonomy.vectors.isa_count
        self.leaf_similarity: np.ndarray = similarity_matrix(
            taxonomy.vectors.generative[leaf_rows][:, start:], analysis.similarity_metric
        )
        """The taxonomic similarity of every pair of leaves, NaN where undefined."""

        self.initial_rates: dict[str, float] = {
            f.label: float(f.initial_rate) for f in result.fluents.base
        }
        self._rules_by_output = {r.output.label: r for r in taxonomy.rules.rules}
        self._free_index = {f.label: i for i, f in enumerate(features.free)}
        self._categories_old = {c.label: c for c in taxonomy.tree.categories}

    # Sets of instances -----------------------------------------------------------------------

    def below(self, category: str) -> np.ndarray:
        """The indices of the instances below a category (every instance for ``THING``)."""
        return self._below[category]

    def column(self, feature: str) -> np.ndarray:
        """A static feature's values over every instance (a one-place event type's column is
        its agent capacity)."""
        return self.values[:, self.position[feature]]

    def scalar_column(self, scalar: str) -> np.ndarray:
        return self.scalar_values[:, self.scalars.index(scalar)]

    def is_pole(self, label: str) -> bool:
        scalar, _, side = label.rpartition(".")
        return side in SCALAR_POLES and scalar in self.scalars

    def pole_parts(self, pole: str) -> tuple[str, str]:
        scalar, _, side = pole.rpartition(".")
        return scalar, side

    # Event types -----------------------------------------------------------------------------

    def able(self, event_type: str) -> np.ndarray:
        """An event type's requirement over the entities: a bool vector for a one-place event
        type, and a bool matrix over ordered pairs (agent, patient) for a two-place event type
        or a category of them (its base relation), with a false diagonal."""
        if self._able is None:
            self._able = able_table(self.definition)
        return self._able[event_type]

    def event_names(self, event_type: str) -> tuple[str, ...]:
        """The labels that can name an event of an event type: the event type, then the
        categories above it, from the nearest. A one-place event type has only its own label."""
        return tuple(reversed(self.event_types[event_type].path))

    def event_types_below(self, label: str) -> tuple[str, ...]:
        """The event types that have events and are named by a label: the label itself when it
        is an event type, or the leaves below it when it is a category."""
        if not self.event_types[label].category:
            return (label,)
        return tuple(leaf for leaf in self.binary_leaves if label in self.event_types[leaf].path)

    def capacity(self, label: str) -> np.ndarray:
        """A patient capacity (``CANBE.<event type>``) over every instance."""
        return self._capacity[label]

    # The fixed test --------------------------------------------------------------------------

    def fixed(
        self,
        category: str,
        restriction: tuple[tuple[str, bool], ...],
        feature: str,
        cone_limit: int = CONE_ENUMERATION_LIMIT,
    ) -> tuple[int | None, str]:
        """Whether a PROPERTY, PART, or one-place capacity is fixed for a category restricted
        by feature literals, and by which test: ``(1, test)`` or ``(0, test)`` when every
        possible member has that value, and ``(None, test)`` otherwise.

        The test is the fixed-by-rule test of the taxonomy generator, with the restriction's
        literals added as fixed. The free features that are defining at the category, and the
        restriction's literals on free features, are held. The other free features in the cone
        are enumerated, with the intervals between the thresholds of every scalar that is free
        to vary. A restriction's literal on a determined feature keeps only the settings that
        satisfy it. Above ``2 ** cone_limit`` settings, the local test is used instead.
        """
        taxonomy = self._taxonomy
        target = taxonomy.features[untranslate(feature)]
        node = None if category == THING else self._categories_old[untranslate(category)]
        n_free = len(taxonomy.features.free)
        if node is None:
            base = np.zeros(n_free, dtype=np.uint8)
            held = np.zeros(n_free, dtype=bool)
        else:
            base = node.free_values.astype(np.uint8).copy()
            held = node.defining_mask().copy()
        filters = []
        for label, positive in restriction:
            restricted = taxonomy.features[untranslate(label)]
            if restricted.free:
                index = self._free_index[restricted.label]
                if held[index] and base[index] != int(positive):
                    return None, EXACT  # no member can satisfy the restriction
                base[index] = int(positive)
                held[index] = True
            else:
                filters.append((restricted, int(positive)))

        rules = taxonomy.rules
        targets = [target] + [f for f, _ in filters]
        cone = sorted({self._free_index[f.label] for t in targets for f in rules.cone(t)})
        open_features = [i for i in cone if not held[i]]
        thresholds = {t.key: t for t in targets for t in rules.thresholds_of(t)}
        scalar_config = taxonomy.config.scalars
        scalars_fixed = node is not None and scalar_config.fixed_below(node.level)
        literal_values: dict[str, np.ndarray] = {}
        groups: dict[int, list[Threshold]] = {}
        for threshold in thresholds.values():
            if not scalars_fixed:
                groups.setdefault(threshold.scalar, []).append(threshold)
        open_groups = [sorted(groups[s], key=lambda t: t.threshold) for s in sorted(groups)]
        rows = 2 ** len(open_features)
        for group in open_groups:
            rows *= len(group) + 1
        if rows > 2**cone_limit:
            return self._fixed_local(node, base, held, target, cone_limit)

        grid = settings_array(len(open_features))
        if open_groups:
            intervals = np.array(
                list(itertools.product(*[range(len(g) + 1) for g in open_groups])), dtype=np.intp
            )
        else:
            intervals = np.zeros((1, 0), dtype=np.intp)
        binary_index = np.repeat(np.arange(grid.shape[0]), intervals.shape[0])
        interval_index = np.tile(np.arange(intervals.shape[0]), grid.shape[0])
        free_values = np.tile(base, (rows, 1))
        free_values[:, open_features] = grid[binary_index]
        if scalars_fixed:
            assert node is not None
            for key, threshold in thresholds.items():
                value = int(node.scalars[threshold.scalar - 1] > threshold.threshold)
                literal_values[key] = np.full(rows, value, dtype=np.uint8)
        for g, group in enumerate(open_groups):
            chosen = intervals[interval_index, g]
            for j, threshold in enumerate(group):
                # a scalar value in interval i makes the first i literals true
                literal_values[threshold.key] = (chosen > j).astype(np.uint8)

        cache: dict[str, np.ndarray] = {}

        def column(item) -> np.ndarray:
            return evaluate_feature(
                item, free_values, taxonomy.features, self._rules_by_output, cache, literal_values
            )

        keep = np.ones(rows, dtype=bool)
        for restricted, value in filters:
            keep &= column(restricted) == value
        outputs = column(target)[keep]
        if len(outputs) and outputs.min() == outputs.max():
            return int(outputs[0]), EXACT
        return None, EXACT

    def _fixed_local(
        self,
        category: Category | None,
        base: np.ndarray,
        held: np.ndarray,
        feature,
        cone_limit: int,
    ) -> tuple[int | None, str]:
        """The local test, for a cone too large to enumerate: the taxonomy generator's test on a
        copy of the category in which the held features are defining. Literals on determined
        features are left out, which can only miss a fixed feature."""
        taxonomy = self._taxonomy
        if feature.free:
            index = self._free_index[feature.label]
            return (int(base[index]) if held[index] else None), LOCAL
        roles = np.where(held, Role.DEFINING_NEW, Role.UNDIAGNOSTIC).astype(np.int8)
        scalars = np.zeros(taxonomy.features.scalar_count) if category is None else category.scalars
        values = taxonomy.rules.compute(base[None, :], scalars[None, :])[0]
        if category is None:
            stand_in = Category(THING, (), 0, None, base, values, roles, scalars=scalars)
            fixed, _ = fixed_by_rule(taxonomy.rules, stand_in, cone_limit, scalars=None)
        else:
            stand_in = dataclasses.replace(category, free_values=base, values=values, roles=roles)
            fixed, _ = fixed_by_rule(taxonomy.rules, stand_in, cone_limit, taxonomy.config.scalars)
        return (int(values[feature.position]) if fixed[feature.position] else None), LOCAL

    # Rules -----------------------------------------------------------------------------------

    def rule_terms(self) -> tuple[RuleTerm, ...]:
        """Every term of the minimal DNF of every feature rule, in rule order and then term
        order: the sufficient conditions of the determined PROPERTY and PART features and of the
        one-place capacities."""
        terms = []
        for rule in self._taxonomy.rules.rules:
            output = translate(rule.output.label)
            for number, term in enumerate(minimal_dnf(rule.table), start=1):
                used = [
                    (item, value)
                    for item, value in zip(rule.inputs, term, strict=True)
                    if value is not None
                ]
                threshold = any(isinstance(item, Threshold) for item, _ in used)
                literals = tuple(
                    (translate(item.label), bool(value))
                    for item, value in used
                    if not isinstance(item, Threshold)
                )
                terms.append(RuleTerm(output, rule.output.type, number, literals, threshold))
        return tuple(terms)

    @property
    def rule_count(self) -> int:
        return len(self._taxonomy.rules.rules)

    # Files -----------------------------------------------------------------------------------

    def meanings_csv(self) -> str:
        """The categories' meaning vectors, as the taxonomy generator writes them to
        ``categories_generative.csv``, with every label in the world's vocabulary."""
        frame = result_frames(self._taxonomy)[MEANINGS_TABLE]
        frame = frame.with_columns(
            pl.Series("label", [translate(v) for v in frame["label"].to_list()], dtype=pl.Utf8)
        )
        frame.columns = [c if c == "label" else translate_column(c) for c in frame.columns]
        return frame.write_csv(None, float_precision=6, null_value="")

    def identity(self) -> dict[str, Any] | None:
        """The world run's identity for ``config.yaml``, when the world came from a source."""
        if self.source is None:
            return None
        return world_identity(self.source, self.rule_set_id)


def leaf_of(world: World, instance: str) -> str:
    return world.instance_leaf[world.instance_index[instance]]


__all__ = [
    "EXACT",
    "LOCAL",
    "THING",
    "CategoryInfo",
    "EventTypeInfo",
    "RuleTerm",
    "World",
    "check_run_folder",
    "leaf_of",
    "load_world",
    "world_hash",
    "world_identity",
]

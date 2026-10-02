"""An independent recomputation of truth, from a taxonomy output folder.

The oracle reads the files a taxonomy run writes (``instances.csv``, ``tree.csv``, ``roles.csv``,
``categories_generative.csv``, ``features.csv``, ``rules.yaml``, ``relations.yaml``,
``projections.csv``, and ``config.yaml``) and judges a logical form given as JSON. It shares no
code with ``semantic_world.corpus``: subject sets are filtered rows of ``instances.csv``, the
fixed test is a brute-force enumeration over the rules' truth tables, and relations are the
expressions of ``relations.yaml`` evaluated over every pair of rows.

A relative clause in a category term is restrictive: it keeps the rows that have the CAN feature,
or that are related to at least one row of the other category. The oracle finds those rows with
plain loops over the pairs.

``truth`` returns True or False, or None for a logical form that cannot be judged: a vacuous
one (an empty subject set), or a quantifier that its predicate does not take.

Events are judged against scenes as ``scenes.jsonl`` holds them (``event``, ``happened``), and
``allows`` says whether the world allows an event. A verb category names the verbs below it, by
``verb_tree.csv``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.taxonomy.expressions import Cmp, Gt, Var, atom_key, parse_expression

THING = "THING"


class Oracle:
    def __init__(
        self,
        folder: Path,
        *,
        z: float = 1.0,
        most: float = 0.7,
        all_grounding: str = "fixed",
        generic: str = "most",
    ) -> None:
        self.z, self.most, self.all_grounding, self.generic = z, most, all_grounding, generic
        instances = pl.read_csv(folder / "instances.csv", infer_schema_length=None)
        self.labels = instances["label"].to_list()
        self.row = {label: i for i, label in enumerate(self.labels)}
        self.n = len(self.labels)
        self.column = {name: instances[name].to_numpy() for name in instances.columns[2:]}
        if (folder / "projections.csv").exists():
            projections = pl.read_csv(folder / "projections.csv", infer_schema_length=None)
            for name in projections.columns:
                if name.startswith("CANBE."):
                    self.column[name] = projections[name].to_numpy()
        tree = pl.read_csv(folder / "tree.csv", infer_schema_length=None)
        self.parent = dict(zip(tree["label"].to_list(), tree["parent"].to_list(), strict=True))
        self.level = dict(zip(tree["label"].to_list(), tree["level"].to_list(), strict=True))
        features = pl.read_csv(folder / "features.csv", infer_schema_length=None)
        self.kind = dict(zip(features["label"].to_list(), features["kind"].to_list(), strict=True))
        roles = pl.read_csv(folder / "roles.csv", infer_schema_length=None)
        self.defining: dict[str, list[str]] = {c: [] for c in self.parent}
        for category, feature, role in zip(
            roles["category"].to_list(),
            roles["feature"].to_list(),
            roles["role"].to_list(),
            strict=True,
        ):
            if role.startswith("defining"):
                self.defining[category].append(feature)
        generative = pl.read_csv(folder / "categories_generative.csv", infer_schema_length=None)
        self.generative = {row["label"]: row for row in generative.iter_rows(named=True)}
        rules = yaml.safe_load((folder / "rules.yaml").read_text(encoding="utf-8")) or []
        self.rules = {rule["output"]: rule for rule in rules}
        config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        self.drift = config["scalars"]["drift"]["values"]
        self.instance_drift = config["scalars"]["instance_drift"]
        self.relations: dict[str, str] = {}
        if (folder / "relations.yaml").exists():
            records = yaml.safe_load((folder / "relations.yaml").read_text(encoding="utf-8"))
            self.relations = {record["label"]: record["expression"] for record in records}
        self._matrices: dict[str, np.ndarray] = {}
        self.verb_parent: dict[str, str | None] = {}
        if (folder / "verb_tree.csv").exists():
            verbs = pl.read_csv(folder / "verb_tree.csv", infer_schema_length=None)
            self.verb_parent = dict(
                zip(verbs["label"].to_list(), verbs["parent"].to_list(), strict=True)
            )

    # Sets of rows ----------------------------------------------------------------------------

    def below(self, category: str) -> np.ndarray:
        if category == THING:
            return np.ones(self.n, dtype=bool)
        return self.column[f"ISA.{category}"] == 1

    def pole(self, pole: str, comparison: np.ndarray) -> np.ndarray:
        scalar, side = pole.rsplit(".", 1)
        values = self.column[scalar].astype(float)
        mean, sd = values[comparison].mean(), values[comparison].std()
        if sd == 0:
            return np.zeros(self.n, dtype=bool)
        return values >= mean + self.z * sd if side == "HIGH" else values <= mean - self.z * sd

    def subject_set(self, term: dict[str, Any]) -> np.ndarray:
        below = self.below(term["category"])
        keep = below.copy()
        for literal in term["restriction"]:
            positive = not literal.startswith("not ")
            name = literal.removeprefix("not ")
            if name.startswith("SC."):
                keep &= self.pole(name, below)
            else:
                keep &= self.column[name] == int(positive)
        for clause in term.get("clauses", ()):
            if clause["kind"] == "can":
                keep &= self.column[clause["feature"]] == 1
                continue
            holds = self.matrix(clause["verb"])
            as_agent = "patient" in clause
            others = np.flatnonzero(self.subject_set(clause["patient" if as_agent else "agent"]))
            for row in np.flatnonzero(keep):
                related = any(
                    holds[row, other] if as_agent else holds[other, row]
                    for other in others
                    if other != row
                )
                if not related:
                    keep[row] = False
        return keep

    def ancestors(self, category: str) -> list[str]:
        found = []
        parent = self.parent[category]
        while parent is not None:
            found.append(parent)
            parent = self.parent[parent]
        return found

    # The fixed test, by brute force ----------------------------------------------------------

    def _inputs(self, feature: str) -> list[tuple[str, float | None]]:
        """A rule's inputs: feature labels, and scalars with their thresholds."""
        rule = self.rules[feature]
        thresholds = rule.get("thresholds", {})
        return [(name, thresholds.get(name)) for name in rule["inputs"]]

    def _cone(self, feature: str, free: set[str], thresholds: set[tuple[str, float]]) -> None:
        if self.kind[feature] == "free":
            free.add(feature)
            return
        for name, threshold in self._inputs(feature):
            if threshold is not None:
                thresholds.add((name, threshold))
            else:
                self._cone(name, free, thresholds)

    def _value(self, feature: str, setting: dict[str, int], literals: dict) -> int:
        if feature in setting:
            return setting[feature]
        index = 0
        for name, threshold in self._inputs(feature):
            if threshold is not None:
                bit = literals[(name, threshold)]
            else:
                bit = self._value(name, setting, literals)
            index = 2 * index + bit
        return int(self.rules[feature]["truth_table"][index])

    def fixed(self, term: dict[str, Any], feature: str) -> int | None:
        """The value that every possible member of a category term has on a feature, or None
        when the feature varies. Every setting of the free features that are not held, and every
        position of every scalar among its thresholds, is tried."""
        category = term["category"]
        held: dict[str, int] = {}
        if category != THING:
            held = {f: int(self.generative[category][f]) for f in self.defining[category]}
        filters: list[tuple[str, int]] = []
        for literal in term["restriction"]:
            value = int(not literal.startswith("not "))
            name = literal.removeprefix("not ")
            if name.startswith("SC."):
                continue  # a pole fixes no binary feature
            if self.kind[name] == "free":
                if held.get(name, value) != value:
                    return None
                held[name] = value
            else:
                filters.append((name, value))
        free: set[str] = set()
        thresholds: set[tuple[str, float]] = set()
        for name in [feature] + [f for f, _ in filters]:
            self._cone(name, free, thresholds)
        open_features = sorted(free - set(held))
        scalars_fixed = (
            category != THING
            and all(d == 0 for d in self.drift[self.level[category] - 1 :])
            and self.instance_drift == 0
        )
        groups: dict[str, list[float]] = {}
        for scalar, threshold in thresholds:
            groups.setdefault(scalar, []).append(threshold)
        scalars = sorted(groups)
        if scalars_fixed:
            positions = [
                [sum(float(self.generative[category][s]) > t for t in groups[s])] for s in scalars
            ]
        else:
            positions = [list(range(len(groups[s]) + 1)) for s in scalars]
        seen: set[int] = set()
        for bits in itertools.product((0, 1), repeat=len(open_features)):
            setting = {**held, **dict(zip(open_features, bits, strict=True))}
            for chosen in itertools.product(*positions):
                literals = {}
                for scalar, position in zip(scalars, chosen, strict=True):
                    # a value above `position` of the sorted thresholds makes those literals true
                    for rank, threshold in enumerate(sorted(groups[scalar])):
                        literals[(scalar, threshold)] = int(rank < position)
                memo = dict(setting)
                if all(self._value(f, memo, literals) == v for f, v in filters):
                    seen.add(self._value(feature, memo, literals))
        return seen.pop() if len(seen) == 1 else None

    # Relations -------------------------------------------------------------------------------

    def matrix(self, verb: str) -> np.ndarray:
        """A relation over every ordered pair of rows (agent, patient), false on the diagonal."""
        if verb not in self._matrices:
            expression = parse_expression(self.relations[verb])
            env: dict[str, np.ndarray] = {}

            def side(name: str) -> np.ndarray:
                role, label = name.split(".", 1)
                values = self.column[label].astype(float)
                return values[:, None] if role == "a" else values[None, :]

            for atom in expression.atoms():
                if isinstance(atom, Var):
                    value = side(atom.name) == 1
                elif isinstance(atom, Gt):
                    value = side(atom.scalar) > atom.threshold
                else:
                    assert isinstance(atom, Cmp)
                    difference = side(atom.agent) - side(atom.patient)
                    value = difference > atom.low
                    if atom.high is not None:
                        value = value & (difference < atom.high)
                env[atom_key(atom)] = np.broadcast_to(value, (self.n, self.n))
            holds = np.broadcast_to(expression.evaluate(env), (self.n, self.n)).copy()
            np.fill_diagonal(holds, False)
            self._matrices[verb] = holds
        return self._matrices[verb]

    # Events ----------------------------------------------------------------------------------

    def names(self, verb: str) -> list[str]:
        """The labels that name an event of a verb: the verb, and the verb categories above."""
        found = [verb]
        while self.verb_parent.get(found[-1]) is not None:
            found.append(self.verb_parent[found[-1]])
        return found

    def allows(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether the world allows an event: the agent has the CAN feature, or the relation
        holds for the agent and the patient."""
        if patient is None:
            return bool(self.column[label][self.row[agent]] == 1)
        return bool(self.matrix(label)[self.row[agent], self.row[patient]])

    def matching(self, form: dict[str, Any], scene: dict[str, Any], aspect: bool) -> list[dict]:
        """The events of a scene, as ``scenes.jsonl`` holds it, that an event-level form could
        report: the same agent and patient, and a verb that the form's label names. With
        ``aspect``, the same aspect too."""
        predicate = form["predicate"]
        label = predicate["verb"] if predicate["kind"] == "verb" else predicate["feature"]
        patient = predicate["patient"]["instance"] if "patient" in predicate else None
        return [
            event
            for step in scene["steps"]
            for event in step
            if event["agent"] == form["subject"]["instance"]
            and event["patient"] == patient
            and label in self.names(event["verb"])
            and (not aspect or event["aspect"] == form["aspect"])
        ]

    def event(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether an event-level form that names no event is true of its scene: such an event,
        of the form's aspect, is among the scene's events."""
        assert form["scene"] == scene["label"]
        return bool(self.matching(form, scene, aspect=True))

    def happened(self, form: dict[str, Any], scenes: list[dict[str, Any]]) -> bool:
        """Whether some event of the scenes has the form's verb, agent, and patient, in either
        aspect."""
        return any(self.matching(form, scene, aspect=False) for scene in scenes)

    # Truth -----------------------------------------------------------------------------------

    def truth(self, form: dict[str, Any]) -> bool | None:
        if form["level"] == "class":
            return self._class(form)
        return self._instance(form)

    def all_holds(self, form: dict[str, Any], value: int) -> bool | None:
        """Whether every member of the subject set has ``value`` on the predicate: the reading
        of ``all`` (1) and ``no`` (0) under the configured grounding."""
        predicate = form["predicate"]
        count, total = self._counts(form)
        by_law = self.all_grounding == "fixed" and not form["subject"].get("clauses")
        if predicate["kind"] in ("is", "has", "can") and by_law:
            return self.fixed(form["subject"], predicate["feature"]) == value
        return count == (total if value else 0)

    def _counts(self, form: dict[str, Any]) -> tuple[int, int]:
        """How many members of the subject set (or pairs) satisfy the predicate, and how many
        there are."""
        predicate = form["predicate"]
        members = self.subject_set(form["subject"])
        kind = predicate["kind"]
        if kind == "verb":
            patients = self.subject_set(predicate["patient"])
            block = self.matrix(predicate["verb"])[np.ix_(members, patients)]
            return int(block.sum()), int(
                members.sum() * patients.sum() - (members & patients).sum()
            )
        name = predicate["projection"] if kind == "projection" else predicate["feature"]
        return int(self.column[name][members].sum()), int(members.sum())

    def _class(self, form: dict[str, Any]) -> bool | None:
        subject, predicate = form["subject"], form["predicate"]
        quantifier, polarity, kind = form["quantifier"], form["polarity"], predicate["kind"]
        if quantifier in ("all", "no") and not polarity:
            return None
        members = self.subject_set(subject)
        if not members.any():
            return None
        if subject.get("clauses") and quantifier in ("all", "no") and self.all_grounding == "fixed":
            return None  # a subject with a relative clause takes all and no only when observed
        category = subject["category"]
        if kind == "scalar":
            if quantifier != "generic" or category == THING:
                return None
            scalar, side = predicate["pole"].rsplit(".", 1)
            values = self.column[scalar].astype(float)
            parent = self.parent[category]
            comparison = self.below(THING if parent is None else parent)
            mean, sd = values[comparison].mean(), values[comparison].std()
            value = values[members].mean()
            if sd == 0:
                has_pole = False
            elif side == "HIGH":
                has_pole = value >= mean + self.z * sd
            else:
                has_pole = value <= mean - self.z * sd
            return bool(has_pole) == polarity
        if kind == "member":
            other = predicate["category"]
            if quantifier not in ("all", "no", "generic") or category in (THING, other):
                return None
            if quantifier == "no" or not polarity:
                return other not in self.ancestors(category) and category not in self.ancestors(
                    other
                )
            return other in self.ancestors(category)
        if kind == "projection" and quantifier in ("all", "no") and self.all_grounding == "fixed":
            return None
        count, total = self._counts(form)
        if total == 0:
            return None
        asserted = count if polarity else total - count
        means = self.generic if quantifier == "generic" else quantifier
        if means == "no":
            return self.all_holds(form, 0)
        if means == "all":
            return self.all_holds(form, 1 if polarity else 0)
        if means == "most":
            return asserted / total >= self.most - 1e-9
        return asserted > 0

    def _instance(self, form: dict[str, Any]) -> bool | None:
        subject, predicate = form["subject"]["instance"], form["predicate"]
        row, kind = self.row[subject], predicate["kind"]
        leaf = self.labels[row].rsplit(".", 1)[0].replace("I", "C", 1)
        path = [leaf] + self.ancestors(leaf)
        if kind in ("is", "has", "can"):
            value = self.column[predicate["feature"]][row] == 1
        elif kind == "projection":
            value = self.column[predicate["projection"]][row] == 1
        elif kind == "member":
            value = predicate["category"] in path
        elif kind == "scalar":
            if predicate["class"] not in path:
                return None
            value = self.pole(predicate["pole"], self.below(predicate["class"]))[row]
        else:
            patient = predicate["patient"]["instance"]
            if patient == subject:
                return None
            value = self.matrix(predicate["verb"])[row, self.row[patient]]
        return bool(value) == form["polarity"]

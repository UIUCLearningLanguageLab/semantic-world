"""An independent recomputation of truth, from a world run folder.

The oracle reads the files a world run writes (``definition.json``, ``entities.csv``,
``derived/capacities.csv``, the taxonomy's ``tree.csv``, ``roles.csv``,
``categories_generative.csv``, and ``config.yaml`` under ``taxonomy/``) and judges a logical
form given as JSON. It shares no code with ``semantic_world.corpus``: subject sets are filtered
rows of ``entities.csv``, derived features and requirements are evaluated by the brute-force
evaluator of ``semantic_world.world.fixtures`` (truth-table lookups over the definition record,
never the runtime), the fixed test is a brute-force enumeration over the rules' truth tables,
and nothing translates a label.

A relative clause in a category term is restrictive: it keeps the rows that are able to be the
agent of the one-place event type, or that are related to at least one row of the other
category. The oracle finds those rows with plain loops over the pairs.

``truth`` returns True or False, or None for a logical form that cannot be judged: a vacuous
one (an empty subject set), or a quantifier that its predicate does not take.

Events are judged against scenes as ``scenes.jsonl`` holds them (histories): ``event`` and
``happened``; ``able`` says whether a binding's requirement holds, and ``legal`` whether it was
legal at some time point of a scene, by replaying the history with the brute-force evaluator. A
category of event types names the event types below it.

States, changes, and what was possible at a time point (stage a7a) are judged on the replayed
states too: ``state`` (the fluent's value at the time point), ``change`` (its value at the time
point and at the next), ``able_now`` (legality at the time point), ``changed`` (the value at
the scene's final time point against ``TIME.1``), and ``made_by`` (whether one event's own
recorded changes, applied alone to the state at the time point, change the fluent).

Causal statements (stage a7b) are judged by re-reading the ``event_types`` block of
``definition.json``: ``causal`` says whether every event type below the statement's event type
has the effect, or the precondition literal, that the statement names; ``observed`` says whether
the statement held after (or before) every event of its type in the given scenes, with at
least one such event, on the replayed states.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.world.fixtures import BruteEvent, BruteForce

THING = "THING"
NEC_ALL, ALL, MOST, SOME, NO, NEC_NO = "nec_all", "all", "most", "some", "no", "nec_no"


class Oracle:
    def __init__(self, folder: Path, *, z: float = 1.0) -> None:
        self.z = z
        folder = Path(folder)
        self.record = json.loads((folder / "definition.json").read_text(encoding="utf-8"))
        entities = pl.read_csv(folder / "entities.csv", infer_schema_length=None)
        self.rows = entities.to_dicts()
        self.brute = BruteForce(self.record, self.rows)
        self.labels = entities["label"].to_list()
        self.row = {label: i for i, label in enumerate(self.labels)}
        self.n = len(self.labels)
        self.symbols = {s["label"]: s for s in self.record["symbols"]}
        self.rules = {r["output"]: r for r in self.record["rules"]}
        self.literals = self.record["literals"]
        self.event_types = {e["label"]: e for e in self.record["event_types"]}
        self.base_fluents = [
            s["label"] for s in self.record["symbols"] if s["kind"] == "fluent" and not s["derived"]
        ]
        self.free = {
            s["label"]
            for s in self.record["symbols"]
            if s["kind"] in ("property", "part") and not s["derived"]
        }
        self.derived = {
            s["label"]
            for s in self.record["symbols"]
            if s["kind"] in ("property", "part") and s["derived"]
        }
        self.scalars = [s["label"] for s in self.record["symbols"] if s["kind"] == "scalar"]
        # Columns: free features and scalars from entities.csv; derived features and one-place
        # capacities by the brute-force evaluator; patient capacities from capacities.csv.
        self.column: dict[str, np.ndarray] = {}
        for name in entities.columns[2:]:
            self.column[name] = entities[name].to_numpy()
        for label in self.derived:
            self.column[label] = np.array(
                [self.brute.feature(label, e) for e in self.labels], dtype=np.int64
            )
        for label, et in self.event_types.items():
            if et["arity"] == 1:
                self.column[label] = np.array(
                    [int(self.brute.able(BruteEvent(label, {"agent": e}))) for e in self.labels],
                    dtype=np.int64,
                )
        capacities = pl.read_csv(folder / "derived" / "capacities.csv", infer_schema_length=None)
        assert capacities["label"].to_list() == self.labels
        for name in capacities.columns:
            if name.startswith("CANBE."):
                self.column[name] = capacities[name].to_numpy()
        # The tree, and the leaf of every entity.
        tree = pl.read_csv(folder / "taxonomy" / "tree.csv", infer_schema_length=None)
        self.parent = {
            label: parent
            for label, parent in zip(tree["label"].to_list(), tree["parent"].to_list(), strict=True)
        }
        self.level = {
            label: level
            for label, level in zip(tree["label"].to_list(), tree["level"].to_list(), strict=True)
        }
        self.leaf = entities["leaf"].to_list()
        self.path = {
            label: [leaf] + self.ancestors(leaf)
            for label, leaf in zip(self.labels, self.leaf, strict=True)
        }
        roles = pl.read_csv(folder / "taxonomy" / "roles.csv", infer_schema_length=None)
        self.defining: dict[str, list[str]] = {c: [] for c in self.parent}
        for category, feature, role in zip(
            roles["category"].to_list(),
            roles["feature"].to_list(),
            roles["role"].to_list(),
            strict=True,
        ):
            if role.startswith("defining"):
                self.defining[category].append(feature)
        generative = pl.read_csv(
            folder / "taxonomy" / "categories_generative.csv", infer_schema_length=None
        )
        self.generative = {
            row["label"]: dict(row.items()) for row in generative.iter_rows(named=True)
        }
        config = yaml.safe_load((folder / "taxonomy" / "config.yaml").read_text(encoding="utf-8"))
        self.drift = config["scalars"]["drift"]["values"]
        self.instance_drift = config["scalars"]["instance_drift"]
        self._matrices: dict[str, np.ndarray] = {}
        self._states: dict[str, list[dict[str, frozenset[str]]]] = {}

    # Sets of rows ----------------------------------------------------------------------------

    def below(self, category: str) -> np.ndarray:
        if category == THING:
            return np.ones(self.n, dtype=bool)
        return np.array([category in self.path[label] for label in self.labels], dtype=bool)

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
            if name.startswith("SCALARDIM."):
                keep &= self.pole(name, below)
            else:
                keep &= self.column[name] == int(positive)
        for clause in term.get("clauses", ()):
            if clause["kind"] == "event_type1":
                keep &= self.column[clause["label"]] == 1
                continue
            holds = self.matrix(clause["label"])
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

    def _rule_of(self, feature: str) -> dict[str, Any]:
        """The rule that computes a derived feature, or the requirement of a one-place event
        type (a binding rule over the agent's features)."""
        if feature in self.event_types:
            return self.rules[self.event_types[feature]["requirement"]["output"]]
        return self.rules[feature]

    def _inputs(self, feature: str) -> list[tuple]:
        """A rule's inputs: ``("feature", label)``, ``("threshold", scalar, threshold)``, or
        ``("constraint", output)``."""
        found = []
        for index in self._rule_of(feature)["inputs"]:
            literal = self.literals[index]
            if literal["kind"] == "feature":
                found.append(("feature", literal["feature"]))
            elif literal["kind"] == "threshold":
                found.append(("threshold", literal["scalar"], literal["threshold"]))
            elif literal["kind"] == "constraint":
                found.append(("constraint", literal["constraint"]))
            else:
                raise AssertionError(f"a static rule reads {literal['kind']}")
        return found

    def _cone(self, feature: str, free: set[str], thresholds: set[tuple[str, float]]) -> None:
        if feature in self.free:
            free.add(feature)
            return
        for item in self._inputs(feature):
            if item[0] == "feature":
                self._cone(item[1], free, thresholds)
            elif item[0] == "threshold":
                thresholds.add((item[1], item[2]))
            else:
                self._cone_rule(item[1], free, thresholds)

    def _cone_rule(self, output: str, free: set[str], thresholds: set[tuple[str, float]]) -> None:
        for index in self.rules[output]["inputs"]:
            literal = self.literals[index]
            if literal["kind"] == "feature":
                self._cone(literal["feature"], free, thresholds)
            elif literal["kind"] == "threshold":
                thresholds.add((literal["scalar"], literal["threshold"]))
            else:
                self._cone_rule(literal["constraint"], free, thresholds)

    def _value(self, feature: str, setting: dict[str, int], literals: dict) -> int:
        if feature in setting:
            return setting[feature]
        return self._rule_value(self._rule_of(feature), setting, literals)

    def _rule_value(self, rule: dict[str, Any], setting: dict[str, int], literals: dict) -> int:
        index = 0
        for i in rule["inputs"]:
            literal = self.literals[i]
            if literal["kind"] == "feature":
                bit = self._value(literal["feature"], setting, literals)
            elif literal["kind"] == "threshold":
                bit = literals[(literal["scalar"], literal["threshold"])]
            else:
                bit = self._rule_value(self.rules[literal["constraint"]], setting, literals)
            index = 2 * index + bit
        return int(rule["truth_table"][index])

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
            if name.startswith("SCALARDIM."):
                continue  # a pole fixes no binary feature
            if name in self.free:
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

    # Requirements ----------------------------------------------------------------------------

    def matrix(self, event_type: str) -> np.ndarray:
        """A two-place event type's requirement (a category's base relation) over every ordered
        pair of rows (agent, patient), false on the diagonal, by the brute-force evaluator."""
        if event_type not in self._matrices:
            holds = np.zeros((self.n, self.n), dtype=bool)
            for a, agent in enumerate(self.labels):
                for p, patient in enumerate(self.labels):
                    if a != p:
                        event = BruteEvent(event_type, {"agent": agent, "patient": patient})
                        holds[a, p] = self.brute.able(event)
            self._matrices[event_type] = holds
        return self._matrices[event_type]

    def able(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether a binding's requirement holds: the one-place event type's for the agent, or
        the two-place event type's (or category's) for the agent and the patient."""
        if patient is None:
            return bool(self.column[label][self.row[agent]] == 1)
        return bool(self.matrix(label)[self.row[agent], self.row[patient]])

    allows = able

    def able_somewhere(self, label: str) -> bool:
        """Whether the requirement of an event type (or the base relation of a category) holds
        for some entity, or some ordered pair of entities."""
        if self.event_types[label]["arity"] == 1:
            return bool((self.column[label] == 1).any())
        return bool(self.matrix(label).any())

    # Events ----------------------------------------------------------------------------------

    def names(self, event_type: str) -> list[str]:
        """The labels that name an event of an event type: the event type, and the categories
        above it."""
        found = [event_type]
        while self.event_types[found[-1]]["parent"] is not None:
            found.append(self.event_types[found[-1]]["parent"])
        return found

    def leaves_below(self, label: str) -> list[str]:
        if self.event_types[label]["kind"] == "event_type":
            return [label]
        return [
            e
            for e, record in self.event_types.items()
            if record["kind"] == "event_type" and label in self.names(e)
        ]

    def states(self, scene: dict[str, Any]) -> list[dict[str, frozenset[str]]]:
        """The state at every time point of a scene, replayed with the brute-force evaluator
        from the entities' initial fluents and the history's initial fluents."""
        if scene["label"] not in self._states:
            state = {
                label: frozenset(f for f in self.base_fluents if row[f])
                for label, row in zip(self.labels, self.rows, strict=True)
            }
            for participant, fluents in scene["initial"].items():
                state[participant] = frozenset(fluents)
            states = [state]
            for step in scene["steps"]:
                events = [
                    BruteEvent(
                        e["type"],
                        {
                            "agent": e["agent"],
                            **({"patient": e["patient"]} if "patient" in e else {}),
                        },
                    )
                    for e in step["events"]
                ]
                for event in events:
                    assert self.brute.legal(event, state), (scene["label"], step["step"], event)
                state = self.brute.apply(state, events)
                states.append(state)
            self._states[scene["label"]] = states
        return self._states[scene["label"]]

    def legal_at(
        self, scene: dict[str, Any], time: int, label: str, agent: str, patient: str | None
    ) -> bool:
        """Whether the binding was legal at ``TIME.<time>`` of the scene."""
        binding = {"agent": agent, **({"patient": patient} if patient is not None else {})}
        state = self.states(scene)[time - 1]
        return any(
            self.brute.legal(BruteEvent(event_type, binding), state)
            for event_type in self.leaves_below(label)
        )

    def fluent_at(self, scene: dict[str, Any], time: int, entity: str, fluent: str) -> bool:
        """A fluent's value, base or derived, of a participant at ``TIME.<time>``."""
        return bool(self.brute.fluent(fluent, entity, self.states(scene)[time - 1]))

    def changed(self, scene: dict[str, Any], entity: str, fluent: str) -> bool:
        """Whether the fluent's value at the scene's final time point differs from its value
        at ``TIME.1``."""
        final = int(scene["final"].split(".")[1])
        return self.fluent_at(scene, final, entity, fluent) != self.fluent_at(
            scene, 1, entity, fluent
        )

    def made_by(
        self, scene: dict[str, Any], time: int, entity: str, fluent: str, event_label: str
    ) -> bool:
        """Whether the event's own recorded changes, applied alone to the state at
        ``TIME.<time>``, change the fluent of the participant."""
        step = next(s for s in scene["steps"] if s["step"] == time)
        event = next((e for e in step["events"] if e["label"] == event_label), None)
        if event is None:
            return False
        if not self.symbols[fluent]["derived"]:
            return any(c["entity"] == entity and c["fluent"] == fluent for c in event["changes"])
        before = self.states(scene)[time - 1]
        after = {e: set(v) for e, v in before.items()}
        for change in event["changes"]:
            if change["to"]:
                after[change["entity"]].add(change["fluent"])
            else:
                after[change["entity"]].discard(change["fluent"])
        after_state = {e: frozenset(v) for e, v in after.items()}
        return self.brute.fluent(fluent, entity, before) != self.brute.fluent(
            fluent, entity, after_state
        )

    def state(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether a state form (``HOLDS``) is true of the scene."""
        time = int(form["time"].split(".")[1])
        value = self.fluent_at(scene, time, form["subject"]["instance"], form["predicate"]["label"])
        return value == form["polarity"]

    def change(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether a change form (``BECOME``) is true of the scene: the fluent had the other
        value at the time point and the stated value at the next."""
        time = int(form["time"].split(".")[1])
        entity, fluent = form["subject"]["instance"], form["predicate"]["label"]
        before = self.fluent_at(scene, time, entity, fluent)
        after = self.fluent_at(scene, time + 1, entity, fluent)
        return before != form["polarity"] and after == form["polarity"]

    def able_now(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether an able_now form is true of the scene: the binding was legal at the time
        point (positive) or was not (negative)."""
        time = int(form["time"].split(".")[1])
        predicate = form["predicate"]
        patient = predicate["patient"]["instance"] if "patient" in predicate else None
        legal_now = self.legal_at(
            scene, time, predicate["label"], form["subject"]["instance"], patient
        )
        return legal_now == form["polarity"]

    def legal(self, scene: dict[str, Any], label: str, agent: str, patient: str | None) -> bool:
        """Whether the binding was legal at some time point of the scene, for the event type or
        for some event type below the category."""
        binding = {"agent": agent, **({"patient": patient} if patient is not None else {})}
        return any(
            self.brute.legal(BruteEvent(event_type, binding), state)
            for event_type in self.leaves_below(label)
            for state in self.states(scene)
        )

    def matching(self, form: dict[str, Any], scene: dict[str, Any]) -> list[dict]:
        """The events of a scene, as ``scenes.jsonl`` holds it, that an event-level form could
        report: the same agent and patient, and an event type that the form's label names."""
        predicate = form["predicate"]
        label = predicate["label"]
        patient = predicate["patient"]["instance"] if "patient" in predicate else None
        return [
            event
            for step in scene["steps"]
            for event in step["events"]
            if event["agent"] == form["subject"]["instance"]
            and event.get("patient") == patient
            and label in self.names(event["type"])
        ]

    def event(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether an event-level form that names no event is true of its scene: such an event
        is among the scene's events, whatever the aspect."""
        assert form["scene"] == scene["label"]
        return bool(self.matching(form, scene))

    def happened(self, form: dict[str, Any], scenes: list[dict[str, Any]]) -> bool:
        """Whether some event of the scenes has the form's event type, agent, and patient."""
        return any(self.matching(form, scene) for scene in scenes)

    # Causal statements -----------------------------------------------------------------------

    def entries(self, event_type: str, kind: str) -> list[dict[str, Any]]:
        """The effects (``effect``) or the precondition literals (``precondition``) of an event
        type, as ``definition.json`` records them."""
        record = self.event_types[event_type]
        return record["effects"] if kind == "effect" else record["precondition"]["literals"]

    def causal(self, form: dict[str, Any]) -> bool | None:
        """Whether a causal statement is true: every event type below its event type (itself,
        for an event type) has the entry. None for a statement about a derived fluent, or about
        a role the event type lacks."""
        subject, predicate = form["subject"], form["predicate"]
        event_type, role = subject["event"], subject["role"]
        fluent, value = predicate["label"], predicate["value"]
        if form["quantifier"] != NEC_ALL or not form["polarity"]:
            return None
        if fluent not in self.base_fluents or role not in self.event_types[event_type]["roles"]:
            return None
        for label in self.leaves_below(event_type):
            if not any(
                e["role"] == role and e["fluent"] == fluent and e["value"] == value
                for e in self.entries(label, predicate["kind"])
            ):
                return False
        return True

    def observed(self, form: dict[str, Any], scenes: list[dict[str, Any]]) -> bool:
        """Whether a causal statement held of every event of its type in the scenes, and at
        least one occurred: the participant in its role had the fluent's value at the time
        point after the event's step (an effect) or at the event's time point (a
        precondition)."""
        subject, predicate = form["subject"], form["predicate"]
        below = set(self.leaves_below(subject["event"]))
        offset = 1 if predicate["kind"] == "effect" else 0
        seen = False
        for scene in scenes:
            for step in scene["steps"]:
                for event in step["events"]:
                    if event["type"] not in below:
                        continue
                    entity = event.get(subject["role"])
                    if entity is None:
                        continue
                    seen = True
                    value = self.fluent_at(scene, step["step"] + offset, entity, predicate["label"])
                    if value != predicate["value"]:
                        return False
        return seen

    # Truth -----------------------------------------------------------------------------------

    def truth(self, form: dict[str, Any], scene: dict[str, Any] | None = None) -> bool | None:
        if form["level"] == "class" and "head" in form["subject"]:
            return self.causal(form)
        if form["level"] == "class":
            return self._class(form)
        if form["level"] in ("state", "change", "able_now"):
            assert scene is not None and scene["label"] == form["scene"]
            judge = {"state": self.state, "change": self.change, "able_now": self.able_now}
            return judge[form["level"]](form, scene)
        return self._instance(form)

    def _counts(self, form: dict[str, Any]) -> tuple[int, int]:
        """How many members of the subject set (or pairs) satisfy the predicate, and how many
        there are."""
        predicate = form["predicate"]
        members = self.subject_set(form["subject"])
        kind = predicate["kind"]
        if kind == "event_type2":
            patients = self.subject_set(predicate["patient"])
            block = self.matrix(predicate["label"])[np.ix_(members, patients)]
            return int(block.sum()), int(
                members.sum() * patients.sum() - (members & patients).sum()
            )
        return int(self.column[predicate["label"]][members].sum()), int(members.sum())

    def _class(self, form: dict[str, Any]) -> bool | None:
        subject, predicate = form["subject"], form["predicate"]
        quantifier, polarity, kind = form["quantifier"], form["polarity"], predicate["kind"]
        category = subject["category"]
        if kind == "scalar":
            if quantifier is not None or category == THING:
                return None
            if subject["restriction"] or subject.get("clauses"):
                return None  # a statement about the category, not about a restricted set
            members = self.subject_set(subject)
            scalar, side = predicate["label"].rsplit(".", 1)
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
        if quantifier not in (NEC_ALL, ALL, MOST, SOME, NO, NEC_NO):
            return None
        if quantifier in (NEC_ALL, ALL, NO, NEC_NO) and not polarity:
            return None
        if quantifier in (NEC_ALL, NEC_NO):
            if kind not in ("property", "part", "event_type1", "member") or subject.get("clauses"):
                return None
        members = self.subject_set(subject)
        if not members.any():
            return None
        if kind == "member":
            other = predicate["label"]
            if quantifier in (MOST, SOME) or category in (THING, other):
                return None
            if quantifier in (NO, NEC_NO):
                return other not in self.ancestors(category) and category not in self.ancestors(
                    other
                )
            return other in self.ancestors(category)
        count, total = self._counts(form)
        if total == 0:
            return None
        asserted = count if polarity else total - count
        if quantifier == NEC_ALL:
            return self.fixed(subject, predicate["label"]) == 1
        if quantifier == NEC_NO:
            return self.fixed(subject, predicate["label"]) == 0
        if quantifier == ALL:
            return count == total
        if quantifier == NO:
            return count == 0
        if quantifier == MOST:
            return asserted > total / 2
        return asserted > 0

    def _instance(self, form: dict[str, Any]) -> bool | None:
        subject, predicate = form["subject"]["instance"], form["predicate"]
        row, kind = self.row[subject], predicate["kind"]
        path = self.path[subject]
        if kind in ("property", "part", "event_type1", "patient_capacity"):
            value = self.column[predicate["label"]][row] == 1
        elif kind == "member":
            value = predicate["label"] in path
        elif kind == "scalar":
            if predicate["class"] not in path:
                return None
            value = self.pole(predicate["label"], self.below(predicate["class"]))[row]
        else:
            patient = predicate["patient"]["instance"]
            if patient == subject:
                return None
            value = self.matrix(predicate["label"])[row, self.row[patient]]
        return bool(value) == form["polarity"]

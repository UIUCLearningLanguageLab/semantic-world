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
has the effect (an unconditional one: since stage b1 an effect with a ``condition`` key is no
entry, and since stage b2 an effect with an ``amount``, a numeric effect, is none either), or
the precondition literal, that the statement names; ``observed`` says whether the statement
held after (or before) every event of its type in the given scenes, with at least one such
event, on the replayed states.

Since stage b2 the replay carries the numeric fluents (``NUMFL`` columns of ``entities.csv``,
the history's ``initial_numeric``), so that a derived fluent that reads a numeric threshold is
judged on the right values; the brute-force evaluator applies the numeric effects and the
drift. Since stage b3b the corpus states them, and the oracle judges every such form on the
replayed values and the ``range`` of the fluent's symbol record: a numeric pole at a time point
(``ABOVE`` or ``BELOW`` against ``RANGE`` at the ``cut`` the form records), a numeric change
(``INCREASE`` or ``DECREASE``: the value at the next time point against the value at the time
point, whatever moved it), a numeric comparative at a time point, and the numeric effect
statements ``RAISES`` and ``LOWERS`` (an unconditional numeric effect of the sign on the role).
A conditional causal statement (stage b3b) is true when every event type below its event type
has the effect under exactly the condition the form's subject records, and ``observed`` counts
only the events whose participants satisfied the condition; a numeric statement is observed
when every event of its type recorded a change of the fluent in its direction. A numeric
condition literal of a form (``NUMFL.1.HIGH`` with its ``cut``, WM.35) matches the record's
numeric threshold literal ``> t`` (the high pole) or ``<= t`` (the low pole) whose threshold
is exactly the point of the range at the form's cut, and its ``observed`` judges the
participant's replayed value against that point as ``pole_at`` does.

Since stage c4a the replay carries the pair fluents and the places (``docs/specs/WORLD_PHASE_C.md``,
the c4a entry of "Build stages"): the pair part of the state is a set of ordered label pairs per
pair fluent, started from the history's ``initial_pairs`` (the whole pair part among the
participants, WM.38) and, for a location fluent, from the location column of ``entities.csv``
with the history's ``initial_places`` written over it, so that every entity is at one place; the
brute-force evaluator reads the places table (``places.csv``) for the place event types' bindings
and requirements, replays every pair effect (both directions of a symmetric pair) and every move
(the total functional replacement: the old place goes with the new one), judges legality with the
pairs (``SAME`` on a location fluent, ``SOME`` and ``SOMEOF``, the compiled literals), and reads a
derived fluent whose rule reads ``SOME`` or ``SOMEOF`` from the replayed pairs; the recorded pair
changes of every event are checked against the replayed pairs, as ``history.replay`` checks them.
``pairs_at``, ``place_at``, and ``pair_holds`` read the replayed pairs; ``able`` and ``legal``
take a place as the patient of a place event type. The corpus states nothing about them in stage
c4a: the readers are for the tests of this stage and of c4b.

Comparisons of scalars (stage b3a) are recomputed from the ``SCALARDIM`` columns of
``entities.csv`` alone: a pole (kind ``comparison``) against the mean and the population standard
deviation of its comparison class, the rows below the class's category or every row for
``ENTITY``, at the threshold the form records under ``z``; a class-level pole compares the mean
of the subject set the same way; a comparative (kind ``comparative``) compares two rows' values,
or two categories' means, strictly. The oracle shares no class statistics with the corpus or the
world package.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.world.fixtures import BruteEvent, BruteForce, BrutePairs

THING = "THING"
PLACE_EVENT_TYPE = "place_event_type"
EVENT_TYPE_KINDS = ("event_type", PLACE_EVENT_TYPE)
"""The kinds of event-type record that name events: an entity event type, and (stage c3a) a
place event type with dynamics."""
ENTITY = "ENTITY"
NEC_ALL, ALL, MOST, SOME, NO, NEC_NO = "nec_all", "all", "most", "some", "no", "nec_no"


class Oracle:
    def __init__(self, folder: Path, *, z: float = 1.0) -> None:
        self.z = z
        folder = Path(folder)
        self.record = json.loads((folder / "definition.json").read_text(encoding="utf-8"))
        entities = pl.read_csv(folder / "entities.csv", infer_schema_length=None)
        self.rows = entities.to_dicts()
        # the places (stage c1), for the place event types' bindings and requirements
        places_file = folder / "places.csv"
        self.place_rows = (
            pl.read_csv(places_file, infer_schema_length=None).to_dicts()
            if places_file.is_file()
            else []
        )
        self.place_labels = [str(row["label"]) for row in self.place_rows]
        self.brute = BruteForce(self.record, self.rows, self.place_rows)
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
        self.numeric_fluents = [
            s["label"] for s in self.record["symbols"] if s["kind"] == "numeric_fluent"
        ]
        self.ranges = {
            s["label"]: (float(s["range"][0]), float(s["range"][1]))
            for s in self.record["symbols"]
            if s["kind"] == "numeric_fluent"
        }
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
        # The pair fluents (stage c2a) and the location fluents among them (stage c3a), whose
        # initial places are the location columns of entities.csv, named by the fluent
        pair_symbols = [s for s in self.record["symbols"] if s["kind"] == "pair_fluent"]
        self.pair_fluents = [s["label"] for s in pair_symbols]
        self.symmetric = {s["label"] for s in pair_symbols if s["symmetry"] == "symmetric"}
        self.location_fluents = [s["label"] for s in pair_symbols if s.get("total")]
        self.place_event_types = [
            e["label"] for e in self.record["event_types"] if e["kind"] == PLACE_EVENT_TYPE
        ]
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
        self._place_matrices: dict[str, np.ndarray] = {}
        self._states: dict[str, list[dict[str, frozenset[str]]]] = {}
        self._numerics: dict[str, list[dict[str, dict[str, float]]]] = {}
        self._pairs: dict[str, list[BrutePairs]] = {}

    # Sets of rows ----------------------------------------------------------------------------

    def below(self, category: str) -> np.ndarray:
        if category in (THING, ENTITY):
            return np.ones(self.n, dtype=bool)
        return np.array([category in self.path[label] for label in self.labels], dtype=bool)

    def pole(self, pole: str, comparison: np.ndarray, z: float | None = None) -> np.ndarray:
        """Which rows are at a pole against a comparison class (a mask of rows), at ``z``
        standard deviations of the class (the oracle's default when none is given): ``HIGH`` at
        least ``z`` above the class's mean, ``LOW`` at least ``z`` below; none when the class
        has no spread."""
        z = self.z if z is None else z
        scalar, side = pole.rsplit(".", 1)
        values = self.column[scalar].astype(float)
        mean, sd = values[comparison].mean(), values[comparison].std()
        if sd == 0:
            return np.zeros(self.n, dtype=bool)
        return values >= mean + z * sd if side == "HIGH" else values <= mean - z * sd

    def subject_set(self, term: dict[str, Any], z: float | None = None) -> np.ndarray:
        below = self.below(term["category"])
        keep = below.copy()
        for literal in term["restriction"]:
            positive = not literal.startswith("not ")
            name = literal.removeprefix("not ")
            if name.startswith("SCALARDIM."):
                keep &= self.pole(name, below, z)
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

    def place_matrix(self, event_type: str) -> np.ndarray:
        """A place event type's requirement (stage c1) over every binding of an entity (rows)
        and a place (columns), by the brute-force evaluator."""
        if event_type not in self._place_matrices:
            holds = np.zeros((self.n, len(self.place_labels)), dtype=bool)
            for a, agent in enumerate(self.labels):
                for p, place in enumerate(self.place_labels):
                    event = BruteEvent(event_type, {"agent": agent, "patient": place})
                    holds[a, p] = self.brute.able(event)
            self._place_matrices[event_type] = holds
        return self._place_matrices[event_type]

    def is_place_event_type(self, label: str) -> bool:
        return label in self.place_event_types

    def able(self, label: str, agent: str, patient: str | None) -> bool:
        """Whether a binding's requirement holds: the one-place event type's for the agent, the
        two-place event type's (or category's) for the agent and the patient, or (stage c3a) the
        place event type's for the agent and the place."""
        if patient is None:
            return bool(self.column[label][self.row[agent]] == 1)
        if self.is_place_event_type(label):
            return bool(self.place_matrix(label)[self.row[agent], self.place_labels.index(patient)])
        return bool(self.matrix(label)[self.row[agent], self.row[patient]])

    allows = able

    def able_somewhere(self, label: str) -> bool:
        """Whether the requirement of an event type (or the base relation of a category) holds
        for some entity, some ordered pair of entities, or some entity at some place."""
        if self.event_types[label]["arity"] == 1:
            return bool((self.column[label] == 1).any())
        if self.is_place_event_type(label):
            return bool(self.place_matrix(label).any())
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
        if self.event_types[label]["kind"] in EVENT_TYPE_KINDS:
            return [label]
        return [
            e
            for e, record in self.event_types.items()
            if record["kind"] in EVENT_TYPE_KINDS and label in self.names(e)
        ]

    def initial_pairs(self, scene: dict[str, Any]) -> BrutePairs:
        """The pair part of a scene's state at ``TIME.1`` (stage c4a): the history's
        ``initial_pairs`` (every pair between entities that holds among the participants, and no
        other, WM.38), and on each location fluent every entity's place of ``entities.csv`` with
        the history's ``initial_places`` written over the participants'."""
        pairs: dict[str, set[tuple[str, str]]] = {f: set() for f in self.pair_fluents}
        for fluent, first, second in scene.get("initial_pairs") or []:
            pairs[str(fluent)].add((str(first), str(second)))
        for fluent in self.location_fluents:
            for label, row in zip(self.labels, self.rows, strict=True):
                pairs[fluent].add((label, str(row[fluent])))
        for participant, at in (scene.get("initial_places") or {}).items():
            for fluent, place in at.items():
                pairs[str(fluent)] = {p for p in pairs[str(fluent)] if p[0] != participant}
                pairs[str(fluent)].add((str(participant), str(place)))
        return {f: frozenset(v) for f, v in pairs.items()}

    def _recorded_pair_changes(self, event: dict[str, Any]) -> set[tuple[str, str, str, bool]]:
        """An event's recorded pair changes, both directions of a symmetric pair, as
        ``(pair fluent, first, second, to)``."""
        found: set[tuple[str, str, str, bool]] = set()
        for change in event["changes"]:
            if "pair" not in change:
                continue
            fluent, first, second, to = (
                str(change["pair"]),
                str(change["first"]),
                str(change["second"]),
                bool(change["to"]),
            )
            found.add((fluent, first, second, to))
            if fluent in self.symmetric:
                found.add((fluent, second, first, to))
        return found

    def states(self, scene: dict[str, Any]) -> list[dict[str, frozenset[str]]]:
        """The state at every time point of a scene, replayed with the brute-force evaluator
        from the entities' initial fluents and the history's initial fluents (the numeric values
        beside them, in ``numerics``, and since stage c4a the pairs, in ``pairs_at``). Every
        event must be legal in the replayed state, and every step's recorded pair changes must
        be exactly the pairs the replay changed, as ``history.replay`` checks."""
        if scene["label"] not in self._states:
            state = {
                label: frozenset(f for f in self.base_fluents if row[f])
                for label, row in zip(self.labels, self.rows, strict=True)
            }
            for participant, fluents in scene["initial"].items():
                state[participant] = frozenset(fluents)
            numeric = self.brute.initial_numeric()
            for participant, values in (scene.get("initial_numeric") or {}).items():
                numeric[participant] = {f: float(v) for f, v in values.items()}
            pairs = self.initial_pairs(scene)
            states = [state]
            numerics = [numeric]
            pair_states = [pairs]
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
                    assert self.brute.legal(event, state, numeric, pairs), (
                        scene["label"],
                        step["step"],
                        event,
                    )
                state, numeric, _ = self.brute.step(state, numeric, events, pairs)
                after = self.brute.pairs_after
                changed = {
                    (fluent, a, b, (a, b) in after[fluent])
                    for fluent in self.pair_fluents
                    for a, b in pairs[fluent] ^ after[fluent]
                }
                recorded = set().union(*(self._recorded_pair_changes(e) for e in step["events"]))
                assert changed == recorded, (scene["label"], step["step"], changed ^ recorded)
                pairs = after
                states.append(state)
                numerics.append(numeric)
                pair_states.append(pairs)
            self._states[scene["label"]] = states
            self._numerics[scene["label"]] = numerics
            self._pairs[scene["label"]] = pair_states
        return self._states[scene["label"]]

    def numerics(self, scene: dict[str, Any]) -> list[dict[str, dict[str, float]]]:
        """The numeric values at every time point of a scene, replayed with ``states``."""
        self.states(scene)
        return self._numerics[scene["label"]]

    def pairs_at(self, scene: dict[str, Any], time: int | None = None) -> Any:
        """The pair part of the state at ``TIME.<time>`` of a scene (stage c4a), or the list
        of them at every time point when no time is given: for each pair fluent, the ordered
        label pairs that hold, a place second for a location fluent."""
        self.states(scene)
        pair_states = self._pairs[scene["label"]]
        return pair_states if time is None else pair_states[time - 1]

    def place_at(self, scene: dict[str, Any], time: int, entity: str, fluent: str) -> str:
        """The place of an entity on a location fluent at ``TIME.<time>`` (stage c4a)."""
        (place,) = [b for a, b in self.pairs_at(scene, time)[fluent] if a == entity]
        return place

    def pair_holds(
        self, scene: dict[str, Any], time: int, fluent: str, first: str, second: str
    ) -> bool:
        """Whether a pair holds at ``TIME.<time>`` (stage c4a)."""
        return (first, second) in self.pairs_at(scene, time)[fluent]

    def _at(self, scene: dict[str, Any], time: int) -> BruteForce:
        """The brute-force evaluator set to the numeric values and the pairs at ``TIME.<time>``."""
        return self.brute.with_numeric(self.numerics(scene)[time - 1]).with_pairs(
            self.pairs_at(scene, time)
        )

    def legal_at(
        self, scene: dict[str, Any], time: int, label: str, agent: str, patient: str | None
    ) -> bool:
        """Whether the binding was legal at ``TIME.<time>`` of the scene (with the pairs and
        the places of the time point since stage c4a: ``SAME`` on a location fluent, ``SOME``,
        ``SOMEOF``, the compiled literals, and a move's or an act's location literal)."""
        binding = {"agent": agent, **({"patient": patient} if patient is not None else {})}
        state = self.states(scene)[time - 1]
        numeric = self.numerics(scene)[time - 1]
        pairs = self.pairs_at(scene, time)
        return any(
            self.brute.legal(BruteEvent(event_type, binding), state, numeric, pairs)
            for event_type in self.leaves_below(label)
        )

    def fluent_at(self, scene: dict[str, Any], time: int, entity: str, fluent: str) -> bool:
        """A fluent's value, base or derived, of a participant at ``TIME.<time>`` (a derived
        fluent whose rule reads ``SOME`` or ``SOMEOF`` of a pair fluent from the replayed
        pairs, stage c4a)."""
        return bool(self._at(scene, time).fluent(fluent, entity, self.states(scene)[time - 1]))

    def changed(
        self, scene: dict[str, Any], entity: str, fluent: str, cut: float | None = None
    ) -> bool:
        """Whether the fluent's value at the scene's final time point differs from its value
        at ``TIME.1``; for a numeric pole (``NUMFL.1.HIGH``, with the form's ``cut``), whether
        the pole holds at one of the two time points and not at the other."""
        final = int(scene["final"].split(".")[1])
        if fluent.startswith("NUMFL."):
            assert cut is not None
            return self.pole_at(scene, final, entity, fluent, cut) != self.pole_at(
                scene, 1, entity, fluent, cut
            )
        return self.fluent_at(scene, final, entity, fluent) != self.fluent_at(
            scene, 1, entity, fluent
        )

    def numeric_at(self, scene: dict[str, Any], time: int, entity: str, fluent: str) -> float:
        """A numeric fluent's value of a participant at ``TIME.<time>``."""
        return float(self.numerics(scene)[time - 1][entity][fluent])

    def pole_at(self, scene: dict[str, Any], time: int, entity: str, pole: str, cut: float) -> bool:
        """Whether a numeric pole holds of a participant at ``TIME.<time>``: ``HIGH`` when the
        value is strictly greater than the point at the fraction ``cut`` of the way through the
        range, ``LOW`` when at most that point (WM.35: the high word is the world's literal
        ``> t`` and the low word its negation ``<= t``)."""
        fluent, side = pole.rsplit(".", 1)
        low, high = self.ranges[fluent]
        value = self.numeric_at(scene, time, entity, fluent)
        point = low + cut * (high - low)
        return value > point if side == "HIGH" else value <= point

    def made_by(
        self,
        scene: dict[str, Any],
        time: int,
        entity: str,
        fluent: str,
        event_label: str,
        direction: str | None = None,
        second: str | None = None,
    ) -> bool:
        """Whether the event's own recorded changes, applied alone to the state at
        ``TIME.<time>``, change the fluent of the participant; for a numeric fluent, whether the
        event recorded a change of it in the ``direction`` (``increase`` or ``decrease``); for a
        pair fluent (stage c4b), whether the event recorded the pair from the participant to
        ``second``, an entity or a place."""
        step = next(s for s in scene["steps"] if s["step"] == time)
        event = next((e for e in step["events"] if e["label"] == event_label), None)
        if event is None:
            return False
        if fluent in self.numeric_fluents:
            return any(
                "pair" not in c
                and c["entity"] == entity
                and c["fluent"] == fluent
                and "from" in c
                and (
                    direction is None
                    or (float(c["to"]) > float(c["from"])) == (direction == "increase")
                )
                for c in event["changes"]
            )
        if fluent in self.pair_fluents:
            # a pair change (stage c4b): the event records the pair, in either direction for a
            # symmetric pair fluent
            assert second is not None
            return any(
                (f, a, b) in {(fluent, entity, second), (fluent, second, entity)}
                if fluent in self.symmetric
                else (f, a, b) == (fluent, entity, second)
                for f, a, b, _ in self._recorded_pair_changes(event)
            )
        if not self.symbols[fluent]["derived"]:
            return any(
                "pair" not in c and c["entity"] == entity and c["fluent"] == fluent
                for c in event["changes"]
            )
        before = self.states(scene)[time - 1]
        numeric_before = self.numerics(scene)[time - 1]
        pairs_before = self.pairs_at(scene, time)
        after = {e: set(v) for e, v in before.items()}
        numeric_after = {e: dict(v) for e, v in numeric_before.items()}
        pairs_after = {f: set(v) for f, v in pairs_before.items()}
        for change in event["changes"]:
            if "pair" in change:
                # a pair change (stage c4a): both directions of a symmetric pair; a move's two
                # changes delete the old place and add the new one
                for fluent_label, first, second, to in self._recorded_pair_changes(
                    {"changes": [change]}
                ):
                    if to:
                        pairs_after[fluent_label].add((first, second))
                    else:
                        pairs_after[fluent_label].discard((first, second))
            elif "from" in change:
                numeric_after[change["entity"]][change["fluent"]] = float(change["to"])
            elif change["to"]:
                after[change["entity"]].add(change["fluent"])
            else:
                after[change["entity"]].discard(change["fluent"])
        after_state = {e: frozenset(v) for e, v in after.items()}
        self.brute.with_numeric(numeric_before).with_pairs(pairs_before)
        was = self.brute.fluent(fluent, entity, before)
        self.brute.with_numeric(numeric_after).with_pairs(
            {f: frozenset(v) for f, v in pairs_after.items()}
        )
        return was != self.brute.fluent(fluent, entity, after_state)

    def state(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether a state form (``HOLDS``) is true of the scene: a Boolean fluent's value, a
        numeric pole against the range at the form's ``cut``, a numeric comparative at the
        time point (with the sentence's word deciding which argument is the greater), or (stage
        c4b) a pair state, the pair of the pair fluent from the subject to the patient, an
        entity or a place, among the replayed pairs."""
        time = int(form["time"].split(".")[1])
        predicate = form["predicate"]
        entity = form["subject"]["instance"]
        if predicate["kind"] == "pair_state":
            other = predicate["patient"]["instance"]
            if other == entity or predicate["label"] not in self.pair_fluents:
                return None  # type: ignore[return-value]
            held = self.pair_holds(scene, time, predicate["label"], entity, other)
            return held == form["polarity"]
        if predicate["kind"] == "comparison":
            if predicate["class"] != "RANGE":
                return None  # type: ignore[return-value]
            held = self.pole_at(scene, time, entity, predicate["label"], float(predicate["cut"]))
            return held == form["polarity"]
        if predicate["kind"] == "comparative":
            other = predicate["patient"]["instance"]
            if other == entity or not form["polarity"]:
                return None  # type: ignore[return-value]
            first = self.numeric_at(scene, time, entity, predicate["label"])
            second = self.numeric_at(scene, time, other, predicate["label"])
            if self._low_word(predicate):
                first, second = second, first
            return first > second
        value = self.fluent_at(scene, time, entity, predicate["label"])
        return value == form["polarity"]

    def change(self, form: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Whether a change form (``BECOME``, or ``INCREASE`` and ``DECREASE`` for a numeric
        fluent) is true of the scene: the fluent had the other value at the time point and the
        stated value at the next; the numeric value at the next time point is greater, or less,
        than at the time point."""
        time = int(form["time"].split(".")[1])
        entity, fluent = form["subject"]["instance"], form["predicate"]["label"]
        direction = form["predicate"].get("direction")
        if form["predicate"]["kind"] == "pair_state":
            # a pair change (stage c4b): the pair lacked the stated value at the time point
            # and has it at the next, a move's new place or lost place included
            other = form["predicate"]["patient"]["instance"]
            if other == entity or fluent not in self.pair_fluents:
                return None  # type: ignore[return-value]
            before = self.pair_holds(scene, time, fluent, entity, other)
            after = self.pair_holds(scene, time + 1, fluent, entity, other)
            return before != form["polarity"] and after == form["polarity"]
        if direction is not None:
            before = self.numeric_at(scene, time, entity, fluent)
            after = self.numeric_at(scene, time + 1, entity, fluent)
            return after > before if direction == "increase" else after < before
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
            self.brute.legal(BruteEvent(event_type, binding), state, numeric, pairs)
            for event_type in self.leaves_below(label)
            for state, numeric, pairs in zip(
                self.states(scene), self.numerics(scene), self.pairs_at(scene), strict=True
            )
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
        type, as ``definition.json`` records them. An effect with a ``condition`` key is left
        out (stage b1): it holds only when its condition held, so it makes no unconditional
        causal statement true. A numeric effect (an ``amount``) and a numeric threshold literal
        (a ``threshold``) are left out too (stage b2): no statement of this stage is about
        them. A pair effect and a pair literal (a ``pair_fluent``) are left out (stage c4a):
        the statements with pair literals are stage c5a's."""
        record = self.event_types[event_type]
        if kind == "effect":
            return [
                e
                for e in record["effects"]
                if not e.get("condition") and "amount" not in e and e.get("kind") != "pair"
            ]
        return [
            lit
            for lit in record["precondition"]["literals"]
            if "threshold" not in lit and "pair_fluent" not in lit
        ]

    def causal(self, form: dict[str, Any]) -> bool | None:
        """Whether a causal statement is true: every event type below its event type (itself,
        for an event type) has the entry: the Boolean effect or literal with the value, the
        numeric effect of the ``direction``'s sign (``raises``: ``+=``), under exactly the
        ``condition`` the form's subject records (none for an unconditional statement; a
        numeric condition literal with its ``cut`` matches the record's threshold literal at
        exactly that point, WM.35, and any other numeric threshold condition is no entry). None
        for a statement about a
        derived fluent, or about a role the event type lacks."""
        subject, predicate = form["subject"], form["predicate"]
        event_type, role = subject["event"], subject["role"]
        fluent, value = predicate["label"], predicate.get("value")
        direction = predicate.get("direction")
        condition = {self._literal_key(c) for c in subject.get("condition", ())}
        if form["quantifier"] != NEC_ALL or not form["polarity"]:
            return None
        if role not in self.event_types[event_type]["roles"]:
            return None
        if fluent in self.numeric_fluents:
            if predicate["kind"] != "effect" or direction not in ("raises", "lowers"):
                return None
        elif fluent not in self.base_fluents or direction is not None:
            return None
        if condition and predicate["kind"] != "effect":
            return None
        for label in self.leaves_below(event_type):
            record = self.event_types[label]
            if predicate["kind"] == "precondition":
                found = any(
                    "pair_fluent" not in lit
                    and lit["role"] == role
                    and lit["fluent"] == fluent
                    and "threshold" not in lit
                    and lit["value"] == value
                    for lit in record["precondition"]["literals"]
                )
            else:
                found = any(
                    self._effect_matches(e, role, fluent, value, direction, condition)
                    for e in record["effects"]
                )
            if not found:
                return False
        return True

    def _literal_key(self, literal: dict[str, Any]) -> tuple:
        """What a form's condition literal says, in the record's terms: a Boolean literal as
        ``(role, symbol, value)``, a numeric pole with its ``cut`` (WM.35) as ``(role, fluent,
        operator, threshold)`` with ``>`` for the high pole and ``<=`` for the low one and the
        threshold the point of the range at the cut. A numeric pole without a cut, or a negated
        one, matches nothing."""
        label = literal["label"]
        if not label.startswith("NUMFL."):
            return (literal["role"], label, bool(literal["value"]))
        fluent, side = label.rsplit(".", 1)
        if "cut" not in literal or not literal["value"]:
            return (literal["role"], label, None)
        low, high = self.ranges[fluent]
        point = low + float(literal["cut"]) * (high - low)
        return (literal["role"], fluent, ">" if side == "HIGH" else "<=", point)

    @staticmethod
    def _record_literal_key(literal: dict[str, Any]) -> tuple:
        """What a record's condition literal says, in the same terms."""
        if literal["kind"] == "numeric":
            return (literal["role"], literal["symbol"], literal["operator"], literal["threshold"])
        return (literal["role"], literal["symbol"], bool(literal["value"]))

    def _effect_matches(
        self,
        effect: dict[str, Any],
        role: str,
        fluent: str,
        value: bool | None,
        direction: str | None,
        condition: set[tuple],
    ) -> bool:
        if effect.get("kind") == "pair" or effect["role"] != role or effect["fluent"] != fluent:
            return False
        literals = (effect.get("condition") or {}).get("literals", [])
        if any(lit["kind"].startswith("pair") for lit in literals):
            return False  # a pair condition makes no statement in phase (c) (WM.59)
        if "amount" in effect:
            if direction is None or (effect["operator"] == "+=") != (direction == "raises"):
                return False
        elif direction is not None or effect["value"] != value:
            return False
        return {self._record_literal_key(lit) for lit in literals} == condition

    def condition_held(self, scene: dict[str, Any], event: dict[str, Any], condition) -> bool:
        """Whether a conditional statement's condition held of an event's participants at the
        event's time point (``step``)."""
        for literal in condition:
            entity = event.get(literal["role"])
            if entity is None:
                return False
            if literal["label"].startswith("BOOLFL."):
                held = self.fluent_at(scene, event["step"], entity, literal["label"])
            elif literal["label"].startswith("NUMFL."):
                # a numeric pole at the form's cut (WM.35)
                held = self.pole_at(
                    scene, event["step"], entity, literal["label"], float(literal["cut"])
                )
            else:
                held = bool(self.column[literal["label"]][self.row[entity]] == 1)
            if held != bool(literal["value"]):
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
        condition = subject.get("condition", ())
        direction = predicate.get("direction")
        seen = False
        for scene in scenes:
            for step in scene["steps"]:
                for event in step["events"]:
                    if event["type"] not in below:
                        continue
                    entity = event.get(subject["role"])
                    if entity is None:
                        continue
                    if not self.condition_held(scene, {**event, "step": step["step"]}, condition):
                        continue
                    seen = True
                    if direction is not None:
                        recorded = any(
                            "pair" not in c
                            and c["entity"] == entity
                            and c["fluent"] == predicate["label"]
                            and "from" in c
                            and ((float(c["to"]) > float(c["from"])) == (direction == "raises"))
                            for c in event["changes"]
                        )
                        if not recorded:
                            return False
                        continue
                    value = self.fluent_at(scene, step["step"] + offset, entity, predicate["label"])
                    if value != predicate["value"]:
                        return False
        return seen

    # Truth -----------------------------------------------------------------------------------

    def truth(self, form: dict[str, Any], scene: dict[str, Any] | None = None) -> bool | None:
        if form["level"] == "class" and "head" in form["subject"]:
            return self.causal(form)
        if form["level"] == "class":
            if form["predicate"]["kind"] in ("comparison", "comparative"):
                return self._class_comparison(form)
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
        z = form.get("z")
        members = self.subject_set(form["subject"], z)
        kind = predicate["kind"]
        if kind == "event_type2":
            patients = self.subject_set(predicate["patient"], z)
            block = self.matrix(predicate["label"])[np.ix_(members, patients)]
            return int(block.sum()), int(
                members.sum() * patients.sum() - (members & patients).sum()
            )
        return int(self.column[predicate["label"]][members].sum()), int(members.sum())

    def _class_comparison(self, form: dict[str, Any]) -> bool | None:
        """A class-level comparison (a pole of the category's mean against its comparison class,
        an ancestor or ``ENTITY``, at the form's ``z``) or comparative (the category's mean
        against another category's, strictly), from the scalar columns alone."""
        subject, predicate = form["subject"], form["predicate"]
        quantifier, polarity, kind = form["quantifier"], form["polarity"], predicate["kind"]
        category = subject["category"]
        if quantifier is not None or category == THING:
            return None
        if subject["restriction"] or subject.get("clauses"):
            return None  # a statement about the category, not about a restricted set
        members = self.subject_set(subject)
        if kind == "comparison":
            scalar, side = predicate["label"].rsplit(".", 1)
            comparison = predicate["class"]
            if comparison != ENTITY and comparison not in self.ancestors(category):
                return None
            z = form.get("z", self.z)  # a form without a recorded z is judged at the oracle's
            values = self.column[scalar].astype(float)
            rows = self.below(comparison)
            mean, sd = values[rows].mean(), values[rows].std()
            value = values[members].mean()
            if sd == 0:
                has_pole = False
            elif side == "HIGH":
                has_pole = value >= mean + z * sd
            else:
                has_pole = value <= mean - z * sd
            return bool(has_pole) == polarity
        other = predicate["patient"]
        if not polarity or "clauses" in other or other["restriction"]:
            return None
        if other["category"] in (THING, category):
            return None
        values = self.column[predicate["label"]].astype(float)
        first, second = values[members].mean(), values[self.subject_set(other)].mean()
        if self._low_word(predicate):
            first, second = second, first  # "smaller than": the sentence's object is the greater
        return bool(first > second)

    @staticmethod
    def _low_word(predicate: dict[str, Any]) -> bool:
        """Whether a comparative's form says the sentence used the low pole's word, so that its
        subject is the lesser argument (the form holds the sentence's subject and object)."""
        return str(predicate.get("word", "")).endswith(".LOW")

    def _class(self, form: dict[str, Any]) -> bool | None:
        subject, predicate = form["subject"], form["predicate"]
        quantifier, polarity, kind = form["quantifier"], form["polarity"], predicate["kind"]
        category = subject["category"]
        if quantifier not in (NEC_ALL, ALL, MOST, SOME, NO, NEC_NO):
            return None
        if quantifier in (NEC_ALL, ALL, NO, NEC_NO) and not polarity:
            return None
        if quantifier in (NEC_ALL, NEC_NO):
            if kind not in ("property", "part", "event_type1", "member") or subject.get("clauses"):
                return None
        members = self.subject_set(subject, form.get("z"))
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
        elif kind == "comparison":
            if predicate["class"] not in path:
                return None
            rows = self.below(predicate["class"])
            value = self.pole(predicate["label"], rows, form.get("z", self.z))[row]

        elif kind == "comparative":
            other = predicate["patient"]["instance"]
            if other == subject or not form["polarity"]:
                return None
            values = self.column[predicate["label"]].astype(float)
            first, second = values[row], values[self.row[other]]
            if self._low_word(predicate):
                first, second = second, first
            return bool(first > second)

        else:
            patient = predicate["patient"]["instance"]
            if patient == subject:
                return None
            value = self.matrix(predicate["label"])[row, self.row[patient]]
        return bool(value) == form["polarity"]

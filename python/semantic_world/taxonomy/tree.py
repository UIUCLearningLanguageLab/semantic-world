"""The tree: superordinates with the similarity bound, roles, children, and distinct leaves.

Superordinates are generated in index order from the ``taxonomy:superordinates`` stream. Roles,
branching counts, children, and distinct-leaf redraws come from the ``taxonomy:tree`` stream.
Categories are visited depth first, in category order (``CATEGORY.1``, ``CATEGORY.1.1``,
``CATEGORY.1.1.1``, ...): a
category's roles are assigned, its children's free features are generated from its generative
vector according to those roles, and then each child is visited in turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from semantic_world.taxonomy.config import Config, SimilarityBound, TreeSettings
from semantic_world.taxonomy.errors import GenerationError
from semantic_world.taxonomy.rules import RuleSet
from semantic_world.taxonomy.similarity import cross_similarity
from semantic_world.taxonomy.streams import Streams


class Role(IntEnum):
    """The role of a free feature at a category, for that category's members."""

    UNDIAGNOSTIC = 0
    CHARACTERISTIC = 1
    DEFINING_NEW = 2
    DEFINING_INHERITED = 3

    @property
    def defining(self) -> bool:
        return self in (Role.DEFINING_NEW, Role.DEFINING_INHERITED)

    @property
    def csv_name(self) -> str:
        return self.name.lower()


@dataclass(eq=False)
class Category:
    label: str
    indices: tuple[int, ...]
    level: int
    parent: Category | None
    free_values: np.ndarray
    """The generative vector over the free features (free-feature order)."""
    values: np.ndarray
    """The generative vector over every non-ISA feature (matrix order)."""
    roles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    """The role of every free feature at this category, as :class:`Role` values."""
    children: list[Category] = field(default_factory=list)
    scalars: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """The category's value on every scalar dimension."""

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_superordinate(self) -> bool:
        return self.parent is None

    def ancestors(self) -> list[Category]:
        """The parent, the grandparent, and so on up to the superordinate."""
        result = []
        node = self.parent
        while node is not None:
            result.append(node)
            node = node.parent
        return result

    def descendants(self) -> list[Category]:
        """Every strict descendant, in category order."""
        result: list[Category] = []
        for child in self.children:
            result.append(child)
            result.extend(child.descendants())
        return result

    def leaves(self) -> list[Category]:
        return (
            [self] if self.is_leaf else [leaf for child in self.children for leaf in child.leaves()]
        )

    def defining_mask(self) -> np.ndarray:
        return (self.roles == Role.DEFINING_NEW) | (self.roles == Role.DEFINING_INHERITED)


@dataclass
class Tree:
    categories: tuple[Category, ...]
    """Every category in category order (depth first by index tuple)."""
    depth: int
    superordinate_tries: int
    """The total number of candidates drawn for the superordinates."""
    superordinate_similarity: tuple[float, float] | None
    """The smallest and largest pairwise similarity among the accepted superordinates on the
    configured scope and metric, or None when there is no bound or fewer than two superordinates."""
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        self._by_label = {c.label: c for c in self.categories}
        self._by_indices = {c.indices: c for c in self.categories}

    def ancestor_label(self, category: Category, level: int) -> str:
        """The label of a category's ancestor at a level (the category itself at its own
        level)."""
        return self._by_indices[category.indices[:level]].label

    def __getitem__(self, label: str) -> Category:
        try:
            return self._by_label[label]
        except KeyError:
            raise KeyError(f"unknown category {label!r}") from None

    @property
    def superordinates(self) -> tuple[Category, ...]:
        return tuple(c for c in self.categories if c.level == 1)

    @property
    def leaves(self) -> tuple[Category, ...]:
        return tuple(c for c in self.categories if c.is_leaf)

    def at_level(self, level: int) -> tuple[Category, ...]:
        return tuple(c for c in self.categories if c.level == level)

    def generative_matrix(self) -> np.ndarray:
        """One row per category: the generative vector over the non-ISA features."""
        return np.stack([c.values for c in self.categories])

    def free_matrix(self) -> np.ndarray:
        return np.stack([c.free_values for c in self.categories])

    def scalar_matrix(self) -> np.ndarray:
        """One row per category: its scalar values, shape ``(categories, scalars)``."""
        return np.stack([c.scalars for c in self.categories])

    def isa_vector(self, category: Category) -> np.ndarray:
        """The ISA features of a category, in category order: 1 for the category and its
        ancestors, 0 otherwise."""
        on = {category.label} | {a.label for a in category.ancestors()}
        return np.array([1 if c.label in on else 0 for c in self.categories], dtype=np.uint8)

    def isa_matrix(self) -> np.ndarray:
        return np.stack([self.isa_vector(c) for c in self.categories])


# ---------------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------------


def generate_tree(config: Config, rules: RuleSet, streams: Streams) -> Tree:
    """Generate the noun categories: superordinates, roles, children, and leaves."""
    return build_tree(
        config.tree_settings(), rules, streams.superordinates, streams.tree, streams.scalars
    )


def build_tree(
    settings: TreeSettings,
    rules: RuleSet,
    rng_superordinates: np.random.Generator,
    rng_tree: np.random.Generator,
    rng_scalars: np.random.Generator | None = None,
) -> Tree:
    """Generate a tree from explicit settings and streams. The world's event-type tree uses this
    with its own settings, an empty rule set over the event-type features, and the
    ``world:event_tree`` stream."""
    return _TreeBuilder(settings, rules, rng_superordinates, rng_tree, rng_scalars).build()


def stochastic_round(x: float, rng: np.random.Generator) -> int:
    """Round up with probability equal to the fractional part, so the expectation is ``x``."""
    whole = int(np.floor(x))
    return whole + (1 if rng.random() < x - whole else 0)


class _TreeBuilder:
    def __init__(
        self,
        settings: TreeSettings,
        rules: RuleSet,
        rng_superordinates: np.random.Generator,
        rng_tree: np.random.Generator,
        rng_scalars: np.random.Generator | None,
    ) -> None:
        self.settings = settings
        self.rules = rules
        self.features = rules.features
        self.rng_superordinates = rng_superordinates
        self.rng_tree = rng_tree
        self.rng_scalars = rng_scalars
        self.base_rates = self.features.base_rates
        self.n_free = len(self.base_rates)
        self.n_scalars = 0 if settings.scalars is None else settings.scalars.count
        self.depth = settings.taxonomy.depth
        self.warnings: list[str] = []
        self.categories: list[Category] = []
        self.leaf_vectors: dict[bytes, str] = {}
        self.tries = 0
        self.similarity_range: tuple[float, float] | None = None
        bound = settings.similarity_bound
        self.scope_columns = self._scope_columns(bound.scope) if bound is not None else None

    def _scope_columns(self, scope: str) -> np.ndarray:
        if scope == "free":
            return self.features.free_positions
        # ``is_has`` and ``all`` are the same scope now that the taxonomy has no CAN features.
        return np.arange(len(self.features), dtype=np.intp)

    def build(self) -> Tree:
        superordinates = self._superordinates()
        for category in superordinates:
            self._visit(category)
        return Tree(
            tuple(self.categories),
            self.depth,
            self.tries,
            self.similarity_range,
            tuple(self.warnings),
        )

    def _compute(self, free_values: np.ndarray, scalars: np.ndarray | None = None) -> np.ndarray:
        scalar_rows = None if scalars is None or self.n_scalars == 0 else scalars[None, :]
        return self.rules.compute(free_values[None, :], scalar_rows)[0]

    # Superordinates ------------------------------------------------------------------------------

    def _superordinates(self) -> list[Category]:
        rng = self.rng_superordinates
        bound = self.settings.similarity_bound
        accepted: list[Category] = []
        accepted_scope: list[np.ndarray] = []
        for index in range(1, self.settings.taxonomy.superordinates + 1):
            label = f"{self.settings.prefix}{index}"
            scalars = self._superordinate_scalars()
            if bound is None:
                free_values = (rng.random(self.n_free) < self.base_rates).astype(np.uint8)
                values = self._compute(free_values, scalars)
                self.tries += 1
            else:
                free_values, values = self._bounded_superordinate(
                    label, bound, accepted_scope, rng, scalars
                )
                accepted_scope.append(values[self.scope_columns])
            accepted.append(
                Category(label, (index,), 1, None, free_values, values, scalars=scalars)
            )
        if bound is not None and len(accepted_scope) >= 2:
            sims = cross_similarity(
                np.stack(accepted_scope), np.stack(accepted_scope), bound.metric
            )
            upper = sims[np.triu_indices(len(accepted_scope), k=1)]
            self.similarity_range = (float(np.nanmin(upper)), float(np.nanmax(upper)))
        return accepted

    def _violation(
        self, scope_values: np.ndarray, accepted_scope: list[np.ndarray], bound: SimilarityBound
    ) -> tuple[float, float]:
        """The largest violation of the bound against the accepted superordinates, and the
        similarity that produced it. An undefined similarity is an infinite violation."""
        if not accepted_scope:
            return 0.0, float("nan")
        sims = cross_similarity(scope_values[None, :], np.stack(accepted_scope), bound.metric)[0]
        if np.isnan(sims).any():
            return float("inf"), float("nan")
        excess = np.zeros_like(sims)
        if bound.max is not None:
            excess = np.maximum(excess, sims - bound.max)
        if bound.min is not None:
            excess = np.maximum(excess, bound.min - sims)
        worst = int(np.argmax(excess))
        return float(excess[worst]), float(sims[worst])

    def _bounded_superordinate(
        self,
        label: str,
        bound: SimilarityBound,
        accepted_scope: list[np.ndarray],
        rng: np.random.Generator,
        scalars: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.scope_columns is not None
        closest: tuple[float, float] = (float("inf"), float("nan"))
        best_candidate: np.ndarray | None = None
        for _ in range(bound.max_tries):
            self.tries += 1
            free_values = (rng.random(self.n_free) < self.base_rates).astype(np.uint8)
            values = self._compute(free_values, scalars)
            violation, similarity = self._violation(
                values[self.scope_columns], accepted_scope, bound
            )
            if violation == 0.0:
                return free_values, values
            if violation < closest[0]:
                closest = (violation, similarity)
                best_candidate = free_values
        if bound.local_search and best_candidate is not None:
            free_values, values, closest = self._local_search(
                best_candidate, closest, accepted_scope, bound, scalars
            )
            if closest[0] == 0.0:
                return free_values, values
        search = " and a local search" if bound.local_search else ""
        raise GenerationError(
            f"could not place superordinate {label} within the similarity bound after "
            f"{bound.max_tries} tries{search}: the closest {bound.metric} similarity reached was "
            f"{closest[1]:.4f} against the bound [{bound.min}, {bound.max}] on the {bound.scope} "
            f"features; loosen superordinates.similarity_bound.min or max, raise max_tries, turn "
            f"on local_search, or add free features"
        )

    def _local_search(
        self,
        start: np.ndarray,
        closest: tuple[float, float],
        accepted_scope: list[np.ndarray],
        bound: SimilarityBound,
        scalars: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """Flip one free feature at a time, always the flip that most reduces the largest
        violation, until the bound holds or no flip helps."""
        assert self.scope_columns is not None
        current = start.copy()
        current_values = self._compute(current, scalars)
        current_violation = closest
        for _ in range(10 * max(self.n_free, 1)):
            if current_violation[0] == 0.0:
                break
            best: tuple[float, float] | None = None
            best_index = -1
            best_values: np.ndarray | None = None
            for i in range(self.n_free):
                candidate = current.copy()
                candidate[i] ^= 1
                values = self._compute(candidate, scalars)
                violation = self._violation(values[self.scope_columns], accepted_scope, bound)
                if best is None or violation[0] < best[0]:
                    best, best_index, best_values = violation, i, values
            if best is None or best[0] >= current_violation[0]:
                break
            current[best_index] ^= 1
            current_values = best_values  # type: ignore[assignment]
            current_violation = best
        return current, current_values, current_violation

    # Roles and children ------------------------------------------------------------------------

    def _visit(self, category: Category) -> None:
        self.categories.append(category)
        rng = self.rng_tree
        level = category.level
        inheritance = self.settings.inheritance
        category.roles = self._assign_roles(category, level, rng)
        if level == self.depth:
            self._check_leaf(category, rng)
            return
        branching = self.settings.taxonomy.branching[level - 1]
        count = int(rng.integers(branching.min, branching.max + 1))
        copy_probability = inheritance.characteristic_probability[level - 1]
        children = []
        for j in range(1, count + 1):
            free_values = self._child_free_values(category, copy_probability, rng)
            scalars = self._child_scalars(category)
            child = Category(
                f"{category.label}.{j}",
                category.indices + (j,),
                level + 1,
                category,
                free_values,
                self._compute(free_values, scalars),
                scalars=scalars,
            )
            children.append(child)
        category.children = children
        for child in children:
            self._visit(child)

    def _assign_roles(self, category: Category, level: int, rng: np.random.Generator) -> np.ndarray:
        inheritance = self.settings.inheritance
        roles = np.full(self.n_free, Role.UNDIAGNOSTIC, dtype=np.int8)
        if category.parent is not None:
            roles[category.parent.defining_mask()] = Role.DEFINING_INHERITED
        available = np.flatnonzero(roles == Role.UNDIAGNOSTIC)
        size = len(available)
        n_defining = stochastic_round(inheritance.proportion_defining[level - 1] * size, rng)
        n_characteristic = stochastic_round(
            inheritance.proportion_characteristic[level - 1] * size, rng
        )
        n_characteristic = min(n_characteristic, size - n_defining)
        if size and n_defining + n_characteristic:
            chosen = rng.choice(available, size=n_defining + n_characteristic, replace=False)
            roles[chosen[:n_defining]] = Role.DEFINING_NEW
            roles[chosen[n_defining:]] = Role.CHARACTERISTIC
        return roles

    def _child_free_values(
        self, parent: Category, copy_probability: float, rng: np.random.Generator
    ) -> np.ndarray:
        """A child's free features from the parent's generative vector and roles. One uniform
        draw per free feature decides the flip of a characteristic feature or the value of an
        undiagnostic one."""
        u = rng.random(self.n_free)
        values = parent.free_values.copy()
        characteristic = parent.roles == Role.CHARACTERISTIC
        values[characteristic] ^= (u[characteristic] >= copy_probability).astype(np.uint8)
        undiagnostic = parent.roles == Role.UNDIAGNOSTIC
        values[undiagnostic] = (u[undiagnostic] < self.base_rates[undiagnostic]).astype(np.uint8)
        return values

    # Scalars ------------------------------------------------------------------------------------

    def _superordinate_scalars(self) -> np.ndarray:
        """Standard normal draws from the ``scalars`` stream, one per dimension."""
        if self.n_scalars == 0 or self.rng_scalars is None:
            return np.zeros(0)
        return self.rng_scalars.normal(size=self.n_scalars)

    def _child_scalars(self, parent: Category) -> np.ndarray:
        """The parent's values plus normal noise with the drift of the parent's level."""
        if self.n_scalars == 0 or self.rng_scalars is None or self.settings.scalars is None:
            return np.zeros(0)
        drift = self.settings.scalars.drift[parent.level - 1]
        return parent.scalars + self.rng_scalars.normal(0.0, drift, size=self.n_scalars)

    def _check_leaf(self, leaf: Category, rng: np.random.Generator) -> None:
        inheritance = self.settings.inheritance
        if not inheritance.require_distinct_leaves:
            return
        parent = leaf.parent
        for attempt in range(inheritance.distinct_max_tries + 1):
            key = leaf.values.tobytes()
            if key not in self.leaf_vectors:
                self.leaf_vectors[key] = leaf.label
                return
            if parent is None or attempt == inheritance.distinct_max_tries:
                break
            copy_probability = inheritance.characteristic_probability[parent.level - 1]
            leaf.free_values = self._child_free_values(parent, copy_probability, rng)
            leaf.values = self._compute(leaf.free_values, leaf.scalars)
        other = self.leaf_vectors[leaf.values.tobytes()]
        raise GenerationError(
            f"leaf {leaf.label} duplicates leaf {other} on every feature after "
            f"{inheritance.distinct_max_tries} redraws; turn off "
            f"inheritance.require_distinct_leaves, lower the defining proportions, or add free "
            f"features"
        )

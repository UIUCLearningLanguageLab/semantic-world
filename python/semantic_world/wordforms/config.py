"""Configuration for the word-form pipeline.

This module loads the YAML configuration described in ``docs/specs/WORDFORM_PIPELINE.md``,
fills in every default, validates every field, and writes the fully resolved configuration back
out as YAML.

Unknown keys are errors. Every error is a :class:`ConfigError` whose message names the source
file and the dotted field path, for example ``data/wordforms/x.yaml: wordforms.count: ...``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SEED_MAX = 2**64 - 1

SOURCES = ("pseudowords", "english", "mixed")
ENGINES = ("piper", "espeak")
FRONTENDS = ("waveform", "logmel", "cochleagram")
ENCODERS = ("fixed", "pretrained")
POOLINGS = ("mean",)
ASSIGNMENT_MODES = ("arbitrary",)
DEVICES = ("auto", "cpu", "cuda", "mps")

LATER_STAGES = {
    "frontends.modulation": "stage 5",
    "augmentation": "stage 5",
    "encoders.learned": "stage 6",
    "assignment.modes": "stage 7",
}


class ConfigError(ValueError):
    """A configuration error. The message names the source file and the field."""

    def __init__(self, source: str, field: str, message: str) -> None:
        self.source = source
        self.field = field
        self.message = message
        super().__init__(f"{source}: {field}: {message}")


# ---------------------------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WordformsConfig:
    source: str
    mixed_proportion_english: float
    count: int
    syllables: dict[int, float]
    """Syllable counts and their probabilities, normalized to sum to 1, in ascending order."""
    initial_stress_probability: float
    exclude_real_words: bool
    min_english_distance: int
    min_lexicon_distance: int

    @property
    def english_count(self) -> int:
        """How many of the words are real English words."""
        if self.source == "english":
            return self.count
        if self.source == "mixed":
            return _round_half_up(self.mixed_proportion_english * self.count)
        return 0

    def resolved(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "mixed_proportion_english": self.mixed_proportion_english,
            "count": self.count,
            "syllables": dict(self.syllables),
            "initial_stress_probability": self.initial_stress_probability,
            "exclude_real_words": self.exclude_real_words,
            "min_english_distance": self.min_english_distance,
            "min_lexicon_distance": self.min_lexicon_distance,
        }


@dataclass(frozen=True)
class TrimConfig:
    threshold_db: float
    margin_ms: float


@dataclass(frozen=True)
class PiperConfig:
    voice: str
    speakers: int
    noise_scale: float
    length_scale: float
    noise_w: float

    def resolved(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "speakers": self.speakers,
            "noise_scale": self.noise_scale,
            "length_scale": self.length_scale,
            "noise_w": self.noise_w,
        }


@dataclass(frozen=True)
class EspeakConfig:
    voice: str
    variants: tuple[str, ...]
    pitch: tuple[int, int]
    rate: tuple[int, int]
    speakers: int
    """The number of speakers; by default one per variant."""

    def resolved(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "variants": list(self.variants),
            "pitch": list(self.pitch),
            "rate": list(self.rate),
            "speakers": self.speakers,
        }


@dataclass(frozen=True)
class PerturbationConfig:
    rate: float
    pitch_semitones: float


@dataclass(frozen=True)
class SynthesisConfig:
    cache_dir: str
    sample_rate: int
    trim: TrimConfig
    tokens_per_speaker: int
    held_out_speaker_proportion: float
    piper: PiperConfig | None
    espeak: EspeakConfig | None
    token_perturbation: PerturbationConfig

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(e for e in ENGINES if getattr(self, e) is not None)

    def resolved(self) -> dict[str, Any]:
        engines: dict[str, Any] = {}
        if self.piper is not None:
            engines["piper"] = self.piper.resolved()
        if self.espeak is not None:
            engines["espeak"] = self.espeak.resolved()
        return {
            "cache_dir": self.cache_dir,
            "sample_rate": self.sample_rate,
            "trim": {"threshold_db": self.trim.threshold_db, "margin_ms": self.trim.margin_ms},
            "tokens_per_speaker": self.tokens_per_speaker,
            "held_out_speaker_proportion": self.held_out_speaker_proportion,
            "engines": engines,
            "token_perturbation": {
                "rate": self.token_perturbation.rate,
                "pitch_semitones": self.token_perturbation.pitch_semitones,
            },
        }


@dataclass(frozen=True)
class LogmelConfig:
    n_mels: int
    window_ms: float
    hop_ms: float

    def resolved(self) -> dict[str, Any]:
        return {"n_mels": self.n_mels, "window_ms": self.window_ms, "hop_ms": self.hop_ms}


@dataclass(frozen=True)
class CochleagramConfig:
    channels: int
    low_hz: float
    high_hz: float
    compression: float
    frame_rate: int

    def resolved(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "compression": self.compression,
            "frame_rate": self.frame_rate,
        }


@dataclass(frozen=True)
class FrontendsConfig:
    logmel: LogmelConfig | None
    cochleagram: CochleagramConfig | None

    @property
    def names(self) -> tuple[str, ...]:
        """The available front ends: ``waveform`` always, and the configured others."""
        names = ["waveform"]
        if self.logmel is not None:
            names.append("logmel")
        if self.cochleagram is not None:
            names.append("cochleagram")
        return tuple(names)

    def resolved(self) -> dict[str, Any]:
        return {
            "logmel": None if self.logmel is None else self.logmel.resolved(),
            "cochleagram": None if self.cochleagram is None else self.cochleagram.resolved(),
        }


@dataclass(frozen=True)
class FixedEmbeddingConfig:
    name: str
    frontend: str
    time_bins: int
    pca_dims: int | None

    encoder = "fixed"

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "frontend": self.frontend,
            "time_bins": self.time_bins,
            "pca_dims": self.pca_dims,
        }


@dataclass(frozen=True)
class PretrainedEmbeddingConfig:
    name: str
    model: str
    layer: int
    pooling: str

    encoder = "pretrained"

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "model": self.model,
            "layer": self.layer,
            "pooling": self.pooling,
        }


EmbeddingConfig = FixedEmbeddingConfig | PretrainedEmbeddingConfig


@dataclass(frozen=True)
class AssignmentConfig:
    mode: str
    meanings: str | None

    def resolved(self) -> dict[str, Any]:
        return {"mode": self.mode, "meanings": self.meanings}


@dataclass(frozen=True)
class Config:
    source: str
    """The configuration file, for messages."""
    name: str
    seed: int
    wordforms: WordformsConfig
    synthesis: SynthesisConfig
    frontends: FrontendsConfig
    embeddings: tuple[EmbeddingConfig, ...]
    augmentation: None
    assignment: AssignmentConfig
    device: str

    def resolved(self) -> dict[str, Any]:
        """The configuration with every default filled in, in the file's key order."""
        return {
            "name": self.name,
            "seed": self.seed,
            "wordforms": self.wordforms.resolved(),
            "synthesis": self.synthesis.resolved(),
            "frontends": self.frontends.resolved(),
            "embeddings": [e.resolved() for e in self.embeddings],
            "augmentation": self.augmentation,
            "assignment": self.assignment.resolved(),
            "device": self.device,
        }

    def with_seed(self, seed: int) -> Config:
        return Config(**{**self.__dict__, "seed": seed})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.resolved(), sort_keys=False, allow_unicode=True)


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------------------------------------
# Reading with validation
# ---------------------------------------------------------------------------------------------

_MISSING = object()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"the boolean {str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"the number {value!r}"
    if isinstance(value, str):
        return f"the string {value!r}"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return f"a value of type {type(value).__name__}"


class _Node:
    """One mapping of the configuration, read field by field with defaults and checks."""

    def __init__(self, source: str, path: str, data: Any) -> None:
        if not isinstance(data, dict):
            raise ConfigError(
                source, path or "<root>", f"expected a mapping, found {_describe(data)}"
            )
        self.source = source
        self.path = path
        self.data = data
        self.seen: set[Any] = set()

    def field(self, key: Any) -> str:
        return f"{self.path}.{key}" if self.path else str(key)

    def error(self, key: Any, message: str) -> ConfigError:
        return ConfigError(self.source, self.field(key), message)

    def get(self, key: str, default: Any = _MISSING, *, nullable: bool = False) -> Any:
        self.seen.add(key)
        if key not in self.data:
            if default is _MISSING:
                raise self.error(key, "is required")
            return default
        value = self.data[key]
        if value is None and not nullable:
            raise self.error(key, "must not be null")
        return value

    def finish(self, later: dict[str, str] | None = None) -> None:
        """Reject keys that were never read. Keys in ``later`` name the build stage that adds
        them."""
        for key in self.data:
            if key not in self.seen:
                if later and key in later:
                    raise self.error(key, f"is not available until {later[key]}")
                valid = ", ".join(sorted(str(k) for k in self.seen))
                raise self.error(key, f"unknown key; valid keys are: {valid}")

    def int(self, key: str, default: Any, *, min: int | None = None, max: int | None = None) -> int:
        value = self.get(key, default)
        return self.check_int(key, value, min=min, max=max)

    def check_int(
        self, key: Any, value: Any, *, min: int | None = None, max: int | None = None
    ) -> int:
        if not _is_int(value):
            raise self.error(key, f"expected an integer, found {_describe(value)}")
        if min is not None and value < min:
            raise self.error(key, f"must be at least {min}, found {value}")
        if max is not None and value > max:
            raise self.error(key, f"must be at most {max}, found {value}")
        return value

    def number(
        self,
        key: str,
        default: Any,
        *,
        min: float | None = None,
        max: float | None = None,
        exclusive_min: bool = False,
    ) -> float:
        value = self.get(key, default)
        return self.check_number(key, value, min=min, max=max, exclusive_min=exclusive_min)

    def check_number(
        self,
        key: Any,
        value: Any,
        *,
        min: float | None = None,
        max: float | None = None,
        exclusive_min: bool = False,
    ) -> float:
        if not _is_number(value):
            raise self.error(key, f"expected a number, found {_describe(value)}")
        if min is not None and (value < min or (exclusive_min and value == min)):
            bound = "more than" if exclusive_min else "at least"
            raise self.error(key, f"must be {bound} {min}, found {value}")
        if max is not None and value > max:
            raise self.error(key, f"must be at most {max}, found {value}")
        return value

    def probability(self, key: str, default: Any) -> float:
        return self.number(key, default, min=0, max=1)

    def bool(self, key: str, default: Any) -> bool:
        value = self.get(key, default)
        if not isinstance(value, bool):
            raise self.error(key, f"expected true or false, found {_describe(value)}")
        return value

    def string(self, key: str, default: Any, *, nullable: bool = False) -> str | None:
        value = self.get(key, default, nullable=nullable)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise self.error(key, f"expected a non-empty string, found {_describe(value)}")
        return value

    def choice(self, key: str, default: Any, choices: tuple[str, ...]) -> str:
        value = self.get(key, default)
        if not isinstance(value, str) or value not in choices:
            options = ", ".join(choices)
            raise self.error(key, f"expected one of {options}, found {_describe(value)}")
        return value

    def mapping(self, key: str, *, nullable: bool = False) -> _Node | None:
        value = self.get(key, {}, nullable=nullable)
        if value is None:
            return None
        return _Node(self.source, self.field(key), value)

    def int_pair(self, key: str, default: Any, *, min: int | None = None) -> tuple[int, int]:
        """A two-element list ``[low, high]`` of integers with ``low <= high``."""
        value = self.get(key, default)
        if not isinstance(value, list) or len(value) != 2:
            raise self.error(key, f"expected a range [low, high], found {_describe(value)}")
        low = self.check_int(key, value[0], min=min)
        high = self.check_int(key, value[1], min=min)
        if low > high:
            raise self.error(key, f"the range minimum {low} exceeds the maximum {high}")
        return low, high

    def strings(self, key: str, default: Any) -> tuple[str, ...]:
        """A non-empty list of distinct non-empty strings."""
        value = self.get(key, default)
        if not isinstance(value, list) or not value:
            raise self.error(key, f"expected a non-empty list, found {_describe(value)}")
        for item in value:
            if not isinstance(item, str) or not item:
                raise self.error(key, f"expected a list of strings, found {_describe(item)}")
        if len(set(value)) != len(value):
            raise self.error(key, "the entries must be distinct")
        return tuple(value)


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def _read_wordforms(node: _Node) -> WordformsConfig:
    source = node.choice("source", "pseudowords", SOURCES)
    proportion = node.probability("mixed_proportion_english", 0.5)
    count = node.int("count", 500, min=1)
    syllables = _read_syllables(node)
    config = WordformsConfig(
        source=source,
        mixed_proportion_english=proportion,
        count=count,
        syllables=syllables,
        initial_stress_probability=node.probability("initial_stress_probability", 0.8),
        exclude_real_words=node.bool("exclude_real_words", True),
        min_english_distance=node.int("min_english_distance", 1, min=1),
        min_lexicon_distance=node.int("min_lexicon_distance", 1, min=1),
    )
    node.finish()
    return config


def _read_syllables(node: _Node) -> dict[int, float]:
    value = node.get("syllables", {1: 0.3, 2: 0.5, 3: 0.2})
    if not isinstance(value, dict) or not value:
        raise node.error("syllables", f"expected a mapping of weights, found {_describe(value)}")
    inner = _Node(node.source, node.field("syllables"), value)
    weights: dict[int, float] = {}
    for key, weight in value.items():
        if not _is_int(key) or key < 1:
            raise inner.error(key, "keys must be positive syllable counts")
        weights[key] = inner.check_number(key, weight, min=0)
    total = sum(weights.values())
    if total <= 0:
        raise node.error("syllables", "at least one weight must be positive")
    return {k: weights[k] / total for k in sorted(weights)}


def _read_piper(node: _Node) -> PiperConfig:
    config = PiperConfig(
        voice=node.string("voice", "en_US-libritts_r-medium"),
        speakers=node.int("speakers", 40, min=1),
        noise_scale=node.number("noise_scale", 0.667, min=0),
        length_scale=node.number("length_scale", 1.0, min=0, exclusive_min=True),
        noise_w=node.number("noise_w", 0.8, min=0),
    )
    node.finish()
    return config


def _read_espeak(node: _Node) -> EspeakConfig:
    variants = node.strings("variants", ["m1", "m3", "m7", "f2", "f4"])
    config = EspeakConfig(
        voice=node.string("voice", "en-us"),
        variants=variants,
        pitch=node.int_pair("pitch", [35, 65], min=0),
        rate=node.int_pair("rate", [150, 190], min=1),
        speakers=node.int("speakers", len(variants), min=1),
    )
    node.finish()
    return config


def _read_synthesis(node: _Node) -> SynthesisConfig:
    trim = node.mapping("trim")
    trim_config = TrimConfig(
        threshold_db=trim.number("threshold_db", -40, max=0),
        margin_ms=trim.number("margin_ms", 20, min=0),
    )
    trim.finish()
    engines_value = node.get("engines", {"piper": {}, "espeak": {}})
    engines = _Node(node.source, node.field("engines"), engines_value)
    piper_node = engines.mapping("piper", nullable=True)
    espeak_node = engines.mapping("espeak", nullable=True)
    engines.finish()
    piper = None if piper_node is None else _read_piper(piper_node)
    espeak = None if espeak_node is None else _read_espeak(espeak_node)
    if piper is None and espeak is None:
        raise node.error("engines", "at least one engine must be configured")
    perturbation = node.mapping("token_perturbation")
    perturbation_config = PerturbationConfig(
        rate=perturbation.number("rate", 0.05, min=0),
        pitch_semitones=perturbation.number("pitch_semitones", 0.5, min=0),
    )
    perturbation.finish()
    config = SynthesisConfig(
        cache_dir=node.string("cache_dir", "runs/wordforms/cache"),
        sample_rate=node.int("sample_rate", 16000, min=1),
        trim=trim_config,
        tokens_per_speaker=node.int("tokens_per_speaker", 2, min=1),
        held_out_speaker_proportion=node.probability("held_out_speaker_proportion", 0.2),
        piper=piper,
        espeak=espeak,
        token_perturbation=perturbation_config,
    )
    node.finish()
    return config


def _read_frontends(node: _Node) -> FrontendsConfig:
    logmel_node = node.mapping("logmel", nullable=True)
    logmel = None
    if logmel_node is not None:
        logmel = LogmelConfig(
            n_mels=logmel_node.int("n_mels", 80, min=1),
            window_ms=logmel_node.number("window_ms", 25, min=0, exclusive_min=True),
            hop_ms=logmel_node.number("hop_ms", 10, min=0, exclusive_min=True),
        )
        logmel_node.finish()
    cochleagram_node = node.mapping("cochleagram", nullable=True)
    cochleagram = None
    if cochleagram_node is not None:
        low = cochleagram_node.number("low_hz", 50, min=0, exclusive_min=True)
        high = cochleagram_node.number("high_hz", 8000, min=low, exclusive_min=True)
        cochleagram = CochleagramConfig(
            channels=cochleagram_node.int("channels", 64, min=1),
            low_hz=low,
            high_hz=high,
            compression=cochleagram_node.number("compression", 0.3, min=0, exclusive_min=True),
            frame_rate=cochleagram_node.int("frame_rate", 100, min=1),
        )
        cochleagram_node.finish()
    node.finish({"modulation": LATER_STAGES["frontends.modulation"]})
    return FrontendsConfig(logmel=logmel, cochleagram=cochleagram)


def _read_embeddings(root: _Node, frontends: FrontendsConfig) -> tuple[EmbeddingConfig, ...]:
    default = [
        {"name": "cochleagram_fixed", "encoder": "fixed", "frontend": "cochleagram"},
        {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel"},
        {"name": "hubert_base", "encoder": "pretrained", "model": "facebook/hubert-base-ls960"},
    ]
    value = root.get("embeddings", default)
    if not isinstance(value, list):
        raise root.error("embeddings", f"expected a list, found {_describe(value)}")
    result: list[EmbeddingConfig] = []
    names: set[str] = set()
    for i, item in enumerate(value):
        node = _Node(root.source, f"embeddings[{i}]", item)
        name = node.string("name", _MISSING)
        if name in names:
            raise node.error("name", f"the name {name!r} is used twice")
        names.add(name)
        encoder = node.get("encoder", _MISSING)
        if encoder == "learned":
            raise node.error(
                "encoder", f"is not available until {LATER_STAGES['encoders.learned']}"
            )
        encoder = node.choice("encoder", _MISSING, ENCODERS)
        if encoder == "fixed":
            frontend = node.choice("frontend", _MISSING, FRONTENDS)
            if frontend not in frontends.names:
                raise node.error("frontend", f"the front end {frontend!r} is not configured")
            pca = node.get("pca_dims", 256, nullable=True)
            result.append(
                FixedEmbeddingConfig(
                    name=name,
                    frontend=frontend,
                    time_bins=node.int("time_bins", 10, min=1),
                    pca_dims=None if pca is None else node.check_int("pca_dims", pca, min=1),
                )
            )
        else:
            result.append(
                PretrainedEmbeddingConfig(
                    name=name,
                    model=node.string("model", _MISSING),
                    layer=node.int("layer", 6, min=0),
                    pooling=node.choice("pooling", "mean", POOLINGS),
                )
            )
        node.finish()
    return tuple(result)


def _read_assignment(node: _Node) -> AssignmentConfig:
    mode = node.get("mode", "arbitrary")
    if isinstance(mode, str) and mode in ("target_correlation", "branch_markers", "acoustic"):
        raise node.error("mode", f"is not available until {LATER_STAGES['assignment.modes']}")
    config = AssignmentConfig(
        mode=node.choice("mode", "arbitrary", ASSIGNMENT_MODES),
        meanings=node.string("meanings", None, nullable=True),
    )
    node.finish()
    return config


def parse_config(data: Any, source: str, *, seed: int | None = None) -> Config:
    """Validate a loaded YAML document and fill in the defaults. ``source`` names the file in
    error messages. ``seed`` overrides the file's master seed."""
    root = _Node(source, "", data)
    name = root.string("name", "default")
    file_seed = root.int("seed", 1, min=0, max=SEED_MAX)
    if seed is not None:
        if not _is_int(seed) or not 0 <= seed <= SEED_MAX:
            raise ConfigError(
                source, "seed", f"the seed override must be an integer in [0, {SEED_MAX}]"
            )
        file_seed = seed
    wordforms = _read_wordforms(root.mapping("wordforms"))
    synthesis = _read_synthesis(root.mapping("synthesis"))
    frontends = _read_frontends(root.mapping("frontends"))
    embeddings = _read_embeddings(root, frontends)
    augmentation = root.get("augmentation", None, nullable=True)
    if augmentation is not None:
        raise root.error(
            "augmentation",
            f"must be null; augmentation is not available until {LATER_STAGES['augmentation']}",
        )
    assignment = _read_assignment(root.mapping("assignment"))
    device = root.choice("device", "auto", DEVICES)
    root.get("provenance", None, nullable=True)  # written by a run; ignored when read back
    root.finish()
    return Config(
        source=source,
        name=name,
        seed=file_seed,
        wordforms=wordforms,
        synthesis=synthesis,
        frontends=frontends,
        embeddings=embeddings,
        augmentation=None,
        assignment=assignment,
        device=device,
    )


def load_config(path: str | Path, *, seed: int | None = None) -> Config:
    """Load and validate a configuration file. ``seed`` overrides the file's master seed."""
    path = Path(path)
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(source, "<file>", f"cannot read the file: {error.strerror}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(source, "<file>", f"invalid YAML: {error}") from error
    if data is None:
        data = {}
    return parse_config(data, source, seed=seed)

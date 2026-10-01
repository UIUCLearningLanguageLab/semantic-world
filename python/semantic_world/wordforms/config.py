"""Configuration for the word-form pipeline.

This module loads the YAML configuration described in ``docs/specs/WORDFORM_PIPELINE.md``,
fills in every default, validates every field, and writes the fully resolved configuration back
out as YAML.

Unknown keys are errors. Every error is a :class:`ConfigError` whose message names the source
file and the dotted field path, for example ``data/wordforms/x.yaml: wordforms.count: ...``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEED_MAX = 2**64 - 1

SOURCES = ("pseudowords", "english", "mixed")
ENGINES = ("piper", "espeak")
ENCODERS = ("fixed", "pretrained", "learned")
POOLINGS = ("mean",)
ASSIGNMENT_MODES = ("arbitrary",)
FUNCTION_SHAPES = ("CV", "CVC", "VC", "V")
"""The shapes of a function word: one syllable with at most one consonant on each side."""
AFFIX_SHAPES = ("C", "VC", "V")
AFFIX_POSITIONS = ("suffix", "prefix")
GLIDES = ("Y", "W")
CLOSED_CLASS_SOURCES = ("pseudo", "english")
DEFAULT_FUNCTION_WORDS = (
    "the", "and", "a", "is", "that", "it", "with", "not", "all", "can", "has", "no", "some",
    "most", "without",
)  # fmt: skip
"""The default function-word glosses, in order of English frequency (wordfreq, large list)."""
DEFAULT_FUNCTION_SHAPES = {"CV": 0.3, "CVC": 0.4, "VC": 0.3}
DEFAULT_AFFIX_SHAPES = {"C": 0.5, "VC": 0.5, "V": 0.0}
"""A bare vowel suffix is available by setting, but never drawn by default."""
DEFAULT_AFFIXES = (("PLURAL", "suffix"), ("PAST", "suffix"), ("PROGRESSIVE", "suffix"))
DEFAULT_PRETRAINED_LAYER = 8
"""The default layer of a pretrained model: the best HuBERT base layer for telling words apart
across speakers in the stage 4 layer sweep."""
DEVICES = ("auto", "cpu", "cuda", "mps")

LATER_STAGES = {
    "assignment.modes": "stage 7",
}
LEARNED_KINDS = ("contrastive", "cpc")
TRAINING_TOKENS = ("clean", "all")
CPC_EMBEDDINGS = ("context", "latents")
CPC_NEGATIVES = ("batch", "clip")
FRONTENDS = ("waveform", "logmel", "cochleagram", "modulation")
NOISE_KINDS = ("white", "pink", "speech", "babble")
AUGMENTED_SPEAKERS = ("all", "train", "held_out")
WORD_EMBEDDING_TOKENS = ("clean", "all")


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
    english_min_zipf: float | None
    """The smallest Zipf frequency of the words that the sound patterns are learned from; None
    uses the whole dictionary."""
    exclude_inflections: bool = True
    """Whether the regular inflections of other dictionary words are left out of the words that
    the sound patterns are learned from."""

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
            "english_min_zipf": self.english_min_zipf,
            "exclude_inflections": self.exclude_inflections,
        }


@dataclass(frozen=True)
class TrimConfig:
    threshold_db: float
    margin_ms: float


@dataclass(frozen=True)
class LevelConfig:
    """Every clip is scaled to the RMS level ``rms_db`` (dB relative to full scale), or lower
    when the scaled clip's peak would exceed ``max_peak``."""

    rms_db: float
    max_peak: float


@dataclass(frozen=True)
class DurationCheckConfig:
    """A clip longer than ``max_ratio`` times the median duration of its word's tokens is
    synthesized again, up to ``max_tries`` tries in all."""

    max_ratio: float
    max_tries: int


@dataclass(frozen=True)
class PiperConfig:
    voice: str
    speakers: int
    noise_scale: float
    length_scale: float
    noise_w: float
    voice_dir: str
    """The folder that holds the voice's ``.onnx`` and ``.onnx.json`` files."""

    def resolved(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "voice_dir": self.voice_dir,
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
    level: LevelConfig
    duration_check: DurationCheckConfig | None
    long_synthesis_ratio: float | None
    """A word is flagged ``long_synthesis`` when the median duration of its Piper tokens is more
    than this many times the median for words with the same number of syllables. None: no flag."""
    tokens_per_speaker: int
    held_out_speaker_proportion: float
    piper: PiperConfig | None
    espeak: EspeakConfig | None
    token_perturbation: PerturbationConfig

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(e for e in ENGINES if getattr(self, e) is not None)

    def resolved(self) -> dict[str, Any]:
        # an engine that is off is written as null, so that the resolved file reloads equal
        engines: dict[str, Any] = {
            "piper": None if self.piper is None else self.piper.resolved(),
            "espeak": None if self.espeak is None else self.espeak.resolved(),
        }
        return {
            "cache_dir": self.cache_dir,
            "sample_rate": self.sample_rate,
            "trim": {"threshold_db": self.trim.threshold_db, "margin_ms": self.trim.margin_ms},
            "level": {"rms_db": self.level.rms_db, "max_peak": self.level.max_peak},
            "duration_check": None
            if self.duration_check is None
            else {
                "max_ratio": self.duration_check.max_ratio,
                "max_tries": self.duration_check.max_tries,
            },
            "long_synthesis_ratio": self.long_synthesis_ratio,
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
class ModulationConfig:
    """Spectrotemporal modulation over the cochleagram: Gabor filters at temporal ``rates`` (Hz)
    and spectral ``scales`` (cycles per octave), with the response averaged in ``bands`` bands of
    cochleagram channels."""

    rates: tuple[float, ...]
    scales: tuple[float, ...]
    bands: int

    def resolved(self) -> dict[str, Any]:
        return {"rates": list(self.rates), "scales": list(self.scales), "bands": self.bands}


@dataclass(frozen=True)
class FrontendsConfig:
    logmel: LogmelConfig | None
    cochleagram: CochleagramConfig | None
    modulation: ModulationConfig | None = None
    store_waveform: bool = False
    """Whether a run stores the waveform front end. Stored waveforms repeat the audio cache."""

    @property
    def names(self) -> tuple[str, ...]:
        """The available front ends: ``waveform`` always, and the configured others."""
        names = ["waveform"]
        if self.logmel is not None:
            names.append("logmel")
        if self.cochleagram is not None:
            names.append("cochleagram")
        if self.modulation is not None:
            names.append("modulation")
        return tuple(names)

    def resolved(self) -> dict[str, Any]:
        return {
            "waveform": {"store": self.store_waveform},
            "logmel": None if self.logmel is None else self.logmel.resolved(),
            "cochleagram": None if self.cochleagram is None else self.cochleagram.resolved(),
            "modulation": None if self.modulation is None else self.modulation.resolved(),
        }


@dataclass(frozen=True)
class FixedEmbeddingConfig:
    name: str
    frontend: str
    time_bins: int
    pca_dims: int | None
    talker_normalization: bool = False
    """Whether each speaker's mean token embedding is subtracted from that speaker's tokens."""

    encoder = "fixed"

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "frontend": self.frontend,
            "time_bins": self.time_bins,
            "pca_dims": self.pca_dims,
            "talker_normalization": self.talker_normalization,
        }


@dataclass(frozen=True)
class PretrainedEmbeddingConfig:
    name: str
    model: str
    layer: int
    pooling: str
    store_layers: bool = False
    """Whether the run also stores the pooled output of every layer (``layers.npy``). The
    evaluation's layer sweep does not need the stored layers."""
    talker_normalization: bool = False

    encoder = "pretrained"

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "model": self.model,
            "layer": self.layer,
            "pooling": self.pooling,
            "store_layers": self.store_layers,
            "talker_normalization": self.talker_normalization,
        }


@dataclass(frozen=True)
class LearnedEmbeddingConfig:
    """An encoder trained on the run's own audio: ``contrastive`` (an acoustic word encoder
    supervised by word identity) or ``cpc`` (self-supervised, by prediction alone)."""

    name: str
    kind: str
    frontend: str
    dims: int
    """The embedding size of the contrastive encoder, and the latent size of the CPC encoder,
    whose embedding is its context of ``hidden`` numbers."""
    hidden: int
    layers: int
    kernel: int
    epochs: int
    batch_size: int
    learning_rate: float
    temperature: float
    """The contrastive loss's temperature."""
    steps_ahead: int
    """How many frames ahead the CPC encoder predicts."""
    negatives: int
    """How many negative frames each CPC prediction is scored against."""
    train_on: str
    """Which training-speaker tokens the encoder trains on: ``clean``, or ``all`` with the
    augmented ones."""
    embedding_from: str = "context"
    """What the CPC encoder's embedding is the mean of: its ``context`` frames (``hidden``
    numbers) or its ``latents`` (``dims`` numbers). The contrastive encoder ignores it."""
    negatives_from: str = "batch"
    """Where the CPC encoder's negative frames come from: the whole ``batch``, or the same
    ``clip``."""
    talker_normalization: bool = False

    encoder = "learned"

    def settings(self) -> dict[str, Any]:
        """The training settings, as the trainer takes them."""
        return {
            "dims": self.dims,
            "hidden": self.hidden,
            "layers": self.layers,
            "kernel": self.kernel,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "steps_ahead": self.steps_ahead,
            "negatives": self.negatives,
            "embedding_from": self.embedding_from,
            "negatives_from": self.negatives_from,
        }

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "kind": self.kind,
            "frontend": self.frontend,
            **self.settings(),
            "train_on": self.train_on,
            "talker_normalization": self.talker_normalization,
        }


EmbeddingConfig = FixedEmbeddingConfig | PretrainedEmbeddingConfig | LearnedEmbeddingConfig


@dataclass(frozen=True)
class WordEmbeddingsConfig:
    """Which tokens make a word's embedding: the training-speaker tokens without augmentation
    (``clean``, the default), or every training-speaker token (``all``)."""

    tokens: str

    def resolved(self) -> dict[str, Any]:
        return {"tokens": self.tokens}


@dataclass(frozen=True)
class TrainingConfig:
    """What every trained encoder (a learned encoder, or a fixed encoder's projection) shares."""

    held_out_word_proportion: float
    """The share of the content words that no trained encoder sees; they test novel words."""

    def resolved(self) -> dict[str, Any]:
        return {"held_out_word_proportion": self.held_out_word_proportion}


@dataclass(frozen=True)
class AssignmentConfig:
    mode: str
    meanings: str | None

    def resolved(self) -> dict[str, Any]:
        return {"mode": self.mode, "meanings": self.meanings}


@dataclass(frozen=True)
class AffixItem:
    gloss: str
    position: str
    """``suffix`` or ``prefix``."""


@dataclass(frozen=True)
class InflectEntry:
    """Which words take which affixes."""

    words: str | tuple[str, ...]
    """``all``, ``none``, or the labels of content words."""
    affixes: tuple[str, ...]
    """The glosses of the affixes."""

    def resolved(self) -> dict[str, Any]:
        words = self.words if isinstance(self.words, str) else list(self.words)
        return {"words": words, "affixes": list(self.affixes)}


@dataclass(frozen=True)
class ClosedClassConfig:
    """Function words, affixes, and inflected forms. The glosses, the affix items, and the
    inflect entries are the closed-class request, given inline or read from a request file."""

    glosses: tuple[str, ...]
    """The glosses of the function words, most frequent first; the order is the label order."""
    function_source: str
    """``pseudo``: generated forms; ``english``: each gloss's English pronunciation."""
    function_shapes: dict[str, float]
    """The weights of the function-word shapes, normalized to sum to 1."""
    min_distance: int
    affixes: tuple[AffixItem, ...]
    affix_source: str
    """``pseudo``: generated forms; ``english``: the English suffixes with their allomorphs."""
    affix_shapes: dict[str, float]
    epenthesis: bool
    glide: str
    """The glide inserted where a vowel meets a vowel at the join and neither vowel decides:
    ``Y`` or ``W``."""
    max_skipped: float
    """A generated affix is rejected when more than this share of the content words cannot take
    it."""
    inflect: tuple[InflectEntry, ...]
    request: str | None = field(default=None, compare=False)
    """The request file that the request was read from. The resolved configuration holds the
    request inline, so this field takes no part in comparisons."""

    def resolved(self) -> dict[str, Any]:
        return {
            "request": None,
            "function_words": {
                "glosses": list(self.glosses),
                "source": self.function_source,
                "shapes": dict(self.function_shapes),
                "min_distance": self.min_distance,
            },
            "affixes": {
                "items": [{"gloss": a.gloss, "position": a.position} for a in self.affixes],
                "source": self.affix_source,
                "shapes": dict(self.affix_shapes),
                "epenthesis": self.epenthesis,
                "glide": self.glide,
                "max_skipped": self.max_skipped,
            },
            "inflect": [entry.resolved() for entry in self.inflect],
        }


@dataclass(frozen=True)
class NoiseConfig:
    kinds: tuple[str, ...]
    """The noise kinds a recipe draws from: white, pink, speech (shaped like the run's own
    speech), or babble (other tokens of the run mixed together)."""
    snr_db: tuple[float, float]
    """The signal-to-noise ratio, drawn uniformly from this range."""
    babble_voices: int
    """How many other tokens make up babble."""

    def resolved(self) -> dict[str, Any]:
        return {
            "kinds": list(self.kinds),
            "snr_db": list(self.snr_db),
            "babble_voices": self.babble_voices,
        }


@dataclass(frozen=True)
class ReverberationConfig:
    rt60: tuple[float, float]
    """The reverberation time in seconds, drawn uniformly from this range."""
    room_m: tuple[float, float]
    """Each side of the (shoebox) room, in meters, drawn uniformly from this range."""

    def resolved(self) -> dict[str, Any]:
        return {"rt60": list(self.rt60), "room_m": list(self.room_m)}


@dataclass(frozen=True)
class SpeedPitchConfig:
    """Speed and pitch perturbation of a cached clip."""

    speed: tuple[float, float]
    """The speed factor (a faster clip is shorter and higher), drawn from this range."""
    pitch_semitones: tuple[float, float]
    """A pitch shift in semitones at the same speed, drawn from this range."""

    def resolved(self) -> dict[str, Any]:
        return {"speed": list(self.speed), "pitch_semitones": list(self.pitch_semitones)}


@dataclass(frozen=True)
class ManipulationConfig:
    """Praat manipulation, as in Praat's "Change gender" command. Each setting is a range that
    the value is drawn from, or None to leave that dimension alone."""

    pitch_median_hz: tuple[float, float] | None
    pitch_range_factor: tuple[float, float] | None
    formant_shift_ratio: tuple[float, float] | None
    duration_factor: tuple[float, float] | None

    def resolved(self) -> dict[str, Any]:
        return {
            "pitch_median_hz": None if self.pitch_median_hz is None else list(self.pitch_median_hz),
            "pitch_range_factor": None
            if self.pitch_range_factor is None
            else list(self.pitch_range_factor),
            "formant_shift_ratio": None
            if self.formant_shift_ratio is None
            else list(self.formant_shift_ratio),
            "duration_factor": None if self.duration_factor is None else list(self.duration_factor),
        }


@dataclass(frozen=True)
class RecipeConfig:
    """One augmentation: the transformations applied, in the order manipulation, speed and
    pitch, reverberation, noise."""

    name: str
    noise: NoiseConfig | None
    reverberation: ReverberationConfig | None
    speed_pitch: SpeedPitchConfig | None
    manipulation: ManipulationConfig | None

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "noise": None if self.noise is None else self.noise.resolved(),
            "reverberation": None if self.reverberation is None else self.reverberation.resolved(),
            "speed_pitch": None if self.speed_pitch is None else self.speed_pitch.resolved(),
            "manipulation": None if self.manipulation is None else self.manipulation.resolved(),
        }


@dataclass(frozen=True)
class AugmentationConfig:
    """Seeded transformations of cached audio. Every recipe is applied to a seeded share of the
    tokens, and each result is a new token with its own record."""

    recipes: tuple[RecipeConfig, ...]
    proportion: float
    """The share of the eligible tokens that each recipe is applied to."""
    speakers: str
    """Which tokens are eligible: those of all speakers, of training speakers, or of held-out
    speakers."""

    def resolved(self) -> dict[str, Any]:
        return {
            "recipes": [r.resolved() for r in self.recipes],
            "proportion": self.proportion,
            "speakers": self.speakers,
        }


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
    closed_class: ClosedClassConfig | None
    """None: the run has content words only."""
    augmentation: AugmentationConfig | None
    """None: no augmented tokens."""
    word_embeddings: WordEmbeddingsConfig
    training: TrainingConfig
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
            "closed_class": None if self.closed_class is None else self.closed_class.resolved(),
            "augmentation": None if self.augmentation is None else self.augmentation.resolved(),
            "word_embeddings": self.word_embeddings.resolved(),
            "training": self.training.resolved(),
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

    def numbers(
        self,
        key: str,
        default: Any,
        *,
        min: float | None = None,
        max: float | None = None,
        exclusive_min: bool = False,
    ) -> tuple[float, ...]:
        """A non-empty list of distinct numbers, in ascending order."""
        value = self.get(key, default)
        if not isinstance(value, list) or not value:
            raise self.error(key, f"expected a non-empty list of numbers, found {_describe(value)}")
        numbers = tuple(
            float(self.check_number(key, item, min=min, max=max, exclusive_min=exclusive_min))
            for item in value
        )
        if len(set(numbers)) != len(numbers) or list(numbers) != sorted(numbers):
            raise self.error(key, "the numbers must be distinct and in ascending order")
        return numbers

    def range(
        self, key: str, default: Any, *, min: float | None = None, exclusive_min: bool = False
    ) -> tuple[float, float]:
        """A range ``[low, high]`` of numbers with ``low <= high``, or one number for a fixed
        value."""
        value = self.get(key, default)
        if _is_number(value):
            value = [value, value]
        if not isinstance(value, list) or len(value) != 2:
            raise self.error(
                key, f"expected a number or a range [low, high], found {_describe(value)}"
            )
        low = self.check_number(key, value[0], min=min, exclusive_min=exclusive_min)
        high = self.check_number(key, value[1], min=min, exclusive_min=exclusive_min)
        if low > high:
            raise self.error(key, f"the range minimum {low} exceeds the maximum {high}")
        return float(low), float(high)

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
    min_zipf = node.get("english_min_zipf", 3.0, nullable=True)
    if min_zipf is not None:
        min_zipf = float(node.check_number("english_min_zipf", min_zipf, min=0, max=8))
    config = WordformsConfig(
        source=source,
        mixed_proportion_english=proportion,
        count=count,
        syllables=syllables,
        initial_stress_probability=node.probability("initial_stress_probability", 0.8),
        exclude_real_words=node.bool("exclude_real_words", True),
        min_english_distance=node.int("min_english_distance", 1, min=1),
        min_lexicon_distance=node.int("min_lexicon_distance", 1, min=1),
        english_min_zipf=min_zipf,
        exclude_inflections=node.bool("exclude_inflections", True),
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
        voice_dir=node.string("voice_dir", "runs/wordforms/voices"),
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
    level = node.mapping("level")
    level_config = LevelConfig(
        rms_db=level.number("rms_db", -24, max=0),
        max_peak=level.number("max_peak", 0.9, min=0, max=1, exclusive_min=True),
    )
    level.finish()
    duration_node = node.mapping("duration_check", nullable=True)
    duration_check = None
    if duration_node is not None:
        duration_check = DurationCheckConfig(
            max_ratio=duration_node.number("max_ratio", 1.8, min=1, exclusive_min=True),
            max_tries=duration_node.int("max_tries", 5, min=1),
        )
        duration_node.finish()
    long_ratio = node.get("long_synthesis_ratio", 1.6, nullable=True)
    if long_ratio is not None:
        long_ratio = float(
            node.check_number("long_synthesis_ratio", long_ratio, min=1, exclusive_min=True)
        )
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
        level=level_config,
        duration_check=duration_check,
        long_synthesis_ratio=long_ratio,
        tokens_per_speaker=node.int("tokens_per_speaker", 2, min=1),
        held_out_speaker_proportion=node.probability("held_out_speaker_proportion", 0.2),
        piper=piper,
        espeak=espeak,
        token_perturbation=perturbation_config,
    )
    node.finish()
    return config


def _read_frontends(node: _Node, sample_rate: int) -> FrontendsConfig:
    waveform_node = node.mapping("waveform")
    store_waveform = waveform_node.bool("store", False)
    waveform_node.finish()
    logmel_node = node.mapping("logmel", nullable=True)
    logmel = None
    if logmel_node is not None:
        logmel = LogmelConfig(
            n_mels=logmel_node.int("n_mels", 80, min=1),
            window_ms=logmel_node.number("window_ms", 25, min=0, exclusive_min=True),
            hop_ms=logmel_node.number("hop_ms", 10, min=0, exclusive_min=True),
        )
        logmel_node.finish()
        if logmel.window_ms < logmel.hop_ms:
            raise logmel_node.error("window_ms", "must be at least hop_ms")
        if round(logmel.hop_ms * sample_rate / 1000.0) < 1:
            raise logmel_node.error("hop_ms", "is shorter than one sample")
    cochleagram_node = node.mapping("cochleagram", nullable=True)
    cochleagram = None
    if cochleagram_node is not None:
        low = cochleagram_node.number("low_hz", 50, min=0, exclusive_min=True)
        high = cochleagram_node.number("high_hz", 8000, min=low, exclusive_min=True)
        cochleagram = CochleagramConfig(
            channels=cochleagram_node.int("channels", 64, min=3),
            low_hz=low,
            high_hz=high,
            compression=cochleagram_node.number("compression", 0.3, min=0, exclusive_min=True),
            frame_rate=cochleagram_node.int("frame_rate", 100, min=1),
        )
        cochleagram_node.finish()
        if cochleagram.high_hz > sample_rate / 2:
            raise cochleagram_node.error(
                "high_hz", f"must be at most half the sample rate ({sample_rate / 2:g} Hz)"
            )
        if sample_rate % cochleagram.frame_rate != 0:
            raise cochleagram_node.error(
                "frame_rate", f"must divide the sample rate ({sample_rate})"
            )
    # the modulation front end is off unless configured
    modulation_node = None
    if node.get("modulation", None, nullable=True) is not None:
        modulation_node = node.mapping("modulation")
    modulation = None
    if modulation_node is not None:
        if cochleagram is None:
            raise node.error("modulation", "needs the cochleagram front end")
        modulation = ModulationConfig(
            rates=modulation_node.numbers("rates", [2, 4, 8, 16, 32], min=0, exclusive_min=True),
            scales=modulation_node.numbers(
                "scales", [0.25, 0.5, 1, 2, 4], min=0, exclusive_min=True
            ),
            bands=modulation_node.int("bands", 8, min=1, max=cochleagram.channels),
        )
        modulation_node.finish()
        if max(modulation.rates) > cochleagram.frame_rate / 2:
            raise modulation_node.error(
                "rates",
                f"must be at most half the cochleagram frame rate ({cochleagram.frame_rate})",
            )
        # the cochleagram samples the frequency axis about (channels - 1) / log2(high / low)
        # times per octave, and a scale above half that cannot be resolved
        per_octave = (cochleagram.channels - 1) / math.log2(
            cochleagram.high_hz / cochleagram.low_hz
        )
        if max(modulation.scales) > per_octave / 2:
            raise modulation_node.error(
                "scales",
                f"must be at most {per_octave / 2:.2f} cycles per octave, half the cochleagram's "
                f"{per_octave:.1f} channels per octave; use more cochleagram channels for finer "
                f"scales",
            )
    node.finish()
    return FrontendsConfig(
        logmel=logmel,
        cochleagram=cochleagram,
        modulation=modulation,
        store_waveform=store_waveform,
    )


def _read_embeddings(root: _Node, frontends: FrontendsConfig) -> tuple[EmbeddingConfig, ...]:
    default = [
        {"name": "cochleagram_fixed", "encoder": "fixed", "frontend": "cochleagram"},
        {"name": "logmel_fixed", "encoder": "fixed", "frontend": "logmel"},
        {"name": "hubert_base", "encoder": "pretrained", "model": "facebook/hubert-base-ls960"},
        {"name": "contrastive_logmel", "encoder": "learned", "kind": "contrastive"},
        {"name": "cpc_logmel", "encoder": "learned", "kind": "cpc"},
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
                    talker_normalization=node.bool("talker_normalization", False),
                )
            )
        elif encoder == "learned":
            frontend = node.choice("frontend", "logmel", FRONTENDS)
            if frontend not in frontends.names:
                raise node.error("frontend", f"the front end {frontend!r} is not configured")
            kind = node.choice("kind", _MISSING, LEARNED_KINDS)
            result.append(
                LearnedEmbeddingConfig(
                    name=name,
                    kind=kind,
                    frontend=frontend,
                    dims=node.int("dims", 128 if kind == "contrastive" else 64, min=1),
                    hidden=node.int("hidden", 128, min=1),
                    layers=node.int("layers", 3, min=1),
                    kernel=node.int("kernel", 5, min=1),
                    epochs=node.int("epochs", 20 if kind == "contrastive" else 10, min=1),
                    batch_size=node.int("batch_size", 64 if kind == "contrastive" else 32, min=2),
                    learning_rate=node.number("learning_rate", 0.001, min=0, exclusive_min=True),
                    temperature=node.number("temperature", 0.1, min=0, exclusive_min=True),
                    steps_ahead=node.int("steps_ahead", 8, min=1),
                    negatives=node.int("negatives", 32, min=1),
                    train_on=node.choice("train_on", "clean", TRAINING_TOKENS),
                    embedding_from=node.choice("embedding_from", "context", CPC_EMBEDDINGS),
                    negatives_from=node.choice("negatives_from", "batch", CPC_NEGATIVES),
                    talker_normalization=node.bool("talker_normalization", False),
                )
            )
        else:
            result.append(
                PretrainedEmbeddingConfig(
                    name=name,
                    model=node.string("model", _MISSING),
                    layer=node.int("layer", DEFAULT_PRETRAINED_LAYER, min=0),
                    pooling=node.choice("pooling", "mean", POOLINGS),
                    store_layers=node.bool("store_layers", False),
                    talker_normalization=node.bool("talker_normalization", False),
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


def _read_shapes(node: _Node, default: dict[str, float], allowed: tuple[str, ...]):
    """The weights of the shapes of a function word or an affix, normalized to sum to 1, in the
    order of ``allowed``."""
    value = node.get("shapes", default)
    if not isinstance(value, dict) or not value:
        raise node.error("shapes", f"expected a mapping of weights, found {_describe(value)}")
    inner = _Node(node.source, node.field("shapes"), value)
    weights: dict[str, float] = {}
    for key, weight in value.items():
        if key not in allowed:
            raise inner.error(key, f"unknown shape; the shapes are {', '.join(allowed)}")
        weights[key] = inner.check_number(key, weight, min=0)
    total = sum(weights.values())
    if total <= 0:
        raise node.error("shapes", "at least one weight must be positive")
    return {k: weights[k] / total for k in allowed if k in weights}


def _gloss(source: str, field_name: str, value: Any) -> str:
    if isinstance(value, bool):
        word = "no" if value is False else "yes"
        raise ConfigError(
            source,
            field_name,
            f"expected a gloss, found {_describe(value)}; YAML reads a bare word such as "
            f'{word} as a boolean, so write it in quotes ("{word}")',
        )
    if not isinstance(value, str) or not value:
        raise ConfigError(source, field_name, f"expected a gloss, found {_describe(value)}")
    return value


def _read_glosses(source: str, field_name: str, value: Any) -> tuple[str, ...]:
    """The glosses of the function words: a list of distinct strings, which may be empty."""
    if not isinstance(value, list):
        raise ConfigError(source, field_name, f"expected a list, found {_describe(value)}")
    glosses = tuple(_gloss(source, f"{field_name}[{i}]", item) for i, item in enumerate(value))
    if len(set(glosses)) != len(glosses):
        raise ConfigError(source, field_name, "the glosses must be distinct")
    return glosses


def _read_affix_items(source: str, field_name: str, value: Any) -> tuple[AffixItem, ...]:
    if not isinstance(value, list):
        raise ConfigError(source, field_name, f"expected a list, found {_describe(value)}")
    items = []
    for i, item in enumerate(value):
        node = _Node(source, f"{field_name}[{i}]", item)
        gloss = _gloss(source, node.field("gloss"), node.get("gloss"))
        items.append(AffixItem(gloss, node.choice("position", "suffix", AFFIX_POSITIONS)))
        node.finish()
    if len({item.gloss for item in items}) != len(items):
        raise ConfigError(source, field_name, "the glosses must be distinct")
    return tuple(items)


def _read_inflect(source: str, field_name: str, value: Any) -> tuple[InflectEntry, ...]:
    if not isinstance(value, list):
        raise ConfigError(source, field_name, f"expected a list, found {_describe(value)}")
    entries = []
    for i, item in enumerate(value):
        node = _Node(source, f"{field_name}[{i}]", item)
        words = node.get("words")
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, str):
                    raise node.error("words", f"expected word labels, found {_describe(word)}")
            words = tuple(words)
        elif words not in ("all", "none"):
            raise node.error(
                "words", f"expected all, none, or a list of word labels, found {_describe(words)}"
            )
        affixes = node.get("affixes")
        if not isinstance(affixes, list) or not affixes:
            raise node.error(
                "affixes", f"expected a non-empty list of glosses, found {_describe(affixes)}"
            )
        glosses = tuple(
            _gloss(source, f"{node.field('affixes')}[{k}]", a) for k, a in enumerate(affixes)
        )
        entries.append(InflectEntry(words, glosses))
        node.finish()
    return tuple(entries)


def _read_request(path: str, config_source: str):
    """The closed-class request of a request file: the glosses of the function words, the affix
    items, and the inflect entries."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            config_source,
            "closed_class.request",
            f"cannot read the request file {path}: {error.strerror}",
        ) from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(path, "<file>", f"invalid YAML: {error}") from error
    node = _Node(path, "", {} if data is None else data)
    glosses = _read_glosses(path, "function_words", node.get("function_words", []))
    items = _read_affix_items(path, "affixes", node.get("affixes", []))
    inflect = _read_inflect(path, "inflect", node.get("inflect", []))
    node.finish()
    return glosses, items, inflect


def _read_closed_class(root: _Node, word_count: int) -> ClosedClassConfig | None:
    node = root.mapping("closed_class", nullable=True)
    if node is None:
        return None
    request = node.string("request", None, nullable=True)
    function_node = node.mapping("function_words")
    affix_node = node.mapping("affixes")
    default_items = [{"gloss": g, "position": p} for g, p in DEFAULT_AFFIXES]
    if request is not None:
        for owner, key in ((function_node, "glosses"), (affix_node, "items"), (node, "inflect")):
            if key in owner.data:
                raise owner.error(
                    key, "must not be given together with closed_class.request, which replaces it"
                )
        owner_source = request
        glosses, items, inflect = _read_request(request, root.source)
        inflect_field = "inflect"
    else:
        owner_source = root.source
        glosses = _read_glosses(
            root.source,
            function_node.field("glosses"),
            function_node.get("glosses", list(DEFAULT_FUNCTION_WORDS)),
        )
        items = _read_affix_items(
            root.source, affix_node.field("items"), affix_node.get("items", default_items)
        )
        inflect_field = node.field("inflect")
        inflect = _read_inflect(root.source, inflect_field, node.get("inflect", []))
    known = {item.gloss for item in items}
    for i, entry in enumerate(inflect):
        for gloss in entry.affixes:
            if gloss not in known:
                raise ConfigError(
                    owner_source,
                    f"{inflect_field}[{i}].affixes",
                    f"{gloss!r} is not the gloss of an affix",
                )
        if isinstance(entry.words, tuple):
            for label in entry.words:
                number = label[2:] if label.startswith("W.") else ""
                if not (number.isdecimal() and 1 <= int(number) <= word_count):
                    raise ConfigError(
                        owner_source,
                        f"{inflect_field}[{i}].words",
                        f"{label!r} is not the label of a content word (W.1 to W.{word_count})",
                    )
    affix_source = affix_node.choice("source", "pseudo", CLOSED_CLASS_SOURCES)
    if affix_source == "english":
        for i, item in enumerate(items):
            if item.position != "suffix":
                raise ConfigError(
                    owner_source,
                    f"{'affixes' if request else affix_node.field('items')}[{i}].position",
                    f"English has no {item.gloss} prefix; with source english, every affix is "
                    f"a suffix",
                )
    config = ClosedClassConfig(
        glosses=glosses,
        function_source=function_node.choice("source", "pseudo", CLOSED_CLASS_SOURCES),
        function_shapes=_read_shapes(function_node, DEFAULT_FUNCTION_SHAPES, FUNCTION_SHAPES),
        min_distance=function_node.int("min_distance", 2, min=1),
        affixes=items,
        affix_source=affix_source,
        affix_shapes=_read_shapes(affix_node, DEFAULT_AFFIX_SHAPES, AFFIX_SHAPES),
        epenthesis=affix_node.bool("epenthesis", True),
        glide=affix_node.choice("glide", "Y", GLIDES),
        max_skipped=affix_node.probability("max_skipped", 0.1),
        inflect=inflect,
        request=request,
    )
    function_node.finish()
    affix_node.finish()
    node.finish()
    return config


def _optional_range(node: _Node, key: str, *, min: float | None = None, exclusive_min=False):
    value = node.get(key, None, nullable=True)
    if value is None:
        return None
    return node.range(key, value, min=min, exclusive_min=exclusive_min)


def _section(node: _Node, key: str) -> _Node | None:
    """A recipe's transformation section: absent or null means the transformation is off."""
    if key not in node.data:
        return None
    return node.mapping(key, nullable=True)


def _read_recipe(node: _Node) -> RecipeConfig:
    name = node.string("name", _MISSING)
    noise_node = _section(node, "noise")
    noise = None
    if noise_node is not None:
        kinds = noise_node.strings("kinds", ["white", "pink", "speech", "babble"])
        for kind in kinds:
            if kind not in NOISE_KINDS:
                raise noise_node.error(
                    "kinds", f"unknown noise {kind!r}; the kinds are {', '.join(NOISE_KINDS)}"
                )
        noise = NoiseConfig(
            kinds=kinds,
            snr_db=noise_node.range("snr_db", [0, 20]),
            babble_voices=noise_node.int("babble_voices", 6, min=1),
        )
        noise_node.finish()
    reverb_node = _section(node, "reverberation")
    reverberation = None
    if reverb_node is not None:
        reverberation = ReverberationConfig(
            rt60=reverb_node.range("rt60", [0.2, 0.8], min=0.05),
            room_m=reverb_node.range("room_m", [3, 6], min=1.5),
        )
        reverb_node.finish()
    speed_node = _section(node, "speed_pitch")
    speed_pitch = None
    if speed_node is not None:
        speed_pitch = SpeedPitchConfig(
            speed=speed_node.range("speed", [0.9, 1.1], min=0, exclusive_min=True),
            pitch_semitones=speed_node.range("pitch_semitones", [-2, 2]),
        )
        speed_node.finish()
    manipulation_node = _section(node, "manipulation")
    manipulation = None
    if manipulation_node is not None:
        manipulation = ManipulationConfig(
            pitch_median_hz=_optional_range(
                manipulation_node, "pitch_median_hz", min=0, exclusive_min=True
            ),
            pitch_range_factor=_optional_range(manipulation_node, "pitch_range_factor", min=0),
            formant_shift_ratio=_optional_range(
                manipulation_node, "formant_shift_ratio", min=0, exclusive_min=True
            ),
            duration_factor=_optional_range(
                manipulation_node, "duration_factor", min=0, exclusive_min=True
            ),
        )
        manipulation_node.finish()
        if all(v is None for v in manipulation.resolved().values()):
            raise node.error("manipulation", "sets nothing; give at least one range")
    node.finish()
    if noise is None and reverberation is None and speed_pitch is None and manipulation is None:
        raise node.error("name", f"the recipe {name!r} applies no transformation")
    return RecipeConfig(name, noise, reverberation, speed_pitch, manipulation)


def _read_augmentation(root: _Node) -> AugmentationConfig | None:
    """The augmentation section; absent or null means no augmented tokens."""
    if root.get("augmentation", None, nullable=True) is None:
        return None
    node = root.mapping("augmentation")
    value = node.get("recipes", _MISSING)
    if not isinstance(value, list) or not value:
        raise node.error("recipes", f"expected a non-empty list, found {_describe(value)}")
    recipes = []
    for i, item in enumerate(value):
        recipes.append(_read_recipe(_Node(node.source, node.field(f"recipes[{i}]"), item)))
    names = [r.name for r in recipes]
    if len(set(names)) != len(names):
        raise node.error("recipes", "the recipe names must be distinct")
    config = AugmentationConfig(
        recipes=tuple(recipes),
        proportion=node.probability("proportion", 1.0),
        speakers=node.choice("speakers", "all", AUGMENTED_SPEAKERS),
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
    frontends = _read_frontends(root.mapping("frontends"), synthesis.sample_rate)
    embeddings = _read_embeddings(root, frontends)
    closed_class = _read_closed_class(root, wordforms.count)
    augmentation = _read_augmentation(root)
    word_node = root.mapping("word_embeddings")
    word_embeddings = WordEmbeddingsConfig(
        tokens=word_node.choice("tokens", "clean", WORD_EMBEDDING_TOKENS)
    )
    word_node.finish()
    training_node = root.mapping("training")
    training = TrainingConfig(
        held_out_word_proportion=training_node.probability("held_out_word_proportion", 0.2)
    )
    training_node.finish()
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
        closed_class=closed_class,
        augmentation=augmentation,
        word_embeddings=word_embeddings,
        training=training,
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

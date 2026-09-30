"""Speakers: a configured number for each engine, with a held-out proportion.

Piper speakers are speaker IDs of a multi-speaker voice. espeak-ng speakers are a voice variant
with a pitch and a rate drawn from the configured ranges. Each engine draws from its own
substream of ``wordforms:speakers``, so changing one engine's speakers never changes the other's.
Held-out speakers never train a learned encoder and never contribute to word embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semantic_world.wordforms.config import Config, _round_half_up
from semantic_world.wordforms.streams import Streams


@dataclass(frozen=True)
class Speaker:
    label: str
    engine: str
    voice: str
    speaker_id: int | None = None
    """The speaker ID within a Piper voice."""
    variant: str | None = None
    """The espeak-ng voice variant, for example ``m3``."""
    pitch: int | None = None
    rate: int | None = None
    """The espeak-ng pitch (0 to 99) and rate (words per minute)."""
    held_out: bool = False

    @property
    def identity(self) -> str:
        """What the speaker sounds like, without the label: the key of cached audio."""
        if self.engine == "piper":
            return f"piper/{self.voice}/{self.speaker_id}"
        return f"espeak/{self.voice}+{self.variant}/p{self.pitch}/s{self.rate}"

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "engine": self.engine,
            "voice": self.voice,
            "speaker_id": self.speaker_id,
            "variant": self.variant,
            "pitch": self.pitch,
            "rate": self.rate,
            "split": "held_out" if self.held_out else "train",
        }


def held_out_count(count: int, proportion: float) -> int:
    """The number of held-out speakers among ``count``: the proportion, rounded half up, leaving
    at least one training speaker."""
    return min(_round_half_up(proportion * count), count - 1) if count > 0 else 0


def draw_speakers(
    config: Config, streams: Streams, piper_voice_speakers: int | None = None
) -> list[Speaker]:
    """The speakers of a run, Piper's first and then espeak-ng's, labeled ``S.1``, ``S.2``, ...
    ``piper_voice_speakers`` is the number of speakers in the Piper voice."""
    synthesis = config.synthesis
    drawn: list[dict[str, Any]] = []
    if synthesis.piper is not None:
        settings = synthesis.piper
        if piper_voice_speakers is None:
            raise ValueError("the number of speakers in the Piper voice is needed")
        if settings.speakers > piper_voice_speakers:
            raise ValueError(
                f"{config.source}: synthesis.engines.piper.speakers: the voice {settings.voice} "
                f"has {piper_voice_speakers} speakers, fewer than {settings.speakers}"
            )
        rng = streams.substream("speakers", "piper")
        ids = rng.choice(piper_voice_speakers, size=settings.speakers, replace=False)
        held = _held_out(rng, settings.speakers, synthesis.held_out_speaker_proportion)
        for i, speaker_id in enumerate(ids):
            drawn.append(
                dict(
                    engine="piper",
                    voice=settings.voice,
                    speaker_id=int(speaker_id),
                    held_out=i in held,
                )
            )
    if synthesis.espeak is not None:
        settings = synthesis.espeak
        rng = streams.substream("speakers", "espeak")
        voices = []
        for i in range(settings.speakers):
            voices.append(
                dict(
                    engine="espeak",
                    voice=settings.voice,
                    variant=settings.variants[i % len(settings.variants)],
                    pitch=int(rng.integers(settings.pitch[0], settings.pitch[1] + 1)),
                    rate=int(rng.integers(settings.rate[0], settings.rate[1] + 1)),
                )
            )
        held = _held_out(rng, settings.speakers, synthesis.held_out_speaker_proportion)
        for i, voice in enumerate(voices):
            drawn.append({**voice, "held_out": i in held})
    return [Speaker(label=f"S.{i + 1}", **fields) for i, fields in enumerate(drawn)]


def _held_out(rng, count: int, proportion: float) -> set[int]:
    k = held_out_count(count, proportion)
    if k == 0:
        return set()
    return {int(i) for i in rng.choice(count, size=k, replace=False)}

"""Errors raised while generating a corpus."""

from __future__ import annotations


class CorpusError(RuntimeError):
    """The corpus generator cannot go on, for example because a taxonomy output folder does not
    match the taxonomy regenerated from its configuration. Configuration errors are
    :class:`~semantic_world.corpus.config.ConfigError` instead."""

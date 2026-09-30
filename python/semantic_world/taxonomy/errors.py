"""Errors raised while generating a taxonomy."""

from __future__ import annotations


class GenerationError(RuntimeError):
    """The generator could not satisfy the configuration.

    The message reports what was tried, how close the generator came, and which parameters to
    loosen. Configuration errors are :class:`~semantic_world.taxonomy.config.ConfigError` instead.
    """

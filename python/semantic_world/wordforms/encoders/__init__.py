"""Layer 4 encoders: each turns a clip into one vector.

- ``fixed``: a front end's frames averaged within equal time bins, with an optional principal
  component projection fitted on training-speaker tokens.
- ``pretrained``: a pretrained speech model from Hugging Face ``transformers``, mean-pooled.

Learned encoders arrive in stage 6.
"""

from semantic_world.wordforms.encoders.fixed import FixedEncoder, fit_pca, time_bin_means

__all__ = ["FixedEncoder", "fit_pca", "time_bin_means"]

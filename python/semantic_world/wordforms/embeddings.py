"""Layer 4: sound embeddings. One vector per token and one vector per word.

Each embedding configuration names an encoder. The encoder gives one vector per token. A word's
embedding is the mean of its tokens' embeddings over training speakers; held-out speakers never
contribute. Token embeddings are kept too, because token variability is part of what models
should face.

A run stores each embedding under ``embeddings/<name>/``: ``tokens.npy`` (tokens by dimensions),
``types.npy`` (words by dimensions), and ``meta.yaml``. A fixed encoder with a projection also
stores ``projection.npz``. A pretrained encoder stores its configured layer only; with
``store_layers: true`` it also stores ``layers.npy`` (tokens by layers by dimensions), the
pooled output of every layer. The evaluation's layer sweep does not need that file.

:class:`SoundEmbeddings` is the interface for other models: the arrays, the word table, and
``embed`` for new word forms.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from semantic_world.wordforms.config import (
    Config,
    EmbeddingConfig,
    FixedEmbeddingConfig,
    LearnedEmbeddingConfig,
    load_config,
)
from semantic_world.wordforms.encoders.fixed import FixedEncoder, Projection, fit_pca
from semantic_world.wordforms.frontends import FrontendStore, make_frontends
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth import MAPPED_SUFFIX, Synthesis
from semantic_world.wordforms.synth import audio as audio_tools
from semantic_world.wordforms.synth.speakers import Speaker


@dataclass
class EmbeddingStore:
    """A stored embedding: token vectors, word vectors, and settings."""

    folder: Path
    meta: dict[str, Any]
    reused: bool = False

    @classmethod
    def load(cls, folder: str | Path) -> EmbeddingStore:
        folder = Path(folder)
        return cls(folder, yaml.safe_load((folder / "meta.yaml").read_text(encoding="utf-8")))

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def tokens(self) -> np.ndarray:
        return np.load(self.folder / "tokens.npy")

    @property
    def types(self) -> np.ndarray:
        return np.load(self.folder / "types.npy")

    @property
    def speaker_means(self) -> np.ndarray | None:
        """Each speaker's mean token embedding, when the embedding is talker-normalized: the
        stored tokens are the encoder's output less their speaker's mean."""
        path = self.folder / "speaker_means.npy"
        return np.load(path) if path.exists() else None

    @property
    def layers(self) -> np.ndarray | None:
        """A pretrained encoder's pooled output for every layer, memory-mapped: tokens by layers
        by dimensions. None unless the embedding was stored with ``store_layers``."""
        path = self.folder / "layers.npy"
        return np.load(path, mmap_mode="r") if path.exists() else None

    def summary(self) -> dict[str, Any]:
        keys = ("encoder", "pretrained", "dims", "tokens", "words")
        summary = {**{k: self.meta[k] for k in keys}, "reused": self.reused}
        if "training" in self.meta:
            summary["training"] = self.meta["training"]
        return summary


def token_layout(words: list[WordForm], synthesis: Synthesis):
    """For each token, the index of its word and of its speaker, and whether its speaker is a
    training speaker."""
    word_index = {word.label: i for i, word in enumerate(words)}
    speaker_index = {speaker.label: i for i, speaker in enumerate(synthesis.speakers)}
    held_out = np.array([speaker.held_out for speaker in synthesis.speakers], dtype=bool)
    token_words = np.array([word_index[t.word] for t in synthesis.tokens], dtype=np.int64)
    token_speakers = np.array([speaker_index[t.speaker] for t in synthesis.tokens], dtype=np.int64)
    return token_words, token_speakers, ~held_out[token_speakers]


def word_embedding_tokens(config: Config, synthesis: Synthesis, train: np.ndarray) -> np.ndarray:
    """Which tokens make the word embeddings: the training-speaker tokens, without the augmented
    ones unless ``word_embeddings.tokens`` is ``all``, and never a control token (the unmapped
    original of a mapped word's token)."""
    if config.word_embeddings.tokens == "all":
        control = np.array([t.control for t in synthesis.tokens], dtype=bool)
        return train & ~control
    clean = np.array([t.clean for t in synthesis.tokens], dtype=bool)
    return train & clean


def training_word_tokens(words: list[WordForm], token_words: np.ndarray) -> np.ndarray:
    """Which tokens are of training words: the words that are not held out from trained
    encoders."""
    training = np.array([not word.held_out for word in words], dtype=bool)
    return training[token_words]


def normalization_basis(words: list[WordForm], synthesis: Synthesis, token_words: np.ndarray):
    """The tokens that a speaker's mean is taken over for talker normalization: the clean tokens
    of the training content words. Content words only, so that closed-class forms never change
    a content word's embedding."""
    clean = np.array([t.clean for t in synthesis.tokens], dtype=bool)
    content = np.array([word.kind == "content" for word in words], dtype=bool)
    return clean & content[token_words] & training_word_tokens(words, token_words)


def speaker_means(tokens: np.ndarray, token_speakers: np.ndarray, basis: np.ndarray, count: int):
    """Each speaker's mean token embedding over the ``basis`` tokens (the speaker's clean tokens
    of training words); zeros for a speaker without such a token."""
    sums = np.zeros((count, tokens.shape[1]), dtype=np.float64)
    totals = np.zeros(count, dtype=np.int64)
    np.add.at(sums, token_speakers[basis], np.asarray(tokens, dtype=np.float64)[basis])
    np.add.at(totals, token_speakers[basis], 1)
    return (sums / np.maximum(totals, 1)[:, None]).astype(np.float32)


def word_means(tokens: np.ndarray, token_words: np.ndarray, chosen: np.ndarray, count: int):
    """Each word's embedding: the mean of its ``chosen`` tokens."""
    types = np.zeros((count, tokens.shape[1]), dtype=np.float64)
    totals = np.zeros(count, dtype=np.int64)
    np.add.at(types, token_words[chosen], np.asarray(tokens, dtype=np.float64)[chosen])
    np.add.at(totals, token_words[chosen], 1)
    if (totals == 0).any():
        raise ValueError("a word has no tokens from training speakers")
    return (types / totals[:, None]).astype(np.float32)


def training_seed(config: Config, embedding: LearnedEmbeddingConfig) -> int:
    """The seed of a learned encoder's training: the ``wordforms:train`` stream, by name."""
    from semantic_world.wordforms.streams import Streams

    return int(Streams(config.seed).substream("train", embedding.name).integers(2**31 - 1))


def fingerprint(embedding: EmbeddingConfig, synthesis: Synthesis, extra: dict[str, Any]) -> str:
    """A hash of the embedding's settings, of what the embedding is computed from, and of every
    token's label, speaker split, and audio."""
    digest = hashlib.sha256()
    digest.update(json.dumps(embedding.resolved(), sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    held_out = {speaker.label: speaker.held_out for speaker in synthesis.speakers}
    for token in synthesis.tokens:
        digest.update(f"{token.label}:{token.sha256}:{held_out[token.speaker]}\n".encode())
    return digest.hexdigest()


def compute_embedding(
    config: Config,
    embedding: EmbeddingConfig,
    words: list[WordForm],
    synthesis: Synthesis,
    frontends: dict[str, FrontendStore] | None,
    folder: str | Path,
    progress: Callable[[int, int], None] | None = None,
    local_only: bool = False,
) -> EmbeddingStore:
    """Compute one embedding for every token and word and store it in ``folder``. When the
    folder already holds the same embedding for the same audio, nothing is computed."""
    folder = Path(folder)
    fixed = isinstance(embedding, FixedEmbeddingConfig)
    learned = isinstance(embedding, LearnedEmbeddingConfig)
    source: dict[str, Any]
    if fixed or learned:
        frontend = make_frontends(config)[embedding.frontend]
        source = frontend.meta()
    else:
        source = {"sample_rate": config.synthesis.sample_rate}
    source["word_embedding_tokens"] = config.word_embeddings.tokens
    if learned or (fixed and embedding.pca_dims is not None) or embedding.talker_normalization:
        # a trained encoder, and the speaker means, depend on which words are held out
        source["held_out_words"] = [word.label for word in words if word.held_out]
    if learned:
        source["train_seed"] = training_seed(config, embedding)
        source["device"] = "cpu" if config.device == "cpu" else "accelerator"
    mark = fingerprint(embedding, synthesis, source)
    keep_layers = not fixed and not learned and embedding.store_layers
    files = ["tokens.npy", "types.npy", "meta.yaml"] + (["layers.npy"] if keep_layers else [])
    if learned:
        files.append("model.pt")
    if all((folder / name).exists() for name in files):
        store = EmbeddingStore.load(folder)
        if store.meta.get("fingerprint") == mark:
            store.reused = True
            return store
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.yaml").unlink(missing_ok=True)  # an unfinished write is never up to date
    token_words, token_speakers, train = token_layout(words, synthesis)
    chosen = word_embedding_tokens(config, synthesis, train)
    training_words = training_word_tokens(words, token_words)
    count = len(synthesis.tokens)
    meta: dict[str, Any] = {
        "name": embedding.name,
        "encoder": embedding.encoder,
        "pretrained": not fixed and not learned,
        "settings": embedding.resolved(),
    }
    if learned:
        from semantic_world.wordforms.device import resolve_device
        from semantic_world.wordforms.encoders import learned as learned_encoders

        stored = (frontends or {}).get(embedding.frontend)
        cache: dict[int, np.ndarray] = {}

        def frames_of(index: int) -> np.ndarray:
            if stored is not None:
                return stored.token_frames(synthesis.tokens[index].label)
            if index not in cache:
                cache[index] = frontend.compute(synthesis.audio(synthesis.tokens[index]))
            return cache[index]

        clean = np.array([t.clean for t in synthesis.tokens], dtype=bool)
        control = np.array([t.control for t in synthesis.tokens], dtype=bool)
        trainable = train & ~control if embedding.train_on == "all" else train & clean
        # Inflected forms never train an encoder, so the inflected forms that a run asks for
        # never change the embedding of a content word or of a function word.
        uninflected = np.array([word.kind != "inflected" for word in words], dtype=bool)
        train_indices = np.flatnonzero(trainable & training_words & uninflected[token_words])
        mean, std = learned_encoders.frame_statistics(frames_of, train_indices)
        frame_source = learned_encoders.FrameSource(frames_of, mean, std)
        device = resolve_device(config.device)
        model, report = learned_encoders.train_encoder(
            embedding.kind,
            embedding.settings(),
            frame_source,
            token_words,
            train_indices,
            training_seed(config, embedding),
            device,
            None if progress is None else (lambda e, n, loss: progress(e, n)),
        )
        learned_encoders.save_model(
            folder / "model.pt", model, embedding.kind, embedding.settings(), mean, std
        )
        # The tokens of inflected forms are encoded in batches of their own, for the same
        # reason: a batch pads its clips to the longest one.
        of_inflected = ~uninflected[token_words]
        groups = [np.flatnonzero(~of_inflected), np.flatnonzero(of_inflected)]
        tokens = learned_encoders.encode_tokens(model, frame_source, count, device, groups=groups)
        meta["frontend"] = embedding.frontend
        meta["learned"] = True
        meta["training"] = report.as_dict()
        meta["device"] = device
    elif fixed:
        encoder = FixedEncoder(frontend, embedding.time_bins)
        stored = (frontends or {}).get(embedding.frontend)
        features = np.zeros((count, encoder.feature_dims), dtype=np.float32)
        for i, token in enumerate(synthesis.tokens):
            if stored is not None:
                frames = stored.token_frames(token.label)
            else:
                frames = frontend.compute(synthesis.audio(token))
            features[i] = encoder.features(frames)
            if progress is not None:
                progress(i + 1, count)
        (folder / "projection.npz").unlink(missing_ok=True)
        if embedding.pca_dims is not None:
            # The projection is fitted on the content words, so that closed-class forms never
            # change a content word's embedding.
            content = np.array([word.kind == "content" for word in words], dtype=bool)
            clean = np.array([t.clean for t in synthesis.tokens], dtype=bool)
            fitted = train & content[token_words] & clean & training_words
            projection = fit_pca(features[fitted], embedding.pca_dims)
            projection.save(folder / "projection.npz")
            tokens = projection.apply(features)
            total = float(np.var(features[fitted].astype(np.float64), axis=0, ddof=1).sum())
            kept = float(projection.explained_variance.sum())
            meta["projection"] = {
                "kind": "pca",
                "fitted_on": "training-speaker tokens"
                + ("" if content.all() else " of content words")
                + ("" if clean.all() else ", without augmented tokens")
                + ("" if training_words.all() else ", without held-out words"),
                "fitted_tokens": int(fitted.sum()),
                "dims": projection.dims,
                "explained_variance": round(kept / total, 6) if total > 0 else None,
            }
        else:
            tokens = features
        meta["frontend"] = embedding.frontend
        meta["feature_dims"] = encoder.feature_dims
    else:
        from semantic_world.wordforms.encoders.pretrained import PretrainedEncoder

        encoder = PretrainedEncoder(
            embedding.model,
            embedding.layer,
            embedding.pooling,
            config.device,
            config.synthesis.sample_rate,
            local_only,
        )
        # The file of every layer's output belongs to an earlier computation of this embedding.
        (folder / "layers.npy").unlink(missing_ok=True)
        layers = None
        if keep_layers:
            layers = np.lib.format.open_memmap(
                folder / "layers.npy",
                mode="w+",
                dtype=np.float32,
                shape=(count, encoder.layers, encoder.dims),
            )
        tokens = np.zeros((count, encoder.dims), dtype=np.float32)
        for i, token in enumerate(synthesis.tokens):
            pooled = encoder.all_layers(synthesis.audio(token))
            tokens[i] = pooled[embedding.layer]
            if layers is not None:
                layers[i] = pooled
            if progress is not None:
                progress(i + 1, count)
        if layers is not None:
            layers.flush()
            del layers
        meta.update(encoder.provenance())
        meta["layers"] = encoder.layers
        meta["layer"] = embedding.layer
        meta["layers_stored"] = keep_layers
        meta["device"] = encoder.device
    tokens = np.ascontiguousarray(tokens, dtype=np.float32)
    (folder / "speaker_means.npy").unlink(missing_ok=True)
    if embedding.talker_normalization:
        # Each speaker's mean over the speaker's clean tokens of training content words is
        # subtracted from every token of that speaker. The means are stored for new forms.
        basis = normalization_basis(words, synthesis, token_words)
        means = speaker_means(tokens, token_speakers, basis, len(synthesis.speakers))
        np.save(folder / "speaker_means.npy", means)
        tokens = np.ascontiguousarray(tokens - means[token_speakers], dtype=np.float32)
    meta["talker_normalization"] = embedding.talker_normalization
    meta["held_out_words"] = int(sum(word.held_out for word in words))
    np.save(folder / "tokens.npy", tokens)
    np.save(folder / "types.npy", word_means(tokens, token_words, chosen, len(words)))
    meta.update(
        {
            "dims": int(tokens.shape[1]),
            "tokens": count,
            "words": len(words),
            "word_embedding": "mean over training-speaker tokens"
            + (", without augmented tokens" if config.word_embeddings.tokens == "clean" else ""),
            "word_embedding_tokens": config.word_embeddings.tokens,
            "fingerprint": mark,
        }
    )
    (folder / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return EmbeddingStore(folder, meta)


def compute_embeddings(
    config: Config,
    words: list[WordForm],
    synthesis: Synthesis,
    frontends: dict[str, FrontendStore] | None,
    folder: str | Path,
    progress: Callable[[str, int, int], None] | None = None,
    local_only: bool = False,
) -> dict[str, EmbeddingStore]:
    """Compute and store every configured embedding under ``folder/embeddings/<name>/``."""
    stores: dict[str, EmbeddingStore] = {}
    for embedding in config.embeddings:
        name = embedding.name
        report = None if progress is None else (lambda i, n, name=name: progress(name, i, n))
        stores[name] = compute_embedding(
            config,
            embedding,
            words,
            synthesis,
            frontends,
            Path(folder) / "embeddings" / name,
            report,
            local_only,
        )
    return stores


# ---------------------------------------------------------------------------------------------
# The interface for other models
# ---------------------------------------------------------------------------------------------


TOKEN_TEXT_COLUMNS = {"augmentation": pl.String, "achieved": pl.String, "mapping": pl.String}
"""Columns of ``tokens.csv`` that are empty in most runs, and text when they are not."""


def _text_column(table: pl.DataFrame, name: str) -> np.ndarray:
    """A text column of the token table, with empty strings for missing values; all empty when a
    run folder from before the column has none."""
    if name not in table.columns:
        return np.full(table.height, "", dtype=object)
    return table[name].fill_null("").to_numpy()


class SoundEmbeddings:
    """The sound embeddings of a run, for use in other models.

    - ``types``: word embeddings, one row per word, in the order of ``words``;
    - ``tokens``: token embeddings, with ``token_words`` and ``token_speakers`` giving each
      token's row in ``words`` and in ``speakers``, ``token_held_out`` marking the tokens of
      held-out speakers, ``token_augmented`` marking augmented tokens, and ``token_mapped``
      marking tokens changed by acoustic mapping. With acoustic mapping, a mapped word's tokens
      are its mapped tokens. The unmapped originals are a control set, and they are left out of
      every array here: ``control`` holds them, with the same fields, for comparison;
    - ``words``, ``speakers``: the word table and the speaker table. The word table holds the
      content words, then the function words, then the inflected forms, and ``word_kinds`` gives
      the kind of each row;
    - ``embed``: embeddings for new word forms, through the same synthesis, front end, and
      frozen encoder;
    - ``to_torch``: the arrays as PyTorch tensors.

    Load one with ``SoundEmbeddings.load(run_folder, name)``. The run's configuration gives the
    audio cache and the Piper voice as paths, which are read from the current folder when they
    are relative, as in the run itself.
    """

    def __init__(
        self, folder: str | Path, name: str, local_only: bool = False, engines=None
    ) -> None:
        self.folder = Path(folder)
        self.name = name
        self.local_only = local_only
        self.engines = engines
        """The synthesis engines for new word forms. None makes the configured engines when a
        clip has to be synthesized."""
        self.config = load_config(self.folder / "config.yaml")
        self.store = EmbeddingStore.load(self.folder / "embeddings" / name)
        self.meta = self.store.meta
        self.words = pl.read_csv(self.folder / "words.csv")
        self.speakers = pl.read_csv(self.folder / "speakers.csv")
        table = pl.read_csv(self.folder / "tokens.csv", schema_overrides=TOKEN_TEXT_COLUMNS)
        self.types = self.store.types
        word_index = {label: i for i, label in enumerate(self.words["label"])}
        speaker_index = {label: i for i, label in enumerate(self.speakers["label"])}
        held_out = (self.speakers["split"] == "held_out").to_numpy()

        def arrays(rows: np.ndarray | None) -> dict[str, Any]:
            part = table if rows is None else table[rows]
            tokens = self.store.tokens if rows is None else self.store.tokens[rows]
            token_speakers = np.array([speaker_index[s] for s in part["speaker"]], dtype=np.int64)
            return {
                "token_table": part,
                "tokens": tokens,
                "token_words": np.array([word_index[w] for w in part["word"]], dtype=np.int64),
                "token_speakers": token_speakers,
                "token_held_out": held_out[token_speakers],
                "token_augmented": part["augmentation"].fill_null("").to_numpy() != "",
                "token_mapped": _text_column(part, "mapping") != "",
            }

        # A run with acoustic mapping keeps the unmapped originals of the mapped tokens as a
        # control set. They are no part of what a learner gets, so they are held apart.
        control = (
            table["control"].fill_null(False).to_numpy()
            if "control" in table.columns
            else np.zeros(table.height, dtype=bool)
        )
        own = arrays(None if not control.any() else np.flatnonzero(~control))
        self.token_table = own["token_table"]
        self.tokens = own["tokens"]
        self.token_words = own["token_words"]
        self.token_speakers = own["token_speakers"]
        self.token_held_out = own["token_held_out"]
        self.token_augmented = own["token_augmented"]
        self.token_mapped = own["token_mapped"]
        self.control: dict[str, Any] = arrays(np.flatnonzero(control))
        """The control tokens of a run with acoustic mapping (the unmapped originals of the
        mapped tokens): ``token_table``, ``tokens``, ``token_words``, ``token_speakers``, and
        ``token_held_out``. Empty arrays in any other run."""
        self._encoder: Any = None
        self._lexicon: dict[str, str] = {}
        for label, arpabet in self.words.select("label", "arpabet").iter_rows():
            self._lexicon.setdefault(arpabet, label)  # of two forms that sound alike, the first
        # the word's own tokens by word, speaker, and token number (a mapped token has its
        # source's number); augmented tokens are not a form's own recording, so ``embed`` never
        # returns them
        self._stored = {
            (word, speaker, int(label.removesuffix(MAPPED_SUFFIX).rsplit(".", 1)[1])): i
            for i, (label, word, speaker) in enumerate(
                self.token_table.select("label", "word", "speaker").iter_rows()
            )
            if not self.token_augmented[i]
        }

    @classmethod
    def load(
        cls, folder: str | Path, name: str, local_only: bool = False, engines=None
    ) -> SoundEmbeddings:
        """The embedding ``name`` of the run in ``folder``. ``local_only`` keeps a pretrained
        model from being downloaded."""
        return cls(folder, name, local_only, engines)

    @staticmethod
    def names(folder: str | Path) -> list[str]:
        """The names of the embeddings stored in a run's folder."""
        root = Path(folder) / "embeddings"
        return sorted(p.name for p in root.iterdir() if (p / "meta.yaml").exists())

    @property
    def dims(self) -> int:
        return int(self.types.shape[1])

    @property
    def pretrained(self) -> bool:
        return bool(self.meta["pretrained"])

    # Speakers and forms ------------------------------------------------------------------------

    def speaker_list(self, speakers=None) -> list[Speaker]:
        """Speakers from labels, :class:`Speaker` objects, or None for the training speakers."""
        table = {row["label"]: row for row in self.speakers.iter_rows(named=True)}
        if speakers is None:
            speakers = [label for label, row in table.items() if row["split"] == "train"]
        result = []
        for speaker in speakers:
            if isinstance(speaker, Speaker):
                result.append(speaker)
                continue
            if speaker not in table:
                raise ValueError(f"unknown speaker {speaker!r}")
            row = table[speaker]
            result.append(
                Speaker(
                    label=row["label"],
                    engine=row["engine"],
                    voice=row["voice"],
                    speaker_id=row["speaker_id"],
                    variant=row["variant"],
                    pitch=row["pitch"],
                    rate=row["rate"],
                    held_out=row["split"] == "held_out",
                )
            )
        return result

    def lexicon_forms(self) -> list[WordForm]:
        """The words of the lexicon as :class:`WordForm` objects, in the order of ``words``."""
        from semantic_world.wordforms.generate import word_form_from_arpabet

        forms = []
        for row in self.words.iter_rows(named=True):
            settings = self.config.wordforms
            form = word_form_from_arpabet(
                row["label"],
                row["arpabet"],
                settings.english_min_zipf,
                settings.exclude_inflections,
            )
            form.spelling = row["spelling"]
            # a run from before the closed-class forms has content words only
            form.kind = row.get("kind") or "content"
            form.gloss = row.get("gloss")
            form.stem = row.get("stem")
            form.affix = row.get("affix")
            form.join = row.get("join")
            weak = row.get("weak_forms")
            form.weak_forms = tuple(weak.split("; ")) if weak else ()
            form.held_out = row.get("split") == "held_out"
            forms.append(form)
        return forms

    @property
    def word_held_out(self) -> np.ndarray:
        """Which words were held out from the training of every trained encoder."""
        if "split" not in self.words.columns:
            return np.zeros(len(self.words), dtype=bool)
        return (self.words["split"] == "held_out").to_numpy()

    @property
    def word_kinds(self) -> np.ndarray:
        """The kind of each word: ``content``, ``function``, or ``inflected``."""
        if "kind" not in self.words.columns:
            return np.full(len(self.words), "content")
        return self.words["kind"].to_numpy()

    def word_forms(self, forms) -> list[WordForm]:
        """Word forms from :class:`WordForm` objects or ARPAbet strings such as ``K AE1 T``."""
        from semantic_world.wordforms.generate import word_form_from_arpabet

        result = []
        for i, form in enumerate(forms):
            if isinstance(form, WordForm):
                result.append(form)
            else:
                settings = self.config.wordforms
                result.append(
                    word_form_from_arpabet(
                        f"N.{i + 1}", form, settings.english_min_zipf, settings.exclude_inflections
                    )
                )
        return result

    # Encoding ----------------------------------------------------------------------------------

    @property
    def encoder(self):
        """The frozen encoder of this embedding."""
        if self._encoder is None:
            settings = self.meta["settings"]
            if self.meta["encoder"] == "learned":
                from semantic_world.wordforms.device import resolve_device
                from semantic_world.wordforms.encoders.learned import LearnedEncoder

                frontend = make_frontends(self.config)[settings["frontend"]]
                self._encoder = LearnedEncoder(
                    frontend, self.store.folder / "model.pt", resolve_device(self.config.device)
                )
            elif self.meta["encoder"] == "fixed":
                frontend = make_frontends(self.config)[settings["frontend"]]
                path = self.store.folder / "projection.npz"
                projection = Projection.load(path) if path.exists() else None
                self._encoder = FixedEncoder(frontend, settings["time_bins"], projection)
            else:
                from semantic_world.wordforms.encoders.pretrained import PretrainedEncoder

                self._encoder = PretrainedEncoder(
                    settings["model"],
                    settings["layer"],
                    settings["pooling"],
                    self.config.device,
                    self.config.synthesis.sample_rate,
                    self.local_only,
                )
        return self._encoder

    def clips(self, forms, speakers=None, token: int = 1) -> list[list[np.ndarray]]:
        """The audio of token ``token`` of each form by each speaker. A clip comes from the audio
        cache when it is there: a stored token of the run through the token table, and any other
        clip through its cache key. Only a clip that is not in the cache is synthesized."""
        from semantic_world.wordforms.streams import Streams
        from semantic_world.wordforms.synth import synthesize_lexicon

        forms = self.word_forms(forms)
        speakers = self.speaker_list(speakers)
        cache = Path(self.config.synthesis.cache_dir)
        result: list[list[np.ndarray | None]] = [[None] * len(speakers) for _ in forms]
        for i, form in enumerate(forms):
            word = self._lexicon.get(form.arpabet)
            missing = []
            for j, speaker in enumerate(speakers):
                row = self._stored.get((word, speaker.label, token)) if word else None
                if row is not None:
                    path = cache / self.token_table["cache_path"][row]
                    result[i][j], _ = audio_tools.read_flac(path)
                else:
                    missing.append(j)
            if missing:
                if token > self.config.synthesis.tokens_per_speaker:
                    raise ValueError(
                        f"token {token} is beyond tokens_per_speaker "
                        f"({self.config.synthesis.tokens_per_speaker})"
                    )
                synthesis = synthesize_lexicon(
                    self.config,
                    Streams(self.config.seed),
                    [form],
                    engines=self.engines,
                    speakers=[speakers[j] for j in missing],
                    check=False,
                )
                wanted = {t.speaker: t for t in synthesis.tokens if t.label.endswith(f".{token}")}
                for j in missing:
                    result[i][j] = synthesis.audio(wanted[speakers[j].label])
        return result

    def embed(self, forms, speakers=None, token: int = 1) -> np.ndarray:
        """Token embeddings for word forms: an array of forms by speakers by dimensions.

        ``forms`` are :class:`WordForm` objects or ARPAbet strings, ``speakers`` are speaker
        labels (the default is every training speaker), and ``token`` is the token number. A form
        that is already in the lexicon gives its stored embedding, within numerical tolerance.
        """
        clips = self.clips(forms, speakers, token)
        encoder = self.encoder
        result = np.stack([np.stack([encoder.encode(clip) for clip in row]) for row in clips])
        means = self.store.speaker_means
        if means is not None:
            # talker normalization: each speaker's stored mean is subtracted
            rows = {label: i for i, label in enumerate(self.speakers["label"])}
            for j, speaker in enumerate(self.speaker_list(speakers)):
                if speaker.label not in rows:
                    raise ValueError(
                        f"talker normalization needs a speaker of the run, not {speaker.label!r}"
                    )
                result[:, j] -= means[rows[speaker.label]]
        return result

    def embed_types(self, forms) -> np.ndarray:
        """Word embeddings for word forms: the mean over every token of every training speaker,
        as for the words of the lexicon. An array of forms by dimensions."""
        tokens = range(1, self.config.synthesis.tokens_per_speaker + 1)
        stacked = np.stack([self.embed(forms, None, k) for k in tokens])
        return stacked.astype(np.float64).mean(axis=(0, 2)).astype(np.float32)

    def to_torch(self) -> dict[str, Any]:
        """The arrays as PyTorch tensors."""
        import torch

        return {
            "types": torch.from_numpy(np.ascontiguousarray(self.types)),
            "tokens": torch.from_numpy(np.ascontiguousarray(self.tokens)),
            "token_words": torch.from_numpy(self.token_words),
            "token_speakers": torch.from_numpy(self.token_speakers),
            "token_held_out": torch.from_numpy(self.token_held_out),
            "token_augmented": torch.from_numpy(self.token_augmented),
            "token_mapped": torch.from_numpy(self.token_mapped),
        }

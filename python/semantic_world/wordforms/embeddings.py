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
    load_config,
)
from semantic_world.wordforms.encoders.fixed import FixedEncoder, Projection, fit_pca
from semantic_world.wordforms.frontends import FrontendStore, make_frontends
from semantic_world.wordforms.generate import WordForm
from semantic_world.wordforms.synth import Synthesis
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
    def layers(self) -> np.ndarray | None:
        """A pretrained encoder's pooled output for every layer, memory-mapped: tokens by layers
        by dimensions. None unless the embedding was stored with ``store_layers``."""
        path = self.folder / "layers.npy"
        return np.load(path, mmap_mode="r") if path.exists() else None

    def summary(self) -> dict[str, Any]:
        keys = ("encoder", "pretrained", "dims", "tokens", "words")
        return {**{k: self.meta[k] for k in keys}, "reused": self.reused}


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
    ones unless ``word_embeddings.tokens`` is ``all``."""
    if config.word_embeddings.tokens == "all":
        return train
    clean = np.array([not t.augmentation for t in synthesis.tokens], dtype=bool)
    return train & clean


def word_means(tokens: np.ndarray, token_words: np.ndarray, chosen: np.ndarray, count: int):
    """Each word's embedding: the mean of its ``chosen`` tokens."""
    types = np.zeros((count, tokens.shape[1]), dtype=np.float64)
    totals = np.zeros(count, dtype=np.int64)
    np.add.at(types, token_words[chosen], np.asarray(tokens, dtype=np.float64)[chosen])
    np.add.at(totals, token_words[chosen], 1)
    if (totals == 0).any():
        raise ValueError("a word has no tokens from training speakers")
    return (types / totals[:, None]).astype(np.float32)


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
    source: dict[str, Any]
    if fixed:
        frontend = make_frontends(config)[embedding.frontend]
        source = frontend.meta()
    else:
        source = {"sample_rate": config.synthesis.sample_rate}
    source["word_embedding_tokens"] = config.word_embeddings.tokens
    mark = fingerprint(embedding, synthesis, source)
    keep_layers = not fixed and embedding.store_layers
    files = ["tokens.npy", "types.npy", "meta.yaml"] + (["layers.npy"] if keep_layers else [])
    if all((folder / name).exists() for name in files):
        store = EmbeddingStore.load(folder)
        if store.meta.get("fingerprint") == mark:
            store.reused = True
            return store
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.yaml").unlink(missing_ok=True)  # an unfinished write is never up to date
    token_words, _, train = token_layout(words, synthesis)
    chosen = word_embedding_tokens(config, synthesis, train)
    count = len(synthesis.tokens)
    meta: dict[str, Any] = {
        "name": embedding.name,
        "encoder": embedding.encoder,
        "pretrained": not fixed,
        "settings": embedding.resolved(),
    }
    if fixed:
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
            clean = np.array([not t.augmentation for t in synthesis.tokens], dtype=bool)
            fitted = train & content[token_words] & clean
            projection = fit_pca(features[fitted], embedding.pca_dims)
            projection.save(folder / "projection.npz")
            tokens = projection.apply(features)
            total = float(np.var(features[fitted].astype(np.float64), axis=0, ddof=1).sum())
            kept = float(projection.explained_variance.sum())
            meta["projection"] = {
                "kind": "pca",
                "fitted_on": "training-speaker tokens"
                + ("" if content.all() else " of content words")
                + ("" if clean.all() else ", without augmented tokens"),
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


class SoundEmbeddings:
    """The sound embeddings of a run, for use in other models.

    - ``types``: word embeddings, one row per word, in the order of ``words``;
    - ``tokens``: token embeddings, with ``token_words`` and ``token_speakers`` giving each
      token's row in ``words`` and in ``speakers``, ``token_held_out`` marking the tokens of
      held-out speakers, and ``token_augmented`` marking augmented tokens;
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
        self.token_table = pl.read_csv(self.folder / "tokens.csv")
        self.types = self.store.types
        self.tokens = self.store.tokens
        word_index = {label: i for i, label in enumerate(self.words["label"])}
        speaker_index = {label: i for i, label in enumerate(self.speakers["label"])}
        self.token_words = np.array([word_index[w] for w in self.token_table["word"]])
        self.token_speakers = np.array([speaker_index[s] for s in self.token_table["speaker"]])
        held_out = (self.speakers["split"] == "held_out").to_numpy()
        self.token_held_out = held_out[self.token_speakers]
        self.token_augmented = self.token_table["augmentation"].fill_null("").to_numpy() != ""
        self._encoder: Any = None
        self._lexicon: dict[str, str] = {}
        for label, arpabet in self.words.select("label", "arpabet").iter_rows():
            self._lexicon.setdefault(arpabet, label)  # of two forms that sound alike, the first
        # the synthesized tokens by word, speaker, and token number; augmented tokens are
        # not a form's own recording, so ``embed`` never returns them
        augmented = self.token_table["augmentation"].fill_null("").to_numpy()
        self._stored = {
            (word, speaker, int(label.rsplit(".", 1)[1])): i
            for i, (label, word, speaker) in enumerate(
                self.token_table.select("label", "word", "speaker").iter_rows()
            )
            if not augmented[i]
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
            forms.append(form)
        return forms

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
            if self.meta["encoder"] == "fixed":
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
        return np.stack([np.stack([encoder.encode(clip) for clip in row]) for row in clips])

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
        }

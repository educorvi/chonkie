"""Small process-local cache for API model instances."""

import json
from collections import OrderedDict
from threading import RLock
from time import monotonic
from typing import Any, Callable, Hashable

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings

_CACHE_TTL_SECONDS = 24 * 60 * 60
_MAX_CACHE_ENTRIES = 16
_cache_lock = RLock()
_cache: OrderedDict[Hashable, tuple[Any, float]] = OrderedDict()


def _prune_expired(now: float) -> None:
    expired_keys = [key for key, (_, last_used) in _cache.items() if now - last_used >= _CACHE_TTL_SECONDS]
    for key in expired_keys:
        del _cache[key]


def _cache_value(
    key: Hashable,
    factory: Callable[[], Any],
) -> Any:
    with _cache_lock:
        now = monotonic()
        cached = _cache.get(key)
        if cached is not None:
            value, _ = cached
            _cache[key] = (value, now)
            _cache.move_to_end(key)
            return value

        _prune_expired(now)
        value = factory()
        _cache[key] = (value, monotonic())
        while len(_cache) > _MAX_CACHE_ENTRIES:
            _cache.popitem(last=False)
        return value


def get_embedding_model(model_name: str) -> BaseEmbeddings:
    """Return a shared embedding backend for the requested model name."""
    key = ("auto", model_name)
    return _cache_value(key, lambda: AutoEmbeddings.get_embeddings(model_name))


def get_sentence_transformer_model(model_name: str, **kwargs: Any) -> BaseEmbeddings:
    """Return a shared SentenceTransformer backend for the requested model name."""
    from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings

    options_key = json.dumps(kwargs, sort_keys=True, separators=(",", ":"), default=repr)
    key = ("sentence-transformer", f"{model_name}:{options_key}")
    return _cache_value(key, lambda: SentenceTransformerEmbeddings(model=model_name, **kwargs))


def get_neural_chunker(config: dict[str, Any]) -> Any:
    """Return a cached NeuralChunker for a pipeline configuration."""
    config_key = json.dumps(config, sort_keys=True, separators=(",", ":"))

    def build() -> Any:
        from chonkie import NeuralChunker

        return NeuralChunker(**config)

    return _cache_value(("neural-chunker", config_key), build)

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from fastembed.text.static_embedding import StaticTextEmbedding
from tests.test_static_embeddings import _write_safetensors


@pytest.fixture
def local_static_model(tmp_path: Path) -> StaticTextEmbedding:
    vocab = {"[UNK]": 0, "[CLS]": 1, "hello": 2, "world": 3, "café": 4, "!": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A", special_tokens=[("[CLS]", 1)]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    embeddings = np.zeros((len(vocab), 256), dtype=np.float32)
    embeddings[:, : len(vocab)] = np.eye(len(vocab), dtype=np.float32)
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"embeddings": ("F32", list(embeddings.shape), embeddings.astype("<f4").tobytes())},
    )
    return StaticTextEmbedding(
        "minishlab/potion-base-8M",
        specific_model_path=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        local_files_only=True,
    )


@pytest.mark.parametrize("legacy_api", [False, True], ids=["native", "legacy-api"])
def test_static_tokenizer_batch_ids(
    local_static_model: StaticTextEmbedding, legacy_api: bool
) -> None:
    model = local_static_model
    assert model.tokenizer is not None
    if legacy_api:
        # Exercise the fallback even when the installed tokenizers has the fast API.
        model.tokenizer = SimpleNamespace(encode_batch=model.tokenizer.encode_batch)
    ids, lengths = model._token_ids(["hello world", "", "café!", "unknown"])
    assert ids.tolist() == [2, 3, 4, 5, 0]
    assert lengths.tolist() == [2, 0, 2, 1]
    assert ids.dtype == lengths.dtype == np.int64
    ids, lengths = model._token_ids([])
    assert ids.size == lengths.size == 0
    assert ids.dtype == lengths.dtype == np.int64


@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_static_tokenizer_public_embedding(
    local_static_model: StaticTextEmbedding, batch_size: int
) -> None:
    model = local_static_model
    documents = ["hello world", "", "café!", "unknown", "hello " * 600]
    result = np.stack(list(model.embed(iter(documents), batch_size=batch_size)))
    expected = np.zeros((5, 256), dtype=np.float32)
    expected[0, [2, 3]] = expected[2, [4, 5]] = 1 / np.sqrt(2)
    expected[4, 2] = 1
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.dtype == np.float32
    assert model.token_count(documents, batch_size=batch_size) == 517
    assert list(model.embed([])) == []


def test_static_tokenizer_prefers_fast_api(local_static_model: StaticTextEmbedding) -> None:
    model = local_static_model
    assert model.tokenizer is not None
    fast_encode = getattr(model.tokenizer, "encode_batch_fast", None)
    if fast_encode is None:
        pytest.skip("Installed tokenizers predates encode_batch_fast")
    fast = Mock(wraps=fast_encode)
    slow = Mock(side_effect=AssertionError("The fast path should be used"))
    model.tokenizer = SimpleNamespace(encode_batch_fast=fast, encode_batch=slow)
    ids, lengths = model._token_ids(["hello", ""])
    assert ids.tolist() == [2]
    assert lengths.tolist() == [1, 0]
    fast.assert_called_once_with(["hello", ""], add_special_tokens=False)
    slow.assert_not_called()

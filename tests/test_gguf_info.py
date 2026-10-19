import struct

from local_forge.gguf_info import inspect_gguf


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _kv_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _write(path, entries: list[bytes]) -> None:
    path.write_bytes(
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(entries))
        + b"".join(entries)
    )


def test_inspect_hybrid_gguf(tmp_path):
    path = tmp_path / "hybrid.gguf"
    _write(
        path,
        [
            _kv_string("general.architecture", "qwen35"),
            _kv_uint32("qwen35.block_count", 32),
            _kv_uint32("qwen35.context_length", 262144),
            _kv_uint32("qwen35.embedding_length", 2560),
            _kv_uint32("qwen35.attention.head_count", 16),
            _kv_uint32("qwen35.attention.head_count_kv", 4),
            _kv_uint32("qwen35.attention.key_length", 256),
            _kv_uint32("qwen35.attention.value_length", 256),
            _kv_uint32("qwen35.full_attention_interval", 4),
        ],
    )
    info = inspect_gguf(path)
    assert info["architecture"] == "qwen35"
    assert info["hybrid"] is True
    assert info["attention_layers"] == 8
    assert info["recommended"]["context_length"] == 131072
    assert info["recommended"]["cache_k"] == "q8_0"
    assert info["recommended"]["cache_v"] == "q4_0"


def test_inspect_full_attention_gguf(tmp_path):
    path = tmp_path / "full.gguf"
    _write(
        path,
        [
            _kv_string("general.architecture", "qwen3"),
            _kv_uint32("qwen3.block_count", 28),
            _kv_uint32("qwen3.context_length", 40960),
            _kv_uint32("qwen3.embedding_length", 1024),
            _kv_uint32("qwen3.attention.head_count", 16),
            _kv_uint32("qwen3.attention.head_count_kv", 8),
            _kv_uint32("qwen3.attention.key_length", 128),
            _kv_uint32("qwen3.attention.value_length", 128),
        ],
    )
    info = inspect_gguf(path)
    assert info["architecture"] == "qwen3"
    assert info["hybrid"] is False
    assert info["attention_layers"] == 28
    assert info["recommended"]["context_length"] == 32768
    assert info["recommended"]["cache_v"] == "q4_0"

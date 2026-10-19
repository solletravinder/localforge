from __future__ import annotations

import struct
from pathlib import Path

MAGIC = b"GGUF"
FIXED_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
INT_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}
WANTED_SUFFIXES = (
    ".block_count",
    ".context_length",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
    ".full_attention_interval",
    ".layer_types",
)


class GGUFError(ValueError):
    pass


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def read(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise GGUFError("Unexpected end of GGUF metadata")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self, limit: int = 32 * 1024 * 1024) -> str:
        length = self.u64()
        if length > limit:
            raise GGUFError("GGUF string metadata is unreasonably large")
        return self.read(length).decode("utf-8", errors="replace")

    def skip(self, count: int) -> None:
        self.handle.seek(count, 1)

    def skip_value(self, value_type: int) -> None:
        size = FIXED_SIZES.get(value_type)
        if size is not None:
            self.skip(size)
            return
        if value_type == 8:
            length = self.u64()
            self.skip(length)
            return
        if value_type == 9:
            element_type = self.u32()
            count = self.u64()
            if element_type in FIXED_SIZES:
                self.skip(FIXED_SIZES[element_type] * count)
                return
            for _ in range(count):
                self.skip_value(element_type)
            return
        raise GGUFError(f"Unsupported GGUF metadata type {value_type}")

    def value(self, value_type: int, capture: bool):
        size = FIXED_SIZES.get(value_type)
        if size is not None:
            raw = self.read(size)
            if not capture:
                return None
            if value_type in INT_TYPES:
                signed = value_type in {1, 3, 5, 11}
                return int.from_bytes(raw, "little", signed=signed)
            if value_type == 7:
                return bool(raw[0])
            if value_type == 6:
                return struct.unpack("<f", raw)[0]
            if value_type == 12:
                return struct.unpack("<d", raw)[0]
            return None
        if value_type == 8:
            text = self.string() if capture else None
            if not capture:
                length = self.u64()
                self.skip(length)
            return text
        if value_type == 9:
            element_type = self.u32()
            count = self.u64()
            if not capture:
                if element_type in FIXED_SIZES:
                    self.skip(FIXED_SIZES[element_type] * count)
                else:
                    for _ in range(count):
                        self.skip_value(element_type)
                return None
            if count > 1_000_000:
                raise GGUFError("GGUF array metadata is unreasonably large")
            return [self.value(element_type, True) for _ in range(count)]
        raise GGUFError(f"Unsupported GGUF metadata type {value_type}")


def _read_metadata(path: Path) -> dict:
    metadata: dict = {}
    with path.open("rb") as handle:
        reader = _Reader(handle)
        if reader.read(4) != MAGIC:
            raise GGUFError("Not a GGUF file")
        version = reader.u32()
        if version not in {2, 3}:
            raise GGUFError(f"Unsupported GGUF version {version}")
        reader.u64()  # tensor count
        kv_count = reader.u64()
        if kv_count > 200_000:
            raise GGUFError("GGUF metadata count is unreasonably large")
        for _ in range(kv_count):
            key = reader.string(limit=4096)
            value_type = reader.u32()
            capture = (
                key == "general.architecture"
                or key.endswith(WANTED_SUFFIXES)
                or ".ssm." in key
            )
            value = reader.value(value_type, capture)
            if capture:
                metadata[key] = value
    return metadata


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _sum_kv_heads(head_kv, attention_indices: list[int], block_count: int) -> int:
    if isinstance(head_kv, int):
        return head_kv * max(1, len(attention_indices))
    if isinstance(head_kv, list):
        values = [item for item in head_kv if isinstance(item, int)]
        if not values:
            return 0
        if len(values) == block_count:
            return sum(values[index] for index in attention_indices if 0 <= index < len(values))
        return sum(values)
    return 0


def inspect_gguf(path_value: str | Path) -> dict:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {path}")
    metadata = _read_metadata(path)
    architecture = str(metadata.get("general.architecture") or "unknown")

    def number(name: str) -> int | None:
        return _as_int(metadata.get(f"{architecture}.{name}"))

    block_count = number("block_count") or 0
    native_context = number("context_length") or 0
    embedding_length = number("embedding_length") or 0
    head_count = number("attention.head_count") or 0
    head_count_kv = metadata.get(f"{architecture}.attention.head_count_kv")
    full_interval = number("full_attention_interval")
    layer_types = metadata.get(f"{architecture}.layer_types")

    attention_indices: list[int] = []
    if isinstance(layer_types, list) and layer_types:
        for index, layer_type in enumerate(layer_types):
            label = str(layer_type).lower()
            if "attention" in label and "linear" not in label and "recurrent" not in label:
                attention_indices.append(index)
    elif full_interval and full_interval > 0 and block_count:
        attention_indices = list(range(full_interval - 1, block_count, full_interval))
    else:
        attention_indices = list(range(block_count))

    hybrid = (
        len(attention_indices) < block_count
        or bool(full_interval)
        or any(".ssm." in key for key in metadata)
        or architecture in {"qwen35", "qwen3next"}
    )
    kv_heads = _sum_kv_heads(head_count_kv, attention_indices, block_count)
    key_length = number("attention.key_length") or (
        embedding_length // head_count if head_count else 0
    )
    value_length = number("attention.value_length") or key_length
    kv_f16_bytes = kv_heads * (key_length + value_length) * 2
    kv_q8q4_bytes = kv_heads * (
        key_length * 2 * 0.53125 + value_length * 2 * 0.28125
    )
    kv_q8q8_bytes = kv_heads * (key_length + value_length) * 2 * 0.53125

    max_context = min(native_context or 32768, 131072)
    candidates = [
        131072,
        98304,
        65536,
        49152,
        32768,
        24576,
        16384,
        12288,
        8192,
        6144,
        4096,
        3072,
        2048,
    ]
    candidate = next(
        (size for size in candidates if size <= max_context),
        max_context,
    )
    kv_budget = 2_600_000_000
    if kv_q8q4_bytes:
        fitting = [
            size
            for size in candidates
            if size <= max_context and kv_q8q4_bytes * size <= kv_budget
        ]
        candidate = fitting[0] if fitting else min(2048, max_context)
    cache_k, cache_v = "q8_0", "q4_0"
    if kv_q8q8_bytes and kv_q8q8_bytes * candidate <= 1_800_000_000:
        cache_v = "q8_0"
    recommended = {
        "context_length": int(candidate),
        "batch": 2048,
        "ubatch": 1024,
        "flash_attn": "on",
        "cache_k": cache_k,
        "cache_v": cache_v,
    }
    return {
        "path": str(path),
        "file_size_gb": round(path.stat().st_size / 1e9, 2),
        "architecture": architecture,
        "block_count": block_count,
        "attention_layers": len(attention_indices),
        "native_context": native_context,
        "embedding_length": embedding_length,
        "head_count": head_count,
        "head_count_kv": head_count_kv,
        "key_length": key_length,
        "value_length": value_length,
        "hybrid": hybrid,
        "model_type": "hybrid" if hybrid else "full-attention",
        "kv_heads_total": kv_heads,
        "kv_f16_mb_per_1k": round(kv_f16_bytes * 1000 / 1e6, 2),
        "kv_q8q4_mb_per_1k": round(kv_q8q4_bytes * 1000 / 1e6, 2),
        "recommended": recommended,
        "recommended_kv_q8q4_gb": round(kv_q8q4_bytes * candidate / 1e9, 2),
    }

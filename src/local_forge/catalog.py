from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    engine: str  # "openvino" or "llama.cpp"
    repo: str
    dir_name: str
    format: str  # "openvino" or "gguf"
    size_gb: float
    modalities: tuple[str, ...]
    license: str
    notes: str = ""
    allow_patterns: tuple[str, ...] = ()
    quant: str = "Q4_K_M"
    local_path: str | None = None

    @property
    def is_local(self) -> bool:
        return self.local_path is not None


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="gemma4-e2b-uncensored-ov",
        name="Gemma 4 E2B Uncensored (OpenVINO INT4)",
        engine="openvino",
        repo="ONTONOUS/gemma-4-E2B-uncensored-int4-ov",
        dir_name="gemma-4-E2B-uncensored-int4-ov",
        format="openvino",
        size_gb=4.28,
        modalities=("text", "image"),
        license="Apache-2.0",
        notes=(
            "Community uncensored conversion of Gemma 4 E2B. Runs on the Intel GPU with "
            "OpenVINO GenAI. Text + vision; no MTP drafter in this build."
        ),
        allow_patterns=("*.xml", "*.bin", "*.json", "tokenizer*"),
    ),
    ModelSpec(
        id="qwen3.5-4b-abliterated",
        name="Qwen3.5 4B Abliterated (GGUF Q4_K_M)",
        engine="llama.cpp",
        repo="mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF",
        dir_name="qwen3.5-4b-abliterated-gguf",
        format="gguf",
        size_gb=3.3,
        modalities=("text", "image"),
        license="Apache-2.0",
        notes=(
            "Huihui abliterated Qwen3.5-4B GGUFs. Vision needs the Q8_0 mmproj file, which is "
            "downloaded automatically. Requires a recent llama.cpp build."
        ),
        allow_patterns=("*Q4_K_M.gguf", "*q4_k_m.gguf", "*mmproj-Q8_0.gguf", "*MMPROJ-Q8_0.gguf"),
    ),
    ModelSpec(
        id="lfm2.5-2.6b-abliterated",
        name="LFM2.5 2.6B Abliterated (GGUF Q4_K_M)",
        engine="llama.cpp",
        repo="alexwkleung/LFM2.5-2.6B-Abliterated-GGUF",
        dir_name="lfm2.5-2.6b-abliterated-gguf",
        format="gguf",
        size_gb=1.7,
        modalities=("text",),
        license="LFM Open License v1.0",
        notes=(
            "Very new abliterated build of Liquid AI's 2.6B agent model. Validate quality and "
            "tool-calling before relying on it. If the Hugging Face repo is unavailable, use "
            "'lforge models add-local' with a GGUF you downloaded manually."
        ),
        allow_patterns=("*Q4_K_M.gguf", "*q4_k_m.gguf"),
    ),
)


@dataclass(frozen=True)
class ImageModelSpec:
    id: str
    name: str
    repo: str
    dir_name: str
    size_gb: float
    license: str
    notes: str


IMAGE_CATALOG: tuple[ImageModelSpec, ...] = (
    ImageModelSpec(
        id="realistic-vision-v6-photoreal",
        name="Realistic Vision V6.0 B1 (OpenVINO FP16)",
        repo="MacrossRamen/Realistic_Vision_V6.0_B1_noVAE-openvino",
        dir_name="realistic-vision-v6-ov",
        size_gb=4.27,
        license="CreativeML OpenRAIL-M",
        notes=(
            "Photoreal SD1.5 checkpoint with no built-in content filter. Good for portraits "
            "and boudoir-style photography. Export is FP16; use 512x768 and 28-30 steps."
        ),
    ),
)


def local_specs(config: AppConfig) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in config.local_models:
        try:
            specs.append(
                ModelSpec(
                    id=str(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    engine=str(item.get("engine") or "llama.cpp"),
                    repo="",
                    dir_name="",
                    format=str(item.get("format") or ("openvino" if item.get("engine") == "openvino" else "gguf")),
                    size_gb=float(item.get("size_gb") or 0),
                    modalities=tuple(item.get("modalities") or ("text",)),
                    license=str(item.get("license") or "unknown"),
                    notes=str(item.get("notes") or "Locally registered model"),
                    local_path=str(item["path"]),
                )
            )
        except KeyError:
            continue
    return specs


def all_specs(config: AppConfig) -> list[ModelSpec]:
    return [*CATALOG, *local_specs(config)]


def find_spec(config: AppConfig, model_id: str | None) -> ModelSpec:
    specs = all_specs(config)
    if not model_id:
        return specs[0]
    for spec in specs:
        if spec.id == model_id:
            return spec
    known = ", ".join(s.id for s in specs)
    raise KeyError(f"Unknown model '{model_id}'. Known models: {known}")


DEFAULT_MODEL_ID = "gemma4-e2b-uncensored-ov"

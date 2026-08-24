"""Explicit, strict weight mapping for a pinned Qwen2.5-0.5B checkpoint."""

from dataclasses import dataclass
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from .model import ModelConfig, TinyDecoder

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def config_from_qwen(raw: dict, max_seq_len: int = 512) -> ModelConfig:
    """Accept full-attention Qwen2 with default RoPE and a bounded context."""
    if raw.get("model_type") != "qwen2" or raw.get("hidden_act") != "silu":
        raise ValueError("Only Qwen2 with a SiLU/SwiGLU MLP is supported")
    if raw.get("use_sliding_window") or raw.get("use_mrope") or raw.get("rope_scaling"):
        raise ValueError("Sliding-window attention and modified RoPE are not supported")
    if raw.get("layer_types") and any(t != "full_attention" for t in raw["layer_types"]):
        raise ValueError("Only full-attention layers are supported")
    if max_seq_len > raw["max_position_embeddings"]:
        raise ValueError("Requested context exceeds the checkpoint's supported context")
    return ModelConfig(
        vocab_size=raw["vocab_size"], hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"], num_layers=raw["num_hidden_layers"],
        num_heads=raw["num_attention_heads"], num_kv_heads=raw["num_key_value_heads"],
        max_seq_len=max_seq_len, rope_theta=raw["rope_theta"], norm_eps=raw["rms_norm_eps"],
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )


def import_qwen_weights(raw_config: dict, source: dict[str, torch.Tensor],
                        *, max_seq_len: int = 512) -> TinyDecoder:
    """Build a float32 CPU model. Reject missing, extra, or misshaped weights.

    Meta construction avoids initializing a second full random model in RAM.
    No inference is delegated to Transformers: only tensors are imported.
    """
    config = config_from_qwen(raw_config, max_seq_len)
    mapping = {"embedding.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight"}
    for layer in range(config.num_layers):
        dest, src = f"layers.{layer}", f"model.layers.{layer}"
        mapping[f"{dest}.attention_norm.weight"] = f"{src}.input_layernorm.weight"
        mapping[f"{dest}.mlp_norm.weight"] = f"{src}.post_attention_layernorm.weight"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"{dest}.{projection}.weight"] = f"{src}.self_attn.{projection}.weight"
            if projection != "o_proj":
                mapping[f"{dest}.{projection}.bias"] = f"{src}.self_attn.{projection}.bias"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"{dest}.{projection}.weight"] = f"{src}.mlp.{projection}.weight"
    mapping["lm_head.weight"] = "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    required = set(mapping.values())
    optional = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if missing := required - source.keys():
        raise ValueError(f"Missing checkpoint weights: {sorted(missing)}")
    if extra := source.keys() - required - optional:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(extra)}")
    if config.tie_word_embeddings and "lm_head.weight" in source:
        if not torch.equal(source["lm_head.weight"], source["model.embed_tokens.weight"]):
            raise ValueError("Checkpoint declares tied embeddings but output weights differ")
    with torch.device("meta"):
        model = TinyDecoder(config)
    expected = model.state_dict()
    converted = {}
    for destination, origin in mapping.items():
        weight = source[origin]
        if tuple(weight.shape) != tuple(expected[destination].shape):
            raise ValueError(f"Wrong shape for {origin}: got {tuple(weight.shape)}, expected {tuple(expected[destination].shape)}")
        if not weight.is_floating_point():
            raise ValueError(f"Expected floating point checkpoint weight: {origin}")
        # Convert each unique source once so tied embeddings share the tensor.
        if origin not in converted:
            converted[origin] = weight.to(device="cpu", dtype=torch.float32)
    model.load_state_dict({dest: converted[src] for dest, src in mapping.items()}, strict=True, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embedding.weight
    return model.eval()


def download_checkpoint(*, local_files_only: bool = False) -> Path:
    """Cache approximately 1 GB of weights outside the repo, at an immutable revision."""
    return Path(snapshot_download(
        MODEL_ID, revision=REVISION, local_files_only=local_files_only,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"],
    ))


@dataclass
class LoadedCheckpoint:
    model: TinyDecoder
    tokenizer: object
    snapshot: Path
    model_id: str = MODEL_ID
    revision: str = REVISION


def load_qwen(*, device="cpu", max_seq_len=512, local_files_only=False) -> LoadedCheckpoint:
    snapshot = download_checkpoint(local_files_only=local_files_only)
    raw = json.loads((snapshot / "config.json").read_text())
    weights = load_file(str(snapshot / "model.safetensors"), device="cpu")
    model = import_qwen_weights(raw, weights, max_seq_len=max_seq_len).to(device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    return LoadedCheckpoint(model, tokenizer, snapshot)

"""Independent Float64 decoder-only Transformer oracle for the two-case workload.

The scorer must compare a student's HBM outputs against this oracle. This
module never writes those HBM outputs for the student program.
"""

from math import sqrt

import numpy as np

from .workload import MODELS as MODELS
from .workload import SCENARIOS, Model


def make_inputs(model: Model, batch: int, prompt: int, continuation: int, seed: int):
    """Deterministic public fixture generator; private seeds stay outside artifacts."""
    if min(batch, prompt, continuation) <= 0 or model.width % model.heads:
        raise ValueError("Invalid model/scenario dimensions")
    rng = np.random.default_rng(seed)
    d, f = model.width, model.ffn
    weights = []
    for _ in range(model.layers):
        layer = {
            "ln1_g": np.ones(d) + rng.normal(0, 0.01, d),
            "ln1_b": rng.normal(0, 0.01, d),
            "wqkv": rng.normal(0, 1 / sqrt(d), (d, 3 * d)),
            "wo": rng.normal(0, 1 / sqrt(d), (d, d)),
            "ln2_g": np.ones(d) + rng.normal(0, 0.01, d),
            "ln2_b": rng.normal(0, 0.01, d),
            "w1": rng.normal(0, 1 / sqrt(d), (d, f)),
            "b1": rng.normal(0, 0.01, f),
            "w2": rng.normal(0, 1 / sqrt(f), (f, d)),
            "b2": rng.normal(0, 0.01, d),
        }
        weights.append({name: value.astype(np.float32) for name, value in layer.items()})
    inputs = rng.normal(size=(batch, prompt + continuation, d)).astype(np.float32)
    return weights, inputs


def _layernorm(x, gamma, beta):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * gamma + beta


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_chunk(
    model: Model,
    x: np.ndarray,
    weights: list[dict],
    history: list[tuple[np.ndarray, np.ndarray]] | None = None,
    store_kv_float32: bool = False,
    score_ranges: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Compute one prefill or teacher-forced chunk with causal history masking."""
    if x.dtype != np.float64 or x.ndim != 3 or x.shape[-1] != model.width:
        raise ValueError("Expected Float64 [batch,time,width] input")
    if len(weights) != model.layers or (history is not None and len(history) != model.layers):
        raise ValueError("Layer count mismatch")
    b, t, d = x.shape
    h, hd = model.heads, model.head_width
    next_history = []
    for layer_id, w in enumerate(weights):
        norm = _layernorm(x, w["ln1_g"], w["ln1_b"])
        qkv = (norm @ w["wqkv"]).reshape(b, t, 3, h, hd)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        past = 0
        if history is not None:
            old_k, old_v = history[layer_id]
            if old_k.shape != old_v.shape or old_k.shape[:2] != (b, h) or old_k.shape[3] != hd:
                raise ValueError("KV history shape mismatch")
            past = old_k.shape[2]
            k = np.concatenate((old_k, k), axis=2)
            v = np.concatenate((old_v, v), axis=2)
        scores = (q @ k.swapaxes(-1, -2)) / sqrt(hd)
        key_index = np.arange(past + t)
        query_limit = past + np.arange(t)
        scores = np.where(
            key_index[None, None, None, :] <= query_limit[None, None, :, None], scores, -np.inf
        )
        if score_ranges is not None:
            active = scores[np.isfinite(scores)]
            score_ranges.append((float(active.min()), float(active.max())))
        exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        context = (probs @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        residual = x + context @ w["wo"]
        hidden = _layernorm(residual, w["ln2_g"], w["ln2_b"])
        x = residual + _gelu(hidden @ w["w1"] + w["b1"]) @ w["w2"] + w["b2"]
        next_history.append(
            (k.astype(np.float32), v.astype(np.float32)) if store_kv_float32 else (k, v)
        )
    if not np.isfinite(x).all():
        raise ValueError("Reference produced non-finite values")
    return x, next_history


def build_history(
    model: Model,
    prompt: np.ndarray,
    weights: list[dict],
    chunk_size: int = 64,
    score_ranges: list[tuple[float, float]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build correct layer KV from historical input outside the timed decode."""
    if prompt.dtype != np.float32 or prompt.ndim != 3 or chunk_size <= 0:
        raise ValueError("Expected FP32 prompt and positive chunk size")
    history = None
    for start in range(0, prompt.shape[1], chunk_size):
        _, history = forward_chunk(
            model,
            prompt[:, start : start + chunk_size].astype(np.float64),
            weights,
            history,
            store_kv_float32=True,
            score_ranges=score_ranges,
        )
    if history is None:
        raise ValueError("Empty historical prompt")
    return history


def run_scenario_inputs(
    model: Model,
    scenario: str,
    weights: list[dict],
    inputs: np.ndarray,
    score_ranges: list[tuple[float, float]] | None = None,
):
    """Oracle for explicit FP32 fixtures, including numerical edge cases."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    batch, context, new_positions = SCENARIOS[scenario]
    if (
        len(weights) != model.layers
        or inputs.dtype != np.float32
        or inputs.shape != (batch, context + new_positions, model.width)
    ):
        raise ValueError("Reference fixture shape or dtype mismatch")
    if scenario.startswith("P"):
        prefill, history = forward_chunk(
            model,
            inputs[:, :context].astype(np.float64),
            weights,
            store_kv_float32=True,
            score_ranges=score_ranges,
        )
        first, history = forward_chunk(
            model,
            inputs[:, context:].astype(np.float64),
            weights,
            history,
            store_kv_float32=True,
            score_ranges=score_ranges,
        )
        hidden = np.concatenate((prefill, first), axis=1)
    else:
        history = build_history(model, inputs[:, :context], weights, score_ranges=score_ranges)
        pieces = []
        for step in range(new_positions):
            piece, history = forward_chunk(
                model,
                inputs[:, context + step : context + step + 1].astype(np.float64),
                weights,
                history,
                store_kv_float32=True,
                score_ranges=score_ranges,
            )
            pieces.append(piece)
        hidden = np.concatenate(pieces, axis=1)
        history = [(key[:, :, context:], value[:, :, context:]) for key, value in history]
    return hidden, history


def run_scenario(model: Model, scenario: str, seed: int):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    batch, context, new_positions = SCENARIOS[scenario]
    weights, inputs = make_inputs(model, batch, context, new_positions, seed)
    return run_scenario_inputs(model, scenario, weights, inputs)

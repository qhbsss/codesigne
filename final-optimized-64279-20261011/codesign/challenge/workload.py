"""Two-case v0.7 workload contract for the programmable challenge."""

import json
from dataclasses import dataclass
from importlib.resources import files


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    width: int
    heads: int
    ffn: int

    @property
    def head_width(self) -> int:
        return self.width // self.heads


def load_workload() -> dict:
    data = json.loads(files("codesign.challenge").joinpath("workload_v07.json").read_text())
    if (
        set(data)
        != {
            "version",
            "status",
            "active_cases",
            "dtype",
            "layernorm_epsilon",
            "gelu",
            "fixture_contract",
            "models",
            "scenarios",
        }
        or data["version"] != "challenge-workload-v0.7"
    ):
        raise ValueError("Unsupported challenge workload")
    if data["dtype"] != "float32" or data["layernorm_epsilon"] != 1e-5:
        raise ValueError("Unsupported numerical contract")
    if data["gelu"] != "tanh-approximation":
        raise ValueError("Unsupported GELU")
    if data["fixture_contract"] != {
        "source": "reference.make_inputs",
        "seed_domain": "nonnegative-integer",
        "decode_history": "reference.build_history",
        "scored_weight_override": False,
        "additional_qk_stress_scales": [0, 4],
    }:
        raise ValueError("Unsupported fixture contract")
    if set(data["models"]) != {"M1", "M2"} or set(data["scenarios"]) != {"P1", "D1"}:
        raise ValueError("Challenge needs M1/P1 and M2/D1")
    if data["active_cases"] != ["M1_P1", "M2_D1"]:
        raise ValueError("v0.7 requires M1_P1 and M2_D1")
    for item in data["models"].values():
        if set(item) != {"layers", "width", "heads", "ffn"} or any(
            type(value) is not int or value <= 0 for value in item.values()
        ):
            raise ValueError("Invalid model shape")
        if item["width"] % item["heads"] or item["ffn"] < item["width"]:
            raise ValueError("Invalid head or FFN width")
    for name, item in data["scenarios"].items():
        if set(item) != {"kind", "batch", "context_tokens", "new_positions"} or any(
            type(item[key]) is not int or item[key] <= 0
            for key in ("batch", "context_tokens", "new_positions")
        ):
            raise ValueError("Invalid scenario shape")
        if item["kind"] != ("prefill" if name.startswith("P") else "decode"):
            raise ValueError("Invalid scenario kind")
        if item["kind"] == "prefill" and item["new_positions"] != 1:
            raise ValueError("Prefill includes exactly one new position")
    return data


WORKLOAD = load_workload()
MODELS = {name: Model(name, **spec) for name, spec in WORKLOAD["models"].items()}
SCENARIOS = {
    name: (spec["batch"], spec["context_tokens"], spec["new_positions"])
    for name, spec in WORKLOAD["scenarios"].items()
}

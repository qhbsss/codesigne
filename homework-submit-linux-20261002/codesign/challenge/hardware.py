"""Strict v0.4 hardware menu and auditable teaching area model."""

import json
from dataclasses import dataclass
from importlib.resources import files


def cost_model() -> dict:
    return json.loads(files("codesign.challenge").joinpath("cost_v04.json").read_text())


@dataclass(frozen=True)
class Hardware:
    version: str
    sm_count: int
    tc_count: int
    vector_lanes: int
    sfu_lanes: int
    reduction_units: int
    rf_ports: str
    shared_kib: int
    dma_depth: int
    dma_engines: int
    sm_noc_bytes_per_cycle: int
    multicast: bool
    noc_bytes_per_cycle: int
    hbm_channels: int
    cache_mib: int
    tc_array: str | None = None
    tc_k_parallel: int | None = None
    shared_banks: int | None = None
    shared_ports: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Hardware":
        optional = {"tc_array", "tc_k_parallel", "shared_banks", "shared_ports"}
        required = set(cls.__dataclass_fields__) - optional
        if not isinstance(data, dict) or not required <= set(data) <= required | optional:
            raise ValueError("Hardware fields missing or unknown")
        obj = cls(**data)
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.version != "challenge-hardware-v0.4":
            raise ValueError("Unsupported hardware version")
        menus = {
            "sm_count": (4, 8, 12, 16, 20, 24, 28, 32),
            "tc_count": (0, 1, 2, 3, 4, 6, 8),
            "vector_lanes": (8, 16, 32, 64),
            "reduction_units": (0, 1, 2),
            "rf_ports": ("2R1W", "4R2W", "8R4W"),
            "shared_kib": (0, 32, 64, 96, 128, 192, 256),
            "dma_depth": (0, 1, 2),
            "dma_engines": (1, 2, 4),
            "sm_noc_bytes_per_cycle": (32, 64, 128),
            "noc_bytes_per_cycle": (64, 128, 256, 512),
            "hbm_channels": (1, 2, 4, 8),
            "cache_mib": (0, 1, 2, 4, 8, 16),
        }
        for key, values in menus.items():
            val = getattr(self, key)
            if type(val) is not type(values[0]) or val not in values:
                raise ValueError(f"Invalid {key}: {val!r}")
        if type(self.multicast) is not bool:
            raise ValueError("multicast must be boolean")
        if type(self.sfu_lanes) is not int or not 1 <= self.sfu_lanes <= self.vector_lanes:
            raise ValueError("sfu_lanes must be between 1 and vector_lanes")
        if self.tc_count:
            if (
                self.tc_array not in ("4x8", "8x8", "8x16")
                or type(self.tc_k_parallel) is not int
                or self.tc_k_parallel not in (1, 2, 4)
            ):
                raise ValueError("TC requires valid array and K parallelism")
        elif self.tc_array is not None or self.tc_k_parallel is not None:
            raise ValueError("TC=0 requires omitted TC subfields")
        if self.shared_kib:
            if type(self.shared_banks) is not int or self.shared_banks not in (1, 2, 4, 8):
                raise ValueError("shared requires valid bank count")
            if self.shared_ports not in ("1R1W", "2R1W"):
                raise ValueError("shared requires valid bank ports")
        elif self.shared_banks is not None or self.shared_ports is not None:
            raise ValueError("shared=0 requires omitted shared subfields")

    def to_dict(self) -> dict:
        self.validate()
        return {
            key: val for key in self.__dataclass_fields__ if (val := getattr(self, key)) is not None
        }

    def area_mm2(self) -> float:
        self.validate()
        a = cost_model()["area_mm2"]
        tc_area = 0.0
        if self.tc_count:
            pm, pn = map(int, self.tc_array.split("x"))
            k = self.tc_k_parallel
            tc_area = (
                self.tc_count
                * pm
                * pn
                / 32
                * (a["tc_base"] + a["tc_k_multiplier"] * k + a["tc_k_reduction"] * (k - 1))
            )
        shared_bank_area = 0.0
        if self.shared_kib:
            shared_bank_area = (
                a["shared_bank"]
                * self.shared_banks
                * a["shared_bank_port_multiplier"][self.shared_ports]
            )
        sm_area = (
            a["sm_control"]
            + a["vector_lane"] * self.vector_lanes
            + a["sfu_lane"] * self.sfu_lanes
            + a["reduction_unit"] * self.reduction_units
            + a["memory_kib"] * (256 * a["rf_port_multiplier"][self.rf_ports] + self.shared_kib)
            + shared_bank_area
            + a["dma_engine"] * self.dma_engines
            + a["dma_async_slot"] * self.dma_engines * self.dma_depth
            + a["sm_noc_byte_per_cycle"] * self.sm_noc_bytes_per_cycle
            + a["multicast"] * self.multicast
            + tc_area
        )
        cache_area = 0.0
        if self.cache_mib:
            cache_area = (
                a["cache_fixed"]
                + a["cache_memory_multiplier"] * a["memory_kib"] * self.cache_mib * 1024
            )
        return (
            a["fixed"]
            + a["hbm_channel"] * self.hbm_channels
            + a["noc_byte_per_cycle"] * self.noc_bytes_per_cycle
            + cache_area
            + self.sm_count * sm_area
        )

    def static_power_w(self) -> float:
        return self.area_mm2() * cost_model()["static_w_per_mm2"]


def baseline_hardware() -> Hardware:
    return Hardware(
        version="challenge-hardware-v0.4",
        sm_count=8,
        tc_count=2,
        tc_array="4x8",
        tc_k_parallel=1,
        vector_lanes=16,
        sfu_lanes=8,
        reduction_units=0,
        rf_ports="4R2W",
        shared_kib=128,
        shared_banks=4,
        shared_ports="1R1W",
        dma_depth=1,
        dma_engines=1,
        sm_noc_bytes_per_cycle=64,
        multicast=False,
        noc_bytes_per_cycle=256,
        hbm_channels=4,
        cache_mib=0,
    )


def raw_label_count() -> int:
    tc = 1 + 6 * 3 * 3
    shared = 1 + 6 * 4 * 2
    return 8 * tc * (8 + 16 + 32 + 64) * 3 * shared * 3 * 3 * 3 * 3 * 2 * 4 * 4 * 6

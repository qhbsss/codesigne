"""Pure v0.4 service formulas and byte-address slices.

These functions have no access to student-provided traffic estimates. The
event scheduler supplies actual M/N/K and addresses from the parsed IR.
"""

from dataclasses import dataclass
from math import ceil, log2

from .hardware import Hardware, cost_model


@dataclass(frozen=True)
class ComputeService:
    compute_cycles: int
    rf_read_bytes: int
    rf_write_bytes: int
    physical_multiplications: int
    physical_k_additions: int
    arithmetic_pj: float


def mma_service(hardware: Hardware, m: int, n: int, k_extent: int) -> ComputeService:
    hardware.validate()
    if hardware.tc_count == 0:
        raise ValueError("MMA.ACC unavailable when TC=0")
    if any(type(v) is not int or v <= 0 for v in (m, n, k_extent)):
        raise ValueError("M/N/K must be positive integers")
    pm, pn = map(int, hardware.tc_array.split("x"))
    blocks_m, blocks_n = ceil(m / pm), ceil(n / pn)
    blocks = blocks_m * blocks_n
    groups = ceil(k_extent / hardware.tc_k_parallel)
    cycles = 2 + ceil(log2(hardware.tc_k_parallel)) + ceil(blocks / hardware.tc_count) * groups
    reads = 4 * (blocks_n * m * k_extent + blocks_m * k_extent * n + m * n)
    writes = 4 * m * n
    multiplications = blocks * pm * pn * hardware.tc_k_parallel * groups
    reductions = blocks * pm * pn * (hardware.tc_k_parallel - 1) * groups
    prices = cost_model()["energy_pj"]
    return ComputeService(
        compute_cycles=cycles,
        rf_read_bytes=reads,
        rf_write_bytes=writes,
        physical_multiplications=multiplications,
        physical_k_additions=reductions,
        arithmetic_pj=(
            multiplications * prices["tc_multiply"] + reductions * prices["tc_k_reduce"]
        ),
    )


def vector_cycles(elements: int, lanes: int) -> int:
    if type(elements) is not int or type(lanes) is not int or min(elements, lanes) <= 0:
        raise ValueError("Invalid vector size/lanes")
    return 2 + ceil(elements / lanes)


def sfu_cycles(elements: int, lanes: int) -> int:
    if type(elements) is not int or type(lanes) is not int or min(elements, lanes) <= 0:
        raise ValueError("Invalid SFU size/lanes")
    return 2 + 4 * ceil(elements / lanes)


def reduction_cycles(elements: int, vector_lanes: int, dedicated_units: int) -> int:
    if (
        type(elements) is not int
        or type(vector_lanes) is not int
        or type(dedicated_units) is not int
        or min(elements, vector_lanes) <= 0
        or dedicated_units < 0
    ):
        raise ValueError("Invalid reduction size/resources")
    throughput = vector_lanes * max(1, dedicated_units)
    return 2 + ceil(elements / throughput) + ceil(log2(elements))


def rf_port_cycles(hardware: Hardware, read_bytes: int, write_bytes: int) -> int:
    if any(type(x) is not int or x < 0 for x in (read_bytes, write_bytes)):
        raise ValueError("Invalid RF byte demand")
    read_ports, write_ports = {
        "2R1W": (2, 1),
        "4R2W": (4, 2),
        "8R4W": (8, 4),
    }[hardware.rf_ports]
    return max(ceil(read_bytes / (16 * read_ports)), ceil(write_bytes / (16 * write_ports)))


def line_segments(address: int, length: int) -> tuple[tuple[int, int], ...]:
    """Return (64 B line base, payload bytes) for a contiguous HBM access."""
    if (
        type(address) is not int
        or type(length) is not int
        or address < 0
        or length <= 0
        or address + length > 2 * 1024**3
    ):
        raise ValueError("Invalid HBM byte range")
    result = []
    cursor = address
    while cursor < address + length:
        line = cursor // 64 * 64
        take = min(line + 64 - cursor, address + length - cursor)
        result.append((line, take))
        cursor += take
    return tuple(result)


def hbm_channel(address: int, channels: int) -> int:
    if type(address) is not int or address < 0 or type(channels) is not int or channels <= 0:
        raise ValueError("Invalid HBM channel query")
    return (address // 256) % channels


def shared_bank(address: int, banks: int) -> int:
    if type(address) is not int or address < 0 or type(banks) is not int or banks <= 0:
        raise ValueError("Invalid shared bank query")
    return (address // 64) % banks

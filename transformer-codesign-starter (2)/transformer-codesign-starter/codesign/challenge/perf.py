"""Deterministic serial service trace for the first teaching baseline.

This is a conservative timing baseline for a single dependent workgroup.  The
multi-workgroup asynchronous scheduler in the challenge proposal is a separate
future version; this module does not claim to model overlap.
"""

from dataclasses import dataclass
from math import ceil

from .cache import GlobalCache
from .hardware import Hardware, cost_model
from .isa import Instruction
from .power import EnergyEvent, max_window_power_w
from .service import mma_service, reduction_cycles, rf_port_cycles, sfu_cycles, vector_cycles


def _lines(desc):
    offset = desc["offset"]
    shape = desc.get("shape", [desc["count"]])
    strides = desc.get("strides", [1])
    if len(shape) == 1 and shape[0] > 0 and strides == [1]:
        return list(range(offset // 16, (offset + shape[0] - 1) // 16 + 1))
    if len(shape) == 2 and min(shape) > 0 and strides == [shape[1], 1]:
        return list(range(offset // 16, (offset + shape[0] * shape[1] - 1) // 16 + 1))
    if len(shape) == 2 and min(shape) > 0 and 1 in strides:
        extent, step, width = (
            (shape[0], strides[0], shape[1])
            if strides[1] == 1
            else (shape[1], strides[1], shape[0])
        )
        found = set()
        for index in range(extent):
            start = offset + index * step
            found.update(range(start // 16, (start + width - 1) // 16 + 1))
        return sorted(found)
    found = set()
    if len(shape) == 1:
        for i in range(shape[0]):
            found.add((offset + i * strides[0]) // 16)
    else:
        for row in range(shape[0]):
            for col in range(shape[1]):
                found.add((offset + row * strides[0] + col * strides[1]) // 16)
    return sorted(found)


@dataclass(frozen=True)
class PerfResult:
    cycles: int
    dynamic_energy_pj: float
    total_energy_pj: float
    average_power_w: float
    peak_window_power_w: float
    peak_power_upper_bound_w: float
    area_mm2: float
    instruction_count: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    cache_hits: int
    cache_misses: int
    resource_stats: dict | None = None


def estimate_serial(hardware: Hardware, program) -> PerfResult:
    """Evaluate actual addresses and hardware service demand in program order."""
    hardware.validate()
    prices = cost_model()["energy_pj"]
    cache = GlobalCache(hardware.cache_mib)
    cycles = 0
    dynamic_energy = 0.0
    peak_op_power = 0.0
    instruction_count = 0
    hbm_write_bytes = 0
    energy_events = []
    for ins in program:
        if not isinstance(ins, Instruction):
            raise ValueError("Expected parsed challenge instruction")
        instruction_count += 1
        op, args = ins.op, ins.args
        duration = 1
        energy = 0.0
        if op == "MMA.ACC":
            result = mma_service(hardware, args["m"], args["n"], args["k"])
            duration = max(
                result.compute_cycles,
                rf_port_cycles(hardware, result.rf_read_bytes, result.rf_write_bytes),
            )
            energy = (
                result.arithmetic_pj
                + (result.rf_read_bytes + result.rf_write_bytes)
                * prices["rf_byte"][hardware.rf_ports]
            )
        elif op == "VEC":
            count = args["dst"]["count"]
            read_bytes = 4 * sum(src["count"] for src in args["src"] if "space" in src)
            write_bytes = 4 * count
            duration = max(
                vector_cycles(count, hardware.vector_lanes),
                rf_port_cycles(hardware, read_bytes, write_bytes),
            )
            energy = (
                count * prices["vector_fma" if args["kind"] == "fma" else "vector_other"]
                + (read_bytes + write_bytes) * prices["rf_byte"][hardware.rf_ports]
            )
        elif op == "REDUCE":
            count = args["src"]["count"]
            read_bytes, write_bytes = 4 * count, 4
            duration = max(
                reduction_cycles(count, hardware.vector_lanes, hardware.reduction_units),
                rf_port_cycles(hardware, read_bytes, write_bytes),
            )
            energy = (
                max(0, count - 1)
                * prices[
                    "reduction_dedicated_merge"
                    if hardware.reduction_units
                    else "reduction_vector_merge"
                ]
                + (read_bytes + write_bytes) * prices["rf_byte"][hardware.rf_ports]
            )
        elif op == "SFU":
            count = args["src"]["count"]
            read_bytes = write_bytes = 4 * count
            duration = max(
                sfu_cycles(count, hardware.sfu_lanes),
                rf_port_cycles(hardware, read_bytes, write_bytes),
            )
            energy = (
                count * prices["sfu"]
                + (read_bytes + write_bytes) * prices["rf_byte"][hardware.rf_ports]
            )
        elif op in ("LD", "ST"):
            src, dst = args["src"], args["dst"]
            count = src["count"]
            payload_bytes = 4 * count
            rf_read = payload_bytes if src["space"] == "RF" else 0
            rf_write = payload_bytes if dst["space"] == "RF" else 0
            duration = 4 + rf_port_cycles(hardware, rf_read, rf_write)
            energy = (rf_read + rf_write) * prices["rf_byte"][hardware.rf_ports]
            if "SH" in (src["space"], dst["space"]):
                energy += payload_bytes * prices["shared_byte"][hardware.shared_ports]
                duration += ceil(payload_bytes / (16 * hardware.shared_banks))
            if "HBM" in (src["space"], dst["space"]):
                address = src if src["space"] == "HBM" else dst
                lines = _lines(address)
                duration += len(lines)  # one 64 B DMA slice per engine
                duration += ceil(payload_bytes / hardware.sm_noc_bytes_per_cycle)
                duration += ceil(payload_bytes / hardware.noc_bytes_per_cycle)
                energy += payload_bytes * (
                    prices["noc_byte_fixed"]
                    + prices["noc_byte_per_width"] * hardware.noc_bytes_per_cycle
                )
                energy += len(lines) * prices["dma_slice"]
                if src["space"] == "HBM":
                    for line in lines:
                        hit = cache.read_line(line * 64)
                        if hit:
                            duration += 8
                            energy += (
                                64 * prices["cache_read_byte"] + prices["cache_query_or_invalidate"]
                            )
                        else:
                            duration += ceil(64 / 32) + 8 * bool(hardware.cache_mib)
                            energy += 64 * prices["hbm_byte"]
                            if hardware.cache_mib:
                                energy += (
                                    prices["cache_query_or_invalidate"]
                                    + 64 * prices["cache_fill_byte"]
                                )
                else:
                    hbm_write_bytes += payload_bytes
                    duration += ceil(payload_bytes / 32)
                    energy += payload_bytes * prices["hbm_byte"]
                    for line in lines:
                        cache.write(line * 64, 64)
        elif op not in ("WG.BEGIN", "WG.END", "WAIT", "BARRIER", "STEP.COMMIT"):
            raise ValueError(f"Unsupported performance opcode: {op}")
        if energy:
            energy_events.append(EnergyEvent(cycles, cycles + duration, energy))
        cycles += duration
        dynamic_energy += energy
        peak_op_power = max(peak_op_power, energy * cost_model()["clock_hz"] / duration / 1e12)
    area = hardware.area_mm2()
    seconds = cycles / cost_model()["clock_hz"]
    static_power = hardware.static_power_w()
    return PerfResult(
        cycles=cycles,
        dynamic_energy_pj=dynamic_energy,
        total_energy_pj=dynamic_energy + static_power * seconds * 1e12,
        average_power_w=dynamic_energy * 1e-12 / seconds + static_power,
        peak_window_power_w=max_window_power_w(area, energy_events),
        peak_power_upper_bound_w=peak_op_power + static_power,
        area_mm2=area,
        instruction_count=instruction_count,
        hbm_read_bytes=cache.traffic.hbm_read_bytes,
        hbm_write_bytes=hbm_write_bytes,
        cache_hits=cache.traffic.hits,
        cache_misses=cache.traffic.misses,
    )

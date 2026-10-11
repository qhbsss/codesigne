"""Typed, bounded low-level IR for challenge microprograms.

Each non-comment line is an opcode and one JSON object. This intentionally
shares the command prototype's readable text form, without sharing its ISA.
"""

import json
from collections.abc import Collection
from dataclasses import dataclass
from string import Formatter

MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_DYNAMIC_INSTRUCTIONS = 10**7
MAX_LOOP_DEPTH = 32
OPS = {
    "WG.BEGIN",
    "WG.END",
    "LD",
    "ST",
    "MMA.ACC",
    "VEC",
    "REDUCE",
    "SFU",
    "WAIT",
    "BARRIER",
    "STEP.COMMIT",
}


@dataclass(frozen=True)
class Instruction:
    op: str
    args: dict
    line: int


def sync_targets(op: str, args: dict, active_groups: Collection[str]) -> tuple[str, ...]:
    """Resolve the workgroups affected by a WAIT or BARRIER instruction.

    The legacy events-only form is unambiguous only with one resident group.
    Explicit targets keep independent workgroups from inheriting each other's
    synchronization delays.
    """
    if op not in {"WAIT", "BARRIER"} or not isinstance(args, dict):
        raise ValueError("Invalid synchronization opcode or descriptor")
    events = args.get("events")
    if not isinstance(events, list) or any(not isinstance(e, str) or not e for e in events):
        raise ValueError("Invalid synchronization events")
    if len(set(events)) != len(events):
        raise ValueError("Duplicate synchronization event")
    active = set(active_groups)
    if set(args) == {"events"}:
        if len(active) != 1:
            raise ValueError("Synchronization target is ambiguous")
        return (next(iter(active)),)
    if op == "WAIT" and set(args) == {"wg", "events"}:
        targets = (args["wg"],)
    elif op == "BARRIER" and set(args) == {"wgs", "events"}:
        wgs = args["wgs"]
        if not isinstance(wgs, list) or not wgs:
            raise ValueError("Invalid barrier workgroups")
        targets = tuple(wgs)
    else:
        raise ValueError("Invalid synchronization fields")
    if any(not isinstance(wg, str) or not wg or wg not in active for wg in targets):
        raise ValueError("Unknown synchronization workgroup")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate barrier workgroup")
    return targets


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value}")


def _prepare(lines: list[str]):
    """Parse all source lines once, including the bodies of zero-trip loops."""
    parsed = []
    matching_end = {}
    stack = []
    for line, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            parsed.append(None)
            continue
        try:
            op, payload = raw.split(maxsplit=1)
            args = json.loads(
                payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line}: expected OPCODE JSON-object") from exc
        if type(args) is not dict:
            raise ValueError(f"line {line}: descriptor must be an object")
        if op == "FOR":
            if set(args) != {"var", "start", "stop", "step"}:
                raise ValueError(f"line {line}: invalid FOR")
            if not isinstance(args["var"], str) or not args["var"].isidentifier():
                raise ValueError(f"line {line}: invalid loop variable")
            if len(stack) >= MAX_LOOP_DEPTH:
                raise ValueError(f"line {line}: loop nesting exceeds {MAX_LOOP_DEPTH}")
            if any(parsed[parent][1]["var"] == args["var"] for parent in stack):
                raise ValueError(f"line {line}: nested duplicate loop variable")
            stack.append(len(parsed))
        elif op == "END.FOR":
            if args or not stack:
                raise ValueError(f"line {line}: invalid or unmatched END.FOR")
            matching_end[stack.pop()] = len(parsed)
        elif op not in OPS:
            raise ValueError(f"line {line}: unknown opcode {op}")
        parsed.append((op, args, line))
    if stack:
        raise ValueError("Unclosed FOR")
    return parsed, matching_end


def _expand(lines, matching_end, start: int, stop: int, env: dict[str, int], count: list[int]):
    index = start
    while index < stop:
        item = lines[index]
        index += 1
        if item is None:
            continue
        op, args, line = item
        if op == "FOR":
            var = args["var"]
            if var in env:
                raise ValueError(f"line {line}: nested duplicate loop variable")
            begin = _integer(_substitute(args["start"], env), "FOR start")
            end = _integer(_substitute(args["stop"], env), "FOR stop")
            step = _integer(_substitute(args["step"], env), "FOR step", 1)
            if end < begin:
                raise ValueError("FOR stop before start")
            tail = matching_end[index - 1]
            if len(range(begin, end, step)) > MAX_DYNAMIC_INSTRUCTIONS:
                raise ValueError("Dynamic instruction limit exceeded")
            for value in range(begin, end, step):
                yield from _expand(lines, matching_end, index, tail, env | {var: value}, count)
            index = tail + 1
            continue
        if op == "END.FOR":
            raise ValueError(f"line {line}: unmatched END.FOR")
        count[0] += 1
        if count[0] > MAX_DYNAMIC_INSTRUCTIONS:
            raise ValueError("Dynamic instruction limit exceeded")
        yield Instruction(op, _substitute(args, env), line)


def _substitute(value, env):
    if isinstance(value, dict):
        if set(value) == {"var"}:
            name = value["var"]
            if name not in env:
                raise ValueError(f"Unknown loop index {name}")
            return env[name]
        if len(value) == 1:
            kind, items = next(iter(value.items()))
            if kind in {"add", "mul", "min", "max", "mod", "ceildiv"}:
                if not isinstance(items, list) or len(items) < 2:
                    raise ValueError("Invalid integer expression")
                terms = [_substitute(item, env) for item in items]
                if any(type(item) is not int for item in terms):
                    raise ValueError("Integer expression contains noninteger")
                if kind == "add":
                    return sum(terms)
                if kind == "mul":
                    result = 1
                    for item in terms:
                        result *= item
                    return result
                if kind == "min":
                    return min(terms)
                if kind == "max":
                    return max(terms)
                if len(terms) != 2 or terms[1] <= 0:
                    raise ValueError("Invalid modulus/divisor")
                return terms[0] % terms[1] if kind == "mod" else -(-terms[0] // terms[1])
        if set(value) == {"index", "scale", "offset"}:
            name = value["index"]
            if name not in env:
                raise ValueError(f"Unknown loop index {name}")
            scale = _integer(value["scale"], "affine scale")
            offset = _integer(value["offset"], "affine offset")
            return env[name] * scale + offset
        return {k: _substitute(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, env) for v in value]
    if isinstance(value, str) and "{" in value:
        try:
            for _, field, spec, conversion in Formatter().parse(value):
                if field is not None and (field not in env or spec or conversion):
                    raise ValueError("Invalid loop string field")
            return value.format_map(env)
        except (KeyError, ValueError) as exc:
            raise ValueError("Unknown or invalid string interpolation") from exc
    return value


def parse(text: str) -> tuple[Instruction, ...]:
    return tuple(iter_parse(text))


def iter_parse(text: str):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("Invalid or oversized program text")
    lines = text.splitlines()
    if not any(line.strip() and not line.lstrip().startswith("#") for line in lines):
        raise ValueError("Empty program")
    parsed, matching_end = _prepare(lines)
    return _expand(parsed, matching_end, 0, len(parsed), {}, [0])

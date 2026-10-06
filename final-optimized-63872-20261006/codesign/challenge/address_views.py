"""Exact word-set operations for validated one- and two-dimensional ISA views."""

from math import gcd, prod


def dimensions(view: dict) -> tuple[list[int], list[int]]:
    """Validate the shape rules shared by the race and functional checkers."""
    if not isinstance(view, dict):
        raise ValueError("Invalid memory operand")
    count = view["count"]
    shape = view.get("shape", [count])
    strides = view.get("strides", [1])
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(shape, list)
        or not isinstance(strides, list)
        or len(shape) not in (1, 2)
        or len(shape) != len(strides)
        or any(type(value) is not int or value <= 0 for value in shape + strides)
        or prod(shape) != count
    ):
        raise ValueError("Invalid memory shape/strides")
    return shape, strides


def bounds(view: dict) -> tuple[int, int]:
    shape, strides = dimensions(view)
    offset = view["offset"]
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid memory offset/count")
    return offset, offset + 1 + sum(
        (extent - 1) * stride for extent, stride in zip(shape, strides)
    )


def positions(view: dict):
    offset = view["offset"]
    shape, strides = dimensions(view)
    if len(shape) == 1:
        return (offset + index * strides[0] for index in range(shape[0]))
    return (
        offset + row * strides[0] + col * strides[1]
        for row in range(shape[0])
        for col in range(shape[1])
    )


def _contains_position(view: dict, position: int) -> bool:
    delta = position - view["offset"]
    shape = view.get("shape", [view["count"]])
    strides = view.get("strides", [1])
    if len(shape) == 1:
        return delta >= 0 and delta % strides[0] == 0 and delta // strides[0] < shape[0]
    first = 0 if shape[0] <= shape[1] else 1
    second = 1 - first
    for index in range(shape[first]):
        rest = delta - index * strides[first]
        if rest >= 0 and rest % strides[second] == 0 and rest // strides[second] < shape[second]:
            return True
    return False


def _contiguous(view: dict) -> bool:
    shape = view.get("shape", [view["count"]])
    strides = view.get("strides", [1])
    if len(shape) == 1:
        return strides[0] == 1
    return (strides[1] == 1 and strides[0] == shape[1]) or (
        strides[0] == 1 and strides[1] == shape[0]
    )


def _row_blocks(view: dict) -> tuple[int, int, int | None] | None:
    """Return equal-width contiguous rows when the view has such a form."""
    if _contiguous(view):
        return 1, view["count"], None
    shape = view.get("shape", [view["count"]])
    strides = view.get("strides", [1])
    if len(shape) == 1:
        return shape[0], 1, strides[0]
    if strides[1] == 1:
        return shape[0], shape[1], strides[0]
    if strides[0] == 1:
        return shape[1], shape[0], strides[1]
    return None


def _rectangular_overlap(left: dict, right: dict) -> bool | None:
    """Answer exact overlap for compatible contiguous-row representations."""
    left_rows = _row_blocks(left)
    right_rows = _row_blocks(right)
    if left_rows is None or right_rows is None:
        return None
    left_count, left_width, left_stride = left_rows
    right_count, right_width, right_stride = right_rows
    if left_count == 1:
        stride = right_stride
    elif right_count == 1:
        stride = left_stride
    elif left_stride == right_stride:
        stride = left_stride
    else:
        return None
    if stride is None:
        return True
    delta = right["offset"] - left["offset"]
    minimum_q = -(-(1 - right_width - delta) // stride)
    maximum_q = (left_width - 1 - delta) // stride
    return max(minimum_q, 1 - left_count) <= min(maximum_q, right_count - 1)


def overlap(left: dict, right: dict) -> bool:
    """Return whether two validated views contain at least one common word."""
    lo, hi = bounds(left)
    other_lo, other_hi = bounds(right)
    if lo >= other_hi or other_lo >= hi:
        return False
    if _contiguous(left) and _contiguous(right):
        return True
    rectangular = _rectangular_overlap(left, right)
    if rectangular is not None:
        return rectangular

    left_shape = left.get("shape", [left["count"]])
    right_shape = right.get("shape", [right["count"]])
    left_strides = left.get("strides", [1])
    right_strides = right.get("strides", [1])
    if len(left_shape) == len(right_shape) == 1:
        if (left["offset"] - right["offset"]) % gcd(left_strides[0], right_strides[0]):
            return False
    smaller, larger = (left, right) if left["count"] <= right["count"] else (right, left)
    if _contiguous(larger):
        large_lo, large_hi = bounds(larger)
        return any(large_lo <= position < large_hi for position in positions(smaller))
    if smaller["count"] <= 64 or len(larger.get("shape", [larger["count"]])) == 1:
        return any(_contains_position(larger, position) for position in positions(smaller))
    occupied = set(positions(smaller))
    return any(position in occupied for position in positions(larger))

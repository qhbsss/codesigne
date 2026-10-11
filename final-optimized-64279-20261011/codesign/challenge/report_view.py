"""Read-only text view of saved two-case challenge resource reports."""

import json
import math
from html import escape
from pathlib import Path

from .score import CASES

MISSING = "—"


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _value(value: object, unit: str = "") -> str:
    if value is None or isinstance(value, bool):
        return MISSING
    if type(value) is int:
        return f"{value:,} {unit}".rstrip()
    if type(value) is float:
        if not math.isfinite(value):
            return MISSING
        return f"{value:,.6g} {unit}".rstrip()
    return MISSING


def _ratio(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return MISSING
    if not math.isfinite(value):
        return MISSING
    return f"{100 * value:.2f}%"


def _text(value: object) -> str:
    return value if isinstance(value, str) and value else MISSING


def _passed(value: object) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return MISSING


def _interval_count(value: object) -> str:
    if type(value) is int and value >= 0:
        return _value(value)
    if isinstance(value, list) and all(type(item) is int and item >= 0 for item in value):
        return json.dumps(value, separators=(",", ":"))
    return MISSING


def _resource_lines(label: str, record: object) -> list[str]:
    record = _object(record)
    busy = _object(record.get("busy_cycles"))
    entities = _object(record.get("entity_count"))
    utilization = _object(record.get("utilization"))
    lines = [f"  {label}: busy 类型={_text(record.get('busy_cycles_kind'))}; 利用率使用报告记录值"]
    if not busy:
        lines.append(f"    资源占用: {MISSING}")
    for name in sorted(busy):
        lines.append(
            f"    {name}: busy={_value(busy[name], '实体·周期')}; "
            f"实体数={_value(entities.get(name))}; "
            f"利用率={_ratio(utilization.get(name))}"
        )
    return lines


def _edges(timeline: dict) -> list[int] | None:
    edges = timeline.get("edges_cycles")
    if (
        not isinstance(edges, list)
        or not 2 <= len(edges) <= 129
        or any(type(edge) is not int for edge in edges)
        or edges[0] != 0
        or any(right <= left for left, right in zip(edges, edges[1:]))
    ):
        return None
    return edges


def _series(record: object, name: str, bins: int) -> list[int | float] | None:
    values = _object(record).get(name)
    if not isinstance(values, list) or len(values) != bins:
        return None
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in values
    ):
        return None
    return values


def _heat(values: list[int | float]) -> str:
    shades = " ▁▂▃▄▅▆▇█"
    return "".join(shades[min(8, max(0, math.ceil(value * 8)))] for value in values)


def _timeline_lines(timeline: object) -> list[str]:
    timeline = _object(timeline)
    edges = _edges(timeline)
    if edges is None:
        return [f"  资源/功率时间箱: {MISSING}"]
    bins = len(edges) - 1
    power = _object(timeline.get("power"))
    average = _series(power, "average_w", bins)
    energy = _series(power, "dynamic_energy_pj", bins)
    shared = _object(timeline.get("shared"))
    noc = _series(_object(shared.get("bytes")), "noc", bins)
    lines = [
        f"  资源/功率时间箱: 版本={_text(timeline.get('version'))}; 箱数={bins}; "
        "每箱为 [起,止) 周期的积分或平均，箱内先后次序未知",
        f"  精确峰值窗口={_value(power.get('exact_peak_window_w'), 'W')}; "
        f"窗口={_value(power.get('window_cycles'), '周期')}",
    ]
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        lines.append(
            f"    箱 {index} [{start},{end}): 平均功率="
            f"{_value(average[index] if average else None, 'W')}; "
            f"动态能量={_value(energy[index] if energy else None, 'pJ')}; "
            f"NoC={_value(noc[index] if noc else None, 'B')}"
        )
    scopes = [(f"SM {sm}", value) for sm, value in sorted(_object(timeline.get("sms")).items())]
    scopes.append(("共享", shared))
    for label, scope in scopes:
        for field, description in (
            ("utilization", "利用率箱"),
            ("bandwidth_utilization", "带宽利用率箱"),
        ):
            for resource, values in sorted(_object(_object(scope).get(field)).items()):
                if (valid := _series({resource: values}, resource, bins)) is not None:
                    lines.append(f"    {label} {resource} {description}: {_heat(valid)}")
    return lines


def render_report(report: dict) -> str:
    """Display recorded fields only; never infer a timeline or verify a signature."""
    if not isinstance(report, dict):
        raise ValueError("Challenge report must be a JSON object")
    cases = _object(report.get("cases"))
    lines = [
        "两场景资源报告（只读展示；未核验报告签名或重新计时）",
        f"版本={_text(report.get('version'))}  模型={_text(report.get('timing_model'))}  "
        f"状态={_text(report.get('status'))}  模式={_text(report.get('mode'))}",
        f"正式 score={_value(report.get('score'))}  "
        f"候选 experimental_score={_value(report.get('experimental_score'))}",
        "单位：周期=模拟时钟周期；能量=pJ；功率=W；面积=mm²；流量=B。",
        "busy 为报告给出的实体·周期；利用率为报告给出的容量归一化百分比。",
    ]
    for case in CASES:
        item = _object(cases.get(case))
        timing = _object(item.get("timing"))
        stats = _object(timing.get("resource_stats"))
        shared = _object(stats.get("shared"))
        lines.extend(
            [
                "",
                f"[{case}] 功能通过={_passed(item.get('functional_passed'))}",
                f"  完成={_value(timing.get('cycles'), '周期')}  "
                f"关键完成={_value(_object(shared.get('critical_completion')).get('cycle'), '周期')}",
                f"  面积={_value(timing.get('area_mm2'), 'mm²')}  "
                f"动态能量={_value(timing.get('dynamic_energy_pj'), 'pJ')}  "
                f"总能量={_value(timing.get('total_energy_pj'), 'pJ')}",
                f"  平均功率={_value(timing.get('average_power_w'), 'W')}  "
                f"峰值窗口功率={_value(timing.get('peak_window_power_w'), 'W')}",
                f"  HBM 读取={_value(timing.get('hbm_read_bytes'), 'B')}  "
                f"HBM 写入={_value(timing.get('hbm_write_bytes'), 'B')}  "
                f"NoC 传输={_value(shared.get('noc_bytes'), 'B')}",
                f"  Cache 查询={_value(shared.get('cache_queries'))}  "
                f"命中={_value(shared.get('cache_hits'))}  "
                f"未命中={_value(shared.get('cache_misses'))}  "
                f"直通读取={_value(shared.get('cache_bypass_reads'), '条')}  "
                f"读取={_value(shared.get('cache_read_bytes'), 'B')}  "
                f"填充={_value(shared.get('cache_fill_bytes'), 'B')}",
            ]
        )
        sms = _object(stats.get("sms"))
        for sm in sorted(sms, key=lambda key: (not str(key).isdigit(), str(key))):
            sm_stats = _object(sms[sm])
            lines.append(
                f"  SM {sm} 计数: 指令={_value(sm_stats.get('instructions'))}; "
                f"发射等待={_value(sm_stats.get('issue_wait_cycles'), '周期')}; "
                f"DMA={_value(sm_stats.get('dma_bytes'), 'B')}; "
                f"RF 读={_value(sm_stats.get('rf_read_bytes'), 'B')}; "
                f"RF 写={_value(sm_stats.get('rf_write_bytes'), 'B')}"
            )
            lines.extend(_resource_lines(f"SM {sm}", sms[sm]))
        lines.extend(_resource_lines("共享资源", shared))
        intervals = _object(shared.get("reservation_intervals"))
        if intervals:
            for name in sorted(intervals):
                lines.append(f"  预约区间 {name}={_interval_count(intervals[name])}")
        else:
            lines.append(f"  预约区间: {MISSING}")
        lines.extend(_timeline_lines(stats.get("timeline")))
    return "\n".join(lines) + "\n"


def render_file(path: Path) -> str:
    return render_report(json.loads(path.read_text(encoding="utf-8")))


def _power_svg(values: list[int | float]) -> str:
    top = max(values)
    bottom = min(values)
    span = max(top - bottom, 1e-12)
    count = len(values)
    points = " ".join(
        f"{index * 600 / max(count - 1, 1):.2f},{110 - (value - bottom) * 100 / span:.2f}"
        for index, value in enumerate(values)
    )
    return (
        '<svg viewBox="0 0 600 120" role="img" aria-label="每箱平均功率 W">'
        '<polyline fill="none" stroke="#1d6f91" stroke-width="2" points="' + points + '"/></svg>'
    )


def _heatmap_html(timeline: dict, edges: list[int]) -> str:
    bins = len(edges) - 1
    rows = []
    scopes = [(f"SM {sm}", value) for sm, value in sorted(_object(timeline.get("sms")).items())]
    scopes.append(("共享", _object(timeline.get("shared"))))
    for label, scope in scopes:
        scope = _object(scope)
        for ratio_field, value_field, ratio_label, value_label, unit in (
            ("utilization", "busy_cycles", "利用率", "busy", "实体·周期"),
            ("bandwidth_utilization", "bytes", "带宽利用率", "传输", "B"),
        ):
            values_by_resource = _object(scope.get(value_field))
            for resource, values in sorted(_object(scope.get(ratio_field)).items()):
                series = _series({resource: values}, resource, bins)
                if series is None:
                    continue
                measured = _series(values_by_resource, resource, bins)
                cells = []
                for index, used in enumerate(series):
                    level = min(8, max(0, math.ceil(used * 8)))
                    detail = (
                        f"{label} {resource}; [{edges[index]},{edges[index + 1]}) 周期; "
                        f"{ratio_label} {_ratio(used)}; "
                        f"{value_label} {_value(measured[index] if measured else None, unit)}"
                    )
                    cells.append(
                        f'<td class="level{level}" title="{escape(detail, quote=True)}"></td>'
                    )
                rows.append(
                    f"<tr><th>{escape(label)} {escape(str(resource))} "
                    f"{ratio_label}</th>{''.join(cells)}</tr>"
                )
    return '<table class="heat"><tbody>' + "".join(rows) + "</tbody></table>"


def render_html(report: dict) -> str:
    """Static HTML from recorded bins; all report text is escaped."""
    if not isinstance(report, dict):
        raise ValueError("Challenge report must be a JSON object")
    sections = []
    for case in CASES:
        stats = _object(
            _object(_object(_object(report.get("cases")).get(case)).get("timing")).get(
                "resource_stats"
            )
        )
        timeline = _object(stats.get("timeline"))
        edges = _edges(timeline)
        if edges is None:
            sections.append(f"<section><h2>{escape(case)}</h2><p>无有效时间箱数据。</p></section>")
            continue
        bins = len(edges) - 1
        power = _object(timeline.get("power"))
        average = _series(power, "average_w", bins)
        energy = _series(power, "dynamic_energy_pj", bins)
        noc = _series(_object(_object(timeline.get("shared")).get("bytes")), "noc", bins)
        rows = []
        for index, (start, end) in enumerate(zip(edges, edges[1:])):
            rows.append(
                f"<tr><td>{index}</td><td>[{start},{end})</td>"
                f"<td>{_value(average[index] if average else None)}</td>"
                f"<td>{_value(energy[index] if energy else None)}</td>"
                f"<td>{_value(noc[index] if noc else None)}</td></tr>"
            )
        curve = _power_svg(average) if average else "<p>平均功率缺失。</p>"
        sections.append(
            f"<section><h2>{escape(case)}：{bins} 个时间箱</h2>"
            "<p>区间为 [起,止) 模拟周期；曲线是每箱平均功率，热图是每箱资源利用率。"
            "箱内事件先后次序未显示；精确窗口峰值另见原报告。</p>"
            + curve
            + _heatmap_html(timeline, edges)
            + '<table class="bins"><thead><tr><th>箱</th><th>周期区间</th>'
            "<th>平均功率 W</th><th>动态能量 pJ</th><th>NoC B</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></section>"
        )
    title = "两场景资源与功率时间箱"
    summary = escape(render_report(report))
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f"<title>{title}</title><style>"
        "body{font:14px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17252f}"
        "section{margin:2rem 0}svg{display:block;width:100%;height:160px;background:#f3f8fa}"
        "table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #ccd5db;padding:.25rem .4rem}"
        ".heat td{width:5px;min-width:5px;padding:0;height:16px}.heat th{text-align:left;white-space:nowrap}"
        ".bins td{text-align:right}.level0{background:#edf2f4}.level1{background:#d2e9f1}"
        ".level2{background:#b4dbe9}.level3{background:#8cc9df}.level4{background:#62b5d2}"
        ".level5{background:#3b9ebf}.level6{background:#287f9e}.level7{background:#1b617e}"
        ".level8{background:#11445c}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "</style><h1>"
        + title
        + "</h1><p>从已保存 JSON 只读生成；未核验签名或重新计时。单位与缺失值见文本详情。</p>"
        + "".join(sections)
        + f"<details><summary>完整文本资源表</summary><pre>{summary}</pre></details></html>"
    )


def write_html(path: Path, report: dict) -> None:
    """Never replace an existing resource view."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(render_html(report))

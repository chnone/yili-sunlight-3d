"""Summarize all sampled windows of each model dwelling, preserving disjoint daylight."""
import csv
import os
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from sun_model import household_addresses, targets


def assign_addresses(model, rows):
    """Add entrance and within-entrance door while preserving the original frontage key."""
    addresses = {b["id"]: household_addresses(b, model.get("selected_unit_layout")) for b in targets(model)}
    for row in rows:
        row["单元"], row["单元内户位"] = addresses[row["楼号"]][row["户位"]]


def _clock_minute(value):
    h, m = map(int, value.split(":"))
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"无效时刻 {value}")
    return h * 60 + m


def _format_minute(value):
    return f"{value//60:02}:{value%60:02}"


def _combined_intervals(rows, day):
    """Sweep endpoints of half-open runs. At shared boundaries use the net change."""
    events = defaultdict(int)
    for row in rows:
        text = row[day+"时段"]
        total = 0
        previous = -1
        for item in ([] if text == "无" else text.split(" / ")):
            start, end = map(_clock_minute, item.split("–"))
            if start < previous or end <= start:
                raise ValueError(f"{day} 时段重复、逆序或为空")
            events[start] += 1
            events[end] -= 1
            total += end - start
            previous = end
        if total != int(row[day+"分钟"]):
            raise ValueError(f"{day} 单点时段与累计分钟不符")
    together, any_lit = [], []
    active, previous = 0, None
    for time, change in sorted(events.items()):
        if previous is not None and active > 0:
            groups = (any_lit, together) if active == len(rows) else (any_lit,)
            for runs in groups:
                if runs and runs[-1][1] == previous:
                    runs[-1] = (runs[-1][0], time)
                else:
                    runs.append((previous,time))
        active += change
        previous = time
    return together, any_lit


def summarize_households(point_rows):
    groups = defaultdict(list)
    for r in point_rows:
        groups[r["楼号"], int(r["住宅楼层"]), r["户位"]].append(r)
    summaries = []
    for (building, floor, unit), rows in groups.items():
        labels = [r["测点"] for r in rows]
        if len(set(labels)) != len(rows) or len(rows) < 2:
            raise ValueError(f"{building} {floor}层{unit}须有至少两个不同测点")
        summary = {"楼号": building, "分期": rows[0]["分期"], "住宅楼层":floor, "户位":unit,
                   "测点数":len(rows), "测点编号":" / ".join(labels)}
        for day in ("大寒","冬至","春分"):
            field = day+"分钟"
            times = [int(r[field]) for r in rows]
            weak, strong = min(times), max(times)
            summary.update({
                day+"较弱点分钟":weak, day+"较弱点小时":f"{weak/60:.4f}",
                day+"较弱测点":" / ".join(p for p,t in zip(labels,times) if t == weak),
                day+"较好点分钟":strong, day+"点间差分钟":strong-weak,
            })
            if day != "春分":
                summary[day+"选层筛查分钟"] = min(int(r[day+"选层筛查分钟"]) for r in rows)
            together, any_lit = _combined_intervals(rows,day)
            for prefix,runs in (("全部点同时直射",together),("至少一点直射",any_lit)):
                summary[day+prefix+"分钟"] = sum(end-start for start,end in runs)
                summary[day+prefix+"时段"] = " / ".join(
                    f"{_format_minute(start)}–{_format_minute(end)}" for start,end in runs) or "无"
        passed = sum(int(r["大寒分钟"]) >= 120 for r in rows)
        summary.update({"大寒满2小时测点数":passed,"大寒全点均满2小时":"是" if passed==len(rows) else "否",
                        "计算年份":rows[0]["计算年份"], "模型校验码":rows[0]["模型校验码"]})
        if "单元" in rows[0]:
            summary.update({field: rows[0][field] for field in ("单元", "单元内户位")})
        summaries.append(summary)
    return summaries


def export_households(point_rows, output_dir):
    summaries = summarize_households(point_rows)
    if not summaries:
        raise ValueError("缺少逐点明细，无法生成户汇总")
    target = Path(output_dir)/"sunlight_households.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile("w",dir=target.parent,prefix=".households-",suffix=".tmp",
                                encoding="utf-8-sig",newline="",delete=False) as stream:
            temp = Path(stream.name)
            writer = csv.DictWriter(stream,fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        os.replace(temp,target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return target

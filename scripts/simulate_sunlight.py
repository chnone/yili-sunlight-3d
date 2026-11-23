"""Shoot direct-sun rays at representative windows and write sunlight_details.csv.

Times in the data are Beijing civil time (UTC+8). Each half-open five-minute
interval uses its midpoint. Durations cover the selected clock window only.
"""
import argparse
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from sun_engine import Mesh, lit_intervals, solar_position, time_bins
from sun_model import DAYS, MODEL, OUTPUT, height, prisms, read_model, receivers, model_digest
from sun_households import assign_addresses, export_households


ZONE = timezone(timedelta(hours=8))
COLUMNS = (
    "楼号", "分期", "住宅楼层", "户位", "测点", "X_m", "Y_m", "窗台标高_m",
    "大寒分钟", "大寒小时", "大寒时段", "大寒主要遮挡",
    "冬至分钟", "冬至小时", "冬至时段", "冬至主要遮挡",
    "春分分钟", "春分小时", "春分时段", "大寒2小时筛查",
    "大寒1分钟复核", "冬至1分钟复核", "春分1分钟复核",
    "大寒选层筛查分钟", "冬至选层筛查分钟",
    "窗位对照偏移_m", "大寒偏西分钟", "大寒偏东分钟", "冬至偏西分钟", "冬至偏东分钟",
    "对照层高_m", "大寒层高对照分钟", "冬至层高对照分钟",
    "大寒保守分钟", "大寒乐观分钟", "冬至保守分钟", "冬至乐观分钟", "计算年份", "模型校验码",
    "单元", "单元内户位",
)


def scenario_mesh(model, nominal_mesh, bound):
    """Keep receivers fixed. Vary only the specified heights of opaque solids."""
    def endpoint(body, nominal, label):
        limits = body.get("roof_height_range")
        if limits is None:
            return nominal
        if len(limits) != 2 or not (limits[0] <= nominal <= limits[1]) or not np.isfinite(limits).all():
            raise ValueError(f"{label} 的 roof_height_range 须包括基准高度且为有限值")
        return limits[bound]

    buildings = []
    changed = False
    for b in model["buildings"]:
        nominal = height(b)
        alternative = endpoint(b, nominal, b["id"])
        parts = []
        for part in b.get("parts", []):
            part_top = endpoint(part, part["top_height"], b["id"])
            changed |= part_top != part["top_height"]
            parts.append(dict(part, top_height=part_top))
        if alternative != nominal and b["is_target"]:
            highest_sill = b["stilt_height"] + (b["floors"] - 1) * b["floor_height"] + .9
            if alternative <= highest_sill:
                raise ValueError(f'{b["id"]} 的屋顶高度区间与最高窗台冲突')
        changed |= alternative != nominal
        buildings.append(dict(b, roof_height=alternative, parts=parts))
    return Mesh.from_prisms(prisms(dict(model, buildings=buildings))) if changed else nominal_mesh


def blocker_summary(row, mesh, weights):
    """Two most frequent nearest blockers, weighted by bin length. Exclude no incidence."""
    blocked = row >= 0
    if not blocked.any():
        return "无"
    counts = np.bincount(row[blocked], weights=weights[blocked], minlength=len(mesh.ids))
    ordered = sorted(np.flatnonzero(counts), key=lambda i: (-counts[i], str(mesh.ids[i])))[:2]
    return " / ".join(f"{mesh.ids[i]}（{counts[i]:.0f}分钟）" for i in ordered)


def count_sun(mesh, points, normals, directions, weights):
    minutes = np.zeros(len(points), dtype=int)
    for direction, weight in zip(directions, weights):
        minutes += (mesh.classify(points, normals, direction) == -1) * weight
    return minutes


def compute_rows(model, year):
    ids = {b["id"]: b for b in model["buildings"]}
    keys, coordinates, normals = receivers(model)
    if not keys:
        raise ValueError("模型中缺少有住宅层数的目标楼栋")
    shift = model["site"].get("probe_shift_check_m", 0.5)
    if not np.isfinite(shift) or shift < 0:
        raise ValueError("窗位对照偏移距离必须为非负有限值")
    sides = [("偏西", *receivers(model, frontage_shift=-shift)[1:]),
             ("偏东", *receivers(model, frontage_shift=shift)[1:])]
    alternate_height = model["site"].get("sill_comparison_floor_height_m", 3.1)
    comparison = coordinates.copy()
    for i, (bid, floor, _, _) in enumerate(keys):
        comparison[i,2] += (alternate_height - ids[bid]["floor_height"]) * (floor - 1)
        if comparison[i,2] >= ids[bid]["ground_elevation"] + height(ids[bid]):
            comparison[i,2] = np.nan
    valid = np.isfinite(comparison[:,2])
    mesh = Mesh.from_prisms(prisms(model))
    scenarios = {"保守": scenario_mesh(model, mesh, 1), "乐观": scenario_mesh(model, mesh, 0)}
    rows = [{"楼号": bid, "分期": ids[bid].get("phase", "待核"), "住宅楼层": floor, "户位": unit,
             "测点": label, "X_m": f"{coordinates[index, 0]:.3f}", "Y_m": f"{coordinates[index, 1]:.3f}",
             "窗台标高_m": f"{coordinates[index, 2]:.2f}"}
            for index, (bid, floor, unit, label) in enumerate(keys)]
    latitude, longitude = model["site"]["latitude"], model["site"]["longitude"]
    for day, (month, date, start, end) in DAYS.items():
        bins = time_bins(datetime(year, month, date, start, tzinfo=ZONE),
                         datetime(year, month, date, end, tzinfo=ZONE))
        weights = np.array([(b-a).total_seconds() / 60 for a,b,_ in bins], dtype=int)
        directions = [solar_position(mid, latitude, longitude).direction for _, _, mid in bins]
        states = np.column_stack([mesh.classify(coordinates, normals, direction) for direction in directions])
        minutes = (states == -1) @ weights
        check_bins = time_bins(bins[0][0], bins[-1][1], step_minutes=1)
        check_suns = (solar_position(t, latitude, longitude).direction for _, _, t in check_bins)
        minute_check = count_sun(mesh, coordinates, normals, check_suns, [1]*len(check_bins))
        for index, record in enumerate(rows):
            record[day+"分钟"] = int(minutes[index])
            record[day+"1分钟复核"] = int(minute_check[index])
            record[day+"小时"] = f"{minutes[index]/60:.4f}"
            periods = lit_intervals(states[index] == -1, bins)
            record[day+"时段"] = " / ".join(f"{a:%H:%M}–{b:%H:%M}" for a,b in periods) or "无"
            if day != "春分":
                record[day+"主要遮挡"] = blocker_summary(states[index], mesh, weights)
        if day == "春分":
            continue
        combined = np.minimum(minutes, minute_check)
        for side, shifted_xyz, shifted_normals in sides:
            values = count_sun(mesh, shifted_xyz, shifted_normals, directions, weights)
            tall_mesh = scenarios["保守"]
            high_side = values if tall_mesh is mesh else count_sun(
                tall_mesh, shifted_xyz, shifted_normals, directions, weights)
            combined = np.minimum(combined, high_side)
            for record, value in zip(rows, values):
                record["窗位对照偏移_m"] = shift
                record[day+side+"分钟"] = int(value)
        alternate_minutes = count_sun(mesh, comparison[valid], normals[valid], directions, weights)
        for record in rows:
            record["对照层高_m"] = alternate_height
            record[day+"层高对照分钟"] = ""
        for i, minutes_count in zip(np.flatnonzero(valid), alternate_minutes):
            rows[i][day+"层高对照分钟"] = int(minutes_count)
        for scenario_name, scenario in scenarios.items():
            if scenario is mesh:
                values = minutes
            else:
                values = count_sun(scenario, coordinates, normals, directions, weights)
            if scenario_name == "保守":
                combined = np.minimum(combined, values)
            if ((scenario_name == "保守" and np.any(values > minutes)) or
                    (scenario_name == "乐观" and np.any(values < minutes))):
                raise RuntimeError("遮挡高度情景的日照分钟数次序异常，请核验测点与遮挡体关系")
            for record, value in zip(rows, values):
                record[day+scenario_name+"分钟"] = int(value)
        for record, value in zip(rows, combined):
            record[day+"选层筛查分钟"] = int(value)
    digest = model_digest(model)
    for record in rows:
        record["计算年份"] = year
        record["模型校验码"] = digest
        record["大寒2小时筛查"] = "是" if record["大寒分钟"] >= 120 else "否"
    assign_addresses(model, rows)
    return rows


def export(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "sunlight_details.csv"
    temp = None
    try:
        with NamedTemporaryFile("w", dir=output_dir, prefix=".sunlight-", suffix=".tmp",
                                encoding="utf-8-sig", newline="", delete=False) as stream:
            temp = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, target)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL, help="楼栋模型 JSON")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="CSV 输出目录")
    parser.add_argument("--year", type=int, help="公历年，默认使用模型 site.year")
    args = parser.parse_args()
    model = read_model(args.model)
    year = args.year if args.year is not None else model["site"]["year"]
    rows = compute_rows(model, year)
    path = export(rows, args.output_dir)
    household_path = export_households(rows, args.output_dir)
    print(f"{year} 年，{len(rows)} 个南窗测点，已写入 {path} 和 {household_path}")


if __name__ == "__main__":
    main()

"""Build the shared ENU geometry and per-floor direct-sun state for animated views."""
import argparse
import csv
import json
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np

from simulate_sunlight import ZONE
from sun_engine import Mesh, lit_intervals, solar_position
from sun_model import DAYS, MODEL, OUTPUT, household_addresses, model_digest, prisms, read_model, receivers, solid_parts, targets


def _scene_building(building):
    ground = building["ground_elevation"]
    main, *parts = solid_parts(building)
    return {
        "id": building["id"],
        "name": building.get("chart_name", building["id"]),
        "group": building.get("group", "residential" if building["is_target"] else "external"),
        **({"kind": building["kind"]} if "kind" in building else {}),
        "is_target": building["is_target"],
        "estimated": bool(building.get("estimated", False)),
        "phase": building.get("phase", ""),
        "floors": building["floors"],
        **main,
        "sill_base": ground + building["stilt_height"] + 0.9 if building["is_target"] else None,
        "floor_height": building["floor_height"],
        "height_review": {
            **building.get("height_review", {}),
            "range": building.get("roof_height_range"),
        },
        "parts": parts,
    }


def _csv_points(path, keys, xyz, digest, year):
    expected = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["楼号"], int(row["住宅楼层"]), row["户位"], row["测点"]
            if key in expected:
                raise ValueError(f"逐点明细出现重复记录 {key}")
            if row["模型校验码"] != digest or int(row["计算年份"]) != year:
                raise ValueError("楼栋模型、年份与逐点明细不一致，请先按同一模型及年份运行 simulate_sunlight.py")
            expected[key] = row
    if expected.keys() != set(keys) or len(keys) != len(set(keys)):
        raise ValueError("逐点明细与动画使用的住宅楼层、户位或测点集合不一致")
    records = [expected[k] for k in keys]
    for key, row, pos in zip(keys, records, xyz):
        fields = ["X_m", "Y_m", "窗台标高_m"]
        if any(row[c] != f"{v:.{precision}f}"
               for c, v, precision in zip(fields, pos, (3, 3, 2))):
            raise ValueError(f"逐点明细坐标与动画模型不一致 {key}")
    return records


def _clock_minutes(moment):
    whole_minutes = 60 * moment.hour + moment.minute
    seconds = moment.second + moment.microsecond / 1e6
    return whole_minutes + seconds / 60 if seconds else whole_minutes


def _solar_knot(moment, sun):
    return {"minute": _clock_minutes(moment), "direction": np.round(sun.direction, 9).tolist()}


def _daylight_bins(midnight, latitude, longitude):
    """Local sunrise/sunset at upper-limb altitude, with geometric-centre ray tests.

    NOAA's apparent rise/set convention uses a geometric solar centre at −0.833°.
    https://gml.noaa.gov/grad/solcalc/calcdetails.html
    Bracket around local morning and afternoon for the Tianjin site. Crossing times
    are stored to the nearest second; the clock's five-minute grid stays fixed.
    """
    def level_at(second):
        moment = midnight + timedelta(seconds=second)
        return solar_position(moment, latitude, longitude).altitude

    def crossing(rising, target):
        left, right = (0, 43200) if rising else (43200, 86400)
        if (level_at(left) < target) != rising or (level_at(right) >= target) != rising:
            raise ValueError(f"{midnight:%Y-%m-%d} 该经纬度在北京时间上下午未分别跨越 {target}°")
        # Stop with adjacent civil seconds, then use the nearer endpoint.
        while right - left > 1:
            middle = (left + right) // 2
            if (level_at(middle) < target) == rising:
                left = middle
            else:
                right = middle
        return min((left, right), key=lambda s: abs(level_at(s) - target))

    sunrise, sunset = (crossing(rising, -.833) for rising in (True, False))
    geometric_rise, geometric_set = (crossing(rising, 0.) for rising in (True, False))
    edges = {sunrise, sunset, geometric_rise, geometric_set}
    # Include both zero-altitude crossings so no bin contains both a positive and
    # negative geometric altitude. Every interior five-minute civil bin keeps its
    # normal midpoint, including all bins in the original CSV analysis windows.
    edges.update(range((sunrise // 300 + 1) * 300, sunset, 300))
    times = [midnight + timedelta(seconds=second) for second in sorted(edges)]
    return [(a, b, a + (b-a)/2) for a, b in zip(times, times[1:])]


def _household_leaders(keys, lit_seconds, limit=5, *, excluded_buildings=()):
    """Return the top dwellings by mean full-day probe duration, including cutoff ties."""
    groups = {}
    excluded = set(excluded_buildings)
    for (bid, floor, unit, label), seconds in zip(keys, lit_seconds, strict=True):
        if bid in excluded:
            continue
        groups.setdefault((bid, floor, unit), {})[label] = int(seconds)
    scores = {home: Fraction(sum(probes.values()), len(probes)) for home, probes in groups.items()}
    rows = []
    last_score = None
    for place, home in enumerate(sorted(groups, key=scores.get, reverse=True), 1):
        score = scores[home]
        if score != last_score:
            if place > limit:
                break
            rank = place
            last_score = score
        bid, floor, unit = home
        rows.append({"rank": rank, "bid": bid, "floor": floor, "unit": unit,
                     "probe_seconds": groups[home], "mean_seconds": float(score)})
    return {"households": len(groups), "rows": rows, "excluded_buildings": list(excluded_buildings)}


def build_scene(model, year, details_path):
    """Return JSON-ready animation data; check every exported point's runs against the CSV.

    Each frame describes a half-open civil-time bin from apparent rise to apparent set.
    Its direction is geometric ENU at the bin midpoint. solar_path contains actual ENU
    vectors at the day boundaries, every bin edge and every midpoint for playback.
    masks[i] stores point i's visible floors with the least significant bit for
    floor 1. A mask beyond JavaScript's
    safe integer range is emitted as a decimal string (BigInt in the viewer).
    """
    keys, xyz, normals = receivers(model)
    if not keys:
        raise ValueError("场景缺少住宅窗台测点")
    digest = model_digest(model)
    records = _csv_points(details_path, keys, xyz, digest, year)
    points = []
    addresses = {b["id"]: household_addresses(b, model.get("selected_unit_layout")) for b in targets(model)}
    index_by_station = {}
    row_station = []
    row_bit = []
    for (bid, floor, unit, label), pos in zip(keys, xyz):
        key = bid, unit, label
        if key not in index_by_station:
            index_by_station[key] = len(points)
            entry, door = addresses[bid][unit]
            points.append({"bid": bid, "unit": unit, "entry": entry, "door": door, "label": label,
                           "x": round(float(pos[0]), 5), "y": round(float(pos[1]), 5)})
        row_station.append(index_by_station[key])
        row_bit.append(1 << (floor - 1))

    day_data = {}
    mesh = Mesh.from_prisms(prisms(model))
    buildings = [_scene_building(b) for b in model["buildings"]]
    by_id = {b["id"]: b for b in buildings}
    excluded = model["site"].get("household_ranking_excluded_buildings", [])
    if not isinstance(excluded, list) or any(not isinstance(bid, str) or bid not in by_id for bid in excluded):
        raise ValueError("排名排除名单须为模型中已知的楼号列表")
    # Renderer indices follow JSON order even if mesh construction order changes.
    display_index = {b["id"]: i for i, b in enumerate(buildings)}
    owner_index = np.array([display_index[bid] for bid in mesh.ids], dtype=np.int32)
    latitude, longitude = model["site"]["latitude"], model["site"]["longitude"]
    for day, (month, date, begin, end) in DAYS.items():
        midnight = datetime(year, month, date, tzinfo=ZONE)
        standard_start = midnight + timedelta(hours=begin)
        standard_end = midnight + timedelta(hours=end)
        bins = _daylight_bins(midnight, latitude, longitude)
        standard = [i for i, (a, b, _) in enumerate(bins)
                    if standard_start <= a and b <= standard_end]
        standard_bins = [bins[i] for i in standard]
        frames = []
        solar_path = []
        history = [[] for _ in keys]
        previous = np.full(len(keys), -3, dtype=np.int32)
        first_hit_seconds = np.zeros(len(mesh.ids), dtype=np.int64)
        lit_seconds = np.zeros(len(keys), dtype=np.int64)
        standard_seconds = np.zeros_like(first_hit_seconds)
        for frame_index, (start, stop, mid) in enumerate(bins):
            sun = solar_position(mid, latitude, longitude)
            solar_path.append(_solar_knot(start, solar_position(start, latitude, longitude)))
            solar_path.append(_solar_knot(mid, sun))
            hits = mesh.classify(xyz, normals, sun.direction)
            for index in np.flatnonzero(hits != previous):
                hit = int(hits[index])
                history[index].append([frame_index, int(owner_index[hit]) if hit >= 0 else hit])
            previous = hits
            duration = int((stop - start).total_seconds())
            lit_seconds[hits == -1] += duration
            blocked_seconds = duration * np.bincount(hits[hits >= 0], minlength=len(mesh.ids))
            first_hit_seconds += blocked_seconds
            if standard_start <= start and stop <= standard_end:
                standard_seconds += blocked_seconds
            masks = [0] * len(points)
            for index in np.flatnonzero(hits == -1):
                masks[row_station[index]] |= row_bit[index]
            frames.append({
                "start": _clock_minutes(start), "end": _clock_minutes(stop),
                "time": mid.strftime("%H:%M:%S"), "direction": np.round(sun.direction, 9).tolist(),
                "altitude": round(sun.altitude, 5), "azimuth": round(sun.azimuth, 5),
                "masks": [mask if mask < 2**53 else str(mask) for mask in masks],
            })
        sunset = bins[-1][1]
        solar_path.append(_solar_knot(sunset, solar_position(sunset, latitude, longitude)))
        # Validate the encoded statutory bins against the same CSV as the heatmaps.
        for key, row, station, bit in zip(keys, records, row_station, row_bit):
            visible = [bool(int(frames[i]["masks"][station]) & bit) for i in standard]
            runs = lit_intervals(visible, standard_bins)
            periods = " / ".join(f"{a:%H:%M}–{b:%H:%M}" for a, b in runs) or "无"
            minutes = sum((b-a).total_seconds() for a, b in runs) / 60
            if periods != row[day+"时段"] or minutes != int(row[day+"分钟"]):
                raise ValueError(f"{key} 在{day}的动画见光时段或分钟与逐点明细不一致")
        blockers = [[None] * by_id[p["bid"]]["floors"] for p in points]
        for station, (_, floor, _, _), runs in zip(row_station, keys, history):
            blockers[station][floor - 1] = runs
        day_data[day] = {"date": midnight.date().isoformat(), "start": frames[0]["start"],
                         "end": frames[-1]["end"], "standard_start": 60*begin, "standard_end": 60*end,
                         "solar_path": solar_path, "frames": frames, "blockers": blockers,
                         "household_leaders": _household_leaders(keys, lit_seconds, excluded_buildings=excluded)}
        for bid, seconds, standard_value in zip(mesh.ids, first_hit_seconds, standard_seconds):
            by_id[bid].setdefault("shadow_impact", {})[day] = round(int(seconds) / 60, 3)
            by_id[bid].setdefault("standard_shadow_impact", {})[day] = int(standard_value // 60)
    return {
        "year": year, "step": 5, "model_digest": digest,
        "site_polygon": model["site_polygon"],
        "buildings": buildings,
        "points": points, "days": day_data,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--year", type=int)
    parser.add_argument("--details", type=Path, default=OUTPUT / "sunlight_details.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = read_model(args.model)
    year = args.year if args.year is not None else model["site"]["year"]
    scene = build_scene(model, year, args.details)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                        encoding="utf-8")
    print(f'{len(scene["points"])} 个平面窗位，{sum(len(d["frames"]) for d in scene["days"].values())} 帧 → {args.out}')


if __name__ == "__main__":
    main()

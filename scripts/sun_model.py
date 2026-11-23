"""Read the editable site model and place representative points on actual polygon edges."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "buildings.json"
CHARTS = ROOT / "charts"
OUTPUT = ROOT / "output"
DAYS = {"大寒": (1, 20, 8, 16), "冬至": (12, 21, 9, 15), "春分": (3, 21, 8, 16)}


def model_digest(model):
    # The CSV fingerprint tracks ray inputs. Display-only ranking preferences
    # may change without invalidating the saved sunlight intervals.
    if "household_ranking_excluded_buildings" in model.get("site", {}):
        site = model["site"].copy()
        del site["household_ranking_excluded_buildings"]
        model = {**model, "site": site}
    data = json.dumps(model, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def read_model(path=MODEL):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [b["id"] for b in model["buildings"]]
    if len(set(ids)) != len(ids):
        raise ValueError("楼栋 ID 必须唯一")
    return model


def height(b):
    """Height above local grade. A drawing height includes the roof and platform."""
    if b.get("roof_height") is not None:
        return float(b["roof_height"])
    return b["stilt_height"] + b["floors"] * b["floor_height"] + b["parapet_height"]


def solid_parts(b):
    """Absolute solid elevations shared by the ray tracer and animation."""
    ground = b["ground_elevation"]
    if b.get("kind") == "bridge" and ("base_height" not in b or b.get("roof_height") is None):
        raise ValueError(f'{b["id"]} 的主桥面须显式填写梁底 base_height 和桥面 roof_height')
    yield {"polygon": b["polygon"], "base": ground + b.get("base_height", 0), "top": ground + height(b),
           **({"centerline": b["centerline"]} if "centerline" in b else {})}
    for part in b.get("parts", []):
        yield {"polygon": part["polygon"], "base": ground + part["base_height"],
               "top": ground + part["top_height"],
               **{key: part[key] for key in ("kind", "centerline") if key in part}}


def prisms(model):
    for b in model["buildings"]:
        for part in solid_parts(b):
            yield b["id"], part["polygon"], part["base"], part["top"]


def building_sort(b):
    import re
    found = re.search(r"\d+", b["id"])
    return (int(found.group()) if found else math.inf), b["id"]


def targets(model):
    return sorted((b for b in model["buildings"] if b["is_target"]), key=building_sort)


def _fraction_list(values, field):
    if not isinstance(values, (list, tuple)) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values):
        raise ValueError(f"{field} 须为有限数值数组")
    return list(values)


def _ordered_interior(values):
    return all(0 < v < 1 for v in values) and all(a < b for a,b in zip(values, values[1:]))


def household_addresses(b, layout=None):
    """Map stable east-to-west frontage keys to (entrance, dwelling within entrance).

    The original frontage names also identify the probes in previously saved
    results. Entrance names describe the modeled entrances, not official numbers.
    """
    n = b["units_per_floor"]
    if type(n) is not int or n < 1:
        raise ValueError(f'{b["id"]} 的每层户数须为正整数')
    if n == 2:
        return {"东户": ("唯一单元", "东户"), "西户": ("唯一单元", "西户")}
    if n == 4:
        return {"东户": ("东单元", "东户"), "东中户": ("东单元", "西户"),
                "西中户": ("西单元", "东户"), "西户": ("西单元", "西户")}
    names = {1: ["整层户"], 3: ["东户", "中户", "西户"]}.get(n)
    if names is None:
        names = [f"东起{i+1}户" for i in range(n)]
    if n == 6 and layout == "north_six":
        pairs = [(entry, door) for entry in ("东单元", "中单元", "西单元") for door in ("东户", "西户")]
        return dict(zip(names, pairs, strict=True))
    return {name: ("", name) for name in names}


def unit_frontages(b):
    """Yield (frontage key, east boundary, west boundary, positions within that unit)."""
    names = household_addresses(b)
    n = len(names)
    boundaries = b.get("unit_boundaries_from_east")
    legacy = b.get("probe_fractions_from_east")
    if legacy is not None:
        legacy = _fraction_list(legacy, "probe_fractions_from_east")
        if len(legacy) != n or not _ordered_interior(legacy):
            raise ValueError("旧版测点须每户一个，按从东向西的全楼面宽比例递增")
        if boundaries is None:
            boundaries = [0, *[(a+b)/2 for a,b in zip(legacy,legacy[1:])], 1]
    if boundaries is None:
        boundaries = [j / n for j in range(n+1)]
    boundaries = _fraction_list(boundaries, "unit_boundaries_from_east")
    if (len(boundaries) != n+1 or boundaries[0] != 0 or boundaries[-1] != 1 or
            any(a >= b for a,b in zip(boundaries,boundaries[1:]))):
        raise ValueError(f'{b["id"]} 的户界须为从 0 到 1 严格递增的 {n + 1} 个比例')
    common = b.get("probe_fractions_within_unit", [.25, .75])
    positions = b.get("unit_probe_fractions", [common for _ in names])
    if not isinstance(positions, (list, tuple)) or len(positions) != n:
        raise ValueError("unit_probe_fractions 须按每层户数从东向西提供每户测点数组")
    for i, name in enumerate(names):
        east, west = boundaries[i:i+2]
        local = _fraction_list(positions[i], f'{b["id"]} {name}户内测点')
        if len(local) < 2 or not _ordered_interior(local):
            raise ValueError(f'{b["id"]} {name} 须有至少两个严格递增且在户界内的测点比例')
        if legacy is not None:
            reference = (legacy[i] - east) / (west - east)
            if not 0 < reference < 1:
                raise ValueError(f'{b["id"]} {name}旧版测点超出指定户界，请核对两种比例')
            # Preserve a manually digitized single-point location alongside the new pair.
            if not any(abs(v-reference) < 1e-10 for v in local):
                local = sorted([*local, reference])
        yield name, east, west, local


def _probe_label(index):
    label = ""
    while index >= 0:
        index, rem = divmod(index, 26)
        label = chr(ord("A") + rem) + label
        index -= 1
    return label


def facade_probes(b, offset=0.1, frontage_shift=0.0):
    """Project each unit's sampling stations to its outermost south-facing wall."""
    if not np.isfinite(frontage_shift) or not np.isfinite(offset) or offset < 0:
        raise ValueError("窗台偏移距离必须为有限值，外墙法向偏移须非负")
    p = np.array(b["polygon"], dtype=float)
    east, north = np.ptp(p, axis=0)
    if min(east, north) <= 0:
        raise ValueError(f'{b["id"]} 的轮廓无面积')
    if np.array_equal(p[0], p[-1]):
        p = p[:-1]
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    q = np.roll(p, -1, axis=0)
    edge = q - p
    lengths = np.linalg.norm(edge, axis=1)
    if np.any(lengths == 0) or area2 == 0:
        raise ValueError(f'{b["id"]} 的轮廓含退化边')
    normals = np.column_stack([edge[:, 1], -edge[:, 0]]) / lengths[:, None] * np.sign(area2)
    az = np.deg2rad(b.get("facade_azimuth", 180.0))
    south = np.array([np.sin(az), np.cos(az)])
    along = np.array([-south[1], south[0]])   # increasing towards the east
    t0, t1 = p @ along, q @ along
    def project(u):
        chosen = None
        for i in np.flatnonzero((normals @ south > 1.e-8) &
                                (u >= np.minimum(t0, t1) - 1.e-9) &
                                (u <= np.maximum(t0, t1) + 1.e-9)):
            point = p[i] + ((u - t0[i]) / (t1[i] - t0[i])) * edge[i]
            candidate = float(point @ south)
            if chosen is None or candidate > chosen[0]:
                chosen = candidate, point, normals[i]
        if chosen is None:
            raise ValueError(f'{b["id"]} 的测点无法投影到南向立面')
        _, point, norm = chosen
        return point + offset * norm, norm

    out = []
    width = np.ptp(t0)
    for name, east_frac, west_frac, local in unit_frontages(b):
        east = t0.max() - east_frac * width
        west = t0.max() - west_frac * width
        stations = east - np.asarray(local) * (east - west)
        # Move the window group together without crossing a household boundary or
        # merging two stations. Reserve at most half the existing clearance.
        gap = min(.0001, (east-stations[0])/2, (stations[-1]-west)/2)
        delta = np.clip(frontage_shift, west+gap-stations[-1], east-gap-stations[0])
        for i, station in enumerate(stations):
            xy, normal = project(station + delta)
            out.append((name, _probe_label(i), xy, normal))
    return out


def receivers(model, frontage_shift=0.0):
    keys, coords, normals = [], [], []
    for b in targets(model):
        probes = facade_probes(b, frontage_shift=frontage_shift)
        for k in range(1, b["floors"] + 1):
            z = b["stilt_height"] + (k - 1) * b["floor_height"] + .9
            if z >= height(b):
                raise ValueError(f'{b["id"]} 住宅第{k}层的窗台超出屋顶')
            for unit, label, xy, norm in probes:
                keys.append((b["id"], k, unit, label))
                coords.append([*xy, b["ground_elevation"] + z])
                normals.append(norm)
    return keys, np.array(coords), np.array(normals)

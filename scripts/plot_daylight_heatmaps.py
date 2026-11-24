"""Export sunrise-to-sunset heatmaps matching the animated viewer's two sampling intervals."""
import argparse
import csv
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from build_sun_animation import VIEWER
from build_sun_animation_data import _daylight_bins, _household_leaders, fine_daylight_seconds
from simulate_sunlight import ZONE
from sun_engine import Mesh
from sun_model import CHARTS, MODEL, OUTPUT, model_digest, prisms, read_model, receivers

DAYS_TO_PLOT = {"大寒": "04_dahan_daylight_heatmap.png", "冬至": "05_dongzhi_daylight_heatmap.png"}


def read_animation(path, model):
    text = path.read_text(encoding="utf-8")
    start_tag = '<script type="application/json" id="scene">'
    if text.count(start_tag) != 1:
        raise ValueError("找不到唯一的内嵌动画场景，请重新生成交互动画。")
    scene = json.loads(text.split(start_tag, 1)[1].split("</script>", 1)[0])
    if scene["model_digest"] != model_digest(model) or scene["step"] != 5:
        raise ValueError("模型或采样步长与动画不一致，请先用本模型重新生成动画。")
    return scene


def day_rows(scene, name, model, mesh, keys, xyz, normals):
    day = scene["days"][name]
    midnight = datetime.fromisoformat(day["date"]).replace(tzinfo=ZONE)
    lat, lon = model["site"]["latitude"], model["site"]["longitude"]
    bins = _daylight_bins(midnight, lat, lon)
    stations = {(p["bid"], p["unit"], p["label"]): i for i, p in enumerate(scene["points"])}
    if len(stations) != len(scene["points"]) or set(stations) != {(b, u, p) for b, _, u, p in keys}:
        raise ValueError("动画中的窗台与当前模型不一致。")
    frames = []
    for (a, b, _), frame in zip(bins, day["frames"], strict=True):
        lo, hi = round(frame["start"] * 60), round(frame["end"] * 60)
        if (lo, hi) != ((a - midnight).total_seconds(), (b - midnight).total_seconds()):
            raise ValueError(f"{name}的回放区间与模型日出日落时间不符。")
        masks = [int(value) for value in frame["masks"]]
        if len(masks) != len(stations):
            raise ValueError(f"{name}的窗台判光数据不全。")
        frames.append((hi - lo, frame["end"] - frame["start"], masks))

    fine = fine_daylight_seconds(mesh, xyz, normals, bins, lat, lon)
    leaders = day["household_leaders"]
    ranking = _household_leaders(keys, fine, excluded_buildings=leaders["excluded_buildings"])
    if (leaders["sample_minutes"] != 1 or ranking["rows"] != leaders["rows"]
            or ranking["households"] != leaders["households"]):
        raise ValueError(f"{name}的 1 分钟复算与动画榜单不一致。")

    homes = {}
    for (bid, floor, unit, label), seconds in zip(keys, fine, strict=True):
        index = stations[bid, unit, label]
        point = scene["points"][index]
        home = homes.setdefault((bid, floor, unit), {
            "bid": bid, "floor": floor, "unit": unit, "entry": point["entry"],
            "door": point["door"], "points": [],
        })
        floor_bit = 1 << (floor - 1)
        window_seconds = 0
        card_minutes = 0.0
        for duration, minutes, masks in frames:
            if masks[index] & floor_bit:
                window_seconds += duration
                card_minutes += minutes
        # Mirror the browser's ordered floating-point addition and Number(toFixed(1)).
        # The CSV also preserves the exact interval length in seconds.
        card_value = Decimal.from_float(card_minutes).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        home["points"].append({"label": label, "window_seconds": window_seconds,
                               "window_display": str(card_value).removesuffix(".0"), "fine_seconds": int(seconds)})
    for home in homes.values():
        points = home["points"]
        home["mean_seconds"] = sum(p["fine_seconds"] for p in points) / len(points)
    return homes


def save_table(results, path):
    # One address per row, so dates can be compared and filtered side by side.
    days = list(results)
    baseline = results[days[0]]
    if any(homes.keys() != baseline.keys() for homes in results.values()):
        raise ValueError("两天的模拟户位不一致。")
    records = []
    for key, home in baseline.items():
        row = {"楼栋": home["bid"], "住宅楼层": home["floor"], "单元": home["entry"], "单元内户位": home["door"]}
        for day in days:
            h = results[day][key]
            row[day + "_户均直射_小时"] = f'{h["mean_seconds"]/3600:.4f}'
            row[day + "_户均直射_秒"] = h["mean_seconds"]
            for p in h["points"]:
                prefix = day + "_" + p["label"]
                row[prefix + "_窗台卡5分钟采样_直射分钟"] = p["window_display"]
                row[prefix + "_窗台卡5分钟采样_直射秒"] = p["window_seconds"]
                row[prefix + "_榜单1分钟采样_直射秒"] = p["fine_seconds"]
        records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader()
        writer.writerows(records)
    print(path)


def main():
    from sun_daylight_chart import plot_daylight_heatmap
    from sun_plot_style import bridge_height_note

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, default=MODEL)
    cli.add_argument("--viewer", type=Path, default=OUTPUT / VIEWER)
    cli.add_argument("--charts-dir", type=Path, default=CHARTS)
    cli.add_argument("--table", type=Path, default=OUTPUT / "全天日照逐户明细.csv")
    args = cli.parse_args()
    model = read_model(args.model)
    scene = read_animation(args.viewer, model)
    buildings = [b for b in scene["buildings"] if b["is_target"]]
    mesh = Mesh.from_prisms(prisms(model))
    keys, xyz, normals = receivers(model)
    results = {name: day_rows(scene, name, model, mesh, keys, xyz, normals) for name in DAYS_TO_PLOT}
    for name, filename in DAYS_TO_PLOT.items():
        day = scene["days"][name]
        plot_daylight_heatmap(name, day["date"], round(day["start"]*60), round(day["end"]*60),
                              buildings, list(results[name].values()), args.charts_dir / filename,
                              bridge_note=bridge_height_note(model))
    save_table(results, args.table)


if __name__ == "__main__":
    main()

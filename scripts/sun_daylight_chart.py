"""Draw sunrise-to-sunset household heatmaps using the viewer's two sampling intervals."""
from collections import defaultdict
from itertools import groupby
import math
import re

import numpy as np

from sun_plot_style import MUTED, plt, save, setup
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase


def _duration(seconds):
    minutes = math.floor(seconds / 60 + .5)
    return f"{minutes // 60}:{minutes % 60:02}"


def _clock(seconds):
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def plot_daylight_heatmap(day, date, start_seconds, end_seconds, buildings, rows, path):
    setup()
    homes = defaultdict(list)
    for row in rows:
        homes[row["bid"]].append(row)
    widths = {}
    # Rows follow the east-to-west station order in the scene.
    for bid, records in homes.items():
        widths[bid] = list(dict.fromkeys((r["entry"], r["door"]) for r in records))
    order = sorted(buildings, key=lambda b: int(re.search(r"\d+", b["id"]).group()))
    bands = []
    band = []
    count = 0
    for building in order:
        width = len(widths[building["id"]])
        if band and (count + width > 10 or len(band) >= 4):
            bands.append(band)
            band, count = [], 0
        band.append(building)
        count += width
    if band:
        bands.append(band)

    palette = colors.LinearSegmentedColormap.from_list(
        "daylight_hours", ["#ab4341", "#e99c69", "#ffe0a3", "#dae8b5", "#87bca2", "#226d62"])
    norm = colors.Normalize(0, max(10, math.ceil(max(r["mean_seconds"] for r in rows) / 3600)))
    point_lines = math.ceil(max(len(r["points"]) for r in rows) / 2)
    cell_width, cell_height = 1.07, .42 + .13 * (point_lines - 1)
    floor_margin, gap = .33, .20
    left, right, header, band_header, band_gap, footer = .45, .45, 1.40, .73, .26, .47
    figure_width = left + right + max(
        sum(len(widths[b["id"]]) * cell_width + floor_margin for b in group) + gap * (len(group) - 1)
        for group in bands)
    band_heights = [max(b["floors"] for b in group) * cell_height + band_header for group in bands]
    figure_height = header + sum(band_heights) + band_gap * (len(bands) - 1) + footer
    fig = plt.figure(figsize=(figure_width, figure_height))

    def label(x, y, value, **style):
        fig.text(x / figure_width, 1 - y / figure_height, value, va="center", **style)

    label(left, .27, f"{day}  |  每层每户全天直射", fontsize=22)
    label(left, .61, f"天津河西 · 一里南侧地块　 {date}　 北京时间　"
          f"日出 {_clock(start_seconds)}　 日落 {_clock(end_seconds)}　 ·　 {len(order)} 栋 / {len(rows)} 户",
          fontsize=10, color=MUTED)
    label(left, .94, "大字为各南窗测点的平均时长（时:分），按 1 分钟复算并四舍五入到分钟。",
          fontsize=10)
    label(left, 1.18, "小字为动画窗台卡 A、B 点各自的全天直射（分钟），按 5 分钟采样。",
          fontsize=10)
    bar_width = 3.05
    bar_left = figure_width - right - bar_width
    label(bar_left, .85, "户均直射时长（小时）", fontsize=9, color=MUTED)
    bar_ax = fig.add_axes([bar_left / figure_width, 1 - 1.04 / figure_height,
                           bar_width / figure_width, .075 / figure_height])
    bar = ColorbarBase(bar_ax, cmap=palette, norm=norm, orientation="horizontal",
                       ticks=np.arange(0, norm.vmax + 1, 2))
    bar.outline.set_visible(False)
    bar_ax.tick_params(length=0, labelsize=8, pad=2)

    band_top = header
    for group, band_height in zip(bands, band_heights):
        x = left
        for b in group:
            bid, floors = b["id"], b["floors"]
            units = widths[bid]
            grid_x = x + floor_margin
            cols = len(units)
            width = cols * cell_width
            label(x, band_top + .13, bid, fontsize=14)
            label(x + floor_margin + width, band_top + .13, b["phase"],
                  ha="right", fontsize=8, color=MUTED)
            idx = 0
            for entry, doors in groupby(units, key=lambda addr: addr[0]):
                n = len(list(doors))
                label(grid_x + (idx + n / 2) * cell_width, band_top + .40, entry,
                      ha="center", fontsize=9, color=MUTED)
                idx += n
            for i, (_, door) in enumerate(units):
                label(grid_x + (i + .5) * cell_width, band_top + .60,
                      door, ha="center", fontsize=9)
            label(x, band_top + .60, "层", fontsize=8, color=MUTED)

            grid_top = band_top + band_header
            height = floors * cell_height
            ax = fig.add_axes([grid_x / figure_width, 1 - (grid_top + height) / figure_height,
                               width / figure_width, height / figure_height])
            matrix = np.full((floors, cols), np.nan)
            for r in homes[bid]:
                floor = r["floor"] - 1
                col = units.index((r["entry"], r["door"]))
                val = r["mean_seconds"] / 3600
                matrix[floor, col] = val
                red, green, blue = palette(norm(val))[:3]
                ink = "#ffffff" if .2126 * red + .7152 * green + .0722 * blue < .51 else "#162f33"
                ax.text(col, floor + .5 - .135 / cell_height, _duration(r["mean_seconds"]), fontsize=12.2,
                        color=ink, ha="center", va="center")
                captions = [f'{probe["label"]} {probe["window_display"]}' for probe in r["points"]]
                for j in range(0, len(captions), 2):
                    ypos = floor + .5 - (.29 + (j // 2) * .13) / cell_height
                    ax.text(col, ypos, "   ".join(captions[j:j+2]), fontsize=7.6,
                            color=ink, ha="center", va="center")
            ax.imshow(matrix, origin="lower", cmap=palette, norm=norm, interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks(range(floors), range(1, floors + 1))
            ax.tick_params(axis="y", labelsize=8, length=0, pad=4)
            for edge in range(1, cols):
                ax.axvline(edge - .5, color="white",
                           lw=1.65 if units[edge - 1][0] != units[edge][0] else .55)
            for edge in range(1, floors):
                ax.axhline(edge - .5, color="white", lw=.55)
            for spine in ax.spines.values():
                spine.set_visible(False)
            x += floor_margin + width + gap
        band_top += band_height + band_gap
    label(left, figure_height - .20, "住宅层从首个住宅层起算。12、13 号楼也列出全栋结果。"
          "计算采用模型南窗的晴天直射。", fontsize=9, color=MUTED)
    save(fig, path)

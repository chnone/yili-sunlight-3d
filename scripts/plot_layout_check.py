"""Plan check, in local east/north metres. The overlay uses the original PDF scan."""
import argparse
from itertools import groupby
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from sun_plot_style import setup, save, plt, MUTED, INK
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D

from sun_model import MODEL, CHARTS, ROOT, read_model, facade_probes, height, targets

BLUE = "#27639b"
LATER = "#72a6ce"
EXT = "#bbc4c6"
BRIDGE = "#557e99"
RED = "#c52839"


def polygon(ax, pts, **opts):
    ax.add_patch(Polygon(np.asarray(pts), closed=True, **opts))


def sheet(ax, pts, pad):
    pts = np.concatenate([np.asarray(p) for p in pts])
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(color="#e0e8e9", lw=.45)
    ax.tick_params(labelsize=8, length=2)
    ax.set_xlabel("正东 X（m）", fontsize=9)
    ax.set_ylabel("正北 Y（m）", fontsize=9)


def north(ax):
    ax.annotate("N", xy=(.96,.86), xytext=(.96,.73), xycoords="axes fraction",
                ha="center", va="top", fontsize=11, weight="bold",
                arrowprops={"arrowstyle":"-|>", "color":INK, "lw":1.5},
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85, "pad":1})


def official_overlay(ax, model, pdf_path):
    """Sample scan in ground coordinates via inverse affine; overlay independent model polygons."""
    affine = np.array(model["registration"]["pixel_to_enu"], dtype=float)
    site = np.asarray(model["site_polygon"])
    west, south = np.min(site, axis=0) - 5
    east, north_y = np.max(site, axis=0) + 5
    width = 1450
    h = (east - west) / width
    pixels = (width, round((north_y - south) / h))
    grid_to_xy = np.array([[h, 0, west], [0, -h, north_y], [0, 0, 1]])
    xy_to_source = np.linalg.inv(np.vstack([affine, [0,0,1]]))
    transform = (xy_to_source @ grid_to_xy)[:2].ravel()
    with tempfile.TemporaryDirectory() as tmp:
        root = str(Path(tmp) / "plan")
        subprocess.run(["pdfimages", "-f", "1", "-l", "1", "-j", str(pdf_path), root], check=True)
        with Image.open(root + "-000.jpg") as original:
            crop = original.convert("L").transform(pixels, Image.Transform.AFFINE, transform,
                                                    Image.Resampling.BILINEAR, fillcolor=255)
    ax.imshow(crop, extent=[west, east, south, north_y], cmap="gray", vmin=0, vmax=255)
    for b in targets(model):
        polygon(ax, b["polygon"], fill=False, edgecolor="#e84b29", linewidth=.8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def make(model, pdf, destination):
    from sun_geometry_check import check_layout
    check_layout(model)
    setup()
    buildings, site = model["buildings"], model["site_polygon"]
    houses = targets(model)
    external = [b for b in buildings if b.get("group") == "external" and b.get("kind") != "bridge"]
    bridges = [b for b in buildings if b.get("kind") == "bridge"]
    accessories = [b for b in buildings if not b["is_target"] and b.get("group") != "external"]
    fig = plt.figure(figsize=(16, 13), layout=None)
    fig.suptitle("一里南侧地块  |  楼位与遮挡校验", x=.056, y=.977, ha="left", fontsize=24)
    fig.text(.057,.942,"天津河西　·　2026.11.02 一期公示总平　/　面积按图校准，长度单位 m　/　相连红点属于同一模拟户", color=MUTED, fontsize=12)
    upper = fig.add_axes([.054,.585,.58,.31])
    lower = fig.add_axes([.055,.095,.56,.37])
    src = fig.add_axes([.663,.646,.31,.223])
    checks = fig.add_axes([.667,.51,.308,.115]); checks.axis("off")
    key = fig.add_axes([.655,.064,.329,.4]); key.axis("off")
    upper.set_title("住宅楼位、模型总高及标注间距", fontsize=14, pad=8)
    lower.set_title("周边遮挡体分布", fontsize=14, pad=8)
    src.set_title("公示原图叠合  /  橙线为模型外轮廓", fontsize=12, pad=8)
    sheet(upper, [site], 5.8)
    decks = [part for b in bridges for part in [b, *b.get("parts", [])] if part.get("kind") != "pier"]
    sheet(lower, [site] + [b["polygon"] for b in [*external, *decks]], 10)
    for part in decks:
        polygon(lower, part["polygon"], facecolor=BRIDGE, edgecolor="#345771", lw=.55)
        if part.get("centerline"):
            lower.plot(*np.array(part["centerline"]).T, lw=.5, color="white", linestyle=(0,(4,4)))
    for ax in (upper, lower):
        polygon(ax, site, facecolor="#f6fafc", edgecolor="#52747d", lw=1.1, linestyle=(0,(4,2)))
        for b in accessories:
            polygon(ax, b["polygon"], facecolor="#c3d9dd", edgecolor="#96aeb4", lw=.6, alpha=.45)
        for b in houses:
            for part in b.get("parts",[]):
                polygon(ax,part["polygon"],facecolor="#9ab8c1",edgecolor="#638a99",lw=.5)
            phased = b.get("phase_group") == "later"
            polygon(ax, b["polygon"], facecolor=LATER if phased else BLUE, edgecolor="#163e62", lw=.7)
            if ax is lower:
                continue
            p = np.array(b["polygon"])
            x, y = (p.min(axis=0) + p.max(axis=0)) / 2
            label = b["id"].replace("号楼","号").replace("#","号")
            upper.text(x,y-.75,f'{label} · {b["units_per_floor"]}户\n{height(b):g} m', ha="center", va="center",
                       fontsize=7.4, linespacing=1, color="white" if not phased else "#122f49")
            for _, group in groupby(facade_probes(b), key=lambda probe: probe[0]):
                pts = np.array([xy for _,_,xy,_ in group])
                upper.plot(*pts.T, color=RED, lw=.9, zorder=5)
                upper.scatter(*pts.T, c=RED, s=13, linewidths=.4, edgecolors="white", zorder=6)
    for d in model.get("dimensions",[]):
        if not d.get("show",True):
            continue
        a, b = np.array(d["a"]), np.array(d["b"])
        upper.annotate("",xy=a,xytext=b,arrowprops={"arrowstyle":"<->","lw":.78,"color":"#54707a",
                                                  "shrinkA":0,"shrinkB":0,"mutation_scale":7})
        mid=(a+b)/2
        hor=abs((a-b)[0])>abs((a-b)[1])
        upper.annotate(f'{d.get("relation","")}{d["value_m"]:g}',xy=mid,xytext=(0,4) if hor else (3,0),textcoords="offset points",
                       fontsize=7, color="#28444b", ha="center" if hor else "left", va="center",
                       bbox={"facecolor":"#fcfdfb","edgecolor":"none","pad":.5,"alpha":.9})
    lower.apply_aspect()
    label_centres = []
    # Nearby wings may be only a few metres apart. Position callouts in print
    # points so their separation survives zooming and a change of map extent.
    offsets = sorted(((x, y) for x in range(-32, 33, 8) for y in range(-32, 33, 8)),
                     key=lambda xy: (np.hypot(*xy), abs(xy[0])))
    box = lower.get_window_extent()
    lo = np.array([box.x0, box.y0]) * 72 / fig.dpi + 6
    hi = np.array([box.x1, box.y1]) * 72 / fig.dpi - 6
    for i,b in enumerate(external,1):
        polygon(lower,b["polygon"],facecolor=EXT,edgecolor="#697b82",lw=.8)
        p=np.array(b["polygon"])
        pos=np.array(b.get("plot_label_xy",p.mean(axis=0)))
        centre = lower.transData.transform(pos) * 72 / fig.dpi
        for offset in offsets:
            candidate = centre + offset
            if (np.all(lo <= candidate) and np.all(candidate <= hi)
                    and all(np.linalg.norm(candidate - other) >= 12 for other in label_centres)):
                break
        label_centres.append(centre + offset)
        leader = dict(arrowstyle="-", color="#485e67", lw=.6, shrinkA=0, shrinkB=0) if any(offset) else None
        lower.annotate(str(i), xy=pos, xytext=offset, textcoords="offset points",
                       ha="center", va="center", fontsize=7, zorder=5, arrowprops=leader,
                       bbox={"boxstyle":"circle,pad=.17","facecolor":"white","edgecolor":"#687a83","lw":.6})
    for road,x,y,rotation in (("墨江道",-6,54,90),("太湖路",231,54,90),
                                ("嘉江道",109,126,0),("沐江道",65,-12,0)):
        lower.text(x,y,road,rotation=rotation,ha="center",va="center",color="#526e78",fontsize=7.6)
    for b in bridges:
        line = np.array(b.get("centerline", b["polygon"]))
        tip = line[np.argmin(line[:,1])] + [0, 25]
        lower.annotate(b.get("chart_name", b["id"]), xy=tip, xytext=(17,0),
                       textcoords="offset points", va="center", color="#345771", fontsize=8,
                       arrowprops=dict(arrowstyle="-",color=BRIDGE,lw=.8))
    north(lower)
    official_overlay(src,model,pdf)
    checks.text(0,.98,"图上距离校准", fontsize=12, va="top", weight="bold")
    checks.text(0,.69,model["registration"]["chart_note"],fontsize=9.5,va="top",linespacing=1.65)
    fig.legend(handles=[Patch(color=BLUE,label="一期申报住宅"),Patch(color=LATER,label="后续示意住宅"),
                        Patch(color=EXT,label="周边建筑"), *([Patch(color=BRIDGE,label="立交桥面（估测）")] if bridges else []),
                        Patch(color="#c3d9dd",label="地块内配套"),
                        Line2D([],[],marker="o",lw=.9,color=RED,label="同户南窗测点")],
               loc="upper left", bbox_to_anchor=(.057,.55),ncol=6,fontsize=9,frameon=False,
               columnspacing=1.05,handlelength=1.3)
    key.text(0,1,"周边体量与模型总高",va="top",fontsize=13,weight="bold")
    count=len(external)
    columns = 2 if count > 17 else 1
    rows = (count+columns-1)//columns
    spacing=min(.062,.84/max(1,rows))
    for i,b in enumerate(external):
        column,row=divmod(i,rows)
        label=b.get("chart_name",b["id"])
        key.text(column/columns,.9-row*spacing,
                 f'{i+1:02}   {label}　{height(b):g} m', va="top", fontsize=7.6 if columns==2 else 9)
    key.text(0,.014,"周边高度为估计值，数据文件内列有来源及情景范围。",fontsize=8,color=MUTED)
    fig.text(.055,.027,"面积与距离以公示图标注为准。后续分期随报建可能调整；窗台点、屋面及外楼估高的口径见模型与报告。", fontsize=9,color=MUTED)
    save(fig,destination)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--pdf",type=Path,default=ROOT/"plan.pdf")
    parser.add_argument("--out",type=Path,default=CHARTS/"00_layout_verification.png")
    a=parser.parse_args()
    make(read_model(a.model),a.pdf,a.out)

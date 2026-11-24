"""Shared headless chart setup, including explicit CJK font selection."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "yili-matplotlib"))
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt

INK, MUTED = "#182f3a", "#425c67"


def bridge_height_note(model):
    """Describe the deck heights used in the rays, relative to the bridge's local grade."""
    bridge = next((b for b in model["buildings"] if b["id"] == "桥W01"), None)
    if not bridge:
        return ""
    decks = {bridge.get("name", "主桥"): bridge["roof_height"]}
    decks.update((part["name"], part["top_height"]) for part in bridge.get("parts", [])
                 if part.get("kind") == "deck" and part.get("name"))
    main = [height for name, height in decks.items() if "主桥" in name]
    labels = []
    if main:
        span = f"{min(main):g}–{max(main):g}" if min(main) != max(main) else f"{main[0]:g}"
        labels.append(f"主桥 {span} 米")
    if "东侧匝道" in decks:
        labels.append(f"东侧匝道 {decks['东侧匝道']:g} 米")
    qualifier = "估算" if bridge.get("estimated") else "模型"
    if not labels:
        return ""
    return f"郁江桥桥面离地高度（{qualifier}）　" + "，".join(labels)


def setup():
    options = [
        os.environ.get("SUNLIGHT_FONT"), "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc"
    ]
    for path in options:
        if path and Path(path).is_file():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        names = {f.name for f in font_manager.fontManager.ttflist}
        family = next((f for f in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC") if f in names), None)
        if not family:
            raise RuntimeError("请设置 SUNLIGHT_FONT 为支持中文的字体文件路径")
    mpl.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#adc0c4",
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
                         "axes.titlelocation": "left", "figure.facecolor": "white"})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(path)

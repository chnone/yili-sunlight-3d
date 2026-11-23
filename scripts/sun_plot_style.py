"""Shared headless chart setup, including explicit CJK font selection."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "yili-matplotlib"))
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager, pyplot as plt

INK, MUTED = "#182f3a", "#425c67"


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

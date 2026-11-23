"""Build the offline viewer and winter-solstice video from the current verified sunlight model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from build_sun_animation_data import build_scene
from sun_height_review import review_text
from sun_model import MODEL, OUTPUT, ROOT, read_model

TEMPLATE = ROOT / "scripts" / "sun_animation_template.html.in"
TOKEN = "__SUN_SCENE__"
VIEWER = "一里日照交互动画.html"
VIDEO = "一里冬至日照动画.mp4"
HEIGHT_AUDIT = "周边楼高复核.md"


def embed_scene(scene_json):
    template = TEMPLATE.read_text(encoding="utf-8")
    orbit_token = "__SUN_ORBIT_JS__"
    if template.count(TOKEN) != 1 or template.count(orbit_token) != 1:
        raise ValueError("动画模板缺少唯一的场景或三维渲染器插入位置")
    template = template.replace(orbit_token, TEMPLATE.with_name("sun_orbit.js").read_text(encoding="utf-8"))
    # A model label must not terminate a script tag, even for locally edited input.
    safe_json = (scene_json.replace("<", r"\u003c")
                 .replace("\u2028", r"\u2028")
                 .replace("\u2029", r"\u2029"))
    return template.replace(TOKEN, safe_json)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, default=MODEL)
    cli.add_argument("--details", type=Path, default=OUTPUT / "sunlight_details.csv")
    cli.add_argument("--year", type=int, help="应与逐点明细的计算年份一致")
    cli.add_argument("--output-dir", type=Path, default=OUTPUT)
    cli.add_argument("--html-only", action="store_true", help="只生成可离线播放的交互动画")
    args = cli.parse_args()
    model = read_model(args.model)
    year = args.year if args.year is not None else model["site"]["year"]
    scene = build_scene(model, year, args.details)
    payload = json.dumps(scene, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir, prefix=".sun-animation-") as staging:
        staging = Path(staging)
        html = staging / VIEWER
        html.write_text(embed_scene(payload), encoding="utf-8")
        audit = staging / HEIGHT_AUDIT
        audit.write_text(review_text(model, scene), encoding="utf-8")
        if not args.html_only:
            data = staging / "scene.json"
            data.write_text(payload, encoding="utf-8")
            movie = staging / VIDEO
            subprocess.run([
                sys.executable, "-B", str(ROOT / "scripts" / "render_sun_animation.py"),
                "--data", str(data), "--output", str(movie),
            ], check=True)
            os.replace(movie, args.output_dir / VIDEO)
            print(args.output_dir / VIDEO)
        os.replace(html, args.output_dir / VIEWER)
        os.replace(audit, args.output_dir / HEIGHT_AUDIT)
        print(args.output_dir / VIEWER)


if __name__ == "__main__":
    main()

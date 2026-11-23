# 天津河西 · 一里地块三维日照模拟

[下载离线交互动画](https://github.com/chnone/yili-sunlight-3d/raw/refs/heads/main/output/一里日照交互动画.html) 后用浏览器打开，即可拖动三维视角，按楼栋、楼层、单元和户位查看南窗的直射时段。默认显示地块。点「西侧立交桥」可以把郁江桥和住宅放在同一画面中。

模型包含 13 栋住宅、368 个模拟户位、736 个窗台测点、44 个周边建筑体量，以及郁江桥解放南路临近地块的一段。桥面和桥墩分别参与遮挡计算。主桥桥面离地暂取 9 米，匝道暂取 7 米，墩位和桥高均为估算。住宅分户及后续分期也包含建模假设。[建模说明](data/建模说明.md) · [周边高度复核](output/周边楼高复核.md)

## 日照结果

交互动画从日出播到日落，窗台见光与遮挡按 5 分钟区间中点追踪。全天住户排名另按 1 分钟采样，取每户各南窗测点的平均直射时长。排除 12、13 号楼整栋后共 296 户参评，两栋楼仍参与遮挡。点“高亮第一名”可在三维图中查看当日所有并列第一户。

数据表分别统计大寒和春分北京时间 08:00–16:00、冬至 09:00–15:00 的直射。大寒另列当地真太阳时 08:00–16:00 的结果，2026 年 1 月 20 日在模型坐标下约对应北京时间 08:22–16:22。住宅详情可比较同一户较弱测点在两种时间带及周边楼高估计范围内的结果。大寒 2 小时用于模型筛查，正式逐户认定需实际房间、窗位、遮挡高度和报审资料。

- [逐窗台明细](output/sunlight_details.csv) · [逐户汇总](output/sunlight_households.csv) · [选房报告](output/采光选房排雷报告.md)
- [楼位校验图](charts/00_layout_verification.png) · [大寒热力图](charts/01_dahan_heatmap.png) · [冬至热力图](charts/02_dongzhi_heatmap.png) · [楼栋对比](charts/03_sunlight_summary_ranking.png)
- [冬至回放视频](output/一里冬至日照动画.mp4) · [10、11 号楼每层六户对照](output/10-11号楼_六户对照/)

## 重算

在仓库根目录运行以下命令。本版在 Python 3.14 上验证。修改 [buildings.json](data/buildings.json) 可调整楼体、桥面、桥墩和测点。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest discover -s scripts -p 'test_sun*.py'
python -B scripts/simulate_sunlight.py
python -B scripts/sun_layout_variant.py
python -B scripts/plot_heatmaps.py
python -B scripts/build_sun_animation.py --html-only
```

生成冬至视频需安装含 libx264 的 FFmpeg，再运行 `python -B scripts/build_sun_animation.py`。绘图及视频需要中文字体，可通过 `SUNLIGHT_FONT` 指定字体文件。重新制作楼位校验图时，将下列官方 PDF 下载到根目录并命名为 `plan.pdf`，安装 Poppler 后运行 `python -B scripts/plot_layout_check.py`。

## 图纸与地图来源

楼位依据天津市规划和自然资源局 2026 年 11 月 2 日公示总平面图，校验图右上角将模型轮廓与该图局部叠合。[公示原文](https://ghhzrzy.tj.gov.cn/ywpd/cxgh_43015/ghgs/202611/t20261102_7648329.html) · [原始 PDF](https://ghhzrzy.tj.gov.cn/ywpd/cxgh_43015/ghgs/202611/W020261102559803301059.pdf)

郁江桥桥段的原始中心线来自 © OpenStreetMap contributors，适用 Open Database License（ODbL）。配准控制点、地图要素编号、影像依据和桥高假设记录在周边几何底稿中。[OpenStreetMap 版权与许可](https://www.openstreetmap.org/copyright) · [周边几何底稿](data/surrounding_geometry.json)

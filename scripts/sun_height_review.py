"""Generate the height audit for the same geometry, dates and rays as the offline viewer."""
from sun_model import height


def review_text(model, scene):
    external = [b for b in scene["buildings"] if b["group"] == "external"]
    buildings = [b for b in external if b.get("kind") != "bridge"]
    bridges = [b for b in external if b.get("kind") == "bridge"]
    original = {b["id"]: b for b in model["buildings"]}
    reviews = {bid: b.get("height_review", {}) for bid, b in original.items()}
    matches = [b for b in external if sum(b["shadow_impact"].values())]
    standard_matches = [b for b in external if sum(b["standard_shadow_impact"].values())]
    days = ["大寒", "冬至", "春分"]
    reviewed = model.get("surrounding_sources", {}).get("height_audit", {}).get("date", "待填写")
    lines = [
        "# 一里南侧地块周边遮挡体高度复核",
        "",
        f"周边楼高资料核对日 {reviewed}。场景含 {len(buildings)} 个周边建筑体量"
        + (f"和 {len(bridges)} 座桥" if bridges else "") + f"。按 {scene['year']} 年三个指定日期和当前基准高度，"
        f"从日出到日落，有 {len(matches)} 个体块曾挡在至少一处住宅窗台与太阳之间，且为该方向上最靠近窗台的遮挡。"
        f"只看原统计时段时，这个数量为 {len(standard_matches)} 个。"
        "可在动画中点选这些体块查看依据，也可点击窗台卡片上的遮挡物。",
        "",
        "## 逐栋高度",
        "",
        "高度单位为米，基准为模型屋顶相对本地地面的高度。层数单列，情景范围用于测试遮挡对高度变化的敏感性。"
        "相邻项目名称、楼号和行政文件各有覆盖范围，表中说明了可对应到什么程度。",
        "",
        "下表中的益发里是本盘南面、隔着沐江道的小区。表中的“前次 → 本次”记录日照模型两次取值。"
        "例如用 10 层、每层 3.1 米、屋面再加 1 米来估算，算阴影时就把屋顶放在离地 32 米处。"
        "实楼的竣工高度仍需实测或查竣工图。",
        "",
        "益发里北排的位置以沐江道、太湖路和中央幼儿园为控制，南排以复兴河为控制。"
        "东侧双迎里用太湖路与嘉江道交叉口以及南北两排住宅对应。"
        "[天地图公众地图服务](https://map.tianditu.gov.cn/)",
        "",
        "|体块|楼名或位置|有资料的层数|前次 → 本次基准（米）|情景范围（米）|核对依据|",
        "|---|---|---:|---:|---:|---|",
    ]
    for b in buildings:
        info = reviews[b["id"]]
        top = height(original[b["id"]])
        before = info.get("previous_roof_m")
        previous = f"{before:g} → " if before is not None else "新增 → "
        limits = b["height_review"]["range"]
        interval = f"{limits[0]:g}–{limits[1]:g}" if limits else "未设定"
        links = [f"[{s['title']}]({s['url']})" for s in info.get("references", [])]
        floors = info.get("floor_evidence_text") or original[b["id"]].get("floors") or "待核"
        lines.append(f"|{b['id']}|{b['name']}|{floors}|{previous}**{top:g}**|{interval}|"
                     f"{info.get('summary', '待核对。')} {' · '.join(links)}|")
    for bridge in bridges:
        info = reviews[bridge["id"]]
        ground = original[bridge["id"]]["ground_elevation"]
        decks = [bridge, *(p for p in bridge["parts"] if p.get("kind") == "deck")]
        piers = sum(p.get("kind") == "pier" for p in bridge["parts"])
        def heights(key):
            values = [p[key] - ground for p in decks]
            return f"{min(values):g}–{max(values):g}" if max(values) != min(values) else f"{min(values):g}"
        refs = " · ".join(f"[{s['title']}]({s['url']})" for s in info.get("references", []))
        hits = "、".join(f"{day} {bridge['shadow_impact'][day]:g}" for day in days)
        lines += ["", f"## {bridge['name']}", "",
                  f"模型含 {len(decks)} 个桥面段和 {piers} 个桥墩。相对模型地面的桥板底高为 {heights('base')} 米，"
                  f"板顶高为 {heights('top')} 米。射线可从桥板与桥墩之间的净空穿过。",
                  f"从日出到日落，该桥作为窗台射线首个遮挡物的累计点·分钟为{hits}。",
                  "", f"{info.get('reviewed_on', '')}　{info.get('summary', '')} {refs}".strip()]
    lines += ["", "## 遮挡过住宅窗台的外部体块", "",
              "一对窗台测点分别计数。一处窗台被同一体块挡住 5 分钟计 5 点·分钟。"
              "表内把各楼层、各户的测点求和，同一条射线只归给遇到的第一个遮挡物。"
              "未列体块仍在模型中参与所有求交。",
              ""]
    for title, obstacles, field in (
            ("日出至日落（北京时间）", matches, "shadow_impact"),
            ("原统计时段（大寒、春分 08:00–16:00，冬至 09:00–15:00）", standard_matches, "standard_shadow_impact")):
        lines += [f"### {title}", "",
                  "|体块|名称|大寒（点·分钟）|冬至（点·分钟）|春分（点·分钟）|", "|---|---|---:|---:|---:|"]
        for b in sorted(obstacles, key=lambda x: -sum(x[field].values())):
            values = [f"{b[field][d]:.3f}".rstrip("0").rstrip(".") for d in days]
            lines.append("|" + "|".join([b["id"], b["name"], *values]) + "|")
        lines.append("")
    lines += ["", "## 地图高度与遥感交叉检查", "",
              "天地图高清底图的三维楼体使用楼层属性 floor 和挤出高度 h。当前相邻楼体的 h 常为层数乘以 3 米，"
              "这项数值用于核对楼体量级。遇到工程层数记录、实际用途或多种量测与地图不一致时，逐栋说明里给出取值依据。",
              "",
              "将模型外轮廓按公示图与 2025-05-24 影像的同一组控制点配准到 WGS84。"
              "交并比 IoU 为轮廓交集面积除以并集面积，最高为 1。GBA 和 3D-GloBFP 列给出各自交并比最大的候选面。"
              "部分 GBA 轮廓取自 3D-GloBFP，两列高度产品的差异不能解释为独立实测误差。"
              "CNBH 列取轮廓内 10 米像元中心的有效正值中位数，括号内为有效像元数 / 落入像元数。"
              "狭翼、裙楼及高塔须特别注意分辨率、配准偏差和屋顶分割影响。",
              "", "|体块|天地图楼体（层 / 米）|GBA 候选高度 / IoU|3D-GloBFP 候选高度 / IoU|CNBH 中位数（像元）|",
              "|---|---|---:|---:|---:|"]
    for b in buildings:
        cross = reviews[b["id"]].get("remote_check", {})
        def candidate(key):
            item = cross.get(key)
            if not item or item["height_m"] <= 0:
                return "无有效配对"
            return f"{item['height_m']:.1f} m / {item['iou']:.2f}"
        raster = cross.get("cnbh", {"median_m": None, "valid": 0, "total": 0})
        median = f"{raster['median_m']:.1f} m" if raster["median_m"] is not None else "无有效值"
        map_note = reviews[b["id"]].get("map_values", "未取得对应值")
        lines.append(f"|{b['id']}|{map_note}|{candidate('gba')}|{candidate('globfp')}|{median}（{raster['valid']}/{raster['total']}）|")
    lines += ["", "研究数据与方法见 [GlobalBuildingAtlas 原始研究](https://essd.copernicus.org/articles/17/6647/2025/)、"
              "[Taylor Geospatial 的 LoD1 分发数据](https://source.coop/tge-labs/globalbuildingatlas-lod1)、"
              "[3D-GloBFP 原始研究](https://essd.copernicus.org/articles/16/5357/2024/)及"
              "[2020 年 CNBH-10m 原始数据](https://zenodo.org/records/7923866)。"
              "GBA 使用 e115_n40_e120_n35 分片，CNBH 使用 X117Y39（WGS 84 / UTM 50N）。",
              "", f"模型校验码 `{scene['model_digest']}`。屋顶机房、女儿墙和局部高低差以当前统一模型为准。", ""]
    return "\n".join(lines)

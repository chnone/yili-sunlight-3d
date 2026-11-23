"""Plot per-household direct sun and report the weaker sampled position in each home."""
import argparse
import csv
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
from sun_model import CHARTS, DAYS, MODEL, OUTPUT, household_addresses, model_digest, read_model, receivers, targets
from sun_plot_style import MUTED, plt, save, setup
from sun_households import assign_addresses, summarize_households
from sun_layout_variant import select_layout
from matplotlib import colors, patches
from matplotlib.lines import Line2D


def load_table(model, path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    if not records or "测点" not in records[0]:
        raise ValueError("明细缺少户内测点，请先运行 simulate_sunlight.py")
    years = {int(r["计算年份"]) for r in records}
    if len(years) != 1 or {r["模型校验码"] for r in records} != {model_digest(model)}:
        raise ValueError("明细与楼栋参数或计算年份不一致，请重新运行仿真")
    expected, _, _ = receivers(model)
    indexed = {(r["楼号"], int(r["住宅楼层"]), r["户位"], r["测点"]): r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(expected):
        raise ValueError("实算测点与模型中的楼栋、楼层、户位、测点标识存在缺失或重复")
    assign_addresses(model, records)
    by_unit = defaultdict(list)
    for bid, floor, unit, probe in expected:
        by_unit[bid, floor, unit].append(indexed[bid, floor, unit, probe])
    by_building = defaultdict(list)
    summaries = summarize_households(indexed[key] for key in expected)
    for summary in summaries:
        bid, floor, unit = summary["楼号"], summary["住宅楼层"], summary["户位"]
        points = by_unit[bid,floor,unit]
        home = {"住宅楼层": floor, "户位": unit, "测点明细": points,
                "单元": summary["单元"], "单元内户位": summary["单元内户位"]}
        for day in DAYS:
            col = day + "分钟"
            dark = min(points, key=lambda r: int(r[col]))
            home[col] = summary[day+"较弱点分钟"]
            home[day+"最长分钟"] = summary[day+"较好点分钟"]
            home[day+"最弱点"] = dark
            home[day+"1分钟复核"] = min(int(p[day+"1分钟复核"]) for p in points)
            if day != "春分":
                home[day+"选层筛查分钟"] = summary[day+"选层筛查分钟"]
                for scenario in ("保守", "乐观", "偏西", "偏东", "层高对照"):
                    field = day + scenario + "分钟"
                    home[field] = min(int(p[field]) for p in points) if all(p[field] != "" for p in points) else None
        by_building[bid].append(home)
    return by_building, years.pop()


def heatmap(model, groups, day, file, year):
    buildings = sorted(targets(model), key=lambda b: b.get("phase_group") == "later")
    parts = []
    columns = []
    for b in buildings:
        labels = list(dict.fromkeys(r["户位"] for r in groups[b["id"]]))
        parts.append((b, len(columns), labels))
        columns.extend(labels)

    nfloor = max(b["floors"] for b in buildings)
    data = np.full((nfloor, len(columns)), np.nan)
    cell_labels = {}
    max_points = 0
    for b, start_col, labels in parts:
        for row in groups[b["id"]]:
            col = start_col + labels.index(row["户位"])
            grid_cell = row["住宅楼层"] - 1, col
            data[grid_cell] = row[day+"分钟"] / 60
            points = row["测点明细"]
            max_points = max(max_points,len(points))
            cell_labels[grid_cell] = "\n".join(f'{p["测点"]} {int(p[day+"分钟"])/60:.2f}' for p in points)

    month, dom, begin, end = DAYS[day]
    palette = colors.LinearSegmentedColormap.from_list(
        "hours", ["#b5303b","#ec8259","#ffcf86","#f4e9a5","#9bd4ab","#399971","#106952"])
    norm = colors.Normalize(0, end-begin)
    # Reserve a separate text line for each point inside every floor/household cell.
    row_height_pt = 5 + 9 * max_points
    fig = plt.figure(figsize=(max(14, len(columns)*.48+2), max(7, nfloor*row_height_pt/(.60*72))))
    fig.suptitle(f"{day}  |  各层各户南向日照", x=.06,y=.968,ha="left",fontsize=23)
    fig.text(.06,.902,f"{year} 年 {month} 月 {dom} 日　·　北京时间 {begin:02}:00–{end:02}:00"
             "　·　每格一户，A / B 等逐行列出各点小时数，底色按本户较弱点",fontsize=12,color=MUTED)
    ax = fig.add_axes([.058,.18,.925,.60])
    ax.set_facecolor("#f0f3f2")
    im = ax.imshow(data,origin="lower",aspect="auto",interpolation="none",cmap=palette,norm=norm)
    for floor, col in zip(*np.nonzero(np.isfinite(data))):
        val = data[floor,col]
        if val < 2:
            ax.add_patch(patches.Rectangle((col-.47,floor-.47),.94,.94,fill=False,lw=1.65,ec="#ad1527",zorder=2))
        brightness = np.dot(palette(norm(val))[:3], [.2126,.7152,.0722])
        ax.text(col,floor,cell_labels[floor,col],ha="center",va="center",fontsize=7.2,linespacing=1.04,
                color="white" if brightness < .50 else "#172b31")
    tick_labels = []
    for b, col, labels in parts:
        ax.axvline(col-.5,color="white",lw=2.0)
        ax.text(col+(len(labels)-1)/2,1.012,b["id"],transform=ax.get_xaxis_transform(),
                ha="center",va="bottom",fontsize=10)
        addresses = household_addresses(b, model.get("selected_unit_layout"))
        tick_labels.extend(addresses[label][1] for label in labels)
        start = col
        for entrance, group in groupby(labels, key=lambda label: addresses[label][0]):
            stop = start + len(list(group))
            if entrance:
                transform = ax.get_xaxis_transform()
                ax.plot([start-.36, stop-.64], [-.031, -.031], transform=transform,
                        color=MUTED, lw=.55, clip_on=False)
                ax.text((start+stop-1)/2, -.039, entrance, transform=transform,
                        ha="center", va="top", fontsize=8, color=MUTED)
            start = stop
    for group, title, color in ((False,"一期申报","#205978"),(True,"后续分期示意","#5a727c")):
        spans = [(col, col+len(names)) for b,col,names in parts if (b.get("phase_group")=="later") == group]
        if spans:
            first, last = spans[0][0], spans[-1][1]
            ax.plot([first-.4,last-.6],[1.07]*2,transform=ax.get_xaxis_transform(),clip_on=False,color=color,lw=1)
            ax.text((first+last-1)/2,1.08,title,transform=ax.get_xaxis_transform(),ha="center",fontsize=10,color=color)
    ax.set_xticks(range(len(columns)), tick_labels, fontsize=8)
    ax.set_yticks(range(nfloor),range(1,nfloor+1))
    ax.set_ylabel("住宅楼层（自首个住宅层起）")
    ax.tick_params(length=0)
    cax = fig.add_axes([.062,.082,.29,.02])
    fig.colorbar(im,cax=cax,orientation="horizontal",ticks=range(end-begin+1))
    fig.text(.38,.086,"红框  本户较弱点 < 2.00 小时",color="#a82937",fontsize=11)
    fig.text(.60,.086,"灰底为空层　·　户位从东向西排列",color=MUTED,fontsize=10)
    fig.text(.062,.023,"横轴按楼栋、单元及单元内户位读数。各点分别累计，逐点日照时段见明细表。",color=MUTED,fontsize=9)
    save(fig,file)


def cutoff(homes, dahan_col=None, winter_col="冬至分钟"):
    """Lowest floor whose homes and all higher homes satisfy both chosen cutoffs."""
    floors = {r["住宅楼层"] for r in homes}
    failed = [r["住宅楼层"] for r in homes if r[winter_col] < 180 or (dahan_col and r[dahan_col] < 120)]
    return min((k for k in floors if k > max(failed,default=0)),default=None)


def avg(homes, day):
    return sum(r[day+"分钟"] for r in homes)/len(homes)/60


def ranked_ids(groups):
    return sorted(groups,key=lambda bid:(-avg(groups[bid],"冬至"),-avg(groups[bid],"大寒"),bid))


def ranking(groups, file):
    ranked = ranked_ids(groups)
    fig,(left,right)=plt.subplots(1,2,figsize=(13.4,8),gridspec_kw={"width_ratios":[1.37,1]})
    fig.subplots_adjust(left=.11,right=.965,top=.79,bottom=.17,wspace=.17)
    fig.suptitle("楼栋日照对比  |  各户较弱点与选层",x=.075,y=.961,ha="left",fontsize=23)
    fig.text(.075,.897,"每户先取各点中较短的时长，全楼各户等权。选房筛查要求大寒 ≥2 小时、冬至 ≥3 小时。",
             fontsize=11,color=MUTED)
    y = np.arange(len(ranked))
    for ax in (left,right):
        ax.set_ylim(len(y)-.35,-.65)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color="#dce5e7",lw=.8)
        for j in y[::2]:
            ax.axhspan(j-.48,j+.48,color="#f1f5f4",zorder=-2)
        for side in ("right","top","left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    for d,delta,color in (("冬至",-.18,"#245887"),("大寒",0,"#c57232"),("春分",.18,"#318163")):
        left.scatter([avg(groups[bid],d) for bid in ranked],y+delta,s=23,color=color,label=d)
    left.set_xlim(0,8.2)
    left.set_xticks(range(9))
    left.set_yticks(y,ranked)
    left.set_xlabel("各户较弱点的全楼平均时长（h）")
    left.set_title("三个日期的户均值",fontsize=14,pad=10)
    left.legend(loc="lower left",bbox_to_anchor=(0,-.19),frameon=False,ncol=3)
    for j,bid in enumerate(ranked):
        homes = groups[bid]
        baseline = cutoff(homes)
        screened = cutoff(homes,"大寒选层筛查分钟","冬至选层筛查分钟")
        for level,dy,color,mark in ((baseline,-.13,"#2f729e","o"),(screened,.13,"#a46236","D")):
            if level:
                right.scatter(level,j+dy,marker=mark,color=color,s=31)
            else:
                right.text(.4,j+dy,"—",ha="center",va="center",color=color,fontsize=8)
    high=max(r["住宅楼层"] for homes in groups.values() for r in homes)
    right.set_xlim(0,high+1)
    right.set_xticks([1,*range(3,high+1,3)])
    right.set_yticks([])
    right.set_xlabel("住宅起始层（— 表示没有满足条件的整层）")
    right.set_title("该层及以上所有户均达到条件",fontsize=14,pad=10)
    right.legend(handles=[Line2D([],[],color="#2f729e",marker="o",ls="",label="冬至基准 ≥3 h"),
                          Line2D([],[],color="#a46236",marker="D",ls="",label="大寒与冬至综合筛查")],
                 loc="lower left",bbox_to_anchor=(-.02,-.19),frameon=False,ncol=2,fontsize=9)
    save(fig,file)


def floor_ranges(values):
    runs=[]
    for k in sorted(set(values)):
        if runs and k==runs[-1][-1]+1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return "、".join(str(g[0]) if len(g)==1 else f"{g[0]}–{g[-1]}" for g in runs) or "无"


def floor_text(k):
    return f"{k} 层" if k else "无"

def home_address(row):
    return row["单元"] + row["单元内户位"]


def unit_start_floors(homes):
    by_unit = defaultdict(list)
    for r in homes:
        by_unit[home_address(r)].append(r)
    return [(unit, cutoff(rows,"大寒选层筛查分钟","冬至选层筛查分钟")) for unit,rows in by_unit.items()]


def verified_north_comparison(model, out_dir, year):
    """Only cite a completed paired run for this exact model, year and floor layout."""
    folder = out_dir / "10-11号楼_六户对照"
    detail_path, unit_path = folder/"sunlight_details.csv", folder/"sunlight_households.csv"
    if not all(path.is_file() for path in (detail_path,unit_path)):
        return {}
    alternate = select_layout(model,"north_six")
    try:
        computed, actual_year = load_table(alternate,detail_path)
        if actual_year != year:
            return {}
        with unit_path.open(encoding="utf-8-sig", newline="") as stream:
            published = list(csv.DictReader(stream))
        keyed = {(bid,r["住宅楼层"],r["户位"]):r for bid,homes in computed.items() for r in homes}
        groups = defaultdict(list)
        seen = set()
        digest = model_digest(alternate)
        for row in published:
            key = (row["楼号"],int(row["住宅楼层"]),row["户位"])
            if key not in keyed or key in seen or int(row["计算年份"]) != year or row["模型校验码"] != digest:
                return {}
            home = {"住宅楼层":key[1], "冬至分钟":int(row["冬至较弱点分钟"]),
                    "大寒选层筛查分钟":int(row["大寒选层筛查分钟"]), "冬至选层筛查分钟":int(row["冬至选层筛查分钟"])}
            if any(keyed[key][field] != value for field,value in home.items()):
                return {}
            groups[key[0]].append(home)
            seen.add(key)
        return groups if seen == set(keyed) else {}
    except (KeyError, ValueError):
        return {}


def report(model, groups, out, year):
    ranked = ranked_ids(groups)
    by_id = {b["id"]: b for b in targets(model)}
    named = {b["id"]:b.get("chart_name",b["id"]) for b in model["buildings"]}
    first = [b for b in ranked if by_id[b].get("phase_group")!="later"]
    all_homes = [home for homes in groups.values() for home in homes]
    samples = sum(len(r["测点明细"]) for r in all_homes)
    short = sum(r["大寒分钟"]<120 for r in all_homes)
    zero_weak = sum(r["冬至分钟"]==0 for r in all_homes)
    zero_all = sum(r["冬至最长分钟"]==0 for r in all_homes)
    counts = sorted({len(r["测点明细"]) for r in all_homes})
    shift = float(all_homes[0]["测点明细"][0]["窗位对照偏移_m"])
    alt = float(all_homes[0]["测点明细"][0]["对照层高_m"])
    per_unit = "、".join(map(str,counts))
    phase = lambda bid: "后续示意" if by_id[bid].get("phase_group")=="later" else "一期申报"
    first_count = sum(len(groups[bid]) for bid in first)
    winner = ranked[0]
    body = ["# 天津河西一里南侧地块采光选房排雷报告","",
            f"{year} 年固定日期晴空模拟按标准层户位布点。{len(ranked)} 栋住宅共 **{len(all_homes)} 个模型户位、{samples} 个测点**，每户 {per_unit} 点。每户各测点分别累计直射时间，按当日最短值筛查。"
            f"基准情景中 **{short} 个户位**有测点大寒不足 2 小时。冬至 09:00–15:00 有 **{zero_weak} 个户位**至少一处为零，其中 **{zero_all} 个户位**全部测点为零。",
            "", f"**优先看房**　冬至户均值最高的是{winner}（{phase(winner)}），全楼各户较弱点平均 **{avg(groups[winner],'冬至'):.2f} 小时**。"]
    if first:
        best = first[0]
        start = cutoff(groups[best],"大寒选层筛查分钟","冬至选层筛查分钟")
        entry = f"一期冬至户均值最高的是{best}，为 {avg(groups[best],'冬至'):.2f} 小时。"
        if start:
            body.append(entry + f"该栋全层筛查从 **{start} 层**起通过。")
        else:
            per_home = "，".join(f"{unit}从 {k} 层起通过" if k else f"{unit}未形成连续到顶层的通过区间"
                                for unit,k in unit_start_floors(groups[best]))
            body.append(entry + per_home+"。")
            top = max(r["住宅楼层"] for r in groups[best])
            swings = [(int(p["冬至分钟"])-int(p["冬至"+side+"分钟"]),side,p)
                      for r in groups[best] if r["住宅楼层"]==top for p in r["测点明细"] for side in ("偏东","偏西")
                      if int(p["冬至分钟"])>=180>int(p["冬至"+side+"分钟"])]
            if swings:
                _,side,probe = max(swings,key=lambda swing:swing[0])
                body.append(f'该栋 {top} 层{home_address(probe)} {probe["测点"]} 点冬至基准为 {probe["冬至分钟"]} 分钟。'
                            f'同一外楼高度下，窗位{side} {shift:g} m 后为 {probe["冬至"+side+"分钟"]} 分钟，具体房源须核对该窗与墙垛的位置。')
    body+=["", "## 户型与测点","",
           f"主模型一期按标准层逐层外推共 {first_count} 个户位。窄楼每层由唯一单元的东户、西户组成。宽楼从东往西依次是东单元东户、东单元西户、西单元东户、西单元西户。公示表列一期住宅户数上限为 266 户，实际分户与销售房号待核实。A、B 分别为同一户内偏东、偏西的窗台测点。",
           "", "按楼号、住宅楼层、单元、单元内户位查[逐测点明细](sunlight_details.csv)或[逐户汇总](sunlight_households.csv)。旧表中的“东中户”对应东单元西户，“西中户”对应西单元东户。"]
    if ("north_six" in model.get("site",{}).get("unit_layout_scenarios",{})
            and all(by_id.get(b,{}).get("units_per_floor")==4 for b in ("10号楼","11号楼"))):
        north_ids = ("10号楼","11号楼")
        alt_count = first_count + sum((6-by_id[b]["units_per_floor"])*by_id[b]["floors"] for b in north_ids)
        body+=["",f"10、11 号楼各可见三组入口。主模型按每层四户。六户对照沿用东、中、西三单元各两户的假设，单列于下表。若其余楼栋采用主模型，一期外推为 {alt_count} 个户位。"]
        alternatives = verified_north_comparison(model,out.parent,year)
        if alternatives:
            body+=["","| 楼栋 | 四户冬至户均 h | 六户冬至户均 h | 四户建议起始层 | 六户建议起始层 |",
                   "|---|---:|---:|---:|---:|"]
            for bid in north_ids:
                main_rows,alt_rows=groups[bid],alternatives[bid]
                four=cutoff(main_rows,"大寒选层筛查分钟","冬至选层筛查分钟")
                six=cutoff(alt_rows,"大寒选层筛查分钟","冬至选层筛查分钟")
                body.append(f"| {bid} | {avg(main_rows,'冬至'):.2f} | {avg(alt_rows,'冬至'):.2f} | {floor_text(four)} | {floor_text(six)} |")
            body+=["","[两栋六户情景的逐户结果](10-11号楼_六户对照/sunlight_households.csv) · [逐点结果](10-11号楼_六户对照/sunlight_details.csv)"]
        else:
            body+=["","运行 `scripts/sun_layout_variant.py` 可生成同年六户对照。"]
    body+=["","## 各栋最低建议楼层","",
           f"基准冬至起始层要求本层及以上每个户位的所有测点均有 3 小时直射。建议起始层同时要求大寒 2 小时与冬至 3 小时，采用高外楼及窗位左右移动 {shift:g} m 的筛查情景，并计入基准几何 1 分钟复算。层号从首个住宅层算起。“无”表示没有某层及以上连续至顶层均通过的区间，个别楼层仍可逐户查询。",
           "", "| 楼栋 | 分期 | 标准层户数 | 大寒户均 h | 冬至户均 h | 春分户均 h | 基准冬至起始层 | 整层建议起始层 | 分户建议起始层（东→西） |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for bid in ranked:
        homes=groups[bid]
        levels = "、".join(f"{unit} {floor_text(k)}" for unit,k in unit_start_floors(homes))
        body.append(f"| {bid} | {phase(bid)} | {by_id[bid]['units_per_floor']} | {avg(homes,'大寒'):.2f} | "
                    f"{avg(homes,'冬至'):.2f} | {avg(homes,'春分'):.2f} | {floor_text(cutoff(homes))} | "
                    f"{floor_text(cutoff(homes,'大寒选层筛查分钟','冬至选层筛查分钟'))} | {levels} |")
    body+=["","## 低楼层与同户两处采光差异","",
           "有任一户触发即列出楼层。每栋展示一个冬至最弱户位。时长范围来自该户全部测点，最后一列为最弱点在冬至各时段首先命中的主要遮挡体。",
           "", "| 楼栋 | 大寒弱点不足 2 h 楼层 | 冬至弱点为零楼层 | 冬至最弱户位 | 本户最短–最长 h | 最弱点 | 主要遮挡 |",
           "|---|---|---|---|---:|---|---|"]
    for bid in ranked:
        homes=groups[bid]
        worst=min(homes,key=lambda r:(r["冬至分钟"],r["住宅楼层"]))
        weak=worst["冬至最弱点"]
        labels=[]
        for hit in weak["冬至主要遮挡"].split(" / "):
            label, delim, remaining=hit.partition("（")
            labels.append(named.get(label,label)+delim+remaining)
        body.append(f'| {bid} | {floor_ranges(r["住宅楼层"] for r in homes if r["大寒分钟"]<120)} | '
                    f'{floor_ranges(r["住宅楼层"] for r in homes if r["冬至分钟"]==0)} | '
                    f'{worst["住宅楼层"]} 层{home_address(worst)} | {worst["冬至分钟"]/60:.2f}–{worst["冬至最长分钟"]/60:.2f} | '
                    f'{weak["测点"]} | {" / ".join(labels)} |')
    largest=max(all_homes,key=lambda r:r["冬至最长分钟"]-r["冬至分钟"])
    obs=largest["冬至最弱点"]
    gap=largest["冬至最长分钟"]-largest["冬至分钟"]
    body+=["", f'冬至同户点间差最大为 **{gap} 分钟**。其中一处为{obs["楼号"]} {obs["住宅楼层"]} 层{home_address(obs)}，'
           f'较弱点累计 {largest["冬至分钟"]} 分钟，较亮点累计 {largest["冬至最长分钟"]} 分钟。']
    for tall in (bid for bid in ranked if by_id[bid]["floors"]==18):
        affected=[]
        for bid in ranked:
            if bid==tall:
                continue
            hits = []
            for home in groups[bid]:
                for p in home["测点明细"]:
                    names = {hit.partition("（")[0] for day in ("大寒","冬至")
                             for hit in p[day+"主要遮挡"].split(" / ")}
                    if tall in names:
                        hits.append(home["住宅楼层"])
                        break
            if hits:
                affected.append(f"{bid} {floor_ranges(hits)} 层")
        if affected:
            body+=["",f"{tall}（18 层）进入以下住宅大寒或冬至测点的前两位遮挡体，涉及"+"、".join(affected)+"。"]
    body+=["","## 窗位、层高与采样复核","",
           f"外楼取基准高度，将每户测点沿立面向西或向东移动 {shift:g} m 后，以下楼栋的大寒、冬至双条件起始层发生变化。"]
    changes = []
    for bid in ranked:
        lvls = [cutoff(groups[bid],"大寒"+s+"分钟","冬至"+s+"分钟") for s in ("","偏西","偏东")]
        if len(set(lvls))>1:
            changes.append("| "+bid+" | "+" | ".join(floor_text(k) for k in lvls)+" |")
    if changes:
        body+=["",f"| 楼栋 | 基准窗位 | 偏西 {shift:g} m | 偏东 {shift:g} m |",
               "|---|---:|---:|---:|",*changes]
    else:
        body[-1]=f"外楼取基准高度，将每户测点向西或向东移动 {shift:g} m 后，各栋双条件起始层均相同。"
    for day,limit in (("大寒",120),("冬至",180)):
        changed=[r for r in all_homes if (r[day+"分钟"]>=limit)!=(r[day+"1分钟复核"]>=limit)]
        near=", ".join(f'{r[day+"最弱点"]["楼号"]} {r["住宅楼层"]} 层{home_address(r)}（{r[day+"分钟"]} → {r[day+"1分钟复核"]} 分钟）'
                       for r in changed[:3])
        alt_valid=[r for r in all_homes if r[day+"层高对照分钟"] is not None]
        height_changed=sum((r[day+"分钟"]>=limit)!=(r[day+"层高对照分钟"]>=limit) for r in alt_valid)
        ex_flips=sum(r[day+"保守分钟"]<limit<=r[day+"乐观分钟"] for r in all_homes)
        body+=["", f"- **{day} {limit//60} 小时筛查**　基准几何细化至 1 分钟采样后，有 {len(changed)} 个户位跨过阈值。"
               + ("例如"+near+"。" if near else "")
               + f"外楼从低值调至高值时有 {ex_flips} 个户位跨过阈值。改按 {alt:g} m 层高布置窗台，在 {len(alt_valid)} 个可比户位中有 {height_changed} 个跨过阈值。"]
    body+=["", "## 数据口径","",
           "- 户内较短值取自各测点独立累计的分钟数。逐户汇总另列「全部点同时直射」及「至少一点直射」的分钟数与时段，分别取各点见光区间的交集、并集。两个代表点在分户图核实前仅用于选房筛查，逐户日照合规还需核对实际窗位、居住空间和审查报告。",
           "- 时间取北京时间。太阳方向依据 NOAA 公式、经度与均时差计算。基准累计以 5 分钟中点判光，相邻见光区间合并。结果描述指定窗口的晴空直射。",
           "- 模型对周边楼高、户界、首层起算面和部分屋面高度采用估值。精细窗洞、实际阳台、室内和下沉空间需结合单体平面与剖面核对。[建模说明](../data/建模说明.md) · [规划公示](https://ghhzrzy.tj.gov.cn/ywpd/cxgh_43015/ghgs/202611/t20261102_7648329.html)", ""]
    for bridge in (b for b in model["buildings"] if b.get("kind") == "bridge"):
        body += [f"- {bridge.get('chart_name', bridge['id'])}的桥面和桥墩分别参与射线求交。"
                 "桥面高度及其估算依据见[周边遮挡体高度复核](周边楼高复核.md)。", ""]
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(body),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--details",type=Path,default=OUTPUT/"sunlight_details.csv")
    parser.add_argument("--charts-dir",type=Path,default=CHARTS)
    parser.add_argument("--report",type=Path,default=OUTPUT/"采光选房排雷报告.md")
    args=parser.parse_args()
    setup()
    model=read_model(args.model)
    groups,year=load_table(model,args.details)
    for day,filename in (("大寒","01_dahan_heatmap.png"),("冬至","02_dongzhi_heatmap.png")):
        heatmap(model,groups,day,args.charts_dir/filename,year)
    ranking(groups,args.charts_dir/"03_sunlight_summary_ranking.png")
    report(model,groups,args.report,year)

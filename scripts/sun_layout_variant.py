"""Recalculate alternative household counts for specified towers in the full obstruction scene."""
import argparse
from pathlib import Path

from simulate_sunlight import compute_rows, export
from sun_model import MODEL, OUTPUT, read_model
from sun_households import export_households


def select_layout(model, name):
    alternative = model["site"]["unit_layout_scenarios"][name]
    overrides = alternative["overrides"]
    eligible = {b["id"] for b in model["buildings"] if b["is_target"]}
    if not overrides or not set(overrides).issubset(eligible):
        raise ValueError("情景必须针对模型内的住宅楼")
    buildings=[]
    for b in model["buildings"]:
        if b["id"] in overrides:
            buildings.append(dict(b, **overrides[b["id"]]))
        else:
            buildings.append(dict(b,is_target=False))
    return dict(model,buildings=buildings,selected_unit_layout=name)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=MODEL)
    parser.add_argument("--scenario",default="north_six")
    parser.add_argument("--year",type=int)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT/"10-11号楼_六户对照")
    args=parser.parse_args()
    model=select_layout(read_model(args.model),args.scenario)
    rows=compute_rows(model,args.year or model["site"]["year"])
    print(export(rows,args.output_dir))
    print(export_households(rows,args.output_dir))

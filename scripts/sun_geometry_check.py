"""Analytic intersection-area check for digitized residential footprints."""
import numpy as np
from sun_engine import _triangulate
from sun_model import targets, facade_probes, receivers, solid_parts
from matplotlib.path import Path


def cross(u, v):
    return u[0]*v[1] - u[1]*v[0]


def triangle_intersection_area(t, clip):
    subject = list(t)
    for a, b in zip(clip,np.roll(clip,-1,axis=0)):
        output=[]
        for start,stop in zip(subject,subject[1:]+subject[:1]):
            d0=cross(b-a,start-a)
            d1=cross(b-a,stop-a)
            if (d0>0) != (d1>0):
                output.append(start+(stop-start)*(d0/(d0-d1)))
            if d1>=0:
                output.append(stop)
        subject=output
        if len(subject)<3:
            return 0.
    pts=np.asarray(subject)-subject[0]
    return abs(sum(cross(v,w) for v,w in zip(pts,np.roll(pts,-1,axis=0))))/2


def check_layout(model):
    buildings = model["buildings"]
    houses = targets(model)
    receiver_keys, _, _ = receivers(model)
    site=Path(model["site_polygon"])
    prepared=[]
    for b in buildings:
        solids = solid_parts(b)
        # Each bridge span and pier occupies a different XY and Z interval.
        parts = list(solids) if b.get("kind") == "bridge" else [next(solids)]
        for part in parts:
            poly,idx=_triangulate(part["polygon"])
            lo = np.r_[poly.min(axis=0), part["base"]]
            hi = np.r_[poly.max(axis=0), part["top"]]
            if not np.isfinite([part["base"], part["top"]]).all() or hi[2] <= lo[2]:
                raise ValueError(f'{b["id"]} 的实体底标高、顶标高次序异常')
            prepared.append((b,poly[idx],lo,hi))
        if b["is_target"]:
            if not site.contains_points(poly,radius=.005).all():
                raise ValueError(f'{b["id"]} 顶点超出地块')
            probes=facade_probes(b)
            if Path(poly).contains_points([xy for _,_,xy,_ in probes],radius=.001).any():
                raise ValueError(f'{b["id"]} 有窗台点落在本楼内部')
    checked=0
    house_pairs=0
    maximum=0.
    for i,(a,aa,lo,hi) in enumerate(prepared):
        for b,bb,other_lo,other_hi in prepared[i+1:]:
            if a is b:
                continue
            # Lower podium and door halls may adjoin or underlie residential prisms.
            if not all(v["is_target"] or v.get("group") == "external" for v in (a,b)):
                continue
            house_pairs += int(a["is_target"] and b["is_target"])
            checked+=1
            if (lo>=other_hi).any() or (other_lo>=hi).any():
                continue
            area=sum(triangle_intersection_area(s,t) for s in aa for t in bb
                     if (s.min(axis=0)<t.max(axis=0)).all() and (t.min(axis=0)<s.max(axis=0)).all())
            maximum=max(maximum,area)
            if area>.015:
                raise ValueError(f'{a["id"]} 与 {b["id"]} 重叠 {area:.3f} m²，请检查轮廓')
    households = {key[:3] for key in receiver_keys}
    print(f"住宅 {len(houses)} 栋、{house_pairs} 对；含外部共 {checked} 对轮廓，最大相交面积 {maximum:.4f} m²；"
          f"模拟 {len(households)} 个楼层户位，南向窗台测点 {len(receiver_keys)} 个。")

"""Encode the winter-solstice sunlight movie with a continuous sunrise-to-sunset clock."""
import argparse
from bisect import bisect_right
from datetime import date
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SIZE = (1920,1080)
MAP_SIZE = (1290,736)
MAP_X, MAP_Y = 54,163
PAPER, CARD, INK = "#F6F5F0", "#FCFCF9", "#193E44"
MUTED, LINE = "#48656A", "#C9D6D1"
SUN, DARK = "#F4B443", "#4C6373"
SELECTED = "#276777"
FPS = 30
MINUTES_PER_SECOND = 12


def font_path(bold):
    linux = "/usr/share/fonts/opentype/noto/NotoSansCJK-"
    paths = [os.environ.get("SUNLIGHT_FONT"),
             "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
             linux+("Bold.ttc" if bold else "Regular.ttc"),
             "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
             "/Library/Fonts/Arial Unicode.ttf"]
    for path in paths:
        if path and Path(path).is_file():
            return path
    raise RuntimeError("请将 SUNLIGHT_FONT 设为可读取的中文字体文件")


@lru_cache(maxsize=60)
def font(size,bold=False):
    return ImageFont.truetype(font_path(bold),size)


def label(draw,x,y,s,size=22,color=INK,bold=False,anchor="lt"):
    draw.text((x,y),s,font=font(size,bold),fill=color,anchor=anchor)


def hhmm(minute):
    rounded=round(minute)
    return f"{rounded//60:02d}:{rounded%60:02d}"


def clock_seconds(minute):
    seconds=round(minute*60)
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


class Atlas:
    def __init__(self, scene):
        self.scene=scene
        self.day=scene["days"]["冬至"]
        self.buildings=scene["buildings"]
        self.points=scene["points"]
        self.focus_id, self.unit="1号楼","西户"
        self.floor=1
        self.house=next(b for b in self.buildings if b["id"]==self.focus_id)
        self.focus=[i for i,p in enumerate(self.points)
                    if p["bid"]==self.focus_id and p["unit"]==self.unit and p["label"] in {"A", "B"}]
        self.focus.sort(key=lambda i:self.points[i]["label"])
        if [self.points[i]["label"] for i in self.focus] != ["A","B"]:
            raise ValueError("缺少 1 号楼首个住宅层西户视频观察点 A 或 B")
        self.plane=self.house["sill_base"]
        site=np.asarray(scene["site_polygon"])
        lo=site.min(axis=0)-[12,33]
        hi=site.max(axis=0)+[18,15]
        # Include the stretch beside the parcel without fitting the entire viaduct.
        for b in self.buildings:
            if b.get("kind") != "bridge":
                continue
            for deck in (b, *(p for p in b["parts"] if p.get("kind") == "deck")):
                ring = np.asarray(deck["polygon"])
                xs = [x for x, y in ring if lo[1] <= y <= hi[1]]
                for (x0,y0), (x1,y1) in zip(ring, np.roll(ring,-1,axis=0)):
                    for y in (lo[1], hi[1]):
                        if min(y0,y1) < y < max(y0,y1):
                            xs.append(x0 + (y-y0)*(x1-x0)/(y1-y0))
                if xs:
                    lo[0] = min(lo[0], min(xs)-8)
                    hi[0] = max(hi[0], max(xs)+8)
        self.scale=min(MAP_SIZE/(hi-lo))
        mid=(lo+hi)/2
        self.west=mid[0]-MAP_SIZE[0]/(2*self.scale)
        self.north=mid[1]+MAP_SIZE[1]/(2*self.scale)
        self.prisms=[]
        for b in self.buildings:
            for part in (b,*b.get("parts",[])):
                if part["top"] > self.plane:
                    self.prisms.append((np.asarray(part["polygon"],float),
                                        max(self.plane,part["base"]),part["top"]))
        all_xy=np.concatenate([p for p,_,_ in self.prisms])
        corners=np.array([[self.west,self.north], [self.west+MAP_SIZE[0]/self.scale,self.north-MAP_SIZE[1]/self.scale]])
        self.max_cast=np.linalg.norm(np.ptp(np.vstack((all_xy,corners)),axis=0))+2
        self.base_map=self.map_background()
        self.shaded_map=Image.blend(self.base_map,Image.new("RGB",MAP_SIZE,"#79939F"),.55)
        self.twilight_map=Image.blend(self.base_map,Image.new("RGB",MAP_SIZE,"#5D6F89"),.23)
        self.boards=self.static_page()
        self.states=np.array([[bool(int(f["masks"][i]) & (1<<(self.floor-1))) for i in self.focus]
                              for f in self.day["frames"]])
        weights=np.array([f["end"]-f["start"] for f in self.day["frames"]])
        self.before=np.vstack(([0.,0.],np.cumsum(self.states*weights[:,None],axis=0)))
        self.starts=[f["start"] for f in self.day["frames"]]
        self.sun_minutes=[k["minute"] for k in self.day["solar_path"]]
        self.sun_dirs=np.array([k["direction"] for k in self.day["solar_path"]])

    def xy(self, coords):
        a=np.asarray(coords)
        return np.column_stack(((a[:,0]-self.west)*self.scale,(self.north-a[:,1])*self.scale)).round().astype(int)

    def pt(self,x,y):
        return round((x-self.west)*self.scale),round((self.north-y)*self.scale)

    def polygon(self, draw, points, fill, stroke=None, width=1):
        px=self.xy(points)
        pts=[tuple(p) for p in px]
        draw.polygon(pts,fill=fill)
        if stroke:
            draw.line([*pts,pts[0]],fill=stroke,width=width,joint="curve")

    def map_background(self):
        img=Image.new("RGB",MAP_SIZE,"#EDF0EA")
        d=ImageDraw.Draw(img)
        for x in range(-100,380,20):
            u,_=self.pt(x,0);d.line((u,0,u,MAP_SIZE[1]),fill="#DDE3DA",width=1)
        for y in range(-100,220,20):
            _,v=self.pt(0,y);d.line((0,v,MAP_SIZE[0],v),fill="#DDE3DA",width=1)
        self.polygon(d,self.scene["site_polygon"],"#F7F8F1","#90A7A1",3)
        for b in self.buildings:
            if b.get("group")=="podium":
                self.polygon(d,b["polygon"],"#E9EFE5","#B1C2B7",2)
        return img

    def static_page(self):
        img=Image.new("RGB",SIZE,PAPER)
        d=ImageDraw.Draw(img)
        label(d,54,34,"天津河西  /  新八大里一里南侧地块",24,MUTED)
        label(d,52,76,"冬至日照",57,INK,True)
        label(d,326,98,"首个住宅层的南向窗",33,INK)
        stamp=date.fromisoformat(str(self.day["date"]))
        label(d,1865,47,stamp.strftime("%Y.%m.%d"),31,INK,True,"rt")
        label(d,1865,96,f'日出 {hhmm(self.day["start"])}    日落 {hhmm(self.day["end"])}　北京时间',24,MUTED,anchor="rt")
        d.line((54,145,1866,145),fill=LINE,width=2)
        x0,x1=1371,1866
        d.rounded_rectangle((x0,163,x1,899),radius=13,fill=CARD,outline=LINE,width=2)
        for y in (420,569):
            d.line((x0+27,y,x1-28,y),fill=LINE,width=2)
        label(d,1402,192,"北京时间",25,MUTED)
        label(d,1402,346,"太阳高度",23,MUTED)
        label(d,1625,346,"太阳方位",23,MUTED)
        label(d,1402,448,"观察户",23,MUTED)
        label(d,1402,491,"1 号楼　首层西户",38,INK,True)
        label(d,1402,590,"南窗测点",23,MUTED)
        label(d,1837,590,"从日出累计至此刻",21,MUTED,anchor="rt")
        label(d,1470,850,hhmm(self.day["start"]),18,MUTED)
        label(d,1828,850,hhmm(self.day["end"]),18,MUTED,anchor="rt")
        d.line((54,923,1866,923),fill=LINE,width=2)
        label(d,56,950,f"图示阴影投至 {self.plane:.1f} m 窗台水平面",26,INK)
        d.ellipse((631,951,653,973),fill=SUN,outline="#875C22",width=2)
        label(d,666,949,"窗台点直射",25,INK)
        d.ellipse((883,951,905,973),fill=DARK,outline="#334953",width=2)
        label(d,918,949,"窗台点无直射",25,INK)
        label(d,1866,950,"阴影连续移动，窗点按采样区间判光",23,MUTED,anchor="rt")
        label(d,56,1017,"底图取自天津市规划和自然资源局一期总平公示。分户与窗位按面宽估计，周边体量采用基准高度。",21,MUTED)
        return img

    def shadow_mask(self,direction):
        ray=np.asarray(direction)
        mask=Image.new("L",MAP_SIZE,0)
        d=ImageDraw.Draw(mask)
        if ray[2]<=0:
            d.rectangle((0,0,*MAP_SIZE),fill=255)
            return mask
        slope=-ray[:2]/ray[2]
        # A shadow cast farther than every visible receiver can stop outside the
        # viewport. This keeps near-horizon projections in Pillow's integer range.
        max_height=self.max_cast/max(np.linalg.norm(slope),1e-12)
        # Projection of the closed solid: both caps plus all side faces.
        # Raster union also preserves concave footprints, shared sides and voids.
        for p,bottom,top in self.prisms:
            low=self.xy(p+min(bottom-self.plane,max_height)*slope)
            high=self.xy(p+min(top-self.plane,max_height)*slope)
            lp,hp=[tuple(x) for x in low],[tuple(x) for x in high]
            d.polygon(lp,fill=255);d.polygon(hp,fill=255)
            for i in range(len(lp)):
                j=(i+1)%len(lp)
                d.polygon([lp[i],lp[j],hp[j],hp[i]],fill=255)
        return mask

    def map_frame(self,frame,direction):
        altitude=math.degrees(math.asin(direction[2]))
        if altitude<=0:
            tile=self.twilight_map.copy()
        else:
            tile=Image.composite(self.shaded_map,self.base_map,self.shadow_mask(direction))
            if altitude<2.5:
                fraction=altitude/2.5
                tile=Image.blend(self.twilight_map,tile,fraction*fraction*(3-2*fraction))
        d=ImageDraw.Draw(tile)
        for b in self.buildings:
            if b["group"]=="podium":
                continue
            if b.get("kind") == "bridge":
                decks = [b, *(p for p in b["parts"] if p.get("kind") == "deck")]
                for deck in decks:
                    self.polygon(d,deck["polygon"],"#A19586","#726856",2)
                    if track := deck.get("centerline"):
                        d.line([tuple(xy) for xy in self.xy(track)], fill="#FAE8BB", width=2)
                track = b.get("centerline", [])
                mid_y = self.north - MAP_SIZE[1]/(2*self.scale)
                for (x0,y0),(x1,y1) in zip(track,track[1:]):
                    if min(y0,y1) <= mid_y < max(y0,y1):
                        u,v = self.pt(x0+(x1-x0)*(mid_y-y0)/(y1-y0), mid_y)
                        d.text((u,v), b["name"].split("（")[0], font=font(21,True), fill=CARD,
                               stroke_width=2, stroke_fill="#625847", anchor="mm")
                        break
                continue
            is_home=b["is_target"]
            selected=b["id"]==self.focus_id
            color=SELECTED if selected else ("#FAF9EF" if is_home else "#C4CDD0")
            line=INK if is_home else "#8B9DA3"
            for part in b.get("parts",[]):
                self.polygon(d,part["polygon"],"#E6E8DD","#91A5A1",1)
            self.polygon(d,b["polygon"],color,line,3 if selected else 2)
            if is_home:
                p=np.asarray(b["polygon"])
                x,y=(p.min(axis=0)+p.max(axis=0))/2
                u,v=self.pt(x,y-.8)
                label(d,u,v,b["id"],22,"white" if selected else INK,selected,"mm")
        for i,p in enumerate(self.points):
            u,v=self.pt(p["x"],p["y"])
            lit=bool(int(frame["masks"][i]) & (1<<(self.floor-1)))
            focus=i in self.focus
            rad=8 if focus else 5
            d.ellipse((u-rad-1,v-rad-1,u+rad+1,v+rad+1),fill=CARD)
            d.ellipse((u-rad,v-rad,u+rad,v+rad),fill=SUN if lit else DARK,
                      outline="#825C23" if lit else "#263C48",width=2)
            if focus:
                d.line((u,v+rad+1,u,v+14),fill=INK,width=2)
                d.rounded_rectangle((u-10,v+14,u+10,v+35),radius=4,fill="#F7F7F0",outline=INK,width=1)
                label(d,u,v+24,p["label"],16,INK,True,"mm")
        for x,y,s in ((107,119,"嘉 江 道"),(101,-15,"沐 江 道")):
            u,v=self.pt(x,y);label(d,u,v,s,21,"#475E60",anchor="mm")
        # Keep labels on the margins separate from the footprint and the shadow layer.
        d.rounded_rectangle((MAP_SIZE[0]-76,15,MAP_SIZE[0]-16,103),radius=8,fill=CARD,outline=LINE)
        u=MAP_SIZE[0]-46
        label(d,u,24,"N",22,INK,True,"mt")
        d.polygon([(u,49),(u-11,80),(u,74)],fill=INK)
        d.polygon([(u,49),(u+11,80),(u,74)],fill="#779797")
        u,v=24,MAP_SIZE[1]-19
        bar=round(20*self.scale)
        d.rounded_rectangle((u-13,v-39,u+bar+19,v+11),radius=5,fill=CARD)
        label(d,u,v-32,"20 m",18,INK)
        d.line((u,v,u+bar,v),fill=INK,width=4)
        d.line((u,v-6,u,v+2),fill=INK,width=2);d.line((u+bar,v-6,u+bar,v+2),fill=INK,width=2)
        return tile

    def frame(self,minute):
        index=max(0,bisect_right(self.starts,minute)-1)
        current=self.day["frames"][index]
        left=max(0,min(bisect_right(self.sun_minutes,minute)-1,len(self.sun_minutes)-2))
        t=(minute-self.sun_minutes[left])/(self.sun_minutes[left+1]-self.sun_minutes[left])
        direction=(1-t)*self.sun_dirs[left]+t*self.sun_dirs[left+1]
        direction/=np.linalg.norm(direction)
        image=self.boards.copy()
        image.paste(self.map_frame(current,direction),(MAP_X,MAP_Y))
        d=ImageDraw.Draw(image)
        d.rectangle((MAP_X,MAP_Y,MAP_X+MAP_SIZE[0]-1,MAP_Y+MAP_SIZE[1]-1),outline=LINE,width=2)
        label(d,1398,232,clock_seconds(minute),64,INK,True)
        label(d,1402,312,f'窗点采样　{current["time"]}',22,MUTED)
        label(d,1402,377,f'{math.degrees(math.asin(direction[2])):.1f}°',31,INK,True)
        label(d,1625,377,f'{math.degrees(math.atan2(direction[0],direction[1]))%360:.1f}°',31,INK,True)
        total=self.day["end"]-self.day["start"]
        bx,ex=1470,1828
        def track_x(t):
            return bx+(ex-bx)*(t-self.day["start"])/total
        for j,i in enumerate(self.focus):
            p=self.points[i]; y=634+j*106
            lit=self.states[index,j]
            d.rounded_rectangle((1401,y,1452,y+43),radius=8,fill=SUN if lit else DARK)
            label(d,1426,y+20,p["label"],27,INK if lit else "white",True,"mm")
            label(d,bx,y+2,"直射" if lit else "无直射",28,INK)
            minutes=self.before[index,j]+(minute-current["start"] if lit else 0)
            label(d,1836,y+5,f"{int(minutes)} 分钟",27,INK,True,"rt")
            by=y+52;hh=16
            d.rectangle((bx,by,ex,by+hh),fill="#B9C7C9")
            for t,entry in enumerate(self.day["frames"]):
                if self.states[t,j]:
                    d.rectangle((round(track_x(entry["start"])),by,round(track_x(entry["end"])),by+hh),
                                fill="#E9AD45")
            v=round(track_x(minute))
            d.line((v,by-5,v,by+hh+5),fill=INK,width=3)
            d.polygon([(v,by-2),(v-5,by-9),(v+5,by-9)],fill=INK)
        return image


def render(data_path,output_path):
    renderer=Atlas(json.loads(Path(data_path).read_text(encoding="utf-8")))
    tool=shutil.which("ffmpeg")
    if not tool:
        raise RuntimeError("视频编码需本机 ffmpeg（含 libx264）")
    out=Path(output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sun-video-",dir=out.parent) as tmp:
        target=Path(tmp)/"winter.mp4"
        command=[tool,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
                 "-video_size",f"{SIZE[0]}x{SIZE[1]}","-framerate",str(FPS),"-i","pipe:0",
                 "-an","-vf","scale=out_color_matrix=bt709,format=yuv420p","-c:v","libx264",
                 "-preset","fast","-crf","18","-colorspace","bt709","-color_primaries","bt709",
                 "-color_trc","bt709","-movflags","+faststart",str(target)]
        with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
            try:
                start,end=renderer.day["start"],renderer.day["end"]
                count=math.ceil((end-start)*FPS/MINUTES_PER_SECOND)+1
                for minute in np.linspace(start,end,count):
                    proc.stdin.write(renderer.frame(minute).tobytes())
                proc.stdin.close()
                if proc.wait():
                    raise RuntimeError("ffmpeg 编码失败")
            finally:
                if proc.poll() is None:
                    proc.kill()
        target.replace(out)
    print(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path,help="动画几何与逐窗射线数据 JSON")
    parser.add_argument("--output",type=Path,default=ROOT/"output"/"一里冬至日照动画.mp4")
    args=parser.parse_args()
    render(args.data,args.output)

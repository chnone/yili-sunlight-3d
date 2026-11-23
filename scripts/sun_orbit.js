// Orthographic ENU renderer. Geometry, depth-tested edges and picking share
// a camera buffer. Advancing the sun only re-shades its visible pixel spans.
class SunOrbit {
  constructor(scene, canvas, initialScope='all') {
    this.scene = scene;
    this.canvas = canvas;
    this.layer = document.createElement('canvas');
    this.ctx = this.layer.getContext('2d');
    this.owners = scene.buildings;
    this.index = new Map(this.owners.map((b, i) => [b.id, i]));
    this.faces = [];
    this.edges = [];
    this.vertices = [];
    this.residentialWalls = new Map();
    for (const [owner, b] of this.owners.entries()) {
      for (const solid of [b, ...(b.parts || [])]) this.extrude(solid, owner);
    }
    this.camera = {yaw: 0, elevation: 0, zoom: 1, x: 0, y: 0, z: 0};
    this.reset(initialScope);
  }

  static triangles(polygon) {
    const cross = (a, b, c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const p = [];
    for (const point of polygon) {
      if (!p.length || Math.hypot(point[0]-p.at(-1)[0], point[1]-p.at(-1)[1]) > 1e-7) p.push(point);
    }
    if (Math.hypot(p[0][0]-p.at(-1)[0], p[0][1]-p.at(-1)[1]) < 1e-7) p.pop();
    for (let k = 0; k < p.length && p.length > 3;) {
      if (Math.abs(cross(p[(k+p.length-1)%p.length],p[k],p[(k+1)%p.length])) < 1e-8) {
        p.splice(k,1); k=0;
      } else k++;
    }
    const area = p.reduce((a,v,i) => a + v[0]*p[(i+1)%p.length][1]-v[1]*p[(i+1)%p.length][0], 0);
    if (area < 0) p.reverse();
    const ids = p.map((_,i)=>i), triangles = [];
    while (ids.length>3) {
      const ear = ids.findIndex((id,j) => {
        const a=p[ids[(j+ids.length-1)%ids.length]], b=p[id], c=p[ids[(j+1)%ids.length]];
        if (cross(a,b,c) <= 1e-8) return false;
        return !ids.some(k => {
          const q=p[k];
          return q!==a && q!==b && q!==c && cross(a,b,q)>=-1e-8 && cross(b,c,q)>=-1e-8 && cross(c,a,q)>=-1e-8;
        });
      });
      if (ear<0) throw Error('三维底面无法三角化，请检查自交轮廓');
      triangles.push([ids[(ear+ids.length-1)%ids.length],ids[ear],ids[(ear+1)%ids.length]]);
      ids.splice(ear,1);
    }
    triangles.push(ids);
    return {p,triangles};
  }

  extrude(solid, owner) {
    const {p,triangles} = SunOrbit.triangles(solid.polygon);
    const start=this.vertices.length, count=p.length;
    const base=p.map(([x,y])=>[x,y,solid.base]), roof=p.map(([x,y])=>[x,y,solid.top]);
    this.vertices.push(...base,...roof);
    const add=(i,j,k,normal,roof)=>this.faces.push({ids:[start+i,start+j,start+k],owner,normal,roof});
    const edge=(i,j)=>this.edges.push([start+i,start+j,owner]);
    for(const [a,b,c] of triangles) {
      add(a+count,b+count,c+count,[0,0,1],true);
      add(c,b,a,[0,0,-1],false);
    }
    for(let i=0;i<count;i++) {
      const j=(i+1)%count, dx=p[j][0]-p[i][0], dy=p[j][1]-p[i][1], len=Math.hypot(dx,dy);
      const normal=[dy/len,-dx/len,0];
      if(solid===this.owners[owner] && solid.is_target){
        if(!this.residentialWalls.has(solid.id))this.residentialWalls.set(solid.id,[]);
        this.residentialWalls.get(solid.id).push({x:p[i][0],y:p[i][1],tx:dx/len,ty:dy/len,len,face:this.faces.length});
      }
      add(i,j,j+count,normal,false);add(i,j+count,i+count,normal,false);
      edge(i+count,j+count);edge(i,i+count);
    }
  }

  reset(scope=this.scope) {
    this.scope=scope;
    const buildings=this.owners.filter(b=>scope==='all' || b.group!=='external');
    this.fit(SunOrbit.extentPoints(buildings));
  }

  focus(b) {
    if(b.kind==='bridge'){
      this.fit(SunOrbit.extentPoints([b,...this.owners.filter(owner=>owner.is_target)]));
    }else this.fit(SunOrbit.extentPoints([b]),true);
  }

  static extentPoints(owners) {
    return owners.flatMap(b=>[b,...(b.parts||[])].flatMap(s=>s.polygon.flatMap(([x,y])=>[[x,y,s.base],[x,y,s.top]])));
  }

  fit(coords, detail=false) {
    const c=this.camera;
    if(!detail) {c.yaw=12*Math.PI/180;c.elevation=44*Math.PI/180}
    c.zoom=1;c.x=0;c.y=0;c.z=0;
    const {right,up,toward}=this.axes();
    const b=[Infinity,-Infinity,Infinity,-Infinity,Infinity,-Infinity];
    for(const v of coords) {
      const vals=[SunOrbit.dot(v,right),SunOrbit.dot(v,up),SunOrbit.dot(v,toward)];
      vals.forEach((val,i)=>{b[2*i]=Math.min(b[2*i],val);b[2*i+1]=Math.max(b[2*i+1],val)});
    }
    const mid=[(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2];
    [c.x,c.y,c.z]=[0,1,2].map(i=>right[i]*mid[0]+up[i]*mid[1]+toward[i]*mid[2]);
    const margin=detail?1.75:1.11;
    this.span=[Math.max(60,b[1]-b[0])*margin,Math.max(55,b[3]-b[2])*margin];
  }
  static dot(v,n){return v[0]*n[0]+v[1]*n[1]+v[2]*n[2]}
  axes(){
    const {yaw,elevation}=this.camera,ca=Math.cos(yaw),sa=Math.sin(yaw),ce=Math.cos(elevation),se=Math.sin(elevation);
    return {right:[ca,-sa,0],up:[sa*se,ca*se,ce],toward:[-sa*ce,-ca*ce,se]};
  }
  scale(w=this.canvas.clientWidth,h=this.canvas.clientHeight) {
    return Math.min(w/this.span[0],h/this.span[1])*this.camera.zoom;
  }
  orbit(dx,dy) {
    const c=this.camera;
    c.yaw=(c.yaw+dx*.009)%(2*Math.PI);
    c.elevation=Math.max(.10,Math.min(1.53,c.elevation+dy*.008));
  }
  pan(dx,dy) {
    const c=this.camera,{right,up}=this.axes(),s=this.scale();
    // Translate the camera parallel to the image plane so the grabbed object
    // follows the pointer even when the view is almost horizontal.
    c.x+=(-dx*right[0]+dy*up[0])/s;c.y+=(-dx*right[1]+dy*up[1])/s;c.z+=dy*up[2]/s;
  }
  zoom(factor) {this.camera.zoom=Math.max(.08,Math.min(24,this.camera.zoom*factor))}
  cameraView(w,h) {
    const c=this.camera, axes=this.axes(),s=this.scale(w,h);
    const ox=w/2-s*SunOrbit.dot([c.x,c.y,c.z],axes.right);
    const oy=h/2+s*SunOrbit.dot([c.x,c.y,c.z],axes.up);
    const P=(x,y,z=0)=>[ox+s*(axes.right[0]*x+axes.right[1]*y),
      oy-s*(axes.up[0]*x+axes.up[1]*y+axes.up[2]*z)];
    return {s,P,axes};
  }

  // Fill a sun-depth or camera-depth buffer. For the latter also record the
  // front triangle, and keep the barycentric gradients for deferred shading.
  static raster(vertices, ids, width, height, depth, faceMap, faceIndex, gradients) {
    const a=vertices[ids[0]], b=vertices[ids[1]], c=vertices[ids[2]];
    const bx=b[0]-a[0],by=b[1]-a[1],cx=c[0]-a[0],cy=c[1]-a[1],den=bx*cy-by*cx;
    if(Math.abs(den)<1e-7)return;
    const ux=cy/den,uy=-cx/den,vx=-by/den,vy=bx/den;
    if(gradients)gradients[faceIndex]={ux,uy,vx,vy};
    const db=b[2]-a[2],dc=c[2]-a[2],stepX=ux*db+vx*dc,stepY=uy*db+vy*dc;
    let p=a,q=b,r=c;
    if(q[1]<p[1])[p,q]=[q,p];
    if(r[1]<p[1])[p,r]=[r,p];
    if(r[1]<q[1])[q,r]=[r,q];
    const longSlope=(r[0]-p[0])/(r[1]-p[1]);
    // Scan the actual triangle interior. At low sun a thin, diagonal shadow
    // triangle's bounding rectangle can contain thousands of empty pixels.
    for(let half=0;half<2;half++){
      const lo=half===0?p:q,hi=half===0?q:r;
      if(hi[1]<=lo[1])continue;
      const first=Math.max(0,Math.ceil(lo[1]-.5)),last=Math.min(height,Math.ceil(hi[1]-.5));
      const shortSlope=(hi[0]-lo[0])/(hi[1]-lo[1]);
      let left=p[0]+(first+.5-p[1])*longSlope,right=lo[0]+(first+.5-lo[1])*shortSlope;
      for(let y=first;y<last;y++,left+=longSlope,right+=shortSlope){
        const x0=Math.max(0,Math.ceil(Math.min(left,right)-.500001));
        const x1=Math.min(width,Math.floor(Math.max(left,right)-.499999)+1);
        let z=a[2]+(x0+.5-a[0])*stepX+(y+.5-a[1])*stepY;
        const end=y*width+x1;
        for(let j=y*width+x0;j<end;j++,z+=stepX){
          if(z>=depth[j]){
            depth[j]=z;
            if(faceMap)faceMap[j]=faceIndex;
          }
        }
      }
    }
  }

  lighting(dir) {
    const prior=this.light?.direction;
    if(prior&&prior.every((value,i)=>value===dir[i]))return this.light;
    const horiz=Math.hypot(dir[0],dir[1]), r=[dir[1]/horiz,-dir[0]/horiz,0],u=[-r[1]*dir[2],r[0]*dir[2],r[1]*dir[0]-r[0]*dir[1]];
    const positions=this.light?.positions || this.vertices.map(()=>[0,0,0]);
    let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(let i=0;i<positions.length;i++){
      const point=this.vertices[i],pos=positions[i];
      pos[0]=SunOrbit.dot(point,r);pos[1]=SunOrbit.dot(point,u);pos[2]=SunOrbit.dot(point,dir);
      minX=Math.min(minX,pos[0]);maxX=Math.max(maxX,pos[0]);
      minY=Math.min(minY,pos[1]);maxY=Math.max(maxY,pos[1]);
    }
    const size=1152,scale=(size-5)/Math.max(maxX-minX,maxY-minY);
    for(const pos of positions){pos[0]=2+(pos[0]-minX)*scale;pos[1]=2+(pos[1]-minY)*scale;}
    const depth=this.light?.depth || new Float32Array(size*size);
    depth.fill(-Infinity);
    for(const face of this.faces){
      if(SunOrbit.dot(face.normal,dir)>0)SunOrbit.raster(positions,face.ids,size,size,depth);
    }
    this.light={size,depth,positions,direction:dir.slice()};
    return this.light;
  }

  static color(rgb,gain) {
    const scale=c=>Math.round(Math.max(0,Math.min(255,c*gain)));
    return ((255<<24)|(scale(rgb[2])<<16)|(scale(rgb[1])<<8)|scale(rgb[0]))>>>0;
  }

  project(view, w, h) {
    // Depth and visibility depend on the camera only. Sun movement, scrubbing
    // and selecting a different household all reuse this projection.
    const ratio=this.pixelRatio, faces=this.faces, faceMap=this.faceMap;
    this.z.fill(-Infinity);this.ids.fill(-1);faceMap.fill(-1);this.colors.fill(0);
    const coords=this.vertices.map(p=>{
      const [x,y]=view.P(...p);
      return [x*ratio,y*ratio,SunOrbit.dot(p,view.axes.toward)];
    });
    const gradients=[];
    for(let i=0;i<faces.length;i++) {
      const face=faces[i];
      if(SunOrbit.dot(face.normal,view.axes.toward)>1e-7)
        SunOrbit.raster(coords,face.ids,w,h,this.z,faceMap,i,gradients);
    }
    // Encode visible silhouette and roof edges separately. Keep edge ownership
    // identical to the solid behind it, including near closely stacked roofs.
    const err=1.15/(ratio*view.s);
    for(const [a,b,owner] of this.edges) {
      const p=coords[a],q=coords[b],dx=q[0]-p[0],dy=q[1]-p[1],dz=q[2]-p[2];
      const steps=Math.ceil(Math.max(Math.abs(dx),Math.abs(dy)));if(!steps)continue;
      for(let i=0;i<=steps;i++){
        const t=i/steps, x=Math.floor(p[0]+dx*t),y=Math.floor(p[1]+dy*t);
        if(x<0||x>=w||y<0||y>=h)continue;
        const k=y*w+x,f=faceMap[k];
        if(f>=0 && faces[f].owner===owner && this.z[k]<=p[2]+dz*t+err)faceMap[k]=-2-owner;
      }
    }
    // Each row is divided into consecutive pixels of the same face. During
    // playback these spans need no per-pixel triangle/depth/visibility tests.
    this.spans=faces.map(()=>[]);this.edgeSpans=this.owners.map(()=>[]);
    for(let y=0;y<h;y++){
      const row=y*w,end=row+w;
      for(let left=row;left<end;){
        const f=faceMap[left];let right=left+1;
        while(right<end&&faceMap[right]===f)right++;
        if(f!==-1){
          const owner=f<0?-2-f:faces[f].owner;
          this.ids.fill(owner,left,right);
          if(f<0)this.edgeSpans[owner].push(left,right);
          else{
            const {ux,uy,vx,vy}=gradients[f],a=coords[faces[f].ids[0]];
            const dx=left-row+.5-a[0],dy=y+.5-a[1];
            this.spans[f].push(left,right,dx*ux+dy*uy,dx*vx+dy*vy);
          }
        }
        left=right;
      }
    }
    this.gradients=gradients;
  }

  shade(frame, palette) {
    const dir=frame.direction,daylight=dir[2]>0;
    const light=daylight?this.lighting(dir):null;
    const colors=this.colors;
    for(let i=0;i<this.faces.length;i++) {
      const spans=this.spans[i];if(!spans.length)continue;
      const face=this.faces[i],sun=daylight?Math.max(0,SunOrbit.dot(face.normal,dir)):0;
      const gain=face.roof?1.06+.10*sun:.76+.28*sun+.05*face.normal[0];
      const base=palette[face.owner],lit=SunOrbit.color(base,gain);
      if(sun<=1e-6){
        for(let j=0;j<spans.length;j+=4)colors.fill(lit,spans[j],spans[j+1]);
        continue;
      }
      const contrast=.21*Math.min(1,sun/.025,dir[2]/.013);
      const shaded=SunOrbit.color(base,gain*(1-contrast)),size=light.size,depth=light.depth;
      const {ux,vx}=this.gradients[i],a=light.positions[face.ids[0]],b=light.positions[face.ids[1]],c=light.positions[face.ids[2]];
      const bx=b[0]-a[0],by=b[1]-a[1],bz=b[2]-a[2],cx=c[0]-a[0],cy=c[1]-a[1],cz=c[2]-a[2];
      const stepX=ux*bx+vx*cx,stepY=ux*by+vx*cy,stepZ=ux*bz+vx*cz;
      const det=bx*cy-cx*by,slX=(bz*cy-cz*by)/det,slY=(bx*cz-cx*bz)/det;
      for(let j=0;j<spans.length;j+=4) {
        const end=spans[j+1],u=spans[j+2],v=spans[j+3];
        let sx=a[0]+u*bx+v*cx,sy=a[1]+u*by+v*cy,sz=a[2]+u*bz+v*cz;
        for(let k=spans[j];k<end;k++,sx+=stepX,sy+=stepY,sz+=stepZ){
          // All solar projections lie within the padded shadow atlas. Compare
          // at the texel centre so grazing rays do not stripe flat surfaces.
          const x=Math.floor(sx),y=Math.floor(sy);
          const z=sz+(x+.5-sx)*slX+(y+.5-sy)*slY;
          colors[k]=depth[y*size+x]>z+.035?shaded:lit;
        }
      }
    }
    for(let i=0;i<this.edgeSpans.length;i++){
      const edges=this.edgeSpans[i],color=SunOrbit.color(palette[i],.68);
      for(let j=0;j<edges.length;j+=2)colors.fill(color,edges[j],edges[j+1]);
    }
  }

  draw(context,frame,selected,exterior,isContributor) {
    const cssW=this.canvas.clientWidth,cssH=this.canvas.clientHeight;
    // The labels and exact ray samples are independent from the colour raster.
    // While a pointer is moving, spend the frame budget on immediate camera
    // feedback. Restore the higher-resolution buffer on release.
    const dragging=this.canvas.classList.contains('dragging');
    const maxWidth=dragging?960:selected.playing?1152:1340;
    const maxRatio=dragging?1.25:1.5;
    this.pixelRatio=Math.min(devicePixelRatio||1,maxRatio,maxWidth/cssW);
    const w=Math.max(1,Math.round(cssW*this.pixelRatio)),h=Math.max(1,Math.round(cssH*this.pixelRatio));
    if(w!==this.layer.width || h!==this.layer.height || !this.pixels) {
      this.layer.width=w;this.layer.height=h;
      this.pixels=this.ctx.createImageData(w,h); this.colors=new Uint32Array(this.pixels.data.buffer);
      this.z=new Float32Array(w*h);this.ids=new Int32Array(w*h);this.faceMap=new Int32Array(w*h);
      this.projectionKey=null;
    }
    const c=this.camera,cameraKey=[cssW,cssH,c.yaw,c.elevation,c.zoom,c.x,c.y,c.z,...this.span].join('|');
    const changed=cameraKey!==this.projectionKey;
    if(changed){
      this.view=this.cameraView(cssW,cssH);
      this.project(this.view,w,h);
      this.projectionKey=cameraKey;
    }
    const paletteKey=[selected.day,selected.bid,exterior].join('|');
    if(paletteKey!==this.paletteKey){
      this.palette=this.owners.map(b=>{
        if(b.kind==='bridge')return b.id===exterior?[85,80,128]:[100,110,137];
        if(b.id===exterior)return [172,89,52];
        if(b.id===selected.bid)return [48,108,121];
        if(b.is_target)return b.phase?.includes('后续')?[145,176,174]:[111,155,160];
        if(b.group==='podium')return [202,212,197];
        return isContributor(b)?[193,154,121]:[180,190,185];
      });
      this.paletteKey=paletteKey;
    }
    const shadeKey=paletteKey+'|'+(frame.direction[2]>0?frame.direction.join(','):'below');
    if(changed||shadeKey!==this.shadeKey){
      this.shade(frame,this.palette);
      this.ctx.putImageData(this.pixels,0,0);
      this.shadeKey=shadeKey;
    }
    if(context)context.drawImage(this.layer,0,0,cssW,cssH);
    return this.view;
  }

  hit(x,y) {
    if(!this.ids)return null;
    const px=Math.floor(x*this.pixelRatio),py=Math.floor(y*this.pixelRatio);
    if(px<0||px>=this.layer.width||py<0||py>=this.layer.height)return null;
    return this.owners[this.ids[px+py*this.layer.width]]||null;
  }

  // Anchor a rank marker to the actual wall immediately inside a sill probe.
  // The strip is a locator symbol. Its width is limited by the wall corners.
  sillMarker(bid, point, sill, floorHeight, halfWidth) {
    let best=null,nearest=Infinity;
    for(const wall of this.residentialWalls.get(bid)){
      const u=Math.max(0,Math.min(wall.len,(point.x-wall.x)*wall.tx+(point.y-wall.y)*wall.ty));
      const dx=wall.x+u*wall.tx-point.x,dy=wall.y+u*wall.ty-point.y;
      const dist=dx*dx+dy*dy;
      if(dist<nearest){nearest=dist;best={wall,u}}
    }
    const {wall,u}=best;
    const ends=[Math.max(0,u-halfWidth),Math.min(wall.len,u+halfWidth)];
    return {ends:ends.map(t=>[wall.x+t*wall.tx,wall.y+t*wall.ty,sill]),height:floorHeight*.48,face:wall.face};
  }

  visibleMarkerPath(marker) {
    const {P}=this.view,ratio=this.pixelRatio,w=this.layer.width,h=this.layer.height;
    const [left,right]=marker.ends.map(p=>P(...p)).sort((a,b)=>a[0]-b[0]);
    const width=right[0]-left[0];
    if(width<.01)return null;
    const rise=marker.height*this.view.s*this.view.axes.up[2];
    const x0=Math.max(0,Math.ceil(left[0]*ratio-.5)),x1=Math.min(w,Math.ceil(right[0]*ratio-.5));
    let path=null;
    for(let x=x0;x<x1;x++){
      const t=((x+.5)/ratio-left[0])/width,bottom=left[1]+t*(right[1]-left[1]);
      const y0=Math.max(0,Math.ceil((bottom-rise)*ratio-.5)),y1=Math.min(h,Math.ceil(bottom*ratio-.5));
      let start=-1;
      for(let y=y0;y<=y1;y++){
        const face=y<y1?this.faceMap[y*w+x]:-1;
        if(face===marker.face||face===marker.face+1){if(start===-1)start=y}
        else if(start!==-1){
          if(!path)path=new Path2D();
          path.rect(x/ratio,start/ratio,1/ratio,(y-start)/ratio);
          start=-1;
        }
      }
    }
    return path;
  }
  visible(x,y,z,owner) {
    const pos=this.view.P(x,y,z),px=Math.floor(pos[0]*this.pixelRatio),py=Math.floor(pos[1]*this.pixelRatio);
    if(px<0||py<0||px>=this.layer.width||py>=this.layer.height)return false;
    const k=px+py*this.layer.width,d=SunOrbit.dot([x,y,z],this.view.axes.toward);
    const same=this.ids[k]===this.index.get(owner);
    // A different solid must still hide a marker near a silhouette.
    return d+(same?1.3:.35)/(this.pixelRatio*this.view.s)>=this.z[k];
  }
}

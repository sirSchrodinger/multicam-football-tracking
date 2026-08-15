"""Gerçek görüntü + saha çizgileri (kalibrasyon) + oyuncu tespit kutuları/ayak-nokta.
'Sahayı nasıl alıyorum, oyuncuları nasıl tespit ediyorum' görseli."""
import json, subprocess, tempfile, sys
from pathlib import Path
import numpy as np, cv2, pandas as pd
sys.path.insert(0,".")
from pitch.homography import PitchHomography
from self_training.pseudo_label import reconstruct_bbox_xyxy

def render(tracks_p, calib_p, video, out_mp4, t0, t1, fps_out=12):
    df=pd.read_parquet(tracks_p); homo=PitchHomography.load(calib_p)
    idc="player_id" if "player_id" in df.columns else "tid"
    bx=reconstruct_bbox_xyxy(df); df=df.assign(x1=bx[:,0],y1=bx[:,1],x2=bx[:,2],y2=bx[:,3])
    win=df[(df.t_sec>=t0)&(df.t_sec<=t1)]
    pids=sorted(df[idc].unique()); rng=np.random.default_rng(0)
    pal={p:tuple(int(c) for c in rng.integers(60,255,3)) for p in pids}
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
    fb={f:g for f,g in win.groupby("frame")}; allf=sorted(fb); ft=np.array([fb[f].t_sec.iloc[0] for f in allf])
    # saha çizgi segmentleri (metre) -> HAM piksel (distorted=True)
    segs=homo._template_segments_m()
    tmp=Path(tempfile.mkdtemp())
    for k,t in enumerate(np.arange(t0,t1,1.0/fps_out)):
        fi=allf[int(np.argmin(np.abs(ft-t)))]; cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,fr=cap.read()
        if not ok: continue
        # 1) SAHA çizgileri (yeşil) — kalibrasyondan, YOĞUN-ÖRNEKLE (fisheye eğrisine otur)
        Hh,Ww=fr.shape[:2]
        for (p0,p1) in segs:
            p0=np.array(p0,float); p1=np.array(p1,float); n=40
            pts_m=p0[None,:]*(1-np.linspace(0,1,n)[:,None])+p1[None,:]*np.linspace(0,1,n)[:,None]
            px=homo.pitch_to_pixel(pts_m,distorted=True)
            for i in range(len(px)-1):
                a,b=px[i],px[i+1]
                if all(np.isfinite(a)) and all(np.isfinite(b)) and -200<a[0]<Ww+200 and -200<a[1]<Hh+200:
                    cv2.line(fr,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),(0,255,0),2)
        # merkez yuvarlak
        cc=homo._dims_m(); th=np.linspace(0,2*np.pi,40); circ=np.c_[cc[0]/2+4*np.cos(th),cc[1]/2+4*np.sin(th)]
        cp=homo.pitch_to_pixel(circ,distorted=True)
        for i in range(len(cp)-1): cv2.line(fr,(int(cp[i,0]),int(cp[i,1])),(int(cp[i+1,0]),int(cp[i+1,1])),(0,255,0),2)
        # 2) OYUNCU tespit kutuları + ayak-nokta
        g=fb[fi]; n=0
        for _,r in g.iterrows():
            col=pal[r[idc]]; cv2.rectangle(fr,(int(r.x1),int(r.y1)),(int(r.x2),int(r.y2)),col,2)
            cv2.circle(fr,(int(r.foot_x),int(r.foot_y)),4,(0,0,255),-1)  # ayak-nokta (konum)
            cv2.putText(fr,f"#{int(r[idc])}",(int(r.x1),int(r.y1)-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1); n+=1
        cv2.putText(fr,"YESIL=saha (kalibrasyon)  KUTU=oyuncu tespit  KIRMIZI nokta=ayak/konum",(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(fr,f"t={t:.1f}s  tespit={n}",(15,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imwrite(str(tmp/f"f{k:05d}.png"),fr)
    cap.release()
    r=subprocess.run(["ffmpeg","-y","-framerate",str(fps_out),"-i",str(tmp/"f%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-crf","23",str(out_mp4)],capture_output=True,text=True)
    for p in tmp.glob("*.png"): p.unlink()
    tmp.rmdir()
    if r.returncode: raise RuntimeError(r.stderr[-400:])
    return out_mp4

if __name__=="__main__":
    # yoğun pencere bul
    df=pd.read_parquet("raw/tracks_active_game.parquet"); idc="player_id" if "player_id" in df.columns else "tid"
    per=df.groupby("frame")[idc].nunique(); per=per.reset_index(); per["t"]=per.frame/df.frame.max()*df.t_sec.max()
    per["w"]=(per.t//10).astype(int); bw=int(per.groupby("w")[idc].mean().idxmax()); t0=max(0,bw*10)
    print(f"yoğun pencere t={t0}-{t0+22}")
    render("raw/tracks_active_game.parquet","calib/cankaya_cam2_v2.json","raw/_active_game.mp4","docs/detection_overlay.mp4",t0,t0+22)
    print("docs/detection_overlay.mp4")

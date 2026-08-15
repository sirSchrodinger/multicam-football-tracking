"""cam2 boyunca aktif-oyun periyodu (hareket-proxy, GPU'suz). Saha-merkez bölgesinde
ardışık-kare farkı = oyuncu hareketi. Yüksek = aktif oyun, düşük = ayakta/duruş."""
import cv2, numpy as np, json
src="raw/cankaya_cam2.mp4"; cap=cv2.VideoCapture(src); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
roi=(slice(120,520),slice(700,1750))  # saha merkez/üst bölge (oyun burada)
rows=[]
for t in range(20, int(n/fps)-5, 20):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)); ok,a=cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(t*fps)+3); ok2,b=cap.read()
    if not(ok and ok2): continue
    ga=cv2.cvtColor(a[roi],cv2.COLOR_BGR2GRAY).astype(np.float32); gb=cv2.cvtColor(b[roi],cv2.COLOR_BGR2GRAY).astype(np.float32)
    mot=float(np.mean(np.abs(ga-gb)>18))  # hareketli-piksel oranı
    rows.append((t,mot))
cap.release()
rows=np.array(rows); 
# 5dk pencerelerde ortalama aktivite
best=None
for i in range(len(rows)):
    t0=rows[i,0]; m=rows[(rows[:,0]>=t0)&(rows[:,0]<t0+300),1].mean()
    if best is None or m>best[1]: best=(t0,m)
json.dump({"timeline":rows.tolist(),"best_5min_start":float(best[0]),"best_activity":float(best[1]),
           "top5":sorted(rows.tolist(),key=lambda r:-r[1])[:5]}, open("overnight_out/activity_scan.json","w"),indent=2)
print(f"aktif periyot: t={best[0]:.0f}s (cam2), aktivite {best[1]:.3f}")
print("en aktif 5 örnek (t,mot):",sorted(rows.tolist(),key=lambda r:-r[1])[:5])

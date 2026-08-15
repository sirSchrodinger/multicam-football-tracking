import json, sys, numpy as np, time
sys.path.insert(0,".")
from pathlib import Path
from pitch.template import PitchTemplate
from pitch import auto_calib
from pitch.venue_autocalib import autocalibrate_venue

clip="raw/cankaya_cam2_clip2400.mp4"
calib=json.load(open("calib/cankaya_cam2_v2.json"))
K=np.array(calib["K"]); dist=np.array(calib["dist"])
cache=Path("overnight_out/cankaya_Lstatic.npy")
t0=time.time()
if cache.exists():
    L=np.load(cache); print(f"L_static cache yüklendi {L.shape}")
else:
    print("L_static kuruluyor (150 frame)...")
    L=auto_calib.build_static_line_map(clip, n_frames=150)
    np.save(cache, L); print(f"L_static kuruldu+cache {L.shape} ({time.time()-t0:.0f}s)")
print(f"line-px frac: {(L>0.35).mean():.3f}")
# bilinen distorsiyonla (geometriyi izole et), aspect-doğru template
for L_dim,W_dim in [(34,18)]:
    tpl=PitchTemplate.seven_a_side(L=L_dim, W=W_dim)
    homo, rep = autocalibrate_venue(clip,"cankaya_auto",template=tpl,
                                    distortion=(K,dist), L_static=L)
    print(f"\n=== template {L_dim}x{W_dim} ===")
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print("MANUEL v2 median_px: 9.87 (karşılaştırma)")

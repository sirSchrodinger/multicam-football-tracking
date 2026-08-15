"""Ortak yardimci: bir kalib json + foot-pikseller -> metrik saha (Xm,Ym)."""
import json, numpy as np, cv2
def load(calib_path):
    F=json.load(open(calib_path))
    return dict(K=np.array(F["K"],float), D=np.array(F["dist"],float),
               Hi2p=np.array(F["H_img2pitch"],float),
               SC=F["scale_anchor"]["scale_factor"],
               L=F["template"]["dims_m"][0], W=F["template"]["dims_m"][1])
def to_pitch(cal, xy):
    """xy: (N,2) foot pixels (distorted) -> (N,2) metre (Xm,Ym)."""
    xy=np.asarray(xy,float).reshape(-1,1,2)
    u=cv2.undistortPoints(xy, cal["K"], cal["D"], P=cal["K"]).reshape(-1,2)
    q=cal["Hi2p"] @ np.c_[u, np.ones(len(u))].T
    return np.c_[q[0]/q[2]*cal["SC"], q[1]/q[2]*cal["SC"]]

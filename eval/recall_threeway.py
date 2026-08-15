#!/usr/bin/env python3
"""CEPHE3 — üç-yollu recall: base vs +tile vs +CLAHE, İNSAN-GT'ye karşı.

İnsan-GT = recall_val/score_summary.json (3 kör sayıcı, far/rest zon başına
n_humans / n_base_covered / n_tile_covered / n_uncovered / n_false_pos).

  base   recall = sum(n_base_covered) / sum(n_humans)
  +tile  recall = sum(n_base_covered + n_tile_covered) / sum(n_humans)
  +CLAHE : sayıcılar CLAHE'yi İŞARETLEMEDİ -> insan-GT'ye doğrudan atfedilemez.
           Bu script CLAHE-tile'ın base+tile ÖTESİNDE ürettiği YENİ far-band kutu
           sayısını ölçer (active_reexport.py ile AYNI mantık) ve overlay üretir;
           recall %% olarak DEĞİL "GT-doğrulanmamış ek kutu" olarak raporlar.

DÜRÜSTLÜK: base & +tile recall = insan-sayımı (kanıtlı). +CLAHE = makine-delta,
GT-atıfsız (sayıcı CLAHE turu yok) -> overlay PNG ile manuel doğrulanmalı.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import recall_eval as R
# validated config (active_reexport ile aynı): TILE_UP=2.0 default, aspect-filtre YOK
FAR = 365.0
DEDUP_CLAHE_PX = 22.0
VAL = ROOT / "recall_val"
OUT = ROOT / "scratchpad/wf2_recall"


def clahe_bgr(img, clip=3.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def foots(xy):
    return np.stack([(xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]], 1) if len(xy) else np.empty((0, 2))


def human_gt_recall():
    s = json.loads((VAL / "score_summary.json").read_text())["reconciled"]
    agg = {"far": dict(H=0, b=0, t=0, unc=0, fp=0), "rest": dict(H=0, b=0, t=0, unc=0, fp=0)}
    for fi, v in s.items():
        for z in ("far", "rest"):
            zz = v.get(z, {})
            agg[z]["H"] += zz.get("n_humans", 0)
            agg[z]["b"] += zz.get("n_base_covered", 0)
            agg[z]["t"] += zz.get("n_tile_covered", 0)
            agg[z]["unc"] += zz.get("n_uncovered", 0)
            agg[z]["fp"] += zz.get("n_false_pos", 0)
    return agg


def main():
    agg = human_gt_recall()
    far, rest = agg["far"], agg["rest"]
    allH = far["H"] + rest["H"]; allB = far["b"] + rest["b"]; allT = allB + far["t"] + rest["t"]
    print("================  İNSAN-GT RECALL (3 kör sayıcı, 8 kare)  ================")
    print(f"{'zone':5} {'H':>4} {'base':>5} {'base%':>7} {'+tile':>6} {'+tile%':>7} {'unc':>4} {'fp':>3}")
    for z, a in (("far", far), ("rest", rest)):
        bp = 100 * a["b"] / a["H"] if a["H"] else 0
        tp = 100 * (a["b"] + a["t"]) / a["H"] if a["H"] else 0
        print(f"{z:5} {a['H']:>4} {a['b']:>5} {bp:>6.1f}% {a['b']+a['t']:>6} {tp:>6.1f}% {a['unc']:>4} {a['fp']:>3}")
    print(f"{'ALL':5} {allH:>4} {allB:>5} {100*allB/allH:>6.1f}% {allT:>6} {100*allT/allH:>6.1f}% "
          f"{far['unc']+rest['unc']:>4} {far['fp']+rest['fp']:>3}")
    print(f"\nbase precision (insan): {'100.0%' if (far['fp']+rest['fp'])==0 else '<100%'} (fp={far['fp']+rest['fp']})")

    # ---- CLAHE-tile makine-delta (base+tile ötesinde), active_reexport ile aynı mantık ----
    print("\n================  +CLAHE makine-delta (GT-atıfsız)  ================")
    model = R._load_model()
    tot_tile = tot_clahe = 0
    overlays = {}
    rows = []
    for fi in R.FROZEN:
        fr = cv2.imread(str(VAL / f"frames/f{fi}.png"))
        b_xy, _ = R._base_detect(model, fr)
        t_xy, _ = R._tile_far(model, fr, b_xy)
        frC = clahe_bgr(fr)
        tc_xy, _ = R._tile_far(model, frC, b_xy)
        # dedup CLAHE-tile vs raw-tile
        if len(tc_xy) and len(t_xy):
            tf = foots(t_xy); cf = foots(tc_xy)
            keep = [i for i in range(len(tc_xy))
                    if np.linalg.norm(tf - cf[i], axis=1).min() >= DEDUP_CLAHE_PX]
            tc_xy = tc_xy[np.array(keep, int)] if keep else tc_xy[:0]
        t_far = int((t_xy[:, 3] < FAR).sum()) if len(t_xy) else 0
        c_far = int((tc_xy[:, 3] < FAR).sum()) if len(tc_xy) else 0
        tot_tile += t_far; tot_clahe += c_far
        rows.append(dict(frame=fi, tile_new_far=t_far, clahe_new_far=c_far))
        print(f"f{fi}: tile_new_far=+{t_far}  clahe_new_far(beyond tile)=+{c_far}")
        if fi in (67153, 37307, 74615):
            vis = fr.copy()
            for a in b_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 0), 2)
            for a in t_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 255), 2)
            for a in tc_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (255, 0, 255), 2)
            cv2.rectangle(vis, (0, 110), (1919, int(FAR)), (255, 0, 0), 1)
            cv2.putText(vis, f"f{fi} green=base yellow=tile magenta=CLAHE-tile(new)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            p = OUT / f"recall_threeway_overlay_f{fi}.png"
            cv2.imwrite(str(p), vis); overlays[fi] = str(p)
    print(f"\nTOTAL 8-kare: tile_new_far=+{tot_tile}  clahe_new_far(beyond tile)=+{tot_clahe}")
    print("(+tile = insan-GT'de 10 far-kurtarma ile uyumlu; +CLAHE GT-doğrulanmamış)")

    out = dict(
        human_gt=dict(
            far=dict(H=far["H"], base=far["b"], base_pct=round(100 * far["b"] / far["H"], 1),
                     tile=far["b"] + far["t"], tile_pct=round(100 * (far["b"] + far["t"]) / far["H"], 1),
                     uncovered=far["unc"], fp=far["fp"]),
            rest=dict(H=rest["H"], base=rest["b"], base_pct=round(100 * rest["b"] / rest["H"], 1)),
            all=dict(H=allH, base=allB, base_pct=round(100 * allB / allH, 1),
                     tile=allT, tile_pct=round(100 * allT / allH, 1),
                     precision_pct=100.0 if (far["fp"] + rest["fp"]) == 0 else None)),
        clahe_machine_delta=dict(tile_new_far_8f=tot_tile, clahe_new_far_8f=tot_clahe,
                                 per_frame=rows, gt_attributed=False,
                                 note="CLAHE sayıcı turu yok; overlay ile manuel doğrula"),
        overlays=overlays)
    (OUT / "recall_threeway.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT / "recall_threeway.json")


if __name__ == "__main__":
    main()

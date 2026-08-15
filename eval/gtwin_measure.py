"""GT'ye karsi tracker/stitch OLCUMU — gercek IDF1 (proxy degil).

Girdi:
  scratchpad/gtwin_tracklets.parquet + gtwin_emb.npy
  scratchpad/gtwin_groups.json          (tracklet -> auto-grup)
  scratchpad/gtwin_gt.json              (adjudication sonucu: identities/purity/disputed)
Olculen prediktorler:
  tracklets   — konservatif tracklet'ler (parcalanma tabani)
  autogroup   — 0.10-esik auto-grup (stitch on-hali)
  botsort     — BoTSORT (Apache trackers), ayni det evreni
  stitch:*    — global_stitch sweep (app_gate x maxgap)
Cikti: scratchpad/gtwin_scores.json + stdout tablo
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from eval.identity_eval import evaluate            # noqa: E402
from detect.global_stitch import stitch_fast as stitch  # noqa: E402 (fast, partition-esdeger)


def build_gt(df: pd.DataFrame) -> np.ndarray:
    """gtwin_gt.json (gtwin_finalize ciktisi): tid2ident dogrudan."""
    gt = json.load(open('scratchpad/gtwin_gt.json'))
    t2i = {int(k): v for k, v in gt['tid2ident'].items()}
    disputed = set(gt.get('disputed_idents', []))
    ident_ids = {n: k for k, n in enumerate(sorted(gt['identities'].keys()))}
    out = np.full(len(df), -1, dtype=int)
    for i, t in enumerate(df.tracklet_id.values):
        n = t2i.get(int(t))
        if n is None or n in disputed:
            continue
        out[i] = ident_ids[n]
    return out


def pred_botsort(df: pd.DataFrame) -> np.ndarray:
    import supervision as sv
    import trackers
    bot = trackers.BoTSORTTracker(
        frame_rate=12.5, enable_cmc=False, lost_track_buffer=25,
        minimum_consecutive_frames=2, track_activation_threshold=0.30,
        high_conf_det_threshold=0.45, minimum_iou_threshold_first_assoc=0.15)
    pred = np.full(len(df), -1, dtype=int)
    for t in sorted(df.t.unique()):
        m = np.where(df.t.values == t)[0]
        xy = df[['x0', 'y0', 'x1', 'y1']].values[m].astype(float)
        cf = df.conf.values[m].astype(float)
        dets = sv.Detections(xyxy=xy, confidence=cf,
                             class_id=np.zeros(len(m), int))
        out = bot.update(dets, None)
        if out.tracker_id is None:
            continue
        # cikti kutularini girdilere esle (xyxy exact -> IoU fallback)
        for k in range(len(out)):
            ob = out.xyxy[k]
            d = np.abs(xy - ob).sum(1)
            j = int(np.argmin(d))
            if d[j] < 2.0:
                pred[m[j]] = int(out.tracker_id[k])
            else:
                ix1 = np.maximum(xy[:, 0], ob[0]); iy1 = np.maximum(xy[:, 1], ob[1])
                ix2 = np.minimum(xy[:, 2], ob[2]); iy2 = np.minimum(xy[:, 3], ob[3])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                ua = ((xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
                      + (ob[2] - ob[0]) * (ob[3] - ob[1]) - inter)
                iou = inter / np.maximum(ua, 1e-9)
                j = int(np.argmax(iou))
                if iou[j] > 0.5:
                    pred[m[j]] = int(out.tracker_id[k])
    return pred


def main():
    df = pd.read_parquet('scratchpad/gtwin_tracklets.parquet').reset_index(drop=True)
    emb = np.load('scratchpad/gtwin_emb.npy')
    gt = build_gt(df)
    n_lab = int((gt >= 0).sum())
    print(f"GT: {len(np.unique(gt[gt >= 0]))} kimlik, {n_lab}/{len(df)} etiketli det")

    preds = {}
    preds['tracklets'] = df.tracklet_id.values
    groups = json.load(open('scratchpad/gtwin_groups.json'))
    t2g = {t: k for k, (g, ts) in enumerate(sorted(groups.items()))
           for t in ts}
    preds['autogroup'] = np.array([t2g.get(int(t), -1)
                                   for t in df.tracklet_id.values])
    import os as _os
    if _os.environ.get('HALISAHA_SKIP_BOTSORT') != '1':
        try:
            preds['botsort'] = pred_botsort(df)
        except Exception as e:  # noqa: BLE001
            print(f"[uyari] botsort kosamadi: {e}")

    for gate in (0.13, 0.16, 0.20):
        for mg in (8.0, 12.0, 20.0):
            ids = stitch(df, emb, app_gate=gate, maxgap=mg, vmax=6.5)
            preds[f'stitch:g{gate}-mg{int(mg)}'] = ids

    scores = {}
    for name, p in preds.items():
        s = evaluate(gt, p)
        scores[name] = s
        print(f"{name:22s} IDF1={s['IDF1']:.3f} IDP={s['IDP']:.3f} "
              f"IDR={s['IDR']:.3f} ids={s['n_pred_ids']:4d} "
              f"pur={s['purity_mean']} frank={s['n_frankenstein_tracks']} "
              f"frag={s['frag_per_gt_mean']}")
    json.dump(scores, open('scratchpad/gtwin_scores.json', 'w'), indent=1)
    print('-> scratchpad/gtwin_scores.json')


if __name__ == '__main__':
    main()

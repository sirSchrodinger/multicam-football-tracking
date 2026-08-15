"""Kimlik-GT'ye karsi ILISKILENDIRME metrikleri (ortak detection-seti uzerinde).

Detection-seviyesi GT kimlik etiketi (gt_id) ile tracker ciktisi (pred_id) AYNI
detection satirlarinda karsilastirilir. Bu DetA'yi OLCMEZ (recall ayri olculdu,
recall_val) — sadece iliskilendirme (AssA/IDF1 tarafi). HOTA = DetA x AssA
ayristirmasinin AssA yarisi.

IDF1 burada klasik tanimla hesaplanir: gt_id x pred_id eslesme matrisinde
Hungarian ile IDTP maksimize edilir; IDF1 = 2*IDTP / (n_gt + n_pred).
gt_id < 0 satirlar (spurious/bilinmiyor) GT evreninden haric tutulur ama
pred paydasina girer (spurious'a kimlik uyduran cezalanir: excl_spurious=False)
ya da tamamen dislanir (excl_spurious=True, varsayilan — detection FP'si
association metrigini kirletmesin).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def idf1(gt_id: np.ndarray, pred_id: np.ndarray, excl_spurious: bool = True) -> dict:
    gt_id = np.asarray(gt_id)
    pred_id = np.asarray(pred_id)
    assert gt_id.shape == pred_id.shape
    if excl_spurious:
        m = gt_id >= 0
        gt_id, pred_id = gt_id[m], pred_id[m]
    gts = np.unique(gt_id[gt_id >= 0])
    prs = np.unique(pred_id[pred_id >= 0])
    gi = {g: k for k, g in enumerate(gts)}
    pi = {p: k for k, p in enumerate(prs)}
    M = np.zeros((len(gts), len(prs)))
    for g, p in zip(gt_id, pred_id):
        if g >= 0 and p >= 0:
            M[gi[g], pi[p]] += 1
    ri, ci = linear_sum_assignment(-M)
    idtp = float(M[ri, ci].sum())
    n_gt = float((gt_id >= 0).sum())
    n_pred = float((pred_id >= 0).sum())
    idp = idtp / n_pred if n_pred else 0.0
    idr = idtp / n_gt if n_gt else 0.0
    f1 = 2 * idtp / (n_gt + n_pred) if (n_gt + n_pred) else 0.0
    return {'IDF1': round(f1, 4), 'IDP': round(idp, 4), 'IDR': round(idr, 4),
            'IDTP': int(idtp), 'n_gt_det': int(n_gt), 'n_pred_det': int(n_pred),
            'n_gt_ids': len(gts), 'n_pred_ids': len(prs)}


def purity_frag(gt_id: np.ndarray, pred_id: np.ndarray) -> dict:
    """Track-saflik + parcalanma ozeti (Frankenstein teshisi)."""
    gt_id = np.asarray(gt_id)
    pred_id = np.asarray(pred_id)
    m = (gt_id >= 0) & (pred_id >= 0)
    gt_id, pred_id = gt_id[m], pred_id[m]
    pur = []
    frank = 0
    for p in np.unique(pred_id):
        gg = gt_id[pred_id == p]
        vals, cnt = np.unique(gg, return_counts=True)
        pur.append(cnt.max() / cnt.sum())
        if len(vals) > 1 and (cnt.sum() - cnt.max()) / cnt.sum() > 0.2:
            frank += 1
    frags = []
    for g in np.unique(gt_id):
        frags.append(len(np.unique(pred_id[gt_id == g])))
    return {'purity_mean': round(float(np.mean(pur)), 4) if pur else None,
            'purity_min': round(float(np.min(pur)), 4) if pur else None,
            'n_frankenstein_tracks': int(frank),
            'frag_per_gt_mean': round(float(np.mean(frags)), 2) if frags else None,
            'frag_per_gt_max': int(np.max(frags)) if frags else None}


def evaluate(gt_id, pred_id, excl_spurious: bool = True) -> dict:
    out = idf1(gt_id, pred_id, excl_spurious=excl_spurious)
    out.update(purity_frag(gt_id, pred_id))
    return out


if __name__ == '__main__':
    # kucuk self-test: mukemmel eslesme + bilinen bozulmalar
    g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    p_perfect = np.array([5, 5, 5, 7, 7, 7, 9, 9, 9])
    r = evaluate(g, p_perfect)
    assert r['IDF1'] == 1.0, r
    p_split = np.array([5, 5, 6, 7, 7, 7, 9, 9, 9])   # 1 bolunme
    r2 = evaluate(g, p_split)
    assert 0.8 < r2['IDF1'] < 1.0, r2
    p_frank = np.array([5, 5, 5, 5, 5, 5, 9, 9, 9])   # 2 kimlik tek track
    r3 = evaluate(g, p_frank)
    assert r3['IDF1'] < 0.9 and r3['n_frankenstein_tracks'] == 1, r3
    print('selftest OK', r, r2, r3)

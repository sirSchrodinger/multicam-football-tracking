"""recovery_search.py — eğitimsiz, CPU-yerel recall-recovery config araması (Track A).

Öneri 1+2'nin çekirdeği: RunPod beklemeden, BUGÜN, frozen insan-GT'ye karşı gerçek
recall deltası üret. RF-DETR'i fine-tune ETMEZ; sadece FarBandRecovery POST-PROC
config'ini (önce recover_far bayrağı) dev/test ayrımıyla optimize eder.

DÜRÜSTLÜK MİMARİSİ (iki ayrı GT yeteneği):
  count-GT yolu (BUGÜN GATE'LENİR):
    recall_score.reconcile -> her (kare, bölge) için 3-kör-sayıcı medyan
    {n_humans, n_base_covered, n_tile_covered, n_false_pos}. recover_far bayrağının
    etkisi DOĞRUDAN okunur: OFF recall = base_covered/H, ON recall = (base+tile)/H.
    Hiç GPU yok, hiç kutu-eşleme yok, hiç uydurma yok. far +%10.9 ZATEN ölçülü.
  box-GT yolu (KİLİTLİ, nokta-konumlu GT bekler):
    tile_thresh/up/clahe gibi İNCE knob'lar kutu-konumlu GT ister (detcache.apply
    kutuyu üretir ama sayı-GT bunları AYIRT EDEMEZ). Nokta-GT gelene dek bu knob'lar
    skor üretmez -> ledger'a "needs_point_gt" yazılır, ASLA sahte delta.

dev/test ayrımı = anti-overfit (N=8 küçük): config DEV(5)'te seçilir, TEST(3)'te
gate edilir. Ek olarak k-fold çapraz-delta + gt_n her satıra yazılır (güven kalibre).
CPU-only; eval/recall_score.reconcile sarmalı.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GT_LABELS = ROOT / "recall_val/gt_labels.json"
FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]


# --------------------------------------------------------------------------- #
# count-GT'den per-kare kapsama (recover_far bayrağı buradan gate'lenir)
# --------------------------------------------------------------------------- #
def per_frame_coverage(labels) -> dict:
    """recall_score.reconcile -> {frame: {far_H,far_base,far_tile,far_fp,rest_H,...}}."""
    from eval.recall_score import reconcile
    if isinstance(labels, dict) and "labels" in labels:
        labels = labels["labels"]
    rec = reconcile(labels)
    out = {}
    for fi, r in rec.items():
        far, rest = r.get("far", {}), r.get("rest", {})
        out[int(fi)] = {
            "far_H": int(far.get("n_humans") or 0),
            "far_base": int(far.get("n_base_covered") or 0),
            "far_tile": int(far.get("n_tile_covered") or 0),
            "far_fp": int(far.get("n_false_pos") or 0),
            "rest_H": int(rest.get("n_humans") or 0),
            "rest_base": int(rest.get("n_base_covered") or 0),
            "rest_fp": int(rest.get("n_false_pos") or 0),
        }
    return out


def _recall_on(cov: dict, frames, recover_far: bool) -> dict:
    """Bir kare alt-kümesinde cfg.recover_far için recall metrikleri (count-GT)."""
    far_H = sum(cov[f]["far_H"] for f in frames)
    rest_H = sum(cov[f]["rest_H"] for f in frames)
    far_cov = sum(cov[f]["far_base"] + (cov[f]["far_tile"] if recover_far else 0)
                  for f in frames)
    rest_cov = sum(cov[f]["rest_base"] for f in frames)
    fp = sum(cov[f]["far_fp"] + cov[f]["rest_fp"] for f in frames)
    tot_H = far_H + rest_H
    tot_cov = far_cov + rest_cov

    def _r(n, d):
        return round(n / d, 4) if d else None
    return {
        "far_recall": _r(far_cov, far_H),
        "rest_recall": _r(rest_cov, rest_H),
        "all_recall": _r(tot_cov, tot_H),
        "fp": int(fp),
        "precision": round(tot_cov / (tot_cov + fp), 4) if (tot_cov + fp) else None,
        "gt_n": int(tot_H), "far_gt_n": int(far_H), "n_frames": len(frames),
    }


# --------------------------------------------------------------------------- #
# dev/test ayrımı (deterministik) + k-fold
# --------------------------------------------------------------------------- #
def split_frames(labels=None, n_dev: int = 5, seed: int = 0) -> dict:
    """Frozen kareleri dev(n_dev)/test(kalan) olarak deterministik böl (seed)."""
    frames = sorted(per_frame_coverage(labels).keys()) if labels is not None else list(FROZEN)
    rng = np.random.default_rng(seed)
    perm = [int(x) for x in rng.permutation(frames)]
    return {"dev": sorted(perm[:n_dev]), "test": sorted(perm[n_dev:]), "seed": seed}


def _kfold_delta(cov: dict, k: int = 4, seed: int = 0) -> dict:
    """recover_far OFF->ON far-recall deltasının k-fold (test-fold) çapraz dağılımı.

    N=8 küçük; tek dev/test bölmesi şanslı olabilir. Her fold'un TEST diliminde
    deltayı ölç -> ortalama + min (en kötü fold). Gürültü üstünde mi gör.
    """
    frames = sorted(cov.keys())
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(frames))
    folds = [perm[i::k] for i in range(k)]
    deltas = []
    for fo in folds:
        if not fo:
            continue
        off = _recall_on(cov, fo, False)["far_recall"]
        on = _recall_on(cov, fo, True)["far_recall"]
        if off is not None and on is not None:
            deltas.append(round(on - off, 4))
    if not deltas:
        return {"folds": 0}
    return {"folds": len(deltas), "far_delta_per_fold": deltas,
            "far_delta_mean": round(float(np.mean(deltas)), 4),
            "far_delta_min": round(float(np.min(deltas)), 4),
            "all_positive": bool(np.min(deltas) > 0)}


# --------------------------------------------------------------------------- #
# default grid  (count-GT: recover_far OFF vs ON@üretim-knob)
# --------------------------------------------------------------------------- #
def default_grid() -> list[dict]:
    """Bugün count-GT ile gate'lenebilir grid: recover_far bayrağı (üretim knob'ları).

    İnce knob varyantları (tile_thresh/up/clahe) box-GT ister; default_grid_box()
    bunları detcache.apply ile üretir ama gate KİLİTLİ (needs_point_gt).
    """
    prod = {"tile_thresh": 0.40, "up": 2.0, "dedup_px": 28.0, "clahe": False,
            "band": "fixed_110_365"}
    return [
        {"recover_far": False, **prod, "_scoreable": "count_gt"},
        {"recover_far": True, **prod, "_scoreable": "count_gt"},
    ]


def default_grid_box() -> list[dict]:
    """İnce-knob grid (box-GT gerektirir; bugün skorlanMAZ -> needs_point_gt)."""
    grid = []
    for tt in (0.35, 0.40, 0.45):
        for up in (2.0, 2.5):
            for cl in (False, True):
                grid.append({"recover_far": True, "tile_thresh": tt, "up": up,
                             "clahe": cl, "dedup_px": 28.0, "band": "density",
                             "_scoreable": "box_gt"})
    return grid


# --------------------------------------------------------------------------- #
# search  (eğitimsiz config araması; DEV'de seç, TEST'te gate)
# --------------------------------------------------------------------------- #
def search(gt_labels=None, grid: list[dict] | None = None,
           split: dict | None = None, cache=None, venue: str = "cankaya_cam2") -> dict:
    """recover_far config araması (count-GT; eğitimsiz).

    gt_labels : gt_labels.json içeriği (liste) ya da None (dosyadan oku).
    grid      : config listesi (default = default_grid()).
    split     : {"dev":[...],"test":[...]} (default = split_frames(seed=0)).
    cache     : (ops) detcache box cache; box-GT yolu için (bugün skorlanmaz).
    Döndürür best_cfg + dev/heldout metrikleri + off referansı + k-fold + uyarılar.
    """
    import json
    if gt_labels is None:
        gt_labels = json.loads(GT_LABELS.read_text())
    cov = per_frame_coverage(gt_labels)
    grid = grid or default_grid()
    split = split or split_frames(gt_labels)
    dev, test = split["dev"], split["test"]

    candidates, skipped = [], []
    for cfg in grid:
        if cfg.get("_scoreable", "count_gt") != "count_gt":
            skipped.append({"cfg": cfg, "reason": "needs_point_gt"})
            continue
        rf = bool(cfg.get("recover_far", True))
        candidates.append({
            "cfg": cfg,
            "dev": _recall_on(cov, dev, rf),
            "test": _recall_on(cov, test, rf),
        })

    # OFF referansı (count-GT yolu daima üretir)
    off_dev = _recall_on(cov, dev, False)
    off_test = _recall_on(cov, test, False)

    # DEV'de seç: far_recall maks; eşitlikte rest_recall yüksek, fp düşük
    def _key(c):
        d = c["dev"]
        return (d["far_recall"] or -1, d["rest_recall"] or -1, -(d["fp"] or 0))
    best = max(candidates, key=_key) if candidates else None

    return {
        "venue": venue,
        "split": {"dev": dev, "test": test, "seed": split.get("seed")},
        "gt_n": int(sum(c["far_H"] + c["rest_H"] for c in cov.values())),
        "far_gt_n": int(sum(c["far_H"] for c in cov.values())),
        "best_cfg": best["cfg"] if best else None,
        "dev_far_recall": best["dev"]["far_recall"] if best else None,
        "heldout_far_recall": best["test"]["far_recall"] if best else None,
        "heldout_rest_recall": best["test"]["rest_recall"] if best else None,
        "heldout_all_recall": best["test"]["all_recall"] if best else None,
        "heldout_precision": best["test"]["precision"] if best else None,
        "fp_delta": (best["test"]["fp"] - off_test["fp"]) if best else None,
        "best_test_metrics": best["test"] if best else None,
        "off_dev_metrics": off_dev,
        "off_test_metrics": off_test,
        "candidates": candidates,
        "kfold": _kfold_delta(cov, k=4, seed=split.get("seed", 0)),
        "skipped_box_gt": skipped,
        "full_set": {
            "off": _recall_on(cov, sorted(cov.keys()), False),
            "best_on": _recall_on(cov, sorted(cov.keys()), True) if best
                       and best["cfg"].get("recover_far") else None,
        },
    }

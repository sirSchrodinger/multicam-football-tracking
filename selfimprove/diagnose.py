"""diagnose.py — ScoreCard -> KÖK-NEDEN yönlendirme (kapalı-döngü kontrolör).

İki kritik tasarım kararı (naif "en düşük ekseni kovala" YANLIŞ olurdu):

1) SEMPTOM != KÖK-NEDEN. match_quality 'motion' eksenini en düşük gösterebilir
   ama motion (saha-dışı sızıntı + ışınlanma) downstream'dir:
     off_field_pct yüksek -> KÖK calibration
     teleport/impossible_step yüksek -> KÖK identity (stitch / re-assoc)
   Bu yüzden symptom->root haritası ile semptom değil KÖKE yönlendiririz.

2) RECALL-ÖNCELİKLİ amaç. Birincil hedef (Alperen): far/karanlık band'de KAÇAN
   oyuncu = recall. recall_heldout.base < RECALL_TARGET olduğu sürece, başka bir
   proxy ekseni sayısal olarak daha düşük olsa bile, BİRİNCİL aksiyon recall
   düzeltici (mine far-band + detector fine-tune). recall hedefe ulaşınca
   proxy-zayıf köke düşülür.

Çıktı Action = {
  "objective": "recall" | "identity" | "calibration",
  "root_cause": str,
  "symptom_axis": str,
  "corrective_module": str,    # mine.far_band_corpus | stitch_tune | recalibrate_flag
  "params": {...},
  "rationale": str,
  "needs_gpu_train": bool,     # True => RunPod improve adımı gerekir
}
"""
from __future__ import annotations

RECALL_TARGET = 0.90          # held-out base recall hedefi (far dahil)
FAR_RECALL_TARGET = 0.85      # far-band ayrı izlenir (en zor)

# proxy eksen -> kök-neden(ler). Düşük skorun GERÇEK kaynağı.
_SYMPTOM_TO_ROOT = {
    "detection": "recall_pipeline",       # düşük tespit = recall (far/dark)
    "identity":  "identity_stitch",       # fragman/teleport = stitch/re-assoc
    "calibration": "calibration",         # reproj px = homografi
    "motion":    None,                    # TÜRETİLMİŞ; off_field->calib, teleport->id
}


def _motion_root(axes: dict) -> str:
    """motion semptomunu kökene ayır: off-field baskınsa calib, teleport baskınsa id."""
    m = axes.get("motion", {})
    off = float(m.get("off_field_pct", 0.0))
    imp = float(m.get("impossible_step_pct", 0.0))
    # off-field saha-dışı sızıntı -> kalibrasyon/foot-nokta; ışınlanma -> kimlik.
    if off >= imp:
        return "calibration"
    return "identity_stitch"


def _route_recall(base: float, far: float, gap_far: bool, ctx: dict) -> dict:
    """RECALL düzelticisini 3'e ayır (loop state'i = context).

      RAY A  recovery_search : recover_far henüz promote değil + frozen-GT var
                               -> CPU-yerel, EĞİTİMSİZ, RunPod'suz (BUGÜN).
      RAY B  mine+improve    : Track A tavana vurdu + LOSO hazır (>=2 saha blind-GT)
                               -> korpus + RunPod fine-tune (GATED).
      gt_request             : LOSO için 2. saha blind-GT yok -> önce etiket topla
                               (label_queue); Track B kilidini açan TEK adım.
    """
    recover_promoted = bool(ctx.get("recover_far_promoted", False))
    frozen_gt = bool(ctx.get("frozen_gt_available", False))
    n_gt_venues = int(ctx.get("n_gt_venues", 0))
    common = {
        "objective": "recall", "root_cause": "recall_pipeline", "symptom_axis": "detection",
        "params": {"emphasize_far": gap_far, "current_base_recall": base,
                   "current_far_recall": far, "target_base": RECALL_TARGET,
                   "target_far": FAR_RECALL_TARGET},
    }
    if not recover_promoted and frozen_gt:
        return {**common, "corrective_module": "recovery_search", "ray": "A",
                "needs_gpu_train": False,
                "rationale": (f"held-out far {far:.0%} < {FAR_RECALL_TARGET:.0%}: recover_far "
                              "henüz promote DEĞİL -> Track A eğitimsiz config araması "
                              "(frozen-GT dev/test, RunPod'suz, bugün)")}
    if n_gt_venues >= 2:
        return {**common, "ray": "B", "needs_gpu_train": True,
                "corrective_module": "mine.far_band_corpus" if gap_far else "mine.detection_corpus",
                "rationale": (f"Track A tüketildi (recover_far promote); LOSO hazır "
                              f"({n_gt_venues} saha) -> mine far-band + RunPod fine-tune")}
    return {**common, "corrective_module": "gt_request", "ray": "gt",
            "needs_gpu_train": False,
            "rationale": (f"Track A tüketildi ama LOSO için {n_gt_venues}<2 saha blind-GT; "
                          "2. saha etiket kuyruğu (mine.label_queue) -> Track B gate açılır")}


def diagnose(scorecard: dict, context: dict | None = None) -> dict:
    """ScoreCard -> kök-neden aksiyonu.

    context (ops, loop'tan): {recover_far_promoted, frozen_gt_available, n_gt_venues}.
    Verilirse recall düzelticisi 3-yol (A/B/gt_request) olarak ayrışır. Verilmezse
    (geriye-dönük uyum: tek-arg çağrılar) recall -> Track B çerçevesi döner.
    """
    axes = scorecard.get("proxy_axes", {})
    rh = scorecard.get("recall_heldout")

    # ---- 1) RECALL-ÖNCELİK: held-out recall hedefin altındaysa her şeyden önce gelir
    if rh is not None:
        base = float(rh.get("base", 0.0))
        far = float(rh.get("far_base", 0.0))
        if base < RECALL_TARGET or far < FAR_RECALL_TARGET:
            gap_far = far < FAR_RECALL_TARGET
            if context is not None:
                return _route_recall(base, far, gap_far, context)
            return {
                "objective": "recall",
                "root_cause": "recall_pipeline",
                "symptom_axis": "detection",
                "corrective_module": "mine.far_band_corpus" if gap_far else "mine.detection_corpus",
                "params": {
                    "emphasize_far": gap_far,
                    "current_base_recall": base,
                    "current_far_recall": far,
                    "target_base": RECALL_TARGET,
                    "target_far": FAR_RECALL_TARGET,
                },
                "rationale": (f"held-out base recall {base:.0%} < hedef {RECALL_TARGET:.0%}"
                              + (f"; far {far:.0%} < {FAR_RECALL_TARGET:.0%} (öncelik far-band)"
                                 if gap_far else "")),
                "needs_gpu_train": True,
            }

    # ---- 2) recall yoksa ya da hedefteyse: proxy-zayıf KÖKE düş
    scored = {k: v.get("score") for k, v in axes.items() if v.get("score") is not None}
    if not scored:
        return {"objective": "none", "root_cause": "no_signal",
                "symptom_axis": None, "corrective_module": None, "params": {},
                "rationale": "ölçülebilir eksen yok", "needs_gpu_train": False}

    symptom = min(scored, key=scored.get)
    if symptom == "motion":
        root = _motion_root(axes)
    else:
        root = _SYMPTOM_TO_ROOT.get(symptom, symptom)

    route = {
        "recall_pipeline": ("recall", "mine.detection_corpus", True),
        "identity_stitch": ("identity", "stitch_tune", False),
        "calibration":     ("calibration", "recalibrate_flag", False),
    }.get(root, ("identity", "stitch_tune", False))
    objective, module, needs_gpu = route

    return {
        "objective": objective,
        "root_cause": root,
        "symptom_axis": symptom,
        "corrective_module": module,
        "params": {k: axes[k] for k in (symptom, "motion") if k in axes},
        "rationale": (f"proxy zayıf eksen '{symptom}' ({scored[symptom]:.0f}/100) "
                      f"-> kök-neden '{root}'"
                      + ("; motion semptomu off-field/teleport oranına göre köke ayrıldı"
                         if symptom == "motion" else "")),
        "needs_gpu_train": needs_gpu,
    }

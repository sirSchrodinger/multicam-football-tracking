"""ledger.py — append-only olay defteri + kalıcı kontrol-durumu.

İki dosya (selfimprove/_state/ altında):
  ledger.jsonl  : her cycle bir satır (append-only; SİLİNMEZ -> ilerleme denetlenebilir).
  state.json    : canlı kontrol durumu = {promoted_weights, baseline_metrics,
                  gt_venues, cycle_seq, last_action}.

Append-only olmak DÜRÜSTLÜK aracı: "geliştirdik" demek için ledger'da KEEP kararı +
held-out delta görünmeli; rollback'ler de kalır (sahte ilerleme gizlenemez).
CPU-only, sadece json/std-lib.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "_state"
LEDGER = STATE_DIR / "ledger.jsonl"
STATE = STATE_DIR / "state.json"

_DEFAULT_STATE = {
    "promoted_weights": "models/weights/checkpoint_best_regular.pth",
    "promoted_weights_sha256": None,
    "baseline_metrics": None,        # drift_guard.evaluate_held_out çıktısı (LOSO; Track B)
    "promoted_config": None,         # Track A: FarBandRecovery knob'ları (production okur)
    "gt_venues": ["cankaya_cam2"],   # blind-GT (recall_score) olan saha listesi
    "cycle_seq": 0,
    "last_action": None,
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        s = json.loads(STATE.read_text())
        for k, v in _DEFAULT_STATE.items():
            s.setdefault(k, v)
        return s
    s = dict(_DEFAULT_STATE)
    STATE.write_text(json.dumps(s, indent=2, ensure_ascii=False))
    return s


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def next_cycle_id() -> int:
    s = load_state()
    return int(s.get("cycle_seq", 0)) + 1


def append(entry: dict) -> dict:
    """Bir cycle kaydını ledger'a ekle + state.cycle_seq ilerlet. Eklenen kaydı döndür."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    s = load_state()
    cid = int(s.get("cycle_seq", 0)) + 1
    rec = {"cycle": cid, "ts": _now(), **entry}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    s["cycle_seq"] = cid
    s["last_action"] = entry.get("action", {}).get("corrective_module") if entry.get("action") else None
    save_state(s)
    return rec


def history(n: int | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return rows[-n:] if n else rows


def sha256_file(path: str | os.PathLike) -> str | None:
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def promote(weights_path: str, baseline_metrics: dict, *, reason: str = "") -> dict:
    """Yeni ağırlığı CANLI yap: state.promoted_weights + baseline_metrics güncelle.

    SADECE gate.gate_or_rollback KEEP dediğinde çağrılır. baseline_metrics, bir
    sonraki turun karşılaştıracağı held-out referansı olur (drift takip noktası).
    """
    s = load_state()
    s["promoted_weights"] = str(weights_path)
    s["promoted_weights_sha256"] = sha256_file(weights_path)
    s["baseline_metrics"] = baseline_metrics
    save_state(s)
    return s


def promote_config(config: dict, evidence: dict, *, venue: str = "",
                   reason: str = "") -> dict:
    """Track A post-proc CONFIG'ini CANLI yap (reversible).

    SADECE gate.config_gate KEEP dediğinde çağrılır. Önceki config 'previous'a yazılır
    (tek satırla geri-al). Production (export_tracks.py --recover-far, harvest_probe)
    state.promoted_config'i okur. evidence = recovery_search held-out kanıtı (gt_n dahil).
    """
    s = load_state()
    prev = s.get("promoted_config")
    s["promoted_config"] = {
        **{k: v for k, v in config.items() if not k.startswith("_")},
        "reversible": True,
        "venue": venue,
        "gt_n": evidence.get("gt_n"),
        "heldout_far_recall": evidence.get("heldout_far_recall"),
        "promoted_ts": _now(),
        "reason": reason,
        "previous": prev,
    }
    save_state(s)
    return s


def last_config_baseline() -> dict | None:
    """Hâlihazırda promote edilmiş config (Track A gate'in karşılaştıracağı taban).

    None -> hiç config promote edilmemiş; bu durumda taban = recover_far OFF
    (recovery_search off_test_metrics zaten üretir).
    """
    return load_state().get("promoted_config")


def progress_summary() -> dict:
    """Ledger'dan ilerleme özeti: kaç cycle, kaç KEEP/rollback, recall trend, config."""
    rows = history()
    keeps = [r for r in rows if r.get("gate", {}).get("decision") == "keep"]
    rolls = [r for r in rows if r.get("gate", {}).get("decision") == "rollback"]
    recall_trend = [
        (r["cycle"], r.get("scorecard", {}).get("recall_heldout", {}).get("base"))
        for r in rows
        if r.get("scorecard", {}).get("recall_heldout", {}).get("base") is not None
    ]
    config_promotions = [
        {"cycle": r["cycle"], "far_delta": r.get("gate", {}).get("deltas", {}).get("far_recall"),
         "cfg": r.get("recovery", {}).get("best_cfg"), "gt_n": r.get("recovery", {}).get("gt_n"),
         "venue": r.get("recovery", {}).get("venue")}
        for r in rows if r.get("promoted_config")
    ]
    s = load_state()
    return {
        "n_cycles": len(rows),
        "n_keep": len(keeps),
        "n_rollback": len(rolls),
        "recall_base_trend": recall_trend,
        "current_promoted_weights": s.get("promoted_weights"),
        "current_promoted_config": s.get("promoted_config"),
        "config_promotions": config_promotions,
    }

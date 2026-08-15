#!/usr/bin/env python3
"""Tek-komut halisaha pipeline: (video|tracks-parquet) + calib -> per-OYUNCU rapor + figur.

Adimlar:
  [export] girdi video ise: export_tracks --calib -> tracks parquet  (GPU; QA-kapisi)
  [stitch] detect.track_stitch.stitch -> *_stitched.parquet + *_player.parquet (per-oyuncu)
  [report] stats.topdown_viz.render_report_figure(player-keyed) -> report_figure.png + report_topdown.json

Calib QA-kapisindan GECMELI (bozuk 65px reddedilir; uydurma yok). Cikti stats_out/<cam>/.
Idempotent. Metre kilidi calib'te scale_anchor varsa acilir; yoksa ciktilar relative_m.

Kullanim:
  venv/bin/python scripts/run_pipeline.py <clip.mp4 | tracks.parquet> <calib.json> \
      [--out stats_out/<cam>] [--video appearance_icin.mp4]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="clip.mp4 (tam) VEYA mevcut tracks.parquet")
    ap.add_argument("calib", help="calib/<cam>.json (QA-kapisindan gecmeli)")
    ap.add_argument("--out", default=None, help="cikti dizini (default stats_out/<cam>)")
    ap.add_argument("--video", default=None,
                    help="appearance icin video (parquet girdisinde opsiyonel)")
    ap.add_argument("--weights", default=None, help="export icin RF-DETR agirligi")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="tam-mac konsolidasyonunu (dedup+uzun-boslik merge) KAPAT, "
                         "ham track_stitch kullan (kisa klip/geriye-uyum icin)")
    ap.add_argument("--expected-players", type=int, default=None,
                    help="cift roster ipucu (12/14/16); roster_estimate'e gecer, ASLA zorlamaz")
    ap.add_argument("--scale-v3", action="store_true",
                    help="QA-gecmis calib'i standart-boyuta snap'le (v3, m approx +-10%%) ve "
                         "raporu metre-yaklasik birimle uret; default kapali -> relative_m")
    ap.add_argument("--scale-fmt", default="7v7", help="--scale-v3 icin format (7v7/8v8/...)")
    ap.add_argument("--scale-height", action="store_true",
                    help="ONERILEN olcek: oyuncu-boyu (~1.75m) tek-goruntu metroloji -> mutlak "
                         "metre (sahadan/kaleden BAGIMSIZ, m approx +-~13%%). --scale-v3'u ezer. "
                         "track parquet gerektirir (export sonrasi).")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    inp = Path(a.input)
    if not inp.exists():
        sys.exit(f"girdi yok: {inp}")
    cam = inp.stem.replace("tracks_", "").replace("_stitched", "").replace("_player", "")
    out = Path(a.out) if a.out else (ROOT / "stats_out" / cam)
    out.mkdir(parents=True, exist_ok=True)

    # 0) OLCEK opsiyonu: --scale-v3 -> QA-gecmis calib'i izotropik standart-boyuta snap'le.
    #    Fail-closed (bozuk calib reddedilir). v3 scale_anchor tasir -> rapor 'm (approx +-10%)'.
    #    Snap edilmezse calib oldugu gibi gecer; scale_anchor null ise rapor relative_m kalir.
    calib_path = a.calib
    if a.scale_v3:
        from pitch.scale_snap import make_v3
        v3_path = str(out / (Path(a.calib).stem + "_v3.json"))
        print(f"[0 scale] standart-boyut snap ({a.scale_fmt}) -> {v3_path} ...", flush=True)
        smeta = make_v3(a.calib, v3_path, fmt=a.scale_fmt)
        print(f"    asserted {smeta.get('asserted_dims_LW')} m ({smeta.get('catalog')}) | "
              f"scale_factor {smeta.get('scale_factor')} | in_band={smeta.get('in_band')} "
              f"| gate_ok={smeta.get('gate_ok')} (approximate +-10%)", flush=True)
        calib_path = v3_path

    # 1) EXPORT (yalniz video girdisinde) ---------------------------------------
    video_path = a.video
    if inp.suffix.lower() in VIDEO_EXT:
        import export_tracks as ET
        tracks_path = str(out / f"tracks_{inp.stem}.parquet")
        print(f"[1/3 export] {inp.name} --calib {a.calib} (GPU) ...", flush=True)
        kw = dict(out_path=tracks_path, calib_path=a.calib)
        if a.weights:
            kw["weights_path"] = a.weights
        tracks_path = ET.run_tracking_export(str(inp), **kw)
        video_path = video_path or str(inp)
    else:
        tracks_path = str(inp)
        print(f"[1/3 export] atlandi (parquet girdisi: {inp.name})", flush=True)

    # 1a) OLCEK (ONERILEN): oyuncu-boyu metroloji -> mutlak metre, sahadan/kaleden BAGIMSIZ.
    #     track parquet gerektirir (boylar oradan); export'tan SONRA, presplit'ten ONCE.
    #     --scale-v3 (katalog) ile uretilen calib'i de ezer (oyuncu-boyu daha dogru).
    if a.scale_height:
        from pitch.height_scale import make_v_height
        vh_path = str(out / (Path(a.calib).stem + "_vheight.json"))
        print(f"[1a scale] oyuncu-boyu metroloji -> {vh_path} ...", flush=True)
        hm = make_v_height(calib_path, vh_path, tracks_path)
        if hm["ok"]:
            print(f"    SAHA {hm['implied_field_m']} m | kamera {hm['camera_height_m']}m | "
                  f"boy medyan {hm['median_height_relm']} CV {hm['height_cv']} | "
                  f"c={hm['scale_factor']} band +-{hm['band_pct']}% (m approx) | n={hm['n_detections']}",
                  flush=True)
            calib_path = vh_path
        else:
            print(f"    UYARI oyuncu-boyu olcek REDDEDILDI ({hm['reasons']}) -> "
                  f"relative_m kalir", flush=True)

    # 1b) ROBUST PRE-STITCH (diagnose sidecar + crossing-coincident teleport bolme)
    import detect.track_robust as TR
    import stats_report
    print("[1b robust] diagnose + presplit ...", flush=True)
    diag = TR.diagnose(tracks_path, calib_path, out_dir=str(out))
    tracks_path = TR.presplit(tracks_path, calib_path)
    print(f"    teleport piksel-aday {diag['mode1a_intra_swaps']['n']} | "
          f"crossing-coincident bolme {diag['mode1a_intra_swaps']['n_split_candidates']} | "
          f"presplit -> {tracks_path}", flush=True)

    # 2) STITCH (146 fragment -> ~14-18 oyuncu) ---------------------------------
    print("[2/3 stitch] tracklet -> oyuncu ...", flush=True)
    if a.no_consolidate:
        import detect.track_stitch as TS
        res = TS.stitch(tracks_path, calib_path, str(out), video_path=video_path,
                        expected_players=a.expected_players)
    else:
        # VARSAYILAN: tam-mac konsolidasyonu (inframe-dedup + uzun-bosluk merge).
        # 12-dk macta 1034 parca -> ~21 kume/14 core; over-merge korumasi korunur.
        from detect.consolidate import consolidate
        res = consolidate(tracks_path, calib_path, str(out), video_path=video_path,
                          expected_players=a.expected_players)
        ds = res.get("dedup_stats", {})
        print(f"    [consolidate] dedup {ds.get('n_dropped')} satir "
              f"({ds.get('dropped_frac',0)*100:.1f}%); floor "
              f"{ds.get('clique_floor_before')}->{ds.get('clique_floor_after')}",
              flush=True)
    rep = res["report"]
    pk = res["player_keyed_path"]
    pk_unbridged = pk  # takim ayrimi icin (bridge'siz; NaN-interp satir yok)

    # 2b) ROBUST POST-STITCH (bridge_gaps + audit_seams + rapor)
    print("[2b robust] bridge_gaps + audit_seams ...", flush=True)
    pk, bmeta = TR.bridge_gaps(pk, calib_path)
    teams = TR.team_label(stats_report.load_tracks(pk), calib_path, video_path)
    audit = TR.audit_seams(res["stitched"], calib_path, video_path)
    jp, mp = TR.robust_report(diag, bmeta, teams, audit, str(out))
    print(f"    bridged {bmeta['bridged_rows']} satir | uzun-bosluk atlanan "
          f"{bmeta['bridged_skipped_long']} | swap-suspect dikis "
          f"{audit['n_ambiguous_seams']}", flush=True)
    print(f"    robust rapor: {jp} ; {mp}")

    # 2c) TAKIM AYRIMI — PRIMARY: jersey-renk 2-means + pas/etkilesim DOGRULAMA
    #     (track_robust.team_label sadece diagnostik; canonical takim ciktisi BU).
    #     Non-destructive: {base}_team.parquet + team JSON; dengesizligi gizlemez.
    print("[2c team] jersey-renk PRIMARY + pas-proxy dogrulama ...", flush=True)
    try:
        from detect.team_split import assign_teams
        # bridge'siz player parquet: interpolasyon NaN'lari renk ornek-cikarimini bozmasin
        tsplit = assign_teams(pk_unbridged, calib_path, str(out), video_path)
        trep = tsplit["report"]
        print(f"    team_sizes {trep.get('team_sizes')} | balance_ok={trep.get('balance_ok')} | "
              f"renk-guven {trep.get('color_separation_confidence')} | "
              f"pas-uyum {trep.get('color_interaction_agreement')} | "
              f"low-conf {len(trep.get('flagged_low_conf', []))}", flush=True)
        print(f"    takim parquet: {tsplit.get('team_parquet_path')}", flush=True)
    except Exception as e:  # video yoksa veya renk sinyali cokerse pipeline durmasin
        print(f"    UYARI takim ayrimi atlandi: {e}", flush=True)

    # 2d) SUREKLI KONUM HARITASI (projenin CEKIRDEGI, Alperen): her core oyuncu HER frame'de
    #     plausible konum (observed/interpolated/predicted) -> recall acigi/occlusion/kadraj-disi
    #     boyunca 14 oyuncu yasar. interpolated/predicted TESPIT DEGIL (conf<1) -> metre HARIC.
    try:
        import detect.player_state as PS
        from pitch.homography import PitchHomography as _PH
        _homo = _PH.load(calib_path)
        _sd = res["stitched"]
        _fps = float(_sd["frame"].max()) / float(_sd["t_sec"].max())
        _core = [int(p) for p, v in rep.get("presence", {}).get("per_player", {}).items()
                 if v.get("role") == "core"]
        state_df, cov = PS.continuous_state(_sd, _homo, _fps, core_ids=_core or None)
        state_path = str(out / "player_state_continuous.parquet")
        state_df.to_parquet(state_path, index=False)
        cs = PS.coverage_summary(cov)
        print(f"[2d harita] surekli konum: {cs['n_players']} core | gozlenen-oran ort "
              f"{cs['mean_observed_frac']} (min {cs['min_observed_frac']}) | dolgu "
              f"{cs['total_interpolated'] + cs['total_predicted']} frame -> {state_path}", flush=True)
    except Exception as e:
        print(f"    UYARI surekli harita atlandi: {e}", flush=True)

    print(f"    {rep['n_tracklets']} tracklet -> {rep['n_clusters']} oyuncu | "
          f"floor {rep['clique_floor']} | ortusme-ihlal {rep['temporal_violations']} | "
          f"top16 %{rep['top16_obs_coverage']*100:.1f} | "
          f"under-merge {rep['n_residual_fragments']} | birim {rep['unit']}", flush=True)
    print(f"    stitched : {res['stitched_path']}")
    print(f"    per-oyuncu: {pk}")

    # 3) REPORT + FIGURE (player-keyed -> per-oyuncu metrik/heatmap) ------------
    from stats.topdown_viz import render_report_figure
    fig = str(out / f"report_figure_{cam}.png")
    print("[3/3 report] top-down figur + rapor ...", flush=True)
    render_report_figure(pk, calib_path, fig)
    print(f"    figur : {fig}")
    print(f"    rapor : {out / 'report_topdown.json'}")
    print(f"\nBITTI -> {out}/")


if __name__ == "__main__":
    main()

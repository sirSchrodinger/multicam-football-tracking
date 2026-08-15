#!/usr/bin/env python3
"""stats/replay2d.py — temiz, kalıcı 2D maç replay (top-down).

scratchpad/integrated_render.py'nin yerini alır. Düzeltilen bug'lar (Alperen 29 Haz):
  1) Ceza sahası + orta yuvarlak gerçeğinden GENİŞ çiziliyordu — kök: (a) FINAL.json
     yanlış-aspect homografi (2.15 vs gerçek ~1.89), (b) marking sabitleri SC ile
     çift-ölçekleniyordu, (c) kalibrasyonda OLMAYAN futbol-boyu ceza sahası uyduruluyordu.
     → v2/vheight homografisi (aspect 1.89, far-reproj 7px) + marking'leri SADECE
       kalibre çizgilerden (solid) + nominal yuvarlak/box (soluk, "tahmini" etiketli).
  2) Uzak defans sebepsiz orta sahaya çekiliyordu — kök: post-process np.interp
     gözlem-boşluklarını DÜZ ÇİZGİYLE köprülüyordu (+ID-swap). → boşluk-kapılı interp:
     yalnız < GAP_MAX_S köprülenir; uzun boşlukta iz KESİLİR (orta sahaya çizgi yok).

Lisans: OpenCV(BSD)+numpy+scipy; MP4 için sistem-ffmpeg üzerinden cv2.VideoWriter.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2

# ---- kalibrasyon (fusion/pitch_map ile uyumlu ama scale_anchor=None'ı da kaldırır) ----
def load_calib(path: str) -> dict:
    F = json.load(open(path))
    sa = F.get("scale_anchor")
    if isinstance(sa, dict):                            # vheight: {scale_factor:...}
        sc = float(sa.get("scale_factor", 1.0)); band = sa.get("label", "rel_m")
    elif isinstance(sa, (list, tuple)):                 # v4 line-calib: ['goal_width', 3.0] -> H metrik, SC=1
        sc = 1.0; band = f"{sa[0]} {sa[1]}m anchor (line-calib)"
    else:                                               # v2: yok -> 1.0
        sc = 1.0; band = "rel_m (kalibre, ölçek doğrulanmadı)"
    tdims = F["template"]["dims_m"]                     # homografinin hedef dikdörtgeni (aspect burada)
    L, W = tdims[0] * sc, tdims[1] * sc                 # metrik saha
    return dict(
        K=np.array(F["K"], float), D=np.array(F["dist"], float),
        Hi2p=np.array(F["H_img2pitch"], float), SC=sc,
        L=float(L), W=float(W), template=F["template"], band=band,
        far_reproj=(F.get("qa") or {}).get("per_zone", {}).get("far"),
    )


def project(cal: dict, foot_xy) -> np.ndarray:
    """(N,2) ayak-pikselleri (distorted) -> (N,2) metre (X=boy, Y=en)."""
    xy = np.asarray(foot_xy, float).reshape(-1, 1, 2)
    u = cv2.undistortPoints(xy, cal["K"], cal["D"], P=cal["K"]).reshape(-1, 2)
    q = cal["Hi2p"] @ np.c_[u, np.ones(len(u))].T
    return np.c_[q[0] / q[2] * cal["SC"], q[1] / q[2] * cal["SC"]]


# ---- post-process: teleport-red + savgol + TAHMİN-li boşluk-tutma + hız-tavanı ----
def postprocess(df, cal, fps, vmax=6.5, gap_max_s=0.5, pred_max_s=2.0, hold_tail_s=1.5,
                savgol_w=15, id_col=None):
    """df: <id>,frame,foot_x,foot_y -> per-frame pid,x,y,status('obs'|'interp'|'pred').

    id_col: kimlik sütunu. None -> OTOMATİK: 'player_id' varsa onu kullan (konsolide
    kimlik; gk_anchor çıktısı), yoksa 'tid' (ham BoT-SORT, FRAGMENTE -> spawn). player_id
    kullanılırsa gürültü (player_id<0) ELENİR.

    Alperen 29 Haz: "aynı oyuncu dediğinde 2sn bekleme payı var, neden yok ediyor adamı;
    özellikle KALECİ kalede oluşup kayboluyor — tahmin mekanizması gelişmeli." Eskiden boşluk
    > 0.5s ise oyuncu SİLİNİYORDU (yeni segment, arada hiç çizilmiyor). Artık 3 katman:
      • boşluk <= gap_max_s (0.5s)        -> 'interp' (iki gözlem arası, gerçek)
      • gap_max_s < boşluk <= pred_max_s  -> 'pred' köprü (fiziksel-olası hızda; SİLME YOK)
      • boşluk > pred_max_s VEYA fiziksel-dışı köprü-hızı -> segment KESİLİR, AMA segment
        sonuna hold_tail_s (1.5s) ileri-TAHMİN kuyruğu eklenir (hız söner -> sabit-tut;
        kaleci ~duruyor -> yerinde kalır). Böylece oyuncu anında kaybolmaz, soluk 'pred'
        olarak ~1.5s asılı kalır, sonraki gözlemde yeniden yakalanır.
    pred konum TESPİT DEĞİL -> status='pred' (render'da içi-boş soluk halka). Sahte
    cross-field iz YOK: köprü yalnız fiziksel-olası hızda (bs<=vmax) kurulur; ID-swap'lı
    uzak sıçrama hâlâ kesilir. [[detect/player_state.continuous_state]] ile aynı felsefe."""
    from scipy.signal import savgol_filter
    if id_col is None:
        id_col = "player_id" if "player_id" in df.columns else "tid"
    df = df.copy()
    if id_col == "player_id":
        df = df[df["player_id"] >= 0]                    # gürültü/eşlemesiz tespitleri at
    P = project(cal, df[["foot_x", "foot_y"]].values)
    df["X"], df["Y"] = P[:, 0], P[:, 1]
    gap_max_f = max(1, int(round(gap_max_s * fps)))
    pred_max_f = max(gap_max_f, int(round(pred_max_s * fps)))
    hold_f = max(1, int(round(hold_tail_s * fps)))
    cp = vmax / fps; L, W = cal["L"], cal["W"]
    out = []
    for tid, g in df.groupby(id_col):
        g = g.sort_values("frame")
        of = g["frame"].values.astype(int); ox = g["X"].values; oy = g["Y"].values
        ot = of / fps
        if len(of) < 5:
            continue
        # 1) teleport-red (anchor'a göre fiziksel-dışı sıçrama at)
        keep = [0]
        for k in range(1, len(of)):
            dt = ot[k] - ot[keep[-1]]; dd = np.hypot(ox[k] - ox[keep[-1]], oy[k] - oy[keep[-1]])
            if dt > 0 and dd / dt <= vmax * 1.2:
                keep.append(k)
        of, ox, oy, ot = of[keep], ox[keep], oy[keep], ot[keep]
        if len(of) < 5:
            continue
        # 2) savgol düzleştir (gözlem-gürültüsü, gerçek hareketi bozmadan)
        w = min(savgol_w, len(ox) if len(ox) % 2 else len(ox) - 1); w = max(5, w if w % 2 else w - 1)
        if len(ox) >= w:
            ox = savgol_filter(ox, w, 2); oy = savgol_filter(oy, w, 2)
        # 3) segment KES yalnız: uzun-boşluk (>pred_max_f) VEYA fiziksel-dışı köprü-hızı.
        #    (Eskiden >gap_max_f'te kesip siliyordu; artık ~2s'e kadar köprülenir.)
        seg_bounds = [0]
        for k in range(1, len(of)):
            gap = of[k] - of[k - 1]; dt = gap / fps
            bs = (np.hypot(ox[k] - ox[k - 1], oy[k] - oy[k - 1]) / dt) if dt > 0 else 0.0
            if gap > pred_max_f or bs > vmax:
                seg_bounds.append(k)
        seg_bounds.append(len(of))
        nseg = len(seg_bounds) - 1
        for si in range(nseg):
            a, b = seg_bounds[si], seg_bounds[si + 1]
            sf, sx, sy = of[a:b], ox[a:b], oy[a:b]
            if len(sf) == 0:
                continue
            ff = np.arange(sf[0], sf[-1] + 1)                # segment-içi her frame
            xi = np.interp(ff, sf, sx); yi = np.interp(ff, sf, sy)
            # sert hız-tavanı (teleport-yok)
            for k in range(1, len(xi)):
                st = np.hypot(xi[k] - xi[k - 1], yi[k] - yi[k - 1])
                if st > cp > 0:
                    r = cp / st; xi[k] = xi[k - 1] + (xi[k] - xi[k - 1]) * r
                    yi[k] = yi[k - 1] + (yi[k] - yi[k - 1]) * r
            obs = set(sf.tolist())
            # her ara-frame'in dahil olduğu gözlem-boşluğu (status için): <=gap_max=interp, üstü=pred
            li = np.clip(np.searchsorted(sf, ff, side="right") - 1, 0, len(sf) - 1)
            ri = np.clip(li + 1, 0, len(sf) - 1)
            local_gap = sf[ri] - sf[li]
            for j, fr in enumerate(ff):
                stt = "obs" if int(fr) in obs else ("interp" if local_gap[j] <= gap_max_f else "pred")
                out.append(dict(pid=int(tid), frame=int(fr), seg=int(si),
                                x=float(xi[j]), y=float(yi[j]), status=stt))
            # 4) İLERİ-TAHMİN KUYRUĞU: segment sonundan coast->hold (kaleci yerinde kalır).
            #    Sonraki segment başlangıcını GEÇMEZ (çift-çizim yok); status='pred'.
            kvw = min(8, len(xi) - 1)
            vx = (xi[-1] - xi[-1 - kvw]) / kvw if kvw >= 1 else 0.0
            vy = (yi[-1] - yi[-1 - kvw]) / kvw if kvw >= 1 else 0.0
            tail_end = ff[-1] + hold_f
            if si + 1 < nseg:
                tail_end = min(tail_end, int(of[seg_bounds[si + 1]]) - 1)
            px, py = float(xi[-1]), float(yi[-1])
            for fr in range(int(ff[-1]) + 1, tail_end + 1):
                vx *= 0.90; vy *= 0.90                        # hız sön -> sabit-tut (asimptotik)
                sp = np.hypot(vx, vy)
                if sp > cp > 0:
                    vx *= cp / sp; vy *= cp / sp
                px = float(np.clip(px + vx, -1, L + 1)); py = float(np.clip(py + vy, -1, W + 1))
                out.append(dict(pid=int(tid), frame=int(fr), seg=int(si),
                                x=px, y=py, status="pred"))
    import pandas as pd
    return pd.DataFrame(out)


# ---- render ----
def _draw_field(im, cal, m2px, S, draw_nominal=True):
    L, W = cal["L"], cal["W"]; tmpl = cal["template"]
    SOLID = (235, 235, 235); NOM = (120, 150, 120)          # kalibre=beyaz, nominal=soluk
    # kalibre çizgiler (template.line_segments_m) — bunlar ÖLÇÜLMÜŞ
    for (p0, p1) in tmpl.get("line_segments_m", []):
        cv2.line(im, m2px(*p0), m2px(*p1), SOLID, 2)
    if not tmpl.get("line_segments_m"):
        cv2.rectangle(im, m2px(0, 0), m2px(L, W), SOLID, 2)
        cv2.line(im, m2px(L / 2, 0), m2px(L / 2, W), SOLID, 2)
    # kaleler (3m ağız — güvenilir çapa)
    for gx in (0.0, L):
        cv2.line(im, m2px(gx, W / 2 - 1.5), m2px(gx, W / 2 + 1.5), (0, 0, 235), 3)
    if draw_nominal:
        # orta yuvarlak + ceza sahası: NOMINAL (kalibre değil) -> soluk, "tahmini" etiketli.
        # Alperen kuralı: ceza sahası DERİNLİĞİ ~yarı-sahanın 1/4'ü = (L/2)/4 = L/8 (biçme-şeritleriyle
        # uyumlu). KESİN ölçü görünen beyaz kutudan gelecek (veri arttıkça line_calib'e eklenecek).
        cv2.circle(im, m2px(L / 2, W / 2), int(3.0 * S), NOM, 1)
        bd, bhw = L / 8.0, 5.0
        for x0, xd in ((0.0, bd), (L, -bd)):
            cv2.rectangle(im, m2px(min(x0, x0 + xd), W / 2 - bhw),
                          m2px(max(x0, x0 + xd), W / 2 + bhw), NOM, 1)


def _fade(c, k, add):
    return tuple(int(k * v + add) for v in c)


_GK_AMBER = (60, 200, 245)                                   # GK = altın/amber (ayırt edici)


def _draw_players_2d(im, f, pids, rg, col, gk_at_f, trail_f, m2px):
    """2D paneline f karesinde oyuncuları çiz (build_replay + build_stacked ortak).
    status'a göre stil: obs=dolu, interp=yarı-soluk dolu, pred=İÇİ-BOŞ soluk halka
    (tahmin: görmüyoruz ama tutuyoruz). gk_at_f = O FRAME'de GK olan pid kümesi (zaman-değişken
    rol; sabit gk_ids yerine per-frame timeline). p in gk_at_f -> amber renk + 'GK' etiketi
    DİNAMİK uygulanır (col precompute'ta sabit amber YOK). Döner: çizilen sayı."""
    n = 0
    for p in pids:
        is_gk = p in gk_at_f                                  # bu frame'de GK mi (rotasyon takibi)
        base = _GK_AMBER if is_gk else col[p]                # amber DİNAMİK (frame-bazlı)
        g = rg[p]; cur = g[g["frame"] <= f]
        if len(cur) == 0 or cur["frame"].iloc[-1] < f - 3:
            continue
        tail = cur[cur["frame"] >= f - trail_f]; rows = list(tail.itertuples())
        for i in range(1, len(rows)):
            if rows[i].seg != rows[i - 1].seg:
                continue                                     # segment kopuşunda iz çizme
            s = rows[i].status
            c = base if s == "obs" else (_fade(base, 0.45, 25) if s == "interp"
                                         else _fade(base, 0.30, 18))
            cv2.line(im, m2px(rows[i - 1].x, rows[i - 1].y), m2px(rows[i].x, rows[i].y), c, 2)
        last = cur.iloc[-1]; x, y = float(last["x"]), float(last["y"]); status = last["status"]
        r = 7 if is_gk else 6; cx, cy = m2px(x, y)
        if status == "obs":
            cv2.circle(im, (cx, cy), r, base, -1); cv2.circle(im, (cx, cy), r, (240, 240, 240), 1)
        elif status == "interp":
            cv2.circle(im, (cx, cy), r, _fade(base, 0.6, 18), -1)
            cv2.circle(im, (cx, cy), r, (180, 180, 180), 1)
        else:                                                # pred -> içi-boş soluk halka
            cv2.circle(im, (cx, cy), max(1, r - 3), _fade(base, 0.35, 12), -1)
            cv2.circle(im, (cx, cy), r, base, 2)
        n += 1
        if is_gk:
            cv2.putText(im, "GK", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GK_AMBER, 2)
    return n


def _gk_at_frame(f, gk_timeline, gk_ids):
    """O frame'de GK olan pid kümesini çöz. gk_timeline ({frame->{'near','far'}}) verilmiş ve
    frame'i kapsıyorsa zaman-değişken rolü kullan; aksi halde SABIT gk_ids'e düş (geriye-uyumlu)."""
    if gk_timeline is not None:
        ent = gk_timeline.get(int(f))
        if ent is not None:
            return {v for v in ent.values() if v is not None}
    return gk_ids


def build_replay(state_df, cal, out_mp4, fps=12, trail_f=45, S=26, pad=46,
                 t0_label=0.0, src_fps=None, title="", gk_ids=None, gk_timeline=None):
    import pandas as pd
    gk_ids = set(gk_ids or [])                               # SABIT kaleci pid'leri (fallback)
    L, W = cal["L"], cal["W"]
    Wp = int(L * S + 2 * pad); Hp = int(W * S + 2 * pad)
    Wp += Wp % 2; Hp += Hp % 2
    def m2px(x, y): return int(pad + x * S), int(pad + (W - y) * S)
    pids = sorted(state_df["pid"].unique())
    rng = np.linspace(0, 179, max(1, len(pids))).astype(int)
    col = {p: tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(h), 210, 255]]]),
                                                 cv2.COLOR_HSV2BGR)[0, 0]) for p, h in zip(pids, rng)}
    # NOT: amber col precompute'ta SABIT atanmaz; _draw_players_2d'de gk_at_f ile DİNAMİK uygulanır
    rg = {p: g.sort_values("frame") for p, g in state_df.groupby("pid")}
    f0, f1 = int(state_df["frame"].min()), int(state_df["frame"].max())
    sfps = src_fps or fps
    out = None
    for f in range(f0, f1 + 1):
        im = np.full((Hp, Wp, 3), 26, np.uint8)
        _draw_field(im, cal, m2px, S)
        gk_at_f = _gk_at_frame(f, gk_timeline, gk_ids)
        n = _draw_players_2d(im, f, pids, rg, col, gk_at_f, trail_f, m2px)
        cv2.rectangle(im, (0, 0), (Wp, 30), (0, 0, 0), -1)
        cv2.putText(im, title or f"2D replay  saha {L:.1f}x{W:.1f}m  {cal['band']}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1)
        cv2.putText(im, f"t={f/sfps - t0_label:.1f}s  aktif={n}  (dolu=gozlem, yari-soluk=interp, "
                    f"ici-bos halka=TAHMIN/tutuluyor; ince-cizgi=nominal marking)", (8, Hp - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        if out is None:
            out = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wp, Hp))
        out.write(im)
    if out is not None:
        out.release()
    return out_mp4


# ---- GK pid'lerini consolidate raporundan oku (doğru pid'e GK etiketi garanti) ----
def gk_pids_from_report(report_path: str) -> set:
    """consolidate_report.json -> {GK_A pid, GK_L pid}. gk_anchor'ın çıktısıyla uyumlu.
    GK etiketi 'anlık en-arka oyuncu' DEĞİL; kale-ucu derin-rezidans + advance-return zinciri.
    DİKKAT: bu rapor consolidate aşamasının GK ÖN-SEED'idir; CEPHE-2 yeniden skorlayıp
    düzeltebilir (örn. endL: 8 -> 13). Kanonik kaynak gk_pids_from_cephe2'dir; bu yalnız
    fallback. resolve_gk_pids() ikisini birleştirir."""
    try:
        r = json.load(open(report_path))
        gs = r.get("gk_summary", {})
        return {gs[e]["player_id"] for e in ("endA", "endL")
                if gs.get(e, {}).get("player_id") is not None}
    except Exception:
        return set()


def gk_pids_from_cephe2(table_path: str) -> set:
    """CEPHE-2 gk_score_table.json -> {uc-basi DÜZELTİLMİŞ gk_pick}. KANONİK GK kaynağı.
    Cephe-2 'anlık en-arka' önyargısını düzeltir: derin-rezidans + advance-return ile her
    kale-ucuna TEK sabit pid atar (verdict[end].gk_pick). consolidate raporu yanlışsa
    (correct=False) bu pick onu geçersiz kılar — pipeline doğru GK'yı bu dosyadan alır."""
    try:
        d = json.load(open(table_path))
        v = d.get("verdict", {})
        return {v[e]["gk_pick"] for e in v if v[e].get("gk_pick") is not None}
    except Exception:
        return set()


def resolve_gk_pids(cephe2_table: str | None = None, report_path: str | None = None) -> tuple[set, str]:
    """GK pid'lerini PARAMETRİK çöz: önce CEPHE-2 düzeltilmiş tablosu, yoksa consolidate raporu.
    Döner (gk_set, kaynak_etiketi). Pipeline GK etiketini bu tek noktadan alır -> CEPHE-2
    yeniden-skorlaması (8->13 gibi) otomatik yansır, stale rapor pid'i gizlice kullanılmaz."""
    if cephe2_table and Path(cephe2_table).exists():
        g = gk_pids_from_cephe2(cephe2_table)
        if g:
            return g, f"cephe2:{Path(cephe2_table).name}"
    if report_path and Path(report_path).exists():
        g = gk_pids_from_report(report_path)
        if g:
            return g, f"report:{Path(report_path).name}"
    return set(), "none"


# ---- v5 > v4 kalibrasyon çözücü (sanity-kapılı; CEPHE-1 v5'i hazır olunca otomatik alınır) ----
def resolve_calib_path(calib_dir: str = "calib", df=None, prefer_v5: bool = True) -> tuple[str, str]:
    """En iyi MEVCUT calib dosyasını seç. Sıra: v5(final, CEPHE-1) > v4 > v2.
    NOT: cankaya_cam2_v5_auto.json BİLEREK auto-sıraya alınmadı — auto-fit ara-ürün, undistort'u
    AŞIRI-bombeli (v4'ten görsel-kötü, far-reproj sayısı iyi olsa da); CLI ile açıkça verilebilir.
    df verilirse her aday için ayak-pikselleri projekte edip in-bounds oranını ölçer; < 0.90
    olan aday DEGENERE sayılıp atlanır (bozuk v5 çıktıyı sessizce bozmasın). Döner (yol, neden)."""
    order = (["cankaya_cam2_v5.json"] if prefer_v5 else []) + \
            ["cankaya_cam2_v4.json", "cankaya_cam2_v2.json"]
    foot = None
    if df is not None and {"foot_x", "foot_y"}.issubset(df.columns):
        d2 = df[df["player_id"] >= 0] if "player_id" in df.columns else df
        foot = d2[["foot_x", "foot_y"]].values
    for name in order:
        p = str(Path(calib_dir) / name)
        if not Path(p).exists():
            continue
        if foot is None or len(foot) == 0:
            return p, f"{name} (var; sanity-atlandı)"
        try:
            cal = load_calib(p); P = project(cal, foot)
            inb = float(((P[:, 0] >= -1) & (P[:, 0] <= cal["L"] + 1) &
                         (P[:, 1] >= -1) & (P[:, 1] <= cal["W"] + 1)).mean())
            if inb >= 0.90:
                return p, f"{name} (in-bounds={inb:.3f})"
        except Exception:
            continue
    # hiçbiri geçmediyse v4'e düş (kanıtlanmış)
    return str(Path(calib_dir) / "cankaya_cam2_v4.json"), "v4 (fallback; aday sanity gecemedi)"


# ---- alpha=1 görüntü-kesmeyen undistort (Alperen: 'görüntüyü kesme') ----
def undistort_alpha1(im, K, D):
    """cv2.undistort(K) köşeleri KIRPAR (barrel düzeltmesi sahneyi dışarı iter). alpha=1
    getOptimalNewCameraMatrix tüm geçerli pikselleri korur (siyah bant ekler, sahne kaybetmez).
    Döner: (undist_img, newK). newK kullanılırsa kalibre çizgileri bu görüntüye projekte edilebilir."""
    h, w = im.shape[:2]
    newK, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
    return cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR), newK


def build_stacked(state_df, cal, video, out_mp4, src_fps, fps=12, t0_label=0.0,
                  trail_f=45, S=22, pad=40, gk_ids=None, title="", preview_png=None,
                  gk_timeline=None):
    """ÜST: alpha=1 full-frame gerçek görüntü (kesilmez) | ALT: 2D top-down replay — senkron.

    Alperen'in 'alpha=1 full-frame stacked çıktı' isteği. 2D panel build_replay ile AYNI
    çizim (GK amber + segment-kapılı iz). Video frame'leri state frame'lerine eşlenir.
    gk_timeline verilirse GK zaman-değişken rol (per-frame); yoksa sabit gk_ids (geriye-uyumlu)."""
    import pandas as pd
    gk_ids = set(gk_ids or [])
    L, W = cal["L"], cal["W"]; K, D = cal["K"], cal["D"]
    Wp = int(L * S + 2 * pad); Hp = int(W * S + 2 * pad); Wp += Wp % 2; Hp += Hp % 2
    def m2px(x, y): return int(pad + x * S), int(pad + (W - y) * S)
    pids = sorted(state_df["pid"].unique())
    rng = np.linspace(0, 179, max(1, len(pids))).astype(int)
    col = {p: tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(h), 210, 255]]]),
                                                 cv2.COLOR_HSV2BGR)[0, 0]) for p, h in zip(pids, rng)}
    # amber DİNAMİK (gk_at_f), col precompute'ta SABIT amber YOK
    rg = {p: g.sort_values("frame") for p, g in state_df.groupby("pid")}
    f0, f1 = int(state_df["frame"].min()), int(state_df["frame"].max())
    cap = cv2.VideoCapture(str(video))
    out = None; first = None
    for f in range(f0, f1 + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        if not ok:
            continue
        und, _ = undistort_alpha1(im, K, D)
        # 2D panel
        b = np.full((Hp, Wp, 3), 26, np.uint8); _draw_field(b, cal, m2px, S)
        gk_at_f = _gk_at_frame(f, gk_timeline, gk_ids)
        n = _draw_players_2d(b, f, pids, rg, col, gk_at_f, trail_f, m2px)
        # üst görüntü genişliğini 2D paneline eşle
        rw = Wp; rh = int(rw * und.shape[0] / und.shape[1]); rh += rh % 2
        topr = cv2.resize(und, (rw, rh))
        cv2.putText(topr, "alpha=1 full-frame (kesilmez)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(b, f"t={f/src_fps - t0_label:.1f}s  aktif={n}  {title}", (8, Hp - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        frame = np.vstack([topr, b])
        frame = frame[:frame.shape[0] - frame.shape[0] % 2, :frame.shape[1] - frame.shape[1] % 2]
        if first is None:
            first = frame.copy()
        if out is None:
            out = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                  (frame.shape[1], frame.shape[0]))
        out.write(frame)
    cap.release()
    if out is not None:
        out.release()
    if preview_png is not None and first is not None:
        cv2.imwrite(str(preview_png), first)
    return out_mp4


if __name__ == "__main__":
    import sys, os, pandas as pd
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)                            # detect paketi icin (script-modu)
    # CEPHE-4 entegre varsayılanlar: konsolide kimlik + v5>v4 calib + GK CEPHE-2'den (parametrik).
    raw = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/wf_identity/consolidated.parquet"
    calib = sys.argv[2] if len(sys.argv) > 2 else None    # None -> resolve_calib_path (v5>v4)
    report = sys.argv[3] if len(sys.argv) > 3 else "scratchpad/wf_identity/consolidate_report.json"
    outdir = sys.argv[4] if len(sys.argv) > 4 else "scratchpad/wf2_pipe"
    cephe2_gk = sys.argv[5] if len(sys.argv) > 5 else "scratchpad/wf2_gk/gk_score_table.json"
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(raw)
    # (a) calib: v5'i de yükle — CEPHE-1 v5'i hazırsa sanity-kapılı olarak otomatik seçilir
    if calib is None:
        calib, calreason = resolve_calib_path("calib", df=df, prefer_v5=True)
    else:
        calreason = "CLI"
    cal = load_calib(calib)
    # (c) GK parametrik: CEPHE-2 düzeltilmiş tablosu (yoksa consolidate raporu) — SABIT fallback
    gk_ids, gk_src = resolve_gk_pids(cephe2_gk, report)
    src_fps = 88294 / 3549.97
    # (c2) ZAMAN-DEĞİŞKEN GK ROL TIMELINE: amator kaleci rotasyonu + kimlik-parçalanmasına dayanıklı.
    #      Her frame {'near','far'} pid; build_replay/build_stacked'e verilir (sabit gk_ids fallback).
    from detect.gk_anchor import gk_role_timeline           # lokal import (dairesel-import önle)
    gk_timeline = gk_role_timeline(df, cal, fps=src_fps)
    # (d) gap-kapılı postprocess: kısa boşluk köprülenir (status=interp), uzun boşluk KESİLİR
    st = postprocess(df, cal, fps=src_fps)        # id_col OTO: player_id (konsolide)
    st.to_parquet(os.path.join(outdir, "replay2d_state.parquet"))

    # ---- ÖLÇÜMLER ----
    print(f"[calib] {Path(calib).name}  ({calreason})  saha {cal['L']:.1f}x{cal['W']:.1f}m  far-reproj {cal['far_reproj']}px")
    print(f"[id]    id_col={'player_id' if 'player_id' in df.columns else 'tid'}")
    print(f"[GK]    gk_ids={sorted(gk_ids)}  kaynak={gk_src}   (rapor={sorted(gk_pids_from_report(report))} -> CEPHE-2 düzeltmesi yansıdı mı?)")
    # zaman-değişken GK rol dağılımı: her uçta hangi pid kaç frame seçildi (rotasyon/parçalanma görünür)
    from collections import Counter
    _ntot = len(gk_timeline) or 1
    for _end in ("near", "far"):
        _c = Counter(gk_timeline[f][_end] for f in gk_timeline)
        _gaps = _c.pop(None, 0)
        _dist = ", ".join(f"pid{p}:{100*n/_ntot:.0f}%" for p, n in _c.most_common())
        print(f"[GK-tl] {_end:>4} kale: {_dist if _dist else '(yok)'}  [gap]={100*_gaps/_ntot:.0f}% frame  ({_ntot} frame)")
    print(f"[post]  {st['pid'].nunique()} oyuncu, {st.groupby('pid')['seg'].nunique().mean():.2f} seg/oyuncu-ort, {len(st)} satır")
    # (d) honest spawn/gap ölçümü: aktif sayısının obs vs interp dağılımı; köprü-yokluğu görünür
    per_f = st.groupby("frame")["status"].agg(
        active="count", obs=lambda s: (s == "obs").sum(), pred=lambda s: (s == "pred").sum())
    print(f"[gap]   aktif/kare: min={per_f['active'].min()} max={per_f['active'].max()} medyan={int(per_f['active'].median())}  "
          f"(obs-payı medyan={float((per_f['obs']/per_f['active']).median()):.2f}; "
          f"pred-tutulan/kare medyan={int(per_f['pred'].median())} — eskiden bunlar SİLİNİYORDU)")
    sc = st["status"].value_counts()
    print(f"[pred]  satır durum dağılımı: obs={int(sc.get('obs',0))} interp={int(sc.get('interp',0))} "
          f"pred={int(sc.get('pred',0))}  (pred = kaleci/oyuncu kısa-boşlukta tahminle TUTULDU, silinmedi)")
    seg_per = st.groupby("pid")["seg"].nunique()
    print(f"[gap]   uzun-boşlukla KESİLEN izler (seg>1): {int((seg_per > 1).sum())}/{len(seg_per)} oyuncu  "
          f"(gözlem-yokluğu GİZLENMEZ; sahte cross-field köprü yok)")

    # 2D-only video + tek-kare önizleme
    build_replay(st, cal, os.path.join(outdir, "replay2d_demo.mp4"), src_fps=src_fps,
                 t0_label=2690, gk_ids=gk_ids, gk_timeline=gk_timeline,
                 title=f"CEPHE4: consolidated pid + {Path(calib).stem} + GK(timeline)")
    # (b) alpha=1 full-frame stacked (KIRPMA YOK) — üst gerçek undistort + alt 2D
    stacked_dims = None
    if os.path.exists(raw) and Path("raw/cankaya_cam2.mp4").exists():
        build_stacked(st, cal, "raw/cankaya_cam2.mp4",
                      os.path.join(outdir, "replay2d_stacked.mp4"), src_fps=src_fps,
                      t0_label=2690, gk_ids=gk_ids, gk_timeline=gk_timeline,
                      title=f"{Path(calib).stem}+consolidated",
                      preview_png=os.path.join(outdir, "replay2d_stacked_preview.png"))
        prev = cv2.imread(os.path.join(outdir, "replay2d_stacked_preview.png"))
        if prev is not None:
            stacked_dims = prev.shape
    # full-frame doğrulama: alpha=1 undistort orijinal 1920x1080'ı KIRPMADAN korur mu?
    cap = cv2.VideoCapture("raw/cankaya_cam2.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 66905); ok, fr = cap.read(); cap.release()
    if ok:
        und, newK = undistort_alpha1(fr, cal["K"], cal["D"])
        print(f"[full]  ham {fr.shape[1]}x{fr.shape[0]} -> alpha=1 undistort {und.shape[1]}x{und.shape[0]} "
              f"(boyut korunur, KIRPMA YOK; siyah-bant=geçersiz piksel)  stacked={stacked_dims}")
    print(f"yazildi {outdir}/replay2d_demo.mp4 + replay2d_stacked.mp4 + state/preview")

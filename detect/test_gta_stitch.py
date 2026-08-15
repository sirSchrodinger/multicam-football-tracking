#!/usr/bin/env python3
"""gta_stitch sentetik testleri — bilinen GROUND-TRUTH ile SPLIT/CONNECT davranisi.

Tum sahneler saha-metresi uzayinda ELLE kurulur (gercek tracker'a gerek yok);
beklenen sonuc modul-disi geometriden GELIR, modulun icine GOMULMEZ.
"""
import numpy as np

from detect.gta_stitch import (
    Tracklet, make_tracklet, split_tracklets, connect_tracklets,
    over_merge_count, tracklets_from_df,
)

FPS = 25.0


def _line(p0, p1, f0, n, fps=FPS, jitter=0.0, seed=0):
    """f0..f0+n-1 kareleri boyunca p0'dan p1'e duz hareket (opsiyonel jitter)."""
    rng = np.random.default_rng(seed)
    frames = np.arange(f0, f0 + n)
    s = np.linspace(0.0, 1.0, n)[:, None]
    xy = (1 - s) * np.asarray(p0, float) + s * np.asarray(p1, float)
    if jitter:
        xy = xy + rng.normal(0, jitter, size=xy.shape)
    return frames, xy


# ----------------------------------------------------------------- SPLITTER --
def test_splitter_teleport_is_cut_into_two():
    # Tek tracklet: 0-39 kareler kimlik A ~(5,5), 40-79 kimlik B ~(28,16).
    # 39->40 sicramasi 1 karede ~25m = imkansiz hiz => 2'ye BOLUNMELI.
    fA, xyA = _line((5, 5), (5.5, 5.2), 0, 40, jitter=0.15, seed=1)
    fB, xyB = _line((28, 16), (28.4, 16.3), 40, 40, jitter=0.15, seed=2)
    tr = make_tracklet(7, np.concatenate([fA, fB]),
                       np.concatenate([xyA, xyB]), fps=FPS)
    subs = split_tracklets([tr], max_speed_mps=8.0, eps_m=2.0, fps=FPS)
    assert len(subs) == 2, f"teleport 2'ye bolunmeliydi, {len(subs)} cikti"
    # her alt-tracklet tek bolgede: biri A yakini, biri B yakini
    cents = sorted(float(s.xy[:, 0].mean()) for s in subs)
    assert cents[0] < 10 and cents[1] > 20
    # satir kaybi yok
    assert sum(len(s) for s in subs) == len(tr)


def test_splitter_smooth_run_not_split():
    # Tek oyuncu sahayi bastan basa kosuyor (0->30m, 100 kare). Genis uzaysal
    # yayilim AMA hiz makul => DBSCAN tek yogunluk-zinciri, BOLUNMEMELI.
    f, xy = _line((2, 10), (30, 10), 0, 100, jitter=0.05, seed=3)
    tr = make_tracklet(1, f, xy, fps=FPS)
    subs = split_tracklets([tr], max_speed_mps=8.0, eps_m=2.0, fps=FPS)
    assert len(subs) == 1, f"surekli kosu bolunmemeliydi, {len(subs)} cikti"


def test_splitter_occlusion_gap_same_identity_not_split():
    # Ayni oyuncu: 0-24 (5,5) civari, sonra 1s bosluk, 50-74 (6.2,5) civari.
    # Bosluk-suresi buyuk (1s) => gerekli hiz dusuk, teleport YOK => BOLUNMEMELI.
    fA, xyA = _line((5, 5), (5.2, 5), 0, 25, jitter=0.05, seed=4)
    fB, xyB = _line((6.0, 5), (6.4, 5), 50, 25, jitter=0.05, seed=5)
    tr = make_tracklet(2, np.concatenate([fA, fB]),
                       np.concatenate([xyA, xyB]), fps=FPS)
    subs = split_tracklets([tr], max_speed_mps=8.0, eps_m=2.0, fps=FPS)
    assert len(subs) == 1, f"occlusion-gap ayni kimlik bolunmemeliydi ({len(subs)})"


def test_splitter_single_frame_glitch_remerged():
    # Tek karelik isinma (spike): A bolgesinden 30m firlayip geri donuyor.
    # DBSCAN spike'i A kumesine yapistirir => ayni baskin etiket => BIRLESTIR.
    f, xy = _line((5, 5), (5.5, 5), 0, 40, jitter=0.05, seed=6)
    xy[20] = [35, 18]   # tek karelik teleport-out
    tr = make_tracklet(3, f, xy, fps=FPS)
    subs = split_tracklets([tr], max_speed_mps=8.0, eps_m=2.0,
                           min_seg_frames=3, fps=FPS)
    assert len(subs) == 1, f"tek-kare glitch yeniden-birlesmeliydi ({len(subs)})"


# ----------------------------------------------------------------- CONNECTOR -
def test_connector_joins_slow_player_across_gap():
    # Bir oyuncunun iki parcasi: 0-24 ~(10,10), 1s bosluk, 50-74 ~(11,10).
    # 0.5m / ~1s = 0.5 m/s << 7 => BAGLANMALI (tek player_id).
    fA, xyA = _line((10, 10), (10.5, 10), 0, 25, jitter=0.03, seed=7)
    fB, xyB = _line((11.0, 10), (11.5, 10), 50, 25, jitter=0.03, seed=8)
    F1 = make_tracklet("F1", fA, xyA, fps=FPS)
    F2 = make_tracklet("F2", fB, xyB, fps=FPS)
    res = connect_tracklets([F1, F2], max_speed_mps=7.0, max_gap_s=5.0, fps=FPS)
    assert res.n_output == 1, f"yavas-bosluk baglanmaliydi, {res.n_output} kume"
    assert res.labels["F1"] == res.labels["F2"]
    assert over_merge_count(res.labels, [F1, F2], fps=FPS) == 0


def test_connector_never_merges_concurrent_players():
    # Iki ESZAMANLI oyuncu (0-49 kareler), uzakta ve yakinda iki sahne dene.
    for (px, py) in [(25, 16), (5.4, 5)]:   # biri uzak, biri 0.4m yakin-markaj
        fA, xyA = _line((5, 5), (5.3, 5), 0, 50, jitter=0.05, seed=9)
        fB, xyB = _line((px, py), (px + 0.3, py), 0, 50, jitter=0.05, seed=10)
        P1 = make_tracklet("P1", fA, xyA, fps=FPS)
        P2 = make_tracklet("P2", fB, xyB, fps=FPS)
        res = connect_tracklets([P1, P2], max_speed_mps=7.0, max_gap_s=10.0,
                                fps=FPS)
        assert res.n_output == 2, f"eszamanli oyuncular birlesti (px={px})"
        assert res.labels["P1"] != res.labels["P2"]
        assert over_merge_count(res.labels, [P1, P2], fps=FPS) == 0


def test_connector_rejects_fast_teleport():
    # F1 0-24 ~(5,5), F2 26-50 ~(35,16): bosluk ~0.08s, mesafe ~31m =>
    # gerekli hiz ~390 m/s >> 7 => BAGLANMAMALI.
    fA, xyA = _line((5, 5), (5.2, 5), 0, 25, jitter=0.02, seed=11)
    fB, xyB = _line((35, 16), (35.2, 16), 26, 25, jitter=0.02, seed=12)
    F1 = make_tracklet("F1", fA, xyA, fps=FPS)
    F2 = make_tracklet("F2", fB, xyB, fps=FPS)
    res = connect_tracklets([F1, F2], max_speed_mps=7.0, max_gap_s=5.0, fps=FPS)
    assert res.n_output == 2, "hizli teleport baglanmamaliydi"
    assert res.labels["F1"] != res.labels["F2"]


def test_connector_respects_max_gap():
    # Yavas ama bosluk cok buyuk (4s > max_gap 2s) => baglanmamali.
    fA, xyA = _line((10, 10), (10.2, 10), 0, 25, jitter=0.02, seed=13)
    fB, xyB = _line((10.5, 10), (10.7, 10), 25 + 100, 25, jitter=0.02, seed=14)
    F1 = make_tracklet("F1", fA, xyA, fps=FPS)
    F2 = make_tracklet("F2", fB, xyB, fps=FPS)
    res = connect_tracklets([F1, F2], max_speed_mps=7.0, max_gap_s=2.0, fps=FPS)
    assert res.n_output == 2, "max_gap asimi baglanmamaliydi"


def test_connector_team_label_blocks_merge():
    # Mekansal/zamansal olarak baglanabilir AMA takimlar farkli => yasak.
    fA, xyA = _line((10, 10), (10.3, 10), 0, 25, jitter=0.02, seed=15)
    fB, xyB = _line((10.6, 10), (10.9, 10), 50, 25, jitter=0.02, seed=16)
    F1 = make_tracklet("F1", fA, xyA, team=0, fps=FPS)
    F2 = make_tracklet("F2", fB, xyB, team=1, fps=FPS)
    res = connect_tracklets([F1, F2], max_speed_mps=7.0, max_gap_s=5.0,
                            use_team=True, fps=FPS)
    assert res.n_output == 2, "takim uyusmazligi baglamayi engellemeliydi"
    # ayni takim olunca AYNI veri baglanir (kontrol)
    G1 = make_tracklet("G1", fA, xyA, team=1, fps=FPS)
    G2 = make_tracklet("G2", fB, xyB, team=1, fps=FPS)
    res2 = connect_tracklets([G1, G2], max_speed_mps=7.0, max_gap_s=5.0,
                             use_team=True, fps=FPS)
    assert res2.n_output == 1, "ayni takim baglanmaliydi"


def test_connector_chain_three_fragments_one_player():
    # Ucе parca tek oyuncu, ardisik bosluklarla => tek player_id, over-merge 0.
    fA, xyA = _line((5, 8), (5.4, 8), 0, 20, jitter=0.02, seed=17)
    fB, xyB = _line((6.0, 8), (6.4, 8), 40, 20, jitter=0.02, seed=18)
    fC, xyC = _line((7.0, 8), (7.4, 8), 80, 20, jitter=0.02, seed=19)
    trs = [make_tracklet(k, f, xy, fps=FPS) for k, (f, xy) in
           zip("ABC", [(fA, xyA), (fB, xyB), (fC, xyC)])]
    res = connect_tracklets(trs, max_speed_mps=7.0, max_gap_s=2.0, fps=FPS)
    assert res.n_output == 1
    assert len(set(res.labels.values())) == 1
    assert over_merge_count(res.labels, trs, fps=FPS) == 0


# ----------------------------------------------- entegrasyon yardimcisi (df) -
def test_tracklets_from_df_roundtrip():
    import pandas as pd
    df = pd.DataFrame({
        "tid": [1, 1, 1, 2, 2],
        "frame": [2, 0, 1, 5, 6],            # bilerek karisik sira
        "t_sec": [0.08, 0.0, 0.04, 0.20, 0.24],
        "pitch_x": [5.0, 4.0, 4.5, 20.0, 20.5],
        "pitch_y": [5.0, 5.0, 5.0, 10.0, 10.0],
    })
    trs = tracklets_from_df(df, fps=FPS)
    assert len(trs) == 2
    t1 = next(t for t in trs if t.tid == 1)
    # kare-artan siralanmis olmali
    assert list(t1.frames) == [0, 1, 2]
    assert np.allclose(t1.xy[0], [4.0, 5.0])


# --------------------------------------------- split+connect uctan uca pipeline
def test_split_then_connect_recovers_pieces():
    # Bir teleport-tracklet'i BOL, sonra ayni-kimlik baska parcayla BAGLA.
    fA, xyA = _line((5, 5), (5.4, 5), 0, 30, jitter=0.05, seed=20)
    fB, xyB = _line((28, 16), (28.4, 16), 30, 30, jitter=0.05, seed=21)
    bad = make_tracklet(99, np.concatenate([fA, fB]),
                        np.concatenate([xyA, xyB]), fps=FPS)
    subs = split_tracklets([bad], max_speed_mps=8.0, eps_m=2.0, fps=FPS)
    assert len(subs) == 2
    # A bolgesinin devami: 90-119 ~(6,5), bolunmus A-parcasiyla baglanmali
    fC, xyC = _line((6.0, 5), (6.4, 5), 90, 30, jitter=0.05, seed=22)
    cont = make_tracklet("cont", fC, xyC, fps=FPS)
    res = connect_tracklets(subs + [cont], max_speed_mps=7.0, max_gap_s=3.0,
                            fps=FPS)
    # A-parcasi + cont ayni player; B ayri => 2 kume, over-merge 0
    assert res.n_output == 2
    assert over_merge_count(res.labels, subs + [cont], fps=FPS) == 0


if __name__ == "__main__":
    for k, v in dict(globals()).items():
        if k.startswith("test_"):
            v()
            print("ok", k)

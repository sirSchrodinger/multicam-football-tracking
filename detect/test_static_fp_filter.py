#!/usr/bin/env python3
"""static_fp_filter testleri — pano/file (piksel-kilitli) atılır, oyuncu korunur."""
from detect.static_fp_filter import flag_static_fp, clean_coco_static


def _box(frame, cx, foot_y, w=20, h=60):
    return dict(frame=frame, cx=cx, foot_y=foot_y, w=w, h=h)


def test_pixel_locked_structure_flagged():
    # 20 karede AYNI pikselde (pano) -> hepsi statik-FP
    boxes = [_box(f, 500.0, 120.0) for f in range(20)]
    flagged = flag_static_fp(boxes, n_frames_total=20, recur_frac=0.5, pos_std_px=6.0)
    assert len(flagged) == 20


def test_moving_player_kept():
    # her karede ilerleyen oyuncu (hareket) -> hicbiri flag degil
    boxes = [_box(f, 200.0 + f * 8, 300.0 + f * 4) for f in range(20)]
    flagged = flag_static_fp(boxes, n_frames_total=20, recur_frac=0.5, pos_std_px=6.0)
    assert len(flagged) == 0


def test_transient_not_flagged():
    # yalnız birkaç karede gorunen sabit nokta (seyrek) -> recur_frac altinda, korunur
    boxes = [_box(f, 700.0, 150.0) for f in range(3)]
    flagged = flag_static_fp(boxes, n_frames_total=30, recur_frac=0.5, pos_std_px=6.0)
    assert len(flagged) == 0


def test_slightly_drifting_standing_player_kept():
    # ayakta ama >6px gezinen oyuncu -> piksel-kilitli DEGIL, korunur
    import numpy as np
    rng = [0, 7, -8, 9, -6, 10, -9, 8, -7, 11, 6, -10]
    boxes = [_box(f, 400.0 + rng[f % len(rng)], 350.0 + rng[(f + 3) % len(rng)])
             for f in range(20)]
    flagged = flag_static_fp(boxes, n_frames_total=20, recur_frac=0.5, pos_std_px=6.0)
    assert len(flagged) == 0


def test_clean_coco_mixed():
    imgs = [dict(id=i, file_name=f"v_{i}.jpg", venue="V") for i in range(12)]
    anns = []
    aid = 0
    for i in range(12):
        # statik pano (cx=600,foot=100) her karede
        anns.append(dict(id=aid, image_id=i, bbox=[590.0, 70.0, 20.0, 30.0])); aid += 1
        # hareketli oyuncu
        anns.append(dict(id=aid, image_id=i, bbox=[100.0 + i * 9, 300.0, 22.0, 60.0])); aid += 1
    coco = dict(images=imgs, annotations=anns, categories=[dict(id=1, name="player")])
    new, rep = clean_coco_static(coco, recur_frac=0.5, pos_std_px=6.0)
    assert rep["total_dropped"] == 12          # 12 pano kutusu atildi
    assert len(new["annotations"]) == 12       # 12 oyuncu kutusu kaldi


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("ok", fn.__name__)

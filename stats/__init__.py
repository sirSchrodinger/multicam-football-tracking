"""Top-down istatistik katmani (stats paketi).

`topdown_stats` : per-frame track parquet'inden (export_tracks semasi) ust-acidan,
guven-agirlikli, durustluk-onceli per-player + per-zone metrik raporu uretir.
Tek projeksiyon kaynagi = pitch.homography.PitchHomography. Tespit YENIDEN
KOSTURULMAZ (parquet + calib I/O). GPU yok, matplotlib yok.
"""

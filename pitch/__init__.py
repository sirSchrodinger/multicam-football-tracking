"""Halısaha pitch-calibration paketi.

Sabit kamera varsayımı: saha homografisi tüm maç boyunca SABİTtir, bu yüzden
kamera/maç başına BİR KEZ kalibre edilir (broadcast PTZ gibi her-frame değil).

Modüller:
  template   : PitchTemplate (parametrik 5v5/7v7 saha modeli)
  homography : PitchHomography (manuel kalibrasyon + apply + persist + QA)
  auto_calib : temporal-median çizgi haritası + chamfer + self-verify gate

Birincil yol %100 lisans-temiz (OpenCV BSD, scipy BSD, numpy) ve KİRALIK GPU
GEREKTİRMEZ.
"""

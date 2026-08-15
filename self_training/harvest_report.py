#!/usr/bin/env python3
"""harvest_report.py — HARVEST_REPORT.md üretir (download + export + rebuild JSON'larından).

Tüm sayılar dosyalardan okunur (uydurma YOK). Çalıştır: rebuild_dataset.py SONRASI.
"""
import json, os
from pathlib import Path
REPORT = Path(os.environ.get("HARVEST_REPORT_DIR", "scratchpad/wf_dataset"))
COCO = Path("self_training/data/coco")

def load(p, d=None):
    p = REPORT / p
    return json.loads(p.read_text()) if p.exists() else (d if d is not None else {})

dl = load("download_manifest.json", [])
exp = load("export_results.json", [])
yields = load("yields.json", {})
stats = json.loads((COCO / "stats.json").read_text()) if (COCO / "stats.json").exists() else {}

lines = []
W = lines.append
W("# HARVEST_REPORT — çok-tesis dataset genişletme (Cephe D)")
W("")
W("Tarih: 2026-06-29. RF-DETR fine-tune dataset'ini tek-tesis-biaslı 3 tesisten")
W("çok-tesise genişletme. Hedef: amatör sabit-kamera dağılım çeşitliliği +")
W("FAR/küçük-oyuncu (zor recall) örnekleri. GPU: yerel 1650 Ti (~3.3 fps).")
W("")
W("## 1) Tesis keşfi (sosyalhalisaha.com)")
W("- Keşif API'si (`overnight_runner.discover_venues`) ÇALIŞIYOR. 7-günlük tarih")
W("  penceresinde ~10 maç/gün, çok sayıda farklı tesis erişilebilir.")
W("- **İndirme bulgusu (s1 vs s2):** CDN'de iki sunucu. `s2.*` mp4 faststart →")
W("  ffmpeg `-ss -c copy` ÇALIŞIR. `s1.*` mp4 faststart DEĞİL (moov sonda) →")
W("  ffmpeg moov-seek'te CDN 404 (curl range 206 çalışıyor → ffmpeg keep-alive).")
W("  s1 tesisleri yerelde ATLANDI (RunPod fix: tam-indir+remux, harvest_runpod.md).")
W("")
W("## 2) İndirilen klipler")
if dl:
    W("| tesis | indirildi | boyut | süre |")
    W("|---|---|---|---|")
    for m in dl:
        W(f"| {m['name']} | {'OK' if m['ok'] else 'BAŞARISIZ'} | "
          f"{m.get('size_mb',0)}MB | {m.get('dur',0)}s |")
    ok_dl = [m for m in dl if m["ok"]]
    fail_dl = [m for m in dl if not m["ok"]]
    W("")
    W(f"İndirilen: **{len(ok_dl)}/{len(dl)}** ({', '.join(m['name'] for m in ok_dl)}). "
      f"Başarısız (s1/non-faststart): {', '.join(m['name'] for m in fail_dl) or '-'}.")
W("")
W("## 3) AKTİF-oyun filtresi (ÖNEMLİ dürüstlük notu)")
W("Sabit `-ss` offseti SIK SIK BOŞ sahaya düşüyor (booking-arası / maç-öncesi).")
W("1. tur (sabit offset, ön-filtresiz): 6 indirilen klipten **3'ü BOŞ** çıktı")
W("(Sporland düşük-bitrate boş saha, Atakum + Rize boş indoor dome — görsel+GPU")
W("ile doğrulandı, 0 track), 3'ü AKTİF (Aydınoğlu/Atlantik/Ayazma).")
W("")
W("**Çözüm — `harvest_probe.py` (AKTİF ön-filtre):** her aday tesiste önce 6 kareye")
W("RF-DETR at, median oyuncu>=5 ise tam export. Boş klipte 7dk GPU israfı YOK.")
pm = load("probe_manifest.json", [])
if pm:
    act = [m for m in pm if m.get("status") == "active"]
    pas = [m for m in pm if m.get("status") == "passive"]
    dlf = [m for m in pm if m.get("status") == "download_fail"]
    W("2. tur (probe ile) sonuç:")
    for m in pm:
        if m.get("status") == "active":
            W(f"- AKTİF (median {m.get('median_players')}): **{m['name']}** -> export ✓")
        elif m.get("status") == "passive":
            W(f"- PASİF (median {m.get('median_players')}): {m['name']} -> SİLİNDİ (GPU yok)")
        elif m.get("status") == "download_fail":
            W(f"- indirilemedi (s1): {m['name']}")
    W("")
    W(f"Probe doğrulaması: **{len(act)} aktif** export, **{len(pas)} boş** elendi "
      f"(7dk/boş GPU tasarrufu), {len(dlf)} s1-indirilemedi. Filtre RUNTIME'da çalıştı.")
W("")
W("## 4) Pseudo-label yield (tracklet-kapısı sonrası)")
if yields:
    W("| tesis | çözünürlük | trk kabul | label rows | kare | small_frac | med h/H |")
    W("|---|---|---|---|---|---|---|")
    for n, fs in yields.items():
        W(f"| {n} | {fs.get('img_h','?')}p | {fs.get('tracklets_accepted',0)} | "
          f"{fs.get('label_rows',0)} | {fs.get('n_frames',0)} | "
          f"{fs.get('small_frac',0)} | {fs.get('med_box_h_norm',0)} |")
    W("")
    W("`small_frac` = box_h/H < 0.06 oranı (çözünürlük-NORMALİZE uzak/küçük oyuncu).")
    W("Mutlak piksel DEĞİL (720p'de yakın oyuncu da küçük olurdu → yanıltıcı).")
W("")
W("## 5) Yeniden kurulan COCO dataset")
if stats:
    W(f"- Toplam: **{stats.get('total_images','?')} kare, {stats.get('total_boxes','?')} kutu**.")
    W(f"- Kullanılabilir tesis: **{stats.get('n_venues_usable','?')}**.")
    W(f"- train tesisleri: {stats.get('train_venues')}")
    W(f"- val tesisleri (held-out, overfit guard): {stats.get('val_venues')}")
    fa = stats.get("far_analysis", {})
    if fa:
        W(f"- train small_frac (uzak/küçük): {fa.get('train',{}).get('small_frac')} "
          f"({fa.get('train',{}).get('boxes')} kutu)")
        W(f"- val small_frac: {fa.get('val',{}).get('small_frac')} "
          f"({fa.get('val',{}).get('boxes')} kutu)")
    # önce/sonra
    W("")
    W("**Önce (28 Haz):** 2 kullanılabilir tesis (Çankaya 622 kare + Kıbrıs 135), 757 kare, 7997 kutu.")
    W(f"**Sonra:** {stats.get('total_images','?')} kare, {stats.get('total_boxes','?')} kutu, "
      f"{stats.get('n_venues_usable','?')} tesis.")
W("")
W("## 6) DÜRÜST sınırlar")
W("- Yeni tesisler **base-only** (recover_far=False): per-venue kalibrasyon yok →")
W("  FarBandRecovery in_pitch filtresi güvenilmez. Far-band tiling AÇIK DEĞİL →")
W("  base'in kaçırdığı en-uzak oyuncular bu tesislerde etiketlenmiyor (tam da")
W("  öğretmek istediğimiz zor durum). Çankaya+Kıbrıs tiling'li.")
W("- Pseudo-label **gürültülü**; box-count ile recall İDDİA EDİLMEDİ (geçmişte")
W("  tiling-dedup şişirdi). Recall = insan-GT (recall_eval, 8 kare).")
W("- s1 tesisleri (havuzun ~yarısı) yerelde indirilemedi (remux fix RunPod'da).")
W("- Boş-saha klipleri (Sporland/Atakum/Rize) dataset'e 0 katkı; AKTİF-filtre ile")
W("  gelecek harvest'te elenecek.")
W("")
W("## 7) RunPod reçetesi")
W("`self_training/harvest_runpod.md`: s1-remux fix + 4090'da 10-15× hızlı etiketleme")
W("+ far-band tiling'i no-calib modda açma planı + reorg+train.")
(REPORT / "HARVEST_REPORT.md").write_text("\n".join(lines))
print(f"yazıldı: {REPORT/'HARVEST_REPORT.md'} ({len(lines)} satır)")

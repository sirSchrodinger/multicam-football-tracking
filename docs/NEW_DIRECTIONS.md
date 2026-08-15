# Halısaha Kalibrasyon — Yeni Yönler: Sentez & Karar

**Tarih:** 2026-07-01
**Bağlam:** Tek-kameradan saha kalibrasyonu darboğaz. Detection (RF-DETR) çözülü.
Gerçek kullanılabilir-2D kapsama **%33-36** (121 gece-saha). %47 saha "bowtie/çöküş".
Tükenmiş lever'lar: threshold, rol-swap, köşe-anchor, multi-cand, RANSAC-line,
conv-penalty, bigger-model. Tek gerçek kazanç: VP-recovery (+5 saha, %33→%36).
**Kritik ders:** gate (convex+düşük-residual+interior) GAMEABLE — tek güvenilir
doğrulayıcı GÖZLE bakmak.

---

## 1. Sıralama (promise × feasibility × darboğazı-kırma-şansı)

| # | Yol | Promise | Feasib. | RunPod | Darboğazı kırar mı? | Gate-overfit/hype riski |
|---|-----|---------|---------|--------|---------------------|--------------------------|
| **1** | **2-kamera joint kalibrasyon** | medium | moderate | Hayır (math local) | **EVET — tek sağlam mekanizma** (geometry-limited %47'yi hedefler) | DÜŞÜK (xview residual fiziksel non-gameable, ama görsel zorunlu) |
| **2** | **Calibration-free relative analiz** | medium | moderate | Hayır | Hayır — **etrafından dolanır** (iddiayı düşürür) | DÜŞÜK gate / **YÜKSEK over-claim** (px-mesafeyi "metre" diye satma) |
| 3 | Scaled self-training (noisy-student) | medium | moderate | Evet | Saf hâli HAYIR (zaten 34→34 plato) — kıran varyant = #1'e karışıyor | ORTA-YÜKSEK ("1000 saha" ölçek hayali + plato tekrarı) |
| 4 | Confounder-aware line detector | low | moderate | Evet | Hayır — premise GÖRSEL audit'te çürüdü (1/9 gerçek hedef) | ORTA (sayı yükselirse şüphe) + false-negative tuzağı |
| 5 | Direct calib regression (yeni H-net) | low | moderate | Hayır | HAYIR — zaten RUN edildi, **kötüleştiriyor** (2.94m vs sub-metre) | **YÜKSEK — "güvenli halüsinasyon"** (makul-ama-yanlış saha gate'i geçer) |

**Darboğazı GERÇEKTEN kırma şansı en yüksek: Yol #1 (2-kamera joint).** Tek yol ki
geometry-limited (head-on / zayıf-perspektif) sahalar için *yeni geometrik bilgi*
ekliyor — diğer hepsi ya aynı tek-kare bilgisinden bir şey çıkarmaya çalışıyor
(çıkmaz: bilgi-teorik limit) ya da iddiayı düşürerek etrafından dolanıyor.

---

## 2. En iyi 1-2 yol + neden + somut ilk-adım

### 🥇 Yol #1 — 2-kamera joint kalibrasyon (ASIL bahis)

**Neden:**
- **Sağlam mekanizma:** her kamera KENDİ yakın-yarısında iyi-koşullu. cam1'in
  uzak-üçtebiri = cam2'nin yakın-üçtebiri. Joint solve, her kameranın güvenilmez
  fence-kilitli uzak çizgisini ATAR, iki yakın-yarıyı çapraz-görüş oyuncu-çakışmasıyla
  diker → iki bowtie kamerayı tek iyi-koşullu fit'e çevirir. Bu, her tek-kamera
  lever'ının (threshold/VP/bigger-model) ~%36'da platolaşmasının *gerçek* sebebini
  aşan tek geometrik argüman.
- **Yerel veride zaten kanıtlandı (kolay vaka):** Çankaya cam1+cam2 üzerinde
  3289 eşzamanlı çapraz-görüş oyuncu çifti (1.0m gate), median anlaşmazlık 0.71m;
  iki bağımsız manuel calib ZATEN ~0.8m tutarlı (rigid offset ~0). Homografi 4 nokta
  ister, elimizde binlerce var.
- **Lisans temiz:** joint solve = custom scipy.least_squares (BSD). Detection RF-DETR
  (Apache), tracking orijinal ByteTrack (MIT) / supervision (MIT). boxmot/Ultralytics
  (AGPL) YOK.
- **RunPod gerekmez** (math local GTX 1650 Ti'de koşar; GPU sadece Phase-1 harvest ölçeği için OPSİYONEL).

**Dürüst caveat:** PRIZE vaka (HER İKİ kamera bowtie → joint recovery) henüz
KANITLANMADI — senkron iki-kamera VİDEO sadece Çankaya (kolay saha) için var.
Night-report union 1/10 = "uçlar simetrik çuvallayabilir" gerçek uyarısı.
Gerçekçi tavan (iki-kamera %77 alt-küme): iyimser %35→%55-70, kötümser %35→%40.

**SOMUT İLK-ADIM (bu gece/yarın, CPU, NO RunPod) — PHASE 0 negative-control:**
1. Çankaya'da cam2'yi iyi-anchor kabul et.
2. cam1'i KASTEN boz: H_cam1'i sadece yakın goal-line + center + fence-kilitli bir
   uzak çizgiyle yeniden çöz → `H_cam1_bad` (bowtie simülasyonu).
3. `calib_train/joint_calib_xview.py` yaz (joint_calib.py'yi genişlet). Parametreler:
   H_cam1 + paylaşılan lens k1,k2 (shared_distortion.json). Residual'lar:
   (a) cam1 SADECE yakın-alan çizgi kısıtları (uzak çizgiyi AT) +
   (b) 3289 eşleşen çift için cam1.to_pitch(foot) == cam2 metrik (cam2=anchor).
   scipy.least_squares (BSD).
4. **SUCCESS:** geri-kazanılan H_cam1, manuel cam1 calib'ine ~0.3-0.5m içinde +
   `H_cam1_bad`'den çok daha iyi.
5. **GÖRSEL doğrula:** pitch grid'i cam1 frame'e reproject (reproject_overlay.py) +
   iki kameranın oyuncu haritasını üst-üste (fusion/pitch_map) — hizalanmalı.

> Bu, veri-lojistiğine dokunmadan ÖNCE bootstrap'ı kontrollü vakada kanıtlar.
> Tek en yüksek-bilgi deneyi. Phase-0 geçerse Phase-1 (15-30 gerçek bowtie saha
> harvest, OPSİYONEL RunPod A4000/3090, sadece RF-DETR batch — eğitim YOK) anlamlı.

---

### 🥈 Yol #2 — Calibration-free relative analiz (ürün-hedge, paralel track)

**Neden (farklı eksen — darboğazı KIRMAZ, ürünü AÇAR):**
- **Compute-düşük + ürün-değeri-yüksek:** kalibre ~%36'dan ~%100 sahaya açılır,
  AMA sadece **coarse/ordinal katman**. RunPod yok, CPU.
- **Ölçülü-dürüst (Çankaya homografi-truth'a karşı notlandı):**
  - SAVUNULABİLİR: top-3 NN hit %91.3, per-player Spearman 0.864, kompaktlık-TREND,
    soft-marking (top-2/3), oyuncu-boyu-birimli vicinity-pressure, tempo.
  - SAVUNULAMAZ (ASLA metre diye satma): exact who-marks-whom (%58), metre-mesafe
    (±%56, -1.7m bias), m² hull, head-on'da kesin formasyon etiketi.

**SOMUT İLK-ADIM (CPU, local):**
1. Team etiketlerini tracks'e bağla: `detect/team_split.py` + `team_vest.py`
   (Çankaya + _active_game), `team` kolonu ekle ("unknown" bucket'ı kabul et).
   Cross-team metrikten ÖNCE overlay'de renk doğrula (vest tek-yönlü, memory).
2. `stats/relative_metrics.py` yaz (numpy/scipy, MIT): SADECE savunulabilir katman,
   her çıktıda açık **"ordinal/relative — not metric"** tag'i.
3. **Doğrulama harness'i** (bu oturum prototiplendi, formalleştir): her kalibre sahada
   calib-free metrik vs homografi-truth — NN-agreement / top-3 / Spearman / bias.
   "Metric" kelimesini bu sayılara gate'le.
4. (Opsiyonel lever) calib-LITE: vp_calib.py horizon/VP + box_h ile Criminisi
   single-view-metrology anizotropik zemin-ölçek → derinlik-axis foreshorten düzelt;
   median hata <%30'a düşerse "approx-metric ±X%" iddiasına terfi ettir.

---

## 3. Gate-overfit / hype bayrakları

- 🚩🚩 **Yol #5 (direct regression)** — EN sinsi. Regresyon-prior geometry-limited
  karede "makul-ama-metrik-yanlış" saha üretir, low-residual+convex gate'i geçer =
  dürüst-çöküşü **güvenli halüsinasyona** çevirir. + 4-nokta res=0.000 tuzağı zaten
  kanıtlandı (3/24 saha res=0 → SUCCESS DEĞİL, bowtie). **Yeni-net KURMA.**
- 🚩 **Yol #3 (self-train saf hâli)** — selffeed_kb.log: pool=59, pseudo τ=0.28 → 8
  kare, 4/4 refresh durgun, 34→34 (+0). Confirmation-bias yapısal (student yalnız
  zaten-çözülen sahadan öğrenir). "~1000 saha" = platform ölçeği, yerelde 27 saha;
  harvest rate-limit'i KANITLANMADAN kazanç kurma.
- 🚩 **Yol #4 (confounder)** — 9-saha GÖRSEL audit premise'i çürüttü: 1/9 gerçek
  hedef (Atakum reklam-panosu). Çoğunluk geometrik-çöküş/zayıf-sinyal/karanlık =
  confounder DEĞİL. false-negative tuzağı (touchline=fence-dibi çakışık) NET-negatif
  riski. Değeri kapsamada değil PRECISION'da (yanlış-ama-emin → dürüst-red).
- ⚠️ **Yol #2** — gate-overfit DÜŞÜK ama **over-claim YÜKSEK**: box_h-normalized
  mesafeyi "metre" diye shipping (kanıtlı yanlış ±%56). Kod+UI'da sert ayrım şart.
- ✅ **Yol #1** — en düşük gate-overfit: xview oyuncu-çakışma residual'ı fiziksel
  non-gameable (senkron doğruysa). AMA failure-correlation gizlenebilir → her
  recovery'yi İKİ frame'de GÖZLE (iki cam oyuncuları çakışmıyorsa H yanlış).

---

## 4. Ürün-değeri-yüksek + compute-düşük  vs  yüksek-risk-yüksek-getiri

**Ürün-değeri-yüksek + compute-düşük (güvenli, hemen ship):**
- **Yol #2 (calib-free)** — bugün ~%100 sahaya coarse-taktik açar, RunPod yok,
  ölçülü-dürüst. Darboğazı çözmez ama ürünü beklerken AÇIK tutar.

**Yüksek-risk / yüksek-getiri (asıl darboğaz bahsi):**
- **Yol #1 (2-cam)** — başarılırsa iki-kamera %77 alt-kümesinde çift-haneli kapsama
  (+%20-35 puan); başarısızsa failure-correlation'dan ~0. Phase-0 ucuz kararı.

**Düşük-getiri / yüksek-risk (ÖNERİLMEZ):**
- **Yol #5** kurma (halüsinasyon). Sadece komşu cheap-deney: TVCalib **MIT solver**
  (SoccerNet ağırlık KULLANMA) drop-in swap → solver-brittle birkaç saha kurtarabilir.
- **Yol #4** düşük-öncelik; sadece precision/gate-boşluğu (gate %54 vs göz %33-36)
  daraltmak için, false-negative-güvenli dead-zone ile.
- **Yol #3** saf hâli ÖNERİLMEZ; tek değerli kısmı = cross-cam privileged pseudo,
  o da Yol #1'e karışıyor (oradan al).

---

## 5. NET TAVSİYE — Sıradaki oturumda NE yapılmalı

**İki paralel track, ikisi de CPU/local, RunPod GEREKMEZ:**

1. **ASIL (geometri):** Yol #1 **PHASE 0 negative-control**'ü koş.
   `calib_train/joint_calib_xview.py` yaz → Çankaya cam1'i kasten boz → yakın-yarı
   çizgi + 3289 xview çift ile joint çöz → recovered H_cam1 manuel'e ~0.3-0.5m mı?
   → grid+iki-oyuncu-harita overlay GÖZLE. **Bu tek deney, prize mekanizmasının
   çalışıp çalışmadığını veri-lojistiğine girmeden DECISIVE söyler.** Geçerse →
   Phase-1 (15-30 bowtie saha harvest, RF-DETR batch RunPod opsiyonel) anlamlı.
   Geçmezse → 2-cam'e veri toplamadan dur, kaybı erken kes.

2. **PARALEL (ürün):** Yol #2 coarse-tier'i ayağa kaldır — team-label tracks'e bağla,
   `stats/relative_metrics.py` (sadece ordinal, "not metric" tag) + Çankaya grading
   harness. Bu, geometri çözülürken ürünü ~%100 sahaya dürüstçe açar.

**YAPMA:** yeni direct-regression net (kanıtlı kötü + halüsinasyon). Saf self-train
ölçek (zaten plato). Confounder eğitimi (premise çürük) — en fazla cheap precision-fix.

**Doğrulama disiplini (her iddiada):** sayıya değil GÖZE güven. Phase-0 = grid+çift-
harita overlay; calib-free = Çankaya homografi-truth harness + bowtie videoda
marking-ok eyeball. Gate sayısı tek başına KANIT DEĞİL.

---

### İlgili dosyalar
- Yeni yazılacak: `<repo-root>/calib_train/joint_calib_xview.py`
- Genişletilecek baz: `calib_train/joint_calib.py`, `fusion/fuse_engine.py` (build_tracklets/_coreg_pairs/coregister)
- Veri: `raw/tracks_cankaya_cam1.parquet` (145k/937tid), `raw/tracks_cankaya_cam2.parquet` (179k/1034tid), `calib/fusion_config.json` (55.1s offset+180° flip), `calib_train/shared_distortion.json`
- Yeni yazılacak: `<repo-root>/stats/relative_metrics.py`
- Mevcut: `detect/team_split.py`, `detect/team_vest.py`, `calib_train/vp_calib.py`, `reproject_overlay.py`

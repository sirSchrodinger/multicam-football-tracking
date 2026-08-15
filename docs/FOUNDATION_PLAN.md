# Halısaha foundation sistemi — derin plan (token-sınırsız araştırma temelli)

**Tarih:** 29 Haz 2026. **Yazan:** Claude (Alperen için). **Durum:** 4 paralel literatür-araştırması (tracking/re-ID, ölçekli SSL detection + top, kalibrasyon/ölçek, oyun-durumu/simetri) + canlı veri-ölçümü ile temellendi. Tüm yöntemler lisans-etiketli (ticari ürün → GPL/AGPL/NC dışla).

---

## 0. Neden bu plan — dürüst mevcut durum

Bugüne kadar **tek sahaya (Çankaya) kilitli** derinleştik. Bu yanlıştı. Sahip olduğumuz gerçek varlık tek bir maç değil; **sosyalhalisaha'nın tüm akışı**. Bu planın amacı: sabit bir sahaya takılmadan, **kamera-agnostik bir foundation sistemi** (player + team + ball + pitch detection) eğitip, **kimlik sürekliliğini** çözüp, sonra **sahanın içini okumak** (pas, dizilim, santra/faul/baraj, kaleci rotasyonu, devre-arası saha değişimi).

**Bugün zayıf olduğumuz yerler (itiraf):**
- Detection tek-tesise overfit; far/karanlık recall hâlâ açık; gece ham FP (hayalet) var.
- Kimlik 12dk'da 1034 parçaya bölünüyordu → konsolidasyonla 21'e indirdik (14 core) ama appearance şans-seviyesi, re-ID çözülmedi.
- Kalibrasyon balıkgözü köşelerini içe büküyor; mutlak metre ±13%.
- Takım/top/oyun-durumu yok.

**Kuzey yıldızı:** her sabit kameraya bir-kez otomatik kalibre olan, oyuncuyu maç boyu karıştırmadan takip eden, topu ve takımı veren, ve **insan gibi sahayı okuyan** (santra mı, faul mu, baraj mı) bir sistem.

---

## 1. Veri: 75k maç — CANLI DOĞRULANDI

`scan_cams.py` (xhr/filtre listeleme API + her maçın `videoSrc` dizisi) ile **28 Haz canlı ölçüm**: 22 maçtan **17'si 2-kameralı (~%77)**, 5'i tek. Yani **iki-kamera kural, istisna değil** — çapraz-kamera self-supervision + füzyon (`fusion/`) için altın. Venue'ler tüm Türkiye'ye yayılı (Erzurum, Gaziantep, İstanbul, Dikmen, Rize, Atakum...). URL deseni temiz: `s{1,2}.sosyalhalisaha.com/matches/disk{N}/shst{ID}s{...}/build/{ts}.{cam}-1.mp4`. 15-gün retention → günlük harvest cron şart.

**Ölçek kanıtı:** ~1000 saha × ~4-5 maç/gün × 15 gün ≈ **60-75k maç penceresi**; ~14 kişi/maç × ~75k ≈ **~1M kişi-örneği** (Alperen'in ~900k tahmini doğru mertebede). Bu, her venue için **kişi-boyu dağılımı (çan eğrisi)** ve **popülasyon-ölçekli kalibrasyon** için yeterli N.

**Harvest planı (Faz 0):**
1. Günlük cron: `scan_cams` tüm sayfalar → maç+kamera+venue+URL kataloğu (D1/sqlite).
2. ~1-2 fps frame örnekle; **perceptual-hash dedup** (ardışık-kare çöpünü at); **venue-stratified** örnekle (çeşitlilik > hacim).
3. DINOv2-embedding indexi (sonra diversity/active-learning için).
4. Fisheye'ı **resample ETME** (kenar çözünürlüğü = uzak oyuncu); geometriyi kalibrasyon halleder.

---

## 2. Detection at scale — çoğu-etiketsiz veriden (RF-DETR, Apache)

Araştırma özeti: **fine-tune'da DETR veri-aç DEĞİL** (veri-açlığı sıfırdan-eğitim miti; Deformable/DN/DINO-DETR çözdü). COCO-checkpoint'ten **~200-500 ÇEŞİTLİ kare** doğru mertebede; darboğaz **zor-koşul kapsaması** (far/karanlık/örtülü/fisheye-kenar), ham sayı değil. SSOD literatürü: **~%10 etiketle tam-süpervizyona birkaç mAP yakın**.

**Reçete (tek 4090):**
1. **Sıfır-etiket seed:** Grounding DINO ("person", **Apache 2303.05499**) + COCO RF-DETR-S, **far-third'de SAHI tile** (`obss/sahi` **MIT 2202.06934**, +5-7 AP küçük-obje, retrain'siz = bizim FarBandRecovery'nin prensipli hâli). Pitch-polygon dışını at (taraftar filtresi zaten var).
2. **Tracking ile temizle + zor-örnek madenciliği:** ByteTrack → uzun-track-üyesi kutuları tut; **temporal FN** (track'te boşluk = kaçan far/karanlık oyuncu) + **orphan FP** (1808.04285 Unsupervised Hard Example Mining). Bu hem temiz pseudo-set hem zor-kare havuzu verir → mevcut tracking/player_state'i yeniden kullanır.
3. **Active learning round-1:** zor-havuzu **entropy + temporal-gap** ile skorla, **Core-Set k-center (DINOv2 embedding)** ile çeşitlendir (ardışık-kare-tuzağını kırar). **~300-500 kare elle etiketle** (gece/far/karanlık/fisheye-kenar/iki-takım). 3-kör-sayıcı protokolü (N=8 pilotundaki gibi).
4. **Seed teacher fine-tune:** RF-DETR-S/M (Apache) COCO'dan; held-out venue'de doğrula.
5. **Noisy-Student / self-train loop (DETR-pratik SSOD):** teacher → büyük havuza SAHI+temporal-filtre+**Soft-Teacher conf-ağırlık** ile pseudo-label; student'ı güçlü-aug ile eğit; **2-3 tur, her tur yeni venue ekle**; **Consistent/Dense-Teacher adaptif eşik** (sabit 0.5 değil). DETR-native isterse **Semi-DETR (2307.08095)**.
6. **Far-recall round-2:** student'ı çok-venue'de koştur, en-kötü far/karanlık temporal-FN kümeleri → **~300-500 kare daha** (tam o kayıplarda) → re-fine-tune. Pilot zaten precision~%100/kayıp=far-recall dedi → bütçeyi tam oraya harca. **Oyuncu toplamı ~600-1000 kare.**

**Lisans temiz stack:** RF-DETR Nano-Large + DINOv2 + Grounding DINO + SAHI (hepsi MIT/Apache). **DIŞLA:** YOLO-World/YOLOv8 (GPL/AGPL), RF-DETR XL/2XL (paid Platform License), DINOv3 (custom Meta license).

---

## 3. Top tespiti — heatmap + temporal (WASB, MIT)

Alan konsensüsü: **N ardışık kare → per-kare Gaussian heatmap → motion/trajectory post**. Temporal-stacking far/bulanık/düşük-kontrast topu kurtarır (tek-kare YOLO/RF-DETR kaçırır).

- **Base: WASB-SBDT (MIT 2311.05237)** — HRNet 3-kare heatmap + motion-gated sabit-ivme tracker; **sabit-kamera futbol (ISSIA) pedigree'si** = bizim domain.
- **Eklenti: TrackNetV4 (MIT 2409.14543)** — frame-diff motion-attention, hızlı/bulanık far-top için.
- **BlurBall (CC-BY-SA, kod-DIŞLA)** — blur-şerit-merkezi etiketleme FİKRİNİ al.
- **Etiketleme (ucuz):** ~30-50 maçta insan topu **~0.5-1s'de bir tıklar**, spline-interpolasyon (top hareketi pürüzsüz) → **~2-4k yoğun etiket ~1-2 günde**. WASB ön-geçişiyle insan sadece düzeltir.
- **Self-train:** sabit-ivme motion-model tutarlı tespitleri pseudo-label olarak geri-besle.

---

## 4. Kimlik sürekliliği — #1 darboğaz, "beni maç boyu karıştırma"

**Araştırmanın en önemli bulgusu (paradigma):** Bu bir **re-ID problemi DEĞİL** — bir **dünya-koordinatında-hareket + offline-stitching + takım-kısıtı** problemi. Gece ~50px aynı-formalı oyuncular çakışan appearance-embedding üretir → derin re-ID **takımları ayırır, oyuncuları değil** (bizim ölçtüğümüz 0.59 vs 0.58 tam bu). Kanıt: SportsMOT SOTA sport-tracker'ları **Kalman'ı atıp** appearance-light koşuyor (Deep-EIoU 77.2 HOTA, GTA-Link 81.04). "15dk sıfır-swap" tek modelden ALINMAZ, bir **stack'ten mühendislikle** kurulur. **En büyük kullanılmayan kaldıracımız = kamera SABİT** → bir homografi + ground-plane hız-kapısı + takım-rengi; broadcast yöntemleri yapamaz, biz yapabiliriz. (Bizim `consolidate` zaten pitch-uzayında hız-kapılı çalışıyor → doğru yoldaymışız.)

**Birebir blueprint'ler (önce bunlar):**
- **GTATrack — SoccerTrack 2025 KAZANANI (2602.00484):** sabit fisheye, 22 oyuncu, 4096×1080 — **neredeyse tam bizim problem**. RF/YOLO detector + **pseudo-label ile uzak-fisheye recall** + OSNet + **Deep-EIoU online** + **GTA-Link offline** split/merge. (repo lisanssız → mimari al, kodu yeniden-yaz.)
- **KIST-GSR — SoccerNet 2025 GSR kazananı (2508.19182):** detect → Deep-EIoU+OSNet → çok-kare pitch-calib → **split-and-merge** (identity-tutarsızlığında böl, re-ID benzerliğinde birleştir) → **takım = jersey-renk kümele + küme-x-pozisyon karşılaştır** (gece ucuz+sağlam).
- **From Broadcast to Minimap (2504.06357):** kazancın çoğu **offline fragmentation-azaltan stitch**, online tracker değil. (Bizim uzun-boşluk merge bunu doğruluyor.)

**Adopte edilecek (hepsi lisans-temiz):**
- **Offline stitch — GTA-Link (MIT 2411.08216):** re-ID embedding'de DBSCAN ile **Splitter** (>1 kimlik içeren tracklet'i kes — occlusion-sonrası kirlenmeyi temizler, aynı-formada bile) + **Connector** (aynı-kimlik birleştir). SportsMOT 81.04 HOTA SOTA. **Bizim `consolidate`'in productize hâli** — merge-affinity'sini appearance yerine **pitch-hareket + takım** yap.
- **Gap-filling — GSI (clean-room, StrongSORT 2202.13514):** Gaussian/GP regresyon boşlukta **eğri** yörünge doldurur (bizim düz-çizgi interp yerine) → **mesafe-altsınırını gerçeğe yaklaştırır** (42 m/dk düşük-sayımın kökü düz-çizgi interp'ti). Global link = **min-cost flow (OR-Tools Apache / SciPy BSD)**.
- **Online — OC-SORT (MIT 2203.14360):** **ORU** (kayıp track yeniden-bulununca son-gerçek↔yeni-gözlem virtual-trajectory + KF yeniden-koş = kör-coast drift'i geri-al) + **OCM** (yön iki *gerçek* tespitten, gürültülü-KF değil → kesen oyuncu takım-arkadaşına eşleşmez). + **ByteTrack low-score** kurtarma (50px/sönük tespitleri geri-alır). **NO-ReID.**
- **Height-adaptive gating (SportMamba fikri, clean-room):** kapı-eşiği box-yüksekliğiyle ölçeklenir → near ve far aynı mantık.
- **Jersey-number sparse re-lock (2405.13896 fikri, re-implement):** yalnız **yakın-kamera iyi-ışıkta** OCR → **tracklet-çoğunluk-oyu** → bir güvenli okuma uzun-occlusion'ı köprüler. (Far/gece okunmaz → load-bearing yapma.)
- **Self-supervised re-ID (ikincil, down-weighted):** **OSNet/torchreid (MIT)** kendi auto-label tracklet'lerimizde fine-tune (off-the-shelf ağırlık zayıf transfer + lisans-tainted). + CycAs çevrim-tutarlılığı (2007.07577) yakın-third'de güvenilir pseudo-ID seed.

**Çapraz-kamera bonus (%77 iki-kameralı!):** zıt-uç ikinci kamera = **MUST-link** (aynı oyuncu iki açıdan) → `fusion/` bunu kullanıyor; far-third belirsizliğini yakın-kamera çözer = tek-kamera re-ID bilgi-limitini kıran en güçlü ücretsiz sinyal.

**Tuzaklar:** appearance'a kimlik-bağlama (gece çöker); **Ultralytics YOLO = AGPL** (her iki kazanan kullanıyor — ticari tuzak, RF-DETR/YOLOX kullan); Deep-EIoU/GTATrack/SportMamba **lisanssız** (re-implement); sabit-CV-Kalman çok-saniye occlusion'da takım-arkadaşına coast eder; transformer tracker'lar (MOTRv2/MeMOTR/MOTIP) veri-aç + çok-GPU → **v2'ye ertele**.

---

## 5. Kalibrasyon + ölçek — çok-venue otomatik (TVCalib MIT optimizer + retrain detector)

Araştırma kritik bulgu: yayınlı SOTA (PnLCalib, No-Bells, TVCalib, KpSFR) **broadcast-trained** → *yöntem* transfer eder, *ağırlık* etmez (domain), ve çoğu **GPL/NC**. **En yüksek-kaldıraç: kendi futsal karelerimizde line/keypoint detector'ı RETRAIN.**

**Pipeline (kamera başına bir-kez):**
1. **Distortion + line birlikte:** uzun saha-çizgileri (LSD/retrain) → **division-model (Fitzgibbon CVPR2001)** çizgi-düzlüğü ile (resample YOK, joint-refine için sakla).
2. **Marking detector (retrain):** HRNet keypoint + line/segment head (PnLCalib/TVCalib mimarisi) **kendi futsal karelerimizde** (çizgi, orta-yuvarlak, ceza, kale-çizgisi, **kale direkleri**). 1000-venue'yi çalıştıran ADIM bu.
3. **Template init:** robust DLT+RANSAC futsal-template'e; köşe off-frame ise **circle-correspondence fallback (2504.20052)** + chamfer/score-align (Shi WACV2022).
4. **Joint nonlinear refine:** focal+pose+residual-distortion, segment-reprojection minimize (**TVCalib MIT differentiable objective**). Ground-homografi → focal, tilt, kamera-yüksekliği.
5. **In-plane metric anchor (focal-bağımsız):** **kale-ağzı 3m (futsal!)** + **orta-yuvarlak 3m yarıçap** + **kale direği 2m dikey** cross-check (Criminisi). Touchline-uzunluğu DEĞİL (güvenmediğimiz şey).
6. **Popülasyon boy-ölçeği (Alperen fikri, DOĞRU ama düzeltilmiş):** ayakta-gated oyuncuların head→foot segmenti → metre; **KDE MODE / 2-bileşen mixture** (ORTALAMA DEĞİL — futsal postür-skew'u ortalamayı aşağı-yanlı yapar); mesafeyle ağırlıkla. Bu, alan-geometrisinin veremediği **1 DoF (focal↔yükseklik) ölçek-kapatıcı** + validator. Literatür precedent: surveillance autocalibration from pedestrian-height (BMVC 2011, ECCV-W 2016) — bizim N (~10^5/venue) onların çok üstünde.
7. **İmkânsız-katalog dedektörü:** `S_height/S_goal` hesapla; >%8-10 ayrışma veya implied-dims futsal-aralık dışı (~25-42m × 16-25m) → **"fiziksel-imkânsız katalog" flag** (Çankaya'da katalog/40×20'yi zaten eledik — bu prensipli hâli).
8. **Auto-QA gate:** metric reprojection RMSE eşik + iki-anchor uyumu → kabul; yoksa **trainable-projected-label / manuel** fallback; sonra **LOCK**.

**Lisans:** **MIT TVCalib optimizer + kendi detector**'ımız ana yol; PnLCalib (GPL-2.0) yalnız referans. SoccerNet ağırlıkları ticari-temiz değil → retrain.

---

## 6. Simetri + temporal cycle-consistency — ÜCRETSİZ eğitim sinyali (Alperen'in içgörüsü = makale-değer)

Alperen: "sahalar simetriktir; uzun koşuda yakın ve uzak yarıdaki oyun benzer duruma gelmeli; devre-arası saha değişirler." Bu, literatürde **fixed-camera sports için yayınlanmamış** bir self-supervision çekirdeği:

**(7) Near/far simetri = far-recall kalibratörü + test-time kayıp.** Devre-arası saha-değişimi olan tam maçta aynı oyuncular yakın ve uzak yarıda **istatistiksel-simetrik** zaman geçirir. Yarı-koşullu (count, occupancy, hız, aralık) histogramları maç-boyu topla; **simetri-fark kaybı** = near-half vs far-half(mirror) divergence (EMD/KL). Kusursuz tespitte ~0; pratikte **pozitif ve far-third'de yoğun** = **etiketsiz far-recall açığı ölçümü** (mevcut recall-ölçüm işimize birebir bağlanır). İki kullanım: (i) far-eşik/NMS'i fark kapanana dek kalibre et; (ii) **Test-Time Training (1909.13231; stream 2307.05014) / STFAR (2303.17937)** auxiliary objective'i — detector bu maçta far'ı near'a benzetecek şekilde adapte olur. Reflection-equivariant backbone (EquiSym 2203.16787; TacticAI Nat.Comms 2024), **partial/learned** simetri (2312.12223, perspektif tam-aynayı bozar).

**(8) Forward/backward + frame-skip cycle-consistency = far-kimlik tamiri.** Bölüm 4'teki CycAs + Path-Consistency; yakın-third'in temiz çevrim-tutarlı kimlikleri **hedef** istatistik, simetri kaybı bu kaliteyi far'a taşır. **Simetri recall/density'i, cycle-consistency kimlik-sürekliliğini tamir eder** — far-third'in iki failure-mode'unu yalnız maçın kendisiyle.

**Novelty (yazılabilir 2 katkı):** (a) **pozisyon-only kickoff/baraj spatial-config tespiti**; (b) **near/far istatistiksel simetri = fixed-camera tracking için self-supervised far-recall sinyali**.

---

## 7. Oyun-durumu — pozisyondan (top YOK gerekmiyor)

Alperen: "sahaya bakıp santra mı, faul mu, baraj mı görürsün; üstün AI göremez mi?" Pozisyondan kurulabilir:

- **(P2) Possession/pas/turnover — PathCRF (2602.12080, top'suz!):** tam-bağlı oyuncu-graf, CRF her t'de tek "possession edge" seçer, Viterbi; edge değişimi = pas/turnover. **Top tespitine gerek YOK** — possession-stream her şeyi besler.
- **(P1) Kickoff/santra:** iki-takım halfway'le temiz-ayrık + ≥1 oyuncu orta-noktada + düşük-hız → bir takım çizgiyi geçer. TacticAI reflection-equivariant skor (mirrored-santra aynı skor) + dense-anchor displacement head (2205.10450) ile kesin timestamp.
- **(P4) Baraj/set-piece:** savunan takımda **collinear küme** (2-4 oyuncu, eşit-aralık, kale-top çizgisine dik, ~6-10m). Line-fit-residual × compactness × kale-yakınlığı.
- **(P3) Formasyon/rol/kaleci/saha-değişimi:** per-kare Hungarian rol-atama + EM rol-Gaussian (Bialkowski 2014/16) → EFPI template (2506.23843) → **SoccerCPD change-point (2206.10926)**. **Devre-arası saha-değişimi** = iki takım-centroid x'inin birlikte işaret-değiştirdiği en-büyük change-point. **Kaleci rotasyonu** = en-derin/en-statik rolü tutan oyuncunun change-point'i.
- **(P5) Faul/stoppage:** maça eğitilmiş "ghost"/next-position model (1703.03121; baller2vec 2102.03291); çok-oyunculu eşzamanlı hız-çöküşü + tahmin-hatası sıçraması = dead-ball; isim veremeyiz (top/ses yok) ama **anı lokalize ederiz**.
- **(P6) Tek weakly-supervised graf event-head:** P1-P5 = otomatik weak-label → Sanford trajectory-self-attention (2004.10299) / ARG (1904.10117) sliding-window 14-node graf → {kickoff, restart, wall, open-play, turnover, stoppage}. SoccerNet mAP@tolerance ile raporla.

---

## 8. Takım ayrımı (gece, dinamik yelek)

Mevcut: gece yelek-tek-yönlü güvenilmez. Foundation yardımı: (a) çok-venue retrain → daha iyi torso-crop; (b) **PathCRF possession** + pas-ağı → topu paslaşanlar aynı takım (renkten bağımsız graf-clustering); (c) ilk-gol-kapısı (yelek-değişimi olayı, P-event olarak tespit). Renk tek başına temelsiz kalır; **pas-grafı + possession takımı renksiz çıkarabilir** (yeni yol).

---

## 9. RunPod yürütme — verimli, maliyet-sınırlı

Kanıtlanmış altyapı (29 Haz): headless-Brave CDP + GraphQL pod aç/SSH/terminate, **cost-guard** (BİTTİ/stall/deadline → indir+terminate), env-zinciri çözülü (torch 2.11+cu128). Plan:
- **Lokal Lenovo (1650Ti 4GB):** harvest, dedup, pseudo-label (Grounding DINO/SAHI CPU/GPU), tracking-clean, active-learning seçim, etiket-araçları, CALIB optimize (TVCalib hafif), oyun-durumu (pozisyon-only, GPU'suz). Yani **çoğu iş lokal**.
- **RunPod 4090 ($0.34-0.69/sa):** yalnız **eğitim** — (1) RF-DETR self-train turları, (2) marking-detector (calib) train, (3) WASB top-model train. Her iş cost-guard'lı, biter-bitmez terminate. Tahmini: detection 2-3 tur × ~$2 + calib ~$2 + ball ~$2 ≈ **<$15 toplam**.
- **Disiplin:** her run held-out-venue gate'i geçmeden promote yok; açık-pod bırakma yok (Alperen #1 kural).

---

## 10. Fazlı yol haritası (dürüst milestone + ölçüm)

| Faz | İş | Çıktı / ölçüm | Süre |
|---|---|---|---|
| **0** | Harvest cron + dedup + venue-stratified sampling | katalog (~75k maç), embedding index | 2-3 gün |
| **1** | Player detection foundation (seed→self-train×3, 2 etiket-turu) | far-recall ↑, çok-venue held-out gate | 1-2 hafta |
| **2** | Kimlik re-ID self-train (CycAs+Path-Consistency) + füzyon | 14 kararlı kimlik, swap↓, far-fragment↓ | 1-2 hafta |
| **3** | Auto-calib (retrain marking + TVCalib + boy-mode ölçek) | çok-venue otomatik H + ölçek-CI + imkânsız-katalog flag | 1-2 hafta |
| **4** | Ball (WASB + interpolated-click + self-train) | top trajectory, possession-ready | 1 hafta |
| **5** | Oyun-durumu (PathCRF possession + kickoff/wall/role/side-swap) | event-stream, watchable annotated replay | 1-2 hafta |
| **6** | Simetri-SSL + cycle-consistency (far-recall + kimlik tamiri) + makale | öz-süpervize far-iyileşme, 2 novelty yazısı | sürekli |

**Birincil sıralama (kaldıraç):** veri → player far-recall → kimlik süreklilik → calib/ölçek → top → oyun-durumu; simetri-SSL tüm aşamaları besler.

---

## 11. Lisans hijyeni (ticari)

**KULLAN (MIT/Apache):** RF-DETR Nano-Large, DINOv2, Grounding DINO, SAHI, WASB-SBDT, TrackNetV4, TVCalib, KpSFR, MI-AOD/CALD, Probabilistic-Teacher/LODS, Soft/Consistent/Dense-Teacher, Unbiased-Teacher, Noisy-Student/STAC.
**DIŞLA:** YOLO-World/YOLOv8/boxmot (GPL/AGPL), PnLCalib/No-Bells (GPL-2.0, yalnız referans), Adaptive-Teacher (CC-BY-NC), BlurBall-kod (CC-BY-SA), orijinal TrackNet/V2 (lisanssız), DINOv3 (custom Meta), RF-DETR XL/2XL (paid), SoccerNet-türevi ağırlık (ticari-temiz değil → retrain).

---

## 12. İlk somut adım (onay sonrası)

Faz 0 + Faz 1 başlangıcı, hepsi lokal, RunPod yalnız eğitimde:
1. `scan_cams` → tam katalog + harvest cron (dry-run önce, [[feedback_watcher_dryrun_first]]).
2. ~20-30 çeşitli venue'den stratified frame örnekle + dedup.
3. Grounding DINO + SAHI sıfır-etiket pseudo-label + pitch-filtre + tracking-temizle.
4. Active-learning ile ~400 kare seç → etiketle (3-kör-sayıcı).
5. RF-DETR fine-tune (RunPod, cost-guard) → held-out venue far-recall ölç.

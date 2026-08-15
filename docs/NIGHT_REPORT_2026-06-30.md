# Halısaha gece otonom run — 30 Haz 2026 (Alperen uyurken)

Mandat: GPU+RunPod sınırsız, yaratıcılıkla loop, dürüstlük > övgü. Asıl hedef: "her sahayı
otonom 2D'ye çeviren motor"u ileriye taşı + Çankaya-ötesi genelleme kanıtla.

## TL;DR (dürüst)
1. **Detection GENELLİYOR** — ft-detektör (RF-DETR) yeni Türk sahalarında oyuncuları temiz yakalıyor.
   Çankaya-only değil. Kanıt: çok-saha grid (`cand/_GENELLEME_GRID*.jpg`).
2. **Asıl darboğaz = auto-calib.** Keyfi gece sahasının sadece **%33'ünde** geometrik-temiz otonom 2D çıkıyor
   (sanity-pass, 121 saha). residual-coverage (%54) YANILTICI — bowtie'leri sayıyor.
3. **Failure dökümü:** bowtie/çöküş %47 (dominant) · çizgi-yok %11 · kara-kare %7. Çözülemez-taban ~%18 → tavan ~%82.
4. **Crop-aware 1024 retrain (RunPod, $0.22): PLATO.** Val far-IoU yükseldi (goalF 0.03→0.31) ama gerçek
   kapsama DEĞİŞMEDİ (%33.1→%33.1 full, %30.6 tiled). → far-çizgi IoU ≠ doğru calib. KEEP-BEST: seg2_hr2 kaldı.
5. **3 ucuz lever dürüstçe ELENDİ:** threshold-recovery, rol-swap, köşe-anchor-solver — hiçbiri bowtie'yi çözmedi.
6. **Çok-aday far-çizgi seçimi: GATE-OVERFIT (REDDEDİLDİ).** İlk bakışta %33→%46.3 (32 recovery) gibi
   göründü AMA GÖRSEL DOĞRULAMA sahteyi ortaya çıkardı: reprojekte iç-işaretler (merkez-yuvarlak/ceza-sahası)
   gerçek çizgilere oturmuyor. 9 combo deneyip gate-geçeni seçmek = confirmation-bias. SHIP EDİLMEDİ.
7. **İç-consistency doğrulayıcı: BUST.** "Reprojekte iç-işaret tahmin-iç-kanala oturmalı" prensibi iyi ama
   model iç-kanalları (center/box/circle) zayıf tahmin ediyor → skor iyi'yi sahte'den ayıramadı.
8. **★ TEMPORAL AGREGASYON: GÜÇLÜ POZİTİF — YENİ SAHALARDA DA GENELLİYOR (path forward kanıtlandı).**
   N-kare seg temporal-median (statik çizgi pekişir, oyuncu/gürültü silinir). ÇOK-SAHA kanıtı:
   | saha | tek-kare res | temporal | |  Çankaya 1.38→**0.23** ✓6× | Aydınoğlu(yeni) 0.71→**0.33** ✓ |
   Aymakoop(yeni) 1.79→**0.76** ✓gürültülü→temiz | Pınar(yeni) bowtie→bowtie ✗.
   **DÜRÜST DÜZELTME (11-saha canlı batch, küçük örneğe güvenme):** temporal-PASS = **3/11 = %27** (tek-kare
   sanity-pass ~%33 ile BENZER). Geçtiği sahalarda residual **0.89m→0.20m (4.4×)**, 5/11'de iyi. YANİ temporal'in
   değeri = çözülebilir sahalarda KALİTE/GÜVENİLİRLİK (4×), **kapsama SIÇRAMASI DEĞİL** — rastgele gece sahalarının
   çoğu yine bowtie/geometrik-limitli, temporal çözmüyor. Erken 4-saha örneği (3/4) İYİMSERDİ. Zor-saha kapsaması = 2-kamera.
   **ALTYAPI WIN:** klip-extraction remote-mp4'te patlıyor AMA farklı-zaman tek-frame çekimi ÇALIŞIYOR →
   temporal calib deploy'da uygulanabilir (frame örnekle, klip gerekmez). get_clip yerine bu yol.

## ⚠️ FRAME-ÖRNEKLEME GÜRÜLTÜSÜ (kritik dürüstlük caveat'i)
Temporal calib, hangi ~10-12 sparse-kareyi çektiğime DUYARLI: aynı saha (Kıbrıs) bir koşuda cam0 PASS 0.307,
başka koşuda cam0 FAIL. → **tüm kapsama yüzdelerim (tek-kare %33, temporal %27) ±gürültü taşıyor; kalitatif
bulgular sağlam ama kesin % oynak.** Stabil sayı için DAHA YOĞUN örnekleme (30-60 kare) + tutarlı timestamp gerek.

## 2-KAMERA KAPSAMA testi (INCONCLUSIVE — gürültülü)
10 iki-kameralı venue, her iki kamerayı temporal-kalibre: cam0 0/10, union 1/10 (Garden Park cam1 rescue 0.564).
AMA cam0 0/10 yukarıdaki frame-gürültüsüyle çelişiyor (canlı batch %27'ydi) → bu koşu şanssız-örneklemli, GÜVENİLMEZ.
TEK gerçek sinyal: Garden Park cam0-fail→cam1-PASS = 2-kameranın zor-sahayı kurtardığı 1 vaka (hipotez zayıf-destek).
DOĞRU 2-kamera çalışması = yoğun-örnekleme + çok venue + cam-eşleştirme (gelecek iş). [[project_halisaha_2cam_reid]] mevcut füzyon altyapısı.

## ★ WORKFLOW KAZANCI: VP-recovery katmanı (6-ajan, adversaryal-doğrulandı, 1 Tem)
6 ajan paralel 6 calib-robustluk yöntemi prototipledi + adversaryal görsel doğruladı (docs/CALIB_ROBUSTNESS_WF.md):
- **KAZANAN: `vanishing_point`** (calib_train/vp_calib.py::calib_vp_strict) — far-rol için çok-aday + VP-geometri
  prefiltre (touchline'lar saha-DIŞI VP'ye yakınsamalı; fence'e kilitli far-line VP testini GEÇEMEZ) + sıkı IC≥0.45 gate.
  **DOĞRU ENTEGRASYON: baseline-PASS aynen korunur (REGRESYON YOK), SADECE baseline-fail'de recovery dene.**
  → **44/121 (~%36) vs baseline ~%33, +5 GENUINE recovery** (idx 6/26/58/60/118: AdaArena/AydınOral/Esenpet/FatihOktay/İfaSpor).
  KENDİM görsel-doğruladım (cand/_VP_RECOVERIES.jpg): mor ceza-sahaları gerçek beyaz kutulara oturuyor (non-gameable sinyal).
- **5 yöntem ELENDİ (görsel-doğrulamada gate-overfit/sahte):** ransac_lines (fence'e kilit, -kayıp), multicomp_interior
  (67 ham ama 0 genuine), ellipse_anchor (çember yok→halüsine), convex_penalty (convex-ama-yanlış), conf_weighted (marjinal).
  DERS: gate (convex+residual+IC) GAMEABLE; sadece VP yanlış-korespondansa (fence-lock) dokunuyor. interior_consistency'nin KENDİSİ de gameable.
- DÜRÜST SINIR: kazanç mütevazı (+5); bowtie'lerin çoğu (~%50) hâlâ çözülmedi (VP sadece non-fence far-line + VP-plausible olanı kurtarır).

## BIG model (BASE48 1280, RunPod 2. deney) — BAŞARISIZ + IC-gameability tuzağı ifşa (1 Tem)
seg2_hr_big.pth (BASE48 1280, 12000 step, ~$0.20). hr2+VP=44'e karşı **big+VP HAM 92/121 (%76) GÖSTERDİ** —
ama GÖRSEL DOĞRULAMA SAHTE: 58VİP kara-kare fit=2.57 "geçti", pembe çember boş çimde, fit 1.1-2.6 berbat residual'lar.
**KÖK: büyük model iç-kanalları (center/box/circle) daha YOĞUN tahmin → interior_consistency ŞİŞİYOR → IC≥0.45 gate
geometrik-yanlış calib'leri geçiriyor.** = IC gate MODEL-BAĞIMLI + gameable. KEEP-BEST: **seg2_hr2 KALIR, big ELENDİ.**
DERS: hem crop-aware hem BASE48 RunPod deneyi darboğazın compute/capacity OLMADIĞINI kanıtladı. Ve %76'yı raporlamadım —
şüpheci görsel-doğrulama yakaladı (her yüksek-sayı gate-overfit). VP-recovery'nin güvenilirliği hr2'nin iç-kanal-kalibrasyonuna bağlı.

## ★★ 2-KAMERA ÇAPRAZ-KALİBRASYON POC — ÇALIŞIYOR (1 Tem, "başka yol")
calib_train/night/twocam_joint_poc.py (Çankaya cam1+cam2 lokal). cam1'i SADECE cam2-metrik-oyuncu-konumu +
cam1-PİKSEL'den kalibre ettim (cam1 ÇİZGİSİ KULLANMADAN). Çapraz-kamera H, cam1-FINAL(çizgi) H ile **median 0.71m, p90 1.37m** uyuşuyor (175 korespondans, RANSAC 45 inlier).
**MEKANİZMA KANITLANDI: çizgisi bowtie/fence-kilitli kamera, partner kameradan (zıt-uç) + ortak oyunculardan kalibre olur.** Venue'lerin %77'si 2-kameralı → single-camera'nın çuvalladığı geometri-limitli sahalara DOĞRUDAN çözüm.
- DÜRÜST: POC eşleştirme için cam1-calib kullandı (oracle); deployment'ta MATCH-FREE gerek = cam2-metrik-anchor'lara cam1'i robust point-set registration (RANSAC homografi, candidate match'ler + temporal-tutarlılık + flip-prior). Çözülebilir, sıradaki adım.
- 0.71m heatmap/zone için kullanılabilir; precision için daha çok inlier/temporal-ortalama. fusion/ paketi (pitch_map, fuse_engine, 55.1s+flip) mevcut altyapı.
- **MATCH-FREE testi (xview_icp.py): SAF-POZİSYON ICP DIVERGE (7.78m)** — 11 oyuncu arası korespondansı pozisyon-only bulmak belirsiz. → çözüm: fusion'ın APPEARANCE+temporal matching'i (3289 çift zaten var) korespondans verir → çapraz-görüş calib. Saf-pozisyon yetmez.
- **NEW_DIRECTIONS.md (5-yol Workflow sentezi): 2-cam joint = #1 (tek yeni-bilgi ekleyen). Diğer 4 elendi:** calib-free(over-claim), self-train(plato), confounder-detektör(premise çürüdü: far-fail çoğu fence değil head-on), direct-regression(2.94m kötü). DÜRÜST AÇIK PRIZE-vaka: HER İKİ kamera bowtie→joint-recovery KANITLANMADI (sadece Çankaya=kolay 2-cam video var); failure-correlation riski (zıt-uç simetrik çuvallayabilir, union 1/10 uyarısı).
- **SONRAKİ (next session): Phase-0 negative-control** (cam1'i near-line+fence-far'a boz → joint-recover via near-lines+cross-view) + 15-30 GERÇEK bowtie 2-cam venue harvest (synced frame-burst+ByteTrack) → joint vs single, HER recovery GÖRSEL-doğrula. Plan: docs/NEW_DIRECTIONS.md.

## ★ 2-CAMERA BUILD — Phase-0 negative-control GERÇEĞİ (1 Tem, "kaliteli yap")
Joint cross-view solver yazıldı (fusion/joint_calib_xview.py). Çankaya Phase-0 negative-control:
- (a) SADECE near-half çizgi: **0.13m (uzak-yarı dahil!)** — near-half çizgiler tek başına TAM kalibre ediyor.
- (b) SADECE cross-view: 0.85m (gürültülü). (c) near+cross: 0.19m (cross-view HAFİF KÖTÜLEŞTİRDİ).
**REVİZE: near-half çizgiler iyiyse cross-view GEREKMİYOR → 2-camera değeri DAR** (sadece near-half DE bozuksa,
o da partner-bozukla korele = failure-correlation, agent'ın union 1/10 uyarısı). POC'taki 0.71m kolay-vaka.
- **Near-half tek-kamera fix testi (üst-%30 kes, bowtie'lerde): 3/83 = negligible** (ve sahte-risk) →
  bowtie fields "fence-locked far + iyi near" DEĞİL, GERÇEKTEN geometri-limitli (head-on/zayıf-perspektif).
- **NİHAİ DÜRÜST HARİTA:** ~%50 saha fundamental geometri-limitli — tek-gece calib fix'i YOK. Kazançlar:
  VP(+5, fence-subset) + temporal(kalite, noise-subset) + 2-cam(dar, prize-vaka kanıtsız). Geometri-limitli çoğunluk
  ya iyi kamera-yerleşimi (kontrol-dışı) ya FARKLI sensing ister. → ÜRÜN: 2-katman (kalibre-olan ~%40 tam-metrik-2D +
  kalan calibration-FREE relative analiz, DÜRÜST-etiketli). Calib'i daha fazla zorlamak bilgi-teorik limit.

## ★★★ VISION-GROUNDED KALİBRASYON (1 Tem, Alperen kırılma-içgörüsü) — KAZANÇ
"Otomatik imkansız" demek YANLIŞTI: GÖREN ajan (insan/multimodal) standart-şablonu oturtabiliyor.
- **vision_calib.py** (4 köşe + ortak-lens -> homografi). Gaziantep (otomatik-bowtie) -> sanity-PASS.
- **Workflow (14 bowtie sahaya ajanlar GÖZLE baktı): 14/14 dış-sınır SANE (otomatik %0!), 7/14 tam-precision.**
  Ajan katma-değeri: ORYANTASYON (goller sol/sağ-mı üst/alt-mı) — seg'in yapamadığı. docs/VISION_CALIB.md.
- "rough 7" kök-neden = ortak-lens her sahanın fisheye'ini modellemiyor (köşe değil). **vision_calib_joint.py**
  (k1,k2 serbest + merkez-yuvarlak kısıtı + lens-prior reg) bunu hedefler; normal corner-cam'de iç-işaret oturur,
  AMA 10Numara=CENTER-cam dejenere'de + kaba-manuel-okumada tam oturmadı (hassas annotation ister).
- **ÜRÜN SONUCU: sane-outer kalibrasyon oyuncu-konumu/heatmap/zone için YETERLİ.** Vision kapsamayı ~%36'dan
  bowtie'lerin %100-sane / %50-tam'ına çıkarır. Sınır YÖNTEMDEYDİ (bilgide değil) — Alperen haklıydı.
- SONRAKİ: vision-WF'i circle-annotation + joint-lens ile re-run (corner-cam rough->good); 1000s saha için
  Alperen-etiketler / ben-ajan-etiketler / yarı-otomatik-köşe-detektör. cand/_EYECAL,_BEST_AUTO,_JOINT_CMP.jpg.

## Metrik dersi (önemli)
residual<0.45 "coverage" = optimizer kısıtı sağladı demek, calib DOĞRU demek DEĞİL (10-DOF overfit).
GERÇEK metrik = SANITY-PASS (convex quad + alan + PCA + kara-kare reddi). Bundan sonra bu kullanılmalı.
Ek ders: SANITY bile YETMİYOR (multicand gate-overfit etti) → asıl validator iç-işaret hizası AMA model-iç-kanalı
zayıf → tek güvenilir yol = TEMPORAL (çok-kare denoise) + insan-onaylı görsel.

## Bowtie kök-sebep (kanıtlanmış)
Far-kanallar ATEŞLİYOR (goalF_max~1.0) ama far-çizgi MİS-LOKALİZE → 4 çizgi dejenere quad → bowtie.
Çok-nedenli: (a) largest-component fence/duvar'a kilit (→ multi-cand çözüyor), (b) head-on/zayıf-perspektif
tek-kareden ill-conditioned (→ çözülemez, TEMPORAL gerekir).

## Path forward (KANITLANMIŞ öncelik sırası)
1. **★ TEMPORAL calib (KANITLANDI, productionize et):** N-kare temporal-median seg → calib. Çankaya'da
   1.38m→0.23m. Deploy'da video zaten var. EN YÜKSEK GETİRİ, RİSKSİZ. auto_calib.build_static_line_map mevcut.
2. **head-on/zayıf-perspektif sahalar:** tek-kameradan ill-conditioned → 2-kamera füzyonu [[project_halisaha_2cam_reid]]
   veya merkez-yuvarlak elips-fit ek kısıt (ölçek+yön çapası).
3. **multi-cand: ŞIP ETME** (gate-overfit). Eğer denenirse iç-işaret-validatörü + insan-onay ŞART.
4. Detection zaten çözülü → enerjiyi calib'e ver.

## Para / altyapı
RunPod $8.13→$7.90 (1 pod, 3090, 62dk, iron-guard auto-terminate). Brave-CDP erişim fix. Pod OFF doğrulandı.

## ★ CAPSTONE: TAM-OTONOM video → 2D (SIFIR insan girdisi)
- `cand/_NV_CAPSTONE.jpg` — **YENİ SAHA Aydınoğlu (Çankaya-DIŞI):** temporal AUTO-calib (res 0.36m PASS) +
  ft-detektör → **14 oyuncu 2D**. Oyuncular saha-içi makul, kamera-köşesi tutarlı. = Çankaya-only DEĞİL kanıtı.
- `cand/_AUTONOMOUS_CAPSTONE.jpg` — Çankaya gerçek maç: temporal AUTO-calib (res 0.23m) + 11 oyuncu → 2D.
Modül: `calib_train/temporal_calib.py::temporal_calibrate(video,seg_fn,N)`. End-state vizyonu çalışıyor (gürültü-limitli sahalarda).

## Artefaktlar
`cand/_AUTONOMOUS_CAPSTONE.jpg` (★ tam-otonom) · `_TEMPORAL_DEMO.jpg` · `_GENELLEME_GRID*.jpg` (detection genelleme) ·
`_VERIFY_RECOVERED.jpg` (multicand-sahte kanıtı) · `_CTRL_GOOD_OVERLAY.jpg` · `_SEGCHAN_DIAG.jpg` (far-çizgi tanı) ·
`_truecov_*.json` (metrik) · `calib_train/night/` (tüm probe+eval) · `calib_train/temporal_calib.py` + `multicand_calib.py` + `interior_check.py` · `scratchpad/autorun/PROGRESS_NIGHT.md`

# PAS & KONUMSAL DEĞERLENDİRME MİMARİSİ (Cephe E)

29 Haz 2026. Nihai hedef (Alperen): mevcut 2D-replay pipeline'ından → **(a) PAS
istatistiği** + **(b) KONUMSAL / dizilim değerlendirmesi**. Bu doküman somut yolu,
hangi mevcut modülün neyi beslediğini ve **dürüst bağımlılık grafiğini** verir.

Tek cümlelik tez: **Konumsal analitiğin büyük kısmı TOP'suz, takım-etiketsiz, GPU'suz
BUGÜN yapılabilir** (`continuous_state` + kalibrasyon yeter). **Pas ise top + takım
gerektirir** ve bu iki bağımlılık zincirinin EN UCUNDADIR. O yüzden ürün sırası
konumsal-ÖNCE, pas-SONRA.

---

## 0. Mevcut sağlam taban (bunların üstüne kuruluyor)

| Modül | Üretir | Durum |
|---|---|---|
| `calib/cankaya_cam2_v2.json` + `stats/replay2d.py:load_calib/project` | piksel→metre (34.0×18.0 m, far-reproj 7px) | DONE, QA-geçti |
| `detect/track_robust.py` + `track_stitch.py` | swap-onarımlı, boşluk-köprülü `player_id` | DONE |
| `detect/player_state.py:continuous_state` | her core oyuncu, aktif span'inde **her frame** (x,y,status,conf) | DONE |
| `stats/replay2d.py:postprocess` | teleport-red + Savgol + boşluk-kapılı interp → (pid,frame,seg,x,y,status) | DONE |
| `stats/topdown_stats.py:accept_calib_qa` | bozuk-calib fail-closed kapısı | DONE |
| `fusion/fuse_engine.py` | 2-kamera zıt-uç füzyonu (uzak-1/3 recall'ı LOKAL kapatır) | DONE |
| `detect/team_split.py` | `team_id`+`team_conf` ekler — **GÜVENİLMEZ** (REFUSE yolu) | kısmi |

Ölçülmüş recall (insan-GT, N=8 kare): base ~%73, +far-tiling ~%82, precision %100.
Kaçanlar **sadece far-üçte-bir** (near/mid %100). Bu, konumsal metrik için kritik:
şeklin uzak kenarı eksik-örnekli → `continuous_state` conf<1 dolgu ile kapatır,
eksik kalanlar `observed_frac` floor'unun altında grileştirilir.

---

## 1. TOP TESPİTİ — pas için ŞART mı? Top'suz proxy ne kadar gider?

### 1a. Top'suz proxy ÖLÇÜLDÜ ve REDDEDİLDİ (uydurma değil, kanıt var)
`docs/BALL_NOTES.md`: oyuncu-yerleşiminden top-çıkarımı denendi.

| proxy | frame-frame hız (medyan/p90) | sonuç |
|---|---|---|
| hız-ağırlıklı oyuncu merkezi | 38 / 90 m/s | çok gürültülü, zıplıyor |
| hız-yakınsama (topa koşan ışınları kes) | 110 / 322 m/s | fiziksel-dışı |

Gerçek top: pas ~15-25 m/s ama trajektori PÜRÜZSÜZ. Proxy'nin sıçraması bunun 2-15
katı → top sinyali off-ball koşu gürültüsünün ALTINDA (herkes topa koşmaz: boşa koşu,
markaj, geri çekilme yakınsamayı bozar). **Sonuç: top-proxy'den PAS sayılamaz.**

### 1b. Pas için top ŞART (tek cümle): EVET
Pas = topun sahibinin değişmesi. "Sahip" topun fiziksel konumu olmadan tanımsız.
Top'suz en fazla **possession-territory pressure** (ağır zaman-yumuşatmalı oyun-yoğunluk
bölgesi) çıkar — bu PAS DEĞİL, "oyun nerede oynanıyor" düşük-bilgi haritası. Onu da
"top" diye etiketlemek `BALL_NOTES.md`'deki sahte-proxy yasağını çiğner. Dürüst etiket:
*territory_pressure (top değil)*.

### 1c. Top tespit yolu (DETECTION_ROADMAP.md #4)
- **Hızlı ara-filtre (BUGÜN, 0 GPU):** RF-DETR kutularında `h/w < 1.3` olanları
  "insan-değil" say → "top insan sanıldı → teleport" bug'ını (ölçülmüş ~457-671 FP kare)
  hemen düşürür. Bu top'u BULMAZ, sadece yanlış-pozitifi temizler.
- **Stage 1 (etiketsiz):** RF-DETR `ball` sınıfı düşük-eşik top-K + far-band tiling,
  UNION sabit-kamera arka-plan-çıkarma blobları, homografi top-boyu prior ile kapılı;
  Viterbi/DP trajektori + açık "top-yok" durumu; havadayken
  `metric_status='airborne_unreliable'`.
- **Stage 2 (SADECE Stage-1 tavanı düşükse):** WASB-SBDT (MIT) ayrı heatmap top-kafası,
  1.5M param 4GB-ucuz, player-dedektöründen compute çalmaz. RunPod gece-etiketleme gerekir.
- **Gece/çözünürlük zorluğu:** top bu çözünürlükte ~birkaç piksel, çoğu karede görünmez.
  Bu yüzden top katmanı **fail-closed**: `eval/ball_eval.py` (yazılacak) görünür-recall +
  precision + FABRİKASYON-AUDIT (insan-görmediği span'de üretilen konum ~0 olmalı).
  precision ≥ ~%90 değilse BETA/seyrek veya REFUSE.

**Karar:** top, konumsal analitiğin DEĞİL, sadece PAS + topla-temas metriklerinin önkoşulu.
Konumsal işi topa bağlama.

---

## 2. PAS OLAYLARI — çift-kapılı (top VE takım)

### 2a. Tanım (top + takım varken)
1. **Ball-carrier zaman serisi:** her frame topun en-yakın oyuncusu (kalibre metre-uzayında,
   `continuous_state` konumları + top konumu). Histerezis + min-süre ile gürültü kır.
2. **Possession-change olayı:** carrier `player_id` değişir.
3. **PAS** = carrier değişti **VE** yeni carrier AYNI takım **VE** arada rakip-teması yok
   (top rakip oyuncu yarıçapından geçmedi). Aksi → **kayıp/top-çalma (turnover)**.
4. Türevler: pas sayısı, isabet %, yön (ileri/yanal/geri, X-ekseni), ilerletici pas
   (kendi yarıdan rakip yarıya), pas uzunluğu, pas ağı (kim→kim matrisi).

### 2b. Bağımlılıklar (ikisi de zor)
- **Top** (Bölüm 1): RunPod + etiketleme, fail-closed.
- **Takım-etiketi** (`detect/team_split.py`): otomatik renk GÜVENİLMEZ
  (`docs/TEAM_NOTES.md`: yelek-sarı ≈ çim-hue, karışık-kıyafet takımının ortak imzası YOK,
  velocity-7-7 stabilite ~%57 = artefakt). Çözüm: **tek-seferlik manuel-seed** (~14 tık/maç)
  VEYA tek-yönlü-yelek modu (sadece yelekli takımı güvenle bul, gerisi "diğer").
  Takım YOKSA → pas SAYILAMAZ (pası turnover'dan ayıramazsın).

### 2c. Ara-ürün: top VAR, takım YOK
Top tespit edilip takım yoksa **honest touch-timeline / carrier-sequence** çıkar:
"top sırayla şu anonim oyunculardan geçti" — pas/turnover ayrımı OLMADAN. Bu hâlâ
satılabilir (topla-temas sayısı, oyuncu-başına topta-süre). Takım gelince pas-ağına yükselir.

### 2d. Top'suz tavan (dürüst sınır)
Top yoksa pas YOK. En fazla `territory_pressure` (2c'siz, ağır-yumuşatma) +
possession-proxy bölgesi. Bunu **PAS diye sunma** — `BALL_NOTES.md` yasağı.

---

## 3. KONUMSAL / DİZİLİM — top GEREKMEZ (asıl kaldıraç, ŞİMDİ)

Hepsi `continuous_state` (veya `replay2d.postprocess`) + kalibrasyondan türetilir.
İkiye ayrılır: **takım-agnostik** (hiçbir ek bağımlılık yok, BUGÜN) ve
**takım-koşullu** (takım-etiketi gerekir, ama TOP gerekmez).

### 3a. Takım-AGNOSTİK — BUGÜN, 0 GPU, ölçüldü ✅
Aşağıdakiler 22s pencerede `bot_raw_tracks` üstünde GERÇEKTEN hesaplandı
(`stats/replay2d.postprocess` çıktısından, ~8 satır scipy):

| metrik | ölçülen değer (bu klip) | nasıl |
|---|---|---|
| sahadaki ort. oyuncu | 12.9 (gerçek ~14, far-recall açığı) | frame başı nokta sayısı |
| topluluk genişliği (Y) | 14.8 m (p10-p90 13.4-16.2) | `y.max-y.min` |
| topluluk derinliği (X) | 27.5 m | `x.max-x.min` |
| dışbükey-zarf alanı | 253 m² (151-337) | `scipy.spatial.ConvexHull.volume` |
| en-yakın-komşu aralığı | 3.83 m | `pdist` min |

Bunlara eklenecek (aynı veri, ek satır): topluluk-centroid, kompaktlık indeksi
(zarf/oyuncu), dağılım (centroid'e ort. uzaklık), bölge-occupancy (def/orta/hücum
üçte-birleri X-bandıyla), tempo (SADECE observed, conf-ağırlıklı hız —
`status=='observed'` filtresi, interp/pred hıza GİRMEZ), Voronoi alan-kontrolü
(`scipy.spatial`, hücreler saha-poligonuna yarı-düzlem kırpılı), occupancy heatmap
(zaten `player_state.occupancy_heatmap` var).

**Dürüstlük:** her metrik `observed_frac` ile etiketli; floor altı frame'ler grileştirilir;
ölçek `rel_m` band'iyle (±%13) sunulur, metre/sprint ASLA exact-mod dışında. Bunlar
**recall-toleranslı** — far'da 1-2 oyuncu eksik olsa bile şekil-trendleri sağlam
(`continuous_state` conf<1 dolgu ile kapatır).

### 3b. Takım-KOŞULLU — takım-etiketi gerekir (TOP gerekmez)
Takım `team_id != -1` olunca (manuel-seed VEYA tek-yönlü-yelek):
- per-takım centroid + iki centroid arası mesafe (takım-ayrımı)
- **defans-hattı yüksekliği:** takımın en-geri saha-oyuncusunun X'i (zaman serisi)
- per-takım genişlik / kompaktlık / hat-arası mesafe (defans-orta-hücum)
- **pressing PROXY:** topa-sahip-değilken takım ne kadar ileri+kompakt (top'suz olduğu
  için PROXY; gerçek pressing top gerektirir → açıkça "proxy" etiketli)
- zone-occupancy per-takım, blok-yüksekliği

**Kapı:** takım-etiketi başarısızsa (REFUSE) bu blok `team_id=-1` → **DEFERRED**, `balance_note`
ile. Sahte 7-7 ile ASLA üretilmez (`TEAM_NOTES.md` değişmezi). Bu blok **GPU/top-bloke
DEĞİL**; sadece tek-seferlik insan-takım-girişine bağlı.

### 3c. RunPod-recall NE'yi keskinleştirir (gate DEĞİL, enhancer)
Far-band fine-tune (`self_training/`) recall'ı %73→? çekince: 3a/3b'de daha çok `observed`,
daha az `interpolated` → özellikle uzak kenarda şekil daha keskin, `observed_frac` floor'u
daha az frame griler. **Ama konumsal analitik recall'ı BEKLEMEZ** — bugünkü %73 +
`continuous_state` dolgusu ile çalışır. Ek olarak `fusion/fuse_engine.py` (2-kamera) uzak-1/3'ü
LOKAL kapatır = recall fine-tune'a kısmi ikame.

---

## 4. DÜRÜST BAĞIMLILIK GRAFİĞİ

```
calib v2 (DONE) ──> replay2d.project / continuous_state (DONE)
                        │
   ┌────────────────────┼─────────────────────────────────────────┐
   │                    │                                          │
   ▼                    ▼                                          ▼
[3a] TAKIM-AGNOSTİK   [3b] TAKIM-KOŞULLU                   [recall fine-tune]
 konumsal şekil        konumsal şekil                      self_training/ (RunPod)
 ✅ BUGÜN, 0 GPU         ▲                                   = 3a/3b'yi KESKİNLEŞTİRİR
 top YOK, takım YOK     │ team_id (manuel-seed ~14 tık       (gate DEĞİL)
                        │  VEYA tek-yönlü-yelek)            + fusion/fuse_engine (lokal far-fix)
                        │  team_split.py REFUSE-yolu
                        │
TOP (Bölüm 1) ──────────┼──> [2c] carrier/touch-timeline (top VAR, takım YOK)
 RF-DETR ball-class      │         │
 / WASB-SBDT (RunPod)    └─────────┴──> [2a/2b] PAS OLAYLARI  ✗ EN UÇTA
 + ball_eval fail-closed                (top VE takım ŞART; çift-kapı)
```

**Okuma:** Sol kol (konumsal) bağımsız ve hazır. Sağ kol (pas) iki ayrı zor zincirin
(top + takım) KESİŞİMİNDE. Recall fine-tune yan-besleyici, kimsenin önkoşulu değil.

| katman | top? | takım? | GPU/RunPod? | etiketleme? | durum |
|---|:--:|:--:|:--:|:--:|---|
| 3a takım-agnostik şekil | ✗ | ✗ | ✗ | ✗ | **BUGÜN yapılabilir** |
| 3b takım-koşullu şekil | ✗ | ✔ | ✗ | seed ~14 tık | takım-kapılı, lokal |
| recall keskinleştirme | ✗ | ✗ | ✔ | far-band ~300-800 | 3a/3b enhancer |
| 2c carrier/touch-timeline | ✔ | ✗ | ✔ | top GT (seyrek) | top-kapılı |
| 2a/2b PAS olayları | ✔ | ✔ | ✔ | top + takım | **en uçta, çift-kapı** |

---

## 5. AŞAMALI PLAN — satılabilir B2B artefakt sırası

Konumsal top'suz + recall-toleranslı olduğu için ÖNCE gelir.

**P1 — 2D replay (Faz 2, prototip var):** `stats/match2d.py` / `replay2d.build_replay`.
Görünür "wow", 0 sunucu-compute, iframe-gömülebilir. (zaten `docs/replay_demo.mp4`)

**P2 — Takım-agnostik konumsal analitik (3a) — YENİ `stats/match_eval.py`:**
`continuous_state` zaman serisi → topluluk şekli (zarf, genişlik, derinlik, kompaktlık,
NN-aralık), tempo (observed-only), bölge-occupancy, Voronoi. observed_frac etiketli,
floor-altı gri. **İlk savunulabilir analitik ürün; top/takım/GPU/etiket YOK.** Replay'in
üstüne overlay panelleri. → `tests/test_match_eval.py`.

**P3 — Tek-seferlik takım-enrollment + takım-koşullu konumsal (3b):** ~14 tık/maç
manuel-seed helper (topdown_viz crop'larını kullan) VEYA tek-yönlü-yelek. Sonra
defans-hattı, hat-arası, pressing-proxy, per-takım kompaktlık. REFUSE-kapılı, DEFERRED+balance_note.

**P4 — Recall fine-tune (RunPod) + 2-cam füzyon:** `self_training/` far-band; `fuse_engine`
zıt-uç. P2/P3'ün uzak-kenarını keskinleştirir, yeni metrik açmaz.

**P5 — Top katmanı + carrier/touch-timeline (2c):** RF-DETR ball-class Stage1 →
gerekirse WASB-SBDT Stage2; `eval/ball_eval.py` fail-closed (precision≥~%90 yoksa REFUSE).
topla-temas sayısı, topta-süre (takımsız).

**P6 — PAS ağı (2a/2b):** top + takım birleşince pas sayısı/isabet/yön/ilerletici/ağ.
**En son** — çünkü iki zor zincirin kesişimi ve her ikisi de fail-closed.

---

## 6. DÜRÜSTLÜK DEĞİŞMEZLERİ (kod-assert seviyesinde)
- Top'suz PAS üretme — `BALL_NOTES.md` proxy yasağı. Top fail-closed (fabrikasyon-audit ~0).
- Sahte 7-7 takım shipleme — `TEAM_NOTES.md`; REFUSE = doğru davranış, başarısızlık değil.
- interp/predicted konumlar TESPİT DEĞİL → hız/tempo/mesafeye GİRMEZ (sadece şekil/occupancy,
  conf-ağırlıklı). `player_state.py` zaten bunu uyguluyor.
- Ölçek `rel_m` (±%13) — metre/sprint sadece doğrulanmış exact-modda.
- Far-recall açığı şekli yanlıtmasın: floor-altı frame gri, sayı 14'e ZORLANMAZ.
- Pas/konumsal her sayı `observed_frac` + ölçek-band + (varsa) takım-conf ile sunulur.

## 7. Reproduce (Bölüm 3a ölçümü)
```
venv/bin/python -c "
from stats.replay2d import load_calib, postprocess
import pandas as pd
cal=load_calib('calib/cankaya_cam2_v2.json')
st=postprocess(pd.read_parquet('scratchpad/bot_raw_tracks.parquet'), cal, fps=88294/3549.97)
# st: pid,frame,seg,x,y,status -> per-frame ConvexHull/width/depth/NN (Bölüm 3a tablosu)
"
```

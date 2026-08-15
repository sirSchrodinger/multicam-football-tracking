# Tek-Kare Kalibrasyon Robustluğu — Yöntem Sentezi (gece-saha, 121 kare)

**Baseline:** largest-component + sanity = **38/121 PASS** (~%31).
(Görev brief'i %33=40/121 diyor; yöntem script'leri 38'i baz aldı — aşağıdaki tüm Δ'lar 38'e göre. Fark gate'in nondeterminizmi/eşik değil, sayım kaynağı.)

**Metrik:** PASS = `rec.ok AND auto_clean2d.sanity(...)`. KRİTİK: bu gate GAMEABLE. Bu yüzden her iddia GÖRSEL doğrulandı (reprojekte saha-çizgisini gerçek beyaz çizgiyle karşılaştır). Sadece görsel-doğrulanmış kurtarma sayılır.

---

## 1. GERÇEKTEN baseline'ı geçen yöntem(ler)

**Yalnızca `vanishing_point`.** Adversaryal olarak bağımsız re-render edilip (adv_overlay.py → adv_overlays/) tek tek Read edildi.

- **8 görsel-doğrulanmış yeni kurtarma** (idx 6, 26, 27, 57, 58, 60, 72, 114): reprojekte sarı merkez-yuvarlak GERÇEK boyalı çembere oturuyor VE ≥1 mor ceza-sahası gerçek beyaz kutuyu izliyor. En güçlü: idx57 (sol kutu beyaz kutuya tam). En zayıf-ama-savunulabilir: idx114.
- **2 regresyon** (idx 40, 78) — ama bunlar head-on dejenere baseline-PASS'lerin DOĞRU reddi, gerçek kayıp değil.
- Net dürüst kazanç: **38 → ~46 GERÇEK PASS** (~%38), yani **≥8 doğrulanmış bowtie kurtarma**.

⚠️ **Headline 78/121 ABARTILI** ve gate-overfit ile şişmiş (düşük-IC kuyruğu konveks-ama-yanlış). Bunu yöntemin kendisi açıkça beyan etti; doğrulanan iddia 78 değil, kapsamı daraltılmış **8**. IC < ~0.25 kuyruğuna GÜVENME.

---

## 2. Gate-overfit olarak elenenler (doğrulanmış kazanç = 0)

| Yöntem | Gate PASS | İddia yeni | Doğrulanan | Verdict | Kök neden |
|---|---|---|---|---|---|
| `multicomp_interior` | 67 | 29 | **0** | failed | Klasik tuzak: far-rol top-3 kombinasyonundan gate-geçeni seç → konveks ama köşeye çökmüş / sadece yakın-yarı oturan H. interior_consistency'nin kendisi de gameable (gürültülü prob-piksele yakınlık 0.4-0.56 verir). |
| `ellipse_anchor` | 46 | 8 | **0** | neutral | Bu sahalarda boyalı merkez-yuvarlak çoğu yok → fitEllipse halüsine blob'a oturuyor. Fallback yapısı 0-regresyon ama anchor sahte. |
| `conf_weighted` | 41 | 9 | **1** (idx35, borderline) | neutral | sqrt(prob)-ağırlıklı TLS gürültüyü hafif bastırıyor ama far-çizgi yine fence/scale'e kilitleniyor. Marjinal gate hareketi, sağlam kazanç yok. |
| `convex_penalty` | 39 | 3 | **0** | failed | Konvekslik cezası sanity'nin zaten baktığını optimizatöre taşıyor; yanlış korespondansı düzeltmiyor → convex-ama-yanlış H. Weight sweep'te new_rec seti tamamen değişti = overfit imzası. 2 gerçek regresyon. |
| `ransac_lines` | 34 | 12 | **0** | loss | "En kalabalık tek çizgi" tam da yanlış çizgiyi (düz fence/duvar tepesi) seçtiriyor; largest-component'in doğru bulduğu 16 sahayı da bozdu. Net kayıp. |

Ortak ders: **gate (convex + düşük residual + IC) yetmez**; far-line yanlış-korespondansını (fence/duvara kilit) düzeltmeyen her yöntem sadece gate'i kandırıyor.

---

## 3. EN İYİ yöntem + entegrasyon

**`vanishing_point`** — `calib_train/night/method_vanishing_point.py`

Neden: far-rol (goalF/touchF) için TEK büyük-bileşen yerine BİRDEN ÇOK aday çıkarıp her konfigi VP-geometri ile sınar (konveks quad + iki vanishing-point saha-DIŞI + ufuk saha-üstü), VP-makul olanları solve'a verir, sanity-geçenler arasından en yüksek interior-consistency'liyi seçer, güçlü baseline-PASS'i korur (regresyon-koruma). Yanlış-korespondans problemine doğrudan dokunan TEK yöntem: fence'e kilitlenen far-line VP testini geçemiyor.

Entegrasyon (1-2 cümle): Ana pipeline'da `calib_from_pred`'in largest-component seçimini, far-rol kanalları için method_vanishing_point'taki **multi-candidate + VP-gate + regresyon-korumalı seçim** ile değiştir; ama **yalnızca IC ≥ ~0.45 eşiğiyle yayınla** (düşük-IC kuyruğu gate-overfit, susturulmalı) — yani kazancı ≥8 doğrulanmış sahaya sınırla, ham 78'i değil.

---

## 4. Dürüst verdict

Tek-kare geometri-robustluğu darboğazı **tam kırmadı, ama küçük gerçek bir çentik attı**: vanishing_point ile **38 → ~46 görsel-doğrulanmış PASS (~%31 → ~%38)**, ≥8 sağlam bowtie kurtarma. Diğer 5 yöntemin tamamı doğrulanmış 0 kazanç (gate-overfit).

Kalan ~%60'ın çoğu (boyalı iç-işaret yok / head-on zayıf-perspektif / far-line fiziksel olarak fence'ten ayırt edilemez) tek kareden **bilgi-teorik olarak kurtarılamaz**. Bu sahalar için yol: **2-kamera füzyonu (zıt-uç) + temporal tutarlılık** — tek-kare ne kadar akıllı olursa olsun bu bariyeri aşamaz. vanishing_point'i entegre et (ucuz, ≥8 net), gerisini 2-cam/temporal'a bırak.

---
*Kaynak: explore (6 yöntem prototip+görsel) + verify (vanishing_point bağımsız adversaryal re-doğrulama, "holds"). Tüm kod yolları calib_train/night/method_*.py.*

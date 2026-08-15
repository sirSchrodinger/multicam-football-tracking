#!/usr/bin/env python3
"""Artifact rapor HTML üret: dürüst auto-calib durumu + teşhis + vision-recovery kanıtı.
Gerçek sayıları _judge_results.json'dan, hero data-URI'leri scratchpad/heroes.json'dan çeker."""
import json, os
from collections import Counter
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
SCR="/tmp/claude-1000/-home-schrodiger/a4885d7e-62c7-4078-8fac-9344d56713e8/scratchpad"
H=json.load(open(f"{SCR}/heroes.json"))
J=json.load(open(f"{ROOT}/cand/_judge_results.json"))['per_idx']
c=Counter(r['final'] for r in J); n=len(J); usable=c.get('GOOD',0)+c.get('DECENT',0)
w=Counter(r.get('worst') for r in J if r['final'] in ('BAD','ROUGH'))
# recovery + boundary (ürün-bar) istatistikleri (varsa)
def _load(p):
    try: return json.load(open(p))
    except Exception: return None
REC=_load(f"{ROOT}/cand/_judge_recover_results.json")
BFULL=_load(f"{ROOT}/cand/_boundary_full.json"); BND=_load(f"{ROOT}/cand/_boundary_results.json")
rec_rows=(REC.get('per_idx',REC) if isinstance(REC,dict) else REC) or []
rec_usable=[r for r in rec_rows if r.get('final') in ('GOOD','DECENT')]
rec_n=len(rec_rows)
# ürün-bar: AUTO tam-80 (_boundary_full), recover 25-örneklem (_boundary_results)
if BFULL:
    A=BFULL['auto']; a_use=A['USABLE']; a_marg=A['MARGINAL']; a_wrong=A['WRONG']; an=A['n']
    rv=[r for r in (BND['per_idx'] if BND else []) if r.get('source')=='recover'] if BND else []
    au25=[r for r in (BND['per_idx'] if BND else []) if r.get('source')=='auto'] if BND else []
    r_use=sum(1 for r in rv if r.get('verdict')=='USABLE'); rn=len(rv)
    a_use25=sum(1 for r in au25 if r.get('verdict')=='USABLE')
    prod=dict(an=an,a_use=a_use,a_marg=a_marg,a_wrong=a_wrong,rn=rn,r_use=r_use,a_use25=a_use25)
else:
    prod=None
WEL={'penalty_box':'ceza sahası','center_circle':'orta yuvarlak','far_goal':'uzak kale',
     'center_line':'orta çizgi','far_touch':'uzak taç','near_touch':'yakın taç','near_goal':'yakın kale'}
VMETA=[('GOOD','sorunsuz','#56b877'),('DECENT','kullanılabilir','#7bb87e'),
       ('ROUGH','tanınır ama kayık','#d9a441'),('BAD','çarpık / yamuk','#d05a48'),
       ('UNSOLVABLE','kare elvermiyor','#7a8288')]

rows_v="".join(
 f"<tr><td><span class='pill' style='--c:{col}'>{k}</span></td><td class='muted'>{desc}</td>"
 f"<td class='num'>{c.get(k,0)}</td><td class='bar'><span style='width:{100*c.get(k,0)/n:.0f}%;background:{col}'></span></td></tr>"
 for k,desc,col in VMETA)
rows_w="".join(
 f"<tr><td>{WEL.get(k,k)}</td><td class='num'>{v}</td>"
 f"<td class='bar'><span style='width:{100*v/max(w.values()):.0f}%;background:#ef6a55'></span></td></tr>"
 for k,v in w.most_common(6))

HTML=f"""<article>
<header class="hero">
  <p class="eyebrow">saha kalibrasyon · dürüst durum raporu · 2 Tem 2026</p>
  <h1>Her halısahayı tek kareden 2D'ye çevirmek</h1>
  <p class="lede">80 farklı tesisin otomatik kalibrasyonu ayrı ayrı yapay-zekâ değerlendiricilere baktırıldı.
  Asıl ders sayıda değil, <b>hangi soruyu sorduğunda</b>: "her çizgi mükemmel mi" diye sorarsan felaket görünüyor;
  "oyuncuyu doğru yere koyuyor mu" diye sorarsan aslında çoğu tesis çalışıyor.</p>
  <div class="bignum">
    <div><span class="big muted">{usable}<span class="slash">/{n}</span></span><span class="cap">"tüm çizgiler otursun" katı-barında — ama bu <b>yanlış soru</b> (kutu/çember kozmetik)</span></div>
    <div class="vs"><span class="big" style="color:var(--good)">{prod['a_use'] if prod else '?'}<span class="slash">/{prod['an'] if prod else '?'}</span></span><span class="cap"><b>oyuncu-konumu barında</b> — yalnız sınır doğru mu (asıl önemli olan)</span></div>
  </div>
</header>

<section>
  <h2>Satılan şey: canlı 2D + oyuncu istatistiği</h2>
  <p>Kalibrasyon "yeterince iyi" olunca ürün bu: gerçek maçtan oyuncuların saha-içi konumu, izleri ve
  per-oyuncu <b>koşu mesafesi + tepe hızı</b>. Aşağısı Çankaya'dan gerçek bir segment — 13 oyuncu, konsolide kimlik.</p>
  <figure class="wide"><img src="{H['product']}" alt="ürün dashboard — Çankaya canlı 2D + istatistik"><figcaption>Çankaya Halı Saha · canlı 2D + per-oyuncu istatistik. Mesafe savgol-düzeltilmiş (recall-boşlukları nedeniyle alt-sınır), ölçek ±%13.</figcaption></figure>
  <p class="note">Bunu üretmek için kalibrasyonun "tüm çizgileri mükemmel" olması gerekmiyor — yalnız <b>sınırı doğru</b>
  olması yeter, yani aşağıdaki <b>%80'lik ürün-barına</b> giren tesisler. Değer buradan çıkıyor.</p>
</section>

<section>
  <h2>Boru hattı — dört adımda (gözle)</h2>
  <p>Tek kamera karesinden 2D'ye giden yol dört adım: <b>(1)</b> oyuncuları bul, ayak-altını kırmızıyla işaretle;
  <b>(2)</b> saha çizgilerini tespit et; <b>(3)</b> kuş-bakışı düzleştir (warp); <b>(4)</b> oyuncuları şematik 2D
  sahaya koy. Aşağısı gerçek bir maçtan (EFT Halı Saha, 14 oyuncu):</p>
  <figure class="wide"><img src="{H['pipeline4']}" alt="4-panel boru hattı: ham+ayak, çizgi, warp, 2D"><figcaption>ham + ayak-tespiti → çizgi tespiti → warp → 2D radar. Warp'ta uzak yarı doğal olarak yayılır (tek oblik kamera); ürün bu smear'i KULLANMAZ — sadece oyuncu ayak-noktalarını 2D'ye projekte eder (sağ-alt panel).</figcaption></figure>
  <p class="note"><b>Warp neden bazen kayıktı — ve düzelttik:</b> otomatik çözüm önce yalnız <b>4 sınır çizgisini</b>
  kullanıyordu, iyi-tespit-edilen <b>orta yuvarlağı</b> hesaba katmıyordu — bu yüzden orta-sahayı 2-14&nbsp;m yanlış
  koyup warp'ı büküyordu. Orta yuvarlak + orta çizgiyi de kısıt yapınca düzeliyor. Aşağıda Serdivan (sol: eski,
  sağ: düzeltilmiş) — çizilen sarı çember artık gerçek beyaz orta yuvarlağa oturuyor. Çemberin güvenle görüldüğü
  sahalarda otomatik uygulanır (bir non-gameable kontrolle: yeni çember gerçek çember-işaretine daha iyi oturmalı).</p>
  <figure class="wide"><img src="{H['centerfix']}" alt="warp center-fix: eski vs düzeltilmiş"><figcaption>Serdivan warp · sol = eski (yalnız 4 sınır, çember kayık) · sağ = orta-yuvarlak kısıtı eklenmiş (çember oturdu).</figcaption></figure>
</section>

<section>
  <h2>Nasıl ölçtük</h2>
  <p>Her tesis için kamera karesine kalibrasyon çizgileri işlenmiş bir overlay üretildi. Sonra <b>bağımsız bir
  ajan her overlay'e baktı</b> ve beş kademeli bir ölçekle puanladı; "kullanılabilir" çıkan <b>8 adayı şüpheci
  ikinci bir ajan yakınlaştırıp çürütmeye çalıştı — 6'sı düştü, yalnız 2'si ayakta kaldı.</b> İkinci göz,
  ceza-sahasının gerçek beyaz çizginin ~1&nbsp;m dışına çizildiğini parlak ön-bölgede yakalayabiliyor; tek bir
  gate sayısının gizlediği tam da bu: geometrik artık düşük olabilir ama saha gözle hâlâ yamuktur.</p>
  <table class="tbl">
    <thead><tr><th>karar</th><th>anlamı</th><th class="num">saha</th><th></th></tr></thead>
    <tbody>{rows_v}</tbody>
  </table>
</section>

<section>
  <h2>Ne bozuluyor</h2>
  <p>Beklentinin aksine baskın hata <em>uzak kale</em> değil. Dört sınır çoğu sahada kabaca oturuyor —
  asıl kayan, <b>iç işaretler</b>: ceza sahası ve orta yuvarlak. Bunlar sabit bir şablondan çiziliyor,
  tesisin gerçek ölçüsüne uymuyor. Gerçek "yamuk" (uzak kalenin bir duvara oturması) daha küçük bir alt-küme.</p>
  <table class="tbl">
    <thead><tr><th>en yanlış çizilen öğe</th><th class="num">saha</th><th></th></tr></thead>
    <tbody>{rows_w}</tbody>
  </table>
</section>

<section>
  <h2>İki otomatik kısayol da çuvalladı</h2>
  <p>Yamuk sahaları ucuza elemek için iki geometrik sinyal denendi. İkisi de görsel yargıyla eşleşmedi:</p>
  <ul class="findings">
    <li><b>Sınır öz-tutarlılığı (BSC)</b> — dört sınırın kendi kanal-aktivasyonuna oturması. Yakın çizgiler
      hep iyi oturduğundan ortalama, asıl-zor uzak kaleyi yıkıyor. Ayrım yok (Youden 0.12).</li>
    <li><b>Uzak-kale grounding (gf@far)</b> — sadece uzak kaleye bakan versiyon. Yamuk'ların bir kısmını
      yakalıyor ama ROUGH'ların çoğunda uzak kale doğru, başka yer (kutu/çember) bozuk — bu yüzden
      grounding onları geçiriyor. Ayrım zayıf (Youden 0.39).</li>
  </ul>
  <p class="takeaway">Ders: katı görsel kaliteyle eşleşen ucuz bir otomatik ölçü <b>yok</b>. Güvenilir tek
  sinyal, kareye bakan bir gözün kendisi.</p>
</section>

<section>
  <h2>Doğru soru: oyuncu nereye düşüyor</h2>
  <p>Ürün, çizgi çizmiyor — oyuncuların saha içindeki <b>konumunu</b> veriyor. Onun için de yalnız <b>dört sınırın</b>
  doğru olması yeter (ceza-sahası ve orta yuvarlak sabit şablondan çizilir, kozmetiktir). Aynı 25 sahaya bu gözle
  bakınca tablo tersine dönüyor. 80 tesisin tümünde ölçüldü: <b>otomatik-kalibrasyonun sınırı {prod['a_use'] if prod else '?'}/{prod['an'] if prod else '?'}
  sahada doğrudan kullanılabilir</b> ({round(100*prod['a_use']/prod['an']) if prod else '?'}%), {prod['a_marg'] if prod else '?'} tanesi marjinal
  (uzak bölge biraz kayık ama iş görür), yalnız {prod['a_wrong'] if prod else '?'} tanesi yanlış — yani <b>{round(100*(prod['a_use']+prod['a_marg'])/prod['an']) if prod else '?'}%
  kullanılabilir-veya-marjinal</b>. Sistem "2/80" felaketi değil; yanlış barla ölçülmüştü.</p>
</section>

<section>
  <h2>Denedim ve elendi: ajan-köşe recovery</h2>
  <p>Otomatik uzak kaleyi ıskaladığında, ajanın kareye bakıp dört köşeyi kendisi okumasını denedim
  ("ben görüyorsam sen de görürsün" içgörünle). Tek tük dramatik düzelme verdi — Arslan'da uzak kale
  duvardan gerçek kaleye geçti:</p>
  <div class="ba">
    <figure><img src="{H['auto13']}" alt="otomatik"><figcaption><span class="tag bad">otomatik</span> uzak kale tavan duvarında</figcaption></figure>
    <figure><img src="{H['rec13']}" alt="ajan-köşeli"><figcaption><span class="tag warn">ajan-köşeli</span> uzak kale kaleye geçti — ama saha hâlâ tam oturmadı</figcaption></figure>
  </div>
  <p class="note"><b>Ama toplamda GERİLEME.</b> Aynı 25 sahalık örneklemde ürün-barıyla: otomatik {prod['a_use25'] if prod else '?'}/25 kullanılabilir,
  ajan-köşeli sadece {prod['r_use'] if prod else '?'}/25. Recovery yalnız <b>2</b> sahada otomatikten iyi, <b>14</b>
  sahada DAHA KÖTÜ yaptı — ajanın köşe tahmini, mevcut otomatik sınırdan daha gürültülü (özellikle head-on/occluded
  karelerde kale↔taç karıştırıyor). Yani recovery kaldıraç değil, <b>elendi</b>. Doğru hamle: otomatiği kullan,
  ürün-barıyla ölç, ve o ince "sınır gerçekten paralel mi" yargısı için insan gözü / geometrik paralellik-metriği.</p>
</section>

<footer>
  <p><b>Sıradaki:</b> ürün-barını (sınır) 121 tesisin tümüne ölç — otomatik-kalibrasyonla, recovery'siz;
  sınır-doğru sahalarda oyuncu-konumu 2D'yi ürün olarak ver. Sınırın "gerçekten paralel mi" ince yargısı için
  otomatik geometrik metrik (gerçek çizgi kaçış-noktaları vs kalibrasyon) — ajan-görsel-yargıdan daha güvenilir.
  Kutu/çember gibi kozmetik işaretlerin kesin metriği isteniyorsa tesis-başı tek ölçüm.</p>
  <p class="sig">halısaha-stats · tek-kamera sabit-kurulum · dürüstlük &gt; gate · 2 Tem 2026 otonom oturum</p>
</footer>
</article>

<style>
:root{{--bg:#14171a;--panel:#1b1f22;--ink:#e9ede9;--muted:#8b9490;--line:#2a2f34;--coral:#ef6a55;
  --good:#56b877;--warn:#d9a441;--bad:#d05a48;}}
*{{box-sizing:border-box}}
article{{max-width:920px;margin:0 auto;padding:56px 24px 40px;color:var(--ink);
  font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  background:var(--bg);
  background-image:radial-gradient(1200px 400px at 70% -5%,rgba(239,106,85,.06),transparent 60%);}}
h1,h2{{font-family:Georgia,"Times New Roman",serif;font-weight:600;letter-spacing:-.01em;text-wrap:balance}}
h1{{font-size:clamp(30px,5vw,46px);line-height:1.08;margin:.2em 0 .3em}}
h2{{font-size:24px;margin:0 0 .5em;padding-top:.2em}}
section{{margin:44px 0;border-top:1px solid var(--line);padding-top:28px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-size:12px;color:var(--coral);margin:0 0 8px;font-weight:600}}
.lede{{font-size:19px;color:#cdd3cd;max-width:64ch}}
.muted{{color:var(--muted)}}
.bignum{{display:flex;gap:40px;flex-wrap:wrap;margin-top:28px;padding:22px 24px;background:var(--panel);
  border:1px solid var(--line);border-radius:12px}}
.bignum>div{{display:flex;flex-direction:column;gap:4px}}
.big{{font-family:Georgia,serif;font-size:52px;line-height:1;font-variant-numeric:tabular-nums}}
.big .slash{{font-size:26px;color:var(--muted)}}
.vs .big{{color:var(--muted)}}
.cap{{font-size:13px;color:var(--muted);max-width:24ch}}
.cap b{{color:var(--ink)}}
p{{max-width:66ch}}
.tbl{{width:100%;border-collapse:collapse;margin-top:14px;font-size:15px}}
.tbl th{{text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  font-weight:600;padding:6px 10px;border-bottom:1px solid var(--line)}}
.tbl td{{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:middle}}
.tbl .num{{text-align:right;font-variant-numeric:tabular-nums;width:64px}}
.tbl .bar{{width:34%}}.tbl .bar span{{display:block;height:8px;border-radius:5px;min-width:2px}}
.pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:700;
  color:var(--bg);background:var(--c)}}
.findings{{list-style:none;padding:0;margin:14px 0;display:flex;flex-direction:column;gap:12px}}
.findings li{{padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--coral);border-radius:8px;max-width:none}}
.takeaway{{font-size:17px;border-left:3px solid var(--coral);padding-left:16px;color:#cdd3cd}}
.ba{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}}
.ba img,.grid3 img,.wide img{{width:100%;border-radius:8px;border:1px solid var(--line);display:block}}
.wide{{margin:16px 0}}
figcaption{{font-size:13px;color:var(--muted);margin-top:8px;line-height:1.4}}
.tag{{display:inline-block;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:700;color:var(--bg);margin-right:5px}}
.tag.good{{background:var(--good)}}.tag.bad{{background:var(--bad)}}
.note{{font-size:14.5px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;max-width:none}}
.note b{{color:var(--ink)}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
footer{{margin-top:44px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:14px}}
.sig{{font-size:12px;letter-spacing:.04em;margin-top:8px;color:#5f676b}}
@media(max-width:640px){{.ba,.grid3{{grid-template-columns:1fr}}.bignum{{gap:22px}}}}
@media print{{
  section{{border-top:none;margin:0;padding:20px 0 8px}}
  h2{{break-after:avoid;padding-top:14px;border-top:1px solid var(--line)}}
  figure,.ba,.bignum,.tbl,.note,.findings li,.wide{{break-inside:avoid}}
  .ba{{break-inside:avoid}} p{{orphans:3;widows:3}}
  header.hero{{break-after:avoid}}
}}
</style>"""
open(f"{ROOT}/cand/rapor.html","w").write(HTML)
print(f"-> {ROOT}/cand/rapor.html  ({len(HTML)//1024} KB)")

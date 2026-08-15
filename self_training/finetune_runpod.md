# RF-DETR fine-tune — RunPod 4090 recipe (recall #1 fix)

**Amaç:** amatör-gece halısaha dağılımına RF-DETR'ı fine-tune et → far-üçte-bir
recall'ı %64 → %85+ (replay'i gerçekten izlenebilir yapar). Yerel 1650 Ti 4GB
fine-tune yapamaz (≥8-16GB gerek) → RunPod 4090.

## Dataset (lokalde HAZIR, overfit-siz)
`self_training/build_dataset.py` üretir: `self_training/data/coco/` (images/ +
train.json + val.json + stats.json). Tasarım:
- **Track-doğrulanmış pseudo-label** (pseudo_label tracklet kapısı: süre/box-cv/
  teleport/in-pitch) → tek-kare FP/gürültü elenir.
- **conf-eşik 0.3 + sıkı tracklet kapısı** → tiling-kurtarılan FAR oyuncular
  (base'in kaçırdığı = öğretilecek zor durumlar) DAHİL. Çankaya starter: 622 kare,
  6858 kutu, %20'si <0.5-conf track-doğrulanmış far.
- **TESİS-bazlı train/val split** → held-out venue = generalization ölçümü =
  OVERFIT GUARD. (Multi-venue harvest `harvest_venues.py` ile; bitince val =
  held-out tesis.)
- DÜRÜST sınır: pseudo-label gürültülü (recall ~%73 base → ~%27 kaçan etiketlenmez).
  Bu yüzden held-out venue val + insan-GT recall_eval ile DOĞRULA, körlemesine
  döngüleme YOK (drift_guard).

## RunPod adımları
1. RunPod Community 4090 pod (~$0.34/sa), PyTorch image.
2. Repo + dataset'i kopyala (`self_training/data/coco/` + `models/weights/checkpoint_best_regular.pth`).
3. rfdetr beklenen yapı (roboflow-stil): `dataset/{train,valid}/` her birinde
   `_annotations.coco.json` + jpg'ler. `reorg_for_rfdetr.py` (aşağıda) bunu kurar.
4. Fine-tune:
   ```python
   from rfdetr import RFDETRLarge
   m = RFDETRLarge(pretrain_weights="checkpoint_best_regular.pth")
   m.train(dataset_dir="dataset", epochs=20, batch_size=4, lr=1e-4,
           output_dir="ft_out")   # ~birkaç saat 4090
   ```
5. Çıktı checkpoint'i indir → `models/weights/checkpoint_ft.pth`.

## Fine-tune SONRASI yerel doğrulama (overfit kontrolü)
- `eval/recall_eval.py` ile 8 insan-GT karesinde recall'ı yeniden ölç (gold-val).
- `eval/match_quality.py` ile aktif-oyun karnesini yeniden çıkar (recall ekseni
  %64'ten ne kadar çıktı).
- BAŞKA tesiste (held-out) recall'a bak → genelliyor mu (overfit değil mi).

## Tahmini maliyet
~2-4 saat 4090 @ $0.34/sa = **~$1-2** (Community). Secure $0.69/sa = ~$2-3.
"""

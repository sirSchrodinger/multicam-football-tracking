"""COCO (train.json/val.json + images/) -> rfdetr beklenen yapı (train/ valid/ +
_annotations.coco.json). RunPod'da fine-tune öncesi çalıştır."""
import json, shutil, sys
from pathlib import Path
src=Path(sys.argv[1] if len(sys.argv)>1 else "self_training/data/coco")
dst=Path(sys.argv[2] if len(sys.argv)>2 else "self_training/data/rfdetr")
for split,j in [("train","train.json"),("valid","val.json")]:
    coco=json.load(open(src/j)); d=dst/split; d.mkdir(parents=True,exist_ok=True)
    for im in coco["images"]:
        s=src/"images"/im["file_name"]
        if s.exists(): shutil.copy(s, d/im["file_name"])
    json.dump(coco, open(d/"_annotations.coco.json","w"))
    print(f"{split}: {len(coco['images'])} kare -> {d}")

"""corner_labels_parts/*.json -> corner_labels.json birleştir (race-free toplama)."""
import json,glob,os
DS='calib_train/corner_labels.json'; PD='calib_train/corner_labels_parts'
lab=json.load(open(DS)) if os.path.exists(DS) else {}
n0=len(lab)
for f in glob.glob(f'{PD}/*.json'):
    idx=os.path.basename(f)[:-5]
    try:
        v=json.load(open(f))
        if v.get('corners') and len(v['corners'])==4: lab[idx]=v
    except: pass
json.dump(lab,open(DS,'w'),ensure_ascii=False)
print(f'merge: {n0} -> {len(lab)} etiket (+{len(lab)-n0})')

"""Halısaha ingest paketi — venue-agnostik tarama → kamera kayıt defteri (registry).

Bu paket yalnızca tarama artefaktlarından (cams_*.tsv + list_*.json) ve mevcut
kalibrasyon JSON'larından beslenir; import-zamanında torch/cv2 gerektirmez.
"""

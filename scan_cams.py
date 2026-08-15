#!/usr/bin/env python3
"""Scan sosyalhalisaha match pages and count cameras per match (videoSrc array length).

Usage: python3 scan_cams.py 08.06.2026 [max_pages] [max_matches]
Output: cams_<date>.tsv  (id, n_cams, date, venue, title, views, first_url)
"""
import json
import re
import sys
import time
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
DELAY = 0.5


def get(url, xhr=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if xhr:
        req.add_header("X-Requested-With", "XMLHttpRequest")
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "08.06.2026"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_matches = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    matches = []
    url = f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"
    for page in range(max_pages):
        try:
            d = json.loads(get(url, xhr=True))
        except Exception as e:
            print(f"[list] page {page} failed: {e}", flush=True)
            break
        if d.get("status") != "success":
            break
        for m in d.get("data", []):
            matches.append(m)
        url = d.get("nextPage")
        print(f"[list] page {page}: +{len(d.get('data', []))} (total {len(matches)}) next={url}", flush=True)
        if not url:
            break
        time.sleep(DELAY)

    out = open(f"cams_{date.replace('.', '')}.tsv", "w")
    out.write("id\tn_cams\tdate\tvenue\ttitle\tviews\tfirst_url\n")
    two_cam = 0
    for i, m in enumerate(matches[:max_matches]):
        mid = m["url"].rstrip("/").split("/")[-1]
        try:
            html = get(m["url"])
        except Exception as e:
            print(f"[match] {mid} fetch failed: {e}", flush=True)
            continue
        mm = re.search(r"videoSrc\s*=\s*(\[.*?\]);", html, re.S)
        if not mm:
            n, first = 0, ""
        else:
            try:
                arr = json.loads(mm.group(1))
                n = len(arr)
                first = arr[0]["url"] if arr else ""
            except Exception:
                n, first = -1, ""
        row = f"{mid}\t{n}\t{m['date']}\t{m['place']['name']}\t{m.get('title', '')}\t{m.get('watch_count', 0)}\t{first}"
        out.write(row + "\n")
        out.flush()
        if n >= 2:
            two_cam += 1
            print(f"[HIT] {row}", flush=True)
        if i % 10 == 0:
            print(f"[match] {i}/{min(len(matches), max_matches)} scanned, {two_cam} two-cam so far", flush=True)
        time.sleep(DELAY)
    out.close()
    print(f"DONE: {min(len(matches), max_matches)} scanned, {two_cam} matches with >=2 cams", flush=True)


if __name__ == "__main__":
    main()

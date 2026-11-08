"""Step 0: download the 32 ABARES reports listed in data/pdfs/manifest.csv (CC BY, about 80 MB).

    python 00_download_data.py
"""
import csv
import os

import requests

from common import PDFS

HEAD = {"User-Agent": "Mozilla/5.0 (GeoAI-Lab RAG tutorial)"}


def main():
    rows = list(csv.DictReader(open(os.path.join(PDFS, "manifest.csv"), encoding="utf-8-sig")))
    for r in rows:
        out = os.path.join(PDFS, r["file"])
        if os.path.exists(out) and open(out, "rb").read(4) == b"%PDF":
            print("have", r["file"])
            continue
        resp = requests.get(r["url"], headers=HEAD, timeout=120)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            print("NOT A PDF (site layout may have changed):", r["url"])
            continue
        open(out, "wb").write(resp.content)
        print(f"got  {r['file']}  {len(resp.content) / 1e6:.1f} MB")
    print(len(rows), "reports in the manifest")


if __name__ == "__main__":
    main()

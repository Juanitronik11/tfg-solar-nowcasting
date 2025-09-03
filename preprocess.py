# caffeinate -d -i python src/preprocess.py
# -*- coding: utf-8 -*-
"""
Scans data/raw/images/YYYYMMDD/ and builds:
  - pairs_brbg.csv  -> raw + BRBG (baseline)
  - pairs_both.csv  -> raw + BRBG + CDOC (for fusion)
Additionally, lists missing days and days without pairs/triples.

Expected per-day file structure:
  * Raw image:            *_11_NE.jpg
  * BRBG mask (with sun): *_1112_BRBG.png
  * CDOC mask (detail):   *_1112_CDOC.png
"""

import os
from glob import glob
import pandas as pd
from datetime import date, timedelta

# ---------- Parameters ----------
START_DATE = date(2018, 1, 1)
END_DATE   = date(2024, 12, 31)

BASE_DIR   = "data/raw/images"
OUT_DIR    = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def date_range(start: date, end: date):
    """Generates all dates between start and end (both inclusive)."""
    d = start
    while d <= end:
        yield d.strftime("%Y%m%d")   # 'YYYYMMDD'
        d += timedelta(days=1)


def _scan_day(day_folder: str):
    """
    Returns three dictionaries {stem -> path} for raw, brbg, cdoc.
    The 'stem' is the part before the first '_' (e.g., 'YYYYMMDDhhmmss').
    """
    raw_dict, brbg_dict, cdoc_dict = {}, {}, {}

    for p in glob(os.path.join(day_folder, "*_11_NE.jpg")):
        stem = os.path.basename(p).split("_")[0]
        raw_dict[stem] = p

    for p in glob(os.path.join(day_folder, "*_1112_BRBG.png")):
        stem = os.path.basename(p).split("_")[0]
        brbg_dict[stem] = p

    for p in glob(os.path.join(day_folder, "*_1112_CDOC.png")):
        stem = os.path.basename(p).split("_")[0]
        cdoc_dict[stem] = p

    return raw_dict, brbg_dict, cdoc_dict


def find_triples_and_gaps(base_dir: str):
    """
    Builds:
      - list of raw+BRBG pairs
      - list of raw+BRBG+CDOC triples
      - lists of missing days and days without pairs/triples
    """
    expected = set(date_range(START_DATE, END_DATE))
    present  = set([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    # 1) Days whose ZIP was never extracted (missing folder)
    missing_folders = sorted(expected - present)

    raws_brbg, brbg_paths, raws_both, brbg_both, cdoc_both = [], [], [], [], []
    days_no_pairs_brbg, days_no_triples = [], []

    # 2) Present days: build pairs and triples
    for day in sorted(present):
        day_folder = os.path.join(base_dir, day)
        raw_dict, brbg_dict, cdoc_dict = _scan_day(day_folder)

        # Raw+brbg pairs (baseline)
        common_pairs = sorted(set(raw_dict) & set(brbg_dict))
        if not common_pairs:
            days_no_pairs_brbg.append(day)
        else:
            raws_brbg.extend([raw_dict[s] for s in common_pairs])
            brbg_paths.extend([brbg_dict[s] for s in common_pairs])

        # Triples raw+brbg+cdoc (for fusion)
        common_triples = sorted(set(raw_dict) & set(brbg_dict) & set(cdoc_dict))
        if not common_triples:
            days_no_triples.append(day)
        else:
            raws_both.extend([raw_dict[s]  for s in common_triples])
            brbg_both.extend([brbg_dict[s] for s in common_triples])
            cdoc_both.extend([cdoc_dict[s] for s in common_triples])

    return (raws_brbg, brbg_paths,
            raws_both, brbg_both, cdoc_both,
            missing_folders, days_no_pairs_brbg, days_no_triples)


if __name__ == "__main__":
    (raws_brbg, brbg_only,
     raws_both, brbg_both, cdoc_both,
     miss_folders, miss_pairs_brbg, miss_triples) = find_triples_and_gaps(BASE_DIR)

    # ---------------------- SUMMARY ----------------------
    print(f"Parejas raw+BRBG: {len(raws_brbg)}")
    print(f"Tríos raw+BRBG+CDOC: {len(raws_both)}")
    print(f"Carpetas ausentes: {len(miss_folders)}")
    print(f"Días PRESENTES sin pares raw+BRBG: {len(miss_pairs_brbg)}")
    print(f"Días PRESENTES sin tríos raw+BRBG+CDOC: {len(miss_triples)}")

    # ---------------------- CSVs -------------------------
    pd.DataFrame({"raw_path": raws_brbg,
                  "mask_brbg_path": brbg_only})\
      .to_csv(f"{OUT_DIR}/pairs_brbg.csv", index=False)

    pd.DataFrame({"raw_path": raws_both,
                  "mask_brbg_path": brbg_both,
                  "mask_cdoc_path": cdoc_both})\
      .to_csv(f"{OUT_DIR}/pairs_both.csv", index=False)

    # ---------------------- Gap TXT files ----------------
    with open(f"{OUT_DIR}/missing_folders.txt", "w") as f:
        f.writelines([d + "\n" for d in miss_folders])

    with open(f"{OUT_DIR}/days_no_pairs_brbg.txt", "w") as f:
        f.writelines([d + "\n" for d in miss_pairs_brbg])

    with open(f"{OUT_DIR}/days_no_triples.txt", "w") as f:
        f.writelines([d + "\n" for d in miss_triples])

    print("Archivos guardados en data/processed/")

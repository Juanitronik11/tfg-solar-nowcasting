#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  caffeinate -d -i python src/check_corrupt_images_and_separate.py

Checks RAW images and both masks (BRBG and CDOC), optionally deletes corrupt
files from disk, and splits the dataset into train/val/test.

Produces:
  • pairs_clean.csv                (raw_path, mask_brbg_path, mask_cdoc_path, stamp, ghi)
  • pairs_train.csv   (2018–2022)
  • pairs_val.csv     (2023)
  • pairs_test.csv    (2024)
Additionally, writes one TXT per column listing paths to corrupt files.

Default input:
  data/processed/pairs_with_ghi_full.csv
  (must contain columns: raw_path, mask_brbg_path, mask_cdoc_path; and optionally stamp, ghi)
"""

from pathlib import Path
from typing import Dict
import pandas as pd
from PIL import Image

# ───────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────
INPUT_CSV  = Path("data/processed/pairs_with_ghi_full.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If you do not want to delete corrupt files from disk, set to False
DELETE_CORRUPT_FILES = True

# If you want to require non-null GHI for training, set to True
REQUIRE_GHI = False


# ───────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────
def is_image_ok(path: Path) -> bool:
    """Attempts to open and fully load the image; returns False if it is corrupt or does not exist."""
    try:
        if not path.exists() or not path.is_file():
            return False
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def remove_corrupt(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Iterates over each path in df[column], deletes corrupt files from disk
    (if DELETE_CORRUPT_FILES=True) and logs them in a TXT file.
    Returns the DataFrame filtered to only "good" rows for that column.
    Treats empty strings and literal 'nan' (as text) as corrupt without touching disk.
    """
    good_idxs, bad_paths = [], []
    total = len(df)

    print(f"\nComprobando {total} rutas en columna '{column}'…")
    for i, path_str in enumerate(df[column].astype(str), start=1):
        if i % 200 == 0:
            print(f"  Procesadas {i}/{total}", end="\r")

        s = path_str.strip().lower()
        if (s == "") or (s == "nan"):
            bad_paths.append(path_str)
            continue

        p = Path(path_str)
        if is_image_ok(p):
            good_idxs.append(i - 1)
        else:
            bad_paths.append(path_str)
            if DELETE_CORRUPT_FILES and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Write TXT listing corrupt paths
    txt_file = OUTPUT_DIR / f"corrupt_{column}.txt"
    with open(txt_file, "w") as f:
        for p in bad_paths:
            f.write(p + "\n")

    print(f"  → Eliminadas/descartadas {len(bad_paths)} imágenes corruptas en '{column}'")
    print(f"    Lista en: {txt_file}")
    return df.iloc[good_idxs].reset_index(drop=True)


def ensure_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures required columns exist; raises if any are missing."""
    required = {"raw_path", "mask_brbg_path", "mask_cdoc_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Faltan columnas en {INPUT_CSV}: {missing}. "
            f"Asegúrate de ejecutar 'preprocess.py' y 'extract_ghi_for_pairs.py' actualizados."
        )
    return df


def split_by_year(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split into train/val/test according to the date encoded in the filename:
      train: 2018–2022, val: 2023, test: 2024.
    Uses 'stamp' if present; otherwise derives it from raw_path.
    """
    df = df.copy()
    if "stamp" in df.columns:
        df["stamp"] = df["stamp"].astype(str).str.extract(r"(\d{14})")[0]
    else:
        df["stamp"] = df["raw_path"].astype(str).str.extract(r"(\d{14})_")[0]

    df["dt"] = pd.to_datetime(df["stamp"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df[df["dt"].notna()].copy()
    df["ymd"] = df["dt"].dt.strftime("%Y%m%d").astype(int)

    return {
        "train": df[(20180101 <= df["ymd"]) & (df["ymd"] <= 20221231)],
        "val":   df[(20230101 <= df["ymd"]) & (df["ymd"] <= 20231231)],
        "test":  df[(20240101 <= df["ymd"]) & (df["ymd"] <= 20241231)],
    }


# ───────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"\nLeídas {len(df)} filas de '{INPUT_CSV}'")

    df = ensure_min_columns(df)

    # Normalize and drop empty/'nan' (text) before any I/O
    for col in ["raw_path", "mask_brbg_path", "mask_cdoc_path"]:
        df[col] = df[col].astype(str)
        bad_empty = (df[col].str.strip() == "") | df[col].str.lower().eq("nan")
        if bad_empty.any():
            n = int(bad_empty.sum())
            print(f"Descartando {n} filas con {col} vacío/NaN")
            df = df[~bad_empty].reset_index(drop=True)

    # Per-column cleaning (intersection of good rows)
    df = remove_corrupt(df, "raw_path")
    df = remove_corrupt(df, "mask_brbg_path")
    df = remove_corrupt(df, "mask_cdoc_path")

    # Require non-null GHI if desired
    if REQUIRE_GHI and ("ghi" in df.columns):
        n_ghi_nan = int(df["ghi"].isna().sum())
        print(f"\nFilas sin GHI: {n_ghi_nan}")
        df = df[df["ghi"].notna()].reset_index(drop=True)

    print(f"\nQuedan {len(df)} filas tras eliminar corruptas en raw+BRBG+CDOC")

    # Ensure useful columns before saving
    if "stamp" not in df.columns:
        df["stamp"] = df["raw_path"].astype(str).str.extract(r"(\d{14})_")[0]
    keep_cols = [c for c in ["raw_path", "mask_brbg_path", "mask_cdoc_path", "stamp", "ghi"] if c in df.columns]

    # Clean CSV
    clean_csv = OUTPUT_DIR / "pairs_clean.csv"
    df[keep_cols].to_csv(clean_csv, index=False)
    print(f"CSV limpio → {clean_csv}")

    # Split by years and save
    parts = split_by_year(df)
    for name, subdf in parts.items():
        out = OUTPUT_DIR / f"pairs_{name}.csv"
        subdf[keep_cols].to_csv(out, index=False)
        print(f"  {name.upper():5}: {len(subdf)} filas → {out}")

    print("\nProceso completado.")


if __name__ == "__main__":
    main()

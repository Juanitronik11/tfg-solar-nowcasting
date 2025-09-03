#!/usr/bin/env python3
# download.py
# Example (keep Mac awake during long downloads):
#   caffeinate -d -i python src/download.py
# caffeinate flags:
#   -d  prevents the display from sleeping
#   -i  prevents idle system sleep
# With this, your download:
#   Survives temporary connection drops.
#   Resumes if your Mac or the server interrupts it.
#   Shows a continuous progress bar.

#
# Script to automatically download and extract the daily ZIPs of images
# from the EKO ASI-16 (SRRLASI) or YES TSI-880 (SRRL) sky cameras.
# Andreas told me the EKO All Sky Imager is more up-to-date.
# "An EKO All Sky Imager (ASI-16) is capturing all-sky images and computing cloud cover since September 26, 2017."
#
# Usage:
#   1) Set START_DATE and END_DATE.
#   2) Choose the camera: CAM = "SRRLASI" or "SRRL".
#   3) Run: python src/download.py
#
# ZIP files will be saved under data/raw/zips/
# and extracted images under data/raw/images/YYYYMMDD/.

import os  # Standard library: filesystem and OS utilities.
import requests
import zipfile
from datetime import date, timedelta
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS: edit as needed
# ──────────────────────────────────────────────────────────────────────────────

# Start and end dates (inclusive) in year, month, day
START_DATE = date(2018, 1, 1)
END_DATE   = date(2024, 12, 31)

# Camera: "SRRLASI" for EKO ASI-16 (recommended), or "SRRL" for YES TSI-880
CAM = "SRRLI"

# Base URL where daily ZIPs are hosted
BASE_URL = f"https://midcdmz.nrel.gov/tsi/{CAM}"

# Output folders, relative to your project root tfg_project/
OUT_ZIP_DIR  = "data/raw/zips"
OUT_IMG_DIR  = "data/raw/images"

# ──────────────────────────────────────────────────────────────────────────────
# END OF PARAMETERS – DO NOT EDIT BELOW THIS LINE
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    """
    Create output directories if they don't exist:
      - OUT_ZIP_DIR  : stores .zip files
      - OUT_IMG_DIR  : stores extracted images
    """
    os.makedirs(OUT_ZIP_DIR, exist_ok=True)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

def download_zip(zip_url: str, zip_path: str) -> bool:
    """
    Download a ZIP file from zip_url and save it to zip_path,
    showing a tqdm progress bar.

    Returns True if the ZIP is already present or downloaded successfully.
    Returns False if an HTTP error (404 or others) occurs and the file is skipped.
    """
    # Skip if already present
    if os.path.exists(zip_path):
        print(f"Already exists: {zip_path}")
        return True

    # Streaming HTTP request
    try:
        resp = requests.get(zip_url, stream=True, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Network error ({e}), skipping {zip_url}")
        return False

    if resp.status_code != 200:
        print(f" Error {resp.status_code} downloading {zip_url}, skipping.")
        return False

    total_bytes = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc=os.path.basename(zip_path),
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        # Write content in 1024-byte chunks
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    return True

def extract_zip(zip_path: str, extract_to: str):
    """
    Extract the contents of zip_path into the extract_to directory.
    """
    # If target directory already exists, assume it was extracted previously
    if os.path.isdir(extract_to):
        print(f"Already extracted: {extract_to}")
        return

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"Extracted: {extract_to}")

def main():
    # Ensure output directories exist
    ensure_dirs()

    # Iterate day by day from START_DATE to END_DATE (inclusive)
    current = START_DATE
    while current <= END_DATE:
        day_str        = current.strftime("%Y%m%d")
        zip_name       = f"{day_str}.zip"
        zip_url        = f"{BASE_URL}/{current.year}/{zip_name}"
        zip_path       = os.path.join(OUT_ZIP_DIR, zip_name)
        extract_folder = os.path.join(OUT_IMG_DIR, day_str)

        # 1) Try to download (or confirm it already exists)
        if download_zip(zip_url, zip_path):
            # 2) If present, extract
            extract_zip(zip_path, extract_folder)
        else:
            # 3) On 404 or failure, remove any partial file and skip the date
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print(f"Skipping date {day_str}")

        # 4) Move to the next day (always)
        current += timedelta(days=1)


if __name__ == "__main__":
    main()

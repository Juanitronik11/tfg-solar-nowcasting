# ──────────────────────────────
# caffeinate -d -i python src/download_bms_api.py
# Download irradiance files from SRRL-BMS
# (Global/Direct/Diffuse, etc.) month by month.
# ──────────────────────────────

#!/usr/bin/env python3
# src/download_bms_api.py
# Download GHI/DNI/DHI from SRRL BMS month-by-month to avoid failures with large time ranges.

import pandas as pd
from pvlib.iotools import read_midc_raw_data_from_nrel  # pvlib API for MIDC
# This function from pvlib accesses raw meteorological and irradiance data
# from NREL's Measurement and Instrumentation Data Center (MIDC).
from datetime import datetime
from dateutil.relativedelta import relativedelta  # to conveniently add months

# ────────────────────────────────────────────────
# 1) Parameters: set your full time range here
# ────────────────────────────────────────────────
START = datetime(2018, 1, 1, 0, 0)    # start: Jan 1, 2018 at midnight
END   = datetime(2024,12,31,23,59)    # end: Dec 31, 2022 at 23:59  (comment kept as-is)

# ────────────────────────────────────────────────
# 2) Helper to iterate month by month
# ────────────────────────────────────────────────
def month_ranges(start, end):
    """
    Generate (month_start, month_end) tuples for each month in [start, end].
    Example: (2018-01-01, 2018-01-31), (2018-02-01, 2018-02-28), ...
    """
    cur = start.replace(day=1)
    while cur <= end:
        # First day of the next month
        nxt = (cur + relativedelta(months=1)).replace(day=1)
        # End of current month is the day before the next month
        fin = min(end, nxt - relativedelta(days=1))
        yield cur, fin
        cur = nxt

# ────────────────────────────────────────────────
# 3) Monthly loop: download, filter, append
# ────────────────────────────────────────────────
frames = []  # list to store each monthly DataFrame
for mes_inicio, mes_fin in month_ranges(START, END):
    try:
        # Read only that month from the MIDC BMS API
        df_mes = read_midc_raw_data_from_nrel(
            site='BMS',
            start=mes_inicio,
            end=mes_fin
        )
    except Exception as e:
        print(f"⚠️  Failed month {mes_inicio:%Y-%m}: {e}")
        continue  # go on to the next month

    # Select and rename only the columns of interest
    df_sel = df_mes[[
        'Global CMP22 (vent/cor) [W/m^2]',  # GHI
        'Direct NIP [W/m^2]',               # DNI
        'Diffuse CM22-1 (vent/cor) [W/m^2]' # DHI
    ]].rename(columns={
        'Global CMP22 (vent/cor) [W/m^2]': 'ghi',
        'Direct NIP [W/m^2]':              'dni',
        'Diffuse CM22-1 (vent/cor) [W/m^2]': 'dhi',
    })

    frames.append(df_sel)
    print(f"Month {mes_inicio:%Y-%m} downloaded, {len(df_sel)} records.")

# ────────────────────────────────────────────────
# 4) Concatenate everything and write final CSV
# ────────────────────────────────────────────────
if not frames:
    print("No monthly data was downloaded. Check dates or connection.")
else:
    # Merge all months into a single DataFrame
    all_irr = pd.concat(frames)
    # Create 'stamp' column identical to your image filename stem
    ts = all_irr.index.to_series().dt.strftime("%Y%m%d%H%M%S")
    all_irr = all_irr.assign(stamp=ts.values)

    # Save only the necessary columns
    out_csv = "data/processed/irradiance.csv"
    all_irr[['stamp','ghi','dni','dhi']].to_csv(out_csv, index=False)
    print(f"→ Total irradiance saved to {out_csv}")

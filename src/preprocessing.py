"""
preprocessing.py
Bekaa Valley Desertification ML Project
----------------------------------------
Loads raw climatic data from 4 stations, computes the De Martonne
aridity index, engineers features, and exports a clean CSV ready
for modeling.

Usage:
    python src/preprocessing.py
Output:
    data/processed/bekaa_processed.csv
"""

import os
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(PROC_DIR, "bekaa_processed.csv")

os.makedirs(PROC_DIR, exist_ok=True)

# ── De Martonne thresholds ────────────────────────────────────────────────────
#   Im = P / (T + 10)   (monthly: Im = 12 * P / (T + 10))
#   Hyper-arid < 5 | Arid 5-10 | Semi-arid 10-20 | Sub-humid 20-30 | Humid >=30
DM_BINS   = [-np.inf, 5, 10, 20, 30, np.inf]
DM_LABELS = ['Hyper-arid', 'Arid', 'Semi-arid', 'Sub-humid', 'Humid']
DM_ENCODE = {l: i for i, l in enumerate(DM_LABELS)}   # ordinal encoding


# ══════════════════════════════════════════════════════════════════════════════
#  FILE READERS
# ══════════════════════════════════════════════════════════════════════════════

def _read_xls_xml(path: str) -> pd.DataFrame:
    """
    Parse XML-based .xls files saved by Excel 2003 XML format.
    Columns expected: date, temp_avg, temp_max, temp_min, precip_sum
    """
    with open(path, 'rb') as f:
        raw = f.read()
    content = raw.decode('utf-8-sig') if raw[:3] == b'\xef\xbb\xbf' else raw.decode('latin-1')

    # Strip namespaces so ElementTree can parse without ns maps
    content = re.sub(r'xmlns[^=]*="[^"]*"', '', content)
    for ns in ['ss:', 'x:', 'o:', 'html:']:
        content = content.replace(ns, '')

    root  = ET.fromstring(content)
    ws    = root.find('.//Worksheet')
    table = ws.find('.//Table')

    rows_data = []
    for row in table.findall('Row'):
        cells = row.findall('Cell')
        rows_data.append([
            (c.find('Data').text if c.find('Data') is not None else None)
            for c in cells
        ])

    # Row 0 = merged header group, Row 1 = column names → data starts at Row 2
    df = pd.DataFrame(rows_data[2:],
                      columns=['date', 'temp_avg', 'temp_max', 'temp_min', 'precip_sum'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precip_sum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[df['date'].notna()].reset_index(drop=True)


def _read_binary_xls(path: str) -> pd.DataFrame:
    """
    Read standard binary .xls files (Ammik format).
    First row = merged header, second row = column names.
    """
    df = pd.read_excel(path, header=1, engine='xlrd')
    df = df.iloc[:, :5].copy()
    df.columns = ['date', 'temp_avg', 'temp_max', 'temp_min', 'precip_sum']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min', 'precip_sum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[df['date'].notna()].reset_index(drop=True)


def _read_temp_only_xls(path: str) -> pd.DataFrame:
    """
    Read Tal Amara temperature files (TA1, TA2, TA3).
    These have only 4 columns: date, temp_avg, temp_max, temp_min.
    """
    df = pd.read_excel(path, header=1, engine='xlrd')
    df = df.iloc[:, :4].copy()
    df.columns = ['date', 'temp_avg', 'temp_max', 'temp_min']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['temp_avg', 'temp_max', 'temp_min']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[df['date'].notna()].reset_index(drop=True)


def _parse_rain_tal_amara(path: str) -> pd.DataFrame:
    """
    Parse Tal Amara wide-format daily rain file.
    Each sheet covers one hydrological year (SEP y1 – AUG y2).
    Aggregates daily values to monthly totals.
    """
    wb = pd.ExcelFile(path, engine='xlrd')
    months_order = ['SEP','OCT','NOV','DEC','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG']
    records = []

    for sheet in wb.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None, engine='xlrd')

        # Find the row that contains 'DATE'
        header_row = None
        for i, row in raw.iterrows():
            if any(str(v).strip().upper() == 'DATE' for v in row.values):
                header_row = i
                break
        if header_row is None:
            continue

        dfs = pd.read_excel(path, sheet_name=sheet, header=header_row, engine='xlrd')
        dfs.columns = [str(c).strip().upper() for c in dfs.columns]

        # Extract the two years from the sheet title (row index 1)
        title = str(raw.iloc[1, 0])
        years = re.findall(r'\d{4}', title)
        if len(years) < 2:
            continue
        y1, y2 = int(years[0]), int(years[1])

        month_map = {
            'SEP': (y1, 9),  'OCT': (y1, 10), 'NOV': (y1, 11), 'DEC': (y1, 12),
            'JAN': (y2, 1),  'FEB': (y2, 2),  'MAR': (y2, 3),  'APR': (y2, 4),
            'MAY': (y2, 5),  'JUN': (y2, 6),  'JUL': (y2, 7),  'AUG': (y2, 8),
        }

        for m in months_order:
            if m not in dfs.columns:
                continue
            vals  = pd.to_numeric(dfs[m], errors='coerce').dropna()
            total = vals.sum() if len(vals) > 0 else np.nan
            yr, mo = month_map[m]
            records.append({'date': pd.Timestamp(yr, mo, 1), 'precip_sum': total})

    rain = (pd.DataFrame(records)
              .sort_values('date')
              .groupby('date', as_index=False)['precip_sum'].sum())
    return rain


# ══════════════════════════════════════════════════════════════════════════════
#  STATION LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ammik(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "Ammik.xls")
    df   = _read_binary_xls(path)
    df['station'] = 'Ammik'
    return df

def load_doures(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "Doures.xls")
    df   = _read_xls_xml(path)
    df['station'] = 'Doures'
    return df

def load_ras_baalbeck(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "Ras Baalbeck.xls")
    df   = _read_xls_xml(path)
    df['station'] = 'Ras_Baalbeck'
    return df

def load_tal_amara(raw_dir: str) -> pd.DataFrame:
    ta_dir = os.path.join(raw_dir, "Tal Amara")

    # Temperature: merge TA1 + TA2 + TA3, keep latest reading per month
    ta1 = _read_temp_only_xls(os.path.join(ta_dir, "TA1.xls"))
    ta2 = _read_temp_only_xls(os.path.join(ta_dir, "TA2.xls"))
    ta3 = _read_temp_only_xls(os.path.join(ta_dir, "TA3.xls"))
    ta_temp = (pd.concat([ta1, ta2, ta3])
                 .drop_duplicates('date', keep='last')
                 .sort_values('date')
                 .reset_index(drop=True))

    # Precipitation: wide daily → monthly totals
    ta_rain = _parse_rain_tal_amara(os.path.join(ta_dir, "Rain.xls"))

    df = (pd.merge(ta_temp, ta_rain, on='date', how='outer')
            .sort_values('date')
            .reset_index(drop=True))
    df['station'] = 'Tal_Amara'
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def compute_de_martonne(precip: pd.Series, temp: pd.Series) -> pd.Series:
    """Monthly De Martonne index: Im = 12 * P / (T + 10)"""
    return (12 * precip) / (temp + 10)

def classify_aridity(dm: pd.Series) -> pd.Series:
    return pd.cut(dm, bins=DM_BINS, labels=DM_LABELS, right=False)

def compute_spi3(precip: pd.Series) -> pd.Series:
    """
    Simple SPI-3: standardize the 3-month rolling precipitation sum
    using mean and std computed over the full series.
    """
    roll = precip.rolling(3, min_periods=2).sum()
    return (roll - roll.mean()) / (roll.std() + 1e-9)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering per station independently
    so lags/rolls don't bleed across station boundaries.
    """
    stations = []
    for station, grp in df.groupby('station'):
        g = grp.sort_values('date').copy()

        # De Martonne index & aridity class
        g['de_martonne']   = compute_de_martonne(g['precip_sum'], g['temp_avg'])
        g['aridity_class'] = classify_aridity(g['de_martonne'])
        g['aridity_code']  = g['aridity_class'].map(DM_ENCODE)

        # SPI-3 drought index
        g['spi3'] = compute_spi3(g['precip_sum'])

        # Lag features (De Martonne)
        for lag in [1, 2, 3]:
            g[f'dm_lag{lag}'] = g['de_martonne'].shift(lag)

        # Rolling means (precipitation & temperature)
        g['precip_roll3']  = g['precip_sum'].rolling(3,  min_periods=2).mean()
        g['temp_roll3']    = g['temp_avg'].rolling(3,    min_periods=2).mean()
        g['dm_roll3']      = g['de_martonne'].rolling(3, min_periods=2).mean()
        g['dm_roll12']     = g['de_martonne'].rolling(12,min_periods=6).mean()

        # Temporal features
        g['month']       = g['date'].dt.month
        g['year']        = g['date'].dt.year
        g['year_trend']  = g['year'] - g['year'].min()
        g['month_sin']   = np.sin(2 * np.pi * g['month'] / 12)
        g['month_cos']   = np.cos(2 * np.pi * g['month'] / 12)

        # Wet season flag (Oct – May)
        g['wet_season'] = g['month'].isin([10, 11, 12, 1, 2, 3, 4, 5]).astype(int)

        stations.append(g)

    return pd.concat(stations, ignore_index=True).sort_values(['station', 'date']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(raw_dir: str = RAW_DIR, output_path: str = OUTPUT_CSV) -> pd.DataFrame:
    print("── Loading raw station data ──────────────────────────────")
    ammik     = load_ammik(raw_dir)
    doures    = load_doures(raw_dir)
    ras       = load_ras_baalbeck(raw_dir)
    tal_amara = load_tal_amara(raw_dir)

    df = pd.concat([ammik, doures, ras, tal_amara], ignore_index=True)
    print(f"   Raw combined shape : {df.shape}")

    print("── Engineering features ──────────────────────────────────")
    df = engineer_features(df)
    print(f"   Processed shape    : {df.shape}")
    print(f"   Columns            : {df.columns.tolist()}")

    print("── Class distribution ────────────────────────────────────")
    print(df['aridity_class'].value_counts())

    print("── Missing values ────────────────────────────────────────")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print(f"── Saving to {output_path} ──────────────────────────────")
    df.to_csv(output_path, index=False)
    print("   ✅ Done!")

    return df


if __name__ == "__main__":
    run()

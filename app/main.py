"""
FastAPI service untuk model prediksi `loss_payload` insiden jaringan.

Logika di file ini SENGAJA dibuat 1:1 sama dengan CELL 19 (contoh inference)
pada `pipeline_loss_payload_cells_new.py`, supaya fitur yang dikirim ke model
saat serving konsisten persis dengan fitur yang dipakai saat training:
1. Preprocessing dasar (severity -> severity_num, durasi -> menit, dst).
2. Feature engineering (log1p, durasi_x_severity).
3. Lookup hourly_baseline dari df_hourly (site_id + hour + day_name).
4. Encoding kategorikal pakai encoder hasil training, fallback ke "unknown".
5. Prediksi -> expm1 kalau model dilatih di skala log (is_log_model).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

# --------------------------------------------------------------------------
# Konfigurasi path (bisa dioverride via environment variable saat run/docker)
# --------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "output_ml_new/best_incident_model.joblib")
HOURLY_DATA_PATH = os.environ.get("HOURLY_DATA_PATH", "baseline_payload_hourly_weekly.csv")

# --------------------------------------------------------------------------
# Load artifacts sekali saat startup
# --------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model artifact tidak ditemukan di '{MODEL_PATH}'. "
        f"Pastikan file 'best_incident_model.joblib' hasil CELL 20 pipeline "
        f"sudah diletakkan di path tersebut (lihat README)."
    )
if not os.path.exists(HOURLY_DATA_PATH):
    raise FileNotFoundError(
        f"File data hourly baseline tidak ditemukan di '{HOURLY_DATA_PATH}'. "
        f"Pastikan 'baseline_payload_hourly_weekly.csv' sudah diletakkan di path tersebut."
    )

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
encoders = artifacts["encoders"]
features = artifacts["features"]
sev_map = artifacts["sev_map"]
is_log_model = artifacts.get("is_log_model", True)

df_hourly = pd.read_csv(HOURLY_DATA_PATH)

app = FastAPI(
    title="Loss Payload Predictor",
    description="API prediksi loss_payload insiden jaringan berdasarkan model ML terlatih.",
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Skema input — field yang dikirim saat request
# --------------------------------------------------------------------------
class IncidentInput(BaseModel):
    site_id: str
    severity: str
    alarm_start_time: str
    duarasi_alaram: str
    payload: str
    baseline_payload: str
    rpmb: str
    availability_full: str
    regional: str
    day_type: Optional[str] = "Weekday"
    rootcausecategory: Optional[str] = "unknown"
    update_impact: Optional[str] = "1"
    url: Optional[str] = ""

    class Config:
        json_schema_extra = {
            "example": {
                "site_id": "SBS087",
                "severity": "Low",
                "alarm_start_time": "2026-04-29 23:57:36",
                "duarasi_alaram": "14:36:59",
                "payload": "0",
                "baseline_payload": "9750,617441",
                "rpmb": "3,259676896",
                "availability_full": "62,53472222",
                "regional": "KALIMANTAN",
                "day_type": "Weekday",
                "rootcausecategory": "Power",
                "update_impact": "1",
                "url": "SBS087",
            }
        }


# --------------------------------------------------------------------------
# Helper functions (identik dengan CELL 1 & CELL 19 pipeline)
# --------------------------------------------------------------------------
def duration_to_minutes(value):
    try:
        h, m, s = str(value).strip().split(":")
        return int(h) * 60 + int(m) + float(s) / 60
    except Exception:
        return np.nan


def count_impacted_sites(url_value):
    if pd.isna(url_value) or str(url_value).strip() == "":
        return 1
    return len(str(url_value).split(";"))


def parse_alarm_start_time(value: str):
    """Parse alarm_start_time. Prioritas ke format asli sistem monitoring/tiketing
    (DD/MM/YYYY HH.MM.SS), dengan fallback ke parser umum pandas (dayfirst=True)
    untuk jaga-jaga kalau ada variasi format lain (mis. ISO)."""
    # Format utama dari sistem sumber: '29/04/2026 23.53.30'
    parsed = pd.to_datetime(value, format="%d/%m/%Y %H.%M.%S", errors="coerce")
    if pd.notna(parsed):
        return parsed

    # Fallback: parser umum, dayfirst=True supaya '05/04/2026' dibaca 5 April, bukan Mei 4
    return pd.to_datetime(value, errors="coerce", dayfirst=True)


def encode_category(col: str, value: str) -> int:
    """Encode satu kolom kategorikal, fallback ke 'unknown' seperti CELL 19."""
    encoder = encoders[col]
    val_str = str(value)
    if val_str in encoder.classes_:
        return int(encoder.transform([val_str])[0])
    if "unknown" in encoder.classes_:
        return int(encoder.transform(["unknown"])[0])
    return 0


def predict_from_raw(raw_input: IncidentInput) -> float:
    row = {}

    # -- Preprocessing dasar --
    severity_key = str(raw_input.severity).lower()
    if severity_key not in sev_map:
        raise HTTPException(
            status_code=400,
            detail=f"severity '{raw_input.severity}' tidak dikenal. "
                   f"Pilihan valid: {list(sev_map.keys())}",
        )
    row["severity_num"] = sev_map[severity_key]

    row["durasi_menit"] = duration_to_minutes(raw_input.duarasi_alaram)

    try:
        row["baseline_payload"] = float(str(raw_input.baseline_payload).replace(",", "."))
        row["payload"] = float(str(raw_input.payload).replace(",", "."))
        row["rpmb"] = float(str(raw_input.rpmb).replace(",", "."))
        row["availability_full"] = float(str(raw_input.availability_full).replace(",", "."))
        row["update_impact"] = float(raw_input.update_impact)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Field numerik tidak valid: {e}")

    alarm_start = parse_alarm_start_time(raw_input.alarm_start_time)
    if pd.isna(alarm_start):
        raise HTTPException(
            status_code=400,
            detail=(
                f"alarm_start_time '{raw_input.alarm_start_time}' tidak bisa diparse. "
                f"Format yang didukung: 'DD/MM/YYYY HH.MM.SS' (contoh: '29/04/2026 23.53.30') "
                f"atau ISO 'YYYY-MM-DD HH:MM:SS' (contoh: '2026-04-29 23:53:30')."
            ),
        )
    row["hour"] = alarm_start.hour
    row["month"] = alarm_start.month
    row["is_peak_hour"] = int(8 <= alarm_start.hour <= 22)

    day_type_str = raw_input.day_type or "Weekday"
    row["impacted_sites_count"] = count_impacted_sites(raw_input.url)

    # -- Feature engineering --
    row["log_baseline_payload"] = np.log1p(row["baseline_payload"])
    row["log_payload"] = np.log1p(row["payload"])
    row["log_rpmb"] = np.log1p(row["rpmb"])
    row["durasi_x_severity"] = row["durasi_menit"] * row["severity_num"]

    # -- Lookup hourly_baseline (persis CELL 19: join site_id + hour + day_name) --
    hb_match = df_hourly[
        (df_hourly["site_id"] == raw_input.site_id)
        & (df_hourly["hour"] == row["hour"])
        & (df_hourly["day_name"].str.lower() == day_type_str.lower())
    ]["avg_payload_gb"]
    row["hourly_baseline"] = float(hb_match.values[0]) if not hb_match.empty else 0.0

    # -- Encoding kategorikal --
    encode_map = {
        "site_id": raw_input.site_id,
        "regional": raw_input.regional,
        "day_type": day_type_str,
        "rootcausecategory": raw_input.rootcausecategory,
    }
    for col, value in encode_map.items():
        row[col] = encode_category(col, value)

    # -- Susun ulang sesuai urutan fitur training & prediksi --
    X_new = pd.DataFrame([{f: row.get(f, np.nan) for f in features}])
    prediction_raw = float(model.predict(X_new)[0])
    final_pred = np.expm1(prediction_raw) if is_log_model else prediction_raw

    return round(final_pred, 2)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI belum tersedia</h1><p>File static/index.html tidak ditemukan.</p>", status_code=500)
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "n_features": len(features)}


@app.post("/predict")
def predict(data: IncidentInput):
    result = predict_from_raw(data)
    return {"predicted_loss_payload": result}
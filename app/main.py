"""
FastAPI service untuk model prediksi `loss_payload` insiden jaringan.

Logika di file ini SENGAJA dibuat konsisten dengan pipeline training
(`pipeline_loss_payload_cells_new.py` + cell tambahan gabung-data-baru),
supaya fitur yang dikirim ke model saat serving sama persis dengan fitur
yang dipakai saat training:
1. Preprocessing dasar (severity -> severity_num, durasi -> menit, dst).
2. Feature engineering (log1p, durasi_x_severity).
3. Lookup hourly_baseline dari df_hourly GABUNGAN (site_id + hour +
   day_name), dikonversi ke MB (avg_payload_gb * 1024).
4. Encoding kategorikal pakai encoder hasil training, fallback ke
   "unknown" (case-insensitive).
5. Prediksi -> expm1 kalau model dilatih di skala log (is_log_model).

REVISI (setelah retrain dengan data tambahan):
- "rpmb" (dan "log_rpmb") sudah dihapus total sejak revisi sebelumnya --
  tidak berubah di revisi ini.
- "update_impact" DIHAPUS dari request/field model -- importance-nya
  paling rendah di feature importance model hasil retrain (di luar
  top-10), dan tidak tersedia di sumber data baru, sehingga didrop dari
  FEATURES_FINAL saat training. Mengirim field ini sekarang tidak
  berpengaruh apa pun ke prediksi, jadi dihapus dari kontrak API supaya
  tidak menyesatkan.
- df_hourly SEKARANG DIGABUNG dari DUA file (file lama + file tambahan),
  PERSIS seperti proses di notebook training -- sebelumnya API cuma baca
  satu file (baseline_payload_hourly_weekly.csv saja), padahal model
  dilatih dari gabungan dua file. Kalau dibiarkan, hourly_baseline yang
  dihitung API bisa beda dari yang dipelajari model saat training. Kalau
  ada baris duplikat kunci (site_id+hour+day_name, umum terjadi karena
  dua file mencakup periode berbeda), diagregasi (rata-rata) sama seperti
  di notebook.
- encode_category SEKARANG case-insensitive -- sebelumnya pencocokan
  kategori (site_id/regional/day_type/rootcausecategory) sensitif huruf
  besar-kecil tanpa peringatan, jadi 'power' dan 'Power' dianggap dua
  kategori berbeda (yang satu jatuh ke fallback 'unknown' secara diam-
  diam). Form web sudah pakai dropdown untuk rootcausecategory/day_type
  jadi risikonya kecil dari situ, tapi endpoint ini tetap dijaga aman
  untuk pemanggil lain (mis. lewat /docs atau sistem lain).
- LOGIKA PREDIKSI DIPISAH ke compute_prediction() (menerima dict biasa,
  bukan IncidentInput), dipakai bareng oleh /predict (1 insiden) dan
  /predict-batch (banyak insiden dari file CSV/Excel, misal data sepekan
  atau sebulan) -- supaya keduanya dijamin konsisten, tidak ada dua
  versi logic yang bisa menghasilkan angka berbeda.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Konfigurasi path (bisa dioverride via environment variable saat run/docker)
# --------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "output_ml_new/best_incident_model.joblib")
HOURLY_DATA_PATH = os.environ.get("HOURLY_DATA_PATH", "baseline_payload_hourly_weekly.csv")
# Opsional -- kosongkan/hapus environment variable ini kalau memang cuma
# ada satu file hourly (mis. belum pernah retrain dengan data tambahan).
HOURLY_DATA_PATH_TAMBAHAN = os.environ.get(
    "HOURLY_DATA_PATH_TAMBAHAN", "baseline_payload_hourly_weekly_tambahan.csv"
)

HOURLY_JOIN_KEYS = ["site_id", "hour", "day_name"]

# Kolom yang WAJIB ada di file upload /predict-batch. Kolom opsional
# (day_type, rootcausecategory) punya default sendiri, lihat BATCH_DEFAULTS.
BATCH_REQUIRED_COLUMNS = [
    "site_id", "severity", "alarm_start_time", "duarasi_alaram",
    "payload", "baseline_payload", "availability_full", "regional",
]
BATCH_DEFAULTS = {"day_type": "Weekday", "rootcausecategory": "unknown"}


# --------------------------------------------------------------------------
# Load artifacts sekali saat startup
# --------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model artifact tidak ditemukan di '{MODEL_PATH}'. "
        f"Pastikan file 'best_incident_model.joblib' hasil training sudah "
        f"diletakkan di path tersebut (lihat README)."
    )

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
encoders = artifacts["encoders"]
features = artifacts["features"]
sev_map = artifacts["sev_map"]
is_log_model = artifacts.get("is_log_model", True)


def _load_combined_hourly(path_lama: str, path_tambahan: str) -> pd.DataFrame:
    """Gabung file hourly baseline lama + tambahan, PERSIS seperti cell
    [4.1]/[4.1b] di notebook training: kalau ada baris dengan kunci
    (site_id, hour, day_name) yang sama di kedua file, diagregasi
    (rata-rata avg_payload_gb) supaya jadi satu baris per kombinasi,
    bukan dipakai mentah-mentah (yang bisa menggandakan hasil join)."""
    frames = []
    if path_lama and os.path.exists(path_lama):
        frames.append(pd.read_csv(path_lama))
    if path_tambahan and os.path.exists(path_tambahan):
        frames.append(pd.read_csv(path_tambahan))

    if not frames:
        raise FileNotFoundError(
            f"Tidak ada file data hourly baseline yang ditemukan. Cek path "
            f"'{path_lama}' dan '{path_tambahan}' (lihat README / environment "
            f"variable HOURLY_DATA_PATH & HOURLY_DATA_PATH_TAMBAHAN)."
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(subset=HOURLY_JOIN_KEYS).any():
        combined = (
            combined.groupby(HOURLY_JOIN_KEYS, as_index=False)["avg_payload_gb"]
            .mean()
        )
    return combined


df_hourly = _load_combined_hourly(HOURLY_DATA_PATH, HOURLY_DATA_PATH_TAMBAHAN)

app = FastAPI(
    title="Loss Payload Predictor",
    description="API prediksi loss_payload insiden jaringan berdasarkan model ML terlatih.",
    version="1.1.0",
)


# --------------------------------------------------------------------------
# Skema input — field yang dikirim saat request /predict
# --------------------------------------------------------------------------
class IncidentInput(BaseModel):
    site_id: str
    severity: str
    alarm_start_time: str
    duarasi_alaram: str
    payload: str
    baseline_payload: str
    availability_full: str
    regional: str
    day_type: Optional[str] = "Weekday"
    rootcausecategory: Optional[str] = "unknown"

    class Config:
        json_schema_extra = {
            "example": {
                "site_id": "SBS087",
                "severity": "Low",
                "alarm_start_time": "2026-04-29 23:57:36",
                "duarasi_alaram": "14:36:59",
                "payload": "0",
                "baseline_payload": "9750,617441",
                "availability_full": "62,53472222",
                "regional": "KALIMANTAN",
                "day_type": "Weekday",
                "rootcausecategory": "Power",
            }
        }


class BatchResultRow(BaseModel):
    row: int
    site_id: Optional[str] = None
    predicted_loss_payload: Optional[float] = None
    error: Optional[str] = None


class BatchResult(BaseModel):
    count: int
    success_count: int
    error_count: int
    results: list[BatchResultRow]


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def duration_to_minutes(value):
    """Konversi durasi 'HH:MM:SS' -> menit. Mendukung titik dua (:) MAUPUN
    titik (.) sebagai pemisah -- beberapa sumber data mentah (mis. export
    langsung dari sistem monitoring) memakai format 'HH.MM.SS', bukan
    'HH:MM:SS'. Titik dinormalisasi jadi titik dua dulu sebelum di-split,
    jadi '14.36.59' dan '14:36:59' menghasilkan angka yang sama."""
    try:
        cleaned = str(value).strip().replace(".", ":")
        h, m, s = cleaned.split(":")
        return int(h) * 60 + int(m) + float(s) / 60
    except Exception:
        return np.nan


def parse_alarm_start_time(value: str):
    """Parse alarm_start_time. Prioritas ke format asli sistem monitoring/tiketing
    (DD/MM/YYYY HH.MM.SS), dengan fallback ke parser umum pandas (dayfirst=True)
    untuk jaga-jaga kalau ada variasi format lain (mis. ISO)."""
    parsed = pd.to_datetime(value, format="%d/%m/%Y %H.%M.%S", errors="coerce")
    if pd.notna(parsed):
        return parsed
    return pd.to_datetime(value, errors="coerce", dayfirst=True)


def encode_category(col: str, value: str) -> int:
    """Encode satu kolom kategorikal, case-insensitive, fallback ke
    'unknown' kalau kategorinya belum pernah dilihat saat training."""
    encoder = encoders[col]
    val_str = str(value)
    lookup = {kelas.lower(): kelas for kelas in encoder.classes_}
    if val_str.lower() in lookup:
        return int(encoder.transform([lookup[val_str.lower()]])[0])
    if "unknown" in encoder.classes_:
        return int(encoder.transform(["unknown"])[0])
    return 0


def compute_prediction(raw: dict) -> float:
    """Logika inti prediksi, menerima dict field mentah (bukan Pydantic
    model) supaya bisa dipakai bareng oleh /predict (1 baris, dari
    IncidentInput.model_dump()) maupun /predict-batch (banyak baris, dari
    satu baris file CSV/Excel yang diupload). Melempar ValueError kalau
    inputnya tidak valid -- pemanggil yang menentukan bagaimana errornya
    ditangani (single: langsung gagal; batch: dicatat per baris, baris
    lain tetap lanjut diproses)."""
    row = {}

    severity_key = str(raw.get("severity", "")).lower()
    if severity_key not in sev_map:
        raise ValueError(
            f"severity '{raw.get('severity')}' tidak dikenal. "
            f"Pilihan valid: {list(sev_map.keys())}"
        )
    row["severity_num"] = sev_map[severity_key]

    row["durasi_menit"] = duration_to_minutes(raw.get("duarasi_alaram"))
    if pd.isna(row["durasi_menit"]):
        raise ValueError(
            f"duarasi_alaram '{raw.get('duarasi_alaram')}' tidak bisa diparse. "
            f"Format yang didukung: 'HH:MM:SS' atau 'HH.MM.SS' (contoh: '14:36:59' atau '14.36.59')."
        )

    try:
        row["baseline_payload"] = float(str(raw.get("baseline_payload")).replace(",", "."))
        row["payload"] = float(str(raw.get("payload")).replace(",", "."))
        row["availability_full"] = float(str(raw.get("availability_full")).replace(",", "."))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Field numerik tidak valid: {e}")

    alarm_start = parse_alarm_start_time(raw.get("alarm_start_time"))
    if pd.isna(alarm_start):
        raise ValueError(
            f"alarm_start_time '{raw.get('alarm_start_time')}' tidak bisa diparse. "
            f"Format yang didukung: 'DD/MM/YYYY HH.MM.SS' (contoh: '29/04/2026 23.53.30') "
            f"atau ISO 'YYYY-MM-DD HH:MM:SS' (contoh: '2026-04-29 23:53:30')."
        )
    row["hour"] = alarm_start.hour
    row["month"] = alarm_start.month
    row["is_peak_hour"] = int(8 <= alarm_start.hour <= 22)

    day_type_str = raw.get("day_type") or "Weekday"

    row["log_baseline_payload"] = np.log1p(row["baseline_payload"])
    row["log_payload"] = np.log1p(row["payload"])
    row["durasi_x_severity"] = row["durasi_menit"] * row["severity_num"]

    # -- Lookup hourly_baseline dari df_hourly GABUNGAN (lihat _load_combined_hourly) --
    hb_match = df_hourly[
        (df_hourly["site_id"] == raw.get("site_id"))
        & (df_hourly["hour"] == row["hour"])
        & (df_hourly["day_name"].str.lower() == day_type_str.lower())
    ]["avg_payload_gb"]
    row["hourly_baseline"] = (float(hb_match.values[0]) * 1024) if not hb_match.empty else 0.0

    encode_map = {
        "site_id": raw.get("site_id"),
        "regional": raw.get("regional"),
        "day_type": day_type_str,
        "rootcausecategory": raw.get("rootcausecategory") or "unknown",
    }
    for col, value in encode_map.items():
        row[col] = encode_category(col, value)

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
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": len(features),
        "hourly_rows": len(df_hourly),
    }


@app.post("/predict")
def predict(data: IncidentInput):
    try:
        result = compute_prediction(data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_loss_payload": result}


@app.post("/predict-batch", response_model=BatchResult)
async def predict_batch(file: UploadFile = File(...)):
    """Prediksi banyak insiden sekaligus dari file CSV/Excel yang diupload
    -- cocok untuk cek data sepekan/sebulan tanpa isi form satu-satu.
    Kolom yang dibutuhkan sama seperti /predict: site_id, severity,
    alarm_start_time, duarasi_alaram, payload, baseline_payload,
    availability_full, regional (wajib), day_type & rootcausecategory
    (opsional, ada default). Satu baris gagal TIDAK menggagalkan baris
    lain -- errornya dicatat per baris di kolom 'error'."""
    filename = (file.filename or "").lower()
    raw_bytes = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(raw_bytes), dtype=str)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(raw_bytes), dtype=str)
        else:
            raise HTTPException(
                status_code=400,
                detail="Format file tidak didukung. Upload file .csv, .xlsx, atau .xls.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca file: {e}")

    missing_cols = [c for c in BATCH_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom wajib berikut tidak ada di file: {missing_cols}. "
                   f"Kolom yang dibutuhkan: {BATCH_REQUIRED_COLUMNS}",
        )

    for col, default in BATCH_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    results: list[BatchResultRow] = []
    success_count = 0
    for idx, series in df.iterrows():
        raw = series.to_dict()
        site_id = raw.get("site_id")
        try:
            pred = compute_prediction(raw)
            results.append(BatchResultRow(row=idx + 1, site_id=site_id, predicted_loss_payload=pred))
            success_count += 1
        except ValueError as e:
            results.append(BatchResultRow(row=idx + 1, site_id=site_id, error=str(e)))

    return BatchResult(
        count=len(results),
        success_count=success_count,
        error_count=len(results) - success_count,
        results=results,
    )
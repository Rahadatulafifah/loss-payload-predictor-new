# ==============================================================================
# PIPELINE PREDIKSI `loss_payload` INSIDEN JARINGAN — VERSI FORMAT CELL
# ==============================================================================
# File ini dibagi memakai penanda cell "# %%" (dikenali VSCode Python
# Interactive Window & Jupyter). Setiap cell berisi KODE LANGSUNG (top-level,
# TIDAK dibungkus dalam fungsi lalu dipanggil belakangan di main()), supaya
# saat satu cell dijalankan (Ctrl+Enter / "Run Cell" di VSCode), output-nya
# LANGSUNG muncul tepat di bawah cell itu juga — persis seperti contoh yang
# Anda tunjukkan sebelumnya.
#
# CARA PAKAI DI VSCODE:
# 1. Pastikan extension "Python" (Microsoft) aktif.
# 2. Buka file ini, akan muncul tulisan "Run Cell" / "Run Below" di atas
#    tiap "# %%".
# 3. Jalankan cell dari atas ke bawah SECARA BERURUTAN (cell di bawah
#    butuh variabel yang dibuat cell di atasnya, sama seperti notebook).
# 4. Output tiap cell akan muncul di panel "Python Interactive" tepat
#    setelah cell tersebut, tidak perlu menunggu seluruh file selesai.
#
# CATATAN: Logika perhitungan/model SAMA PERSIS dengan pipeline sebelumnya
# (pipeline_loss_payload_fixed.py) — yang berubah hanya cara penyajian
# kode: dari fungsi+main() menjadi cell-cell top-level dengan penjelasan
# di atas tiap cell.
# ==============================================================================
#
# ==============================================================================
# PETA STRUKTUR PIPELINE (7 BAGIAN BESAR)
# ==============================================================================
# Tidak ada satu pun logika/rumus/urutan komputasi yang diubah di bawah ini.
# Yang ditambahkan hanyalah header penanda bagian besar (supaya struktur
# pipeline terlihat jelas), dan CELL 5 (dulu satu cell campuran) DIPECAH
# jadi CELL 5A (Preprocessing) & CELL 5B (Feature Engineering) — isi kodenya
# sama persis, cuma dipisah berdasarkan mana yang "membersihkan data" vs
# mana yang "membuat fitur baru".
#
#   1. DATA COLLECTION          -> CELL 2
#   2. EDA (Exploratory Data
#      Analysis)                -> CELL 3, CELL 4, CELL 4B  (audit data mentah,
#                                   termasuk dokumentasi kolom dipakai/tidak)
#                                   CELL 6, CELL 7  (eksplorasi data setelah
#                                   preprocessing & feature engineering —
#                                   perlu kolom log/derived yang baru dibuat
#                                   di CELL 5, makanya posisinya di sini)
#   3. PREPROCESSING             -> CELL 5A (pembersihan dasar)
#                                   CELL 8   (encoding kategorikal — lanjutan
#                                   preprocessing, posisinya tetap setelah
#                                   EDA seperti kode aslinya)
#   4. FEATURE ENGINEERING       -> CELL 5B (termasuk penggabungan dengan
#                                   data hourly_weekly -> fitur baru
#                                   `hourly_baseline`)
#   5. MODEL TRAINING            -> CELL 9, CELL 10, CELL 11, CELL 12, CELL 13
#   6. EVALUASI                  -> CELL 14, CELL 15, CELL 16, CELL 17, CELL 18
#   7. REPORTING                 -> CELL 20 (simpan model), CELL 21 (laporan)
#      (+ CELL 19 = contoh inference, bonus di luar 7 bagian utama, posisinya
#       di antara Evaluasi dan Reporting karena dia "mencoba" model final)
#
# ==============================================================================
# RINGKASAN PERUBAHAN SETELAH DIGABUNG (MERGE) DENGAN hourly_weekly.csv
# ==============================================================================
# Dibanding versi sebelumnya (hanya pakai 1 sumber data: data inap saja),
# perubahan konkret ada di titik-titik berikut (masing-masing ditandai
# komentar "PERUBAHAN SETELAH MERGER" persis di lokasinya):
#   - CELL 0  : tambah konstanta FILE_HOURLY, tambah "hourly_baseline" ke
#               FEATURES_FINAL.
#   - CELL 2  : tambah proses load df_hourly dari FILE_HOURLY.
#   - CELL 3  : tambah pengecekan missing value untuk df_hourly.
#   - CELL 5B : tambah proses LEFT JOIN df_eda dengan df_hourly (berdasarkan
#               site_id + hour + hari) untuk membuat fitur baru
#               `hourly_baseline`.
#   - CELL 19 : tambah logika lookup ke df_hourly saat membuat 1 baris
#               input baru untuk contoh inference.
#   - CELL 21 : laporan akhir diperluas jadi menyebutkan 2 sumber file
#               (File Utama & File Pendukung), bukan cuma 1 file seperti
#               laporan versi sebelumnya.
# ==============================================================================
#
# ==============================================================================
# RINGKASAN REVISI TERBARU (permintaan review kedua)
# ==============================================================================
# 1. LOG_BASELINE_PAYLOAD dijelaskan lebih detail (apa/kenapa/dari mana) —
#    lihat komentar panjang tepat di atas baris pembuatannya di CELL 5B.
# 2. `rpmb` dan turunannya (`log_rpmb`) DIHAPUS TOTAL dari pipeline —
#    kolom ini dianggap tidak relevan untuk memprediksi loss_payload.
#    Dihapus dari: FEATURES_FINAL (CELL 0), konversi numerik (CELL 5A),
#    transformasi log & tabel korelasi (CELL 5B, CELL 7), contoh
#    inference (CELL 19).
# 3. `url` DIKEMBALIKAN, begitu juga fungsi `count_impacted_sites()`
#    (CELL 1) yang menghitung jumlah site terdampak dari kolom `url`
#    (dipisah ';'). Hasilnya, kolom `impacted_sites_count`, TETAP DIBUAT
#    di CELL 5B (Feature Engineering) untuk keperluan EDA/audit -- TAPI
#    SENGAJA TIDAK dimasukkan ke `FEATURES_FINAL`, jadi TIDAK ikut dilatih
#    ke model dan TIDAK muncul di feature importance. Kolom `url` cuma
#    dipakai sebagai bahan mentah untuk logika perhitungan itu, bukan
#    fitur model.
# 4. MERGE hourly_baseline DIPASTIKAN TIDAK MERATA-RATAKAN weekday &
#    weekend jadi satu angka. Baris test/insiden dengan day_type
#    "weekday" hanya boleh dicocokkan (JOIN) ke baris baseline yang
#    day_name-nya juga "weekday", begitu juga "weekend" -> "weekend".
#    Ini sudah jadi cara kerja merge sejak awal (join memakai kolom
#    day_name_type == day_name sebagai salah satu key), tapi di revisi
#    ini logikanya dipertegas dengan validasi eksplisit + penjelasan
#    detail di CELL 5B supaya tidak disalahartikan sebagai "dirata-rata".
# 5. SATUAN payload pada data hourly baseline diganti dari GB ke MB.
#    Kolom sumber `avg_payload_gb` dikonversi eksplisit (dikali 1024)
#    sebelum dipakai, lalu diberi nama `avg_payload_mb` agar jelas
#    satuannya, baru kemudian di-join dan disimpan sebagai fitur
#    `hourly_baseline` (dalam MB). Alur pipeline (7 tahap besar) TIDAK
#    berubah, hanya isi & satuan datanya.
# 6. CELL 4B (BARU): dokumentasi eksplisit kolom mana dari 19 kolom
#    mentah yang dipakai vs tidak, dan ALASANNYA masing-masing (bukan
#    cuma "missing value" -- ada juga alasan redundan, ID unik, teks
#    bebas, dsb). `alarm_clear_time` dipakai di sini untuk audit
#    konsistensi durasi terhadap `duarasi_alaram`, tapi TIDAK dijadikan
#    fitur model tersendiri karena redundan dengan durasi_menit.
# ==============================================================================

# %%
# ------------------------------------------------------------------------
# CELL 0: IMPORT LIBRARY & KONFIGURASI GLOBAL
# ------------------------------------------------------------------------
# Penjelasan tiap library:
# - os          : membuat folder output, mengecek ukuran file yang disimpan.
# - warnings    : mematikan warning yang tidak penting agar output bersih.
# - numpy       : operasi numerik (log1p, expm1, array, dsb).
# - pandas      : struktur data tabel (DataFrame) untuk baca/olah data.
# - matplotlib  : membuat grafik/plot (disimpan sebagai gambar .png).
# - seaborn     : mempercantik tampilan grafik matplotlib.
# - joblib      : menyimpan (serialize) model machine learning ke file.
# - sklearn     : algoritma machine learning klasik (split data, encoding,
#                 metrik evaluasi, Random Forest, pencarian hyperparameter).
# - xgboost     : algoritma gradient boosting XGBoost.
#
# Variabel konfigurasi (FILE_MAIN, TARGET, FEATURES_FINAL, dst) dikumpulkan
# di satu tempat supaya mudah diubah tanpa mencari-cari di dalam kode, dan
# dipakai berulang kali oleh cell-cell di bawah.
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

FILE_MAIN = "inap_ticketing_incident_loss_payload_2026.xlsx"
FILE_HOURLY = "baseline_payload_hourly_weekly.csv"  # << PERUBAHAN SETELAH MERGER: sumber data ke-2, tidak ada di versi sebelumnya (yang hanya pakai data inap)
TARGET = "loss_payload"
OUTPUT_DIR = "output_ml_new"
RANDOM_STATE = 42

SEV_MAP = {"low": 1, "minor": 2, "major": 3, "critical": 4}

# << REVISI: "rpmb"/"log_rpmb" dihapus dari daftar fitur -- lihat
# "RINGKASAN REVISI TERBARU" di atas. "impacted_sites_count" TETAP
# DIHITUNG di CELL 5B (dari url), TAPI SENGAJA TIDAK dimasukkan ke sini
# -- jadi cuma untuk EDA/audit, tidak ikut dilatih ke model / tidak
# muncul di feature importance.
FEATURES_FINAL = [
    "severity_num", "durasi_menit", "baseline_payload", "payload",
    "availability_full", "update_impact", "hour",
    "month", "is_peak_hour", "regional", "site_id", "day_type",
    "rootcausecategory", "log_baseline_payload", "log_payload",
    "durasi_x_severity",
    "hourly_baseline",  # << PERUBAHAN SETELAH MERGER: fitur baru dari hasil join dengan FILE_HOURLY, satuan sudah dalam MB (lihat CELL 5B)
]

CAT_COLS = ["site_id", "regional", "day_type", "rootcausecategory"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Konfigurasi siap. Folder output:", OUTPUT_DIR)


# %%
# ------------------------------------------------------------------------
# CELL 1: FUNGSI BANTU (HELPER FUNCTIONS)
# ------------------------------------------------------------------------
# Tiga fungsi kecil ini TETAP dibuat sebagai fungsi (bukan kode flat)
# karena dipakai berulang kali di banyak baris berbeda lewat `.apply()` —
# kalau ditulis ulang manual di tiap tempat pemakaiannya, kode jadi
# panjang dan rawan salah ketik. Fungsi-fungsi ini sendiri tidak butuh
# ditampilkan outputnya (tidak ada yang perlu di-print), karena baru
# menghasilkan nilai kalau dipanggil pada data di cell-cell berikutnya.
#
# - to_numeric_safe   : ubah teks angka format koma ("12,5") jadi float
#                        Python (12.5). Gagal konversi -> NaN (aman, tidak
#                        bikin program berhenti).
# - duration_to_minutes: ubah teks durasi "HH:MM:SS" jadi total menit.
# - count_impacted_sites: hitung jumlah site terdampak dari kolom `url`
#                        (dipisah ';'), kosong/NaN dianggap 1 site.
#                        << REVISI: dikembalikan -- dipakai untuk
#                        menghitung `impacted_sites_count` di CELL 5B,
#                        TAPI hasilnya hanya untuk EDA/audit, sengaja
#                        TIDAK dimasukkan ke FEATURES_FINAL (tidak ikut
#                        melatih model).
def to_numeric_safe(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


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


print("Fungsi bantu siap dipakai: to_numeric_safe, duration_to_minutes, count_impacted_sites")


# %%
# ==============================================================================
# BAGIAN 1: DATA COLLECTION
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 2 (TAHAP 1): LOAD DATA
# ------------------------------------------------------------------------
# Memuat dua sumber data mentah dari disk:
# - df_raw    : data insiden utama dari Excel. Semua kolom dibaca sebagai
#               string (dtype=str) supaya format angka desimal koma
#               ("12,5") dan format durasi teks ("02:15:30") tidak rusak
#               saat dibaca pandas.
# - df_hourly : data baseline payload rata-rata per jam/hari/site (CSV).
#               << PERUBAHAN SETELAH MERGER: sumber data ini BARU. Versi
#               sebelumnya cuma load 1 file (data inap). Sekarang load
#               2 file lalu digabung nanti di CELL 5B.
# Ini gerbang masuk (entry point) seluruh pipeline — semua cell berikutnya
# butuh kedua data ini.
df_raw = pd.read_excel(FILE_MAIN, dtype=str)
df_hourly = pd.read_csv(FILE_HOURLY)  # << PERUBAHAN SETELAH MERGER

print("Shape Data Utama  :", df_raw.shape)
print("Shape Data Hourly :", df_hourly.shape)
print("Kolom Data Utama  :", list(df_raw.columns))


# %%
# ==============================================================================
# BAGIAN 2: EDA (EXPLORATORY DATA ANALYSIS) — BAGIAN 2a: AUDIT DATA MENTAH
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 3 (TAHAP 2): CEK MISSING VALUE
# ------------------------------------------------------------------------
# Mengecek dan menampilkan jumlah + persentase nilai kosong (NaN) di
# setiap kolom, untuk data utama maupun data hourly. Tahap ini murni
# diagnosa/audit kualitas data, TIDAK mengubah data apapun. Penting
# dilakukan sejak awal supaya missing value yang tidak terdeteksi tidak
# diam-diam membuat model bias atau error saat training.
# << PERUBAHAN SETELAH MERGER: blok pengecekan untuk df_hourly di bawah
# ini BARU — versi sebelumnya cuma cek df_raw karena cuma ada 1 sumber data.
missing_raw = df_raw.isnull().sum()
missing_raw_pct = (missing_raw / len(df_raw) * 100).round(1)
table_raw = pd.DataFrame({"missing": missing_raw, "pct": missing_raw_pct})

print("--- Missing Value: Data Utama (Incident) ---")
if missing_raw.sum() > 0:
    print(table_raw[table_raw["missing"] > 0].to_string())
else:
    print("Tidak ada missing value pada data utama.")
print()

missing_hourly = df_hourly.isnull().sum()
missing_hourly_pct = (missing_hourly / len(df_hourly) * 100).round(1)
table_hourly = pd.DataFrame({"missing": missing_hourly, "pct": missing_hourly_pct})

print("--- Missing Value: Data Hourly Baseline ---")
if missing_hourly.sum() > 0:
    print(table_hourly[table_hourly["missing"] > 0].to_string())
else:
    print("Tidak ada missing value pada data hourly baseline.")


# %%
# ------------------------------------------------------------------------
# CELL 4 (TAHAP 3): CEK KONSISTENSI TARGET vs BASELINE_PAYLOAD
# ------------------------------------------------------------------------
# Mengecek konsistensi antara kolom target (`loss_payload`) dan kolom
# `baseline_payload` setelah keduanya dipaksa jadi numerik: berapa baris
# NaN di keduanya sekaligus, dan berapa baris NaN hanya di salah satu
# (menandakan kemungkinan anomali di sumber data). Kedua kolom ini krusial
# karena target adalah yang mau diprediksi, dan baseline_payload adalah
# fitur utama.
loss_na = to_numeric_safe(df_raw[TARGET]).isna()
base_na = to_numeric_safe(df_raw["baseline_payload"]).isna()

print("loss_payload NaN         :", loss_na.sum())
print("baseline_payload NaN     :", base_na.sum())
print("Keduanya NaN bersamaan   :", (loss_na & base_na).sum())
print("Hanya salah satu NaN     :", (loss_na ^ base_na).sum())


# %%
# ------------------------------------------------------------------------
# CELL 4B (TAHAP 3b): KOLOM APA SAJA YANG DIPAKAI vs TIDAK, DAN KENAPA
# ------------------------------------------------------------------------
# Data mentah (df_raw) punya 19 kolom. Sebelum masuk Preprocessing lebih
# jauh, di sini didokumentasikan SECARA EKSPLISIT kolom mana yang dipakai
# ke pipeline dan mana yang tidak, beserta ALASANNYA masing-masing --
# supaya keputusan "dibuang" atau "dipakai" tidak tersembunyi di
# tengah-tengah kode.
#
# === KOLOM YANG DIPAKAI (14 kolom + url utk fitur turunan) ===
#   site_id            -> fitur kategorikal (encoded) + kunci JOIN ke
#                          df_hourly untuk membuat hourly_baseline.
#   severity           -> diubah ke severity_num (SEV_MAP) -> fitur model,
#                          juga dipakai di fitur interaksi durasi_x_severity.
#   alarm_start_time   -> sumber fitur waktu: hour, month, day_name_type
#                          (weekday/weekend), is_peak_hour.
#   alarm_clear_time   -> DIPAKAI, tapi BUKAN sebagai fitur model
#                          tersendiri -- dipakai untuk AUDIT/cross-check
#                          konsistensi durasi terhadap duarasi_alaram (lihat
#                          blok "cross-check" di bawah). Tidak dijadikan
#                          fitur terpisah karena informasinya REDUNDAN
#                          dengan durasi_menit (durasi sudah didapat dari
#                          duarasi_alaram) -- kalau keduanya dimasukkan
#                          sebagai fitur, model hanya belajar hal yang
#                          sama dua kali (multikolinearitas) tanpa
#                          menambah sinyal baru.
#   duarasi_alaram     -> durasi_menit (fitur model).
#   payload            -> fitur model + log_payload.
#   baseline_payload   -> fitur model + log_baseline_payload.
#   loss_payload       -> TARGET (yang diprediksi, bukan fitur input).
#   availability_full  -> fitur model (missing kecil, ~0.7% -> baris ikut
#                          terbuang natural lewat dropna() saat CELL 9).
#   regional           -> fitur kategorikal (encoded).
#   day_type           -> fitur kategorikal (encoded) -- kolom mentah ini
#                          BERBEDA dari `day_name_type` (turunan dari
#                          alarm_start_time yang dipakai khusus untuk
#                          kunci JOIN weekday/weekend ke df_hourly).
#   rootcausecategory  -> fitur kategorikal (encoded).
#   update_impact      -> fitur model.
#   url                -> HANYA dipakai untuk menghitung fitur turunan
#                          `impacted_sites_count` (CELL 5B). Kolom `url`
#                          itu sendiri maupun `impacted_sites_count`
#                          SENGAJA TIDAK dimasukkan ke FEATURES_FINAL --
#                          jadi cuma untuk EDA/audit, tidak ikut dilatih
#                          ke model.
#
# === KOLOM YANG TIDAK DIPAKAI (5 kolom) ===
#   ticket_id          -> ID unik administratif per tiket (1 nilai unik
#                          per baris). BUKAN karena missing value, tapi
#                          karena ID unik tidak merepresentasikan
#                          karakteristik insiden apa pun -- kalau
#                          di-encode malah berisiko jadi bentuk data
#                          leakage/overfitting (model "menghafal" ID,
#                          bukan belajar pola).
#   rpmb               -> DIHAPUS sesuai keputusan eksplisit. Selain
#                          alasan bisnis, secara data korelasinya ke
#                          target juga SANGAT LEMAH (vs target asli
#                          ~-0.007, vs log target ~-0.079) dan punya
#                          missing value (~0.6%) -- kombinasi lemah +
#                          tidak esensial membuatnya aman dibuang.
#   rootcausedetail    -> deskripsi bebas/lebih rinci dari
#                          rootcausecategory yang sudah dipakai. Bukan
#                          soal missing value, tapi granularitas kategori
#                          teksnya berpotensi sangat banyak & tidak
#                          konsisten (free text), sehingga rawan membuat
#                          Label Encoding tidak stabil (kategori baru
#                          terus muncul saat data bertambah) tanpa
#                          menambah sinyal baru dibanding rootcausecategory.
#   order_status       -> status administratif tiket (mis. open/closed),
#                          bukan karakteristik TEKNIS insiden yang
#                          menjelaskan besarnya loss_payload.
#   sitetype           -> tidak dipakai di pipeline SAAT INI (bukan
#                          karena missing value) -- ini kandidat fitur
#                          yang masuk akal untuk eksplorasi lanjutan
#                          (mis. tipe site bisa berhubungan dengan pola
#                          traffic), tapi belum divalidasi dampaknya di
#                          revisi ini. Kalau mau ditambahkan sebagai
#                          fitur model, beri tahu supaya bisa disiapkan
#                          encoding-nya juga.
#
# -- Cross-check alarm_clear_time vs duarasi_alaram (audit, bukan fitur) --
# Tujuannya memverifikasi bahwa kolom teks durasi (duarasi_alaram) yang
# dipakai sebagai sumber durasi_menit itu KONSISTEN dengan selisih waktu
# alarm_clear_time - alarm_start_time. Kalau banyak yang tidak konsisten,
# itu sinyal kualitas data yang perlu diketahui (meski tetap dipakai
# duarasi_alaram sebagai sumber utama, sesuai desain pipeline sejak awal).
_start_dt = pd.to_datetime(df_raw["alarm_start_time"], errors="coerce")
_clear_dt = pd.to_datetime(df_raw["alarm_clear_time"], errors="coerce")
_durasi_dari_clear = (_clear_dt - _start_dt).dt.total_seconds() / 60
_durasi_dari_teks = df_raw["duarasi_alaram"].apply(duration_to_minutes)

_selisih = (_durasi_dari_clear - _durasi_dari_teks).abs()
_valid_check = _selisih.notna()
_toleransi_menit = 1  # toleransi pembulatan kecil

print(f"Baris yang bisa dicek (kedua durasi valid) : {_valid_check.sum()} dari {len(df_raw)}")
print(f"Baris konsisten (selisih <= {_toleransi_menit} menit)   : "
      f"{(_selisih[_valid_check] <= _toleransi_menit).sum()}")
print(f"Baris TIDAK konsisten (selisih > {_toleransi_menit} menit) : "
      f"{(_selisih[_valid_check] > _toleransi_menit).sum()}")
print("Catatan: durasi_menit (fitur model) tetap bersumber dari "
      "duarasi_alaram, BUKAN dari alarm_clear_time -- cek di atas murni "
      "audit kualitas data, alarm_clear_time tidak dijadikan fitur "
      "terpisah karena redundan dengan durasi_menit.")


# %%
# ------------------------------------------------------------------------
# CELL 4C (TAHAP 3c): KOLOM DATA HOURLY BASELINE (df_hourly) -- DIPAKAI vs TIDAK
# ------------------------------------------------------------------------
# df_hourly (FILE_HOURLY) punya 7 kolom: yearweek, hour, site_id, remark,
# avg_payload_gb, avg_traffic_erl, day_name. Sama seperti df_raw, di sini
# didokumentasikan eksplisit kolom mana yang dipakai dan alasannya.
#
# === DIPAKAI (3 kolom) ===
#   site_id        -> kunci JOIN ke df_eda (bareng hour & day_name).
#   hour            -> kunci JOIN ke df_eda.
#   day_name        -> kunci JOIN ke df_eda (dicocokkan ke day_name_type
#                       insiden -- weekday hanya ketemu weekday, weekend
#                       hanya ketemu weekend; lihat penjelasan detail di
#                       CELL 5B, TIDAK dirata-ratakan).
#   avg_payload_gb  -> nilai baseline payload itu sendiri. DIKONVERSI ke
#                       MB (dikali 1024) sebelum dipakai -> jadi fitur
#                       `hourly_baseline` (lihat CELL 5B).
#
# === TIDAK DIPAKAI (3 kolom) ===
#   yearweek        -> menurut hasil audit missing value sebelumnya,
#                       kolom ini nyaris 100% kosong pada data yang
#                       pernah diperiksa -- sekalipun terisi, granularitas
#                       per-minggu ini tidak dibutuhkan karena baseline di
#                       pipeline ini memang dimaksudkan sebagai pola
#                       historis rata-rata per site+jam+jenis hari
#                       (weekday/weekend), bukan per minggu spesifik.
#   remark          -> kolom catatan/anotasi bebas (freetext), bukan
#                       angka atau kategori yang bisa langsung dipakai
#                       sebagai fitur; isinya juga tidak terstandarisasi.
#   avg_traffic_erl -> metrik traffic dalam satuan Erlang, BEDA dimensi
#                       dari payload (Erlang mengukur intensitas trafik
#                       suara/panggilan, bukan volume data). Tidak
#                       dipakai di pipeline ini karena target (loss_payload)
#                       adalah kehilangan payload data, bukan trafik
#                       suara -- tapi ini kandidat fitur tambahan yang
#                       masuk akal untuk eksplorasi lanjutan kalau
#                       relevan dengan jenis insiden tertentu.
print("Kolom df_hourly dipakai   : site_id, hour, day_name, avg_payload_gb (dikonversi ke MB)")
print("Kolom df_hourly TIDAK dipakai : yearweek, remark, avg_traffic_erl")


# %%
# ==============================================================================
# BAGIAN 3: PREPROCESSING
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 5A (TAHAP 4a): PREPROCESSING — PEMBERSIHAN DATA DASAR
# ------------------------------------------------------------------------
# Ini murni PEMBERSIHAN data mentah (teks, tanggal, durasi) menjadi bentuk
# yang valid/konsisten — BELUM membuat fitur baru untuk model (itu baru
# terjadi di CELL 5B: Feature Engineering).
#
# Catatan: sebelumnya cell ini (CELL 5A + 5B) digabung jadi satu cell besar
# tanpa pemisahan eksplisit "preprocessing" vs "feature engineering". Di
# sini kodenya DIPECAH jadi dua cell — isi/logikanya sama persis, cuma
# batasnya dibuat jelas.
#
# Langkah di dalamnya:
# a. Target & kolom numerik dasar: target diubah ke numerik, baris tanpa
#    target dibuang (tidak bisa dipakai supervised learning). Kolom
#    numerik lain (payload, baseline_payload, rpmb, availability_full,
#    update_impact) juga diubah ke numerik.
# b. Fitur waktu dasar: hour, month, day_name_type (weekday/weekend),
#    is_peak_hour (1 jika jam 08:00-22:00) — hasil parsing tanggal mentah,
#    belum "fitur turunan" dalam arti feature engineering.
# c. Durasi insiden: durasi_menit dari teks "HH:MM:SS" -> angka menit.
#
# << REVISI: "rpmb" DIHAPUS dari daftar kolom yang dikonversi ke numerik,
# karena kolom ini sudah tidak dipakai sama sekali di pipeline.
# "impacted_sites_count" TIDAK dikonversi di sini -- kolom ini dihitung
# lewat fungsi count_impacted_sites(url) di CELL 5B, bukan kolom mentah.
df_eda = df_raw.copy()

# -- Target & kolom numerik dasar --
df_eda[TARGET] = to_numeric_safe(df_eda[TARGET])
baris_sebelum = len(df_eda)
df_eda = df_eda[df_eda[TARGET].notna()].copy()
print(f"Baris sebelum buang target kosong : {baris_sebelum}")
print(f"Baris setelah buang target kosong : {len(df_eda)} "
      f"(terbuang: {baris_sebelum - len(df_eda)})")

for col in ["payload", "baseline_payload", "availability_full", "update_impact"]:
    df_eda[col] = to_numeric_safe(df_eda[col])

# -- Fitur waktu (parsing dasar) --
alarm_dt = pd.to_datetime(df_eda["alarm_start_time"], errors="coerce")
df_eda["hour"] = alarm_dt.dt.hour.fillna(0).astype(int)
df_eda["month"] = alarm_dt.dt.month
df_eda["day_name_type"] = alarm_dt.dt.dayofweek.apply(lambda x: "weekend" if x >= 5 else "weekday")
df_eda["is_peak_hour"] = df_eda["hour"].between(8, 22).astype(int)
print("Fitur waktu dibuat: hour, month, day_name_type, is_peak_hour")

# -- Durasi insiden --
df_eda["durasi_menit"] = df_eda["duarasi_alaram"].apply(duration_to_minutes)
print(f"Fitur durasi_menit -> min={df_eda['durasi_menit'].min():.1f}, "
      f"median={df_eda['durasi_menit'].median():.1f}, "
      f"max={df_eda['durasi_menit'].max():.1f}")

print(f"Preprocessing selesai. Total kolom pada df_eda saat ini: {df_eda.shape[1]}")


# %%
# ==============================================================================
# BAGIAN 4: FEATURE ENGINEERING
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 5B (TAHAP 4b): FEATURE ENGINEERING — MEMBUAT FITUR BARU
# ------------------------------------------------------------------------
# Melanjutkan df_eda hasil CELL 5A (preprocessing), di sini fitur-fitur
# BARU dibuat/diturunkan dari kolom yang sudah bersih:
# a. Gabung dengan baseline per jam (left join site_id+hour+hari) supaya
#    jumlah baris tidak berubah; hasil dinamakan hourly_baseline.
#    << PERUBAHAN SETELAH MERGER: langkah join ini BARU. Di versi
#    sebelumnya (hanya data inap) tidak ada df_hourly sama sekali,
#    sehingga fitur `hourly_baseline` juga tidak ada.
# b. Fitur turunan lain: transformasi log1p (log_baseline_payload,
#    log_payload), severity_num, dan fitur interaksi durasi_x_severity
#    (durasi lama + severity tinggi biasanya lebih merusak).
#
# << REVISI: "log_rpmb" DIHAPUS (sumbernya, kolom rpmb, sudah tidak
# dipakai). "impacted_sites_count" DIKEMBALIKAN -- dihitung lagi dari
# kolom `url` lewat count_impacted_sites(), TAPI kolom ini SENGAJA TIDAK
# dimasukkan ke FEATURES_FINAL (lihat CELL 0), jadi hanya tersedia untuk
# EDA/audit dan TIDAK ikut dilatih ke model / tidak muncul di feature
# importance.
df_eda["impacted_sites_count"] = df_eda["url"].apply(count_impacted_sites)
#
# ---------------------------------------------------------------------
# BAGAIMANA MERGE-nya BEKERJA — DIJELASKAN DETAIL (bukan dirata-rata!)
# ---------------------------------------------------------------------
# `df_hourly` (FILE_HOURLY) berisi baseline payload per kombinasi
# site_id + hour + day_name (day_name isinya cuma dua nilai: "weekday"
# atau "weekend"), yaitu payload rata-rata historis pada jam & jenis
# hari tersebut, DI LUAR periode insiden -- ini jadi acuan "kondisi
# normal" untuk dibandingkan dengan payload saat insiden terjadi.
#
# PENTING: merge ini BUKAN mengambil satu angka rata-rata gabungan
# (weekday+weekend dicampur) untuk tiap site_id+hour. LEFT JOIN di
# bawah memakai 3 kolom sebagai kunci penggabungan sekaligus:
#   site_id  (site mana)
#   hour     (jam berapa)
#   day_name_type / day_name  (apakah baris itu "weekday" atau "weekend")
# Karena day_name_type (dari insiden) ikut jadi kunci join dan harus
# SAMA PERSIS dengan day_name (dari baseline), maka:
#   - baris insiden dengan day_name_type == "weekday" HANYA akan
#     dicocokkan ke baris df_hourly yang day_name-nya juga "weekday".
#   - baris insiden dengan day_name_type == "weekend" HANYA akan
#     dicocokkan ke baris df_hourly yang day_name-nya juga "weekend".
# Baris weekday tidak akan pernah tercampur/dirata-rata dengan baris
# weekend, dan sebaliknya -- masing-masing dites/dicocokkan sesuai
# jenis harinya sendiri-sendiri. Baris yang tidak ketemu pasangannya di
# df_hourly (site/jam/jenis hari tidak ada datanya) akan bernilai NaN
# lalu diisi 0 (fillna(0)) supaya tidak mengurangi jumlah baris data.
#
# -- Konversi satuan: GB -> MB --
# Kolom asal di df_hourly ("avg_payload_gb") disimpan dalam Gigabyte,
# sedangkan seluruh kolom payload lain di pipeline ini (baseline_payload,
# payload) sudah dalam Megabyte. Supaya satuannya konsisten dan bisa
# dibandingkan apple-to-apple oleh model, nilainya dikonversi eksplisit
# (1 GB = 1024 MB) sebelum dipakai, dan disimpan di kolom baru
# "avg_payload_mb" (kolom GB aslinya tidak diubah/tidak dihapus, hanya
# ditambah kolom hasil konversi).
df_hourly["avg_payload_mb"] = df_hourly["avg_payload_gb"] * 1024  # << REVISI: GB -> MB

# -- Gabung dengan baseline per jam (left join, jumlah baris tidak berubah) --
df_eda = df_eda.merge(
    df_hourly[["site_id", "hour", "day_name", "avg_payload_mb"]],
    left_on=["site_id", "hour", "day_name_type"],
    right_on=["site_id", "hour", "day_name"],
    how="left",
)  # << PERUBAHAN SETELAH MERGER: seluruh blok merge ini baru
df_eda.rename(columns={"avg_payload_mb": "hourly_baseline"}, inplace=True)
df_eda["hourly_baseline"] = pd.to_numeric(df_eda["hourly_baseline"]).fillna(0)
print(f"Baris valid setelah integrasi dengan baseline hourly (satuan MB): {len(df_eda)}")
print("Cek merge per jenis hari (weekday vs weekend dicocokkan terpisah, tidak dirata-rata):")
print(df_eda.groupby("day_name_type")["hourly_baseline"].agg(["count", "mean"]).round(2).to_string())

# -- Fitur turunan lain --
# ---------------------------------------------------------------------
# log_baseline_payload — PENJELASAN DETAIL (apa / kenapa / dari mana)
# ---------------------------------------------------------------------
# APA    : hasil transformasi log1p (yaitu log(x + 1)) dari kolom
#          `baseline_payload`. "+1" dipakai (bukan log biasa) supaya
#          nilai baseline_payload = 0 tetap aman dihitung (log(0) tidak
#          terdefinisi / -infinity, sedangkan log1p(0) = 0).
# DARI MANA : dihitung langsung dari kolom `baseline_payload` yang
#          sudah dibersihkan & dikonversi ke numerik di CELL 5A (bukan
#          dari df_hourly / hourly_baseline -- dua hal ini berbeda:
#          baseline_payload = payload normal site tsb per insiden,
#          hourly_baseline  = payload normal per jam+hari dari histori).
# KENAPA : distribusi baseline_payload pada data insiden jaringan
#          biasanya sangat "miring ke kanan" (right-skewed) -- banyak
#          site dengan baseline kecil, sedikit site dengan baseline
#          sangat besar. Transformasi log1p:
#          1. Memampatkan rentang nilai ekstrem supaya lebih mendekati
#             distribusi normal/simetris.
#          2. Menstabilkan variansi (nilai besar tidak lagi
#             "mendominasi" secara tidak proporsional).
#          3. Membuat hubungan dengan log(target) lebih linear -- lihat
#             tabel korelasi di CELL 7, log_baseline_payload biasanya
#             berkorelasi lebih kuat dengan log_target dibanding
#             baseline_payload skala asli.
#          4. Sama seperti target (loss_payload) yang juga dilatih di
#             skala log1p (lihat CELL 6), menyamakan skala fitur utama
#             dengan skala target membantu model tree-based belajar
#             pola yang lebih stabil.
df_eda["log_baseline_payload"] = np.log1p(df_eda["baseline_payload"])
df_eda["log_payload"] = np.log1p(df_eda["payload"])
df_eda["severity_num"] = df_eda["severity"].str.lower().map(SEV_MAP)
df_eda["durasi_x_severity"] = df_eda["durasi_menit"] * df_eda["severity_num"]
print("Fitur turunan dibuat: log_baseline_payload, "
      "log_payload, severity_num, durasi_x_severity, hourly_baseline (MB), "
      "impacted_sites_count (khusus EDA, tidak dipakai model)")
print(f"Total kolom pada df_eda setelah feature engineering: {df_eda.shape[1]}")


# %%
# ==============================================================================
# BAGIAN 2: EDA (EXPLORATORY DATA ANALYSIS) — BAGIAN 2b: SETELAH PREPROCESSING
# & FEATURE ENGINEERING
# ==============================================================================
# Eksplorasi lanjutan ini butuh kolom-kolom hasil CELL 5A/5B (misalnya
# log_payload, hourly_baseline), makanya posisinya di sini, bukan di awal.
# ------------------------------------------------------------------------
# CELL 6 (TAHAP 5): EDA — DISTRIBUSI TARGET
# ------------------------------------------------------------------------
# Membuat & menyimpan grafik distribusi loss_payload dalam dua versi:
# skala asli dan skala log1p. Data loss_payload biasanya sangat "miring"
# (skewed) — banyak insiden kecil, sedikit insiden ekstrem besar. Grafik
# ini membuktikan secara visual kenapa model nanti dilatih pada skala log,
# bukan skala asli.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribusi Target", fontsize=13, fontweight="bold")

axes[0].hist(df_eda[TARGET].dropna(), bins=60, color="#1565C0", edgecolor="white", alpha=0.85)
axes[0].set_title("loss_payload (skala asli)")
axes[0].set_xlabel("loss_payload")
axes[0].set_ylabel("Count")
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].hist(np.log1p(df_eda[TARGET].dropna()), bins=60, color="#E65100", edgecolor="white", alpha=0.85)
axes[1].set_title("log1p(loss_payload)")
axes[1].set_xlabel("log1p(loss_payload)")
axes[1].set_ylabel("Count")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
eda_plot_path = f"{OUTPUT_DIR}/eda_target_distribution.png"
plt.savefig(eda_plot_path, dpi=150, bbox_inches="tight")
plt.show()  # supaya grafik langsung tampil di panel Interactive VSCode

print("Statistik ringkas loss_payload (skala asli):")
print(df_eda[TARGET].describe().round(2).to_string())
print(f"Grafik distribusi target disimpan di: {eda_plot_path}")


# %%
# ------------------------------------------------------------------------
# CELL 7 (TAHAP 6): EDA — TABEL KORELASI
# ------------------------------------------------------------------------
# Menghitung korelasi (Pearson) antara fitur numerik utama dengan target,
# baik skala asli maupun skala log target. Membantu memahami fitur mana
# yang paling berhubungan linear dengan target, sekaligus jadi bukti
# tambahan bahwa fitur log lebih berkorelasi dengan log target.
num_cols = [
    "baseline_payload", "log_baseline_payload",
    "payload", "log_payload",
    "hourly_baseline", "availability_full",
    "update_impact", "durasi_menit",
]
log_target = np.log1p(df_eda[TARGET])

corr_raw = df_eda[num_cols + [TARGET]].corr()[TARGET].drop(TARGET)
corr_log = df_eda[num_cols].assign(log_target=log_target).corr()["log_target"].drop("log_target")

corr_table = pd.DataFrame({
    "vs target asli": corr_raw,
    "vs log target": corr_log,
}).sort_values("vs log target", ascending=False)

print("Tabel Korelasi Fitur Numerik:")
print(corr_table.round(3).to_string())


# %%
# ==============================================================================
# BAGIAN 3: PREPROCESSING (LANJUTAN) — ENCODING KATEGORIKAL
# ==============================================================================
# Masih bagian Preprocessing: mengubah kolom kategorikal jadi angka supaya
# siap dipakai model. Posisinya tetap di sini (setelah EDA), sama seperti
# kode aslinya, karena encoding tidak dibutuhkan untuk EDA di CELL 6-7.
# ------------------------------------------------------------------------
# CELL 8 (TAHAP 7): ENCODING KOLOM KATEGORIKAL
# ------------------------------------------------------------------------
# Meng-encode kolom kategorikal (site_id, regional, day_type,
# rootcausecategory) menjadi angka via LabelEncoder, karena model ML
# butuh input numerik. Nilai kosong diisi "unknown" dulu sebelum
# di-encode, supaya nanti saat inference ada kategori fallback untuk
# data baru yang belum pernah dilihat model. `encoders` disimpan supaya
# bisa dipakai ulang persis di tahap inference nanti.
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_eda[col] = df_eda[col].fillna("unknown").astype(str)
    df_eda[col] = le.fit_transform(df_eda[col])
    encoders[col] = le
    print(f"Kolom '{col}' di-encode -> {len(le.classes_)} kategori unik")


# %%
# ==============================================================================
# BAGIAN 5: MODEL TRAINING
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 9 (TAHAP 8): SPLIT DATA (TRAIN/TEST)
# ------------------------------------------------------------------------
# Menyiapkan data final untuk modeling:
# 1. Ambil hanya kolom FEATURES_FINAL + TARGET, buang baris yang punya
#    nilai kosong (dropna) -- model tidak bisa dilatih dengan NaN.
# 2. Buat target 2 skala: y_raw (asli) dan y_log (log1p dari y_raw).
# 3. Split 80% train / 20% test dengan RANDOM_STATE tetap supaya hasil
#    selalu sama tiap dijalankan ulang (reproducible).
df_model = df_eda[FEATURES_FINAL + [TARGET]].dropna()
X = df_model[FEATURES_FINAL]
y_raw = df_model[TARGET]
y_log = np.log1p(y_raw)

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Total data untuk modeling (setelah dropna) : {len(df_model)}")
print(f"Jumlah fitur yang dipakai                   : {len(FEATURES_FINAL)}")
print(f"Data training                                : {X_train.shape[0]} baris")
print(f"Data testing                                 : {X_test.shape[0]} baris")


# %%
# ------------------------------------------------------------------------
# CELL 10 (TAHAP 9): TRAINING RANDOM FOREST
# ------------------------------------------------------------------------
# Melatih SATU model Random Forest Regressor pada data training, dengan
# target skala log (y_train_log). Skala log dipakai karena loss_payload
# sangat skewed (lihat EDA) -- model tree-based lebih stabil belajar pada
# skala log dibanding skala asli yang rentangnya sangat lebar.
# Hyperparameter: 500 pohon (n_estimators), kedalaman tidak dibatasi
# (max_depth=None), memakai semua core CPU (n_jobs=-1).
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
print("Melatih model Random Forest... Mohon tunggu.")
rf_model.fit(X_train, y_train_log)
print("Pelatihan Random Forest selesai.")
print(f"Jumlah pohon (n_estimators) : {rf_model.n_estimators}")


# %%
# ------------------------------------------------------------------------
# CELL 11 (TAHAP 10): EXPORT TRAINING & TEST SET
# ------------------------------------------------------------------------
# Menyimpan data training & testing (fitur + target skala asli) sebagai
# CSV terpisah, untuk keperluan audit/dokumentasi di luar pipeline
# (misalnya dibuka manual di Excel).
df_train_export = X_train.copy()
df_train_export[TARGET] = y_train_raw.values
train_set_path = f"{OUTPUT_DIR}/training_set.csv"
df_train_export.to_csv(train_set_path, index=True)

df_test_export = X_test.copy()
df_test_export[TARGET] = y_test_raw.values
test_set_path = f"{OUTPUT_DIR}/test_set.csv"
df_test_export.to_csv(test_set_path, index=True)

print(f"Training set disimpan di : {train_set_path} ({len(df_train_export)} baris)")
print(f"Test set disimpan di     : {test_set_path} ({len(df_test_export)} baris)")


# %%
# ------------------------------------------------------------------------
# CELL 12 (TAHAP 11): TUNING XGBOOST
# ------------------------------------------------------------------------
# Mencari kombinasi hyperparameter terbaik untuk XGBRegressor memakai
# RandomizedSearchCV (mencoba 20 kombinasi acak dari ruang pencarian, tiap
# kombinasi dievaluasi dengan 5-fold cross-validation di dalam data
# training, dinilai dari skor R2). Setelah ketemu kombinasi terbaik,
# model final otomatis dilatih ulang pada seluruh X_train dengan
# parameter tersebut.
param_dist = {
    "n_estimators": [200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 5, 6, 7],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}
search = RandomizedSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    param_distributions=param_dist,
    n_iter=20,
    scoring="r2",
    cv=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
print("Tuning XGBoost (skala log)... Mencoba 20 kombinasi hyperparameter x 5-fold CV.")
search.fit(X_train, y_train_log)
xgb_model = search.best_estimator_

print("Tuning selesai.")
print("Best params  :", search.best_params_)
print(f"Best CV R2 (di dalam training set) : {search.best_score_:.4f}")


# %%
# ------------------------------------------------------------------------
# CELL 13 (TAHAP 12): CROSS-VALIDATION (SKALA LOG -> DIKEMBALIKAN KE ASLI)
# ------------------------------------------------------------------------
# Mengukur seberapa stabil performa tiap model di berbagai potongan data
# berbeda (bukan cuma satu kali split train/test). Untuk tiap fold:
# 1. Buat model baru dengan hyperparameter identik (fresh, belum pernah
#    dilatih).
# 2. Latih pada data training fold (skala log).
# 3. Prediksi pada data validasi fold (skala log), kembalikan ke skala
#    asli lewat expm1, clip minimal 0 (loss_payload tidak boleh negatif).
# 4. Hitung R2 di skala asli.
# Dilakukan 5 kali (5-fold) dengan KFold(shuffle=True) untuk XGBoost dan
# Random Forest, supaya perbandingan performa kedua model adil
# (apple-to-apple, sama-sama skala log).
def hitung_cv_r2_skala_log(model, X_all, y_raw_all, y_log_all, cv=5, random_state=RANDOM_STATE):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    X_arr = X_all.reset_index(drop=True)
    y_raw_arr = y_raw_all.reset_index(drop=True)
    y_log_arr = y_log_all.reset_index(drop=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr), start=1):
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_arr.iloc[train_idx], y_log_arr.iloc[train_idx])
        pred_log = model_clone.predict(X_arr.iloc[val_idx])
        pred_raw = np.maximum(0, np.expm1(pred_log))
        fold_r2 = r2_score(y_raw_arr.iloc[val_idx], pred_raw)
        scores.append(fold_r2)
        print(f"  Fold {fold_idx}/{cv} -> R2 = {fold_r2:.4f}")

    return np.array(scores)


print("--- Cross-Validation XGBoost ---")
xgb_cv = hitung_cv_r2_skala_log(xgb_model, X, y_raw, y_log)
print(f"XGBoost CV R2: {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")
print()

print("--- Cross-Validation Random Forest ---")
rf_cv = hitung_cv_r2_skala_log(rf_model, X, y_raw, y_log)
print(f"Random Forest CV R2: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")


# %%
# ==============================================================================
# BAGIAN 6: EVALUASI
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 14 (TAHAP 13): EVALUASI TEST SET & PEMILIHAN MODEL TERBAIK
# ------------------------------------------------------------------------
# Menghitung MAE, RMSE, R2 di skala asli untuk kedua model (prediksi
# di-expm1 dari skala log, lalu di-clip minimal 0), lalu memilih model
# dengan R2 tertinggi sebagai best_model. Jika Random Forest R2 >= XGBoost
# R2, Random Forest dipilih (aturan tie-break berpihak ke Random Forest).
y_pred_xgb_log = xgb_model.predict(X_test)
y_pred_xgb = np.maximum(0, np.expm1(y_pred_xgb_log))
metrics_xgb = {
    "MAE": mean_absolute_error(y_test_raw, y_pred_xgb),
    "RMSE": np.sqrt(mean_squared_error(y_test_raw, y_pred_xgb)),
    "R2": r2_score(y_test_raw, y_pred_xgb),
}
print(f"XGBoost      -> MAE={metrics_xgb['MAE']:,.2f}  RMSE={metrics_xgb['RMSE']:,.2f}  R2={metrics_xgb['R2']:.4f}")

y_pred_rf_log = rf_model.predict(X_test)
y_pred_rf = np.maximum(0, np.expm1(y_pred_rf_log))
metrics_rf = {
    "MAE": mean_absolute_error(y_test_raw, y_pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_test_raw, y_pred_rf)),
    "R2": r2_score(y_test_raw, y_pred_rf),
}
print(f"Random Forest-> MAE={metrics_rf['MAE']:,.2f}  RMSE={metrics_rf['RMSE']:,.2f}  R2={metrics_rf['R2']:.4f}")

results_df = pd.DataFrame({"XGBoost": metrics_xgb, "Random Forest": metrics_rf}).T
print("\nPerbandingan Metrik Evaluasi (keduanya skala log -> expm1):")
print(results_df.round(4).to_string())

if metrics_rf["R2"] >= metrics_xgb["R2"]:
    best_model, best_model_name = rf_model, "Random Forest"
    best_preds, best_metrics = y_pred_rf, metrics_rf
else:
    best_model, best_model_name = xgb_model, "XGBoost"
    best_preds, best_metrics = y_pred_xgb, metrics_xgb

best_is_log_model = True  # kedua kandidat dilatih di skala log
print(f"\n-> Model TERBAIK terpilih: {best_model_name} (R2={best_metrics['R2']:.4f})")


# %%
# ------------------------------------------------------------------------
# CELL 15 (TAHAP 14): FEATURE IMPORTANCE MODEL TERBAIK
# ------------------------------------------------------------------------
# Mengambil skor feature_importances_ dari model terbaik (baik Random
# Forest maupun XGBoost punya atribut ini), lalu mengurutkan dari yang
# paling berpengaruh. Berguna untuk memahami fitur mana yang paling
# banyak dipakai model saat membuat keputusan prediksi.
feat_imp = pd.DataFrame({
    "Feature": FEATURES_FINAL,
    "Importance": best_model.feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(f"Best model: {best_model_name}")
print(f"R2={best_metrics['R2']:.4f}  MAE={best_metrics['MAE']:,.2f}  RMSE={best_metrics['RMSE']:,.2f}")
print("\nTop-10 Feature Importance:")
print(feat_imp.head(10).to_string(index=False))


# %%
# ------------------------------------------------------------------------
# CELL 16 (TAHAP 15): ANALISIS ERROR PER KELOMPOK BESARAN NILAI
# ------------------------------------------------------------------------
# Satu angka MAE/R2 global bisa menyembunyikan fakta bahwa model sangat
# akurat untuk insiden kecil tapi buruk untuk insiden besar (atau
# sebaliknya). Di sini, error dipecah per kelompok besaran nilai aktual
# (< 10k, 10k-100k, 100k-1jt, > 1jt) supaya kelihatan di mana model kuat
# dan di mana model masih lemah.
eval_df = pd.DataFrame({"actual": y_test_raw.values, "predicted": best_preds})
eval_df["abs_error"] = (eval_df["actual"] - eval_df["predicted"]).abs()
eval_df["ape"] = eval_df["abs_error"] / (eval_df["actual"] + 1)

bins = [(0, 1e4, "< 10k"), (1e4, 1e5, "10k - 100k"), (1e5, 1e6, "100k - 1jt"), (1e6, np.inf, "> 1jt")]

print(f"{'Kelompok':<14}{'N':>7}{'MAE':>16}{'Median APE':>14}")
for lo, hi, label in bins:
    mask = (eval_df["actual"] >= lo) & (eval_df["actual"] < hi)
    if mask.sum() > 0:
        subset = eval_df[mask]
        mae_kelompok = subset["abs_error"].mean()
        ape_kelompok = subset["ape"].median() * 100
        print(f"{label:<14}{mask.sum():>7}{mae_kelompok:>16,.0f}{ape_kelompok:>13.1f}%")


# %%
# ------------------------------------------------------------------------
# CELL 17 (TAHAP 16): DASHBOARD EVALUASI (VISUAL)
# ------------------------------------------------------------------------
# Membuat satu dashboard visual (5 panel) merangkum evaluasi model
# terbaik: Feature Importance, Actual vs Predicted (skala asli & log),
# Residual Distribution, dan perbandingan Cross-Validation R2 antara
# XGBoost vs Random Forest. Satu gambar ringkas ini memudahkan siapa pun
# menilai kualitas model secara visual.
fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 3, figure=fig)
fig.suptitle(f"Model Evaluation - {best_model_name}", fontsize=13, fontweight="bold")

ax_fi = fig.add_subplot(gs[:, 0])
ax_fi.barh(feat_imp["Feature"], feat_imp["Importance"], color=sns.color_palette("Blues_r", len(feat_imp)))
ax_fi.invert_yaxis()
ax_fi.set_title("Feature Importance")
ax_fi.set_xlabel("Importance")
ax_fi.spines[["top", "right"]].set_visible(False)

ax_avp = fig.add_subplot(gs[0, 1])
ax_avp.scatter(y_test_raw, best_preds, alpha=0.3, s=15, color="#1565C0", edgecolors="none")
lo_val = min(y_test_raw.min(), best_preds.min())
hi_val = max(y_test_raw.max(), best_preds.max())
ax_avp.plot([lo_val, hi_val], [lo_val, hi_val], "r--", linewidth=1.5)
ax_avp.set_title("Actual vs Predicted")
ax_avp.set_xlabel("Actual")
ax_avp.set_ylabel("Predicted")
ax_avp.spines[["top", "right"]].set_visible(False)

ax_res = fig.add_subplot(gs[0, 2])
residuals = y_test_raw.values - best_preds
ax_res.hist(residuals, bins=60, color="#43A047", edgecolor="white", alpha=0.85)
ax_res.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax_res.set_title("Residual Distribution")
ax_res.set_xlabel("Actual - Predicted")
ax_res.set_ylabel("Count")
ax_res.spines[["top", "right"]].set_visible(False)

ax_log = fig.add_subplot(gs[1, 1])
ax_log.scatter(np.log1p(y_test_raw), np.log1p(np.clip(best_preds, 0, None)), alpha=0.3, s=15, color="#6A1B9A", edgecolors="none")
ax_log.plot([0, np.log1p(hi_val)], [0, np.log1p(hi_val)], "r--", linewidth=1.5)
ax_log.set_title("Actual vs Predicted (skala log)")
ax_log.set_xlabel("log1p(Actual)")
ax_log.set_ylabel("log1p(Predicted)")
ax_log.spines[["top", "right"]].set_visible(False)

ax_cv = fig.add_subplot(gs[1, 2])
ax_cv.bar(["XGBoost", "Random Forest"], [xgb_cv.mean(), rf_cv.mean()],
          yerr=[xgb_cv.std(), rf_cv.std()], color=["#1565C0", "#E65100"], capsize=5, width=0.5)
ax_cv.set_ylim(0, 1.05)
ax_cv.set_title("Cross-Validation R2 (5-fold, skala log -> expm1)")
ax_cv.set_ylabel("R2")
ax_cv.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
dashboard_path = f"{OUTPUT_DIR}/evaluation_report.png"
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Residual rata-rata (actual - predicted) : {residuals.mean():,.2f}")
print(f"Dashboard evaluasi disimpan di: {dashboard_path}")


# %%
# ------------------------------------------------------------------------
# CELL 18 (TAHAP 17): EXPORT HASIL PREDIKSI TEST SET
# ------------------------------------------------------------------------
# Menyimpan seluruh hasil prediksi model terbaik pada test set ke CSV
# (lengkap dengan kolom error), lalu menampilkan 10 baris dengan error
# absolut terbesar -- berguna untuk investigasi kasus yang paling meleset.
df_predictions = X_test.copy()
df_predictions["actual_loss_payload"] = y_test_raw.values
df_predictions["predicted_loss_payload"] = best_preds
df_predictions["absolute_error"] = np.abs(y_test_raw.values - best_preds)
df_predictions["error_pct"] = (
    df_predictions["absolute_error"] /
    (df_predictions["actual_loss_payload"].abs() + 1e-9) * 100
)
df_predictions = df_predictions.sort_values("absolute_error", ascending=False)
predictions_path = f"{OUTPUT_DIR}/predictions_test_set.csv"
df_predictions.to_csv(predictions_path, index=True)

print(f"Seluruh hasil prediksi test set disimpan di: {predictions_path}")
print("10 prediksi dengan error terbesar:")
print(df_predictions[["actual_loss_payload", "predicted_loss_payload", "absolute_error", "error_pct"]]
      .head(10).round(2).to_string())


# %%
# ==============================================================================
# BAGIAN TAMBAHAN (DI LUAR 7 BAGIAN UTAMA): CONTOH INFERENCE / DEPLOYMENT
# ==============================================================================
# Mencoba model final pada 1 insiden baru, sebagai jembatan sebelum masuk
# ke Reporting.
# ------------------------------------------------------------------------
# CELL 19 (TAHAP 18): CONTOH INFERENCE PADA INSIDEN BARU
# ------------------------------------------------------------------------
# Memprediksi loss_payload untuk SATU insiden baru yang datanya masih
# mentah (raw_input), dengan MEREPLIKASI PERSIS seluruh feature
# engineering yang dipakai saat training, supaya fitur yang dikirim ke
# model konsisten dengan yang dipelajari model:
# 1. Basic preprocessing (severity->angka, durasi->menit, kolom numerik
#    string->float, waktu dipecah jadi hour/month/is_peak_hour).
# 2. Engineered features (log1p, durasi_x_severity) -- rumus sama persis
#    dengan Cell 5.
# 3. Hourly baseline lookup dari df_hourly, DICOCOKKAN PERSIS ke jenis
#    hari (weekday/weekend) insiden ini -- bukan diambil dari rata-rata
#    gabungan semua hari -- lalu dikonversi ke MB (default 0 jika tidak
#    ditemukan).
# 4. Encoding kategorikal memakai `encoders` yang SAMA dari Cell 8,
#    fallback ke "unknown" untuk kategori baru.
# 5. Prediksi, dikembalikan ke skala asli via expm1 (karena model dilatih
#    di skala log).
#
# << REVISI: "rpmb" dihapus dari raw_input (dan turunan log_rpmb) karena
# sudah tidak dipakai di pipeline. "url" DIKEMBALIKAN -- dipakai untuk
# menghitung impacted_sites_count via count_impacted_sites(), tapi
# nilainya HANYA dicetak untuk info, TIDAK dimasukkan ke `row` / X_new,
# karena impacted_sites_count sengaja tidak ada di FEATURES_FINAL
# (bukan fitur model) -- lihat "RINGKASAN REVISI TERBARU" di header file.
raw_input = {
    "severity": "major",
    "duarasi_alaram": "02:15:30",
    "baseline_payload": "1200.5",
    "payload": "300.2",
    "availability_full": "99.2",
    "update_impact": 1,
    "alarm_start_time": "2026-05-10 14:30:00",
    "day_type": "Weekday",
    "url": "site-A;site-B",
    "site_id": "SITE001",
    "regional": "Jawa Barat",
    "rootcausecategory": "hardware",
}

row = {}
row["severity_num"] = SEV_MAP.get(str(raw_input["severity"]).lower(), np.nan)
row["durasi_menit"] = duration_to_minutes(raw_input["duarasi_alaram"])
row["baseline_payload"] = float(str(raw_input["baseline_payload"]).replace(",", "."))
row["payload"] = float(str(raw_input["payload"]).replace(",", "."))
row["availability_full"] = float(str(raw_input["availability_full"]).replace(",", "."))
row["update_impact"] = float(raw_input.get("update_impact", 1))

# Hanya untuk info/EDA -- TIDAK dimasukkan ke row/X_new (bukan fitur model)
impacted_sites_count_info = count_impacted_sites(raw_input.get("url", ""))

alarm_start = pd.to_datetime(raw_input["alarm_start_time"], errors="coerce")
row["hour"] = alarm_start.hour
row["month"] = alarm_start.month
row["is_peak_hour"] = int(8 <= alarm_start.hour <= 22)
row["day_type_str"] = raw_input.get("day_type", "Weekday")

row["log_baseline_payload"] = np.log1p(row["baseline_payload"])
row["log_payload"] = np.log1p(row["payload"])
row["durasi_x_severity"] = row["durasi_menit"] * row["severity_num"]

# << PERUBAHAN SETELAH MERGER: blok lookup ke df_hourly ini meniru proses
# join yang sama seperti di CELL 5B supaya konsisten dengan training --
# dicocokkan persis ke day_name yang sama (weekday/weekend), lalu
# dikonversi ke MB (1 GB = 1024 MB), BUKAN diambil dari rata-rata semua hari.
hb_match = df_hourly[
    (df_hourly["site_id"] == raw_input["site_id"]) &
    (df_hourly["hour"] == row["hour"]) &
    (df_hourly["day_name"].str.lower() == row["day_type_str"].lower())
]["avg_payload_gb"]
row["hourly_baseline"] = (hb_match.values[0] * 1024) if not hb_match.empty else 0  # << REVISI: GB -> MB

encode_map = {
    "site_id": raw_input["site_id"],
    "regional": raw_input["regional"],
    "day_type": row["day_type_str"],
    "rootcausecategory": raw_input.get("rootcausecategory", "unknown"),
}
for col, value in encode_map.items():
    encoder = encoders[col]
    val_str = str(value)
    if val_str in encoder.classes_:
        row[col] = encoder.transform([val_str])[0]
    else:
        row[col] = encoder.transform(["unknown"])[0] if "unknown" in encoder.classes_ else 0
        print(f"  Peringatan: kategori '{val_str}' pada kolom '{col}' belum pernah "
              f"dilihat saat training -> fallback ke 'unknown'.")

X_new = pd.DataFrame([{f: row.get(f, np.nan) for f in FEATURES_FINAL}])
prediction_raw = best_model.predict(X_new)[0]
final_pred = np.expm1(prediction_raw) if best_is_log_model else prediction_raw
final_pred = round(float(final_pred), 2)

print("Data mentah input      :", raw_input)
print(f"impacted_sites_count   : {impacted_sites_count_info} (info EDA saja, bukan input model)")
print(f"Model dipakai          : {best_model_name}")
print(f"Predicted loss_payload : {final_pred:,.2f}")


# %%
# ==============================================================================
# BAGIAN 7: REPORTING
# ==============================================================================
# ------------------------------------------------------------------------
# CELL 20 (TAHAP 19): SIMPAN MODEL & ARTIFACTS
# ------------------------------------------------------------------------
# Menyimpan SEMUA hal yang dibutuhkan untuk memakai model ini di masa
# depan (deployment/inference) dalam satu file .joblib: model itu sendiri,
# seluruh encoder kategorikal, daftar fitur, mapping severity, dan flag
# skala target model -- supaya saat model dipakai ulang di tempat lain,
# tidak perlu mengingat-ingat encoder mana yang cocok dengan model mana.
model_artifacts = {
    "model": best_model,
    "encoders": encoders,
    "features": FEATURES_FINAL,
    "sev_map": SEV_MAP,
    "is_log_model": best_is_log_model,
}
model_path = f"{OUTPUT_DIR}/best_incident_model.joblib"
joblib.dump(model_artifacts, model_path)

print(f"Model dan artifacts berhasil disimpan di: {model_path}")
print(f"Ukuran file: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")


# %%
# ------------------------------------------------------------------------
# CELL 21 (TAHAP 20): TULIS LAPORAN AKHIR
# ------------------------------------------------------------------------
# Merangkum seluruh hasil pipeline (jumlah data, hasil cross-validation,
# tabel evaluasi, top-10 feature importance, metrik model terbaik) jadi
# satu laporan teks rapi, disimpan sebagai .txt sekaligus dicetak ke
# terminal, supaya siapa pun bisa memahami hasil akhir tanpa membaca ulang
# seluruh log eksekusi yang panjang.
# << PERUBAHAN SETELAH MERGER: baris "File Pendukung" di bawah ini baru
# ditambahkan supaya laporan menyebutkan KEDUA sumber data (dulu cuma ada
# 1 baris "File" karena hanya ada 1 sumber data).
report = f"""Network Incident Loss Payload Predictor

DATA
File Utama      : {FILE_MAIN}
File Pendukung  : {FILE_HOURLY}
Baris total : {len(df_raw)}
Baris valid : {len(df_model)} (setelah buang baris tanpa target dan fitur tidak lengkap)
Jumlah fitur : {len(FEATURES_FINAL)}

TARGET
Transformasi : log1p saat training, expm1 saat prediksi
Outlier : tidak dipotong (nilai ekstrem adalah insiden valid)

MODEL
Train/Test : 80/20
Kedua model (XGBoost & Random Forest) dilatih dan di-CV pada skala log,
lalu prediksi di-expm1 sebelum dievaluasi di skala asli -- perbandingan apple-to-apple.
XGBoost CV R2 : {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})
RF CV R2 : {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})
Best model : {best_model_name}

EVALUASI (skala asli)
{results_df.round(4).to_string()}

TOP-10 FEATURE IMPORTANCE
{feat_imp.head(10).to_string(index=False)}

BEST METRICS
R2 : {best_metrics['R2']:.4f}
MAE : {best_metrics['MAE']:,.2f}
RMSE : {best_metrics['RMSE']:,.2f}
"""
report_path = f"{OUTPUT_DIR}/pipeline_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

metrics_summary_path = f"{OUTPUT_DIR}/metrics_summary.csv"
results_df.round(4).to_csv(metrics_summary_path)

print(f"Laporan disimpan di: {report_path}")
print(f"Ringkasan metrik disimpan di: {metrics_summary_path}")
print(report)
print("PIPELINE SELESAI")
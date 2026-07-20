# Network Incident Loss Payload Predictor

Model machine learning untuk memprediksi `loss_payload` (volume traffic yang hilang saat insiden jaringan) dari data historis insiden telekomunikasi, digabung dengan data baseline traffic per jam per site.

Penggunaan model adalah post-incident analysis: seluruh data insiden diasumsikan sudah lengkap saat prediksi dijalankan, termasuk `baseline_payload`. Model ditujukan untuk pelaporan dan audit loss otomatis, bukan prediksi real-time.

## Hasil Model

Model dilatih pada baris valid hasil gabungan data insiden utama dan data baseline hourly (setelah membuang baris tanpa target dan fitur tidak lengkap). Dua model dibandingkan (Random Forest dan XGBoost, keduanya dilatih dan di-cross-validation pada skala log lalu dievaluasi di skala asli), dan model terbaik dipilih otomatis berdasarkan skor R2.

| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Random Forest | *(isi dari output_ml_new/pipeline_report.txt)* | | |
| XGBoost (tuned) | *(isi dari output_ml_new/pipeline_report.txt)* | | |

Target ditransformasi dengan log1p sebelum training untuk menangani distribusi yang sangat skewed, lalu dikembalikan ke skala asli dengan expm1 saat prediksi (dikontrol lewat flag `is_log_model` yang tersimpan di dalam artifact model).

## Struktur Folder

```
Model_Prediksi_New/
  inap_ticketing_incident_loss_payload_2026.xlsx   Dataset utama (insiden)
  baseline_payload_hourly_weekly.csv               Dataset baseline traffic per jam/hari/site
  loss_payload_predictor_new_fixed.ipynb           Notebook utama (versi cell, 2 sumber data)
  requirements.txt                                 Daftar library
  README.md                                        Dokumen ini
  .venv/                                            Virtual environment Python
  .dockerignore                                     Daftar file/folder yang tidak ikut masuk image
  Dockerfile                                        Konfigurasi Docker
  docker-compose.yml                                Opsional, mempersingkat build+run
  app/
    main.py                                         API server (FastAPI) + serve UI
    static/
      index.html                                    Halaman web (form input + hasil prediksi)
  output_ml_new/                                    Output (dibuat otomatis saat notebook dijalankan)
    best_incident_model.joblib                       Satu file berisi model + encoders + features + sev_map + is_log_model
    metrics_summary.csv                              Ringkasan metrik
    pipeline_report.txt                               Laporan pipeline
    eda_target_distribution.png                       Grafik distribusi target
```

## Persyaratan Sistem

- Python 3.11
- VS Code dengan extension Python dan Jupyter dari Microsoft
- Docker Desktop (untuk menjalankan API lewat container)

## Setup Pertama Kali

1. Buka VS Code dan masuk ke folder project.

2. Buka terminal.

3. Aktifkan virtual environment:

```
.venv\Scripts\activate
```

Jika berhasil akan muncul `(.venv)` di awal baris.

4. Install library:

```
pip install -r requirements.txt
```

## Menjalankan Notebook

1. Pastikan virtual environment aktif.
2. Buka `loss_payload_predictor_new_fixed.ipynb`.
3. Pilih kernel Python dari folder `.venv` lewat tombol **Select Kernel** di pojok kanan atas.
4. Jalankan semua sel dari atas ke bawah (bertanda `# %%`), berurutan — sel di bawah butuh variabel dari sel di atasnya, sama seperti notebook biasa.
5. Output tersimpan otomatis di folder `output_ml_new`, termasuk `best_incident_model.joblib` yang dipakai API.

Catatan: sel hyperparameter tuning (RandomizedSearchCV) adalah bagian paling lama, bisa beberapa menit tergantung spesifikasi laptop.
## Prediksi Data Baru (dari Notebook)

Setelah notebook dijalankan penuh, buka cell contoh inference (CELL 19). Isi variabel `raw_input` dengan data insiden baru lalu jalankan selnya. Field yang dibutuhkan: `site_id`, `severity`, `alarm_start_time`, `duarasi_alaram`, `payload`, `baseline_payload`, `rpmb`, `availability_full`, `regional`, `day_type`, `rootcausecategory`, `update_impact`, `url`.

## Catatan Teknis Penting

- Angka desimal pada dataset memakai koma sebagai pemisah (mis. `"9750,617441"`). Fungsi `to_numeric_safe` (notebook) dan konversi `.replace(",", ".")` (API) menanganinya otomatis.
- Format tanggal `alarm_start_time` yang didukung API ada dua: format asli sistem monitoring `DD/MM/YYYY HH.MM.SS` (mis. `29/04/2026 23.53.30`) sebagai prioritas utama, dengan fallback ke format ISO `YYYY-MM-DD HH:MM:SS`.
- Target `loss_payload` ditraining dengan transformasi log1p untuk menangani distribusi yang sangat skewed. Hasil prediksi otomatis dikembalikan ke skala asli dengan expm1, dikontrol oleh flag `is_log_model` di dalam artifact model — bukan diasumsikan tetap seperti versi sebelumnya.
- Outlier pada target tidak dipotong. Nilai loss yang sangat besar adalah insiden nyata, bukan error pencatatan.
- `loss_payload` dan `baseline_payload` selalu kosong bersamaan pada sebagian baris. Baris tanpa target dibuang karena tidak bisa dipakai untuk training.
- Data baseline hourly (`baseline_payload_hourly_weekly.csv`) digabung ke data insiden lewat left join berdasarkan `site_id` + `hour` + nama hari (`day_name`), menghasilkan fitur `hourly_baseline`. Baris insiden yang site/jam/harinya tidak ada di data hourly tetap dipakai, dengan `hourly_baseline` diisi 0.

## Kolom yang Tidak Dipakai Langsung dan Alasannya

| Kolom | Alasan |
|---|---|
| ticket_id | Identitas unik, tidak ada nilai prediktif |
| alarm_start_time (mentah) | Sudah diturunkan menjadi `hour`, `month`, `is_peak_hour` |
| alarm_clear_time | Redundan dengan durasi |
| duarasi_alaram (mentah) | Sudah dikonversi menjadi `durasi_menit` |
| rootcausedetail | Sudah diwakili `rootcausecategory` |
| url (mentah) | Sudah diekstrak menjadi `impacted_sites_count` |

## Fitur yang Digunakan Model (19 fitur)

| Fitur | Sumber | Keterangan |
|---|---|---|
| severity_num | Dataset | Ordinal: Low=1, Minor=2, Major=3, Critical=4 |
| durasi_menit | Diturunkan | Konversi dari HH:MM:SS |
| baseline_payload | Dataset | Traffic normal site, prediktor terkuat |
| payload | Dataset | Traffic aktual saat insiden |
| rpmb | Dataset | Request Per Minute Baseline |
| availability_full | Dataset | Persentase availability |
| update_impact | Dataset | Numerik langsung |
| impacted_sites_count | Diturunkan | Jumlah site dari kolom `url` |
| hour | Diturunkan | Jam mulai alarm |
| month | Diturunkan | Bulan mulai alarm |
| is_peak_hour | Dibuat | 1 jika jam 08.00–22.00 |
| regional | Dataset | Label encoded |
| day_type | Dataset | Weekday/Weekend, label encoded |
| rootcausecategory | Dataset | Label encoded |
| log_baseline_payload | Dibuat | log1p(baseline_payload) |
| log_payload | Dibuat | log1p(payload) |
| log_rpmb | Dibuat | log1p(rpmb) |
| durasi_x_severity | Dibuat | Interaksi durasi dan keparahan |
| **hourly_baseline** | **Digabung dari file kedua** | **Rata-rata payload historis di jam & hari yang sama (hasil join dengan `baseline_payload_hourly_weekly.csv`)** |

## Metrik Evaluasi

| Metrik | Keterangan |
|---|---|
| R2 | Proporsi variasi target yang dijelaskan model, 1.0 berarti sempurna |
| MAE | Rata-rata selisih absolut prediksi terhadap aktual |
| RMSE | Akar rata-rata kuadrat error, lebih sensitif terhadap error besar |

## Deployment API

Model dijalankan sebagai API menggunakan FastAPI, lengkap dengan halaman web sederhana untuk input manual (tidak cuma lewat `/docs`). Pastikan file `output_ml_new/best_incident_model.joblib` dan `baseline_payload_hourly_weekly.csv` sudah tersedia di lokasi yang sesuai sebelum menjalankan API (lihat struktur folder di atas).

### Menjalankan API Tanpa Docker

Aktifkan virtual environment lalu jalankan dari folder root project:

```
uvicorn app.main:app --reload
```

- Halaman UI: http://127.0.0.1:8000
- Dokumentasi interaktif (Swagger): http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### Menjalankan API dengan Docker

1. Pastikan Docker Desktop sudah terbuka.

2. Build image:

```
docker build -t loss-payload-api .
```

3. Jalankan container:

```
docker run -p 8000:8000 loss-payload-api
```

API dan UI berjalan di http://localhost:8000, sama seperti tanpa Docker.

Atau, pakai docker compose supaya build+run jadi satu perintah:

```
docker compose up --build
```

Tekan `Ctrl+C` untuk menghentikan.

### Memindahkan ke Server Perusahaan

1. Export image menjadi satu file:

```
docker save -o loss-payload-api.tar loss-payload-api
```

2. Kirim file `loss-payload-api.tar` ke server perusahaan.

3. Di server perusahaan, load dan jalankan:

```
docker load -i loss-payload-api.tar
docker run -p 8000:8000 loss-payload-api
```

API langsung berjalan tanpa perlu install apapun di server.

### Menggunakan Halaman UI

Buka http://localhost:8000, ada dua cara mengisi data insiden:

1. **Isi manual** — isi tiap field satu per satu, pilih severity lewat tombol berwarna, pilih day type lewat dropdown.
2. **Quick fill dari JSON** — tempel JSON data insiden (format sama seperti contoh di bawah, termasuk trailing comma dari copy-paste kode Python) ke kotak di bagian atas, klik **Isi ke form**, semua field otomatis terisi dan siap direview sebelum submit.

Klik **Run prediction** untuk menjalankan prediksi. Hasilnya tampil di panel kanan dengan warna sesuai severity yang dipilih.

### Contoh Request Prediksi (via `/predict` atau Swagger)

Kirim POST request ke `/predict` dengan body JSON berikut:

```json
{
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
  "url": "SBS087"
}
```

Response:

```json
{
  "predicted_loss_payload": 1162.7
}
```

### Environment Variable (Opsional)

Path artifact model dan data hourly bisa dioverride tanpa ubah kode, lewat environment variable saat `docker run` atau `uvicorn`:

| Variable | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `output_ml_new/best_incident_model.joblib` | Lokasi file artifact model |
| `HOURLY_DATA_PATH` | `baseline_payload_hourly_weekly.csv` | Lokasi file data baseline hourly |

Contoh:

```
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/output_ml_new/best_incident_model.joblib \
  -e HOURLY_DATA_PATH=/app/baseline_payload_hourly_weekly.csv \
  loss-payload-api
```
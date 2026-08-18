# Streamlit Analisis Streamline & SST Indonesia

## Struktur repository

```text
streamlit-streamline-sst/
├── app.py
├── requirements.txt
└── README.md
```

## Jalankan di komputer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload `app.py` dan `requirements.txt`.
3. Buka Streamlit Community Cloud.
4. Pilih repository tersebut.
5. Pilih file utama `app.py`.
6. Deploy.

Aplikasi membutuhkan koneksi internet karena data GFS dan NOAA OISST
diambil langsung melalui endpoint THREDDS/OPeNDAP.

## Catatan

Versi ini mempertahankan logika utama notebook:
- GFS 0,25° → angin 850 mb + geopotential height → rata-rata dasarian.
- NOAA OISST → SST dan anomali SST → rata-rata dasarian.
- Streamline memakai parameter density yang dapat diubah dari sidebar.
- Peta dapat diunduh dalam format PNG 300 dpi.

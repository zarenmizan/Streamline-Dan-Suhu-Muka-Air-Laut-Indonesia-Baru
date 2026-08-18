import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.ndimage as ndimage
from datetime import date, timedelta
import io

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Streamline & SST Indonesia",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    color: #666;
    margin-top: 0;
}
.stDownloadButton button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌏 Analisis Streamline 850 mb & SST Indonesia</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Analisis dasarian berbasis GFS 0,25° dan NOAA OISST High Resolution</p>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR - PENGATURAN TANGGAL FLEKSIBEL
# ============================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

# Menentukan default rentang 7 hari terakhir
today = date.today()
default_start = today - timedelta(days=7)
default_end = today

rentang_tanggal = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(default_start, default_end),
    max_value=today + timedelta(days=10),
    help="Pilih tanggal awal dan tanggal akhir analisis."
)

# Validasi input tanggal (memastikan user memilih 2 tanggal: awal & akhir)
if isinstance(rentang_tanggal, (tuple, list)) and len(rentang_tanggal) == 2:
    tanggal_mulai, tanggal_akhir = rentang_tanggal
elif isinstance(rentang_tanggal, (tuple, list)) and len(rentang_tanggal) == 1:
    tanggal_mulai = rentang_tanggal[0]
    tanggal_akhir = rentang_tanggal[0]
else:
    tanggal_mulai = rentang_tanggal
    tanggal_akhir = rentang_tanggal

tahun = tanggal_mulai.year

st.sidebar.divider()
st.sidebar.subheader("🌀 Streamline")

density = st.sidebar.slider(
    "Kerapatan streamline", 1.0, 10.0, 7.0, 0.5,
    help="Semakin besar nilai, semakin rapat garis streamline."
)

skip_speed = st.sidebar.slider(
    "Jarak label kecepatan", 5, 40, 20, 1,
    help="Semakin besar nilai, semakin jarang angka kecepatan ditampilkan."
)

low_filter = st.sidebar.slider(
    "Ukuran filter pusat L", 20, 100, 60, 5,
    help="Ukuran filter yang lebih besar akan menyaring pusat tekanan rendah yang kecil."
)

st.sidebar.divider()
st.sidebar.subheader("🗺️ Area Peta")

lon_min, lon_max = st.sidebar.slider(
    "Bujur", 80, 160, (90, 145), 1
)
lat_min, lat_max = st.sidebar.slider(
    "Lintang", -30, 30, (-15, 15), 1
)

st.sidebar.divider()
st.sidebar.caption(
    "Sumber data: NCEP GFS 0,25° (UCAR THREDDS) dan NOAA OISST High Resolution."
)

# ============================================================
# URL DATA
# ============================================================
GFS_URL = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"

def noaa_url(prefix, year):
    return f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/{prefix}.{year}.nc"

# ============================================================
# CACHE DATA - agar tidak download ulang setiap interaksi
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_gfs(start_str, end_str, lon0, lon1, lat0, lat1):
    ds = xr.open_dataset(GFS_URL)

    var_u = ds["u-component_of_wind_isobaric"]
    var_v = ds["v-component_of_wind_isobaric"]
    var_h = ds["Geopotential_height_isobaric"]

    vert_dim_u = [d for d in var_u.dims if "isobaric" in d][0]
    vert_dim_v = [d for d in var_v.dims if "isobaric" in d][0]
    vert_dim_h = [d for d in var_h.dims if "isobaric" in d][0]

    level_u = 85000 if float(ds[vert_dim_u].max()) > 2000 else 850
    level_v = 85000 if float(ds[vert_dim_v].max()) > 2000 else 850
    level_h = 85000 if float(ds[vert_dim_h].max()) > 2000 else 850

    lat_values = ds["lat"].values
    lat_slice = slice(lat0, lat1) if lat_values[0] < lat_values[-1] else slice(lat1, lat0)

    # Menambahkan penanganan jam agar slicing waktu Xarray presisi
    t_start = f"{start_str}T00:00:00"
    t_end = f"{end_str}T23:59:59"

    time_selected = ds["time"].sel(time=slice(t_start, t_end))
    n_time = int(time_selected.sizes.get("time", 0))

    if n_time == 0:
        ds.close()
        return None, None, None, 0

    common = dict(
        lon=slice(lon0, lon1),
        lat=lat_slice,
        time=slice(t_start, t_end),
    )

    u = var_u.sel(**common).sel({vert_dim_u: level_u}, method="nearest").mean(dim="time").load()
    v = var_v.sel(**common).sel({vert_dim_v: level_v}, method="nearest").mean(dim="time").load()
    hgt = var_h.sel(**common).sel({vert_dim_h: level_h}, method="nearest").mean(dim="time").load()

    ds.close()
    return u, v, hgt, n_time


@st.cache_data(show_spinner=False, ttl=3600)
def load_sst(year, start_str, end_str, lon0, lon1, lat0, lat1):
    url_sst = noaa_url("sst.day.mean", year)
    url_anom = noaa_url("sst.day.anom", year)

    ds_sst = xr.open_dataset(url_sst)
    ds_anom = xr.open_dataset(url_anom)

    selected_time = ds_sst["time"].sel(time=slice(start_str, end_str))
    n_time = int(selected_time.sizes.get("time", 0))

    if n_time == 0:
        ds_sst.close()
        ds_anom.close()
        return None, None, 0

    sst = (
        ds_sst["sst"]
        .sel(
            time=slice(start_str, end_str),
            lon=slice(lon0, lon1),
            lat=slice(lat0, lat1),
        )
        .mean(dim="time")
        .load()
    )

    anom = (
        ds_anom["anom"]
        .sel(
            time=slice(start_str, end_str),
            lon=slice(lon0, lon1),
            lat=slice(lat0, lat1),
        )
        .mean(dim="time")
        .load()
    )

    ds_sst.close()
    ds_anom.close()
    return sst, anom, n_time


# ============================================================
# FUNGSI PETA STREAMLINE
# ============================================================
def make_streamline(u, v, hgt, start_date, end_date,
                    density_value, skip_value, filter_size,
                    extent):
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND, facecolor="#FFFF66",
        edgecolor="black", linewidth=0.5, zorder=1
    )
    ax.add_feature(cfeature.OCEAN, facecolor="#E6F2FF", zorder=0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)

    lons, lats = np.meshgrid(u.lon.values, u.lat.values)

    ax.streamplot(
        lons, lats,
        u.values, v.values,
        color="blue",
        linewidth=1.0,
        density=density_value,
        arrowsize=1.2,
        zorder=2,
        transform=ccrs.PlateCarree(),
    )

    speed_kt = np.sqrt(u.values**2 + v.values**2) * 1.94384

    for i in range(0, len(u.lat), skip_value):
        for j in range(0, len(u.lon), skip_value):
            spd = speed_kt[i, j]
            if np.isfinite(spd):
                ax.text(
                    lons[i, j], lats[i, j],
                    f"{int(round(spd))} kt",
                    color="red",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=3,
                    transform=ccrs.PlateCarree(),
                )

    data_hgt = hgt.values
    safe_filter = min(filter_size, max(3, min(data_hgt.shape) - 1))
    local_min = ndimage.minimum_filter(data_hgt, size=safe_filter) == data_hgt
    min_indices = np.where(local_min)

    for y, x in zip(min_indices[0], min_indices[1]):
        if 10 < x < len(hgt.lon) - 10 and 10 < y < len(hgt.lat) - 10:
            if np.isfinite(data_hgt[y, x]):
                lon_L = float(hgt.lon.values[x])
                lat_L = float(hgt.lat.values[y])

                ax.text(
                    lon_L, lat_L, "L",
                    color="red",
                    fontsize=18,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                    bbox=dict(
                        boxstyle="circle,pad=0.2",
                        edgecolor="red",
                        facecolor="white",
                        alpha=0.9,
                        lw=1.5,
                    ),
                    transform=ccrs.PlateCarree(),
                )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=0.5,
        color="gray"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        "Analisis Streamline 850 mb & Tekanan Rendah\n"
        f"Rata-rata Dasarian ({start_date} s/d {end_date})",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    fig.text(
        0.5, 0.01,
        "Data: NCEP GFS 0,25° | Stasiun Klimatologi Jawa Tengah",
        ha="center", va="center",
        fontsize=11, fontweight="bold", style="italic"
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ============================================================
# FUNGSI PETA SST
# ============================================================
def make_sst_map(data, start_date, end_date, extent, anomaly=False):
    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND, facecolor="white",
        edgecolor="black", zorder=2
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=2)

    lons, lats = np.meshgrid(data.lon.values, data.lat.values)

    if anomaly:
        levels = np.arange(-2.5, 2.75, 0.25)
        plot = ax.contourf(
            lons, lats, data.values,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        label = "Anomali Suhu (°C)"
        title = "Anomali Suhu Muka Laut (SSTA) Wilayah Indonesia"
    else:
        warna_jma = [
            "#8A2BE2", "#0000FF", "#1E90FF", "#00FFFF",
            "#00FF00", "#ADFF2F", "#FFD700", "#FFA500",
            "#FF4500", "#FF0000", "#FF00FF"
        ]
        batas_sst = [
            25.0, 25.5, 26.0, 26.5, 27.0, 27.5,
            28.0, 28.5, 29.0, 29.5, 30.0, 31.0
        ]
        cmap = mcolors.ListedColormap(warna_jma)
        norm = mcolors.BoundaryNorm(batas_sst, cmap.N)

        plot = ax.contourf(
            lons, lats, data.values,
            levels=batas_sst,
            cmap=cmap,
            norm=norm,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        label = "Suhu Muka Laut (°C)"
        title = "Suhu Muka Laut (SST) Wilayah Indonesia"

    cbar = plt.colorbar(
        plot, ax=ax,
        orientation="horizontal",
        pad=0.08, shrink=0.8
    )
    cbar.set_label(label, fontsize=11, fontweight="bold")

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=0.5,
        color="gray"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        f"{title}\nRata-rata Dasarian ({start_date} s/d {end_date})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    fig.text(
        0.5, 0.01,
        "Data: NOAA OISST High Resolution | Stasiun Klimatologi Jawa Tengah",
        ha="center", va="center",
        fontsize=10, fontweight="bold", style="italic"
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def fig_to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=300,
        bbox_inches="tight", pad_inches=0.3
    )
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# RINGKASAN PARAMETER
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Periode", f"{tanggal_mulai} – {tanggal_akhir}")
c2.metric("Level Angin", "850 mb")
c3.metric("Resolusi GFS", "0,25°")
c4.metric("Produk SST", "NOAA OISST")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🌀 Streamline 850 mb",
    "🌊 SST & SSTA",
    "ℹ️ Informasi"
])

# ============================================================
# TAB 1 - STREAMLINE
# ============================================================
with tab1:
    st.subheader("🌀 Analisis Streamline 850 mb")

    if st.button("🚀 Buat Peta Streamline", type="primary", use_container_width=True):
        start_str = tanggal_mulai.isoformat()
        end_str = tanggal_akhir.isoformat()

        try:
            with st.spinner("Mengakses GFS UCAR dan menghitung rata-rata dasarian..."):
                u, v, hgt, n_time_gfs = load_gfs(
                    start_str, end_str,
                    lon_min, lon_max, lat_min, lat_max
                )

            if n_time_gfs == 0:
                st.warning(
                    "Tidak ada data GFS pada periode yang dipilih. "
                    "Dataset GFS 'Best' UCAR adalah dataset operasional real-time "
                    "dan tidak berfungsi sebagai arsip jangka panjang."
                )
                st.info(
                    "Coba pilih periode yang masih berada dalam rentang data GFS "
                    "yang tersedia saat ini."
                )
            else:
                with st.spinner("Membuat peta streamline..."):
                    fig = make_streamline(
                        u, v, hgt,
                        tanggal_mulai, tanggal_akhir,
                        density, skip_speed, low_filter,
                        [lon_min, lon_max, lat_min, lat_max]
                    )

                st.pyplot(fig, use_container_width=True)
                st.success(
                    f"Peta berhasil dibuat menggunakan {n_time_gfs} timestep GFS "
                    f"pada periode {tanggal_mulai} s/d {tanggal_akhir}."
                )

                png = fig_to_png(fig)
                st.download_button(
                    "⬇️ Download Peta Streamline (PNG)",
                    data=png,
                    file_name=f"streamline_850mb_{tanggal_mulai}_{tanggal_akhir}.png",
                    mime="image/png",
                    use_container_width=True,
                )
                plt.close(fig)

        except Exception as e:
            st.error("Gagal mengambil atau memproses data GFS.")
            st.exception(e)

# ============================================================
# TAB 2 - SST & SSTA
# ============================================================
with tab2:
    st.subheader("🌊 Suhu Muka Laut (SST) dan Anomali SST (SSTA)")

    if st.button("🚀 Buat Peta SST & SSTA", type="primary", use_container_width=True):
        start_str = tanggal_mulai.isoformat()
        end_str = tanggal_akhir.isoformat()

        try:
            with st.spinner("Mengakses NOAA OISST dan menghitung rata-rata dasarian..."):
                sst, anom, n_time_sst = load_sst(
                    tanggal_mulai.year, start_str, end_str,
                    lon_min, lon_max, lat_min, lat_max
                )

            if n_time_sst == 0:
                st.error("Tidak ada data SST pada periode yang dipilih.")
            else:
                with st.spinner("Membuat peta SST..."):
                    fig_sst = make_sst_map(
                        sst,
                        tanggal_mulai, tanggal_akhir,
                        [lon_min, lon_max, lat_min, lat_max],
                        anomaly=False
                    )

                st.pyplot(fig_sst, use_container_width=True)
                st.success(
                    f"Peta SST/SSTA berhasil dibuat dari {n_time_sst} hari data NOAA OISST."
                )
                png_sst = fig_to_png(fig_sst)
                st.download_button(
                    "⬇️ Download Peta SST (PNG)",
                    data=png_sst,
                    file_name=f"sst_{tanggal_mulai}_{tanggal_akhir}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_sst",
                )
                plt.close(fig_sst)

                with st.spinner("Membuat peta SSTA..."):
                    fig_anom = make_sst_map(
                        anom,
                        tanggal_mulai, tanggal_akhir,
                        [lon_min, lon_max, lat_min, lat_max],
                        anomaly=True
                    )

                st.pyplot(fig_anom, use_container_width=True)
                png_anom = fig_to_png(fig_anom)
                st.download_button(
                    "⬇️ Download Peta SSTA (PNG)",
                    data=png_anom,
                    file_name=f"ssta_{tanggal_mulai}_{tanggal_akhir}.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_ssta",
                )
                plt.close(fig_anom)

        except Exception as e:
            st.error("Gagal mengambil atau memproses data NOAA OISST.")
            st.exception(e)

# ============================================================
# TAB 3 - INFORMASI
# ============================================================
with tab3:
    st.markdown("""
### Tentang aplikasi

Aplikasi ini mengubah workflow pada notebook menjadi antarmuka interaktif
berbasis **Streamlit**.

**Produk yang tersedia:**
- Streamline angin 850 mb.
- Deteksi pusat tekanan rendah (L).
- Label kecepatan angin dalam knot.
- SST rata-rata dasarian.
- Anomali SST/SSTA rata-rata dasarian.

**Sumber data:**
- **NCEP GFS 0,25°** melalui UCAR THREDDS untuk angin 850 mb dan geopotential height.
- **NOAA OISST High Resolution** melalui NOAA PSL untuk SST dan anomali SST.

**Catatan penting:**
- Data diambil secara online ketika tombol analisis dijalankan.
- Hasil dirata-ratakan pada periode dasarian yang dipilih.
- Cache digunakan agar permintaan data yang sama tidak selalu diunduh ulang.
- Aplikasi memeriksa jumlah timestep/hari yang benar-benar tersedia sebelum
  membuat peta, sehingga periode kosong tidak lagi dianggap sebagai data valid.
- Dataset GFS `Best` UCAR adalah **time series forecast operasional real-time**,
  bukan arsip jangka panjang. Untuk periode yang sudah terlalu lama, gunakan
  arsip GFS dari NCAR/RDA.
- Untuk deployment Streamlit Cloud, koneksi internet dari server harus dapat
  mengakses endpoint THREDDS/OPeNDAP sumber data.
""")

st.divider()
st.caption(
    "Dikembangkan dari notebook: BIKIN STREAMLINE DAN SST INDO_EDIT.ipynb | "
    "Stasiun Klimatologi Jawa Tengah"
)

"""
Aplikasi Streamlit: Streamline Angin & SST Wilayah Indonesia
Sumber data: NOAA/PSL (OPeNDAP) & UCAR THREDDS (GFS)

Dikembangkan dari notebook riset milik pengguna (Stasiun Klimatologi Jawa Tengah).
"""

import io
from datetime import date, timedelta

import numpy as np
import streamlit as st
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.ndimage as ndimage

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Streamline & SST Indonesia",
    page_icon="🌊",
    layout="wide",
)

FOOTER_TEXT = "Data: NOAA/PSL & NCEP GFS  |  @ Stasiun Klimatologi Jawa Tengah"


def tambah_grid_manual(ax, extent):
    """Gambar garis grid + label lat/lon tanpa draw_labels=True.

    Cartopy punya bug lama pada gridliner (draw_labels=True) yang kadang
    gagal membentuk poligon batas peta ('Points of LinearRing do not
    form a closed linestring') saat figure di-render. Cara ini memakai
    tick manual yang hasilnya sama persis secara visual tapi tidak lewat
    kode gridliner yang bermasalah.
    """
    ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5, color='gray', zorder=5)

    langkah = 5 if (extent[1] - extent[0]) > 20 else 2
    xticks = np.arange(np.floor(extent[0] / langkah) * langkah, extent[1] + langkah, langkah)
    yticks = np.arange(np.floor(extent[2] / langkah) * langkah, extent[3] + langkah, langkah)
    xticks = xticks[(xticks >= extent[0]) & (xticks <= extent[1])]
    yticks = yticks[(yticks >= extent[2]) & (yticks <= extent[3])]

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=9)


def tambah_fitur_aman(ax, feature, **kwargs):
    """Tambahkan fitur peta (LAND/OCEAN/COASTLINE/BORDERS) dengan aman.

    Beberapa geometri di data Natural Earth kadang tidak valid secara
    ketat untuk shapely versi baru (error 'Points of LinearRing do not
    form a closed linestring'). Daripada bikin seluruh proses gagal,
    lapisan yang bermasalah cukup dilewati saja.
    """
    try:
        ax.add_feature(feature, **kwargs)
    except Exception:
        pass

# Warna & batas SST ala JMA (dipakai untuk peta SST rata-rata)
WARNA_JMA = ['#8A2BE2', '#0000FF', '#1E90FF', '#00FFFF', '#00FF00',
             '#ADFF2F', '#FFD700', '#FFA500', '#FF4500', '#FF0000', '#FF00FF']
BATAS_SST = [25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 31.0]


# ==========================================================
# FUNGSI AMBIL DATA (di-cache biar tidak fetch ulang tiap interaksi)
# ==========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def ambil_data_streamline_ncep(tahun, tgl_mulai, tgl_akhir, lon_min, lon_max, lat_min, lat_max, level_mb):
    """Ambil data U, V, HGT dari NCEP/NCAR Reanalysis (data historis, ada lag 2-3 minggu)."""
    url_u = f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/uwnd.{tahun}.nc"
    url_v = f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/vwnd.{tahun}.nc"
    url_h = f"https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.{tahun}.nc"

    ds_u = xr.open_dataset(url_u)
    ds_v = xr.open_dataset(url_v)
    ds_h = xr.open_dataset(url_h)

    lon_slice = slice(lon_min, lon_max)
    lat_slice = slice(lat_max, lat_min)  # NCEP lat: 90 -> -90, jadi dibalik

    u = ds_u['uwnd'].sel(level=level_mb, time=slice(tgl_mulai, tgl_akhir),
                          lon=lon_slice, lat=lat_slice).mean(dim='time')
    v = ds_v['vwnd'].sel(level=level_mb, time=slice(tgl_mulai, tgl_akhir),
                          lon=lon_slice, lat=lat_slice).mean(dim='time')
    h = ds_h['hgt'].sel(level=level_mb, time=slice(tgl_mulai, tgl_akhir),
                         lon=lon_slice, lat=lat_slice).mean(dim='time')

    lons, lats = np.meshgrid(u.lon.values, u.lat.values)
    return u.values, v.values, h.values, lons, lats


@st.cache_data(show_spinner=False, ttl=3600)
def ambil_data_streamline_gfs(tgl_mulai, tgl_akhir, lon_min, lon_max, lat_min, lat_max, level_mb):
    """Ambil data U, V, HGT dari GFS 0.25° real-time via UCAR THREDDS."""
    url = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"
    ds = xr.open_dataset(url)

    var_u = ds['u-component_of_wind_isobaric']
    var_v = ds['v-component_of_wind_isobaric']
    var_h = ds['Geopotential_height_isobaric']

    vert_dim_u = [d for d in var_u.dims if 'isobaric' in d][0]
    vert_dim_v = [d for d in var_v.dims if 'isobaric' in d][0]
    vert_dim_h = [d for d in var_h.dims if 'isobaric' in d][0]

    def pilih_level(ds, vert_dim, level_mb):
        return level_mb * 100 if ds[vert_dim].max() > 2000 else level_mb

    lv_u = pilih_level(ds, vert_dim_u, level_mb)
    lv_v = pilih_level(ds, vert_dim_v, level_mb)
    lv_h = pilih_level(ds, vert_dim_h, level_mb)

    lon_slice = slice(lon_min, lon_max)
    lat_slice = slice(lat_max, lat_min)

    u = var_u.sel(lon=lon_slice, lat=lat_slice, time=slice(tgl_mulai, tgl_akhir)) \
        .sel({vert_dim_u: lv_u}, method='nearest').mean(dim='time')
    v = var_v.sel(lon=lon_slice, lat=lat_slice, time=slice(tgl_mulai, tgl_akhir)) \
        .sel({vert_dim_v: lv_v}, method='nearest').mean(dim='time')
    h = var_h.sel(lon=lon_slice, lat=lat_slice, time=slice(tgl_mulai, tgl_akhir)) \
        .sel({vert_dim_h: lv_h}, method='nearest').mean(dim='time')

    lons, lats = np.meshgrid(u.lon.values, u.lat.values)
    return u.values, v.values, h.values, lons, lats


@st.cache_data(show_spinner=False, ttl=3600)
def ambil_data_sst(tahun, tgl_mulai, tgl_akhir, lon_min, lon_max, lat_min, lat_max):
    """Ambil data SST harian & anomalinya dari NOAA OISST v2 High Res."""
    url_sst = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{tahun}.nc"
    url_anom = f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.anom.{tahun}.nc"

    ds_sst = xr.open_dataset(url_sst)
    ds_anom = xr.open_dataset(url_anom)

    lon_slice = slice(lon_min, lon_max)
    lat_slice = slice(lat_min, lat_max)  # OISST lat: -90 -> 90 (naik)

    sst = ds_sst['sst'].sel(time=slice(tgl_mulai, tgl_akhir), lon=lon_slice, lat=lat_slice).mean(dim='time')
    anom = ds_anom['anom'].sel(time=slice(tgl_mulai, tgl_akhir), lon=lon_slice, lat=lat_slice).mean(dim='time')

    lons, lats = np.meshgrid(sst.lon.values, sst.lat.values)
    return sst.values, anom.values, lons, lats


# ==========================================================
# FUNGSI VISUALISASI
# ==========================================================
def plot_streamline(u, v, hgt, lons, lats, extent, judul, tgl_mulai, tgl_akhir,
                     density, tampilkan_kecepatan, tampilkan_L, skip):
    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    tambah_fitur_aman(ax, cfeature.LAND, facecolor='#FFFF66', edgecolor='black', linewidth=0.5, zorder=1)
    tambah_fitur_aman(ax, cfeature.OCEAN, facecolor='#E6F2FF', zorder=0)
    tambah_fitur_aman(ax, cfeature.BORDERS, linestyle=':', alpha=0.7, zorder=1)
    tambah_fitur_aman(ax, cfeature.COASTLINE, linewidth=0.8, zorder=1)

    ax.streamplot(lons, lats, u, v, color='blue', linewidth=1.0, density=density,
                  arrowsize=1.2, zorder=2, transform=ccrs.PlateCarree())

    if tampilkan_kecepatan:
        speed_kt = np.sqrt(u ** 2 + v ** 2) * 1.94384
        for i in range(0, u.shape[0], skip):
            for j in range(0, u.shape[1], skip):
                lon_val, lat_val, spd_val = lons[i, j], lats[i, j], speed_kt[i, j]
                if extent[0] <= lon_val <= extent[1] and extent[2] <= lat_val <= extent[3] and not np.isnan(spd_val):
                    ax.text(lon_val, lat_val, f"{int(round(spd_val))} kt", color='red', fontsize=9,
                            fontweight='bold', ha='center', va='center', zorder=3, transform=ccrs.PlateCarree())

    if tampilkan_L:
        local_min = ndimage.minimum_filter(hgt, size=60) == hgt
        min_idx = np.where(local_min)
        for y, x in zip(min_idx[0], min_idx[1]):
            if 10 < x < hgt.shape[1] - 10 and 10 < y < hgt.shape[0] - 10 and not np.isnan(hgt[y, x]):
                ax.text(lons[y, x], lats[y, x], 'L', color='red', fontsize=18, fontweight='bold',
                        ha='center', va='center', zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.2", edgecolor='red', facecolor='white', alpha=0.9, lw=1.5),
                        transform=ccrs.PlateCarree())

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    tambah_grid_manual(ax, extent)

    plt.title(f'{judul}\nRata-rata ({tgl_mulai} s/d {tgl_akhir})', fontsize=15, fontweight='bold', pad=15)
    plt.figtext(0.5, 0.01, FOOTER_TEXT, ha='center', va='center', fontsize=10, fontweight='bold',
                color='black', style='italic')
    fig.tight_layout()
    return fig


def plot_sst(data, lons, lats, extent, judul, tgl_mulai, tgl_akhir, is_anomali):
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    tambah_fitur_aman(ax, cfeature.LAND, facecolor='white', edgecolor='black', zorder=2)
    tambah_fitur_aman(ax, cfeature.COASTLINE, linewidth=0.8, zorder=2)

    if is_anomali:
        plot = ax.contourf(lons, lats, data, levels=np.arange(-2.5, 2.75, 0.25),
                            cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=1)
        label = 'Anomali Suhu (°C)'
    else:
        cmap_sst = mcolors.ListedColormap(WARNA_JMA)
        norm_sst = mcolors.BoundaryNorm(BATAS_SST, cmap_sst.N)
        plot = ax.contourf(lons, lats, data, levels=BATAS_SST, cmap=cmap_sst, norm=norm_sst,
                            extend='both', transform=ccrs.PlateCarree(), zorder=1)
        label = 'Suhu Muka Laut (°C)'

    cbar = plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label(label, fontsize=11, fontweight='bold')

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    tambah_grid_manual(ax, extent)

    plt.title(f'{judul}\nRata-rata ({tgl_mulai} s/d {tgl_akhir})', fontsize=14, fontweight='bold', pad=15)
    plt.figtext(0.5, -0.02, FOOTER_TEXT, ha='center', va='center', fontsize=10, fontweight='bold',
                color='black', style='italic')
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    buf.seek(0)
    return buf


# ==========================================================
# SIDEBAR - PENGATURAN WILAYAH (dipakai bersama semua tab)
# ==========================================================
st.sidebar.header("⚙️ Pengaturan Wilayah")
lon_min = st.sidebar.number_input("Longitude min", value=90.0, step=1.0)
lon_max = st.sidebar.number_input("Longitude max", value=145.0, step=1.0)
lat_min = st.sidebar.number_input("Latitude min", value=-15.0, step=1.0)
lat_max = st.sidebar.number_input("Latitude max", value=15.0, step=1.0)
extent = [lon_min, lon_max, lat_min, lat_max]

st.sidebar.caption(
    "Default: wilayah Indonesia (90–145°E, 15°S–15°N). "
    "Ubah sesuai kebutuhan (mis. fokus ke satu region)."
)

st.title("🌬️🌊 Streamline Angin & SST Wilayah Indonesia")
st.caption("Dibangun dari script analisis milik Stasiun Klimatologi Jawa Tengah — data NOAA/PSL & NCEP GFS")

tab_streamline, tab_sst = st.tabs(["🌬️ Streamline Angin", "🌊 SST & Anomali"])

# ==========================================================
# TAB 1 — STREAMLINE
# ==========================================================
with tab_streamline:
    with st.form("form_streamline"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sumber = st.selectbox(
                "Sumber data",
                ["GFS Real-time (UCAR THREDDS)", "NCEP Reanalysis (historis, lag 2-3 minggu)"],
                help="GFS cocok untuk kondisi terkini/forecast. NCEP Reanalysis cocok untuk data historis dasarian.",
            )
        with col2:
            level_mb = st.selectbox("Level tekanan (mb)", [925, 850, 700, 500], index=1)
        with col3:
            density = st.slider("Kerapatan garis (density)", 2.0, 10.0, 7.0, 0.5)

        col4, col5 = st.columns(2)
        with col4:
            tgl_mulai = st.date_input("Tanggal mulai", value=date.today() - timedelta(days=10))
        with col5:
            tgl_akhir = st.date_input("Tanggal akhir", value=date.today())

        col6, col7 = st.columns(2)
        with col6:
            tampilkan_kecepatan = st.checkbox("Tampilkan label kecepatan (knot)", value=True)
        with col7:
            tampilkan_L = st.checkbox("Tampilkan pusat tekanan rendah (L)", value=True)

        submit_streamline = st.form_submit_button("🗺️ Buat Peta Streamline", use_container_width=True)

    if submit_streamline:
        if tgl_mulai > tgl_akhir:
            st.error("Tanggal mulai harus sebelum tanggal akhir.")
        else:
            try:
                with st.spinner(f"Mengambil data dari {sumber}..."):
                    if sumber.startswith("GFS"):
                        u, v, hgt, lons, lats = ambil_data_streamline_gfs(
                            str(tgl_mulai), str(tgl_akhir), lon_min, lon_max, lat_min, lat_max, level_mb)
                    else:
                        tahun = tgl_mulai.year
                        u, v, hgt, lons, lats = ambil_data_streamline_ncep(
                            tahun, str(tgl_mulai), str(tgl_akhir), lon_min, lon_max, lat_min, lat_max, level_mb)

                if np.all(np.isnan(u)):
                    st.warning("⚠️ Semua data kosong untuk rentang tanggal/wilayah ini. Coba ganti tanggal.")
                else:
                    with st.spinner("Menggambar peta..."):
                        fig = plot_streamline(
                            u, v, hgt, lons, lats, extent,
                            f"Analisis Streamline {level_mb} mb & Tekanan Rendah",
                            tgl_mulai, tgl_akhir, density, tampilkan_kecepatan, tampilkan_L,
                            skip=max(1, u.shape[0] // 12),
                        )
                    st.pyplot(fig, use_container_width=True)
                    st.download_button(
                        "⬇️ Unduh PNG", data=fig_to_png_bytes(fig),
                        file_name=f"streamline_{tgl_mulai}_{tgl_akhir}.png", mime="image/png",
                    )
            except Exception as e:
                st.error(f"Gagal mengambil/memproses data: {e}")

# ==========================================================
# TAB 2 — SST & ANOMALI
# ==========================================================
with tab_sst:
    with st.form("form_sst"):
        col1, col2 = st.columns(2)
        with col1:
            tgl_mulai_sst = st.date_input("Tanggal mulai ", value=date.today() - timedelta(days=10), key="sst_mulai")
        with col2:
            tgl_akhir_sst = st.date_input("Tanggal akhir ", value=date.today(), key="sst_akhir")

        submit_sst = st.form_submit_button("🌡️ Buat Peta SST & Anomali", use_container_width=True)

    if submit_sst:
        if tgl_mulai_sst > tgl_akhir_sst:
            st.error("Tanggal mulai harus sebelum tanggal akhir.")
        else:
            try:
                tahun = tgl_mulai_sst.year
                with st.spinner("Mengambil data SST dari NOAA OISST..."):
                    sst, anom, lons, lats = ambil_data_sst(
                        tahun, str(tgl_mulai_sst), str(tgl_akhir_sst), lon_min, lon_max, lat_min, lat_max)

                if np.all(np.isnan(sst)):
                    st.warning("⚠️ Semua data kosong untuk rentang tanggal/wilayah ini. Coba ganti tanggal.")
                else:
                    colA, colB = st.columns(2)
                    with colA:
                        fig_sst = plot_sst(sst, lons, lats, extent, "Suhu Muka Laut (SST) Wilayah Indonesia",
                                           tgl_mulai_sst, tgl_akhir_sst, is_anomali=False)
                        st.pyplot(fig_sst, use_container_width=True)
                        st.download_button("⬇️ Unduh PNG (SST)", data=fig_to_png_bytes(fig_sst),
                                            file_name=f"sst_mean_{tgl_mulai_sst}_{tgl_akhir_sst}.png",
                                            mime="image/png", key="dl_sst")
                    with colB:
                        fig_anom = plot_sst(anom, lons, lats, extent, "Anomali Suhu Muka Laut (SSTA) Wilayah Indonesia",
                                            tgl_mulai_sst, tgl_akhir_sst, is_anomali=True)
                        st.pyplot(fig_anom, use_container_width=True)
                        st.download_button("⬇️ Unduh PNG (Anomali)", data=fig_to_png_bytes(fig_anom),
                                            file_name=f"sst_anomaly_{tgl_mulai_sst}_{tgl_akhir_sst}.png",
                                            mime="image/png", key="dl_anom")
            except Exception as e:
                st.error(f"Gagal mengambil/memproses data: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan Streamlit • Data: NOAA/PSL & NCEP GFS via OPeNDAP/THREDDS")

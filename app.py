# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
import datetime as dt
import tempfile
import urllib.request
import contextlib
import os
from urllib.error import HTTPError, URLError
from io import BytesIO

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="RAP/RUC Visualizer", layout="wide")
st.title("RAP / RUC Weather Visualization")

# -------------------- Constants --------------------
CONUS_EXTENT = [-125, -66.5, 20, 55]

# Era boundaries (practical for NCEI archive):
RUC2_TO_RUC130 = dt.datetime(2008, 10, 29, 23, tzinfo=dt.UTC)  # ruc2anl_252 (GRIB1) -> ruc2anl_130 (GRIB2)
RUC_TO_RAP     = dt.datetime(2012, 5, 1, 12, tzinfo=dt.UTC)    # RAP replaces RUC

PRODUCTS = [
    "Dewpoint, MSLP, and wind barbs",
    "500 mb wind and height",
    "850 mb wind and height",
    "Surface-based CAPE and wind barbs",
    "Convective precipitation",
]

# -------------------- UI --------------------
col1, col2, col3, col4 = st.columns(4)
utc_year_default = dt.datetime.now(dt.UTC).year
with col1:
    year = st.number_input("Year (UTC)", 1999, utc_year_default, 2014)
with col2:
    month = st.number_input("Month", 1, 12, 4)
with col3:
    day = st.number_input("Day", 1, 31, 28)
with col4:
    hour = st.number_input("Hour (UTC)", 0, 23, 0)

col5, col6 = st.columns([2, 2])
with col5:
    product = st.selectbox("Product", PRODUCTS, index=0)
with col6:
    generate_btn = st.button("Generate", type="primary")

# -------------------- Color tables (match your scripts) --------------------
def dewpoint_cmap():
    colors = [
        (152,109, 77),(150,108, 76),(148,107, 76),(146,106, 75),(144,105, 75),(142,104, 74),
        (140,102, 74),(138,101, 73),(136,100, 72),(134, 99, 72),(132, 98, 71),(130, 97, 71),
        (128, 96, 70),(126, 95, 70),(124, 94, 69),(122, 93, 68),(120, 91, 68),(118, 90, 67),
        (116, 89, 67),(114, 88, 66),(113, 87, 66),(109, 85, 64),(107, 84, 64),(105, 83, 63),
        (103, 82, 63),(101, 80, 62),( 99, 79, 61),( 97, 78, 61),( 95, 77, 60),( 93, 76, 60),
        ( 91, 75, 59),( 89, 74, 59),( 87, 73, 58),( 85, 72, 57),( 83, 71, 57),( 81, 69, 56),
        ( 79, 68, 56),( 77, 67, 55),( 75, 66, 55),( 73, 65, 54),( 71, 64, 54),( 69, 63, 53),
        ( 77, 67, 52),( 81, 71, 56),( 86, 76, 60),( 90, 80, 65),( 94, 85, 69),( 99, 89, 73),
        (103, 94, 77),(107, 98, 81),(112,103, 86),(116,107, 90),(120,112, 94),(125,116, 98),
        (129,121,103),(133,125,107),(138,130,111),(142,134,115),(146,139,119),(151,143,124),
        (155,148,128),(159,152,132),(164,157,137),(168,161,141),(173,166,145),(189,179,156),
        (189,179,156),(188,184,161),(193,188,165),(201,197,173),(201,197,173),(210,206,182),
        (223,220,194),(227,224,198),(231,229,202),(235,233,207),(240,238,211),(244,242,215),
        (230,245,230),(215,240,215),(200,234,200),(185,229,185),(170,223,170),(155,218,155),
        (140,213,140),(125,207,125),(110,202,110),( 95,196, 95),( 80,191, 80),( 65,186, 65),
        ( 48,174, 48),( 44,163, 44),( 39,153, 39),( 35,142, 35),( 30,131, 30),( 26,121, 26),
        ( 21,110, 21),( 17, 99, 17),( 12, 89, 12),(  8, 78,  8),( 97,163,175),( 88,150,160),
        ( 80,137,146),( 71,123,131),( 62,110,116),( 54, 97,102),( 45, 84, 87),( 36, 70, 72),
        ( 28, 57, 58),( 19, 44, 43),(102,102,154),( 96, 94,148),( 89, 86,142),( 83, 78,136),
        ( 77, 70,130),( 70, 62,124),( 64, 54,118),( 58, 46,112),( 51, 38,106),( 45, 30,100),
        (114, 64,113),(120, 69,115),(125, 75,117),(131, 80,118),(136, 86,120),(142, 91,122),
        (147, 97,124),(153,102,125),(158,108,127),(164,113,129)
    ]
    cols = [(r/255,g/255,b/255) for r,g,b in colors]
    return ListedColormap(cols), BoundaryNorm(np.linspace(-40, 90, len(cols)+1), len(cols))

def wind500_cmap():
    pw500speed_colors = [
        (230,244,255),(219,240,254),(209,235,254),(198,231,253),(188,227,253),(177,223,252),
        (167,219,252),(156,214,251),(146,210,251),(135,206,250),(132,194,246),(129,183,241),
        (126,171,237),(123,160,232),(121,148,228),(118,136,223),(115,125,219),(112,113,214),
        (109,102,210),(106, 90,205),(118, 96,207),(131,102,208),(143,108,210),(156,114,211),
        (168,120,213),(180,126,214),(193,132,216),(205,138,217),(218,144,219),(230,150,220),
        (227,144,217),(224,138,214),(221,132,211),(218,126,208),(215,120,205),(212,114,202),
        (209,108,199),(206,102,196),(203, 96,193),(200, 90,190),(196, 83,186),(192, 76,182),
        (188, 69,178),(184, 62,174),(180, 55,170),(176, 48,166),(172, 41,162),(168, 34,158),
        (164, 27,154),(160, 20,150),(164, 16,128),(168, 14,117),(172, 14,117),(176, 12,106),
        (180,  8, 95),(184,  6, 73),(188,  4, 62),(192,  2, 51),(200,  0, 40),(200,  0, 40),
        (202,  4, 42),(204,  8, 44),(208, 12, 44),(210, 20, 50),(212, 24, 52),(212, 24, 52),
        (214, 28, 54),(218, 36, 58),(220, 40, 60),(222, 44, 62),(224, 48, 64),(226, 52, 66),
        (228, 56, 68),(230, 60, 70),(232, 64, 72),(234, 68, 74),(236, 72, 76),(238, 76, 78),
        (240, 80, 80),(241, 96, 82),(242,112, 84),(243,128, 86),(244,144, 88),(245,160, 90),
        (246,176, 92),(247,192, 94),(248,208, 96),(249,224, 98),(250,240,100),(247,235, 97),
        (244,230, 94),(241,225, 91),(238,220, 88),(235,215, 85),(232,210, 82),(229,205, 79),
        (226,200, 76),(223,195, 73),(220,190, 70),(217,185, 67),(214,180, 64),(211,175, 61),
        (208,170, 58),(205,165, 55),(202,160, 52),(199,155, 49),(196,150, 46),(193,145, 43),
        (190,140, 40),(187,135, 37),(184,130, 34),(181,125, 31),(178,120, 28),(175,115, 25),
        (172,110, 22),(169,105, 19),(166,100, 16),(163, 95, 13),(160, 90, 10),(160, 90, 10)
    ]
    cols = [(r/255,g/255,b/255) for r,g,b in pw500speed_colors]
    levels = np.linspace(20, 140, len(cols)+1)
    return ListedColormap(cols), BoundaryNorm(levels, len(cols)), levels

def wind850_cmap():
    pw850speed_colors = [
        (240,248,255),(219,240,254),(198,231,253),(177,223,252),(156,214,251),(135,206,250),
        (129,183,241),(123,160,232),(118,136,223),(112,113,214),(106, 90,205),(131,102,208),
        (156,114,211),(180,126,214),(205,138,217),(230,150,220),(224,138,214),(218,126,208),
        (212,114,202),(206,102,196),(200, 90,190),(192, 76,182),(184, 62,174),(176, 48,166),
        (168, 34,158),(160, 20,150),(168, 16,128),(176, 12,106),(184,  4, 62),(200,  0, 40),
        (200,  0, 40),(208, 16, 48),(212, 24, 52),(216, 32, 56),(220, 40, 60),(224, 48, 64),
        (228, 56, 68),(232, 64, 72),(236, 72, 76),(240, 80, 80),(242,112, 84),(244,144, 88),
        (246,176, 92),(248,208, 96),(250,240,100),(244,230, 94),(238,220, 88),(232,210, 82),
        (226,200, 76),(220,190, 70),(214,180, 64),(208,170, 58),(202,160, 52),(196,150, 46),
        (190,140, 40),(184,130, 34),(178,120, 28),(172,110, 22),(166,100, 16),(160, 90, 10)
    ]
    cols = [(r/255,g/255,b/255) for r,g,b in pw850speed_colors]
    levels = np.linspace(20, 80, len(cols)+1)
    return ListedColormap(cols), BoundaryNorm(levels, len(cols)), levels

def cp_cmap_and_norm():
    # white -> 0.1 mm then jet -> 10 mm
    jet = plt.get_cmap("jet")
    whites = [(1,1,1)] * 10
    n_colors = 99
    jet_colors = [jet(i / (n_colors - 1)) for i in range(n_colors)]
    custom = LinearSegmentedColormap.from_list("custom_jet", whites + jet_colors, N=109)
    norm = mcolors.Normalize(vmin=0, vmax=10)
    return custom, norm

# -------------------- Data access helpers --------------------
@st.cache_data(show_spinner=False)
def download_grib(year, month, day, hour):
    """
    Tries multiple URL patterns + filenames and verifies non-trivial file size.

    Matches your samples:
      - RAP new path (minutes in name):
        https://www.ncei.noaa.gov/data/rapid-refresh/access/rap-130-13km/analysis/202505/20250516/rap_130_20250516_0000_000.grb2
      - RAP historical path (minutes in name):
        https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/201404/20140428/rap_130_20140428_0000_000.grb2
      - RUC2 GRIB1 historical (minutes in name):
        https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/200702/20070202/ruc2anl_252_20070202_0000_000.grb
    """
    when = dt.datetime(int(year), int(month), int(day), int(hour), tzinfo=dt.UTC)
    yyyymm = when.strftime("%Y%m")
    yyyymmdd = when.strftime("%Y%m%d")

    def looks_like_grib(path: str) -> bool:
        try:
            sz = os.path.getsize(path)
        except OSError:
            return False
        return sz > 100_000  # avoid saving HTML index/error pages

    # Filename templates: try minutes and seconds flavors
    if when < RUC2_TO_RUC130:
        # RUC2 (grid 252), GRIB1
        fn_templates = [
            f"ruc2anl_252_{when.strftime('%Y%m%d_%H%M')}_000.grb",
            f"ruc2anl_252_{when.strftime('%Y%m%d_%H%S')}_000.grb",
        ]
        roots = [
            f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{yyyymm}/{yyyymmdd}/",
        ]
    elif when < RUC_TO_RAP:
        # RUC (grid 130), GRIB2
        fn_templates = [
            f"ruc2anl_130_{when.strftime('%Y%m%d_%H%M')}_000.grb2",
            f"ruc2anl_130_{when.strftime('%Y%m%d_%H%S')}_000.grb2",
        ]
        roots = [
            f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{yyyymm}/{yyyymmdd}/",
        ]
    else:
        # RAP (grid 130), GRIB2
        fn_templates = [
            f"rap_130_{when.strftime('%Y%m%d_%H%M')}_000.grb2",
            f"rap_130_{when.strftime('%Y%m%d_%H%S')}_000.grb2",
        ]
        roots = [
            f"https://www.ncei.noaa.gov/data/rapid-refresh/access/rap-130-13km/analysis/{yyyymm}/{yyyymmdd}/",
            f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{yyyymm}/{yyyymmdd}/",
        ]

    tmpdir = tempfile.mkdtemp()
    last_err = None

    for fname in fn_templates:
        for root in roots:
            url = root + fname
            local = os.path.join(tmpdir, fname)
            try:
                # Some servers don't support HEAD; just download and check size.
                urllib.request.urlretrieve(url, local)
                if looks_like_grib(local):
                    return local
                else:
                    last_err = RuntimeError(f"Downloaded but too small: {url}")
            except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
                last_err = e
                continue

    raise RuntimeError(f"Could not locate GRIB for {when:%Y-%m-%d %H:00} UTC. Last error: {last_err}")

def open_grbs(path):
    return pygrib.open(path)

# Flexible selector that tries multiple labels across eras
def select_one(grbs, **kwargs):
    try:
        return grbs.select(**kwargs)[0]
    except Exception:
        return None

def pick_var(grbs, attempts):
    """
    attempts: list[dict] for pygrib.select(**kwargs) attempts
    """
    for a in attempts:
        g = select_one(grbs, **a)
        if g is not None:
            return g
    raise KeyError(f"Could not find variable with attempts: {attempts}")

# -------------------- Plot helpers --------------------
def base_ax():
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    try:
        ax.add_feature(cfeature.STATES, linewidth=0.4)
    except Exception:
        pass
    return ax

def finish_and_return(fig, title):
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "Plotted by Sekai Chandra (@Sekai_WX)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- Product renderers --------------------
def plot_dew_mslp_barbs(grbs, when):
    d2m = pick_var(grbs, [
        {"name":"2 metre dewpoint temperature"},
        {"name":"2 m above ground Dew point temperature"},
    ])
    u10 = pick_var(grbs, [
        {"name":"10 metre U wind component", "level":10},
        {"name":"U component of wind", "typeOfLevel":"heightAboveGround", "level":10},
    ])
    v10 = pick_var(grbs, [
        {"name":"10 metre V wind component", "level":10},
        {"name":"V component of wind", "typeOfLevel":"heightAboveGround", "level":10},
    ])
    mslp = pick_var(grbs, [
        {"name":"MSLP (MAPS System Reduction)"},
        {"shortName":"mslet"},  # occasional alternate
    ])

    dewK = d2m.values
    dewF = (dewK - 273.15)*9/5 + 32.0
    u = u10.values
    v = v10.values
    p = mslp.values / 100.0
    lats, lons = d2m.latlons()

    cmap, norm = dewpoint_cmap()

    fig = plt.figure(figsize=(13.5, 9))
    ax = base_ax()

    cf = ax.contourf(
        lons, lats, dewF,
        levels=np.linspace(-40, 90, cmap.N+1),
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="both"
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Dewpoint (°F)")

    levels = np.arange(900, 1100, 2)  # 2 mb spacing
    cs = ax.contour(lons, lats, p, levels=levels, colors="black", linewidths=0.8, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)

    ax.barbs(lons[::10, ::10], lats[::10, ::10], u[::10, ::10], v[::10, ::10], length=6, transform=ccrs.PlateCarree())

    title = f"Dewpoint (°F), MSLP (mb), and 10 m wind — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_wind_height(grbs, level_mb, is_500, when):
    u = pick_var(grbs, [{"name":"U component of wind", "level":level_mb}])
    v = pick_var(grbs, [{"name":"V component of wind", "level":level_mb}])
    z = pick_var(grbs, [{"name":"Geopotential height", "level":level_mb}])

    uu = u.values; vv = v.values
    zvals = z.values  # gpm
    lats, lons = z.latlons()

    wspd_kt = np.sqrt(uu**2 + vv**2) * 1.94384
    if is_500:
        cmap, bnorm, levels = wind500_cmap()
        h_levels = np.arange(480, 600, 6)  # dam
        name = "500 mb wind & height"
    else:
        cmap, bnorm, levels = wind850_cmap()
        h_levels = np.arange(120, 180, 3)  # dam
        name = "850 mb wind & height"

    fig = plt.figure(figsize=(13.5, 9))
    ax = base_ax()

    cf = ax.contourf(lons, lats, wspd_kt, levels=levels, cmap=cmap, norm=bnorm, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Wind Speed (kt)")

    cs = ax.contour(lons, lats, zvals/10.0, levels=h_levels, colors="black", linewidths=1.0, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")

    ax.barbs(lons[::10, ::10], lats[::10, ::10], uu[::10, ::10], vv[::10, ::10], length=6, transform=ccrs.PlateCarree())

    title = f"{name} — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_sbcape_barbs(grbs, when):
    # Prefer surface CAPE; fall back to common pressureFromGroundLayer proxies
    cape_grb = select_one(grbs, name="Convective available potential energy", typeOfLevel="surface")
    if cape_grb is None:
        cape_grb = select_one(grbs, name="Convective available potential energy", typeOfLevel="pressureFromGroundLayer", level=25500) \
                   or select_one(grbs, name="Convective available potential energy", typeOfLevel="pressureFromGroundLayer", level=9000) \
                   or select_one(grbs, name="Convective available potential energy", typeOfLevel="pressureFromGroundLayer", level=18000)
    if cape_grb is None:
        raise KeyError("Could not locate SBCAPE in this file.")

    u10 = pick_var(grbs, [
        {"name":"10 metre U wind component", "level":10},
        {"name":"U component of wind", "typeOfLevel":"heightAboveGround", "level":10},
    ])
    v10 = pick_var(grbs, [
        {"name":"10 metre V wind component", "level":10},
        {"name":"V component of wind", "typeOfLevel":"heightAboveGround", "level":10},
    ])

    cape = np.clip(cape_grb.values, 0, 7000)
    lats, lons = cape_grb.latlons()
    u = u10.values; v = v10.values

    fig = plt.figure(figsize=(13.5, 9))
    ax = base_ax()

    turbo = plt.get_cmap("turbo")
    cf = ax.contourf(lons, lats, cape, levels=np.linspace(0,7000,71), cmap=turbo, extend="max", transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("SBCAPE (J/kg) — capped at 7000")

    ax.barbs(lons[::10, ::10], lats[::10, ::10], u[::10, ::10], v[::10, ::10], length=6, transform=ccrs.PlateCarree())

    title = f"Surface-based CAPE (J/kg) & 10 m wind — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_conv_precip(grbs, when):
    cp = pick_var(grbs, [{"name":"Convective precipitation (water)", "typeOfLevel":"surface"}])
    cp_mm = cp.values  # kg m^-2 == mm
    lats, lons = cp.latlons()

    cm, norm = cp_cmap_and_norm()

    fig = plt.figure(figsize=(13.5, 9))
    ax = base_ax()
    cf = ax.contourf(
        lons, lats, cp_mm,
        levels=np.linspace(0,10,51),
        cmap=cm, norm=norm,
        transform=ccrs.PlateCarree(), extend="max"
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Convective Precipitation (mm)")

    title = f"Convective Precipitation — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

# -------------------- Run --------------------
if generate_btn:
    local_path = None
    try:
        when = dt.datetime(int(year), int(month), int(day), int(hour), tzinfo=dt.UTC)
        st.info(f"Fetching RAP/RUC analysis for {when:%Y-%m-%d %H:00} UTC …")
        local_path = download_grib(year, month, day, hour)

        with open_grbs(local_path) as grbs:
            if product == "Dewpoint, MSLP, and wind barbs":
                buf = plot_dew_mslp_barbs(grbs, when)
            elif product == "500 mb wind and height":
                buf = plot_wind_height(grbs, 500, is_500=True, when=when)
            elif product == "850 mb wind and height":
                buf = plot_wind_height(grbs, 850, is_500=False, when=when)
            elif product == "Surface-based CAPE and wind barbs":
                buf = plot_sbcape_barbs(grbs, when)
            elif product == "Convective precipitation":
                buf = plot_conv_precip(grbs, when)
            else:
                st.error("Unknown product.")
                st.stop()

        st.success("Visualization generated!")
        st.image(buf, caption=f"{product} — {when:%Y-%m-%d %H:00 UTC}", use_column_width=True)
        st.download_button(
            "Download PNG",
            data=buf,
            file_name=f"RAP_RUC_{product.replace(' ','_')}_{when:%Y%m%d%H}.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("If an hour 404s in one path, the app already falls back to alternates. Try ±1–2 hours if truly missing.")
    finally:
        try:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

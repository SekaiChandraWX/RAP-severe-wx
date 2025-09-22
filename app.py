import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pygrib
import datetime as dt
import tempfile
import urllib.request
import os
from urllib.error import HTTPError, URLError
from io import BytesIO

# -------------------- Page --------------------
st.set_page_config(page_title="RAP/RUC Visualizer", layout="wide")
st.title("RAP / RUC Weather Visualization")

# -------------------- Constants --------------------
CONUS_EXTENT = [-125, -66.5, 20, 55]

# Era boundaries
RUC2_TO_RUC130 = dt.datetime(2008, 10, 29, 23, tzinfo=dt.UTC)  # ruc2anl_252 (GRIB1) -> ruc2anl_130 (GRIB2)
RUC_TO_RAP     = dt.datetime(2012, 5, 1, 12, tzinfo=dt.UTC)    # RAP replaces RUC

PRODUCTS = [
    "Dewpoint, MSLP, and wind barbs",
    "500 mb wind and height",
    "850 mb wind and height",
    "Surface-based CAPE and wind barbs",
    "Precipitation (convective water)",
    "Total precipitation",  # Added total precip option
]

# -------------------- UI --------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year (UTC)", 1999, dt.datetime.now(dt.UTC).year, 2014)
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
    # your RAP dewpoint palette (°F), preserved 1:1
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

# -------------------- Download & open --------------------
@st.cache_data(show_spinner=False)
def download_grib(year, month, day, hour):
    when = dt.datetime(int(year), int(month), int(day), int(hour), tzinfo=dt.UTC)
    yyyymm = when.strftime("%Y%m")
    yyyymmdd = when.strftime("%Y%m%d")

    def ok(path: str) -> bool:
        try:
            return os.path.getsize(path) > 100_000
        except OSError:
            return False

    if when < RUC2_TO_RUC130:
        fn_templates = [
            f"ruc2anl_252_{when.strftime('%Y%m%d_%H%M')}_000.grb",
            f"ruc2anl_252_{when.strftime('%Y%m%d_%H%S')}_000.grb",
        ]
        roots = [f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{yyyymm}/{yyyymmdd}/"]
    elif when < RUC_TO_RAP:
        fn_templates = [
            f"ruc2anl_130_{when.strftime('%Y%m%d_%H%M')}_000.grb2",
            f"ruc2anl_130_{when.strftime('%Y%m%d_%H%S')}_000.grb2",
        ]
        roots = [f"https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{yyyymm}/{yyyymmdd}/"]
    else:
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
                urllib.request.urlretrieve(url, local)
                if ok(local):
                    return local
            except (HTTPError, URLError, TimeoutError) as e:
                last_err = e
                continue
    raise RuntimeError(f"Could not locate GRIB for {when:%Y-%m-%d %H:00} UTC. Last error: {last_err}")

def open_grbs(path):  # fresh handle per plot
    return pygrib.open(path)

# -------------------- Era helpers --------------------
def is_pre_oct_2008(when):
    return when < RUC2_TO_RUC130

# -------------------- Field finders (FIXED to get exact variables) --------------------
def get_mslp(grbs, when):
    if is_pre_oct_2008(when):
        # EXACT: 223: MSLP (MAPS System Reduction) | Type meanSea | Level: 0 | Units: Pa
        try:
            return grbs.select(name="MSLP (MAPS System Reduction)", typeOfLevel="meanSea")[0]
        except Exception:
            raise KeyError("Pre-2008 MSLP (MAPS System Reduction) not found.")
    else:
        # Post-2008: common variants
        for tries in (
            {"shortName": "mslet"},
            {"shortName": "prmsl"},
            {"shortName": "msl"},
            {"name": "Mean sea level pressure"},
            {"name": "MSLP (ETA model reduction)"},
        ):
            try:
                return grbs.select(**tries)[0]
            except Exception:
                continue
        raise KeyError("Post-2008 MSLP not found.")

def get_sbcape(grbs, when):
    if is_pre_oct_2008(when):
        # EXACT: 234: Convective available potential energy | Type: surface | Level: 0 | Units: J kg**-1
        try:
            return grbs.select(name="Convective available potential energy", typeOfLevel="surface")[0]
        except Exception:
            raise KeyError("Pre-2008 SBCAPE (surface) not found.")
    else:
        # Post-2008: normal SBCAPE encodings
        for tries in (
            {"name": "Convective available potential energy", "typeOfLevel": "surface"},
            {"shortName": "cape", "typeOfLevel": "surface"},
            {"name": "Convective available potential energy", "typeOfLevel": "pressureFromGroundLayer", "level": 25500},
            {"name": "Convective available potential energy", "typeOfLevel": "pressureFromGroundLayer", "level": 9000},
            {"name": "Convective available potential energy", "typeOfLevel": "pressureFromGroundLayer", "level": 18000},
        ):
            try:
                return grbs.select(**tries)[0]
            except Exception:
                continue
        raise KeyError("Post-2008 SBCAPE not found.")

def get_conv_precip(grbs, when):
    # EXACT specifications you provided:
    # PRE 10/2008: 239: Convective precipitation (water) | Type: surface | Level: 0 | Units: kg m**-2
    # POST 10/2008 (2025): 252: Convective precipitation (water) | Type: surface | Level: 0 | Units: kg m**-2
    # (2014): 238: Convective precipitation (water) | Type: surface | Level: 0 | Units: kg m**-2
    
    try:
        return grbs.select(name="Convective precipitation (water)", typeOfLevel="surface")[0]
    except Exception:
        # Fallback attempts
        for tries in (
            {"shortName": "cp", "typeOfLevel": "surface"},
            {"name": "Convective precipitation", "typeOfLevel": "surface"},
        ):
            try:
                return grbs.select(**tries)[0]
            except Exception:
                continue
        raise KeyError("Convective precipitation (water) at surface not found.")

def get_total_precip(grbs, when):
    # Try to get total precipitation (convective + stratiform)
    for tries in (
        {"name": "Total precipitation", "typeOfLevel": "surface"},
        {"shortName": "tp", "typeOfLevel": "surface"},
        {"name": "Precipitation rate", "typeOfLevel": "surface"},
        {"shortName": "prate", "typeOfLevel": "surface"},
        {"name": "Large scale precipitation", "typeOfLevel": "surface"},
        {"shortName": "lsp", "typeOfLevel": "surface"},
    ):
        try:
            return grbs.select(**tries)[0]
        except Exception:
            continue
    raise KeyError("Total precipitation not found.")

# -------------------- Debugging function --------------------
def list_precip_variables(grbs):
    """List all precipitation-related variables in the GRIB file for debugging"""
    precip_vars = []
    for msg in grbs:
        name = getattr(msg, 'name', '')
        shortName = getattr(msg, 'shortName', '')
        typeOfLevel = getattr(msg, 'typeOfLevel', '')
        level = getattr(msg, 'level', 0)
        if any(term in name.lower() for term in ['precip', 'rain', 'convective']) or \
           any(term in shortName.lower() for term in ['tp', 'cp', 'prate', 'lsp']):
            precip_vars.append(f"{name} | {shortName} | {typeOfLevel} | {level}")
    return precip_vars

# -------------------- Base map + finishing --------------------
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
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.07)
    fig.text(0.5, 0.015, "Plotted by Sekai Chandra (@Sekai_WX)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, dpi=240, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------- Products --------------------
def plot_dew_mslp_barbs(grbs, when):
    # 2m dewpoint and 10m wind are present in both eras with these labels
    d2m = grbs.select(name="2 metre dewpoint temperature")[0]
    u10 = grbs.select(name="10 metre U wind component")[0]
    v10 = grbs.select(name="10 metre V wind component")[0]
    mslp = get_mslp(grbs, when)

    dewF = (np.array(d2m.values) - 273.15) * 9/5 + 32.0
    u = np.array(u10.values); v = np.array(v10.values)
    pmb = np.array(mslp.values) / 100.0
    lats, lons = d2m.latlons()

    cmap, norm = dewpoint_cmap()

    fig = plt.figure(figsize=(13.0, 8.6))
    ax = base_ax()

    cf = ax.contourf(
        lons, lats, dewF,
        levels=np.linspace(-40, 90, cmap.N+1),
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), extend="both"
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, aspect=28)
    cbar.set_label("Dewpoint (°F)")

    cs = ax.contour(lons, lats, pmb, levels=np.arange(900, 1100, 2),
                    colors="black", linewidths=0.8, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)

    ax.barbs(lons[::10, ::10], lats[::10, ::10], u[::10, ::10], v[::10, ::10],
             length=6, transform=ccrs.PlateCarree())

    title = f"Dewpoint (°F), MSLP (mb), and 10 m wind — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_wind_height(grbs, level_mb, is_500, when):
    u = grbs.select(name="U component of wind", level=level_mb)[0]
    v = grbs.select(name="V component of wind", level=level_mb)[0]
    z = grbs.select(name="Geopotential height", level=level_mb)[0]

    uu = u.values; vv = v.values
    zvals = z.values
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

    fig = plt.figure(figsize=(13.0, 8.6))
    ax = base_ax()

    cf = ax.contourf(lons, lats, wspd_kt, levels=levels, cmap=cmap, norm=bnorm, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, aspect=28)
    cbar.set_label("Wind Speed (kt)")

    cs = ax.contour(lons, lats, zvals/10.0, levels=h_levels, colors="black", linewidths=1.0, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")

    ax.barbs(lons[::10, ::10], lats[::10, ::10], uu[::10, ::10], vv[::10, ::10], length=6, transform=ccrs.PlateCarree())

    title = f"{name} — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_sbcape_barbs(grbs, when):
    cape_grb = get_sbcape(grbs, when)
    u10 = grbs.select(name="10 metre U wind component")[0]
    v10 = grbs.select(name="10 metre V wind component")[0]

    cape = np.clip(np.array(cape_grb.values, dtype=float), 0, 7000)
    lats, lons = cape_grb.latlons()
    u = np.array(u10.values); v = np.array(v10.values)

    fig = plt.figure(figsize=(13.0, 8.6))
    ax = base_ax()

    turbo = plt.get_cmap("turbo")
    cf = ax.contourf(lons, lats, cape, levels=np.linspace(0,7000,71), cmap=turbo,
                     transform=ccrs.PlateCarree(), extend="max")
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, aspect=28)
    cbar.set_label("SBCAPE (J/kg) — capped at 7000")

    ax.barbs(lons[::10, ::10], lats[::10, ::10], u[::10, ::10], v[::10, ::10], length=6, transform=ccrs.PlateCarree())

    title = f"Surface-based CAPE (J/kg) & 10 m wind — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_conv_precip(grbs, when):
    try:
        cp = get_conv_precip(grbs, when)
        data = np.array(cp.values, dtype=float)  # kg m^-2 == mm accumulation
        lats, lons = cp.latlons()

        # Mask out very small values to avoid plotting noise
        data = np.where(data < 0.01, np.nan, data)
        
        # Get statistics for better scaling
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            st.warning("No valid precipitation data found in this file.")
            vmax = 25.0  # Default scale
        else:
            data_max = np.nanmax(valid_data)
            data_99th = np.nanpercentile(valid_data, 99)
            vmax = max(10.0, min(data_99th * 1.2, data_max))

        levels = np.linspace(0, vmax, 51)
        cm = plt.get_cmap("turbo")

        fig = plt.figure(figsize=(13.0, 8.6))
        ax = base_ax()
        
        # Only plot where data > 0.01 mm
        cf = ax.contourf(lons, lats, data, levels=levels, cmap=cm,
                         transform=ccrs.PlateCarree(), extend="max")
        cbar = fig.colorbar(cf, ax=ax, pad=0.01, aspect=28)
        cbar.set_label("Convective precipitation (mm)")

        title = f"Convective Precipitation — {when:%B %d, %Y %H:00 UTC}"
        return finish_and_return(fig, title)
    
    except KeyError as e:
        st.error(f"Could not find convective precipitation variable: {e}")
        # Show available precipitation variables for debugging
        precip_vars = list_precip_variables(grbs)
        if precip_vars:
            st.info("Available precipitation-related variables:")
            for var in precip_vars:
                st.text(var)
        raise

def plot_total_precip(grbs, when):
    try:
        tp = get_total_precip(grbs, when)
        data = np.array(tp.values, dtype=float)  # kg m^-2 == mm accumulation
        lats, lons = tp.latlons()

        # Mask out very small values
        data = np.where(data < 0.01, np.nan, data)
        
        # Get statistics for better scaling
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            st.warning("No valid total precipitation data found in this file.")
            vmax = 25.0
        else:
            data_max = np.nanmax(valid_data)
            data_99th = np.nanpercentile(valid_data, 99)
            vmax = max(10.0, min(data_99th * 1.2, data_max))

        levels = np.linspace(0, vmax, 51)
        cm = plt.get_cmap("Blues")  # Use Blues for total precip

        fig = plt.figure(figsize=(13.0, 8.6))
        ax = base_ax()
        
        cf = ax.contourf(lons, lats, data, levels=levels, cmap=cm,
                         transform=ccrs.PlateCarree(), extend="max")
        cbar = fig.colorbar(cf, ax=ax, pad=0.01, aspect=28)
        cbar.set_label("Total precipitation (mm)")

        title = f"Total Precipitation — {when:%B %d, %Y %H:00 UTC}"
        return finish_and_return(fig, title)
    
    except KeyError as e:
        st.error(f"Could not find total precipitation variable: {e}")
        # Show available precipitation variables for debugging
        precip_vars = list_precip_variables(grbs)
        if precip_vars:
            st.info("Available precipitation-related variables:")
            for var in precip_vars:
                st.text(var)
        raise

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
            elif product == "Precipitation (convective water)":
                buf = plot_conv_precip(grbs, when)
            elif product == "Total precipitation":
                buf = plot_total_precip(grbs, when)
            else:
                st.error("Unknown product.")
                st.stop()

        st.success("Visualization generated.")
        st.image(buf, caption=f"{product} — {when:%Y-%m-%d %H:00 UTC}", use_column_width=True)
        st.download_button(
            "Download PNG",
            data=buf,
            file_name=f"RAP_RUC_{product.replace(' ','_').replace('(','').replace(')','').replace(',','_')}_{when:%Y%m%d%H}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try a different date/time, or check if the GRIB file exists for this period.")
    finally:
        # keep cached file on disk; handles are closed above
        pass
# --- robust helpers -----------------------------------------------------------
def select_try(grbs, tries):
    """Return first pygrib message that matches any of the attempt dicts."""
    for t in tries:
        try:
            g = grbs.select(**t)[0]
            return g
        except Exception:
            continue
    return None

def find_mslp(grbs):
    """MSLP across eras (GRIB1/2)."""
    # 1) explicit names/shortNames that appear across inventories
    tries = [
        {"name": "MSLP (MAPS System Reduction)"},
        {"name": "MSLP (ETA model reduction)"},
        {"name": "Mean sea level pressure"},
        {"name": "MSL Pressure reduced to MSL"},
        {"shortName": "mslet"},
        {"shortName": "prmsl"},
        {"shortName": "msl"},
        {"typeOfLevel": "meanSea"}  # generic
    ]
    g = select_try(grbs, tries)
    if g is not None:
        return g

    # 2) absolute fallback: first message that *looks* like MSLP
    for msg in grbs:
        if getattr(msg, "typeOfLevel", "") == "meanSea":
            return msg
    raise KeyError("MSLP not found by name or meanSea scan")

def find_sbcape(grbs):
    """SBCAPE across eras."""
    tries = [
        {"name": "Convective available potential energy", "typeOfLevel": "surface"},
        {"shortName": "cape", "typeOfLevel": "surface"},
        {"name": "Convective available potential energy",
         "typeOfLevel": "pressureFromGroundLayer", "level": 25500},
        {"name": "Convective available potential energy",
         "typeOfLevel": "pressureFromGroundLayer", "level": 9000},
        {"name": "Convective available potential energy",
         "typeOfLevel": "pressureFromGroundLayer", "level": 18000},
    ]
    g = select_try(grbs, tries)
    if g is not None:
        return g

    # substring fallback for GRIB1 oddities
    for msg in grbs:
        if (getattr(msg, "typeOfLevel", "") in ("surface", "pressureFromGroundLayer")
            and "available potential energy" in getattr(msg, "name", "").lower()):
            return msg
    raise KeyError("Could not locate SBCAPE in this file.")

def extract_precip_mm(grbs):
    """
    Returns (field_mm, label, is_rate) choosing the best available precip:
    1) Total Precipitation (kg m^-2 == mm)
    2) CP + LSP (both kg m^-2)
    3) Precipitation rate * 3600 (mm h^-1)
    """
    # Total precipitation (2020s RAP commonly provides it)
    total = select_try(grbs, [
        {"name": "Total Precipitation", "typeOfLevel": "surface"},
        {"shortName": "tp", "typeOfLevel": "surface"},
    ])
    if total is not None:
        data = np.array(total.values, dtype=float)
        if np.nanmax(data) > 0.05:
            return data, "Total Precip (mm)", False

    # Large-scale + convective accumulations (present in 2007/2014 lists)
    lsp = select_try(grbs, [
        {"name": "Large scale precipitation (non-convective)", "typeOfLevel": "surface"},
        {"shortName": "lsp", "typeOfLevel": "surface"},
    ])
    cp = select_try(grbs, [
        {"name": "Convective precipitation (water)", "typeOfLevel": "surface"},
        {"shortName": "cp", "typeOfLevel": "surface"},
    ])
    if (lsp is not None) or (cp is not None):
        lsp_mm = np.zeros_like(lsp.values if lsp is not None else cp.values, dtype=float)
        cp_mm  = np.zeros_like(lsp_mm)
        if lsp is not None:
            lsp_mm = np.array(lsp.values, dtype=float)
        if cp is not None:
            cp_mm = np.array(cp.values, dtype=float)
        total_mm = lsp_mm + cp_mm
        if np.nanmax(total_mm) > 0.05:
            return total_mm, "Large-scale + Convective Precip (mm)", False

    # Instantaneous precip rate → hourly equivalent
    prate = select_try(grbs, [
        {"name": "Precipitation rate", "typeOfLevel": "surface"},
        {"shortName": "prate", "typeOfLevel": "surface"},
        {"shortName": "cprat", "typeOfLevel": "surface"},  # sometimes convective rate
    ])
    if prate is not None:
        mmph = np.array(prate.values, dtype=float) * 3600.0  # kg m^-2 s^-1 -> mm hr^-1
        if np.nanmax(mmph) > 0.05:
            return mmph, "Precipitation rate (mm h⁻¹)", True

    # Last resort—if we truly have all zeros, return the most informative thing we found
    if total is not None:
        return np.array(total.values, dtype=float), "Total Precip (mm)", False
    if lsp is not None or cp is not None:
        lsp_mm = np.array(lsp.values if lsp is not None else 0.0, dtype=float)
        cp_mm  = np.array(cp.values if cp is not None else 0.0, dtype=float)
        return lsp_mm + cp_mm, "Large-scale + Convective Precip (mm)", False
    if prate is not None:
        return np.array(prate.values, dtype=float) * 3600.0, "Precipitation rate (mm h⁻¹)", True

    raise KeyError("No usable precipitation variable found in this file.")

# --- product renderers (only the pieces that change) --------------------------
def plot_conv_precip(grbs, when):
    field, label, is_rate = extract_precip_mm(grbs)
    lats, lons = grbs[1].latlons()  # any field with same grid; first msg is safe here

    # color map tuned for precip
    cm = plt.get_cmap("turbo")
    vmax = 50 if not is_rate else 100  # mm accum vs mm/hr
    levels = np.linspace(0, vmax, 51)

    fig = plt.figure(figsize=(13.5, 9))
    ax = base_ax()
    cf = ax.contourf(lons, lats, field, levels=levels, cmap=cm,
                     transform=ccrs.PlateCarree(), extend="max")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(label)

    title = f"Precipitation — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

def plot_dew_mslp_barbs(grbs, when):
    d2m = select_try(grbs, [
        {"name": "2 metre dewpoint temperature"},
        {"name": "2 m above ground Dew point temperature"},
        {"shortName": "dpt"},
    ])
    if d2m is None:
        raise KeyError("2 m dewpoint not found")

    u10 = select_try(grbs, [
        {"name": "10 metre U wind component", "level": 10},
        {"name": "U component of wind", "typeOfLevel": "heightAboveGround", "level": 10},
    ])
    v10 = select_try(grbs, [
        {"name": "10 metre V wind component", "level": 10},
        {"name": "V component of wind", "typeOfLevel": "heightAboveGround", "level": 10},
    ])
    mslp = find_mslp(grbs)

    dewF = (np.array(d2m.values) - 273.15) * 9/5 + 32.0
    u = np.array(u10.values); v = np.array(v10.values)
    p = np.array(mslp.values) / 100.0
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

    cs = ax.contour(lons, lats, p, levels=np.arange(900, 1100, 2),
                    colors="black", linewidths=0.8, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8)

    ax.barbs(lons[::10, ::10], lats[::10, ::10], u[::10, ::10], v[::10, ::10],
             length=6, transform=ccrs.PlateCarree())

    title = f"Dewpoint (°F), MSLP (mb), and 10 m wind — {when:%B %d, %Y %H:00 UTC}"
    return finish_and_return(fig, title)

# --- trim whitespace everywhere ------------------------------------------------
def finish_and_return(fig, title):
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    # tighter margins: reduce bottom whitespace without clipping the colorbar
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.07)
    fig.text(0.5, 0.015, "Plotted by Sekai Chandra (@Sekai_WX)",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, dpi=220, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

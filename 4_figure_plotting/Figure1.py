from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import Point
import numpy as np
from pyproj import Transformer
from osgeo import gdal


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def add_north_arrow(ax, x, y, arrow_length=0.08, fontsize=14, lw=2.0):
    """Add a north arrow in axes fraction coordinates."""
    ax.annotate(
        '', xy=(x, y + arrow_length), xytext=(x, y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', lw=lw, color='black')
    )
    ax.text(
        x, y + arrow_length + 0.015, 'N',
        transform=ax.transAxes, fontsize=fontsize, fontweight='bold',
        ha='center', va='bottom'
    )


def add_scale_bar(ax, length_km=100, location=(0.35, 0.03), fontsize=8, bar_frac=0.015):
    """Add a two-segment scale bar in projected coordinates (meters)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    total_bar_m = 2 * length_km * 1000
    x_center = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    x0 = x_center - total_bar_m / 2
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]

    length_m = length_km * 1000
    bar_height = (ylim[1] - ylim[0]) * bar_frac

    ax.add_patch(plt.Rectangle(
        (x0, y0), length_m, bar_height,
        facecolor='white', edgecolor='black', linewidth=1.0, zorder=6
    ))
    ax.add_patch(plt.Rectangle(
        (x0 + length_m, y0), length_m, bar_height,
        facecolor='black', edgecolor='black', linewidth=1.0, zorder=6
    ))

    label_y = y0 - (ylim[1] - ylim[0]) * 0.005
    ax.text(x0, label_y, '0', fontsize=fontsize, ha='center', va='top', zorder=6)
    ax.text(x0 + length_m, label_y, f'{length_km}', fontsize=fontsize, ha='center', va='top', zorder=6)
    ax.text(x0 + 2 * length_m, label_y, f'{2 * length_km} km', fontsize=fontsize, ha='center', va='top', zorder=6)


def add_scale_bar_m(ax, length_m=10, location=(0.50, 0.04), fontsize=8, bar_frac=0.015):
    """Add a scale bar in meters for the detail panel."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_center = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    x0 = x_center - length_m / 2
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    bar_height = (ylim[1] - ylim[0]) * bar_frac

    ax.add_patch(plt.Rectangle(
        (x0, y0), length_m, bar_height,
        facecolor='white', edgecolor='black', linewidth=1.0, zorder=6
    ))
    ax.add_patch(plt.Rectangle(
        (x0 + length_m, y0), length_m, bar_height,
        facecolor='black', edgecolor='black', linewidth=1.0, zorder=6
    ))

    label_y = y0 - (ylim[1] - ylim[0]) * 0.005
    ax.text(x0, label_y, '0', fontsize=fontsize, ha='center', va='top', zorder=6)
    ax.text(x0 + length_m, label_y, f'{length_m}', fontsize=fontsize, ha='center', va='top', zorder=6)
    ax.text(x0 + 2 * length_m, label_y, f'{2 * length_m} m', fontsize=fontsize, ha='center', va='top', zorder=6)


def add_lat_lon_grid(ax, crs, interval=5, fontsize=9):
    """Add lat/lon grid lines and labels on a projected-CRS axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    transformer_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    transformer_to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    lon_min, lat_min = transformer_to_ll.transform(xlim[0], ylim[0])
    lon_max, lat_max = transformer_to_ll.transform(xlim[1], ylim[1])

    lon_ticks = np.arange(np.floor(lon_min / interval) * interval,
                          np.ceil(lon_max / interval) * interval + 1, interval)
    lat_ticks = np.arange(np.floor(lat_min / interval) * interval,
                          np.ceil(lat_max / interval) * interval + 1, interval)

    for lon in lon_ticks:
        lats = np.linspace(lat_min - 2, lat_max + 2, 200)
        xs, ys = transformer_to_proj.transform(np.full_like(lats, lon), lats)
        ax.plot(xs, ys, color='gray', linewidth=0.4, linestyle='--', zorder=1)

    for lat in lat_ticks:
        lons = np.linspace(lon_min - 2, lon_max + 2, 200)
        xs, ys = transformer_to_proj.transform(lons, np.full_like(lons, lat))
        ax.plot(xs, ys, color='gray', linewidth=0.4, linestyle='--', zorder=1)

    x_pad = (xlim[1] - xlim[0]) * 0.02
    y_pad = (ylim[1] - ylim[0]) * 0.02

    for lon in lon_ticks:
        x, _ = transformer_to_proj.transform(lon, lat_min)
        if xlim[0] <= x <= xlim[1]:
            ax.text(x, ylim[0] - y_pad, f'{abs(lon):.0f}\u00b0W',
                    fontsize=fontsize, ha='center', va='top')

    for lat in lat_ticks:
        _, y = transformer_to_proj.transform(lon_min, lat)
        if ylim[0] <= y <= ylim[1]:
            ax.text(xlim[0] - x_pad, y, f'{lat:.0f}\u00b0N',
                    fontsize=fontsize, ha='center', va='center', rotation=90)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def read_rgb_with_gdal(tif_path):
    """Read a 3-band GeoTIFF as an RGB numpy array + extent using GDAL."""
    ds = gdal.Open(str(tif_path))
    gt = ds.GetGeoTransform()
    xmin = gt[0]
    xmax = gt[0] + gt[1] * ds.RasterXSize
    ymax = gt[3]
    ymin = gt[3] + gt[5] * ds.RasterYSize
    extent = [xmin, xmax, ymin, ymax]

    r = ds.GetRasterBand(1).ReadAsArray()
    g = ds.GetRasterBand(2).ReadAsArray()
    b = ds.GetRasterBand(3).ReadAsArray()
    ds = None

    rgb = np.dstack([r, g, b])
    return rgb, extent


def style_spines(ax):
    """Show black spines, hide ticks."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def add_panel_label(ax, label, color='black'):
    """Add panel label (A. B. C. D.) inside upper-left with white background."""
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left', zorder=10,
            color=color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', linewidth=0.5, alpha=0.9))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "1_research_area" / "Boundaries" / "StudyArea"
    plot_dir = root / "1_research_area" / "Boundaries"
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data for panel (a): Study area map ---
    gdf_states = gpd.read_file(data_dir / "States_CO_Plateau.shp")
    gdf_co = gpd.read_file(data_dir / "CO_Plateau.shp")
    if gdf_states.crs != gdf_co.crs:
        gdf_states = gdf_states.to_crs(gdf_co.crs)

    research_point = gpd.GeoDataFrame(
        geometry=[Point(-109.42, 38.67)], crs='EPSG:4326'
    ).to_crs(gdf_co.crs)

    # --- Data for panel (b): RGB ortho with all plot boundaries ---
    tif_path = root / "1_research_area" / "Biocrust_Flight2_021222_RGB_Ortho_Metashape_utm83.tif"
    rgb, extent = read_rgb_with_gdal(tif_path)

    gdf_bplots = gpd.read_file(plot_dir / "BPlots_utm83.shp")

    treatments = ['Control', 'AlteredP', 'Warmed', 'WarmAltP']
    treatment_labels = {
        'Control': 'Control', 'AlteredP': 'AltP',
        'Warmed': 'Warmed', 'WarmAltP': 'AltP + Warmed',
    }
    treatment_colors = {
        'Control':  '#2ca02c',
        'AlteredP': '#1f77b4',
        'Warmed':   '#d62728',
        'WarmAltP': '#ff7f0e',
    }
    treatment_linestyles = {
        'Control': '-', 'AlteredP': '--',
        'Warmed': ':', 'WarmAltP': '-.',
    }
    blocks = ['B1', 'B2', 'B3', 'B4', 'B5']

    plot_gdfs = {}
    for block in blocks:
        for treat in treatments:
            shp = plot_dir / f'BPlots_{block}_{treat}.shp'
            if shp.exists():
                plot_gdfs[(block, treat)] = gpd.read_file(shp)

    # -----------------------------------------------------------------------
    # Layout: top row: (a) left, (c) center, (d) right
    #         bottom row: (b) spanning full width
    # -----------------------------------------------------------------------
    fig_w, fig_h = 10, 8
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Data aspect ratios (width/height): A=0.974, C=1.629, D=0.783
    ar_a, ar_c, ar_d = 0.974, 1.629, 0.783

    gap = 0.012
    margin_l = 0.02
    margin_r = 0.01

    # Compute widths for equal-height panels
    avail = 1.0 - margin_l - margin_r - 2 * gap
    sum_ar = ar_a + ar_c + ar_d
    # Each width = (ar / sum_ar) * avail
    wa = ar_a / sum_ar * avail
    wc = ar_c / sum_ar * avail
    wd = ar_d / sum_ar * avail
    # Actual height from width: top_h = wa / ar_a * fig_w / fig_h
    top_h_actual = wa / ar_a * fig_w / fig_h

    top_y = 1.0 - top_h_actual - 0.01
    bot_h = top_y - 0.04
    bot_y = 0.02

    x_a = margin_l
    x_c = x_a + wa + gap
    x_d = x_c + wc + gap

    ax_a = fig.add_axes([x_a, top_y, wa, top_h_actual])
    ax_c = fig.add_axes([x_c, top_y, wc, top_h_actual])
    ax_d = fig.add_axes([x_d, top_y, wd, top_h_actual])
    ax_b = fig.add_axes([margin_l, bot_y, 1.0 - margin_l - margin_r, bot_h])

    # ===================================================================
    # Panel A: Study area map
    # ===================================================================
    gdf_states.plot(ax=ax_a, color='#9eaab2', edgecolor='white', linewidth=0.8)
    gdf_co.plot(ax=ax_a, color='#c8bb8e', edgecolor='#555555', linewidth=0.8)
    research_point.plot(ax=ax_a, color='red', marker='*', markersize=200,
                        edgecolor='darkred', linewidth=0.5, zorder=5)
    ax_a.annotate(
        'Castle Valley',
        xy=(research_point.geometry.x.iloc[0], research_point.geometry.y.iloc[0]),
        xytext=(8, -8), textcoords='offset points',
        fontsize=10, fontweight='bold', color='black', zorder=6
    )

    style_spines(ax_a)
    add_lat_lon_grid(ax_a, gdf_co.crs, interval=5, fontsize=9)
    add_north_arrow(ax_a, x=0.90, y=0.80, arrow_length=0.12, fontsize=12, lw=2.0)
    add_scale_bar(ax_a, length_km=200, location=(0.75, 0.05), fontsize=8, bar_frac=0.018)
    add_panel_label(ax_a, 'a.')

    # Legend for states and CO Plateau (bottom-left)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_a = [
        Patch(facecolor='#c8bb8e', edgecolor='#555555', linewidth=0.8, label='Colorado Plateau'),
        Patch(facecolor='#9eaab2', edgecolor='#6b6b6b', linewidth=0.8, label='States'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=10, label='Castle Valley'),
    ]
    ax_a.legend(handles=legend_a, loc='lower left', fontsize=7,
                frameon=True, framealpha=0.9, edgecolor='none')

    # ===================================================================
    # Panel B: All plots overview
    # ===================================================================
    ax_b.imshow(rgb, extent=extent, origin='upper', aspect='auto')

    # Overall boundary — thick black
    gdf_bplots.boundary.plot(ax=ax_b, color='black', linewidth=4, zorder=2)

    # Individual plot boundaries — thick
    for (block, treat), gdf in plot_gdfs.items():
        color = treatment_colors[treat]
        ls = treatment_linestyles[treat]
        for _, row in gdf.iterrows():
            xs, ys = row.geometry.exterior.xy
            ax_b.plot(xs, ys, color=color, linestyle=ls,
                      linewidth=3, zorder=3)

    # Zoom extent
    bnd = gdf_bplots.total_bounds
    pad_x = (bnd[2] - bnd[0]) * 0.01
    pad_y = (bnd[3] - bnd[1]) * 0.02
    ax_b.set_xlim(bnd[0] - pad_x, bnd[2] + pad_x)
    ax_b.set_ylim(bnd[1] - pad_y, bnd[3] + pad_y)

    # Labels below each plot — much larger
    for (block, treat), gdf in plot_gdfs.items():
        color = treatment_colors[treat]
        cx = gdf.geometry.centroid.x.values[0]
        b = gdf.total_bounds

        label_text = f'{block} ({treatment_labels[treat]})'
        ax_b.annotate(
            label_text,
            xy=(cx, b[1]),
            xytext=(0, -5), textcoords='offset points',
            fontsize=8, fontweight='bold', color='white',
            ha='center', va='top', zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                      alpha=0.8, edgecolor='none'),
            arrowprops=dict(arrowstyle='-', color=color,
                            lw=0.5, shrinkA=0, shrinkB=1)
        )

    style_spines(ax_b)
    add_scale_bar_m(ax_b, length_m=5, location=(0.45, 0.06), fontsize=14, bar_frac=0.020)
    add_north_arrow(ax_b, x=0.93, y=0.80, arrow_length=0.12, fontsize=15, lw=2.5)
    add_panel_label(ax_b, 'b.')

    # Legend — larger, bottom right
    legend_handles = []
    for t in treatments:
        legend_handles.append(FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle='square,pad=0',
            facecolor='none', edgecolor=treatment_colors[t],
            linestyle=treatment_linestyles[t], linewidth=2.5,
            label=treatment_labels[t]
        ))
    ax_b.legend(handles=legend_handles, loc='lower right', fontsize=12,
                frameon=True, framealpha=0.9, edgecolor='none',
                handlelength=2.5, handleheight=1.8)

    # Yellow box indicating B1 zoom area — larger with more padding
    b1_bounds_all = np.array([
        plot_gdfs[('B1', t)].total_bounds for t in treatments
        if ('B1', t) in plot_gdfs
    ])
    b1_xmin = b1_bounds_all[:, 0].min()
    b1_ymin = b1_bounds_all[:, 1].min()
    b1_xmax = b1_bounds_all[:, 2].max()
    b1_ymax = b1_bounds_all[:, 3].max()
    b1_pad = 1.5
    rect = plt.Rectangle(
        (b1_xmin - b1_pad, b1_ymin - b1_pad),
        (b1_xmax - b1_xmin) + 1.5 * b1_pad,
        (b1_ymax - b1_ymin) + 1.5 * b1_pad,
        linewidth=3, edgecolor='yellow', facecolor='none',
        linestyle='-', zorder=4
    )
    ax_b.add_patch(rect)

    # ===================================================================
    # Panel C: Zoomed B1
    # ===================================================================
    ax_c.imshow(rgb, extent=extent, origin='upper', aspect='auto')

    # Overlay B1 plot boundaries — thick
    for treat in treatments:
        key = ('B1', treat)
        if key in plot_gdfs:
            gdf = plot_gdfs[key]
            color = treatment_colors[treat]
            ls = treatment_linestyles[treat]
            for _, row in gdf.iterrows():
                xs, ys = row.geometry.exterior.xy
                ax_c.plot(xs, ys, color=color, linestyle=ls,
                          linewidth=4, zorder=3)

            # Labels: AltP on right, others on top
            cx = gdf.geometry.centroid.x.values[0]
            b = gdf.total_bounds
            label_text = f'B1 ({treatment_labels[treat]})'

            if treat == 'AlteredP':
                # Put label to the right of the plot
                ax_c.annotate(
                    label_text,
                    xy=(b[2], (b[1] + b[3]) / 2),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='white',
                    ha='left', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                              alpha=0.85, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.8, shrinkA=0, shrinkB=1)
                )
            else:
                # Put label on top of the plot
                ax_c.annotate(
                    label_text,
                    xy=(cx, b[3]),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='white',
                    ha='center', va='bottom', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                              alpha=0.85, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', color=color,
                                    lw=0.8, shrinkA=0, shrinkB=1)
                )

    # Zoom to B1 area
    zoom_pad = 0.3
    ax_c.set_xlim(b1_xmin - zoom_pad, b1_xmax + zoom_pad)
    ax_c.set_ylim(b1_ymin - zoom_pad * 0.5, b1_ymax + zoom_pad)
    style_spines(ax_c)
    add_scale_bar_m(ax_c, length_m=1, location=(0.80, 0.08), fontsize=11, bar_frac=0.020)
    add_north_arrow(ax_c, x=0.92, y=0.80, arrow_length=0.12, fontsize=12, lw=2.0)
    add_panel_label(ax_c, 'c.')

    # ===================================================================
    # Panel D: Schematic of 15 subplots within one plot (rotated view)
    # ===================================================================
    from scipy.ndimage import rotate as ndrotate

    gdf_ex = plot_gdfs[('B1', 'Warmed')]
    geom_ex = gdf_ex.geometry.iloc[0]

    mrr = geom_ex.minimum_rotated_rectangle
    mrr_coords = np.array(mrr.exterior.coords[:-1])

    edge_lengths = []
    for i in range(4):
        dx = mrr_coords[(i+1) % 4][0] - mrr_coords[i][0]
        dy = mrr_coords[(i+1) % 4][1] - mrr_coords[i][1]
        edge_lengths.append(np.sqrt(dx**2 + dy**2))
    long_idx = np.argmax(edge_lengths[:2])
    dx_long = mrr_coords[(long_idx+1) % 4][0] - mrr_coords[long_idx][0]
    dy_long = mrr_coords[(long_idx+1) % 4][1] - mrr_coords[long_idx][1]
    plot_angle = np.degrees(np.arctan2(dy_long, dx_long))
    rot_angle = 90.0 - plot_angle

    ex_bounds = gdf_ex.total_bounds
    pad = 1.0
    crop_xmin = ex_bounds[0] - pad
    crop_xmax = ex_bounds[2] + pad
    crop_ymin = ex_bounds[1] - pad
    crop_ymax = ex_bounds[3] + pad

    ds = gdal.Open(str(tif_path))
    gt = ds.GetGeoTransform()
    col_min = max(0, int((crop_xmin - gt[0]) / gt[1]))
    col_max = min(ds.RasterXSize, int((crop_xmax - gt[0]) / gt[1]) + 1)
    row_min = max(0, int((crop_ymax - gt[3]) / gt[5]))
    row_max = min(ds.RasterYSize, int((crop_ymin - gt[3]) / gt[5]) + 1)
    crop_w = col_max - col_min
    crop_h = row_max - row_min
    crop_r = ds.GetRasterBand(1).ReadAsArray(col_min, row_min, crop_w, crop_h)
    crop_g = ds.GetRasterBand(2).ReadAsArray(col_min, row_min, crop_w, crop_h)
    crop_b = ds.GetRasterBand(3).ReadAsArray(col_min, row_min, crop_w, crop_h)
    ds = None
    crop_rgb = np.dstack([crop_r, crop_g, crop_b])

    rotated = ndrotate(crop_rgb, rot_angle, reshape=True, order=1, mode='constant', cval=0)
    ax_d.imshow(rotated, origin='upper', aspect='auto')

    img_h, img_w = rotated.shape[:2]

    def geo_to_rotated_pixel(gx, gy):
        px = (gx - (gt[0] + col_min * gt[1])) / gt[1]
        py = (gy - (gt[3] + row_min * gt[5])) / gt[5]
        cx_p = crop_w / 2
        cy_p = crop_h / 2
        angle_rad = np.radians(rot_angle)
        dx_p = px - cx_p
        dy_p = py - cy_p
        rx = dx_p * np.cos(angle_rad) - dy_p * np.sin(angle_rad)
        ry = dx_p * np.sin(angle_rad) + dy_p * np.cos(angle_rad)
        rx += img_w / 2
        ry += img_h / 2
        return rx, ry

    mrr_px = np.array([geo_to_rotated_pixel(c[0], c[1]) for c in mrr_coords])

    center = mrr_px.mean(axis=0)
    top = mrr_px[mrr_px[:, 1] < center[1]]
    bottom = mrr_px[mrr_px[:, 1] >= center[1]]
    top = top[top[:, 0].argsort()]
    bottom = bottom[bottom[:, 0].argsort()]
    tl, tr = top[0], top[1]
    bl, br = bottom[0], bottom[1]

    v_right = tr - tl
    v_down = bl - tl
    # Shrink the grid and shift upward to avoid circles at bottom
    inset_lr = 0.20   # left-right inset
    inset_top = 0.03  # top inset (smaller to include vegetation)
    inset_bot = 0.28  # bottom inset (larger to avoid circles)
    origin = tl + inset_lr * v_right + inset_top * v_down
    v_r = v_right * (1 - 2 * inset_lr)
    v_d = v_down * (1 - inset_top - inset_bot)

    from matplotlib.patches import Polygon as MplPolygon

    # Colors: 1-12 have both spectra + fractional cover, 13-15 have spectra only
    color_both = '#6baed6'     # blue — spectra + fractional cover
    color_spectra = '#fdae6b'  # orange — spectra only

    nrows, ncols = 5, 3
    subplot_num = 1
    for r in range(nrows):
        for c in range(ncols):
            p0 = origin + (c / ncols) * v_r + (r / nrows) * v_d
            p1 = origin + ((c + 1) / ncols) * v_r + (r / nrows) * v_d
            p2 = origin + ((c + 1) / ncols) * v_r + ((r + 1) / nrows) * v_d
            p3 = origin + (c / ncols) * v_r + ((r + 1) / nrows) * v_d

            if subplot_num <= 12:
                fc = color_both
            else:
                fc = color_spectra

            cell_poly = MplPolygon(
                [p0, p1, p2, p3],
                closed=True, facecolor=fc, edgecolor='black',
                alpha=0.75, linewidth=0.8, zorder=4
            )
            ax_d.add_patch(cell_poly)

            pc = (p0 + p2) / 2
            ax_d.text(
                pc[0], pc[1], str(subplot_num),
                fontsize=10, fontweight='bold', color='black',
                ha='center', va='center', zorder=5
            )
            subplot_num += 1

    # Cyan box aligned with grid edges
    g_tl = origin
    g_tr = origin + v_r
    g_br = origin + v_r + v_d
    g_bl = origin + v_d
    grid_corners = np.array([g_tl, g_tr, g_br, g_bl, g_tl])
    ax_d.plot(grid_corners[:, 0], grid_corners[:, 1],
              color='cyan', linewidth=2.0, zorder=3)

    all_px = grid_corners[:, 0]
    all_py = grid_corners[:, 1]
    px_pad = (all_px.max() - all_px.min()) * 0.10
    py_pad = (all_py.max() - all_py.min()) * 0.15
    ax_d.set_xlim(all_px.min() - px_pad, all_px.max() + px_pad)
    ax_d.set_ylim(all_py.max() + py_pad, all_py.min() - py_pad)
    style_spines(ax_d)
    add_panel_label(ax_d, 'd.')

    # Legend for subplot measurement types
    from matplotlib.patches import Patch as LegPatch
    legend_d = [
        LegPatch(facecolor=color_both, edgecolor='black', linewidth=0.8,
                 alpha=0.75, label='Spectra + Fractional cover'),
        LegPatch(facecolor=color_spectra, edgecolor='black', linewidth=0.8,
                 alpha=0.75, label='Spectra only'),
    ]
    ax_d.legend(handles=legend_d, loc='lower center', fontsize=7,
                frameon=True, framealpha=0.9, edgecolor='none',
                ncol=1)

    # ===================================================================
    # Cross-panel arrows using figure coordinates
    # ===================================================================
    from matplotlib.patches import FancyArrowPatch

    # Arrow from red star in Panel A to center of Panel B
    star_x_data = research_point.geometry.x.iloc[0]
    star_y_data = research_point.geometry.y.iloc[0]
    star_fig = ax_a.transData.transform((star_x_data, star_y_data))
    star_fig = fig.transFigure.inverted().transform(star_fig)

    # Target: top-left area of Panel B (near the black boundary)
    # Use the top-left corner of the BPlots boundary in figure coords
    # Shift target lower and right from the top-left corner
    bnd_topleft_data = (bnd[0] + (bnd[2] - bnd[0]) * 0.15,
                        bnd[3] - (bnd[3] - bnd[1]) * 0.15)
    bnd_topleft_fig = ax_b.transData.transform(bnd_topleft_data)
    target_b = fig.transFigure.inverted().transform(bnd_topleft_fig)

    arrow1 = FancyArrowPatch(
        star_fig, target_b,
        transform=fig.transFigure,
        arrowstyle='->', mutation_scale=15,
        color='red', linewidth=3, zorder=10,
        clip_on=False
    )
    fig.patches.append(arrow1)

    # Arrow from top-center of yellow box in Panel B to Panel C
    yellow_top_center_data = ((b1_xmin - b1_pad + b1_xmax + 1.5 * b1_pad * 0.5) / 2,
                               b1_ymax + 1.5 * b1_pad * 0.5)
    yellow_fig = ax_b.transData.transform(yellow_top_center_data)
    yellow_fig = fig.transFigure.inverted().transform(yellow_fig)

    # Target: bottom-center of Panel C
    c_bbox = ax_c.get_position()
    target_c = (c_bbox.x0 + c_bbox.width * 0.5, c_bbox.y0)

    arrow2 = FancyArrowPatch(
        yellow_fig, target_c,
        transform=fig.transFigure,
        arrowstyle='->', mutation_scale=15,
        color='cyan', linewidth=3, zorder=10,
        clip_on=False
    )
    fig.patches.append(arrow2)

    # Arrow from right-center of B1(Warmed) in Panel C to Panel D
    gdf_warmed = plot_gdfs[('B1', 'Warmed')]
    bw = gdf_warmed.total_bounds  # xmin, ymin, xmax, ymax
    warmed_rc_data = (bw[2], (bw[1] + bw[3]) / 2)
    warmed_rc_fig = ax_c.transData.transform(warmed_rc_data)
    warmed_rc_fig = fig.transFigure.inverted().transform(warmed_rc_fig)

    # Target: left-center of Panel D
    d_bbox = ax_d.get_position()
    target_d = (d_bbox.x0, d_bbox.y0 + d_bbox.height * 0.5)

    arrow3 = FancyArrowPatch(
        warmed_rc_fig, target_d,
        transform=fig.transFigure,
        arrowstyle='->', mutation_scale=15,
        color='red', linewidth=1.5, zorder=10,
        clip_on=False
    )
    fig.patches.append(arrow3)

    # Save
    out_file = out_dir / 'Figure1.png'
    fig.savefig(out_file, dpi=500, bbox_inches='tight', pad_inches=0.1,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()

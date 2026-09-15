import numpy as np
import rasterio

from terrain import (
    load_dem,
    calculate_slope,
    calculate_slope_score,
    calculate_surface_score,
    calculate_tent_score,
)


# ============================================================
# PERCORSI DATI
# ============================================================

dem_path = (
    "data/raw/"
    "Copernicus_DSM_10_N42_00_E013_00_DEM.tif"
)

worldcover_path = (
    "data/processed/"
    "worldcover_aligned.tif"
)


# ============================================================
# CARICAMENTO DEM
# ============================================================

elevation, cell_size_x, cell_size_y, crs, bounds = (
    load_dem(dem_path)
)

print("DEM reale caricato.")

print("\nDimensioni:")
print(elevation.shape)

print("\nCRS:")
print(crs)

print("\nDimensione cella in metri:")
print("X:", cell_size_x)
print("Y:", cell_size_y)

print("\nBounds:")
print(bounds)


# ============================================================
# CALCOLO PENDENZA
# ============================================================

slope = calculate_slope(
    elevation,
    cell_size_x,
    cell_size_y
)


# ============================================================
# SLOPE SCORE
# ============================================================

slope_score = calculate_slope_score(
    slope
)


# ============================================================
# CARICAMENTO WORLDCOVER
# ============================================================

with rasterio.open(worldcover_path) as worldcover_file:
    worldcover = worldcover_file.read(1)
    worldcover_crs = worldcover_file.crs

print("\nWorldCover allineato caricato.")

print("Dimensioni:")
print(worldcover.shape)

print("CRS:")
print(worldcover_crs)


# ============================================================
# SURFACE SCORE
# ============================================================

surface_score = calculate_surface_score(
    worldcover
)


# ============================================================
# TENT SCORE COMBINATO
# ============================================================

tent_score = calculate_tent_score(
    slope_score,
    surface_score
)


# ============================================================
# RISULTATI
# ============================================================

print("\nPendenza reale:")

print(
    "Min:",
    np.nanmin(slope)
)

print(
    "Max:",
    np.nanmax(slope)
)

print(
    "Media:",
    np.nanmean(slope)
)


print("\nSlope Score:")

print(
    "Min:",
    np.nanmin(slope_score)
)

print(
    "Max:",
    np.nanmax(slope_score)
)

print(
    "Media:",
    np.nanmean(slope_score)
)


print("\nSurface Score:")

print(
    "Min:",
    np.nanmin(surface_score)
)

print(
    "Max:",
    np.nanmax(surface_score)
)

print(
    "Media:",
    np.nanmean(surface_score)
)


print("\nTent Score v0.1:")

print(
    "Min:",
    np.nanmin(tent_score)
)

print(
    "Max:",
    np.nanmax(tent_score)
)

print(
    "Media:",
    np.nanmean(tent_score)
)


# ============================================================
# DISTRIBUZIONE TENT SCORE
# ============================================================

print("\nDistribuzione Tent Score:")

print(
    "0.00 - 0.20:",
    np.mean(tent_score < 0.20) * 100,
    "%"
)

print(
    "0.20 - 0.40:",
    np.mean(
        (tent_score >= 0.20) &
        (tent_score < 0.40)
    ) * 100,
    "%"
)

print(
    "0.40 - 0.60:",
    np.mean(
        (tent_score >= 0.40) &
        (tent_score < 0.60)
    ) * 100,
    "%"
)

print(
    "0.60 - 0.80:",
    np.mean(
        (tent_score >= 0.60) &
        (tent_score < 0.80)
    ) * 100,
    "%"
)

print(
    "0.80 - 1.00:",
    np.mean(
        tent_score >= 0.80
    ) * 100,
    "%"
)
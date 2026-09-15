import numpy as np
import rasterio
from terrain import (
    load_dem,
    calculate_slope,
    calculate_slope_score,
    calculate_surface_score,
    calculate_tent_score,
    detect_flat_areas,
    calculate_candidate_mask,
    calculate_area_scores,
    calculate_terrain_quality,
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

elevation, cell_size_x, cell_size_y, crs, bounds, transform = (
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
# RILEVAMENTO AREE PIANEGGIANTI
# ============================================================

flat_areas = detect_flat_areas(
    slope,
    cell_size_x,
    cell_size_y,
    max_slope=10.0,
    min_area_m2=5000
)

area_score = calculate_area_scores(
    slope,
    cell_size_x,
    cell_size_y,
    max_slope=10.0,
)

print("\nAree pianeggianti:")

flat_area_m2 = (
    np.sum(flat_areas)
    * cell_size_x
    * cell_size_y
)

print(
    "Superficie valida:",
    flat_area_m2,
    "m²"
)

print(
    "Percentuale area:",
    np.mean(flat_areas) * 100,
    "%"
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
# CANDIDATE TERRAIN
# ============================================================

candidate_mask = calculate_candidate_mask(
    flat_areas,
    surface_score,
    min_surface_score=0.5
)

candidate_area_m2 = (
    np.sum(candidate_mask)
    * cell_size_x
    * cell_size_y
)

print("\nCandidate Terrain:")

print(
    "Superficie candidata:",
    candidate_area_m2,
    "m²"
)

print(
    "Percentuale area:",
    np.mean(candidate_mask) * 100,
    "%"
)

# ============================================================
# ESPORTAZIONE CANDIDATE MASK
# ============================================================

output_path = (
    "data/processed/"
    "candidate_mask.tif"
)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=candidate_mask.shape[0],
    width=candidate_mask.shape[1],
    count=1,
    dtype="uint8",
    crs=crs,
    transform=transform,
) as output:
    output.write(
        candidate_mask.astype(np.uint8),
        1
    )

print(
    "\nCandidate mask salvata:",
    output_path
)

# ============================================================
# TENT SCORE COMBINATO
# ============================================================

terrain_quality = calculate_terrain_quality(
    slope_score,
    surface_score,
    area_score,
)

print("\nTerrain Quality:")
print("Min:", terrain_quality.min())
print("Max:", terrain_quality.max())
print("Media:", terrain_quality.mean())

terrain_quality_output = "data/processed/terrain_quality.tif"

with rasterio.open(
    terrain_quality_output,
    "w",
    driver="GTiff",
    height=terrain_quality.shape[0],
    width=terrain_quality.shape[1],
    count=1,
    dtype="float32",
    crs=crs,
    transform=transform,
) as output:
    output.write(
        terrain_quality.astype(np.float32),
        1,
    )

print(
    "\nTerrain Quality salvata:",
    terrain_quality_output
)

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
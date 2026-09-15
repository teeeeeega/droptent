import numpy as np
import rasterio
from scipy.ndimage import sobel


# ============================================================
# CARICAMENTO DEM
# ============================================================

def load_dem(path):
    with rasterio.open(path) as dem:
        elevation = dem.read(1)
        transform = dem.transform
        bounds = dem.bounds
        crs = dem.crs

    # Dimensione della cella in gradi
    cell_size_x_deg = transform.a
    cell_size_y_deg = abs(transform.e)

    # Latitudine centrale della tile
    center_lat = (bounds.top + bounds.bottom) / 2

    # Conversione approssimativa gradi -> metri
    meters_per_degree_lat = 111320
    meters_per_degree_lon = (
        111320 * np.cos(np.radians(center_lat))
    )

    cell_size_x = (
        cell_size_x_deg * meters_per_degree_lon
    )

    cell_size_y = (
        cell_size_y_deg * meters_per_degree_lat
    )

    return (
        elevation,
        cell_size_x,
        cell_size_y,
        crs,
        bounds,
    )


# ============================================================
# CALCOLO PENDENZA
# ============================================================

def calculate_slope(
    elevation,
    cell_size_x,
    cell_size_y
):
    dx = (
        sobel(elevation, axis=1)
        / (8 * cell_size_x)
    )

    dy = (
        sobel(elevation, axis=0)
        / (8 * cell_size_y)
    )

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dx**2 + dy**2
            )
        )
    )

    return slope


# ============================================================
# CLASSIFICAZIONE PENDENZA
# ============================================================

def classify_slope(slope):
    classification = np.zeros_like(
        slope,
        dtype=np.uint8
    )

    classification[slope <= 5] = 1

    classification[
        (slope > 5) &
        (slope <= 10)
    ] = 2

    classification[slope > 10] = 3

    return classification


# ============================================================
# SUITABILITY PENDENZA - VERSIONE DISCRETA
# ============================================================

def calculate_suitability(classification):
    suitability = np.zeros_like(
        classification,
        dtype=float
    )

    suitability[
        classification == 1
    ] = 1.0

    suitability[
        classification == 2
    ] = 0.5

    suitability[
        classification == 3
    ] = 0.0

    return suitability


# ============================================================
# SLOPE SCORE CONTINUO
# ============================================================

def calculate_slope_score(slope):
    score = np.clip(
        1 - (slope / 10),
        0,
        1
    )

    return score


# ============================================================
# WORLDCOVER -> SURFACE SCORE
# ============================================================

def calculate_surface_score(worldcover):
    """
    Assegna un punteggio preliminare alle classi ESA WorldCover.

    Questi valori sono sperimentali e NON rappresentano
    ancora il modello definitivo di Droptent.
    """

    surface_score = np.zeros_like(
        worldcover,
        dtype=float
    )

    # 10 = Tree cover
    surface_score[worldcover == 10] = 0.25

    # 20 = Shrubland
    surface_score[worldcover == 20] = 0.50

    # 30 = Grassland
    surface_score[worldcover == 30] = 1.00

    # 40 = Cropland
    surface_score[worldcover == 40] = 0.30

    # 50 = Built-up
    surface_score[worldcover == 50] = 0.00

    # 60 = Bare / sparse vegetation
    surface_score[worldcover == 60] = 0.90

    # 70 = Snow and ice
    surface_score[worldcover == 70] = 0.00

    # 80 = Permanent water bodies
    surface_score[worldcover == 80] = 0.00

    # 90 = Herbaceous wetland
    surface_score[worldcover == 90] = 0.00

    # 95 = Mangroves
    surface_score[worldcover == 95] = 0.00

    # 100 = Moss and lichen
    surface_score[worldcover == 100] = 0.50

    return surface_score


# ============================================================
# TENT SCORE
# ============================================================

def calculate_tent_score(
    slope_score,
    surface_score
):
    """
    Prima versione combinata del Tent Score.

    Il punteggio finale è il prodotto dei due fattori:
    - pendenza
    - superficie
    """

    tent_score = (
        slope_score *
        surface_score
    )

    return tent_score


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

print("\nWorldCover allineato caricato.")

print("Dimensioni:")
print(worldcover.shape)

print("CRS:")
print(worldcover_file.crs)


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
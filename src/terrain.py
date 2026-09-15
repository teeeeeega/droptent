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
    transform,
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
# RILEVAMENTO AREE PIANEGGIANTI
# ============================================================

def detect_flat_areas(
    slope,
    cell_size_x,
    cell_size_y,
    max_slope=10.0,
    min_area_m2=5000
):
    """
    Individua aree di terreno sufficientemente pianeggianti
    e mantiene soltanto quelle con una superficie minima.

    Parametri:
    - slope: array della pendenza in gradi
    - cell_size_x: dimensione della cella in metri sull'asse X
    - cell_size_y: dimensione della cella in metri sull'asse Y
    - max_slope: pendenza massima accettata
    - min_area_m2: superficie minima dell'area in m²

    Restituisce:
    - flat_mask: maschera booleana delle aree pianeggianti
    """

    from scipy import ndimage

    # Celle che rispettano la soglia di pendenza
    flat_mask = slope <= max_slope

    # Individuazione delle componenti connesse
    structure = ndimage.generate_binary_structure(
        2,
        2
    )

    labeled_areas, num_areas = ndimage.label(
        flat_mask,
        structure=structure
    )

    # Superficie di una singola cella
    cell_area_m2 = (
        cell_size_x *
        cell_size_y
    )

    # Dimensione di ogni area in m²
    area_sizes_m2 = (
        np.bincount(
            labeled_areas.ravel()
        )
        * cell_area_m2
    )

    # Manteniamo soltanto le aree abbastanza grandi
    valid_areas = (
        area_sizes_m2 >= min_area_m2
    )

    valid_areas[0] = False

    flat_mask = valid_areas[
        labeled_areas
    ]

    return flat_mask
# ============================================================
# CANDIDATE TERRAIN MASK
# ============================================================

def calculate_candidate_mask(
    flat_areas,
    surface_score,
    min_surface_score=0.5
):
    """
    Identifica le aree che soddisfano contemporaneamente:

    - pendenza accettabile
    - superficie sufficientemente adatta

    Restituisce una maschera booleana.
    """

    candidate_mask = (
        flat_areas &
        (surface_score >= min_surface_score)
    )

    return candidate_mask
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
    surface_score[worldcover == 60] = 0.70

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
def calculate_terrain_quality(
    slope_score,
    surface_score,
    area_score,
    water_score,
    road_score,
):
    terrain_quality = (
        slope_score * 0.35
        + surface_score * 0.225
        + area_score * 0.175
        + water_score * 0.125
        + road_score * 0.125
    )

    return np.clip(terrain_quality, 0, 1)

def calculate_area_score(
    area_m2,
    min_area_m2=500,
    ideal_area_m2=5000,
):
    area_score = (
        (area_m2 - min_area_m2)
        / (ideal_area_m2 - min_area_m2)
    )

    return np.clip(area_score, 0, 1)
def calculate_area_scores(
    slope,
    cell_size_x,
    cell_size_y,
    max_slope=10.0,
):
    flat_mask = slope <= max_slope

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 2)

    labeled_areas, num_areas = ndimage.label(
        flat_mask,
        structure=structure,
    )

    cell_area_m2 = cell_size_x * cell_size_y

    area_sizes = ndimage.sum(
        flat_mask,
        labeled_areas,
        range(1, num_areas + 1),
    )

    area_sizes_m2 = area_sizes * cell_area_m2

    area_scores = np.zeros(
        num_areas + 1,
        dtype=float,
    )

    area_scores[1:] = calculate_area_score(
        area_sizes_m2
    )

    return area_scores[labeled_areas]
def calculate_water_distance(
    candidate_mask,
    water_gdf,
    transform,
    crs,
    cell_size_x,
    cell_size_y,
):
    """
    Calcola la distanza minima dall'acqua per ogni pixel candidato.

    La distanza viene restituita in metri.
    """
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt

    # Trasforma l'acqua nel CRS del DEM
    water_projected = water_gdf.to_crs(crs)

    # Crea una maschera vuota
    water_mask = np.zeros(
        candidate_mask.shape,
        dtype=bool,
    )

    # Rasterizza le geometrie dell'acqua
    from rasterio.features import rasterize

    shapes = [
        (geometry, 1)
        for geometry in water_projected.geometry
        if geometry is not None
        and not geometry.is_empty
        and geometry.geom_type in ("Polygon", "MultiPolygon")
    ]

    water_mask = rasterize(
        shapes,
        out_shape=candidate_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    # Distanza in pixel dall'acqua
    distance_pixels = distance_transform_edt(
        ~water_mask
    )

    # Conversione in metri
    distance_m = distance_pixels * np.mean(
        (cell_size_x, cell_size_y)
    )

    # Manteniamo la distanza solo nelle aree candidate
    # Imposta a 0 la distanza nelle aree non candidate
    distance_m[~candidate_mask] = 0

    return distance_m


def calculate_water_score(distance_m):
    """
    Calcola uno score in base alla distanza dall'acqua.
    """

    water_score = np.zeros_like(
        distance_m,
        dtype=float,
    )

    water_score[distance_m <= 100] = 1.0

    mask = (
        (distance_m > 100)
        & (distance_m <= 500)
    )

    water_score[mask] = (
        1.0
        - 0.5 * (
            (distance_m[mask] - 100)
            / 400
        )
    )

    mask = (
        (distance_m > 500)
        & (distance_m <= 1000)
    )

    water_score[mask] = (
        0.5
        * (
            1
            - (distance_m[mask] - 500)
            / 500
        )
    )

    return water_score
def calculate_road_distance(
    candidate_mask,
    roads_gdf,
    transform,
    crs,
    cell_size_x,
    cell_size_y,
    highway_types,
):
    import rasterio.features
    from scipy.ndimage import distance_transform_edt

    roads = roads_gdf[
        roads_gdf["highway"].apply(
            lambda value: any(
                highway in highway_types
                for highway in (
                    value if isinstance(value, list)
                    else [value]
                )
            )
        )
    ]

    roads = roads.to_crs(crs)

    road_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in roads.geometry if geom is not None],
        out_shape=candidate_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    distance_pixels = distance_transform_edt(road_mask == 0)

    cell_size = (cell_size_x + cell_size_y) / 2

    distance_m = distance_pixels * cell_size

    distance_m[candidate_mask == 0] = 0

    return distance_m

def calculate_track_distance(
    candidate_mask,
    roads_gdf,
    transform,
    crs,
    cell_size_x,
    cell_size_y,
):
    import rasterio.features
    from scipy.ndimage import distance_transform_edt

    tracks = roads_gdf[
        roads_gdf["highway"].apply(
            lambda value: any(
                highway == "track"
                for highway in (
                    value if isinstance(value, list)
                    else [value]
                )
            )
        )
    ]

    tracks = tracks.to_crs(crs)

    track_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in tracks.geometry if geom is not None],
        out_shape=candidate_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    distance_pixels = distance_transform_edt(track_mask == 0)

    cell_size = (cell_size_x + cell_size_y) / 2

    distance_m = distance_pixels * cell_size

    distance_m[candidate_mask == 0] = 0

    return distance_m

def calculate_road_score(distance_m):
    road_score = np.zeros_like(distance_m, dtype=float)

    # 0-100 m: troppo vicino alla strada
    road_score[distance_m <= 100] = 0.0

    # 100-300 m: aumenta da 0 a 0.5
    mask = (distance_m > 100) & (distance_m <= 300)
    road_score[mask] = (
        0.5 * ((distance_m[mask] - 100) / 200)
    )

    # 300-1000 m: aumenta da 0.5 a 1
    mask = (distance_m > 300) & (distance_m <= 1000)
    road_score[mask] = (
        0.5 + 0.5 * ((distance_m[mask] - 300) / 700)
    )

    # 1-3 km: diminuisce da 1 a 0.5
    mask = (distance_m > 1000) & (distance_m <= 3000)
    road_score[mask] = (
        1.0 - 0.5 * ((distance_m[mask] - 1000) / 2000)
    )

    # Oltre 3 km: diminuisce da 0.5 a 0
    mask = distance_m > 3000
    road_score[mask] = (
        0.5 * (1 - (distance_m[mask] - 3000) / 3000)
    )

    return np.clip(road_score, 0, 1)

def calculate_track_score(distance_m):
    track_score = np.zeros_like(distance_m, dtype=float)

    # Entro 100 m: accesso molto facile
    track_score[distance_m <= 100] = 1.0

    # 100-500 m: diminuisce da 1 a 0.8
    mask = (distance_m > 100) & (distance_m <= 500)
    track_score[mask] = (
        1.0 - 0.2 * ((distance_m[mask] - 100) / 400)
    )

    # 500-1000 m: diminuisce da 0.8 a 0.5
    mask = (distance_m > 500) & (distance_m <= 1000)
    track_score[mask] = (
        0.8 - 0.3 * ((distance_m[mask] - 500) / 500)
    )

    # 1-2 km: diminuisce da 0.5 a 0.2
    mask = (distance_m > 1000) & (distance_m <= 2000)
    track_score[mask] = (
        0.5 - 0.3 * ((distance_m[mask] - 1000) / 1000)
    )

    # Oltre 2 km: diminuisce fino a 0
    mask = distance_m > 2000
    track_score[mask] = (
        0.2 * (1 - (distance_m[mask] - 2000) / 2000)
    )

    return np.clip(track_score, 0, 1)

def calculate_trail_distance(
    candidate_mask,
    roads_gdf,
    transform,
    crs,
    cell_size_x,
    cell_size_y,
):
    import rasterio.features
    from scipy.ndimage import distance_transform_edt

    trail_types = [
        "path",
        "footway",
        "cycleway",
        "bridleway",
    ]

    trails = roads_gdf[
        roads_gdf["highway"].apply(
            lambda value: any(
                highway in trail_types
                for highway in (
                    value if isinstance(value, list)
                    else [value]
                )
            )
        )
    ]

    trails = trails.to_crs(crs)

    trail_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in trails.geometry if geom is not None],
        out_shape=candidate_mask.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    distance_pixels = distance_transform_edt(trail_mask == 0)

    cell_size = (cell_size_x + cell_size_y) / 2

    distance_m = distance_pixels * cell_size

    distance_m[candidate_mask == 0] = 0

    return distance_m

def calculate_trail_score(distance_m):
    trail_score = np.zeros_like(distance_m, dtype=float)

    # Entro 100 m: accessibilità eccellente
    trail_score[distance_m <= 100] = 1.0

    # 100-300 m: diminuisce da 1 a 0.8
    mask = (distance_m > 100) & (distance_m <= 300)
    trail_score[mask] = (
        1.0 - 0.2 * ((distance_m[mask] - 100) / 200)
    )

    # 300-500 m: diminuisce da 0.8 a 0.6
    mask = (distance_m > 300) & (distance_m <= 500)
    trail_score[mask] = (
        0.8 - 0.2 * ((distance_m[mask] - 300) / 200)
    )

    # 500-1000 m: diminuisce da 0.6 a 0.4
    mask = (distance_m > 500) & (distance_m <= 1000)
    trail_score[mask] = (
        0.6 - 0.2 * ((distance_m[mask] - 500) / 500)
    )

    # 1-2 km: diminuisce da 0.4 a 0.15
    mask = (distance_m > 1000) & (distance_m <= 2000)
    trail_score[mask] = (
        0.4 - 0.25 * ((distance_m[mask] - 1000) / 1000)
    )

    # Oltre 2 km: diminuisce fino a 0
    mask = distance_m > 2000
    trail_score[mask] = (
        0.15 * (1 - (distance_m[mask] - 2000) / 2000)
    )

    return np.clip(trail_score, 0, 1)

def calculate_protected_area_mask(
    protected_gdf,
    transform,
    crs,
    shape,
):
    import rasterio.features

    protected_gdf = protected_gdf.to_crs(crs)

    protected_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in protected_gdf.geometry if geom is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    return protected_mask.astype(bool)
def extract_candidate_zones(
    terrain_quality,
    candidate_mask,
    cell_size_x,
    cell_size_y,
    min_score=0.6,
    min_area_m2=5000,
):
    from scipy import ndimage

    quality_mask = (
    candidate_mask
    & (terrain_quality >= 0.6)
)

    labels, num_features = ndimage.label(
        quality_mask,
        structure=np.ones((3, 3), dtype=int),
    )

    if num_features == 0:
        return []

    areas = ndimage.sum(
        quality_mask,
        labels,
        index=np.arange(1, num_features + 1),
    )

    pixel_area = cell_size_x * cell_size_y

    zones = []

    for label_id, pixel_count in enumerate(areas, start=1):
        area_m2 = pixel_count * pixel_area

        if area_m2 < min_area_m2:
            continue

        zone_mask = labels == label_id

        scores = terrain_quality[zone_mask]

        zones.append(
            {
                "label": label_id,
                "area_m2": area_m2,
                "mean_score": float(scores.mean()),
                "max_score": float(scores.max()),
            }
        )

    zones.sort(
        key=lambda zone: zone["mean_score"],
        reverse=True,
    )

    return zones


def export_candidate_zones(
    terrain_quality,
    candidate_mask,
    transform,
    crs,
    cell_size_x,
    cell_size_y,
    min_score=0.9,
    min_area_m2=5000,
    output_path="data/processed/candidate_zones.gpkg",
):
    import geopandas as gpd
    from scipy import ndimage
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import rasterio.features

    quality_mask = (
    candidate_mask
    & (terrain_quality >= 0.6)
)

    labels, num_features = ndimage.label(
        quality_mask,
        structure=np.ones((3, 3), dtype=int),
    )

    if num_features == 0:
        return

    areas = ndimage.sum(
        quality_mask,
        labels,
        index=np.arange(1, num_features + 1),
    )

    pixel_area = cell_size_x * cell_size_y

    zones = []

    for label_id, pixel_count in enumerate(areas, start=1):
        area_m2 = pixel_count * pixel_area

        if area_m2 < min_area_m2:
            continue

        zone_mask = labels == label_id

        scores = terrain_quality[zone_mask]

        mean_score = float(scores.mean())
        max_score = float(scores.max())
        if mean_score < min_score:
         continue

        geometries = rasterio.features.shapes(
            zone_mask.astype("uint8"),
            mask=zone_mask,
            transform=transform,
        )

        polygons = [
            shape(geometry)
            for geometry, value in geometries
            if value == 1
        ]

        if not polygons:
            continue

        geometry = unary_union(polygons)

        zones.append(
            {
                "geometry": geometry,
                "label": label_id,
                "area_m2": area_m2,
                "mean_score": mean_score,
                "max_score": max_score,
            }
        )

    if not zones:
        return

    gdf = gpd.GeoDataFrame(
        zones,
        crs=crs,
    )

    gdf.to_file(
        output_path,
        layer="candidate_zones",
        driver="GPKG",
    )

    print(
        f"Candidate zones esportate: {output_path}"
    )


def rank_candidate_zones(
    zones,
    score_weight=0.7,
    area_weight=0.3,
):
    """
    Ordina le candidate combinando qualità media e dimensione dell'area.
    """

    if not zones:
        return []

    scores = np.array(
        [zone["mean_score"] for zone in zones],
        dtype=float,
    )

    areas = np.array(
        [zone["area_m2"] for zone in zones],
        dtype=float,
    )

    score_min = scores.min()
    score_max = scores.max()

    area_min = areas.min()
    area_max = areas.max()

    if score_max > score_min:
        normalized_scores = (
            (scores - score_min)
            / (score_max - score_min)
        )
    else:
        normalized_scores = np.ones_like(scores)

    if area_max > area_min:
        normalized_areas = (
            (areas - area_min)
            / (area_max - area_min)
        )
    else:
        normalized_areas = np.ones_like(areas)

    for zone, normalized_score, normalized_area in zip(
        zones,
        normalized_scores,
        normalized_areas,
    ):
        ranking_score = (
            normalized_score * score_weight
            + normalized_area * area_weight
        )

        zone["ranking_score"] = float(ranking_score)

    zones.sort(
        key=lambda zone: zone["ranking_score"],
        reverse=True,
    )

    return zones
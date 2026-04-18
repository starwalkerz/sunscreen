"""
Southern Vegetation Occlusion Module (SVOM)
Post-processing module for Nordic rooftop solar suitability estimation.

This module filters detected buildings based on vegetation obstruction
in the southern direction (azimuth 135°-225°) which significantly impacts
solar irradiance in Nordic environments (Finland).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
import pyproj
from functools import partial


class SVOM:
    """
    Southern Vegetation Occlusion Module for filtering rooftop detections
    based on southern vegetation obstruction.
    """

    NDVI_VEGETATION_THRESHOLD = 0.5
    SOUTHERN_AZIMUTH_MIN = 135.0
    SOUTHERN_AZIMUTH_MAX = 225.0
    BUFFER_DISTANCE_METERS = 50.0
    VALID_ROOF_THRESHOLD = 0.3

    def __init__(
        self,
        ndvi_tiff_path: Union[str, Path],
        crs_epsg: int = 3067,
        buffer_distance: float = 50.0,
        azimuth_min: float = 135.0,
        azimuth_max: float = 225.0,
        ndvi_threshold: float = 0.5,
        valid_roof_threshold: float = 0.3
    ):
        """
        Initialize the SVOM with NDVI data.

        Args:
            ndvi_tiff_path: Path to Sentinel-2 NDVI summer composite GeoTIFF
            crs_epsg: EPSG code for target CRS (default: 3067 for Finland)
            buffer_distance: Buffer distance in meters around buildings
            azimuth_min: Minimum azimuth angle for southern sector (degrees)
            azimuth_max: Maximum azimuth angle for southern sector (degrees)
            ndvi_threshold: NDVI threshold for vegetation detection
            valid_roof_threshold: Minimum score for valid roof
        """
        self.ndvi_tiff_path = Path(ndvi_tiff_path)
        self.crs_epsg = crs_epsg
        self.buffer_distance = buffer_distance
        self.azimuth_min = azimuth_min
        self.azimuth_max = azimuth_max
        self.ndvi_threshold = ndvi_threshold
        self.valid_roof_threshold = valid_roof_threshold

        self._ndvi_data = None
        self._transform = None
        self._crs = None

    def _load_ndvi(self) -> None:
        """Load and cache NDVI data from GeoTIFF."""
        if self._ndvi_data is not None:
            return

        with rasterio.open(self.ndvi_tiff_path) as src:
            self._ndvi_data = src.read(1)
            self._transform = src.transform
            self._crs = src.crs

    def _pixel_to_geotransform(
        self,
        row: int,
        col: int
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to geospatial coordinates."""
        x = self._transform[0] + col * self._transform[1] + row * self._transform[2]
        y = self._transform[3] + col * self._transform[4] + row * self._transform[5]
        return x, y

    def _geo_to_pixel(
        self,
        x: float,
        y: float
    ) -> Tuple[int, int]:
        """Convert geospatial coordinates to pixel coordinates."""
        inv_transform = ~self._transform
        col, row = inv_transform * (x, y)
        return int(row), int(col)

    def _create_southern_sector(
        self,
        centroid: Point,
        buffer_distance: float
    ) -> Polygon:
        """
        Create a southern sector geometry using azimuth-based angular geometry.

        The sector is defined as the intersection of:
        1. A circular buffer around the building centroid
        2. A wedge defined by azimuth angles (southern direction)

        Args:
            centroid: Building centroid point
            buffer_distance: Buffer radius in meters

        Returns:
            Polygon representing the southern sector
        """
        cx, cy = centroid.x, centroid.y

        transformer = pyproj.Transformer.from_crs(
            self.crs_epsg,
            self.crs_epsg,
            always_xy=True
        )

        buffer_circle = centroid.buffer(buffer_distance)

        num_points = 360
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        southern_mask = (
            (angles >= np.radians(self.azimuth_min)) &
            (angles <= np.radians(self.azimuth_max))
        )

        southern_angles = angles[southern_mask]

        if len(southern_angles) < 3:
            return buffer_circle

        wedge_points = [centroid.coords[0]]
        for angle in southern_angles:
            dx = buffer_distance * np.cos(angle)
            dy = buffer_distance * np.sin(angle)
            wedge_points.append((cx + dx, cy + dy))
        wedge_points.append(centroid.coords[0])

        try:
            wedge_polygon = Polygon(wedge_points)
            sector = buffer_circle.intersection(wedge_polygon)
            return sector
        except Exception:
            return buffer_circle

    def _get_southern_pixels(
        self,
        building_polygon: Polygon,
        buffer_distance: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract NDVI pixel values within the southern sector of the buffer zone.

        Args:
            building_polygon: Building footprint polygon
            buffer_distance: Buffer distance in meters

        Returns:
            Tuple of (row_indices, col_indices) for southern sector pixels
        """
        self._load_ndvi()

        buffered = building_polygon.buffer(buffer_distance)
        minx, miny, maxx, maxy = buffered.bounds

        row_min, col_min = self._geo_to_pixel(minx, maxy)
        row_max, col_max = self._geo_to_pixel(maxx, miny)

        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(self._ndvi_data.shape[0], row_max + 1)
        col_max = min(self._ndvi_data.shape[1], col_max + 1)

        if row_min >= row_max or col_min >= col_max:
            return np.array([]), np.array([])

        window = self._ndvi_data[row_min:row_max, col_min:col_max]

        centroid = building_polygon.centroid

        rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
        rows = np.broadcast_to(rows, window.shape)
        cols = np.broadcast_to(cols, window.shape)

        coords = np.array([
            [self._pixel_to_geotransform(r, c) for r, c in zip(row_arr, col_arr)]
            for row_arr, col_arr in zip(rows, cols)
        ])

        lons = coords[:, :, 0]
        lats = coords[:, :, 1]

        dlon = lons - centroid.x
        dlat = lats - centroid.y
        azimuths = np.arctan2(dlat, dlon)
        azimuths = (np.degrees(azimuths) + 90) % 360

        southern_mask = (
            (azimuths >= self.azimuth_min) &
            (azimuths <= self.azimuth_max)
        )

        southern_rows = rows[southern_mask]
        southern_cols = cols[southern_mask]

        return southern_rows, southern_cols

    def compute_vegetation_score(
        self,
        building_polygon: Polygon,
        buffer_distance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute vegetation occlusion score for a building.

        Args:
            building_polygon: Building footprint as Shapely Polygon
            buffer_distance: Optional custom buffer distance (defaults to instance value)

        Returns:
            Dictionary containing:
                - vegetation_score: S = 1 - (vegetation_pixels / total_southern_pixels)
                - vegetation_ratio: Ratio of vegetation pixels
                - total_southern_pixels: Total pixels in southern sector
                - vegetation_pixels: Number of vegetation pixels (NDVI > 0.5)
                - valid_roof: Boolean indicating if roof is suitable
        """
        if buffer_distance is None:
            buffer_distance = self.buffer_distance

        self._load_ndvi()

        southern_rows, southern_cols = self._get_southern_pixels(
            building_polygon,
            buffer_distance
        )

        if len(southern_rows) == 0:
            return {
                'vegetation_score': 1.0,
                'vegetation_ratio': 0.0,
                'total_southern_pixels': 0,
                'vegetation_pixels': 0,
                'valid_roof': True,
                'error': 'No southern sector pixels found'
            }

        ndvi_values = self._ndvi_data[southern_rows, southern_cols]

        vegetation_mask = ndvi_values > self.ndvi_threshold
        vegetation_pixels = np.sum(vegetation_mask)
        total_pixels = len(ndvi_values)

        vegetation_ratio = vegetation_pixels / total_pixels
        vegetation_score = 1.0 - vegetation_ratio
        valid_roof = vegetation_score >= self.valid_roof_threshold

        return {
            'vegetation_score': float(vegetation_score),
            'vegetation_ratio': float(vegetation_ratio),
            'total_southern_pixels': int(total_pixels),
            'vegetation_pixels': int(vegetation_pixels),
            'valid_roof': bool(valid_roof)
        }

    def filter_buildings(
        self,
        buildings: List[Polygon],
        buffer_distance: Optional[float] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a list of building polygons based on vegetation occlusion.

        Args:
            buildings: List of building footprints as Shapely Polygons
            buffer_distance: Optional custom buffer distance

        Returns:
            Tuple of (valid_buildings, invalid_buildings) where each is a list
            of dictionaries with building geometry and scores
        """
        valid_buildings = []
        invalid_buildings = []

        for i, building in enumerate(buildings):
            result = self.compute_vegetation_score(building, buffer_distance)
            building_result = {
                'building_id': i,
                'geometry': building,
                **result
            }

            if result['valid_roof']:
                valid_buildings.append(building_result)
            else:
                invalid_buildings.append(building_result)

        return valid_buildings, invalid_buildings


def load_building_polygons(
    polygons: List,
    crs_epsg: int = 3067
) -> List[Polygon]:
    """
    Convert various polygon formats to Shapely Polygons with proper CRS.

    Args:
        polygons: List of polygons in various formats (GeoJSON dicts, coordinate lists, etc.)
        crs_epsg: Target CRS EPSG code

    Returns:
        List of Shapely Polygons
    """
    from shapely.geometry import shape

    shapely_polygons = []

    for poly in polygons:
        if isinstance(poly, Polygon):
            shapely_polygons.append(poly)
        elif isinstance(poly, dict):
            if 'geometry' in poly:
                shapely_polygons.append(shape(poly['geometry']))
            else:
                shapely_polygons.append(shape(poly))
        elif isinstance(poly, (list, tuple)):
            if len(poly) >= 3 and isinstance(poly[0][0], (list, tuple)):
                shapely_polygons.append(Polygon(poly))
            else:
                shapely_polygons.append(Polygon(poly))
        else:
            raise TypeError(f"Unsupported polygon format: {type(poly)}")

    return shapely_polygons


def create_svom_from_tiff(
    ndvi_tiff_path: Union[str, Path],
    **kwargs
) -> SVOM:
    """
    Factory function to create an SVOM instance from NDVI GeoTIFF path.

    Args:
        ndvi_tiff_path: Path to Sentinel-2 NDVI summer composite
        **kwargs: Additional arguments for SVOM initialization

    Returns:
        SVOM instance
    """
    return SVOM(ndvi_tiff_path=ndvi_tiff_path, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Southern Vegetation Occlusion Module (SVOM)'
    )
    parser.add_argument(
        '--ndvi-tiff',
        required=True,
        help='Path to NDVI GeoTIFF'
    )
    parser.add_argument(
        '--buildings',
        nargs='+',
        help='Building polygon coordinates (json files or WKT)'
    )
    parser.add_argument(
        '--buffer',
        type=float,
        default=50.0,
        help='Buffer distance in meters'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Valid roof threshold'
    )

    args = parser.parse_args()

    svom = SVOM(
        ndvi_tiff_path=args.ndvi_tiff,
        buffer_distance=args.buffer,
        valid_roof_threshold=args.threshold
    )

    print(f"SVOM initialized with:")
    print(f"  NDVI threshold: {svom.ndvi_threshold}")
    print(f"  Southern sector: {svom.azimuth_min}° - {svom.azimuth_max}°")
    print(f"  Buffer distance: {svom.buffer_distance}m")
    print(f"  Valid roof threshold: {svom.valid_roof_threshold}")
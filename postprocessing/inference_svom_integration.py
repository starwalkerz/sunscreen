"""
Inference pipeline integration for SVOM post-processing.

This script demonstrates how to integrate the Southern Vegetation Occlusion Module
(SVOM) with the existing rooftop detection pipeline.

The module is designed to work with building polygon outputs from the CNN-based
rooftop detection model and filter them based on southern vegetation obstruction.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import argparse

from shapely.geometry import Polygon, shape, mapping

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from postprocessing.svom_ndvi_filter import SVOM, load_building_polygons
except ImportError:
    from svom_ndvi_filter import SVOM, load_building_polygons


class RooftopInferenceWithSVOM:
    """
    Enhanced inference pipeline that integrates SVOM post-processing
    for Nordic rooftop solar suitability estimation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        ndvi_tiff_path: Optional[str] = None,
        svom_config: Optional[Dict] = None,
        crs_epsg: int = 3067
    ):
        """
        Initialize the inference pipeline with SVOM.

        Args:
            model_path: Path to trained CNN model weights
            ndvi_tiff_path: Path to Sentinel-2 NDVI summer composite
            svom_config: Dictionary of SVOM configuration parameters
            crs_epsg: EPSG code for coordinate reference system
        """
        self.model_path = model_path
        self.crs_epsg = crs_epsg

        self.svom = None
        if ndvi_tiff_path:
            config = svom_config or {}
            self.svom = SVOM(
                ndvi_tiff_path=ndvi_tiff_path,
                crs_epsg=crs_epsg,
                **config
            )

    def _detect_rooftops(self, image_path: str) -> List[Polygon]:
        """
        Run the CNN model to detect rooftops.

        This is a placeholder - replace with actual model inference code.

        Args:
            image_path: Path to input aerial image

        Returns:
            List of detected building polygons
        """
        pass

    def _polygons_from_segmentation(
        self,
        mask: np.ndarray,
        transform: tuple,
        min_area: float = 10.0
    ) -> List[Polygon]:
        """
        Convert segmentation mask to building polygons.

        Args:
            mask: Binary segmentation mask
            transform: Geotransform for pixel-to-geo conversion
            min_area: Minimum polygon area in square meters

        Returns:
            List of building polygons
        """
        from skimage import measure

        contours = measure.find_contours(mask, 0.5)
        polygons = []

        for contour in contours:
            if len(contour) < 3:
                continue

            coords = np.array([
                [transform[0] + c[1] * transform[1] + c[0] * transform[2],
                 transform[3] + c[1] * transform[4] + c[0] * transform[5]]
                for c in contour
            ])

            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area >= min_area:
                    polygons.append(poly)
            except Exception:
                continue

        return polygons

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        include_invalid: bool = False
    ) -> Dict:
        """
        Process a single image through detection + SVOM filtering.

        Args:
            image_path: Path to input aerial image
            output_path: Optional path to save results
            include_invalid: Whether to include invalid roofs in output

        Returns:
            Dictionary containing detection results with SVOM scores
        """
        building_polygons = self._detect_rooftops(image_path)

        if not self.svom:
            return {
                'image_path': image_path,
                'buildings': [
                    {'geometry': mapping(p), 'valid_roof': True}
                    for p in building_polygons
                ],
                'svom_applied': False
            }

        valid_buildings, invalid_buildings = self.svom.filter_buildings(
            building_polygons
        )

        results = {
            'image_path': image_path,
            'total_detections': len(building_polygons),
            'valid_roofs': len(valid_buildings),
            'invalid_roofs': len(invalid_buildings),
            'svom_applied': True,
            'svom_config': {
                'buffer_distance': self.svom.buffer_distance,
                'azimuth_min': self.svom.azimuth_min,
                'azimuth_max': self.svom.azimuth_max,
                'ndvi_threshold': self.svom.ndvi_threshold,
                'valid_roof_threshold': self.svom.valid_roof_threshold
            },
            'valid_buildings': valid_buildings
        }

        if include_invalid:
            results['invalid_buildings'] = invalid_buildings

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, default=mapping)

        return results

    def process_batch(
        self,
        image_dir: str,
        output_dir: str,
        file_pattern: str = '*.png'
    ) -> List[Dict]:
        """
        Process a batch of images.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for output results
            file_pattern: Glob pattern for image files

        Returns:
            List of result dictionaries
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for image_path in image_dir.glob(file_pattern):
            output_path = output_dir / f"{image_path.stem}_results.json"
            result = self.process_image(
                str(image_path),
                str(output_path),
                include_invalid=True
            )
            results.append(result)

        return results


def append_svom_to_building(
    building_result: Dict,
    ndvi_tiff_path: str,
    crs_epsg: int = 3067,
    **svom_kwargs
) -> Dict:
    """
    Standalone function to append SVOM scores to a single building.

    This function can be called during inference to add vegetation
    occlusion scores to each detected building.

    Args:
        building_result: Dictionary containing building geometry
        ndvi_tiff_path: Path to NDVI GeoTIFF
        crs_epsg: Coordinate reference system EPSG code
        **svom_kwargs: Additional SVOM configuration

    Returns:
        Updated building_result with vegetation_score and valid_roof
    """
    svom = SVOM(ndvi_tiff_path=ndvi_tiff_path, crs_epsg=crs_epsg, **svom_kwargs)

    if isinstance(building_result.get('geometry'), dict):
        building_poly = shape(building_result['geometry'])
    elif isinstance(building_result.get('geometry'), Polygon):
        building_poly = building_result['geometry']
    else:
        raise ValueError("Building geometry not found in building_result")

    score = svom.compute_vegetation_score(building_poly)

    building_result['vegetation_score'] = score['vegetation_score']
    building_result['vegetation_ratio'] = score['vegetation_ratio']
    building_result['valid_roof'] = score['valid_roof']
    building_result['total_southern_pixels'] = score['total_southern_pixels']
    building_result['vegetation_pixels'] = score['vegetation_pixels']

    return building_result


def integrate_with_existing_inference(
    detected_buildings: List[Dict],
    ndvi_tiff_path: str,
    crs_epsg: int = 3067,
    **svom_kwargs
) -> List[Dict]:
    """
    Integrate SVOM filtering with existing inference pipeline.

    This function should be called in the inference loop after
    building detection, before saving results.

    Args:
        detected_buildings: List of detected building dictionaries
        ndvi_tiff_path: Path to Sentinel-2 NDVI summer composite
        crs_epsg: EPSG code for coordinate reference system
        **svom_kwargs: Additional SVOM configuration

    Returns:
        List of buildings with appended vegetation_score and valid_roof
    """
    svom = SVOM(ndvi_tiff_path=ndvi_tiff_path, crs_epsg=crs_epsg, **svom_kwargs)

    enriched_buildings = []

    for building in detected_buildings:
        if isinstance(building.get('geometry'), dict):
            building_poly = shape(building['geometry'])
        elif isinstance(building.get('geometry'), Polygon):
            building_poly = building['geometry']
        else:
            enriched_buildings.append({
                **building,
                'vegetation_score': None,
                'valid_roof': None,
                'error': 'Invalid geometry'
            })
            continue

        score = svom.compute_vegetation_score(building_poly)

        enriched_buildings.append({
            **building,
            'vegetation_score': score['vegetation_score'],
            'vegetation_ratio': score['vegetation_ratio'],
            'valid_roof': score['valid_roof'],
            'total_southern_pixels': score['total_southern_pixels'],
            'vegetation_pixels': score['vegetation_pixels']
        })

    return enriched_buildings


def filter_valid_roofs(
    buildings: List[Dict],
    min_score: float = 0.3
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter buildings by valid_roof flag.

    Args:
        buildings: List of building dictionaries with valid_roof field
        min_score: Minimum vegetation_score threshold

    Returns:
        Tuple of (valid_buildings, invalid_buildings)
    """
    valid = [b for b in buildings if b.get('valid_roof', False)]
    invalid = [b for b in buildings if not b.get('valid_roof', True)]

    return valid, invalid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rooftop inference with SVOM post-processing'
    )
    parser.add_argument(
        '--model',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--ndvi-tiff',
        required=True,
        help='Path to Sentinel-2 NDVI summer composite'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input image or directory'
    )
    parser.add_argument(
        '--output',
        help='Output path (file or directory)'
    )
    parser.add_argument(
        '--crs',
        type=int,
        default=3067,
        help='EPSG code for CRS (default: 3067 Finland)'
    )
    parser.add_argument(
        '--buffer',
        type=float,
        default=50.0,
        help='Buffer distance in meters (default: 50)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Valid roof threshold (default: 0.3)'
    )
    parser.add_argument(
        '--include-invalid',
        action='store_true',
        help='Include invalid roofs in output'
    )

    args = parser.parse_args()

    pipeline = RooftopInferenceWithSVOM(
        model_path=args.model,
        ndvi_tiff_path=args.ndvi_tiff,
        svom_config={
            'buffer_distance': args.buffer,
            'valid_roof_threshold': args.threshold
        },
        crs_epsg=args.crs
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        results = pipeline.process_batch(
            str(input_path),
            args.output or str(input_path / 'results'),
            include_invalid=args.include_invalid
        )
        print(f"Processed {len(results)} images")
    else:
        result = pipeline.process_image(
            str(input_path),
            args.output,
            include_invalid=args.include_invalid
        )
        print(f"Found {result['valid_roofs']} valid roofs out of {result['total_detections']} detections")
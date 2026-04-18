"""
Postprocessing package for rooftop solar suitability estimation.
"""

from .svom_ndvi_filter import SVOM, load_building_polygons, create_svom_from_tiff

__all__ = ['SVOM', 'load_building_polygons', 'create_svom_from_tiff']
__version__ = '1.0.0'
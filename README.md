# Rooftop CNN Detection with SVOM

CNN-based rooftop detection with Southern Vegetation Occlusion Module (SVOM) for Nordic solar suitability estimation.

## Installation

### From PyPI (if published)
```bash
pip install rooftop-cnn-detection
```

### From source
```bash
git clone https://github.com/raphaelattias/rooftop-cnn-detection.git
cd rooftop-cnn-detection
pip install -e .
```

### Development installation
```bash
git clone https://github.com/raphaelattias/rooftop-cnn-detection.git
cd rooftop-cnn-detection
pip install -e ".[dev]"
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- rasterio 1.2+
- shapely 1.8+
- pyproj 3.0+

Install all dependencies:
```bash
pip install -r requirements.txt
```

## SVOM Post-Processing Module

The Southern Vegetation Occlusion Module (SVOM) filters rooftop detections based on southern vegetation obstruction, which significantly impacts solar irradiance in Nordic environments (Finland).

### Quick Start

```python
from postprocessing.svom_ndvi_filter import SVOM
from shapely.geometry import Polygon

# Initialize SVOM with NDVI data
svom = SVOM(
    ndvi_tiff_path='path/to/finland_ndvi_summer_composite.tif',
    crs_epsg=3067,  # Finland CRS
    buffer_distance=50.0,  # meters
    valid_roof_threshold=0.3
)

# Building polygon from detection model
building_polygon = Polygon([(385000, 6670000), (385050, 6670000), 
                            (385050, 6670050), (385000, 6670050)])

# Compute vegetation score
result = svom.compute_vegetation_score(building_polygon)

print(f"Vegetation Score: {result['vegetation_score']}")
print(f"Valid Roof: {result['valid_roof']}")
```

### Output Fields

- `vegetation_score`: S = 1 - (vegetation_pixels / total_southern_pixels), range [0.0, 1.0]
- `vegetation_ratio`: Ratio of vegetation pixels in southern sector
- `total_southern_pixels`: Total pixels in southern sector (azimuth 135°-225°)
- `vegetation_pixels`: Pixels with NDVI > 0.5 (vegetation)
- `valid_roof`: True if vegetation_score >= 0.3

### Integration with Inference Pipeline

```python
from postprocessing.inference_svom_integration import integrate_with_existing_inference

# After rooftop detection
detected_buildings = [...]  # from CNN model

# Add SVOM scores
enriched_buildings = integrate_with_existing_inference(
    detected_buildings=detected_buildings,
    ndvi_tiff_path='path/to/ndvi.tif',
    crs_epsg=3067
)

# Filter valid roofs
valid_roofs = [b for b in enriched_buildings if b['valid_roof']]
```

### Command Line Usage

```bash
# Process single image
python -m postprocessing.inference_svom_inference \
    --ndvi-tiff path/to/ndvi.tif \
    --input path/to/image.png \
    --output results.json

# Process batch
python -m postprocessing.inference_svom_inference \
    --ndvi-tiff path/to/ndvi.tif \
    --input path/to/images/ \
    --output results/
```

## Training

```bash
python run.py
```

## NDVI Data Preparation

For SVOM, use Sentinel-2 summer composite (June-August) from last 3-5 years:

1. **Source**: Copernicus Hub (scihub.copernicus.eu) or Google Earth Engine
2. **Bands**: B4 (Red) and B8 (NIR)
3. **Formula**: NDVI = (B8 - B4) / (B8 + B4)
4. **Resolution**: 10m or 20m
5. **Format**: GeoTIFF with CRS (EPSG:3067 for Finland)

## Project Structure

```
rooftop-cnn-detection/
├── model/              # U-Net model
├── process_data/       # Data loading and preprocessing
├── train/              # Training scripts
├── loss/               # Custom loss functions
├── plots/              # Visualization
├── hyperparameters/    # Hyperparameter selection
├── labelling_tool/     # Image labeling tools
├── postprocessing/    # SVOM post-processing module
│   ├── svom_ndvi_filter.py
│   ├── inference_svom_integration.py
│   └── __init__.py
└── run.py              # Main training script
```

## License

MIT

## Citation

```bibtex
@software{rooftop-cnn-detection,
  title={Rooftop CNN Detection with SVOM},
  author={OverfitTeam},
  year={2020},
  url={https://github.com/raphaelattias/rooftop-cnn-detection}
}
```
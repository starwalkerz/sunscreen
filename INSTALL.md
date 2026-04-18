# Installation Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/raphaelattias/rooftop-cnn-detection.git
cd rooftop-cnn-detection

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 3. Install package in editable mode
pip install -e .

# 4. Verify installation
python -c "from postprocessing.svom_ndvi_filter import SVOM; print('SVOM installed successfully')"
```

## Alternative Installation

### Install with all dependencies manually

```bash
# Clone and navigate to directory
git clone https://github.com/raphaelattias/rooftop-cnn-detection.git
cd rooftop-cnn-detection

# Install dependencies first
pip install -r requirements.txt

# Then install package
pip install -e .
```

### Install only SVOM post-processing module

```bash
# If you already have the base repo and only need SVOM
cd rooftop-cnn-detection
pip install -e .

# Import directly
from postprocessing.svom_ndvi_filter import SVOM
```

## Server Deployment

### Production Server Install

```bash
# Clone
git clone https://github.com/raphaelattias/rooftop-cnn-detection.git

# Install with production dependencies
pip install -e .

# Or use specific Python version
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

### Docker Installation

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app
WORKDIR /app

# Install Python dependencies and package
RUN pip install --no-cache-dir -e .

# Run inference
CMD ["python", "-m", "postprocessing.inference_svom_integration", "--help"]
```

## Verify SVOM Works

```python
from shapely.geometry import Polygon
from postprocessing.svom_ndvi_filter import SVOM

# Create test polygon (example coordinates in EPSG:3067)
test_polygon = Polygon([
    (385000, 6670000),
    (385050, 6670000),
    (385050, 6670050),
    (385000, 6670050)
])

# Initialize (requires valid NDVI GeoTIFF)
svom = SVOM(ndvi_tiff_path='path/to/ndvi.tif', crs_epsg=3067)

# Compute score
result = svom.compute_vegetation_score(test_polygon)
print(f"Vegetation Score: {result['vegetation_score']}")
print(f"Valid Roof: {result['valid_roof']}")
```

## Troubleshooting

### GDAL/rasterio errors on Linux
```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev gdal-bin

# Then reinstall rasterio
pip install --force-reinstall rasterio
```

### pyproj errors
```bash
pip install --force-reinstall pyproj
```

### Import errors
```bash
# Ensure you're in the correct directory
cd rooftop-cnn-detection
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## NDVI Data Preparation

For SVOM to work, you need a Sentinel-2 NDVI summer composite:

1. Download from: https://scihub.copernicus.eu/ or Google Earth Engine
2. Use bands B4 (Red) and B8 (NIR)
3. Calculate: NDVI = (B8 - B4) / (B8 + B4)
4. Export as GeoTIFF in EPSG:3067 (Finland) or your local CRS

## Support

For issues, check: https://github.com/raphaelattias/rooftop-cnn-detection/issues
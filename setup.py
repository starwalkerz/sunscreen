[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rooftop-cnn-detection"
version = "1.0.0"
description = "CNN-based rooftop detection with SVOM post-processing for Nordic solar suitability"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "OverfitTeam", email = "team@example.com"},
]
keywords = ["deep-learning", "remote-sensing", "solar", "rooftop", "segmentation", "nordic"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: GIS",
]

dependencies = [
    "numpy>=1.20.0",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "scikit-learn>=0.24.0",
    "opencv-python>=4.5.0",
    "matplotlib>=3.3.0",
    "rasterio>=1.2.0",
    "shapely>=1.8.0",
    "pyproj>=3.0.0",
    "scikit-image>=0.18.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=21.0",
    "flake8>=3.9.0",
]

[project.urls]
Homepage = "https://github.com/raphaelattias/rooftop-cnn-detection"
Repository = "https://github.com/raphaelattias/rooftop-cnn-detection"

[tool.setuptools]
packages = ["model", "process_data", "loss", "plots", "train", "hyperparameters", "labelling_tool", "postprocessing"]

[tool.setuptools.package-data]
model = ["*.pt"]
plots = ["**/*"]
documents = ["**/*"]
figures = ["**/*"]
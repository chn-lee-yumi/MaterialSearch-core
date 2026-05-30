# MaterialSearch Core

Core Python library for [**MaterialSearch**](https://github.com/chn-lee-yumi/MaterialSearch) project.

TODO: Not compatible with new PyTorch / Transformers version yet.

## Installation

```bash
pip3 install materialsearch-core
```

## Usage

### Scan

```python
import os

os.environ['ASSETS_PATH'] = r'your_path' # Use environment variable to set the scan path. More options can see config.py
# Remember to set the environment variables before importing materialsearch_core

from materialsearch_core.scan import scanner

scanner.init()  # Initialise the scanner
scanner.scan()  # Scan
print(scanner.get_status()) # Check scan status
```

### Search

```python
# Search images by text
from materialsearch_core.search import search_image_by_text_path_time
result = search_image_by_text_path_time("Flower")
print(result)

# Search similar images by one image (image path or imageId)
from materialsearch_core.search import search_image_by_image
result = search_image_by_image(r"your_image_path")
# result = search_image_by_image(1)
print(result)

# Search videos by text
from materialsearch_core.search import search_video_by_text_path_time
result = search_video_by_text_path_time("Flower")
print(result)

# Search videos by one image (image path or imageId)
from materialsearch_core.search import search_video_by_image
result = search_video_by_image(r"your_image_path")
# result = search_video_by_image(8)
print(result)
```

## Building and Distributing

Remember to update the version number in `materialsearch_core/__init__.py` and `pyproject.toml` before building.

Install dependencies before building:

```bash
pip3 install -U build twine packaging
````

### Test Environment

```bash
python3 -m build
python3 -m twine upload --repository testpypi dist/* --verbose
python3 -m pip install -U --force-reinstall --index-url https://test.pypi.org/simple/ --no-deps materialsearch-core
# or
python3 -m pip install --force-reinstall --no-deps dist/materialsearch_core-*.whl
```

### Production Environment

```bash
python3 -m build
python3 -m twine upload dist/*
python3 -m pip install materialsearch-core
```

Or use GitHub Actions to build and publish. Create a release will trigger the workflow.

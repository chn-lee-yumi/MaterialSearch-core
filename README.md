# MaterialSearch Core

Core Python library for [**MaterialSearch**](https://github.com/chn-lee-yumi/MaterialSearch) project.

## Installation

```bash
pip3 install materialsearch-core
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
python3 -m pip install -U --index-url https://test.pypi.org/simple/ --no-deps materialsearch-core
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

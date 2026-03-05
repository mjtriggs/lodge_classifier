Running Cleaning:

```
lodge-normalise-names \
  --input data/input/reduced-data-2.csv \
  --output data/processed/lodges_normalised.csv
```

Run Sample:

```
lodge-classify --input sample_input.csv --manual data/manual/manual_curation_template.csv
```

```
lodge-classify --input data/test_files/input/lodge_name_sample_04_n100_seed12345.csv --manual data/manual/manual_curation_template.csv
```

```
deactivate 2>/dev/null || true
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install --only-binary=:all: numpy==2.4.2 pandas
python -m pip install -e ".[dev]"   # or your normal install route
```
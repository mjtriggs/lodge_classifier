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
lodge-classify --input data/test_files/input/lodge_name_sample_03_n100_seed12345.csv --manual data/manual/manual_curation_template.csv
```
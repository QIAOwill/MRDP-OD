# Synthetic sample data

All files in this directory are deterministic, fully synthetic examples created
only to demonstrate the input schema and exercise the code. They do not contain
or approximate the study's confidential source observations. The three directory
names are retained because they are protocol identifiers used by the code.

Each region contains five tab-separated UTF-8 files: OD flow, city-static,
city-dynamic, OD-pair-static, and OD-pair-weather data. Run
`python scripts/generate_sample_data.py` to reproduce them.

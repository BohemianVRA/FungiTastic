# 🌦️ Climate Time Series

## Overview
Each observation comes with **20 years of historical climate data** for its location, supporting distribution modeling and temporal studies.

---

## What’s Included

- **Monthly values:** Mean, min, max temperature, total precipitation.
- **Bioclimatic variables:** 19 features (e.g., seasonality), averaged over 1981–2010.
- **Coverage:** Data up to 2020 for all observations; newer obs may have missing final years.

---

## Data Source

- Extracted from CHELSA and other open-access climate datasets.

---

## Usage Notes

- Format: time series per observation (CSV or torch tensor).
- Missing years for 2020–2024 observations are noted in the data.

---

## Learn More

- See [Metadata](metadata.md) for site locations and environment.
- See [Satellite Data](satellite.md) for environmental context.

# 🛰️ Satellite Data

## Overview
FungiTastic provides **multi-band satellite image patches** for nearly all European observation sites.  
These images capture the environmental context around the specimen.

---

## Data Details

- **Type:** 64×64 pixel image patches, 10m spatial resolution.
- **Bands:** Red, Green, Blue, Near-infrared (NIR), Elevation, Land cover.
- **Source:** Sentinel-2A, Ecodatacube, ASTER, ESA WorldCover.

---

## File Format

- Provided as torch tensors with shape `[6 × 64 × 64]` per patch.
- Each observation is centered on its location.

---

## Usage Notes

- Useful for remote sensing research, species distribution modeling, and combining with in-situ photos.

---

## Learn More

- [Metadata](metadata.md) for location and land cover info.
- [Climate Data](climate.md) for long-term weather variables.

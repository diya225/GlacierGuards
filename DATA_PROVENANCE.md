# Data provenance

## Label source: GLO / GLOID lake polygons (CC BY 4.0)
Rawlins, L., Watson, C. S., Bhambri, R., Khadka, N., & Chand, M. B. *Datasets supporting - Glacial Lake Observatory (GLO): A dataset of
glacial lakes in Nepal and transboundary catchments (2017-2024)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19370146
(concept DOI 10.5281/zenodo.17802333; the record README cites the concept DOI with year 2025. Confirm DOI and year before publication).
- Files used: `S1_20172024_NTB_GLOID_v1.02.gpkg`, `S2_20172024_NTB_GLOID_v1.02.gpkg` (per-year polygons; S1 window 1 Jul-30 Aug, S2 window 1 May-30 Nov).
- Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ . Attribution required.
- Polygons are model-derived (DeepLabV3); the dataset reports F1 of about 0.82 (S1) and 0.92 (S2) for 2020. Masks here are therefore weak labels, not hand-digitised truth.
- Acknowledgements requested by the authors: GLO-FHICC project, UKRI Future Leaders Fellowship MR/Y016564/1, Sentinel-1, Sentinel-2, ALOS AW3D30.
- Processing: polygons with TS_OUTLIER = TRUE, or relative area uncertainty above the sensor 75th percentile, are excluded as target labels;
  lake-years where S1 and S2 polygons disagree (IoU < 0.5) are dropped (see reports/mask_manifest.csv).

## Imagery (accessed via Google Earth Engine)
- Sentinel-1 GRD (COPERNICUS/S1_GRD): contains modified Copernicus Sentinel data 2017-2024, processed by ESA. Median composite, Enhanced-Lee filtered.
- Sentinel-2 L2A harmonized (COPERNICUS/S2_SR_HARMONIZED): contains modified Copernicus Sentinel data 2017-2024. SCL-masked median composite (clouds, shadow, cirrus, saturation removed; snow kept).
- Copernicus DEM GLO-30 (COPERNICUS/DEM/GLO30_2024_1): (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the EU and ESA; all rights reserved.

## Not used
South Lhonak (no GLOID polygon) is an inference-only lake and is excluded from every training, validation and test split.

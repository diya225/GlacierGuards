# GLOF Early Warning System: Section 7 Completed
### Manual Tasks: Decisions, Procedures and Reference Data

This document completes Section 7 of `GLOF_EWS_Project_Pipeline.md`. Each task below gives a recommendation, the procedure to follow, and the evidence behind it. Items I could not verify are marked **[VERIFY]**. Do those checks yourself before relying on the value.

## Task overview

| # | Task | Output | Blocks which milestone |
|---|---|---|---|
| 1 | Finalize lake cohort | `configs/lakes.yaml` | M2 (ingestion) |
| 2 | Digitize ground-truth masks | ~100 binary masks | M4, M5 (segmentation) |
| 3 | Mark moraine-crest points | `crest_points` in `lakes.yaml` | M7 (freeboard) |
| 4 | Research hindcast events | Event table + success criterion | M8 (evaluation) |
| 5 | Create accounts and keys | Local `.env` | M2 (ingestion) |
| 6 | Decide risk-score weights | `configs/pipeline.yaml` | M7 (risk score) |
| 7 | Licensing and attribution | `provenance.csv` | Final report |
| 8 | Compute budget | GPU plan | M5, M6 (training) |

**Suggested order:** Tasks 5 and 1 first (they unblock ingestion), then Tasks 2 and 3 in parallel while ingestion runs, then Task 4. Tasks 6, 7 and 8 can be done at any point before Milestone 7.

**Correction to earlier guidance:** an earlier message in this project gave the area-to-volume formula as V = 0.0365 × A^1.35. I could not confirm it in any source. Do not use it. The verified relations are in Task 6.

---

## Task 1: Finalize the lake cohort

### Recommendation

Use 8 core lakes plus 1–2 quiet control lakes. Restrict the cohort to **moraine-dammed** lakes (see Task 6 for why).

| # | Lake | Region | Role in the project |
|---|---|---|---|
| 1 | South Lhonak | Sikkim, India | Main hindcast case. Moraine-dammed, about 5200 m, at 27°54.741'N, 88°11.857'E. Burst on 4 October 2023. |
| 2 | Gongbatongsha Co | Poiqu basin, Tibet | Second burst inside the Sentinel-1 era (5 July 2016). A small lake that drained completely, so it tests the limits of an area-trend signal. |
| 3 | Imja Tsho | Khumbu, Nepal | Coordinates 27.898°N, 86.928°E. Artificially lowered by about 3.4 m in 2016, which gives a known level drop to test against. |
| 4 | Tsho Rolpa | Rolwaling, Nepal | Altitude about 4,580 m, roughly 90–100 million m³ stored. Lowered by about 3 m in 2000. |
| 5 | Lower Barun | Nepal | ICIMOD/UNDP critical group. |
| 6 | Thulagi | Nepal | ICIMOD/UNDP critical group. |
| 7 | Lumding Tsho | Nepal | ICIMOD/UNDP critical group. |
| 8 | Hongu 2 / Chamlang South | Nepal | ICIMOD/UNDP critical group. |
| C1–C2 | Quiet controls | Choose from ICIMOD Rank III | Never burst. Needed to measure false alarms. |

Nepal's government has named Thulagi, Lower Barun, Lumding and Hongu as the four highest-risk lakes, which supports including them.

### Procedure

1. Download the ICIMOD/UNDP inventory (47 potentially dangerous lakes ranked into three categories) and the GLIMS outlines. **[VERIFY]** the license terms on each download page.
2. For every lake, load the outline in QGIS and compute the centroid. Take the altitude from the Copernicus GLO-30 DEM at the centroid.
3. Coordinates above are verified only for South Lhonak and Imja. For all other lakes, use the centroid from the inventory or GLIMS outline instead of a remembered number.
4. Pick the two controls from Rank III lakes of similar size and elevation to your core lakes, with no recorded outburst.

### `lakes.yaml` entry template

```yaml
- id: south_lhonak
  name: South Lhonak
  lat: 27.9124        # from 27°54.741'N; confirm against outline centroid
  lon: 88.1976        # from 88°11.857'E; confirm against outline centroid
  alt_m: 5200
  dam_type: moraine
  role: hindcast_positive   # hindcast_positive | monitored | control
  burst_date: 2023-10-04    # null if none
  source: ICIMOD/GLIMS + published case studies
  crest_points: []          # filled in Task 3
```

### Limits to state in the report

With two bursts in the Sentinel era, your hindcast validation is effectively n = 2. Say this plainly in the write-up.

---

## Task 2: Ground-truth labeling plan

### Amount and allocation

- About 12 dates per core lake and 6 per control lake, roughly **100 masks**.
- At about 15–20 minutes per mask (with a pre-drafted outline) this is roughly 30 person-hours. Split it four ways.
- Keep whole lakes out for testing, to match the leave-one-lake-out design in Milestone 8.

### Choosing dates

| Season | Typical months | Notes |
|---|---|---|
| Pre-monsoon | Apr–May | Often the best clear-sky optical. |
| Monsoon | Jun–Sep | Optical mostly cloudy. Label from SAR and check against the nearest clear optical date. |
| Post-monsoon | Oct–Nov | Best optical conditions. |
| Winter | Dec–Feb | Ice and snow cover. See the rule below. |

Spread dates across several years per lake, not just one.

### Labeling protocol

1. Pick a cloud-free Sentinel-2 image for the target date.
2. Pair it with the Sentinel-1 pass within about ±3 days.
3. Generate a draft outline with an NDWI threshold, then hand-correct the polygon in QGIS. This is much faster than tracing from scratch.
4. Export the polygon with `qgis_export_helper.py` as a binary mask on the same grid as the stacked tensors.
5. **Winter rule:** decide once, as a team, and apply it to everyone. Recommended: label the full lake outline even when frozen or snow-covered, and include some winter dates so the model learns the difference. NDWI fails on ice, so winter drafts need more manual work.

### Quality check

Have a second person independently re-label about 10% of the masks. Compute IoU between the two versions. This is your annotation-noise ceiling. Quote it next to the model's IoU in the report.

### Data-density caveat

Sentinel-1B stopped working in December 2021, and Sentinel-1C entered service afterward. Revisit frequency therefore drops in the middle of your record. **[VERIFY]** the actual ASF coverage per lake before promising a date spacing to the temporal model.

---

## Task 3: Moraine-crest reference points

### Definitions

- **Freeboard** is the vertical distance between the lake surface and the lowest point on the dam crest. It reflects the minimum wave height needed for overtopping.
- Dam stability also depends on the width-to-height ratio of the dam, its downstream slope, its material, and whether it has an ice core.

### Procedure (per lake)

1. Load Copernicus GLO-30 in QGIS with a hillshade and 5 m contours.
2. Identify the terminal moraine on the downstream side, where the outlet stream exits.
3. Trace the crest as a polyline.
4. Place 5–10 points along it. **Always include the lowest point.**
5. Record lat, lon and DEM elevation for each point.
6. Cross-check positions against high-resolution imagery and, where possible, SRTM.
7. Record dam height (crest minus downstream toe elevation) and dam width, so a width-to-height ratio can be computed.

### Template

```yaml
crest_points:
  - {lat: 0.0, lon: 0.0, elev_m: 0.0, is_lowest: true}
  - {lat: 0.0, lon: 0.0, elev_m: 0.0, is_lowest: false}
dam:
  height_m: 0.0
  width_m: 0.0
  has_outlet_channel: false
```

### Two problems to plan for

1. **Outlet lakes.** A study of Nepal lakes notes that lakes with an outlet channel have freeboard of zero by definition. For those lakes, vertical freeboard carries little information.
2. **DEM resolution.** In my assessment, a 30 m DEM is coarse relative to dam crests that are tens to hundreds of metres wide, so vertical values are noisy.

### Recommended proxy design

| Signal | Role | Computation |
|---|---|---|
| Horizontal gap | Primary | Distance from segmented shoreline to the nearest crest point. |
| Vertical freeboard | Secondary | Lowest crest elevation minus the median DEM elevation along the shoreline. |

Treat both as **relative trends**, not absolute safety margins. For lakes with `has_outlet_channel: true`, move the freeboard weight onto the horizontal gap.

---

## Task 4: Hindcast events and precursor evidence

### Event 1: South Lhonak, 4 October 2023 (main case)

| Item | Detail |
|---|---|
| Lake type | Moraine-dammed, elevation about 5200 m. |
| Pre-event growth | Lake length grew about 575 m in 2023, reaching 2.946 km, per Sentinel-1 backscatter analysis. |
| Volume trend | Average 2015–2023 volume was about 102.4 million m³, which is 55.6% above the 2014–2016 figure of 65.8 million m³. |
| Field volume | Storage capacity measured in surveys (2014, 2016) was 65.81 ± 2.5 million m³. |
| Deformation | Sentinel-1 InSAR (Jan 2021–Mar 2024) showed significant deformation in the moraine around the lake before the GLOF. |
| Trigger | A landslide dumped about 38.31 million m³ of debris into the lake. |
| Impact | 24 fatalities and more than 70 people missing. |

### Event 2: Gongbatongsha Co, 5 July 2016

| Item | Detail |
|---|---|
| Setting | Small lake in the Poiqu basin, Tibet, described as inconspicuous. |
| Chain of events | Debris flow into the lake from the headwall slope, lake overfill, frontal moraine breach, then transformation into a debris flow that hit Nepal. |
| Impact | The flood reached about 40 km downstream, with total economic loss above US$70 million. |
| Modeling note | The lake drained completely, and slow area growth is unlikely to have been a strong precursor. This makes it a good test of your limits. |

### Intervention checks

| Lake | Intervention | Test |
|---|---|---|
| Imja Tsho | Lowered about 3.4 m in 2016 | Does volume or level proxy drop as expected? |
| Tsho Rolpa | Lowered about 3 m in 2000 | Pre-Sentinel, so use Landsat only. |

### How to frame the hindcast

Both bursts were triggered by sudden mass movements into the lake. Your area, volume and freeboard signals measure **susceptibility**, not trigger timing.

- **Success criterion (recommended):** the risk score for the lake sits in the top tier of the cohort for the months before the event, and stays above the quiet controls.
- **Do not claim:** that the system predicted the event date.
- **Report:** lead time above threshold, false-alarm rate on the controls, and results for both events even if Gongbatongsha is a miss.

Pre-2014 events (for example Dig Tsho 1985 and Cirenmaco 1981) predate Sentinel-1 and can only be examined with Landsat.

---

## Task 5: Accounts and keys

| Order | Account | What to produce | Used by |
|---|---|---|---|
| 1 | NASA Earthdata Login | Username and password; confirm HyP3 credit balance | `asf_hyp3_fetch.py` |
| 2 | Google Earth Engine | Cloud Project ID; service account JSON key registered with Earth Engine | `gee_fetch.py` |
| 3 | Copernicus Data Space | OAuth client ID and secret (only if not using GEE for Sentinel) | Optional |
| 4 | USGS EROS | Login (only if pulling Landsat outside GEE) | `landsat_fetch.py` |
| 5 | GitHub / Docker Hub | Optional | Version control, image hosting |

### Google Earth Engine steps

1. Register for non-commercial use and confirm your eligibility under the current terms **[VERIFY]**.
2. Create a Google Cloud Project.
3. Enable the Earth Engine API for that project.
4. Create a service account and download its JSON key.
5. Register the service account with Earth Engine. **[VERIFY]** the current signup flow, since Google changes it.

### Secrets hygiene

- Put every secret in a local `.env` file, listed in `.gitignore`.
- Commit only `.env.example` with placeholder values.
- Each teammate uses their own keys.
- Never paste keys into a chat with an AI agent or into a commit.
- Rotate any key that has been exposed.

`.env.example`:

```
GEE_PROJECT_ID=
GEE_SERVICE_ACCOUNT_JSON=
EARTHDATA_USERNAME=
EARTHDATA_PASSWORD=
CDSE_CLIENT_ID=
CDSE_CLIENT_SECRET=
```

---

## Task 6: Risk-score weighting

### Restrict the cohort to moraine-dammed lakes

Moraine and ice dams fail by different mechanisms:

- Moraine dams fail by trigger-driven overtopping, for example an ice avalanche or rockfall generating a wave.
- Ice dams fail by hydrostatic flotation, and they often reseal and refill.
- Geometric predictors like dam width-to-height ratio are meaningless for ice dams.
- All 21 of ICIMOD's dangerous Nepal lakes are moraine-dammed.

Limit the cohort to moraine dams and state this in the report. Do not build separate scoring branches.

### Starting weights (`configs/pipeline.yaml`)

| Component | Weight | Rationale |
|---|---|---|
| Area-growth anomaly (z-score vs the lake's own seasonal baseline) | 0.40 | Lake expansion rate is a standard hazard factor, alongside freeboard, dam crest width and mass-movement likelihood. |
| Freeboard / horizontal-gap decline rate | 0.35 | Reflects shrinking dam margin. |
| Volume anomaly | 0.25 | Kept lowest because volume is derived from area by a power law, so it partly double-counts the first term. |

```yaml
risk:
  weights: {area_growth: 0.40, freeboard: 0.35, volume: 0.25}
  outlet_lake_fallback: {area_growth: 0.45, freeboard: 0.30, volume: 0.25}  # gap-only freeboard
  ewma_lambda: 0.2
  cusum: {k_sigma: 0.5, h_sigma: 4.5}
```

The EWMA and CUSUM values are textbook defaults, not from the reference papers. Justify them in the report.

### Do not tune to the hindcast

With two burst events, tuning the weights until the hindcast looks good will overfit. Instead:

1. Run a grid over plausible weight combinations.
2. Report how stable the cohort ranking and the South Lhonak result are across the grid.
3. Pick the final weights on literature grounds, and show the grid as a robustness check.

### Area-to-volume scaling (Milestone 7)

| Source | Relation | Notes |
|---|---|---|
| Huggel et al. (2002) | V = 0.104 · A^1.42 | Widely applied. |
| Wang et al. (2012), Himalaya | V = 0.0354 · A^1.3724 | Depth relation D = 0.087 · A^0.434 is published alongside it. |

- **[VERIFY]** the units (area and volume) in each original paper before coding. Unit mix-ups are the most common bug here.
- **Sanity test:** run your implementation on South Lhonak and compare with the field-surveyed 65.81 ± 2.5 million m³. You should land in the same order of magnitude.
- **Known weakness:** volume-area relations can give misleadingly high R² values because volume depends on area, and studies recommend using more than one approach.
- **Practice:** compute two relations, report volume as a range, and use only the relative change in the risk score.

---

## Task 7: Licensing and attribution

Maintain `provenance.csv` with these columns: `dataset, product, dates, access_route, license, attribution_text, citation`.

| Source | License status | Action |
|---|---|---|
| Sentinel-1 / Sentinel-2 | Free and open (Copernicus) | Use the standard acknowledgement: contains modified Copernicus Sentinel data [year], with "processed by ESA" or "processed by ASF/HyP3" as relevant. |
| Landsat | USGS public domain | Cite USGS/NASA. |
| Copernicus DEM GLO-30 | Free, with mandatory acknowledgement wording | **[VERIFY]** Copy the exact statement from the current product guide. |
| Google Earth Engine | Free for non-commercial research | Cite Gorelick et al. (2017). **[VERIFY]** the current terms. |
| ICIMOD and GLIMS inventories | Not verified | **[VERIFY]** the license on each download page and cite each report. |
| HyP3 / ASF | Free with Earthdata login | Follow ASF's current credit wording. **[VERIFY]** |

Also record for each dataset: the exact acquisition dates used and the date you downloaded it, for the data provenance appendix.

---

## Task 8: Compute budget

### Estimate (rough, my own numbers)

| Item | Estimate |
|---|---|
| Dataset size | ~10 lakes × ~100 dates × a handful of 256×256 tiles, so thousands of tiles. |
| One U-Net + EfficientNet-B3 run | Probably 30–60 minutes on a T4-class GPU. |
| Runs needed | About 10 leave-one-lake-out folds, 3 ablations, and temporal-model training. |
| Total | On the order of 20–30 GPU-hours. |

### Where to run

1. **Kaggle notebooks (primary):** offers a weekly free GPU quota, roughly 30 hours. **[VERIFY]** the current limit.
2. **Colab free tier (backup):** sessions can end unpredictably.
3. Only pay for Colab Pro or a cloud GPU if you hit these limits.

### Practical tips

- Save checkpoints every epoch to Drive or Kaggle output.
- Use mixed precision training.
- Run ablations with fewer epochs than the final model.
- Let Google Earth Engine do the heavy preprocessing so you are not downloading terabytes.

### Technical note

With a 10-channel input, `segmentation_models_pytorch` adapts the first convolution of the pretrained encoder. ImageNet pretraining helps less than it would for 3-channel RGB. Mention this in the write-up and consider a run without pretrained weights as an extra ablation.

---

## Final checklist

- [ ] Accounts created; `.env` populated and gitignored (Task 5)
- [ ] `lakes.yaml` filled with 8 core lakes + 1–2 controls, coordinates from real outlines (Task 1)
- [ ] ~100 masks digitized; 10% double-labeled; winter rule agreed (Task 2)
- [ ] Crest points, dam height and width recorded for every lake (Task 3)
- [ ] Hindcast events tabulated; success criterion written down before running (Task 4)
- [ ] Weights set from literature; weight-grid robustness check planned (Task 6)
- [ ] Two area-volume relations implemented and sanity-tested (Task 6)
- [ ] `provenance.csv` started; acknowledgement wording copied from official sources (Task 7)
- [ ] GPU plan chosen; checkpointing enabled (Task 8)

---

## Sources

**South Lhonak**
- Sequence of events that led to the South Lhonak lake outburst flood, Scientific Reports: https://www.nature.com/articles/s41598-026-35895-7
- South Lhonak Glacial System: cascade investigation using satellite remote sensing (EGU abstract): https://ui.adsabs.harvard.edu/abs/2024EGUGA..2618788S/abstract
- Retrospective analysis of the South Lhonak GLOF using InSAR and optical images, Remote Sensing 2024: https://doi.org/10.3390/rs16132307

**Gongbatongsha**
- Transition of a small Himalayan GLOF to a giant transborder flood and debris flow, Scientific Reports: https://www.nature.com/articles/s41598-022-16337-6

**Inventories and cohort**
- ICIMOD, new glacial lake inventory report (47 potentially dangerous lakes): https://www.icimod.org/new-glacial-lake-inventory-report-released-47-potentially-dangerous-glacial-lakes-ranked/
- Imja Tsho: https://en.wikipedia.org/wiki/Imja_Tsho
- Glacial lake outburst flood (Tsho Rolpa details): https://en.wikipedia.org/wiki/Glacial_lake_outburst_flood
- Nepal's most dangerous glacial lake, PreventionWeb: https://www.preventionweb.net/news/nepal-worries-about-its-most-dangerous-glacial-lake
- Six dangerous glacial lakes in the Nepalese Himalaya (lowering of Imja and Rolpa): https://www.researchgate.net/publication/331934661_The_state_of_six_dangerous_glacial_lakes_in_the_Nepalese_Himalaya

**Dam stability and hazard criteria**
- Lake volume and potential hazards of moraine-dammed glacial lakes, The Cryosphere: https://tc.copernicus.org/articles/17/591/2023/
- Rounce et al., a new remote hazard and risk assessment framework for glacial lakes in the Nepal Himalaya: https://scholar.colorado.edu/downloads/q237hs684
- Emmer and Vilímek, lake and breach hazard assessment for moraine-dammed lakes: https://nhess.copernicus.org/articles/13/1551/2013/nhess-13-1551-2013.pdf
- Moraine vs ice dam failure mechanisms and Nepal dam-type statistics: https://arxiv.org/pdf/2608.12422

**Area-volume scaling**
- Cook and Quincey (2015), Estimating the volume of Alpine glacial lakes: https://esurf.copernicus.org/articles/3/559/2015/esurf-3-559-2015.pdf
- Muñoz et al. (2020), table of published relations including Wang et al. (2012): https://researchgate.net/publication/339023984
- In-situ bathymetry of four western Himalaya lakes, Journal of Glaciology: https://www.cambridge.org/core/journals/journal-of-glaciology/article/insitu-bathymetry-and-volume-estimation-of-four-glacial-lakes-in-western-himalaya/2AFBFBBA22F767EFAE288E368C25318A

# Milestone 6: temporal analysis of six lakes (results and notes)



## 1. Summary
- **What was done:** a lake-specific area series and a drop detector, built for 6 lakes with the final segmentation model `models/seg_fused_test.pt` (never trained on any of them): four outburst lakes (South Lhonak, Lower Ngole, Jinwuco, Chamai) and two quiet lakes (Imja Tsho, Thulagi). The plan listed five lakes; Chamai is the sixth and is a miss.
- **Outburst events detected:** 2 of 4 with the plain detector (South Lhonak, Lower Ngole), 3 of 4 with the exploratory per-track mode (adds Jinwuco). Chamai is missed.
- **Detection is not prediction:** every detection is at the first Sentinel-1 pass after the event.
- **The deep forecaster (ConvLSTM, South Lhonak only) was no better than "same as last scene"**; it is not used.
- **Status:** exploratory. The thresholds and extra rules were chosen after seeing these lakes (section 5); the journal-hardening work is deferred (section 8).

## 2. Method
- **Series:** one 10-channel stack per **Sentinel-1 date** (`src/preprocessing/stack_by_s1.py`); optical channels come from a clear Sentinel-2 scene within 5 days when one exists, otherwise the model runs radar-only. (The older `coregister_stack.py` stacks one image per Sentinel-2 date, which left Imja Tsho with 11 dates in 7 years.) South Lhonak was rebuilt this way on 2026-11-06; its earlier 98-date results are in `reports/archive/south_lhonak_burst_detection_s2dates.csv` (noise 0.081 km2, flagged 2023-10-06, 0 of 61 false alarms).
- **Lake region (ROI):** the independent manual outline from the Third Pole inventory plus a margin (400 m for lakes of about 0.5 km2 and larger, 100 m for smaller), with other inventory lakes cut out; for outburst lakes the last snapshot before the event (`scripts/make_lake_roi.py --tp-year`). Area is measured only inside the ROI.
- **Detector** (`src/training/burst_detection.py`): flags a drop larger than 4 robust standard deviations (MAD) between scenes at most 15 days apart, with the deviation estimated on pre-event pairs. Extras: a **persistence confirmation** (next scene stays at least half the drop below the pre-drop level), a **per-track mode** (`--per-track`, section 5) and a guard `--min-sigma` that reports NOT EVALUABLE when the noise estimate is degenerate.
- **Commands:** `scripts/fetch_lake_series.sh LAKE Y0 Y1` (set `END_MD=12-01` for October events), then `scripts/build_lake_series.sh LAKE CHECKPOINT "ROI_ARGS" BURST_DATE|none`, then `python scripts/summarize_burst_detection.py` (writes `reports/m6_multi_lake_summary.csv`). Lakes need an entry in `configs/lakes.yaml`.

## 3. Results table (`reports/m6_multi_lake_summary.csv`)

Plain detector (all Sentinel-1 tracks mixed):

| Lake | Role | Dates | Median area (km2) | Manual outline (km2) | Normal change (km2) | Smallest detectable drop | False alarms (with persistence) | Detection |
|---|---|---|---|---|---|---|---|---|
| South Lhonak | burst 2023-10-04 | 175 | 1.47 | 1.67 | 0.140 | 0.56 km2 (38% of area) | 2 of 158 (0) | 2023-10-07 |
| Lower Ngole, Thame | burst 2024-08-16 | 304 | 0.032 | 0.045 | 0.006 | 0.024 km2 (76%) | 2 of 277 (0) | 2024-08-19 |
| Jinwuco | burst 2020-06-26 | 122 | 0.50 | 0.556 | 0.151 | 0.60 km2 (above the lake area) | 0 of 72 (0) | none |
| Chamai | burst 2021-10-05 | 175 | 0.09 | 0.219 | 0.051 | 0.20 km2 (above the lake area) | 0 of 131 (0) | none |
| Imja Tsho | quiet | 178 | 1.56 | 1.72 | 0.078 | 0.31 km2 (20%) | 2 of 171 (2) | none (no event) |
| Thulagi | quiet | 265 | 0.85 | 0.97 | 0.168 | 0.67 km2 (79%) | 0 of 258 (0) | none (no event) |

Per-track mode (exploratory; each scene compared with the previous scene of the same orbit track):

| Lake | Normal change (km2) | Smallest detectable drop | False alarms | Detection |
|---|---|---|---|---|
| South Lhonak | 0.019 | 5% of area | 6 of 144 | 2023-10-07 (36 sigma) |
| Lower Ngole, Thame | 0.005 | 58% | 3 of 255 | 2024-08-19 (5.4 sigma) |
| Jinwuco | 0.022 | 18% | 2 of 64 | 2020-07-03 (12 sigma) |
| Chamai | 0.020 | 85% | 8 of 120 | 2021-10-25 (not a usable detection) |
| Imja Tsho | 0.025 | 6% | 7 of 160 | none (no event) |
| Thulagi | 0.020 | 9% | 8 of 241 | none (no event) |

## 4. The lakes

### 4.1 South Lhonak, Sikkim (outburst 2023-10-04): detected
- **Lake:** 27.912 N, 88.198 E, moraine-dammed, Third Pole inventory lake 4272 (1.67 km2 in 2022). No GLOID polygon exists for it, so it was never in any training split.
- **Series:** 175 Sentinel-1 dates, 2017 to 2023.
- **Result:** the plain detector flags 2023-10-07 (drop 0.78 km2, 5.8 sigma), 3 days after the burst date, confirmed by persistence; 2 of 158 pre-event false alarms (0 confirmed). Per-track mode flags the same date at 36 sigma.
- **Notes:** the earlier Sentinel-2-date series flagged 2023-10-06 at 8.6 sigma with 0 of 61 false alarms; the difference is the series definition, not the lake. The model's area inside the ROI is 1.34 to 1.55 km2 before the burst (manual outline 1.67 km2; IoU 0.83 to 0.97 on 2022 dates). The ConvLSTM forecaster was no better than "same as last scene" (error 0.084 vs 0.083 km2) and showed no pre-burst anomaly. Unexplained: after the burst the area returns to about 1.4 km2 within seven weeks (one published source reports a 26% reduction; we see about 43% at the first pass). A possible August to September 2023 rise (1.68 km2 vs 1.46 ± 0.07 km2 in five earlier years) is exploratory only.

### 4.2 Lower Ngole lake, Thame, Khumbu (outburst 2024-08-16): detected
- **Lake:** 27.828 N, 86.573 E, moraine-dammed, 4,754 m. GLOID lake `GLO_86.57281_27.82796` (0.048 km2 in 2017 to 2023, 0.013 km2 in 2024); HMAGLOFDB lists it as "Tomutse Lake" (0.045 km2, volume 340,000 m3); inventory lake 4556. ICIMOD calls the event "Thyanbo"; that this is the same lake is not verified.
- **Series:** 304 Sentinel-1 dates, 2017 to 2024 (12-day repeat only since 2022, when Sentinel-1B ended).
- **Result:** area about 0.03 km2 before and 0.007 km2 after; flagged 2024-08-19 (drop 0.028 km2, 4.5 sigma), 3 days after, confirmed. 2 of 277 false alarms (2020-07-12, 2021-10-22), both single-scene dips removed by the persistence rule.
- **Notes:** the model sees about 0.03 km2 against the 0.045 km2 outline (about 30% low). Only a near-total drainage is detectable at this size. The neighbouring upper Ngole lake was not evaluable (section 6).

### 4.3 Jinwuco, Yigong Zangbo basin, south-east Tibet (outburst 2020-06-26): detected in per-track mode only
- **Lake:** 30.356 N, 93.631 E, moraine-dammed, 4,447 m, inventory lake 4880 (0.556 km2 in 2018, 0.252 km2 in 2020). HMAGLOFDB event 508, volume 9.5 million m3 (Zheng et al. 2021). Outside the training region.
- **Series:** 122 Sentinel-1 dates, 2017 to 2021 (two tracks: 77 descending, median 0.41 km2; 143 ascending, median 0.51 km2).
- **Result:** the area steps from about 0.5 km2 (2017 to 2019) to about 0.2 to 0.25 km2 from July 2020 and stays there. The plain detector misses it: the alternating tracks make the noise 0.151 km2, so the 0.60 km2 threshold is above the lake area. Per-track mode flags 2020-07-03 (drop 0.26 km2, 12 sigma) with 2 of 64 false alarms (2019-08-09, 2019-10-08).
- **Notes:** timing is uncertain. The descending scene of 2020-06-28 (two days after the database date) still reads 0.538 km2 and the ascending scene of 2020-07-03 reads 0.223 km2, so the satellite puts the drop between 28 June and 3 July. The database date may be a few days early or the drainage was not instantaneous; the 7-day gap is not detector delay.

### 4.4 Chamai, Langqen Zangbo basin, western Tibet (outburst 2021-10-05): missed
- **Lake:** 31.915 N, 78.841 E, moraine-dammed, 5,600 m, inventory lake 3132 (0.229 km2 in 2020, 0.063 km2 in 2022). HMAGLOFDB event 731, volume 0.6 million m3 (Ma et al. 2024). Outside the training region.
- **Series:** 175 Sentinel-1 dates, 2017 to 2022 (148 Sentinel-2 scenes, a dry region).
- **Result:** not detected. Per-track mode flags 2021-10-25, 20 days late, inside the normal autumn decline, with 8 of 120 false alarms: not a usable detection.
- **Notes:** at 5,600 m the lake freezes every autumn (September 0.14 to 0.20 km2, October 0.08 to 0.18, November 0 to 0.08 in 2017 to 2021), so the drop is gradual and hidden in the seasonal decline. The model also sees only about half of the outline (0.09 vs 0.22 km2) and reads zero on many scenes. The permanent change shows at annual scale (September median 0.20 km2 in 2021, 0.075 km2 in 2022) but cannot be timed as an event.

### 4.5 Imja Tsho, Khumbu (quiet lake)
- **Lake:** 27.898 N, 86.928 E, moraine-dammed, outlet channel, inventory lake 4514 (1.72 km2); a test lake, so the model never saw it. No outburst in the Sentinel era (it was lowered in 2016, per the build plan).
- **Series:** 178 Sentinel-1 dates, 2017 to 2023 (two tracks: ascending 12, median 1.62 km2; descending 121, median 1.55 km2). Area grew from 1.43 km2 (2017) to 1.69 km2 (2023).
- **Result:** 2 of 171 pairs flagged, both single-scene effects: 2018-10-17 (a 22% dip that recovers over 12 days, probably surface snow or ice) and 2020-07-14 (the return from a single high outlier of 2.19 km2). The persistence rule does not remove them. Per-track mode: 7 of 160.

### 4.6 Thulagi, Gandaki basin (quiet lake)
- **Lake:** 28.487 N, 84.486 E, moraine-dammed, inventory lake 3483 (0.97 km2); a test lake. No outburst in the Sentinel era.
- **Series:** 265 Sentinel-1 dates, 2017 to 2023, from three tracks that disagree strongly (track 85: median 0.66 km2, track 19: 0.85, track 158: 0.93), which makes the plot zig-zag and the plain noise large (0.168 km2). Area 0.83 km2 in 2017 to 0.89 km2 in 2023.
- **Result:** 0 of 258 pairs flagged, but the plain threshold is 0.67 km2 (79% of the area), so only a near-total drainage would be caught. Per-track mode cuts the noise to 0.020 km2 (9% of area) at the cost of 8 of 241 flags.

## 5. Cross-cutting findings
- **Sensitivity depends on the lake.** The smallest detectable drop ranges from about 5% to over 100% of the area. State it next to every detection claim.
- **Sentinel-1 orbit tracks inflate the noise.** Each lake has two or three relative-orbit tracks with different viewing geometry, and the model's area differs by track. Comparing each scene with the previous scene of the same track cuts the noise by a factor of 2 to 8 (table in section 3) but gives more flags on quiet lakes, because outlier scenes now cross a tighter threshold, and has no persistence confirmation. It was added after seeing the noise on the quiet lakes, so it is a post-hoc fix, not a validated detector.
- **Persistence rule:** added after seeing Imja Tsho, Lower Ngole and South Lhonak, fixed before Thulagi was looked at. It removes Lower Ngole's two false alarms but not Imja Tsho's. Do not tune it further on these lakes.
- **Model bias:** areas are below the manual outlines (about 10% at Imja Tsho and Thulagi, about 30% at Lower Ngole, about 60% at Chamai).
- **Event dates:** HMAGLOFDB dates carry their own uncertainty (see Jinwuco).

## 6. Not evaluable (data deleted on the owner's instruction)
The rule: if data are pulled for a lake and it proves not evaluable, delete the data and keep the finding here.
1. **Gongbatongsha Co (burst 2016-07-05).** The lake was tiny (0.017 km2 in Sattar et al. 2022, Sci Rep 12:12421; another source gives about 0.14 km2) and neither GLOID nor the inventory maps it (it drained before 2017). Sentinel-1 scenes over the area start on 2015-10-30 and none exist between June and August 2016, so only one scene precedes the burst; Sentinel-2 L2A is almost absent before 2017. The corrected coordinates (28.0781 N, 86.0700 E; the earlier ones were about 1 km off) were kept in `configs/lakes.yaml`.
2. **Upper Ngole lake (Thame, 4,890 m, GLOID `GLO_86.56368_27.83346`, 0.034 km2 in 2017 growing to 0.104 km2 in 2024).** Same 2024 cascade (it overtopped; the lower lake breached) and only about 0.013 km2 lost. The model reads zero area in 80 to 100% of scenes before 2020, so the series is degenerate and the noise estimate collapses. It is also the same event as Lower Ngole, not an independent one.
3. **Rejected without downloading:** the July 2025 Rasuwa (Lende Khola) flood (a supraglacial lake on Pyurepu Glacier formed from March 2025, no earlier record, mechanism disputed); Langmale 2017 (0.03 to 0.05 km2); Birendra Taal 2024 (no area change); Thorthormi 2019 and 2023; ice-dammed Karakoram lakes (Kyagar, Shisper); small Karakoram and Pamir lakes.
4. **Not selected:** an unnamed 2018-07-08 event (28.244 N, 86.321 E; GLOID 0.214 to 0.117 km2; evidence is villager reports plus imagery). GLOID also shows other lakes dropping by over 50% in a year (for example `GLO_86.50218_28.03313` in 2021, `GLO_82.20876_29.93047` in 2022); none is verified as an outburst.

Event source: HMAGLOFDB v1.3.0 (Steiner and Shrestha, ICIMOD; CC0; Zenodo concept DOI 10.5281/zenodo.7271187); file `data/external/hmaglofdb/.../Database/GLOFs/HMAGLOFDB.csv` (latin-1 encoding).

## 7. Data and files
- **Kept per lake:** predictions and lake-region masks (`data/predictions/{lake}`), scene metadata JSONs and DEM (`data/raw/{lake}`), per-date stacks (`data/interim/{lake}/stacked_s1date`). South Lhonak additionally keeps its raw downloads, the older stack folder and its mask, QA and draft files.
- **Deleted (2026-11-06) because they can be refetched or rebuilt:** raw Sentinel images and filtered intermediates of the finished lakes, the training-lake raw, interim and processed data (the Kaggle zip holds the training tiles), South Lhonak's old 256-pixel tiles. To re-predict a lake the stacks suffice; to rebuild stacks rerun `scripts/fetch_lake_series.sh` (30 to 60 minutes per lake).
- **Reports:** per lake one CSV and one figure in `reports/`, per-track tables in `reports/per_track/`, the old South Lhonak numbers in `reports/archive/`, the summary in `reports/m6_multi_lake_summary.csv`.

## 8. Limits and deferred work
- Four events and two quiet lakes: no statistics beyond the counts above. Rules were set after seeing the data and have not been tested on unseen lakes or events.
- Detection is not prediction; no precursor has been shown.
- Deferred until after Milestones 7 to 10 (owner decision): freeze the detector and test it only on unseen lakes; 10 to 15 more quiet lakes for the false-alarm rate; a baseline detector (for example an NDWI area series); a detectable-drop against lake-size curve; the South Lhonak post-burst recovery check; running or dropping the attention forecaster.
- Before Milestone 7: decide which series is "the method" for the risk score and record it in `docs/PREREGISTRATION.md`.

#!/usr/bin/env bash
# From downloaded scenes to a lake-specific area series and burst-detector result, for one lake in lakes.yaml.
#   scripts/build_lake_series.sh LAKE CHECKPOINT "ROI_ARGS" BURST_DATE
# Examples
#   scripts/build_lake_series.sh imja_tsho models/seg_fused_test.pt "--tp-id 4514 --buffer 400 --exclude-others" none
#   scripts/build_lake_series.sh thame_lower_ngole models/seg_fused_test.pt "--tp-id 4556 --buffer 100 --exclude-others" 2024-08-16
# Tiling is skipped on purpose (nothing here needs tiles and they cost gigabytes).
set -eu
LAKE=$1; CKPT=$2; ROI_ARGS=$3; BURST=${4:-none}
PY=".venv/Scripts/python.exe"; export PYTHONUTF8=1
[ "$BURST" = "none" ] && BURST=2100-01-01
echo "$(date +%H:%M) [$LAKE] preprocessing"
$PY src/preprocessing/sar_preprocess.py --lake "$LAKE" > "logs/${LAKE}_pre1.log" 2>&1
$PY src/preprocessing/optical_indices.py --lake "$LAKE" > "logs/${LAKE}_pre2.log" 2>&1
$PY src/preprocessing/stack_by_s1.py --lake "$LAKE" > "logs/${LAKE}_pre3.log" 2>&1   # one stack per Sentinel-1 date
echo "$(date +%H:%M) [$LAKE] stacked dates: $(ls data/interim/$LAKE/stacked_s1date/*.tif | wc -l)"
echo "$(date +%H:%M) [$LAKE] predicting masks"
$PY src/training/predict_masks.py --lake "$LAKE" --ckpt "$CKPT" --stack-subdir stacked_s1date > "logs/${LAKE}_predict.log" 2>&1
echo "$(date +%H:%M) [$LAKE] lake region + area series"
$PY scripts/make_lake_roi.py --lake "$LAKE" --stack-subdir stacked_s1date $ROI_ARGS | tee "logs/${LAKE}_roi.log" | head -3
echo "$(date +%H:%M) [$LAKE] burst detector (burst date $BURST)"
$PY src/training/burst_detection.py --lake "$LAKE" --burst-date "$BURST" | tee "logs/${LAKE}_detect.log"

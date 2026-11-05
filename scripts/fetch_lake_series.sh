#!/usr/bin/env bash
# Download the June-October Sentinel-1 / Sentinel-2 history and the DEM for one lake from lakes.yaml, with retries.
# Earth Engine sometimes returns a temporary 503; gee_fetch.py skips files it already has, so reruns are safe.
#   scripts/fetch_lake_series.sh imja_tsho 2017 2023 [s1 s2]
set -u
LAKE=$1; Y0=$2; Y1=$3; shift 3
SENSORS=${*:-"s1 s2"}
END_MD=${END_MD:-11-01}   # season end (month-day); use 12-01 for lakes whose event is in October
PY=".venv/Scripts/python.exe"
for y in $(seq "$Y0" "$Y1"); do
  for s in $SENSORS; do
    ok=0
    for t in 1 2 3 4 5; do
      if PYTHONUTF8=1 $PY src/ingestion/gee_fetch.py --lake "$LAKE" --sensor "$s" --start "$y-06-01" --end "$y-$END_MD" >/dev/null 2>&1; then ok=1; break; fi
      sleep 20
    done
    echo "$(date +%H:%M) $LAKE $y $s ok=$ok"
  done
done
for t in 1 2 3 4 5; do
  PYTHONUTF8=1 $PY src/ingestion/gee_fetch.py --lake "$LAKE" --sensor dem >/dev/null 2>&1 && break
  sleep 20
done
echo "$(date +%H:%M) $LAKE DONE s1=$(ls data/raw/$LAKE/s1/*.tif 2>/dev/null | wc -l) s2=$(ls data/raw/$LAKE/s2/*.tif 2>/dev/null | wc -l) dem=$(ls data/raw/$LAKE/dem/*.tif 2>/dev/null | wc -l)"

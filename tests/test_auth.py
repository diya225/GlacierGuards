"""
test_auth.py — Verify that all API credentials are working before running the pipeline.

Usage:
    python tests/test_auth.py

Checks:
  1. GEE service account authentication + a trivial API call
  2. NASA Earthdata credentials (ASF search connectivity)
  3. CDSE OAuth credentials (optional, warns if not set)
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_gee() -> bool:
    """Test GEE authentication with a trivial API call."""
    print("\n" + "─" * 50)
    print("  Test 1: Google Earth Engine")
    print("─" * 50)
    try:
        from src.utils.auth import initialize_gee
        import ee

        initialize_gee()
        # Trivial call: get the list of S1 images over a tiny bbox, count ≥ 0
        point = ee.Geometry.Point([88.1976, 27.9124])  # South Lhonak
        count = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(point)
            .filterDate("2023-01-01", "2023-06-01")
            .size()
            .getInfo()
        )
        print(f"  ✅ GEE authenticated. Found {count} S1 images over South Lhonak (2023 H1).")
        return True
    except Exception as exc:
        print(f"  ❌ GEE authentication FAILED: {exc}")
        print()
        print("  How to fix:")
        print("  1. Make sure GEE_PROJECT_ID is set in your .env")
        print("  2. Make sure ServiceAccountKey.json is in the project root")
        print("  3. Go to https://earthengine.google.com/ → your project →")
        print("     confirm Earth Engine API is enabled in Google Cloud Console")
        print("  4. Make sure the service account is registered with Earth Engine:")
        print("     https://code.earthengine.google.com/register")
        return False


def test_earthdata() -> bool:
    """Test NASA Earthdata credentials by running a trivial ASF search."""
    print("\n" + "─" * 50)
    print("  Test 2: NASA Earthdata / ASF")
    print("─" * 50)
    try:
        import asf_search as asf
        from src.utils.auth import get_earthdata_credentials

        username, password = get_earthdata_credentials()
        # Tiny search — just check credentials are accepted
        session = asf.ASFSession().auth_with_creds(username, password)
        print(f"  ✅ NASA Earthdata authenticated as: {username}")
        return True
    except Exception as exc:
        print(f"  ❌ NASA Earthdata authentication FAILED: {exc}")
        print()
        print("  How to fix:")
        print("  1. Make sure EARTHDATA_USERNAME and EARTHDATA_PASSWORD are in .env")
        print("  2. Confirm your account at https://urs.earthdata.nasa.gov/")
        print("  3. Make sure your account has approved the ASF application:")
        print("     → Log in → Applications → Approve 'ASF Data Access'")
        return False


def test_cdse() -> bool:
    """Test Copernicus Data Space Ecosystem OAuth credentials."""
    print("\n" + "─" * 50)
    print("  Test 3: Copernicus Data Space (CDSE) — optional")
    print("─" * 50)
    try:
        import requests
        from src.utils.auth import get_cdse_credentials

        client_id, client_secret = get_cdse_credentials()
        resp = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            print(f"  ✅ CDSE OAuth token obtained successfully (client_id: {client_id[:12]}…)")
            return True
        else:
            print(f"  ❌ CDSE returned HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except EnvironmentError:
        print("  ⚠️  CDSE credentials not set — skipping (optional, only needed if not using GEE for Sentinel).")
        return True  # Not a blocker
    except Exception as exc:
        print(f"  ❌ CDSE test FAILED: {exc}")
        return False


def main() -> None:
    print("\n" + "═" * 50)
    print("  GLOF EWS — API Credential Tests")
    print("═" * 50)

    results = {
        "GEE": test_gee(),
        "NASA Earthdata": test_earthdata(),
        "CDSE": test_cdse(),
    }

    print("\n" + "═" * 50)
    print("  Summary")
    print("═" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 All credential tests passed! Ready to run the pipeline.")
    else:
        print("  ⚠️  Fix the failing credentials above before running ingestion.")
    print()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

"""
auth.py — Centralised authentication helpers for GEE, ASF/HyP3, and CDSE.
All credentials are read exclusively from the project .env file.
"""

import os
import json
import logging
from pathlib import Path

import ee
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Resolve project root regardless of which script calls this module
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env():
    """Load .env from the project root (idempotent)."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        logger.warning(
            ".env file not found at %s. "
            "Make sure environment variables are set before running.",
            env_path,
        )


def initialize_gee() -> str:
    """
    Authenticate with Google Earth Engine using a service account.

    Reads from environment:
        GEE_PROJECT_ID            — GCP project ID that has Earth Engine enabled
        GEE_SERVICE_ACCOUNT_JSON  — path to the service account key JSON

    Returns:
        The GEE project ID string.

    Raises:
        FileNotFoundError: if the service account JSON is missing.
        ee.EEException:    if Earth Engine rejects the credentials.
    """
    _load_env()

    project_id = os.getenv("GEE_PROJECT_ID")
    if not project_id:
        raise EnvironmentError(
            "GEE_PROJECT_ID is not set. "
            "Check your .env file or environment variables."
        )

    sa_json_rel = os.getenv("GEE_SERVICE_ACCOUNT_JSON", "./ServiceAccountKey.json")
    sa_json_path = Path(sa_json_rel)
    if not sa_json_path.is_absolute():
        sa_json_path = PROJECT_ROOT / sa_json_rel

    if not sa_json_path.exists():
        raise FileNotFoundError(
            f"GEE service account JSON not found at: {sa_json_path}\n"
            "1. Download the JSON key from Google Cloud Console → IAM → Service Accounts.\n"
            "2. Place it in the project root and set GEE_SERVICE_ACCOUNT_JSON in .env."
        )

    with open(sa_json_path) as fh:
        key_data = json.load(fh)

    service_account_email = key_data.get("client_email")
    if not service_account_email:
        raise ValueError(
            "Could not read 'client_email' from the service account JSON. "
            "Make sure the file is a valid GCP service account key."
        )

    credentials = ee.ServiceAccountCredentials(
        service_account_email, str(sa_json_path)
    )
    ee.Initialize(credentials, project=project_id)
    logger.info("GEE initialised — project: %s, SA: %s", project_id, service_account_email)
    return project_id


def get_earthdata_credentials() -> tuple[str, str]:
    """
    Return (username, password) for NASA Earthdata / ASF / HyP3.

    Reads from environment:
        EARTHDATA_USERNAME
        EARTHDATA_PASSWORD
    """
    _load_env()
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise EnvironmentError(
            "EARTHDATA_USERNAME or EARTHDATA_PASSWORD is not set. "
            "Create a free account at https://urs.earthdata.nasa.gov/ "
            "and add credentials to .env."
        )
    return username, password


def get_cdse_credentials() -> tuple[str, str]:
    """
    Return (client_id, client_secret) for Copernicus Data Space Ecosystem (CDSE).

    Reads from environment:
        CDSE_CLIENT_ID
        CDSE_CLIENT_SECRET
    """
    _load_env()
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is not set. "
            "Register at https://dataspace.copernicus.eu/ and create an OAuth client."
        )
    return client_id, client_secret

########################################################################################################################

import typing as T  # isort: split

from functools import cache
from pathlib import Path

import json5

########################################################################################################################

_REGION_V1_FILE = "region_v1.json"
_REGION_V2_FILE = "region_v2.json"


@cache
def _load_region_v1_data() -> T.List[T.Dict[str, T.Any]]:
    """Load API v1 region data from JSON file."""
    # Get path to JSON file (same directory as this module)
    module_dir = Path(__file__).parent
    json_path = module_dir / _REGION_V1_FILE

    with open(json_path, "r", encoding="utf-8") as f:
        return json5.load(f)


@cache
def _load_region_v2_data() -> T.List[T.Dict[str, T.Any]]:
    """Load API v2 region data from JSON file."""
    # Get path to JSON file (same directory as this module)
    module_dir = Path(__file__).parent
    json_path = module_dir / _REGION_V2_FILE

    with open(json_path, "r", encoding="utf-8") as f:
        return json5.load(f)


@cache
def _load_country_servers() -> T.Dict[str, T.Optional[T.List[str]]]:
    """Build country->servers mapping from v1 and v2 region data."""
    v1_data = _load_region_v1_data()
    v2_data = _load_region_v2_data()
    country_servers: T.Dict[str, T.Optional[T.List[str]]] = {}

    # Process API v1 countries from region_v1.json (grouped format)
    for group in v1_data:
        countries = group.get("countries", [])

        # Handle offline group (no cloudUrl)
        if "cloudUrl" not in group:
            for country in countries:
                country_servers[country.upper()] = None
            continue

        # Handle online group (has cloudUrl/appID/appKey)
        server_url = group["cloudUrl"]
        for country in countries:
            country_servers[country.upper()] = [server_url]

    # Process API v2 countries from region_v2.json (grouped format)
    for group in v2_data:
        region = group.get("region")
        countries = group.get("countries", [])

        # Handle offline region (no api_configuration)
        if region == "OFFLINE":
            for country in countries:
                country_upper = country.upper()
                # Skip if already in v1 (v1 takes precedence)
                if country_upper not in country_servers:
                    country_servers[country_upper] = None

            continue

        # Handle online regions with api_configuration
        api_cfg = group.get("api_configuration", {})
        prod_base = api_cfg.get("prod_base_url", "")
        prod_secondary = api_cfg.get("prod_secondary_base_url", "")

        if not prod_base:
            # No servers configured for this region
            continue

        # Build server list (primary + optional secondary)
        servers = [prod_base]
        if prod_secondary:  # Only add if not empty string
            servers.append(prod_secondary)

        # Apply to all countries in this group
        for country in countries:
            country_upper = country.upper()
            # Skip if already in v1 (v1 takes precedence)
            if country_upper not in country_servers:
                country_servers[country_upper] = servers

    return country_servers


@cache
def _load_server_credentials() -> T.Dict[str, T.Tuple[str, str]]:
    """Build server->credentials mapping from API v1 region data."""
    region_data = _load_region_v1_data()
    server_credentials = {}

    # Extract credentials for each v1 group (skip offline groups)
    for group in region_data:
        # Only process groups with cloudUrl (online groups)
        if "cloudUrl" in group:
            server_url = group["cloudUrl"]
            app_id = group["appID"]
            app_key = group["appKey"]
            server_credentials[server_url] = (app_id, app_key)

    return server_credentials


def get_servers_for_country_code(country_code: str) -> T.Optional[T.List[str]]:
    """
    Get OMRON Connect API servers for a country code.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'US', 'JP', 'DE')

    Returns:
        List of server URLs to try (in order), or None if country not supported
    """
    country_servers = _load_country_servers()
    country_code = country_code.upper()

    return country_servers.get(country_code)


def get_credentials_for_server(server_url: str) -> T.Optional[T.Tuple[str, str]]:
    """
    Get API v1 credentials (appID, appKey) for a server URL.

    Args:
        server_url: API server URL (e.g., 'https://data-jp.omronconnect.com/api')

    Returns:
        Tuple of (appID, appKey), or None if server doesn't use v1 API credentials
    """
    server_credentials = _load_server_credentials()
    return server_credentials.get(server_url)


########################################################################################################################

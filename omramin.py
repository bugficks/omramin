#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click>=8.1.7",
#   "garminconnect>=0.2.25",
#   "garth>=0.5.21; python_version>='3.14'",
#   "httpx[http2,cli,brotli]>=0.28.1",
#   "inquirer>=3.4.0",
#   "json5>=0.10.0",
#   "keyring>=24.3.0",
#   "keyrings.alt>=5.0.0; sys_platform == 'win32'",
#   "python-dateutil>=2.9.0.post0",
#   "pytz>=2025.1",
# ]
# ///
########################################################################################################################
# pylint: disable=too-many-lines

import typing as T  # isort: split

import asyncio
import binascii
import csv
import dataclasses
import enum
import logging
import logging.config
import os
import pathlib
import platform
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import cache, wraps

import click
import garminconnect as GC
import garth
import inquirer
import keyring
from dateutil import parser as dateutil_parser
from httpx import HTTPStatusError

import omronconnect as OC
import utils as U

########################################################################################################################

__version__ = "0.2.0pre0"

########################################################################################################################


@dataclasses.dataclass
class Options:
    """Options for sync operations"""

    write_to_garmin: bool = True
    overwrite: bool = False
    ble_filter: str = "BLEsmart_"


@dataclasses.dataclass
class MeasurementSyncHandler:
    """Handler for syncing specific measurement types"""

    fetch_garmin_data: T.Callable[[GC.Garmin, str, str], T.Dict[str, T.Any]]  # e.g., garmin_get_weighins
    delete_measurement: T.Callable  # e.g., gc.delete_weigh_in
    add_measurement: T.Callable[[datetime, T.Any, Options], None]  # e.g., add scale/BP measurement
    measurement_type: T.Type  # OC.WeightMeasurement or OC.BPMeasurement
    delete_key_field: str  # "samplePk" or "version"
    log_name: str  # "weigh-in" or "blood pressure"


########################################################################################################################

CONFIG_FILENAME = "config.json"
CONFIG_DIR_NAME = "omramin"

PATH_DEFAULT_CONFIG = f"~/.{CONFIG_DIR_NAME}/{CONFIG_FILENAME}"
PATH_XDG_CONFIG = f"~/.config/{CONFIG_DIR_NAME}/{CONFIG_FILENAME}"

DEFAULT_CONFIG = {
    "version": 1,
    "garmin": {},
    "omron": {
        "devices": [],
    },
}

OMRON_DEVICE_LIST_DAYS = 30


class KeyringBackend(enum.StrEnum):
    SYSTEM = "system"
    FILE = "file"
    ENCRYPTED = "encrypted"


LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "http": {
            "class": "logging.StreamHandler",
            "formatter": "http",
            "stream": "ext://sys.stderr",
        },
    },
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname).1s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "http": {
            "format": "[%(asctime)s] [%(levelname).1s] (%(name)s) - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": logging.INFO,
            "formatter": "root",
        },
        "omronconnect": {
            "handlers": ["default"],
            "level": logging.INFO,
            "propagate": False,
            "formatter": "root",
        },
        "garminconnect": {
            "handlers": ["default"],
            "level": logging.INFO,
            "formatter": "root",
        },
        "httpx": {
            "handlers": ["http"],
            "level": logging.WARNING,
            "formatter": "root",
        },
        "httpcore": {
            "handlers": ["http"],
            "level": logging.WARNING,
            "formatter": "root",
        },
    },
}

########################################################################################################################

_E = os.environ.get

logging.config.dictConfig(LOGGING_CONFIG)

L = logging.getLogger("")

# Individual env var debug enables (for targeted debugging)
if _E("OMRAMIN_OMRON_DEBUG"):
    logging.getLogger("omronconnect").setLevel(logging.DEBUG)
    L.debug("OMRON debug enabled")

if _E("OMRAMIN_GARMIN_DEBUG"):
    logging.getLogger("garminconnect").setLevel(logging.DEBUG)
    logging.getLogger("garminconnect").disabled = False
    L.debug("Garmin debug enabled")


def _configure_logging_levels(debug: bool = False) -> None:
    """Configure logging levels based on debug flag."""

    if debug:
        # Debug: everything at DEBUG level
        logging.getLogger("").setLevel(logging.DEBUG)
        logging.getLogger("omronconnect").setLevel(logging.DEBUG)
        logging.getLogger("garminconnect").setLevel(logging.DEBUG)
        logging.getLogger("garminconnect").disabled = False
        L.debug("Debug mode enabled (all loggers at DEBUG level)")


########################################################################################################################
class LoginError(Exception):
    pass


def prompt_credentials(
    service_name: str,
    email: str = "",
    extra_fields: T.Optional[list] = None,
) -> T.Optional[dict]:
    """
    Prompt for credentials with defaults.

    Args:
        service_name: Display name (e.g., "Garmin", "OMRON")
        email: Existing email to show as default
        extra_fields: List of dicts with field configs:
            [{"name": "country", "message": "...", "type": "text", "default": "", "validate": ...}]
            type can be: "text", "password", "confirm"

    Returns:
        Dict with "email", "password", and any extra field names, or None if cancelled
    """

    questions = [
        inquirer.Text(
            name="email",
            message="> Enter email",
            default=email,
            validate=lambda _, x: x != "",
        ),
        inquirer.Password(
            name="password",
            message="> Enter password",
            validate=lambda _, x: x != "",
        ),
    ]

    # Add extra fields
    if extra_fields:
        for field in extra_fields:
            field_type = field.get("type", "text")
            if field_type == "text":
                questions.append(
                    inquirer.Text(
                        name=field["name"],
                        message=field["message"],
                        default=field.get("default", ""),
                        validate=field.get("validate"),
                    )
                )

            elif field_type == "confirm":
                questions.append(
                    inquirer.Confirm(
                        name=field["name"],
                        message=field["message"],
                        default=field.get("default", False),
                    )
                )

    L.info(f"{service_name} login")
    answers = inquirer.prompt(questions)
    return answers


def garmin_login(config_path: str) -> T.Optional[GC.Garmin]:
    """Login to Garmin Connect"""

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    gcCfg = config.get("garmin", {})

    # Get email from config or env
    email = _E("GARMIN_EMAIL") or gcCfg.get("email", "")

    def get_mfa() -> str:
        return inquirer.text(message="> Enter MFA/2FA code")

    logged_in = False
    try:
        # Load tokens from keyring (requires email)
        tokendata = None
        if email:
            tokendata = load_service_tokens(config_path, "garmin", email, migrate_from_config=config)

        if not tokendata:
            raise FileNotFoundError

        is_cn = gcCfg.get("is_cn", False)
        gc = GC.Garmin(email=email, is_cn=is_cn, prompt_mfa=get_mfa)

        L.debug(f"Attempting Garmin login using cached tokens for {email}")
        logged_in = gc.login(tokendata)
        if not logged_in:
            L.debug("Garmin cached tokens invalid, attempting password login")
            raise FileNotFoundError

        L.debug(f"Garmin login successful using cached tokens for {email}")

    except (FileNotFoundError, binascii.Error, KeyError):
        # Precedence: env > config > prompt
        password = _E("GARMIN_PASSWORD")
        is_cn_str = _E("GARMIN_IS_CN") or str(gcCfg.get("is_cn", "false"))
        is_cn = is_cn_str.lower() in ("true", "1", "yes")

        # Prompt for credentials if not from env vars
        if not (email and password and (_E("GARMIN_IS_CN") or "is_cn" in gcCfg)):
            answers = prompt_credentials(
                service_name="Garmin",
                email=email,
                extra_fields=[
                    {
                        "name": "is_cn",
                        "message": "> Is this a Chinese account?",
                        "type": "confirm",
                        "default": is_cn,
                    }
                ],
            )
            if not answers:
                raise LoginError("Invalid input") from None

            email = answers["email"]
            password = answers["password"]
            is_cn = answers["is_cn"]

        if not email or not password:
            raise LoginError("Missing credentials") from None

        gc = GC.Garmin(email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa)
        try:
            logged_in = gc.login()

        except Exception as login_error:  # pylint: disable=broad-except
            L.error(f"Login failed: {login_error}")
            L.info("Please re-enter your credentials")

            # Re-prompt for all fields with existing values as defaults
            answers = prompt_credentials(
                service_name="Garmin",
                email=email,
                extra_fields=[
                    {
                        "name": "is_cn",
                        "message": "> Is this a Chinese account?",
                        "type": "confirm",
                        "default": is_cn,
                    }
                ],
            )
            if not answers:
                raise LoginError("Invalid input") from login_error

            email = answers["email"]
            password = answers["password"]
            is_cn = answers["is_cn"]

            # Retry login with new credentials
            gc = GC.Garmin(email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa)
            logged_in = gc.login()

        if logged_in:
            # Save credentials to config (if not from env vars)
            config_changed = False
            if not _E("GARMIN_EMAIL"):
                gcCfg["email"] = email
                config_changed = True

            if not _E("GARMIN_IS_CN"):
                gcCfg["is_cn"] = is_cn
                config_changed = True

            if config_changed:
                try:
                    config["garmin"] = gcCfg
                    U.json_save(config_path, config)

                except (OSError, IOError, ValueError) as e:
                    L.warning(f"Failed to save config: {e}")

            # Save tokens to keyring
            save_service_tokens(config_path, "garmin", email, gc.garth.dumps())

    except garth.exc.GarthHTTPError:
        L.error("Failed to login to Garmin Connect", exc_info=True)
        return None

    if not logged_in:
        L.error("Failed to login to Garmin Connect")
        return None

    L.info("Logged in to Garmin Connect")
    return gc


def migrateconfig_path(config: dict) -> dict:
    version = config.get("version", 0)

    if version < 1:
        if "server" in config.get("omron", {}):
            del config["omron"]["server"]

        config["version"] = 1

    return config


def get_keyring_file_path(config_path: str) -> str:
    """Determine keyring file path from config path or env var."""

    if file_path := _E("OMRAMIN_KEYRING_FILE"):
        return str(pathlib.Path(file_path).expanduser().resolve())

    config_path_obj = pathlib.Path(config_path).expanduser().resolve()
    return str(config_path_obj.parent / f"{config_path_obj.stem}.tokens.json")


def get_keyring_password() -> T.Optional[str]:
    """Get password for encrypted keyring backend."""

    password = _E("OMRAMIN_KEYRING_PASSWORD")
    if not password:
        raise ValueError(
            "Encrypted keyring backend requires password. Set OMRAMIN_KEYRING_PASSWORD environment variable."
        )

    return password


def _detect_best_backend() -> str:
    """Detect best keyring backend for current environment.

    Returns:
        Backend type as string ('system', 'file', or 'encrypted')
    """
    # Windows Credential Manager has 2560-byte limit (UTF-16 encoding)
    # OMRON OAuth tokens often exceed this limit - use file backend on Windows
    if platform.system() == "Windows":
        L.debug("Detected Windows, using file backend (WinVault has 2560-byte limit)")
        return KeyringBackend.FILE.value

    # In Docker, always use file backend
    if os.path.exists("/.dockerenv"):
        L.debug("Detected Docker environment, using file backend")
        return KeyringBackend.FILE.value

    kr = keyring.get_keyring()

    # If not a ChainerBackend, check if it's a known system keyring
    if kr.__class__.__name__ != "ChainerBackend":
        # Direct system keyring (macOS.Keyring, WinVaultKeyring, SecretService.Keyring)
        if _is_system_keyring(kr):
            L.debug(f"Detected system keyring: {kr.__class__.__module__}.{kr.__class__.__name__}")
            return KeyringBackend.SYSTEM.value

        L.debug(f"Unknown keyring backend: {kr.__class__.__name__}, using file backend")
        return KeyringBackend.FILE.value

    # ChainerBackend: check if it contains any system keyrings
    # pylint: disable-next=import-outside-toplevel
    from keyring.backends.chainer import ChainerBackend

    if isinstance(kr, ChainerBackend):
        for backend in kr.backends:
            if _is_system_keyring(backend):
                L.debug(f"Found system keyring in chain: {backend.__class__.__module__}.{backend.__class__.__name__}")
                return KeyringBackend.SYSTEM.value

        L.debug("ChainerBackend contains no system keyrings, using file backend")
        return KeyringBackend.FILE.value

    # Fallback
    L.debug("Using system keyring backend")
    return KeyringBackend.SYSTEM.value


def _is_system_keyring(backend: keyring.backend.KeyringBackend) -> bool:
    """Check if a keyring backend is a platform system keyring.

    Returns:
        True if backend is macOS Keychain, Windows Credential Manager, or Linux Secret Service
    """

    module = backend.__class__.__module__
    classname = backend.__class__.__name__

    # macOS Keychain
    if module == "keyring.backends.macOS" and classname == "Keyring":
        return True

    # Windows Credential Manager
    if module == "keyring.backends.Windows" and classname == "WinVaultKeyring":
        return True

    # Linux Secret Service (GNOME Keyring, KWallet, etc.)
    if module == "keyring.backends.SecretService" and classname == "Keyring":
        return True

    return False


@cache
def get_keyring_backend(config_path: str) -> keyring.backend.KeyringBackend:
    """Get configured keyring backend instance.

    Respects standard keyring environment variables with precedence:
    1. PYTHON_KEYRING_BACKEND (standard keyring library variable)
    2. OMRAMIN_KEYRING_BACKEND (app-specific override)
    3. Auto-detection based on environment

    Cached per config_path to avoid duplicate initialization.
    """
    # Check for standard keyring environment variable first
    python_backend = _E("PYTHON_KEYRING_BACKEND")
    if python_backend:
        try:
            # Import and instantiate the backend class
            module_name, class_name = python_backend.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            backend_class = getattr(module, class_name)
            kr = backend_class()

            # Configure file path for file-based backends
            if hasattr(kr, "file_path"):
                kr.file_path = get_keyring_file_path(config_path)

            if hasattr(kr, "keyring_key"):
                kr.keyring_key = get_keyring_password()

            L.debug(f"Using keyring backend from PYTHON_KEYRING_BACKEND: {python_backend}")
            return kr

        except Exception as e:  # pylint: disable=broad-except
            L.warning(f"Failed to load PYTHON_KEYRING_BACKEND '{python_backend}': {e}")
            # Fall through to OMRAMIN_KEYRING_BACKEND logic

    # Fall back to app-specific configuration
    backend_str = _E("OMRAMIN_KEYRING_BACKEND")

    # Auto-detect if not explicitly set
    if not backend_str:
        backend_str = _detect_best_backend()

    backend_str = backend_str.lower()

    # Auto-upgrade to cryptfile backend if password is set and cryptfile is installed
    if backend_str == KeyringBackend.FILE.value and _E("OMRAMIN_KEYRING_PASSWORD"):
        try:
            # pylint: disable-next=import-outside-toplevel
            from keyrings.cryptfile.cryptfile import CryptFileKeyring

            # cryptfile is available, use it
            kr = CryptFileKeyring()
            kr.file_path = get_keyring_file_path(config_path)
            if hasattr(kr, "keyring_key"):
                kr.keyring_key = get_keyring_password()

            return kr

        except ImportError:
            L.debug("CryptFileKeyring not available")
            backend_str = KeyringBackend.ENCRYPTED.value

        except Exception as e:  # pylint: disable=broad-except
            L.warning(f"CryptFileKeyring failed to initialize, falling back to EncryptedKeyring: '{e}'", exc_info=True)
            backend_str = KeyringBackend.ENCRYPTED.value

    try:
        backend_type = KeyringBackend(backend_str)

    except ValueError:
        L.warning(f"Unknown keyring backend '{backend_str}', using system backend")
        backend_type = KeyringBackend.SYSTEM

    if backend_type == KeyringBackend.SYSTEM:
        return keyring.get_keyring()

    if backend_type == KeyringBackend.FILE:
        try:
            # pylint: disable-next=import-outside-toplevel
            from keyrings.alt.file import PlaintextKeyring

        except ImportError as e:
            raise ImportError(
                "File backend requires 'keyrings.alt' package. Install with: pip install keyrings.alt"
            ) from e

        kr = PlaintextKeyring()
        kr.file_path = get_keyring_file_path(config_path)
        return kr

    if backend_type == KeyringBackend.ENCRYPTED:
        try:
            # pylint: disable-next=import-outside-toplevel
            from keyrings.alt.file import EncryptedKeyring

        except ImportError as e:
            raise ImportError(
                "Encrypted backend requires 'keyrings.alt' package. Install with: pip install keyrings.alt"
            ) from e

        kr = EncryptedKeyring()
        kr.file_path = get_keyring_file_path(config_path)
        kr.keyring_key = get_keyring_password()
        return kr

    return None


def _keyring_id(service: str, email: str) -> T.Tuple[str, str]:
    """Email-based keyring ID.

    Args:
        service: Service type ('garmin' or 'omron')
        email: User's email address

    Returns:
        Tuple of (service_name, username) for keyring lookup
        Format: ("omramin+garmin:user@example.com", "user@example.com")
    """

    if not email:
        raise ValueError("Email is required for keyring ID")

    return f"omramin+{service}:{email}", email


def load_service_tokens(
    config_path: str, service: str, email: str, *, migrate_from_config: T.Optional[dict] = None
) -> T.Optional[str]:
    """Load tokens for a specific service from keyring backend.

    Args:
        config_path: Path to config file (used for legacy file migration)
        service: Service type ('garmin' or 'omron')
        email: User's email address
        migrate_from_config: Config dict to migrate tokens from (optional)

    Returns:
        Token data string or None if not found
    """

    if not email:
        L.debug(f"Cannot load {service} tokens: email not available")
        return None

    # Try loading from keyring backend
    try:
        backend = get_keyring_backend(config_path)
        service_name, username = _keyring_id(service, email)
        tokendata = backend.get_password(service_name, username)
        if tokendata:
            L.debug(f"Loaded {service} tokens from keyring backend")
            return tokendata

    except Exception as e:  # pylint: disable=broad-except
        L.debug(f"Could not load {service} tokens from keyring: {e}")

    # Try migration from config.json (embedded tokens)
    if migrate_from_config and service in migrate_from_config:
        service_cfg = migrate_from_config[service]
        tokendata = service_cfg.get("tokendata")
        if tokendata:
            L.debug(f"Migrating {service} tokens from {CONFIG_FILENAME}")
            if save_service_tokens(config_path, service, email, tokendata):
                L.info(f"Migrated {service} tokens from {CONFIG_FILENAME}")

                # Clean up obsolete tokendata from config
                del service_cfg["tokendata"]
                try:
                    U.json_save(config_path, migrate_from_config)
                    L.debug(f"Removed obsolete {service} tokendata from {CONFIG_FILENAME}")

                except Exception as e:  # pylint: disable=broad-except
                    # Log warning but don't fail - tokens ARE in keyring (primary goal)
                    L.warning(f"Could not remove obsolete {service} tokendata from config: {e}")
                    L.warning("Consider manually removing it or making config writable")

            return tokendata

    L.debug(f"No {service} tokens found")
    return None


def save_service_tokens(config_path: str, service: str, email: str, tokendata: str) -> bool:
    """Save tokens for a specific service to keyring backend.

    Args:
        config_path: Path to config file
        service: Service type ('garmin' or 'omron')
        email: User's email address
        tokendata: Token data to save

    Returns:
        True if successful, False otherwise
    """

    if not email:
        L.error(f"Cannot save {service} tokens: email not available")
        return False

    try:
        backend = get_keyring_backend(config_path)
        service_name, username = _keyring_id(service, email)
        L.debug(
            f"Saving {service} token: length={len(tokendata)}, "
            f"service={service_name!r}, username={username!r}, "
            f"preview={tokendata[:50]}..."
        )
        backend.set_password(service_name, username, tokendata)
        L.debug(f"Saved {service} tokens to keyring backend")
        return True

    except Exception as e:  # pylint: disable=broad-except
        L.error(f"Failed to save {service} tokens: {e}")
        return False


def clear_service_tokens(config_path: str, service: str, email: str) -> bool:
    """Clear tokens for a service from keyring backend.

    Args:
        config_path: Path to configuration file
        service: Service name ('garmin' or 'omron')
        email: User email address

    Returns:
        True if tokens were found and cleared, False if no tokens existed
    """

    if not email:
        L.error(f"Cannot clear {service} tokens: email not available")
        return False

    try:
        backend = get_keyring_backend(config_path)
        service_name, username = _keyring_id(service, email)

        existing = backend.get_password(service_name, username)
        if not existing:
            L.debug(f"No {service} tokens found for {email}")
            return False

        backend.delete_password(service_name, username)
        L.debug(f"Cleared {service} tokens for {email}")
        return True

    except Exception as e:  # pylint: disable=broad-except
        L.error(f"Failed to clear {service} tokens: {e}")
        return False


def omron_login(config_path: str) -> T.Optional[OC.OmronClient]:
    """Login to OMRON connect"""

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    config = migrateconfig_path(config)
    ocCfg = config.get("omron", {})

    # Get email from config or env (support legacy "username" field)
    email = _E("OMRON_EMAIL") or ocCfg.get("email", "") or ocCfg.get("username", "")

    refreshToken = None
    oc = None
    try:
        # Load tokens from keyring (requires email)
        tokendata = None
        if email:
            tokendata = load_service_tokens(config_path, "omron", email, migrate_from_config=config)

        country = ocCfg.get("country", "")

        if not tokendata or not country:
            raise FileNotFoundError

        oc = OC.OmronClient(country)
        # OmronConnect2 requires email for token refresh
        L.debug(f"Attempting OMRON token refresh for {email} (country={country})")
        refreshToken = oc.refresh_oauth2(tokendata, email=email)
        # Save the new refresh token (OMRON rotates tokens on each refresh)
        if refreshToken and refreshToken != tokendata and email:
            if save_service_tokens(config_path, "omron", email, refreshToken):
                L.debug("OMRON refresh token updated and saved")

            else:
                L.warning("Failed to save refreshed token")

    except Exception as e:  # pylint: disable=broad-except
        if isinstance(e, HTTPStatusError):
            L.error(f"Failed to login to OMRON connect: '{e.response.reason_phrase}: {e.response.status_code}'")
            if e.response.status_code != 403:
                raise

        elif not isinstance(e, FileNotFoundError):
            L.debug(f"Token refresh failed with {type(e).__name__}: {e}", exc_info=True)

        # Precedence: env > config > prompt
        password = _E("OMRON_PASSWORD")
        country = _E("OMRON_COUNTRY") or ocCfg.get("country", "")

        # Prompt for credentials if any are missing
        if not (email and password and country):
            answers = prompt_credentials(
                service_name="OMRON",
                email=email,
                extra_fields=[
                    {
                        "name": "country",
                        "message": "> Enter country code (e.g. 'US')",
                        "type": "text",
                        "default": country,
                        "validate": lambda _, x: len(x) == 2,
                    }
                ],
            )
            if not answers:
                raise LoginError("Invalid input") from None

            email = answers["email"]
            password = answers["password"]
            country = answers["country"]

        if not email or not password or not country:
            raise LoginError("Missing credentials") from None

        oc = OC.OmronClient(country)
        try:
            refreshToken = oc.login(email, password)

        except Exception as login_error:  # pylint: disable=broad-except
            L.error(f"Login failed: {login_error}")
            L.info("Please re-enter your credentials")

            # Re-prompt for all fields with existing values as defaults
            answers = prompt_credentials(
                service_name="OMRON",
                email=email,
                extra_fields=[
                    {
                        "name": "country",
                        "message": "> Enter country code (e.g. 'US')",
                        "type": "text",
                        "default": country,
                        "validate": lambda _, x: len(x) == 2,
                    }
                ],
            )
            if not answers:
                raise LoginError("Invalid input") from login_error

            email = answers["email"]
            password = answers["password"]
            country = answers["country"]

            if not email or not password or not country:
                raise LoginError("Missing credentials") from None

            # Retry login with new credentials
            oc = OC.OmronClient(country)
            refreshToken = oc.login(email, password)

        if refreshToken:
            # Save credentials to config (if not from env vars)
            config_changed = False
            if not _E("OMRON_EMAIL"):
                ocCfg["email"] = email
                config_changed = True

            if not _E("OMRON_COUNTRY"):
                ocCfg["country"] = country
                config_changed = True

            if config_changed:
                try:
                    config["omron"] = ocCfg
                    U.json_save(config_path, config)

                except (OSError, IOError, ValueError) as e:  # pylint: disable=redefined-outer-name
                    L.warning(f"Failed to save config: {e}")

            # Save tokens to keyring
            save_service_tokens(config_path, "omron", email, refreshToken)

    if refreshToken:
        L.info("Logged in to OMRON connect")
        return oc

    L.error("Failed to login to OMRON connect")
    return None


@contextmanager
def config_write_handler(config_path: str, config: dict) -> T.Generator[dict, None, None]:
    """Context manager for safe config writes with helpful error messages.

    Usage:
        with config_write_handler(config_path, config):
            # Modify config
            config["omron"]["devices"].append(device)

    Args:
        config_path: Path to config file
        config: Config dictionary to save

    Raises:
        OSError, IOError, PermissionError, ValueError: On save failures
    """

    try:
        yield config

        pathlib.Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        U.json_save(config_path, config)

    except (OSError, IOError, PermissionError, ValueError) as e:
        if isinstance(e, PermissionError) or "read-only" in str(e).lower():
            L.error("Cannot modify config: file is read-only")
            L.info("To manage devices, either:")
            L.info(f"  1. Make {CONFIG_FILENAME} writable, OR")
            L.info(f"  2. Edit {CONFIG_FILENAME} manually and restart")

        else:
            L.error(f"Failed to save configuration: {e}")

        raise


def require_config_exists(config_path: str) -> None:
    """Check if config file exists and exit with error if not."""

    if not pathlib.Path(config_path).exists():
        L.error(f"Config file '{config_path}' not found.")
        L.info("Run 'omramin init' to create initial configuration.")
        raise SystemExit(1)


def requires_config(func: T.Callable) -> T.Callable:
    """Decorator to ensure config file exists before command executes."""

    @wraps(func)
    def wrapper(ctx: click.Context, *args, **kwargs):
        config_path = ctx.obj["config_path"]
        require_config_exists(config_path)
        return func(ctx, *args, **kwargs)

    return wrapper


def omron_ble_scan(macAddrsExistig: T.List[str], opts: Options) -> T.List[str]:
    """Scan for Omron devices in pairing mode"""

    # Lazy import - only fail if BLE scanning is actually used
    try:
        # pylint: disable-next=import-outside-toplevel
        import bleak

    except ImportError as e:
        raise ImportError(
            "Bluetooth scanning requires 'bleak' package. Install with: pip install 'omramin[bluetooth]'"
        ) from e

    devsFound: T.Dict[str, str] = {}

    async def scan() -> None:
        L.info("Scanning for Omron devices in pairing mode ...")
        L.info("Press Ctrl+C to stop scanning")
        while True:
            devices_map: T.Dict[str, T.Tuple[T.Any, T.Any]] = await bleak.BleakScanner.discover(
                return_adv=True,
                timeout=1,
            )
            sorted_devices = sorted(devices_map.items(), key=lambda x: x[1][1].rssi, reverse=True)
            for macAddr, (bleDev, advData) in sorted_devices:
                devName = (bleDev.name or "").strip()

                # Extract MAC address from device name on MacOS
                if platform.system() == "Darwin" and devName.upper().startswith("BLESMART_"):
                    try:
                        _mac_len = 12
                        _mac_str = devName[-_mac_len:].upper()
                        # Validate it's actually hex before parsing
                        if len(_mac_str) == _mac_len and all(c in "0123456789ABCDEF" for c in _mac_str):
                            macAddr = ":".join(_mac_str[i : i + 2] for i in range(0, _mac_len, 2))

                        else:
                            continue

                    except (IndexError, ValueError):
                        continue

                if macAddr in devsFound:
                    continue

                if macAddr in macAddrsExistig:
                    continue

                if opts.ble_filter and not devName.upper().startswith(opts.ble_filter.upper()):
                    continue

                serial = OC.ble_mac_to_serial(macAddr)
                devsFound[macAddr] = serial
                L.info(f"+ {macAddr} {bleDev.name} {serial} {advData.rssi}")

    try:
        asyncio.run(scan())

    except bleak.exc.BleakError as e:
        L.error(f"Bleak error: {e}")

    except KeyboardInterrupt:
        pass

    return list(devsFound.keys())


DeviceType = T.Dict[str, T.Any]


def device_new(
    *,
    macaddr: str,
    name: T.Optional[str],
    category: T.Optional[OC.DeviceCategory],
    user: T.Optional[int],
    enabled: T.Optional[bool],
) -> T.Optional[DeviceType]:
    questions = []
    if name is None:
        questions.append(
            inquirer.Text(
                name="name",
                message="Name of the device",
                default="",
            )
        )

    if category is None:
        questions.append(
            inquirer.List(
                "category",
                message="Type of the device",
                choices=list(OC.DeviceCategory.__members__.keys()),
                default="SCALE",
            )
        )

    if user is None:
        questions.append(
            inquirer.List(
                "user",
                message="User number on the device",
                default=1,
                choices=[1, 2, 3, 4],
            )
        )

    if enabled is None:
        questions.append(
            inquirer.List(
                name="enabled",
                message="Enable device",
                default=True,
                choices=[True, False],
            )
        )

    device = {
        "macaddr": macaddr,
        "name": name,
        "category": category,
        "user": user,
        "enabled": enabled,
    }

    if questions:
        answers = inquirer.prompt(questions)
        if not answers:
            return None

        device.update(answers)

    return device


def device_edit(device: DeviceType) -> bool:
    questions = [
        inquirer.Text(
            name="name",
            message="Name of the device",
            default=device.get("name", ""),
        ),
        inquirer.List(
            "category",
            message="Type of the device",
            choices=["SCALE", "BPM"],
            default=device.get("category", "SCALE"),
        ),
        inquirer.List(
            "user",
            message="User number on the device",
            default=device.get("user", 1),
            choices=[1, 2, 3, 4],
        ),
        inquirer.List(
            name="enabled",
            message="Enable device",
            default=device.get("enabled", True),
            choices=[True, False],
        ),
    ]

    answers = inquirer.prompt(questions)
    if not answers:
        return False

    device["name"] = answers["name"] or OC.ble_mac_to_serial(device["macaddr"])
    device["category"] = answers["category"]
    device["user"] = answers["user"]
    device["enabled"] = answers["enabled"]

    return True


def omron_sync_device_to_garmin(
    oc: OC.OmronClient, gc: GC.Garmin, ocDev: OC.OmronDevice, startLocal: int, endLocal: int, opts: Options
) -> None:
    if endLocal - startLocal <= 0:
        L.info("Invalid date range")
        return

    startdateStr = datetime.fromtimestamp(startLocal).date().isoformat()
    enddateStr = datetime.fromtimestamp(endLocal).date().isoformat()

    L.info(f"Start synchronizing device '{ocDev.name}' from {startdateStr} to {enddateStr}")
    L.debug(
        f"Device details: MAC={ocDev.macaddr}, Serial={ocDev.serial}, User={ocDev.user}, Category={ocDev.category.name}"
    )

    measurements = oc.get_measurements(ocDev, searchDateFrom=int(startLocal * 1000), searchDateTo=int(endLocal * 1000))
    if not measurements:
        L.info("No new measurements")
        return

    L.info(f"Downloaded {len(measurements)} entries from 'OMRON connect' for '{ocDev.name}'")
    first_ts = datetime.fromtimestamp(measurements[0].measurementDate / 1000).isoformat()
    last_ts = datetime.fromtimestamp(measurements[-1].measurementDate / 1000).isoformat()
    L.debug(f"Measurement date range: {first_ts} to {last_ts}")

    # get measurements from Garmin Connect for the same date range
    category = "weigh-ins" if ocDev.category == OC.DeviceCategory.SCALE else "blood pressure"
    L.debug(f"Fetching existing Garmin {category} for date range {startdateStr} to {enddateStr}")

    if ocDev.category == OC.DeviceCategory.SCALE:
        gcData = garmin_get_weighins(gc, startdateStr, enddateStr)
        sync_scale_measurements(gc, gcData, measurements, opts)

    elif ocDev.category == OC.DeviceCategory.BPM:
        gcData = garmin_get_bp_measurements(gc, startdateStr, enddateStr)
        sync_bp_measurements(gc, gcData, measurements, opts)


def sync_measurements(
    gcData: T.Dict[str, T.Any],
    measurements: T.List[OC.MeasurementTypes],
    handler: MeasurementSyncHandler,
    opts: Options,
) -> None:
    """Generic measurement sync with deduplication.

    Args:
        gcData: Existing Garmin data mapped by key
        measurements: List of measurements from OMRON
        handler: Handler for measurement-specific operations
        opts: Sync options
    """

    L.debug(f"Starting sync of {len(measurements)} {handler.log_name} measurements")
    L.debug(f"Found {len(gcData)} existing {handler.log_name} records in Garmin Connect")

    skipped = 0
    added = 0

    for measurement in measurements:
        # Common timestamp handling
        tz = measurement.timeZone
        ts = measurement.measurementDate / 1000
        dtUTC = U.utcfromtimestamp(ts)
        dtLocal = datetime.fromtimestamp(ts, tz=tz)

        datetimeStr = dtLocal.isoformat(timespec="seconds")
        dateStr = dtLocal.date().isoformat()
        lookup = f"{dtUTC.date().isoformat()}:{dtUTC.timestamp()}"

        L.debug(f"Processing measurement: local={datetimeStr}, UTC={dtUTC.isoformat()}, lookup={lookup}")

        # Common deduplication logic
        if lookup in gcData.values():
            if opts.overwrite:
                L.warning(f"  ! '{datetimeStr}': removing {handler.log_name}")
                for key, val in gcData.items():
                    if val == lookup and opts.write_to_garmin:
                        handler.delete_measurement(key=key, cdate=dateStr)

            else:
                L.info(f"  - '{datetimeStr}' {handler.log_name} already exists")
                L.debug(f"Skipping duplicate {handler.log_name} (lookup={lookup})")
                skipped += 1
                continue

        # Call handler's add function (handler knows the expected type)
        L.debug(f"Adding new {handler.log_name} for {datetimeStr}")
        handler.add_measurement(dtLocal, measurement, opts)  # type: ignore[arg-type]
        added += 1

    L.debug(f"Sync complete: {added} added, {skipped} skipped")


def _add_scale_measurement(gc: GC.Garmin, dtLocal: datetime, wm: OC.WeightMeasurement, opts: Options) -> None:
    """Add a scale measurement to Garmin Connect."""

    datetimeStr = dtLocal.isoformat(timespec="seconds")
    L.info(f"  + '{datetimeStr}' adding weigh-in: {wm.weight} kg ")
    L.debug(
        f"Scale measurement details: weight={wm.weight}kg, BMI={wm.bmiValue}, body_fat={wm.bodyFatPercentage}%, "
        f"skeletal_muscle={wm.skeletalMusclePercentage}%, visceral_fat={wm.visceralFatLevel}"
    )
    if opts.write_to_garmin:
        L.debug(f"Writing scale measurement to Garmin Connect at {datetimeStr}")
        gc.add_body_composition(
            timestamp=datetimeStr,
            weight=wm.weight,
            percent_fat=wm.bodyFatPercentage if wm.bodyFatPercentage > 0 else None,
            percent_hydration=None,
            visceral_fat_mass=None,
            bone_mass=None,
            muscle_mass=((wm.skeletalMusclePercentage * wm.weight) / 100 if wm.skeletalMusclePercentage > 0 else None),
            basal_met=wm.restingMetabolism if wm.restingMetabolism > 0 else None,
            active_met=None,
            physique_rating=None,
            metabolic_age=wm.metabolicAge if wm.metabolicAge > 0 else None,
            visceral_fat_rating=wm.visceralFatLevel if wm.visceralFatLevel > 0 else None,
            bmi=wm.bmiValue,
        )
        L.debug("Scale measurement written successfully")


def _add_bp_measurement(gc: GC.Garmin, dtLocal: datetime, bpm: OC.BPMeasurement, opts: Options) -> None:
    """Add a blood pressure measurement to Garmin Connect."""

    datetimeStr = dtLocal.isoformat(timespec="seconds")

    notes = bpm.notes
    if bpm.movementDetect:
        notes = f"{notes}, Body Movement detected"

    if bpm.irregularHB:
        notes = f"{notes}, Irregular heartbeat detected"

    if not bpm.cuffWrapDetect:
        notes = f"{notes}, Cuff wrap error"

    if notes:
        notes = notes.lstrip(", ")

    L.info(f"  + '{datetimeStr}' adding blood pressure ({bpm.systolic}/{bpm.diastolic} mmHg, {bpm.pulse} bpm)")
    L.debug(f"BP measurement details: systolic={bpm.systolic}, diastolic={bpm.diastolic}, pulse={bpm.pulse}")
    L.debug(
        f"BP quality flags: movement={bpm.movementDetect}, irregular_hb={bpm.irregularHB}, "
        f"cuff_ok={bpm.cuffWrapDetect}, notes='{notes}'"
    )
    if opts.write_to_garmin:
        L.debug(f"Writing BP measurement to Garmin Connect at {datetimeStr}")
        gc.set_blood_pressure(
            timestamp=datetimeStr, systolic=bpm.systolic, diastolic=bpm.diastolic, pulse=bpm.pulse, notes=notes
        )
        L.debug("BP measurement written successfully")


def sync_scale_measurements(
    gc: GC.Garmin, gcData: T.Dict[str, T.Any], measurements: T.List[OC.MeasurementTypes], opts: Options
):
    """Sync scale measurements using the generic sync_measurements function."""

    handler = MeasurementSyncHandler(
        fetch_garmin_data=garmin_get_weighins,
        delete_measurement=lambda key, cdate: gc.delete_weigh_in(weight_pk=key, cdate=cdate),
        add_measurement=lambda dtLocal, wm, opts: _add_scale_measurement(gc, dtLocal, wm, opts),
        measurement_type=OC.WeightMeasurement,
        delete_key_field="samplePk",
        log_name="weigh-in",
    )
    sync_measurements(gcData, measurements, handler, opts)


def sync_bp_measurements(
    gc: GC.Garmin, gcData: T.Dict[str, T.Any], measurements: T.List[OC.MeasurementTypes], opts: Options
):
    """Sync blood pressure measurements using the generic sync_measurements function."""

    handler = MeasurementSyncHandler(
        fetch_garmin_data=garmin_get_bp_measurements,
        delete_measurement=lambda key, cdate: gc.delete_blood_pressure(version=key, cdate=cdate),
        add_measurement=lambda dtLocal, bpm, opts: _add_bp_measurement(gc, dtLocal, bpm, opts),
        measurement_type=OC.BPMeasurement,
        delete_key_field="version",
        log_name="blood pressure",
    )
    sync_measurements(gcData, measurements, handler, opts)


def garmin_get_bp_measurements(gc: GC.Garmin, startdate: str, enddate: str) -> T.Dict[str, str]:
    # search dates are in local time
    gcData = gc.get_blood_pressure(startdate=startdate, enddate=enddate)

    # reduce to list of measurements
    _gcMeasurements = [metric for x in gcData["measurementSummaries"] for metric in x["measurements"]]

    # map of garmin-key:omron-key
    gcMeasurements = {}
    for metric in _gcMeasurements:
        # use UTC for comparison
        dtUTC = datetime.fromisoformat(f"{metric['measurementTimestampGMT']}Z")
        gcMeasurements[metric["version"]] = f"{dtUTC.date().isoformat()}:{dtUTC.timestamp()}"

    L.info(f"Downloaded {len(gcMeasurements)} bpm measurements from 'Garmin Connect'")
    return gcMeasurements


def garmin_get_weighins(gc: GC.Garmin, startdate: str, enddate: str) -> T.Dict[str, str]:
    # search dates are in local time
    gcData = gc.get_weigh_ins(startdate=startdate, enddate=enddate)

    # reduce to list of allWeightMetrics
    _gcWeighins = [metric for x in gcData["dailyWeightSummaries"] for metric in x["allWeightMetrics"]]

    # map of garmin-key:omron-key
    gcWeighins = {}
    for metric in _gcWeighins:
        # use UTC for comparison
        dtUTC = U.utcfromtimestamp(int(metric["timestampGMT"]) / 1000)
        gcWeighins[metric["samplePk"]] = f"{dtUTC.date().isoformat()}:{dtUTC.timestamp()}"

    L.info(f"Downloaded {len(gcWeighins)} weigh-ins from 'Garmin Connect'")

    return gcWeighins


########################################################################################################################
class DateRangeException(Exception):
    pass


def calculate_date_range(days: int) -> T.Tuple[int, int]:
    days = max(days, 0)
    today = datetime.combine(datetime.today().date(), datetime.max.time())
    start = today - timedelta(days=days)
    start = datetime.combine(start, datetime.min.time())
    startLocal = start.timestamp()
    endLocal = today.timestamp()
    if endLocal - startLocal <= 0:
        raise DateRangeException()

    return int(startLocal), int(endLocal)


def parse_date_string(date_str: str) -> datetime:
    """Parse flexible date formats (yyyymmdd, yyyy-mm-dd, etc.) to datetime at midnight."""

    try:
        parsed_date = dateutil_parser.parse(date_str)
        return datetime.combine(parsed_date.date(), datetime.min.time())

    except (ValueError, TypeError, dateutil_parser.ParserError) as e:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected formats: yyyymmdd, yyyy-mm-dd, yyyy/mm/dd, etc. Error: {e}"
        ) from e


def calculate_date_range_from_options(
    *,
    from_date: T.Optional[str] = None,
    to_date: T.Optional[str] = None,
    days: T.Optional[int] = None,
) -> T.Tuple[int, int]:
    """Calculate date range supporting --from/--to/--days combinations. Future dates clamped to today."""

    if from_date and to_date and days is not None:
        raise DateRangeException(
            "Cannot specify all three options (--from, --to, --days) together. "
            "Use --from with --to, --from with --days, or --to with --days."
        )

    today_end = datetime.combine(datetime.today().date(), datetime.max.time())
    today_start = datetime.combine(datetime.today().date(), datetime.min.time())

    # Pattern 1: --from DATE --to DATE (explicit range)
    if from_date and to_date:
        start = parse_date_string(from_date)
        end = parse_date_string(to_date)
        end = datetime.combine(end.date(), datetime.max.time())

    # Pattern 2: --from DATE --days N (start date plus N days)
    elif from_date and days is not None:
        days = max(days, 0)
        start = parse_date_string(from_date)
        end = start + timedelta(days=days)
        end = datetime.combine(end.date(), datetime.max.time())

    # Pattern 3: --to DATE --days N (end date minus N days)
    elif to_date and days is not None:
        days = max(days, 0)
        end = parse_date_string(to_date)
        end = datetime.combine(end.date(), datetime.max.time())
        start = end - timedelta(days=days)
        start = datetime.combine(start.date(), datetime.min.time())

    # Pattern 4: --days N only (backward compatibility)
    elif days is not None:
        return calculate_date_range(days)

    # Pattern 5: --from DATE only (from date to today)
    elif from_date:
        start = parse_date_string(from_date)
        end = today_end

    # Pattern 6: --to DATE only (single day on specified date)
    elif to_date:
        date = parse_date_string(to_date)
        start = datetime.combine(date.date(), datetime.min.time())
        end = datetime.combine(date.date(), datetime.max.time())

    else:
        start = today_start
        end = today_end

    end = min(end, today_end)
    start = min(start, today_start)

    startLocal = int(start.timestamp())
    endLocal = int(end.timestamp())

    if endLocal - startLocal <= 0:
        raise DateRangeException(
            f"Invalid date range: start date must be before end date. Start: {start.date()}, End: {end.date()}"
        )

    return startLocal, endLocal


def filter_devices(
    devices: T.List[T.Dict[str, T.Any]],
    *,
    devnames: T.Optional[T.List[str]] = None,
    category: T.Optional[OC.DeviceCategory] = None,
) -> T.List[T.Dict[str, T.Any]]:
    devices = [d for d in devices if d["enabled"]]
    if category:
        devices = [d for d in devices if d["category"] == category.name]

    if devnames:
        devices = [d for d in devices if d["name"] in devnames or d["macaddr"] in devnames]

    return devices


def find_device(
    devices: T.List[T.Dict[str, T.Any]],
    identifier: str,
    *,
    prompt_if_empty: bool = False,
    prompt_message: str = "Select device",
) -> T.Optional[T.Dict[str, T.Any]]:
    """Find device by name or MAC address with optional interactive prompt.

    Args:
        devices: List of device dictionaries
        identifier: Device name or MAC address to find
        prompt_if_empty: If True and identifier is empty, prompt user to select
        prompt_message: Message for interactive prompt

    Returns:
        Device dictionary if found, None otherwise
    """

    if not identifier and prompt_if_empty:
        macaddrs = [d["macaddr"] for d in devices]
        identifier = inquirer.list_input(prompt_message, choices=sorted(macaddrs))

    if not identifier:
        return None

    return next(
        (d for d in devices if d.get("name") == identifier or d.get("macaddr") == identifier),
        None,
    )


########################################################################################################################


def _get_xdg_config_path() -> pathlib.Path:
    """Get XDG config path: $XDG_CONFIG_HOME/omramin/config.json or ~/.config/omramin/config.json"""

    xdg_config_home = _E("XDG_CONFIG_HOME")
    if xdg_config_home:
        return pathlib.Path(xdg_config_home) / CONFIG_DIR_NAME / CONFIG_FILENAME

    return pathlib.Path(PATH_XDG_CONFIG).expanduser().resolve()


def _get_platform_config_path() -> pathlib.Path:
    """Get platform-native config path without checking if it exists.

    Returns:
        - Windows: %APPDATA%/omramin/config.json
        - macOS: ~/Library/Application Support/omramin/config.json
        - Linux/Unix: ~/.config/omramin/config.json (XDG)
    """

    system = platform.system()

    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return pathlib.Path(appdata) / CONFIG_DIR_NAME / CONFIG_FILENAME

        return pathlib.Path.home() / "AppData" / "Roaming" / CONFIG_DIR_NAME / CONFIG_FILENAME

    elif system == "Darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / CONFIG_DIR_NAME / CONFIG_FILENAME

    else:
        return _get_xdg_config_path()


def _get_default_config_path() -> pathlib.Path:
    """Get default config path with precedence:

    1. OMRAMIN_CONFIG env var (if set)
    2. ./config.json (if exists in current directory)
    3. XDG_CONFIG_HOME path (if XDG_CONFIG_HOME env var is set AND path exists)
    4. ~/.config/omramin/config.json (if exists)
    5. Platform-native path (if exists)
    6. ~/.omramin/config.json (if exists)
    7. Defaults to platform-native path for new installations
    """

    if env_path := _E("OMRAMIN_CONFIG"):
        return pathlib.Path(env_path).expanduser().resolve()

    if pathlib.Path(f"./{CONFIG_FILENAME}").exists():
        return pathlib.Path(f"./{CONFIG_FILENAME}").resolve()

    if _E("XDG_CONFIG_HOME"):
        xdg_path = _get_xdg_config_path()
        if xdg_path.exists():
            return xdg_path

    xdg_default_path = pathlib.Path(PATH_XDG_CONFIG).expanduser().resolve()
    if xdg_default_path.exists():
        return xdg_default_path

    platform_path = _get_platform_config_path()
    if platform_path.exists():
        return platform_path

    legacy_path = pathlib.Path(PATH_DEFAULT_CONFIG).expanduser().resolve()
    if legacy_path.exists():
        return legacy_path

    return platform_path


# envvar="OMRAMIN_CONFIG" would break:
# OMRAMIN_CONFIG=/wrong/path omramin --config /right/path xxx
# Would incorrectly use /wrong/path instead of /right/path
@click.group()
@click.version_option(__version__)
@click.option(
    "--config",
    "config_path",
    type=click.Path(writable=True, dir_okay=False),
    default=_get_default_config_path(),
    show_default=True,
    help="Config file path",
)
@click.option(
    "--keyring-backend",
    type=click.Choice(list(KeyringBackend), case_sensitive=False),
    default=None,
    show_default=False,
    envvar="OMRAMIN_KEYRING_BACKEND",
    help="Token storage backend: system (auto-detect), file (plaintext), encrypted (password-protected)",
)
@click.option(
    "--keyring-file",
    type=click.Path(writable=True, dir_okay=False),
    default=None,
    show_default=False,
    envvar="OMRAMIN_KEYRING_FILE",
    help="File path for file/encrypted backends (default: {config}.tokens.json)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    envvar="OMRAMIN_DEBUG",
    help="Enable debug logging (shows detailed HTTP traffic and internal operations)",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: click.Path,
    keyring_backend: T.Optional[str],
    keyring_file: T.Optional[str],
    debug: bool,
):
    """Sync data from 'OMRON connect' to 'Garmin Connect'

    Global options must be specified BEFORE the command:
        omramin --debug sync --days 1
    """

    _config_path = str(config_path)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = _config_path

    if keyring_backend:
        os.environ["OMRAMIN_KEYRING_BACKEND"] = keyring_backend.lower()

    if keyring_file:
        os.environ["OMRAMIN_KEYRING_FILE"] = str(pathlib.Path(keyring_file).expanduser().resolve())

    _configure_logging_levels(debug=debug)


########################################################################################################################


@cli.command(name="list")
@click.pass_context
@requires_config
def list_devices(ctx: click.Context):
    """List all configured devices."""

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        L.error(f"Config file '{config_path}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])
    if not devices:
        L.info("No devices configured.")
        return

    for device in devices:
        L.info("-" * 40)
        L.info(f"Name:{' ':<8}{device.get('name', 'Unknown')}")
        L.info(f"MAC Address:{' ':<1}{device.get('macaddr', 'Unknown')}")
        L.info(f"Category:{' ':<4}{device.get('category', 'Unknown')}")
        L.info(f"User:{' ':<8}{device.get('user', 'Unknown')}")
        L.info(f"Enabled:{' ':<5}{device.get('enabled', 'Unknown')}")

    if devices:
        L.info("-" * 40)


########################################################################################################################


@cli.command(name="add")
@click.option(
    "--macaddr",
    "-m",
    required=False,
    help="MAC address of the device to add. If not provided, scan for new devices.",
)
@click.option(
    "--name",
    "-n",
    required=False,
    help="Name of the device to add. If not provided, the serial number will be used.",
)
@click.option(
    "--category",
    "-c",
    required=False,
    type=click.Choice(list(OC.DeviceCategory.__members__.keys()), case_sensitive=False),
    help="Category of the device (SCALE or BPM).",
)
@click.option(
    "--user",
    "-u",
    required=False,
    type=click.INT,
    default=1,
    show_default=True,
    help="User number on the device (1-4).",
)
@click.option("--ble-filter", help="BLE device name filter", default=Options().ble_filter, show_default=True)
@click.pass_context
def add_device(
    ctx: click.Context,
    macaddr: T.Optional[str],
    name: T.Optional[str],
    category: T.Optional[OC.DeviceCategory],
    user: T.Optional[int],
    ble_filter: T.Optional[str],
):
    """Add a new Omron device to the configuration.

    This function allows adding a new Omron device either by providing a MAC address directly
    or by scanning for available devices.

    \b
    Examples:
        # Scan and select device interactively
        python omramin.py add
    \b
        # Add device by MAC address
        python omramin.py add -m 00:11:22:33:44:55
        python omramin.py add -m 00:11:22:33:44:55 -c scale -n "My Scale" -u 3


    """

    config_path = ctx.obj["config_path"]

    opts = Options()
    if ble_filter is not None:
        opts.ble_filter = ble_filter

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    devices = config.get("omron", {}).get("devices", [])

    if not macaddr:
        macAddrs = [d["macaddr"] for d in devices]

        try:
            bleDevices = omron_ble_scan(macAddrs, opts)

        except ImportError as e:
            L.error(str(e))
            L.info("Alternative: Add device manually with MAC address")
            L.info("Example: omramin add --macaddr 00:11:22:33:44:55")
            return

        if not bleDevices:
            L.info("No devices found.")
            return

        # make sure we don't add the same device twice
        tmp = bleDevices.copy()
        for scanned in bleDevices:
            if any(d["macaddr"] == scanned for d in devices):
                tmp.remove(scanned)

        bleDevices = tmp

        if not bleDevices:
            L.info("No new devices found.")
            return

        macaddr = inquirer.list_input("Select device", choices=sorted(bleDevices))

    if macaddr:
        if not U.is_valid_macaddr(macaddr):
            L.error(f"Invalid MAC address: {macaddr}")
            return

        if macaddr in [d["macaddr"] for d in devices]:
            L.info(f"Device '{macaddr}' already exists.")
            return

        if device := device_new(macaddr=macaddr, name=name, category=category, user=user, enabled=True):
            config["omron"]["devices"].append(device)
            try:
                with config_write_handler(config_path, config):
                    pass  # device already appended to config

                L.info("Device(s) added successfully.")

            except (OSError, IOError, PermissionError, ValueError):
                pass  # Error already logged by context manager


########################################################################################################################


@cli.command(name="config")
@click.argument("devname", required=True, type=str, nargs=1)
@click.pass_context
@requires_config
def edit_device(ctx: click.Context, devname: str):
    """Edit device configuration."""

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        L.error(f"Config file '{config_path}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])
    if not devices:
        L.info("No devices configured.")
        return

    device = find_device(devices, devname, prompt_if_empty=True, prompt_message="Select device to configure")
    if not device:
        L.info(f"No device found with identifier: '{devname}'")
        return

    if device_edit(device):
        try:
            with config_write_handler(config_path, config):
                pass  # device modified in-place

            L.info(f"Device '{devname}' configured successfully.")

        except (OSError, IOError, PermissionError, ValueError):
            pass  # Error already logged by context manager


########################################################################################################################


@cli.command(name="remove")
@click.argument("devname", required=True, type=str, nargs=1)
@click.pass_context
@requires_config
def remove_device(ctx: click.Context, devname: str):
    """Remove a device by name or MAC address."""

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        L.error(f"Config file '{config_path}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])

    device = find_device(devices, devname, prompt_if_empty=True, prompt_message="Select device to remove")
    if not device:
        L.info(f"No device found with identifier: {devname}")
        return

    devices.remove(device)
    try:
        with config_write_handler(config_path, config):
            pass  # device already removed

        L.info(f"Device '{devname}' removed successfully.")

    except (OSError, IOError, PermissionError, ValueError):
        pass  # Error already logged by context manager


########################################################################################################################


@cli.command(name="sync")
@click.argument("devnames", required=False, nargs=-1)
@click.option(
    "--category",
    "-c",
    "device_category",
    required=False,
    type=click.Choice(list(OC.DeviceCategory.__members__.keys()), case_sensitive=False),
)
@click.option("--days", default=None, type=click.INT, help="Number of days. Can combine with --from or --to.")
@click.option(
    "--from",
    "from_date",
    type=click.STRING,
    default=None,
    help="Start date (yyyymmdd or yyyy-mm-dd). Can combine with --days or --to.",
)
@click.option(
    "--to",
    "to_date",
    type=click.STRING,
    default=None,
    help="End date (yyyymmdd or yyyy-mm-dd). Can combine with --days or --from.",
)
@click.option(
    "--overwrite", is_flag=True, default=Options().overwrite, show_default=True, help="Overwrite existing measurements."
)
@click.option(
    "--no-write",
    is_flag=True,
    default=not Options().write_to_garmin,
    show_default=True,
    help="Do not write to Garmin Connect.",
)
@click.pass_context
@requires_config
def sync_device(
    ctx: click.Context,
    devnames: T.List[str],
    device_category: T.Optional[str],
    days: T.Optional[int],
    from_date: T.Optional[str],
    to_date: T.Optional[str],
    overwrite: bool,
    no_write: bool,
):
    """Sync DEVNAMES... to Garmin Connect.

    \b
    DEVNAMES: List of Names or MAC addresses for the device to sync. [default: ALL]

    \b
    Date Range Options:
        --days N              Sync last N days from today (default: 0, today only)
        --from DATE           Start date (yyyymmdd or yyyy-mm-dd format)
        --to DATE             End date (yyyymmdd or yyyy-mm-dd format)
        --from DATE --to DATE Explicit date range
        --from DATE --days N  Start date plus N days
        --to DATE --days N    End date minus N days

    \b
    Examples:
        # Sync all devices for today only
        omramin sync
    \b
        # Sync all devices for the last 7 days
        omramin sync --days 7
    \b
        # Sync a specific device for the last 1 day
        omramin sync "my scale" --days 1
    \b
        # Sync with explicit date range
        omramin sync --from 20240101 --to 20240131
    \b
        # Sync with separators in date format
        omramin sync --from 2024-01-01 --to 2024-01-31
    \b
        # Start date plus 7 days
        omramin sync --from 20240101 --days 7
    \b
        # End date minus 7 days
        omramin sync --to 20240131 --days 7
    """

    config_path = ctx.obj["config_path"]

    opts = Options()
    opts.overwrite = overwrite
    opts.write_to_garmin = not no_write

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        L.error(f"Config file '{config_path}' not found.")
        return

    category = OC.DeviceCategory[device_category] if device_category else None

    devices = config.get("omron", {}).get("devices", [])
    if not devices:
        L.info("No devices configured.")
        return

    # Default to 0 days if no date options provided
    if days is None and from_date is None and to_date is None:
        days = 0

    try:
        startLocal, endLocal = calculate_date_range_from_options(
            from_date=from_date,
            to_date=to_date,
            days=days,
        )

    except DateRangeException as e:
        L.error(f"Invalid date range: {e}")
        return

    except ValueError as e:
        L.error(f"Date parsing error: {e}")
        return

    # filter devices by enabled, category and name/mac address
    devices = filter_devices(devices, devnames=devnames, category=category)
    if not devices:
        L.info("No matching devices found")
        return

    try:
        gc = garmin_login(config_path)

    except LoginError:
        L.info("Failed to login to Garmin Connect.")
        return

    try:
        oc = omron_login(config_path)

    except LoginError:
        L.info("Failed to login to OMRON connect.")
        return

    if not oc or not gc:
        L.info("Failed to login to OMRON connect or Garmin Connect.")
        return

    for device in devices:
        ocDev = OC.OmronDevice(**device)
        omron_sync_device_to_garmin(oc, gc, ocDev, startLocal, endLocal, opts=opts)
        L.info(f"Device '{device['name']}' successfully synced.")


########################################################################################################################


########################################################################################################################
# Garmin authentication commands
########################################################################################################################


@cli.group(name="garmin")
@click.pass_context
def garmin_group(_ctx: click.Context) -> None:
    """Manage Garmin Connect authentication."""


@garmin_group.command(name="login")
@click.option("--email", help="Garmin email address")
@click.option("--password", help="Garmin password (WARNING: exposes password in shell history)")
@click.option("--is-cn", is_flag=True, default=None, help="Chinese account flag")
@click.option("--force", is_flag=True, help="Force re-login even if tokens exist")
@click.pass_context
def garmin_login_cmd(ctx: click.Context, email: str, password: str, is_cn: bool, force: bool) -> None:
    """Login to Garmin Connect and save authentication tokens.

    By default, skips login if valid tokens already exist. Use --force to re-login.

    Credential precedence: CLI option > Environment variable > config.json > interactive prompt
    """

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    config = migrateconfig_path(config)
    garminCfg = config.get("garmin", {})

    # Get email from CLI, env, or config
    final_email = email or _E("GARMIN_EMAIL") or garminCfg.get("email", "")

    # Check if already logged in (unless --force)
    if not force and final_email:
        tokendata = load_service_tokens(config_path, "garmin", final_email, migrate_from_config=config)
        if tokendata:
            L.info(f"Already logged in as {final_email}. Use --force to re-login.")
            return

    # Set CLI options as env vars temporarily to give them precedence
    original_env = {}
    try:
        if email:
            original_env["GARMIN_EMAIL"] = os.environ.get("GARMIN_EMAIL")
            os.environ["GARMIN_EMAIL"] = email

        if password:
            original_env["GARMIN_PASSWORD"] = os.environ.get("GARMIN_PASSWORD")
            os.environ["GARMIN_PASSWORD"] = password

        if is_cn is not None:
            original_env["GARMIN_IS_CN"] = os.environ.get("GARMIN_IS_CN")
            os.environ["GARMIN_IS_CN"] = "1" if is_cn else "0"

        # Use existing login logic
        gc = garmin_login(config_path)
        if gc:
            login_email = _E("GARMIN_EMAIL") or garminCfg.get("email", "")
            L.info(f"Successfully logged in to Garmin as {login_email}")

        else:
            L.error("Failed to login to Garmin")

    except LoginError as e:
        L.error(f"Garmin login failed: {e}")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)

            else:
                os.environ[key] = value


@garmin_group.command(name="logout")
@click.option("--email", help="Email address (default: from config)")
@click.option("--clear-config", is_flag=True, help="Also clear email/is_cn from config")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def garmin_logout_cmd(ctx: click.Context, email: str, clear_config: bool, yes: bool) -> None:
    """Logout from Garmin Connect by clearing authentication tokens.

    By default, only clears tokens (keeps config entries). Use --clear-config to also remove
    email/is_cn from config file.
    """

    config_path = ctx.obj["config_path"]

    # Get email from option or config
    if not email:
        try:
            config = U.json_load(config_path)
            config = migrateconfig_path(config)
            garminCfg = config.get("garmin", {})
            email = garminCfg.get("email", "")

        except FileNotFoundError:
            pass

    if not email:
        L.error("Cannot logout: email not found in config and --email not provided")
        return

    # Confirmation prompt
    if not yes:
        action = "clear tokens and config entries" if clear_config else "clear tokens"
        click.echo(f"Logout from Garmin Connect ({email}): {action}")
        if not click.confirm("Continue?", default=False):
            L.info("Logout cancelled")
            return

    # Clear tokens
    tokens_cleared = clear_service_tokens(config_path, "garmin", email)
    if tokens_cleared:
        L.info(f"Cleared Garmin tokens for {email}")

    else:
        L.info(f"No Garmin tokens found for {email}")

    # Clear config if requested
    if clear_config:
        try:
            config = U.json_load(config_path)
            config = migrateconfig_path(config)

            if "garmin" in config:
                garminCfg = config["garmin"]
                removed = []
                for key in ["email", "is_cn"]:
                    if key in garminCfg:
                        del garminCfg[key]
                        removed.append(key)

                if removed:
                    with config_write_handler(config_path, config):
                        pass  # config_write_handler saves automatically

                    L.info(f"Removed from config: {', '.join(removed)}")

                else:
                    L.info("No Garmin config entries to remove")

            else:
                L.info("No Garmin config section found")

        except FileNotFoundError:
            L.info("No config file found, nothing to clear")

        except (OSError, IOError, PermissionError) as e:
            L.warning(f"Config file is read-only, cannot clear config entries: {e}")
            L.info("Tokens were cleared successfully")


########################################################################################################################
# OMRON authentication commands
########################################################################################################################


@cli.group(name="omron")
@click.pass_context
def omron_group(_ctx: click.Context):
    """Manage OMRON Connect authentication."""


@omron_group.command(name="login")
@click.option("--email", help="OMRON email address")
@click.option("--password", help="OMRON password (WARNING: exposes password in shell history)")
@click.option("--country", help="Country code (e.g., 'US')")
@click.option("--force", is_flag=True, help="Force re-login even if tokens exist")
@click.pass_context
def omron_login_cmd(ctx: click.Context, email: str, password: str, country: str, force: bool):
    """Login to OMRON Connect and save authentication tokens.

    By default, skips login if valid tokens already exist. Use --force to re-login.

    Credential precedence: CLI option > Environment variable > config.json > interactive prompt
    """

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    config = migrateconfig_path(config)
    ocCfg = config.get("omron", {})

    # Get email from CLI, env, or config (support legacy "username" field)
    final_email = email or _E("OMRON_EMAIL") or ocCfg.get("email", "") or ocCfg.get("username", "")

    # Check if already logged in (unless --force)
    if not force and final_email:
        tokendata = load_service_tokens(config_path, "omron", final_email, migrate_from_config=config)
        if tokendata:
            L.info(f"Already logged in as {final_email}. Use --force to re-login.")
            return

    # Set CLI options as env vars temporarily to give them precedence
    original_env = {}
    try:
        if email:
            original_env["OMRON_EMAIL"] = os.environ.get("OMRON_EMAIL")
            os.environ["OMRON_EMAIL"] = email

        if password:
            original_env["OMRON_PASSWORD"] = os.environ.get("OMRON_PASSWORD")
            os.environ["OMRON_PASSWORD"] = password

        if country:
            original_env["OMRON_COUNTRY"] = os.environ.get("OMRON_COUNTRY")
            os.environ["OMRON_COUNTRY"] = country

        # Use existing login logic
        oc = omron_login(config_path)
        if oc:
            login_email = _E("OMRON_EMAIL") or ocCfg.get("email", "") or ocCfg.get("username", "")
            L.info(f"Successfully logged in to OMRON as {login_email}")

        else:
            L.error("Failed to login to OMRON")

    except LoginError as e:
        L.error(f"OMRON login failed: {e}")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)

            else:
                os.environ[key] = value


@omron_group.command(name="logout")
@click.option("--email", help="Email address (default: from config)")
@click.option("--clear-config", is_flag=True, help="Also clear email/country from config")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def omron_logout_cmd(ctx: click.Context, email: str, clear_config: bool, yes: bool) -> None:
    """Logout from OMRON Connect by clearing authentication tokens.

    By default, only clears tokens (keeps config entries). Use --clear-config to also remove
    email/country from config file.
    """

    config_path = ctx.obj["config_path"]

    # Get email from option or config
    if not email:
        try:
            config = U.json_load(config_path)
            config = migrateconfig_path(config)
            ocCfg = config.get("omron", {})
            email = ocCfg.get("email", "") or ocCfg.get("username", "")

        except FileNotFoundError:
            pass

    if not email:
        L.error("Cannot logout: email not found in config and --email not provided")
        return

    # Confirmation prompt
    if not yes:
        action = "clear tokens and config entries" if clear_config else "clear tokens"
        click.echo(f"Logout from OMRON Connect ({email}): {action}")
        if not click.confirm("Continue?", default=False):
            L.info("Logout cancelled")
            return

    # Clear tokens
    tokens_cleared = clear_service_tokens(config_path, "omron", email)
    if tokens_cleared:
        L.info(f"Cleared OMRON tokens for {email}")

    else:
        L.info(f"No OMRON tokens found for {email}")

    # Clear config if requested
    if clear_config:
        try:
            config = U.json_load(config_path)
            config = migrateconfig_path(config)

            if "omron" in config:
                ocCfg = config["omron"]
                removed = []
                for key in ["email", "username", "country"]:
                    if key in ocCfg:
                        del ocCfg[key]
                        removed.append(key)

                if removed:
                    with config_write_handler(config_path, config):
                        pass  # config_write_handler saves automatically

                    L.info(f"Removed from config: {', '.join(removed)}")

                else:
                    L.info("No OMRON config entries to remove")

            else:
                L.info("No OMRON config section found")

        except FileNotFoundError:
            L.info("No config file found, nothing to clear")

        except (OSError, IOError, PermissionError) as e:
            L.warning(f"Config file is read-only, cannot clear config entries: {e}")
            L.info("Tokens were cleared successfully")


@omron_group.command(name="list")
@click.option(
    "--days",
    type=int,
    default=OMRON_DEVICE_LIST_DAYS,
    show_default=True,
    help="Limit to devices active in last N days (API v1 only, ignored by API v2)",
)
@click.option(
    "--all",
    "fetch_all",
    is_flag=True,
    help="Fetch all historical devices (API v1: may be slow, API v2: same as default)",
)
@click.pass_context
def omron_list_devices_cmd(ctx: click.Context, days: int, fetch_all: bool) -> None:
    """List all devices registered with OMRON Connect.

    Fetches and displays devices from OMRON Connect account,
    including their category, model name, MAC address, and user number.

    \b
    Notes:
    - API v1 (Asia/Pacific): Supports pagination and date filtering
    - API v2 (Europe/Americas): Returns all active devices (ignores --days)
    - Use --all to fetch complete device history (API v1 only)
    """

    config_path = ctx.obj["config_path"]

    # Determine days parameter
    days_param = None if fetch_all else days

    if fetch_all:
        L.info("Fetching all historical devices (may take a while)...")

    try:
        # Login to OMRON
        oc = omron_login(config_path)
        if not oc:
            L.error("Failed to login to OMRON")
            return

        # Fetch registered devices
        devices = oc.get_registered_devices(days=days_param)

        if not devices:
            L.info("No devices found in OMRON Connect")
            return

        # Display devices in formatted output
        L.info(f"Found {len(devices)} device(s) registered with OMRON Connect:\n")

        for device in devices:
            click.echo(f"Device: {device.name}")
            click.echo(f"  Category:    {device.category.name}")
            click.echo(f"  MAC Address: {device.macaddr}")
            click.echo(f"  User Number: {device.user}")
            click.echo()

    except LoginError as e:
        L.error(f"OMRON login failed: {e}")

    except Exception as e:  # pylint: disable=broad-except
        L.error(f"Failed to fetch devices: {e}")


@omron_group.command(name="export")
@click.argument("devnames", required=False, nargs=-1)
@click.option(
    "--category",
    "-c",
    "device_category",
    required=True,
    type=click.Choice(list(OC.DeviceCategory.__members__.keys()), case_sensitive=False),
)
@click.option("--days", default=None, type=click.INT, help="Number of days. Can combine with --from or --to.")
@click.option(
    "--from",
    "from_date",
    type=click.STRING,
    default=None,
    help="Start date (yyyymmdd or yyyy-mm-dd). Can combine with --days or --to.",
)
@click.option(
    "--to",
    "to_date",
    type=click.STRING,
    default=None,
    help="End date (yyyymmdd or yyyy-mm-dd). Can combine with --days or --from.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
@requires_config
def export_measurements(
    ctx,
    devnames: T.Optional[T.List[str]],
    device_category: str,
    days: T.Optional[int],
    from_date: T.Optional[str],
    to_date: T.Optional[str],
    output_format: T.Optional[str],
    output: T.Optional[str],
):
    """Export device measurements to CSV or JSON format.

    \b
    Date Range Options:
        --days N              Export last N days from today (default: 0, today only)
        --from DATE           Start date (yyyymmdd or yyyy-mm-dd format)
        --to DATE             End date (yyyymmdd or yyyy-mm-dd format)
        --from DATE --to DATE Explicit date range
        --from DATE --days N  Start date plus N days
        --to DATE --days N    End date minus N days

    \b
    Examples:
        # Export scale measurements for today only
        omramin omron export --category scale
    \b
        # Export scale measurements for last 30 days
        omramin omron export --category scale --days 30 -o output.csv
    \b
        # Export with explicit date range
        omramin omron export --category BPM --from 20240101 --to 20240131 --format json
    \b
        # Export with separators in date format
        omramin omron export --category BPM --from 2024-01-01 --to 2024-01-31
    \b
        # Start date plus 30 days
        omramin omron export --category scale --from 20240101 --days 30
    \b
        # End date minus 7 days
        omramin omron export --category BPM --to 20240131 --days 7
    """

    config_path = ctx.obj["config_path"]

    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        L.error(f"Config file '{config_path}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])
    category = OC.DeviceCategory[device_category]

    # Default to 0 days if no date options provided
    if days is None and from_date is None and to_date is None:
        days = 0

    try:
        startLocal, endLocal = calculate_date_range_from_options(
            from_date=from_date,
            to_date=to_date,
            days=days,
        )

    except DateRangeException as e:
        L.error(f"Invalid date range: {e}")
        return

    except ValueError as e:
        L.error(f"Date parsing error: {e}")
        return

    # filter devices by enabled, category and name/mac address
    devices = filter_devices(devices, devnames=devnames, category=category)
    if not devices:
        L.info("No matching devices found")
        return

    startdateStr = datetime.fromtimestamp(startLocal).date().isoformat()
    enddateStr = datetime.fromtimestamp(endLocal).date().isoformat()

    try:
        oc = omron_login(config_path)

    except LoginError:
        L.info("Failed to login to OMRON connect.")
        return

    if not oc:
        return

    exportdata: T.Dict[str, T.Tuple[OC.OmronDevice, T.List[OC.MeasurementTypes]]] = {}
    for device in devices:
        ocDev = OC.OmronDevice(**device)
        L.info(f"Exporting device '{ocDev.name}' from {startdateStr} to {enddateStr}")

        measurements = oc.get_measurements(
            ocDev, searchDateFrom=int(startLocal * 1000), searchDateTo=int(endLocal * 1000)
        )
        if measurements:
            exportdata[ocDev.serial] = (ocDev, measurements)

    if not exportdata:
        L.info("No measurements found")
        return

    if not output:
        output = (
            f"omron_{category.name}_{datetime.fromtimestamp(startLocal).date()}_"
            f"{datetime.fromtimestamp(endLocal).date()}.{output_format}"
        )

    if output_format == "json":
        export_json(output, exportdata)

    else:
        export_csv(output, exportdata)

    L.info(f"Exported {len(exportdata)} measurements to {output}")


def export_csv(
    output: str,
    exportdata: T.Dict[str, T.Tuple[OC.OmronDevice, T.List[OC.MeasurementTypes]]],
) -> None:
    with open(output, "w", newline="\n", encoding="utf-8") as f:
        writer = None
        for ocDev, measurements in exportdata.values():
            for m in measurements:
                dt = datetime.fromtimestamp(m.measurementDate / 1000, tz=m.timeZone)
                row = {
                    "timestamp": dt.isoformat(),
                    "deviceName": ocDev.name,
                    "deviceCategory": ocDev.category.name,
                }
                row.update(dataclasses.asdict(m))

                if writer is None:
                    writer = csv.DictWriter(
                        f, fieldnames=row.keys(), quotechar='"', quoting=csv.QUOTE_ALL, lineterminator="\n"
                    )
                    writer.writeheader()

                writer.writerow(row)


def export_json(
    output: str,
    exportdata: T.Dict[str, T.Tuple[OC.OmronDevice, T.List[OC.MeasurementTypes]]],
) -> None:
    data = []
    for ocDev, measurements in exportdata.values():
        for m in measurements:
            dt = datetime.fromtimestamp(m.measurementDate / 1000, tz=m.timeZone)
            entry = {
                "timestamp": dt.isoformat(),
                "deviceName": ocDev.name,
                "deviceCategory": ocDev.category.name,
            }
            entry.update(dataclasses.asdict(m))
            data.append(entry)

    U.json_save(output, data)


########################################################################################################################
# Init command - Interactive setup wizard
########################################################################################################################


@cli.command(name="init")
@click.option("--skip-garmin", is_flag=True, help="Skip Garmin Connect authentication")
@click.option("--skip-omron", is_flag=True, help="Skip OMRON Connect authentication")
@click.option("--skip-devices", is_flag=True, help="Skip device discovery/selection")
@click.option(
    "--discovery-method",
    type=click.Choice(["api", "ble", "ask"], case_sensitive=False),
    default="ask",
    help="Device discovery method (default: ask)",
)
@click.option("--ble-filter", help="BLE device name filter", default=Options().ble_filter, show_default=True)
@click.pass_context
def init_cmd(
    ctx: click.Context,
    skip_garmin: bool,
    skip_omron: bool,
    skip_devices: bool,
    discovery_method: str,
    ble_filter: str,
) -> None:
    """Initialize omramin: setup authentication and discover devices.

    \b
    This interactive wizard guides you through:
      1. Authenticating with Garmin Connect
      2. Authenticating with OMRON Connect
      3. Discovering and configuring your devices

    \b
    Use --skip-* options to skip specific steps.
    """

    config_path = ctx.obj["config_path"]

    # Welcome banner
    click.echo()
    click.echo("=" * 60)
    click.echo("Welcome to omramin initialization!")
    click.echo("=" * 60)
    click.echo()
    click.echo("This will guide you through:")
    click.echo("  1. Authenticating with Garmin Connect")
    click.echo("  2. Authenticating with OMRON Connect")
    click.echo("  3. Discovering and configuring your devices")
    click.echo()

    # Create temp config file for this session
    temp_fd, temp_config_path = tempfile.mkstemp(suffix=".json", prefix="omramin-init-")
    os.close(temp_fd)

    # Load existing config or start with defaults
    try:
        config = U.json_load(config_path)

    except FileNotFoundError:
        config = DEFAULT_CONFIG.copy()

    config = migrateconfig_path(config)

    # Write initial state to temp config
    U.json_save(temp_config_path, config)

    # Track what was accomplished for summary
    @dataclasses.dataclass
    class Summary:
        garmin_email: T.Optional[str] = None
        garmin_status: T.Optional[bool] = None
        omron_email: T.Optional[str] = None
        omron_country: T.Optional[str] = None
        omron_status: T.Optional[bool] = None
        devices_added: list[dict[str, T.Any]] = dataclasses.field(default_factory=list)

    summary = Summary()

    # Step 1: Garmin Authentication
    if not skip_garmin:
        click.echo("=" * 60)
        click.echo("Step 1: Garmin Connect Authentication")
        click.echo("=" * 60)
        click.echo()

        try:
            garmin_login(temp_config_path)
            config = U.json_load(temp_config_path)

            garminCfg = config.get("garmin", {})
            summary.garmin_email = garminCfg.get("email", "unknown")
            summary.garmin_status = True
            click.echo()

        except Exception as e:  # pylint: disable=broad-except
            L.error(f"Garmin authentication failed: {e}")
            click.echo()
            if not click.confirm("Continue without Garmin Connect?", default=True):
                L.info("Setup cancelled")
                return

            summary.garmin_status = False

    else:
        L.info("Skipping Garmin Connect authentication (--skip-garmin)")
        click.echo()

    # Step 2: OMRON Authentication
    oc: T.Optional[OC.OmronClient] = None
    if not skip_omron:
        click.echo("=" * 60)
        click.echo("Step 2: OMRON Connect Authentication")
        click.echo("=" * 60)
        click.echo()

        try:
            oc = omron_login(temp_config_path)
            config = U.json_load(temp_config_path)

            omronCfg = config.get("omron", {})
            summary.omron_email = omronCfg.get("email", "unknown")
            summary.omron_country = omronCfg.get("country", "unknown")
            summary.omron_status = True
            click.echo()

        except Exception as e:  # pylint: disable=broad-except
            L.error(f"OMRON authentication failed: {e}")
            click.echo()
            if not click.confirm("Continue without OMRON Connect?", default=True):
                L.info("Setup cancelled")
                return

            summary.omron_status = False

    else:
        L.info("Skipping OMRON Connect authentication (--skip-omron)")
        click.echo()

    # Step 3: Device Discovery
    if not skip_devices:
        if not summary.omron_status or oc is None:
            L.warning("Cannot discover devices without OMRON authentication")
            L.info("You can add devices later with 'omramin add'")

        else:
            click.echo("=" * 60)
            click.echo("Step 3: Device Discovery")
            click.echo("=" * 60)
            click.echo()

            # Determine discovery method
            method: str | None = discovery_method.lower()
            if method == "ask":
                questions = [
                    inquirer.List(
                        "method",
                        message="How would you like to discover devices?",
                        choices=[
                            ("Fetch from OMRON Connect (recommended)", "api"),
                            ("Scan via Bluetooth", "ble"),
                        ],
                        default="api",
                    )
                ]
                answers = inquirer.prompt(questions)
                if not answers:
                    L.info("Device discovery cancelled")
                    method = None

                else:
                    method = answers["method"]

            discoveredDevices: T.List[DeviceType] = []

            # API Discovery
            if method == "api":
                try:
                    click.echo()
                    L.info("Fetching devices from OMRON Connect...")
                    apiDevices = oc.get_registered_devices(days=OMRON_DEVICE_LIST_DAYS)

                    if not apiDevices:
                        L.info("No devices found in your OMRON account")
                        click.echo()
                        if click.confirm("Would you like to scan via Bluetooth instead?", default=True):
                            method = "ble"

                    else:
                        L.info(f"Found {len(apiDevices)} device(s) from API")

                        existingDevices = config.get("omron", {}).get("devices", [])
                        existingMacs = {d["macaddr"] for d in existingDevices}

                        apiDevicesDict = [d.to_dict() for d in apiDevices]
                        apiMacs = {d["macaddr"] for d in apiDevicesDict}

                        allDevices: T.List[DeviceType] = []
                        defaultSelected: T.List[int] = []

                        for device in apiDevicesDict:
                            allDevices.append(device)
                            # Check ALL API devices
                            defaultSelected.append(len(allDevices) - 1)

                        # Add devices from config that are NOT in API (unchecked by default)
                        # These are likely old/removed devices
                        for device in existingDevices:
                            if device["macaddr"] not in apiMacs:
                                allDevices.append(device)

                        if not allDevices:
                            L.info("No devices found")

                        else:
                            L.info(f"Found {len(existingDevices)} device(s) in current config")
                            click.echo()

                            choices = []
                            for i, d in enumerate(allDevices):
                                isExisting = d["macaddr"] in existingMacs
                                inApi = d["macaddr"] in apiMacs

                                prefix = "+ "
                                if not inApi:
                                    prefix = "- "

                                elif isExisting:
                                    prefix = "= "

                                label = f"{prefix}{d['name']} ({d['category']}) - {d['macaddr']}"
                                choices.append((label, i))

                                # API devices checked, old config-only devices unchecked
                                questions = [
                                    inquirer.Checkbox(
                                        "selected",
                                        message=(
                                            "Select devices to add/keep/remove in config "
                                            "(space to toggle, enter to confirm)"
                                        ),
                                        choices=choices,
                                        default=defaultSelected,
                                    )
                                ]

                            answers = inquirer.prompt(questions)
                            if not answers:
                                L.info("Device selection cancelled")

                            elif not answers["selected"]:
                                L.info("No devices selected")

                            else:
                                discoveredDevices = [allDevices[i] for i in answers["selected"]]

                except Exception as e:  # pylint: disable=broad-except
                    L.error(f"API device fetch failed: {e}")
                    click.echo()
                    if click.confirm("Would you like to try Bluetooth scanning instead?", default=True):
                        method = "ble"

            # BLE Discovery (fallback or explicit)
            if method == "ble":
                try:
                    click.echo()
                    opts = Options(ble_filter=ble_filter)
                    devices = config.get("omron", {}).get("devices", [])
                    macAddrs = [d["macaddr"] for d in devices]
                    bleDevices = omron_ble_scan(macAddrs, opts)

                    if not bleDevices:
                        L.info("No new devices found via Bluetooth")

                    else:
                        for mac in bleDevices:
                            newDevice = device_new(
                                macaddr=mac,
                                name=None,
                                category=None,
                                user=None,
                                enabled=None,
                            )
                            if newDevice:
                                discoveredDevices.append(newDevice)

                except ImportError as e:
                    L.error(str(e))
                    L.info("Alternative: Use API discovery method instead")
                    L.info("Run: omramin init --discovery-method api")

                except Exception as e:  # pylint: disable=broad-except
                    L.error(f"Bluetooth scanning failed: {e}")

            # Save devices to config
            if discoveredDevices:
                click.echo()

                if "omron" not in config:
                    config["omron"] = {"devices": []}

                if "devices" not in config["omron"]:
                    config["omron"]["devices"] = []

                # API gets complete list vs. BLE only currently available devices
                # API path: REPLACE entire device list
                # BLE path: MERGE with existing devices
                if method == "api":
                    L.info(f"Configuring {len(discoveredDevices)} device(s)...")
                    config["omron"]["devices"] = discoveredDevices

                else:
                    L.info(f"Adding {len(discoveredDevices)} device(s)...")
                    config["omron"]["devices"].extend(discoveredDevices)

                summary.devices_added = [{"name": d["name"], "category": d["category"]} for d in discoveredDevices]

                try:
                    U.json_save(temp_config_path, config)
                    L.info("Configuration saved to temp")

                except (OSError, IOError, PermissionError) as e:
                    L.error(f"Failed to save configuration: {e}")
                    L.info("You may need to edit the config file manually")

    else:
        L.info("Skipping device discovery (--skip-devices)")
        click.echo()

    # Determine if setup was completed or cancelled
    setup_complete = (
        (summary.garmin_status is not False)
        and (summary.omron_status is not False)
        and (skip_devices or summary.devices_added or not summary.omron_status)
    )

    # Summary
    click.echo()
    click.echo("=" * 60)
    click.echo("Setup Complete" if setup_complete else "Setup Cancelled")
    click.echo("=" * 60)
    click.echo()
    click.echo("Summary:")

    # Garmin status
    if summary.garmin_status is True:
        click.echo(f"  Garmin:  authenticated ({summary.garmin_email})")

    elif summary.garmin_status is False:
        click.echo("  Garmin:  authentication failed")

    else:
        click.echo("  Garmin:  skipped")

    # OMRON status
    if summary.omron_status is True:
        click.echo(f"  OMRON:   authenticated ({summary.omron_email}, {summary.omron_country})")

    elif summary.omron_status is False:
        click.echo("  OMRON:   authentication failed")

    else:
        click.echo("  OMRON:   skipped")

    # Devices status
    if summary.devices_added:
        click.echo(f"  Devices: {len(summary.devices_added)} device(s) configured")
        for device in summary.devices_added:
            click.echo(f"    - {device['name']} ({device['category']})")

    else:
        click.echo("  Devices: none configured")

    # Next steps
    if summary.devices_added:
        click.echo()
        click.echo("Next steps:")
        click.echo("  - Run 'omramin sync --days 7' to sync measurements")
        click.echo("  - Run 'omramin list' to see configured devices")

    elif summary.omron_status:
        click.echo()
        click.echo("Next steps:")
        click.echo("  - Run 'omramin add' to add devices")
        click.echo("  - Run 'omramin list' to see configured devices")

    # Merge temp config to final if setup completed
    if setup_complete:
        try:
            temp_config = U.json_load(temp_config_path)
            with config_write_handler(config_path, temp_config):
                pass

            L.info("Configuration saved successfully")

        except Exception as e:  # pylint: disable=broad-except
            L.error(f"Failed to save final configuration: {e}")

    try:
        os.unlink(temp_config_path)

    except Exception:  # pylint: disable=broad-except
        pass

    click.echo()


########################################################################################################################

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter  # Click handles argument parsing

########################################################################################################################

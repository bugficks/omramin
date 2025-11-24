#!/usr/bin/env python3
########################################################################################################################

import typing as T  # isort: split

import asyncio
import binascii
import csv
import dataclasses
import logging
import logging.config
import os
import pathlib
from datetime import datetime, timedelta

import bleak
import click
import garminconnect as GC
import garth
import inquirer

import omronconnect as OC
import utils as U
from regionserver import get_server_for_country_code

########################################################################################################################

__version__ = "0.1.1"

########################################################################################################################


class Options:
    def __init__(self):
        self.write_to_garmin = True
        self.overwrite = False
        self.ble_filter = "BLEsmart_"


########################################################################################################################

PATH_DEFAULT_CONFIG = "~/.omramin/config.json"

DEFAULT_CONFIG = {
    "garmin": {},
    "omron": {
        "server": "",
        "devices": [],
    },
}

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
        "omron": {
            "handlers": ["default"],
            "level": logging.INFO,
            "formatter": "root",
        },
        "httpx": {
            "handlers": ["http"],
            "level": logging.WARNING,
        },
        "httpcore": {
            "handlers": ["http"],
            "level": logging.WARNING,
        },
    },
}

########################################################################################################################

_E = os.environ.get

logging.config.dictConfig(LOGGING_CONFIG)
L = logging.getLogger("")
# L.setLevel(logging.DEBUG)


########################################################################################################################
class LoginError(Exception):
    pass


def garmin_login(_config: str) -> T.Optional[GC.Garmin]:
    """Login to Garmin Connect"""

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
        return None

    gcCfg = config["garmin"]

    def get_mfa():
        return inquirer.text(message="> Enter MFA/2FA code")

    logged_in = False
    try:
        tokendata = gcCfg.get("tokendata", "")
        if not tokendata:
            raise FileNotFoundError

        email = gcCfg["email"]
        is_cn = gcCfg["is_cn"]
        gc = GC.Garmin(email=email, is_cn=is_cn, prompt_mfa=get_mfa)

        logged_in = gc.login(tokendata)
        if not logged_in:
            raise FileNotFoundError

    except (FileNotFoundError, binascii.Error):
        L.info("Garmin login")
        questions = [
            inquirer.Text(
                name="email",
                message="> Enter email",
                validate=lambda _, x: x != "",
            ),
            inquirer.Password(
                name="password",
                message="> Enter password",
                validate=lambda _, x: x != "",
            ),
            inquirer.Confirm(
                "is_cn",
                message="> Is this a Chinese account?",
                default=False,
            ),
        ]
        answers = inquirer.prompt(questions)
        if not answers:
            # pylint: disable-next=raise-missing-from
            raise LoginError("Invalid input")

        email = answers["email"]
        password = answers["password"]
        is_cn = answers["is_cn"]

        gc = GC.Garmin(email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa)
        logged_in = gc.login()
        if logged_in:
            gcCfg["email"] = email
            gcCfg["is_cn"] = is_cn
            gcCfg["tokendata"] = gc.garth.dumps()

            try:
                U.json_save(_config, config)

            except (OSError, IOError, ValueError) as e:
                L.error(f"Failed to save configuration: {e}")

    except garth.exc.GarthHTTPError:
        L.error("Failed to login to Garmin Connect", exc_info=True)
        return None

    if not logged_in:
        L.error("Failed to login to Garmin Connect")
        return None

    L.info("Logged in to Garmin Connect")
    return gc


def omron_login(_config: str) -> T.Optional[OC.OmronConnect]:
    """Login to OMRON connect"""

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
        return None

    ocCfg = config["omron"]

    refreshToken = None
    try:
        server = ocCfg.get("server", "")
        tokendata = ocCfg.get("tokendata", "")
        if not tokendata:
            raise FileNotFoundError

        oc = OC.get_omron_connect(server)
        refreshToken = oc.refresh_oauth2(tokendata)
        if not refreshToken:
            raise FileNotFoundError

    except FileNotFoundError:
        L.info("Omron login")
        questions = [
            inquirer.Text(
                name="email",
                message="> Enter email",
                validate=lambda _, x: x != "",
            ),
            inquirer.Password(
                name="password",
                message="> Enter password",
                validate=lambda _, x: x != "",
            ),
            inquirer.Text(
                "country",
                message="> Enter country code (e.g. 'US')",
                validate=lambda _, x: get_server_for_country_code(x),
            ),
        ]
        answers = inquirer.prompt(questions)
        if not answers:
            # pylint: disable-next=raise-missing-from
            raise LoginError("Invalid input")

        email = answers["email"]
        password = answers["password"]
        country = answers["country"]
        server = get_server_for_country_code(country)

        oc = OC.get_omron_connect(server)
        refreshToken = oc.login(email=email, password=password, country=country)
        if refreshToken:
            tokendata = refreshToken
            ocCfg["email"] = email
            ocCfg["server"] = server
            ocCfg["tokendata"] = tokendata
            ocCfg["country"] = country
            ocCfg["server"] = server

            try:
                U.json_save(_config, config)

            except (OSError, IOError, ValueError) as e:
                L.error(f"Failed to save configuration: {e}")

    if refreshToken:
        L.info("Logged in to OMRON connect")
        return oc

    L.error("Failed to login to OMRON connect")
    return None


def omron_ble_scan(macAddrsExistig: T.List[str], opts: Options) -> T.List[str]:
    """Scan for Omron devices in pairing mode"""

    devsFound = {}

    async def scan():
        L.info("Scanning for Omron devices in pairing mode ...")
        L.info("Press Ctrl+C to stop scanning")
        while True:
            devices = await bleak.BleakScanner.discover(return_adv=True, timeout=1)
            devices = list(sorted(devices.items(), key=lambda x: x[1][1].rssi, reverse=True))
            for macAddr, (bleDev, advData) in devices:
                if macAddr in devsFound:
                    continue

                if macAddr in macAddrsExistig:
                    continue

                devName = bleDev.name or ""

                if opts.ble_filter and not devName.upper().startswith(opts.ble_filter.upper()):
                    continue

                serial = OC.ble_mac_to_serial(macAddr)
                devsFound[macAddr] = serial
                L.info(f"+ {macAddr} {bleDev.name} {serial} {advData.rssi}")

    try:
        asyncio.run(scan())
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
    oc: OC.OmronConnect, gc: GC.Garmin, ocDev: OC.OmronDevice, startLocal: int, endLocal: int, opts: Options
) -> None:
    if endLocal - startLocal <= 0:
        L.info("Invalid date range")
        return

    startdateStr = datetime.fromtimestamp(startLocal).date().isoformat()
    enddateStr = datetime.fromtimestamp(endLocal).date().isoformat()

    L.info(f"Start synchronizing device '{ocDev.name}' from {startdateStr} to {enddateStr}")

    measurements = oc.get_measurements(ocDev, searchDateFrom=int(startLocal * 1000), searchDateTo=int(endLocal * 1000))
    if not measurements:
        L.info("No new measurements")
        return

    L.info(f"Downloaded {len(measurements)} entries from 'OMRON connect' for '{ocDev.name}'")

    # get measurements from Garmin Connect for the same date range
    if ocDev.category == OC.DeviceCategory.SCALE:
        gcData = garmin_get_weighins(gc, startdateStr, enddateStr)
        sync_scale_measurements(gc, gcData, measurements, opts)
    elif ocDev.category == OC.DeviceCategory.BPM:
        gcData = garmin_get_bp_measurements(gc, startdateStr, enddateStr)
        sync_bp_measurements(gc, gcData, measurements, opts)


def sync_scale_measurements(
    gc: GC.Garmin, gcData: T.Dict[str, T.Any], measurements: T.List[OC.MeasurementTypes], opts: Options
):
    for measurement in measurements:
        tz = measurement.timeZone
        ts = measurement.measurementDate / 1000
        dtUTC = U.utcfromtimestamp(ts)
        dtLocal = datetime.fromtimestamp(ts, tz=tz)

        datetimeStr = dtLocal.isoformat(timespec="seconds")
        dateStr = dtLocal.date().isoformat()
        lookup = f"{dtUTC.date().isoformat()}:{dtUTC.timestamp()}"

        if lookup in gcData.values():
            if opts.overwrite:
                L.warning(f"  ! '{datetimeStr}': removing weigh-in")
                for samplePk, val in gcData.items():
                    if val == lookup and opts.write_to_garmin:
                        gc.delete_weigh_in(weight_pk=samplePk, cdate=dateStr)
            else:
                L.info(f"  - '{datetimeStr}' weigh-in already exists")
                continue

        wm = T.cast(OC.WeightMeasurement, measurement)

        L.info(f"  + '{datetimeStr}' adding weigh-in: {wm.weight} kg ")
        if opts.write_to_garmin:
            gc.add_body_composition(
                timestamp=datetimeStr,
                weight=wm.weight,
                percent_fat=wm.bodyFatPercentage if wm.bodyFatPercentage > 0 else None,
                percent_hydration=None,
                visceral_fat_mass=None,
                bone_mass=None,
                muscle_mass=(
                    (wm.skeletalMusclePercentage * wm.weight) / 100 if wm.skeletalMusclePercentage > 0 else None
                ),
                basal_met=wm.restingMetabolism if wm.restingMetabolism > 0 else None,
                active_met=None,
                physique_rating=None,
                metabolic_age=wm.metabolicAge if wm.metabolicAge > 0 else None,
                visceral_fat_rating=wm.visceralFatLevel if wm.visceralFatLevel > 0 else None,
                bmi=wm.bmiValue,
            )


def sync_bp_measurements(
    gc: GC.Garmin, gcData: T.Dict[str, T.Any], measurements: T.List[OC.MeasurementTypes], opts: Options
):
    for measurement in measurements:
        tz = measurement.timeZone
        ts = measurement.measurementDate / 1000
        dtUTC = U.utcfromtimestamp(ts)
        dtLocal = datetime.fromtimestamp(ts, tz=tz)

        datetimeStr = dtLocal.isoformat(timespec="seconds")
        dateStr = dtLocal.date().isoformat()
        lookup = f"{dtUTC.date().isoformat()}:{dtUTC.timestamp()}"

        if lookup in gcData.values():
            if opts.overwrite:
                L.warning(f"  ! '{datetimeStr}': removing blood pressure measurement")
                for version, val in gcData.items():
                    if val == lookup and opts.write_to_garmin:
                        gc.delete_blood_pressure(version=version, cdate=dateStr)
            else:
                L.info(f"  - '{datetimeStr}' blood pressure already exists")
                continue

        bpm = T.cast(OC.BPMeasurement, measurement)

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

        if opts.write_to_garmin:
            gc.set_blood_pressure(
                timestamp=datetimeStr, systolic=bpm.systolic, diastolic=bpm.diastolic, pulse=bpm.pulse, notes=notes
            )


def garmin_get_bp_measurements(gc: GC.Garmin, startdate: str, enddate: str):
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


def garmin_get_weighins(gc: GC.Garmin, startdate: str, enddate: str):
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


########################################################################################################################
class CommonCommand(click.Command):
    """Common options for all commands"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._defaultconfig = PATH_DEFAULT_CONFIG

        for p in [_E("OMRAMIN_CONFIG", PATH_DEFAULT_CONFIG), "./config.json"]:
            if pathlib.Path(p).expanduser().resolve().exists():
                self._defaultconfig = p

        self.params[0:0] = [
            click.Option(
                ("--config", "_config"),
                type=click.Path(writable=True, dir_okay=False),
                default=pathlib.Path(self._defaultconfig).expanduser().resolve(),
                show_default=True,
                help="Config file",
            ),
        ]


@click.group()
@click.version_option(__version__)
def cli():
    """Sync data from 'OMRON connect' to 'Garmin Connect'"""


########################################################################################################################


@cli.command(name="list", cls=CommonCommand)
def list_devices(_config: str):
    """List all configured devices."""

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
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


@cli.command(name="add", cls=CommonCommand)
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
def add_device(
    macaddr: T.Optional[str],
    name: T.Optional[str],
    category: T.Optional[OC.DeviceCategory],
    user: T.Optional[int],
    ble_filter: T.Optional[str],
    _config: str,
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
    opts = Options()
    opts.ble_filter = ble_filter

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        config = DEFAULT_CONFIG

    devices = config.get("omron", {}).get("devices", [])

    if not macaddr:
        macAddrs = [d["macaddr"] for d in devices]
        bleDevices = omron_ble_scan(macAddrs, opts)
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
                U.json_save(_config, config)
                L.info("Device(s) added successfully.")

            except (OSError, IOError, ValueError) as e:
                L.error(f"Failed to save configuration: {e}")


########################################################################################################################


@cli.command(name="config", cls=CommonCommand)
@click.argument("devname", required=True, type=str, nargs=1)
def edit_device(devname: str, _config: str):
    """Edit device configuration."""

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])
    if not devices:
        L.info("No devices configured.")
        return

    if not devname:
        macaddrs = [d["macaddr"] for d in devices]
        devname = inquirer.list_input("Select device to configure", choices=sorted(macaddrs))

    device = next((d for d in devices if d.get("name") == devname or d.get("macaddr") == devname), None)
    if not device:
        L.info(f"No device found with identifier: '{devname}'")
        return

    if device_edit(device):
        try:
            U.json_save(_config, config)
            L.info(f"Device '{devname}' configured successfully.")

        except (OSError, IOError, ValueError) as e:
            L.error(f"Failed to save configuration: {e}")


########################################################################################################################


@cli.command(name="remove", cls=CommonCommand)
@click.argument("devname", required=True, type=str, nargs=1)
def remove_device(devname: str, _config: str):
    """Remove a device by name or MAC address."""

    try:
        config = U.json_load(_config)
    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
        return

    devices = config.get("omron", {}).get("devices", [])

    if not devname:
        macaddrs = [d["macaddr"] for d in devices]
        devname = inquirer.list_input("Select device to remove", choices=sorted(macaddrs))

    device = next((d for d in devices if d.get("name") == devname or d.get("macaddr") == devname), None)

    if not device:
        L.info(f"No device found with identifier: {devname}")
        return

    devices.remove(device)
    try:
        U.json_save(_config, config)
        L.info(f"Device '{devname}' removed successfully.")

    except (OSError, IOError, ValueError) as e:
        L.error(f"Failed to save configuration: {e}")


########################################################################################################################


@cli.command(name="sync", cls=CommonCommand)
@click.argument("devnames", required=False, nargs=-1)
@click.option(
    "--category",
    "-c",
    "_category",
    required=False,
    type=click.Choice(list(OC.DeviceCategory.__members__.keys()), case_sensitive=False),
)
@click.option("--days", default=0, show_default=True, type=click.INT, help="Number of days to sync from today.")
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
def sync_device(
    devnames: T.List[str],
    _category: T.Optional[str],
    days: int,
    overwrite: bool,
    no_write: bool,
    _config: str,
):
    """Sync DEVNAMES... to Garmin Connect.

    \b
    DEVNAMES: List of Names or MAC addresses for the device to sync. [default: ALL]

    \b
    Examples:
        # Sync all devices for the last 7 days
        python omramin.py sync --days 7
    \b
        # Sync a specific device for the last 1 day
        python omramin.py sync "my scale" --days 1
    or
        python omramin.py sync 00:11:22:33:44:55 "my scale" --days 1
    """

    opts = Options()
    opts.overwrite = overwrite
    opts.write_to_garmin = not no_write

    try:
        config = U.json_load(_config)

    except FileNotFoundError:
        L.error(f"Config file '{_config}' not found.")
        return

    category = OC.DeviceCategory[_category] if _category else None

    devices = config.get("omron", {}).get("devices", [])
    if not devices:
        L.info("No devices configured.")
        return

    try:
        startLocal, endLocal = calculate_date_range(days)
    except DateRangeException:
        L.info("Invalid date range")
        return

    # filter devices by enabled, category and name/mac address
    devices = filter_devices(devices, devnames=devnames, category=category)
    if not devices:
        L.info("No matching devices found")
        return

    try:
        gc = garmin_login(_config)
    except LoginError:
        L.info("Failed to login to Garmin Connect.")
        return

    try:
        oc = omron_login(_config)
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


@cli.command(name="export", cls=CommonCommand)
@click.argument("devnames", required=False, nargs=-1)
@click.option(
    "--category",
    "-c",
    "_category",
    required=True,
    type=click.Choice(list(OC.DeviceCategory.__members__.keys()), case_sensitive=False),
)
@click.option("--days", default=0, show_default=True, type=click.INT, help="Number of days to sync from today.")
@click.option(
    "--format",
    "_format",
    type=click.Choice(["csv", "json"], case_sensitive=False),
    default="csv",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def export_measurements(
    devnames: T.Optional[T.List[str]],
    _category: str,
    days: int,
    _format: T.Optional[str],
    output: T.Optional[str],
    _config: str,
):
    """Export device measurements to CSV or JSON format."""

    config = U.json_load("config.json")
    devices = config.get("omron", {}).get("devices", [])
    category = OC.DeviceCategory[_category]

    try:
        startLocal, endLocal = calculate_date_range(days)
    except DateRangeException:
        L.info("Invalid date range")
        return

    # filter devices by enabled, category and name/mac address
    devices = filter_devices(devices, devnames=devnames)
    if not devices:
        L.info("No matching devices found")
        return

    startdateStr = datetime.fromtimestamp(startLocal).date().isoformat()
    enddateStr = datetime.fromtimestamp(endLocal).date().isoformat()

    try:
        oc = omron_login(_config)
    except LoginError:
        L.info("Failed to login to OMRON connect.")
        return

    if not oc:
        return

    exportdata = {}
    for device in devices:
        ocDev = OC.OmronDevice(**device)
        L.info(f"Exporting device '{ocDev.name}' from {startdateStr} to {enddateStr}")

        measurements = oc.get_measurements(
            ocDev, searchDateFrom=int(startLocal * 1000), searchDateTo=int(endLocal * 1000)
        )
        if measurements:
            exportdata[ocDev] = measurements

    if not exportdata:
        L.info("No measurements found")
        return

    if not output:
        output = (
            f"omron_{category.name}_{datetime.fromtimestamp(startLocal).date()}_"
            f"{datetime.fromtimestamp(endLocal).date()}.{_format}"
        )

    if _format == "json":
        export_json(output, exportdata)

    else:
        export_csv(output, exportdata)

    L.info(f"Exported {len(exportdata)} measurements to {output}")


def export_csv(output: str, exportdata: T.Dict[OC.OmronDevice, T.List[OC.MeasurementTypes]]) -> None:
    with open(output, "w", newline="\n", encoding="utf-8") as f:
        writer = None
        for ocDev, measurements in exportdata.items():
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


def export_json(output: str, exportdata: T.Dict[OC.OmronDevice, T.List[OC.MeasurementTypes]]) -> None:
    data = []
    for ocDev, measurements in exportdata.items():
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

if __name__ == "__main__":
    pathlib.Path("~/.omramin").expanduser().resolve().mkdir(parents=True, exist_ok=True)

    cli()

########################################################################################################################

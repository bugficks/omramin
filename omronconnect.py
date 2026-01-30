########################################################################################################################

import typing as T  # isort: split

import datetime
import enum
import hashlib
import json
import logging
import re
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from typing import get_type_hints

import httpx
import pytz
from httpx import HTTPStatusError

import utils as U
from httpxlogtransport import HttpxLogTransport, transport_set_logger
from regionserver import get_credentials_for_server, get_servers_for_country_code

########################################################################################################################

L = logging.getLogger("omronconnect")
transport_set_logger(L)

_debugSaveResponse = False

########################################################################################################################
# Monkey-patch httpx GZipDecoder to handle OMRON servers that claim gzip but send uncompressed data


def _patched_gzip_decode(self, data: bytes) -> bytes:
    """Patched decode that handles servers lying about Content-Encoding: gzip"""

    try:
        return self.decompressor.decompress(data)

    except zlib.error as exc:
        try:
            data.decode("utf-8")
            L.debug(f"GZip decompression failed: {exc}")
            return data

        except UnicodeDecodeError:
            raise httpx.DecodingError(str(exc)) from exc


########################################################################################################################


class Gender(enum.IntEnum):
    MALE = 1
    FEMALE = 2


class VolumeUnit(enum.IntEnum):
    MGDL = 24577
    MMOLL = 24593


class LengthUnit(enum.IntEnum):
    CM = 4098
    INCH = 4113
    KM = 4099
    MILE = 4112


class WeightUnit(enum.IntEnum):
    G = 8192
    KG = 8195
    LB = 8208
    ST = 8224


class ValueUnit(enum.IntEnum):
    STEPS = 61536
    BPM = 61600
    PERCENTAGE = 61584
    KCAL = 16387


class BPUnit(enum.IntEnum):
    MMHG = 20496
    KPA = 20483


class TemperatureUnit(enum.IntEnum):
    CELSIUS = 12288
    FAHRENHEIT = 12304


########################################################################################################################


class ValueType(enum.StrEnum):
    EVENT_RECORD = "0"  # ("%d", 0, 0, -1, -1),
    MMHG_MAX_FIGURE = "1"  # ("%1$,3.0f", 1, 20496, R.string.msg0000808, R.string.msg0020959),
    KPA_MAX_FIGURE = "1"  # ("%.1f", 1, 20483, R.string.msg0000809, R.string.msg0020993),
    MMHG_MIN_FIGURE = "2"  # ("%1$,3.0f", 2, 20496, R.string.msg0000808, R.string.msg0020959),
    KPA_MIN_FIGURE = "2"  # ("%.1f", 2, 20483, R.string.msg0000809, R.string.msg0020993),
    BPM_FIGURE = "3"  # ("%1$,3.0f", 3, 61600, R.string.msg0000815, R.string.msg0020960),
    ATTITUDE_FIGURE = "4"  # ("%.0f", 4, -1, -1, -1),
    ROOM_TEMPERATURE_FIGURE = "5"  # ("%.0f", 5, -1, R.string.msg0000823, R.string.msg0020972),
    ARRHYTHMIA_FLAG_FIGURE = "6"  # ("%.0f", 6, -1, -1, -1),
    BODY_MOTION_FLAG_FIGURE = "7"  # ("%.0f", 7, -1, -1, -1),
    POSTURE_GUIDE = "20"  # ("%.0f", 20, -1, -1, -1),
    KEEP_UP_CHECK_FIGURE = "8"  # ("%.0f", 8, -1, -1, -1),
    PULSE_QUIET_CHECK_FIGURE = "9"  # ("%.0f", 9, -1, -1, -1),
    CONTINUOUS_MEASUREMENT_COUNT_FIGURE = "10"  # ("%1$,3.0f", 10, -1, R.string.msg0000821, R.string.msg0000821),
    ARTIFACT_COUNT_FIGURE = "11"  # ("%1$,3.0f", 11, -1, R.string.msg0000821, R.string.msg0000821),
    IRREGULAR_PULSE_COUNT_FIGURE = "37"  # ("%1$,3.0f", 37, -1, R.string.msg0020505, R.string.msg0020505),
    MEASUREMENT_MODE_FIGURE = "38"  # ("%.0f", 38, -1, -1, -1),
    NOCTURNAL_ERROR_CODE_FIGURE = "41"  # ("%.0f", 41, -1, -1, -1),
    NNOCTURNAL_ERROR_CODE_DISPLAY_FIGURE = "45"  # ("%.0f", 45, -1, -1, -1),
    DISORDERED_PULSE_RATE = "51"  # ("%.1f", 51, -1, -1, -1),
    KG_FIGURE = "257"  # ("%.2f", 257, 8195, R.string.msg0000803, R.string.msg0020986),
    KG_SKELETAL_MUSCLE_MASS_FIGURE = "294"  # ("%.1f", 294, 8195, R.string.msg0000803, R.string.msg0020986),
    KG_BODY_FAT_MASS_FIGURE = "295"  # ("%.1f", 295, -1, R.string.msg0000803, R.string.msg0020986),
    LB_FIGURE = "257"  # ("%.1f", 257, 8208, R.string.msg0000804, R.string.msg0020994),
    ST_FIGURE = "257"  # ("%.0f", 257, 8224, R.string.msg0000805, R.string.msg0020995),
    BODY_FAT_PER_FIGURE = "259"  # ("%.1f", 259, 61584, R.string.msg0000817, R.string.msg0000817),
    VISCERAL_FAT_FIGURE = "264"  # ("%1$,3.0f", 264, -1, R.string.msg0000816, R.string.msg0020987),
    VISCERAL_FAT_FIGURE_702T = "264"  # ("%1$,3.1f", 264, -1, R.string.msg0000816, R.string.msg0020987),
    RATE_SKELETAL_MUSCLE_FIGURE = "261"  # ("%.1f", 261, 61584, R.string.msg0000817, R.string.msg0000817),
    RATE_SKELETAL_MUSCLE_BOTH_ARMS_FIGURE = "275"  # ("%.1f", 275, 61584, R.string.msg0000817, R.string.msg0000817),
    RATE_SKELETAL_MUSCLE_BODY_TRUNK_FIGURE = "277"  # ("%.1f", 277, -1, R.string.msg0000817, R.string.msg0000817),
    RATE_SKELETAL_MUSCLE_BOTH_LEGS_FIGURE = "279"  # ("%.1f", 279, -1, R.string.msg0000817, R.string.msg0000817),
    RATE_SUBCUTANEOUS_FAT_FIGURE = "281"  # ("%.1f", 281, -1, R.string.msg0000817, R.string.msg0000817),
    RATE_SUBCUTANEOUS_FAT_BOTH_ARMS_FIGURE = "283"  # ("%.1f", 283, -1, R.string.msg0000817, R.string.msg0000817),
    RATE_SUBCUTANEOUS_FAT_BODY_TRUNK_FIGURE = "285"  # ("%.1f", 285, -1, R.string.msg0000817, R.string.msg0000817),
    RATE_SUBCUTANEOUS_FAT_BOTH_LEGS_FIGURE = "287"  # ("%.1f", 287, -1, R.string.msg0000817, R.string.msg0000817),
    BIOLOGICAL_AGE_FIGURE = "263"  # ("%1$,3.0f", 263, 61568, R.string.msg0000822, R.string.msg0020989),
    BASAL_METABOLISM_FIGURE = "260"  # ("%1$,3.0f", 260, 16387, R.string.msg0000824, R.string.msg0020988),
    BMI_FIGURE = "262"  # ("%.1f", 262, -1, -1, -1),
    BLE_BMI_FIGURE = "292"  # ("%.1f", 292, -1, -1, -1),
    VISCERAL_FAT_CHECK_FIGURE = "265"  # ("%.0f", 265, -1, -1, -1),
    RATE_SKELETAL_MUSCLE_CHECK_FIGURE = "266"  # ("%.0f", 266, -1, -1, -1),
    RATE_SKELETAL_MUSCLE_BOTH_ARMS_CHECK_FIGURE = "276"  # ("%.0f", 276, -1, -1, -1),
    RATE_SKELETAL_MUSCLE_BODY_TRUNK_CHECK_FIGURE = "278"  # ("%.0f", 278, -1, -1, -1),
    RATE_SKELETAL_MUSCLE_BOTH_LEGS_CHECK_FIGURE = "280"  # ("%.0f", 280, -1, -1, -1),
    RATE_SUBCUTANEOUS_FAT_CHECK_FIGURE = "282"  # ("%.0f", 282, -1, -1, -1),
    RATE_SUBCUTANEOUS_FAT_BOTH_ARMS_CHECK_FIGURE = "284"  # ("%.0f", 284, -1, -1, -1),
    RATE_SUBCUTANEOUS_FAT_BODY_TRUNK_CHECK_FIGURE = "286"  # ("%.0f", 286, -1, -1, -1),
    RATE_SUBCUTANEOUS_FAT_BOTH_LEGS_CHECK_FIGURE = "288"  # ("%.0f", 288, -1, -1, -1),
    IMPEDANCE_FIGURE = "267"  # ("%.0f", 267, -1, -1, -1),
    WEIGHT_FFM_FIGURE = "268"  # ("%.0f", 268, -1, -1, -1),
    AVERAGE_WEIGHT_FIGURE = "269"  # ("%.0f", 269, -1, -1, -1),
    AVERAGE_WEIGHT_FFM_FIGURE = "270"  # ("%.0f", 270, -1, -1, -1),
    MMOLL_FIGURE = "2305"  # ("%.1f", 2305, 24593, R.string.msg0000811, R.string.msg0020975),
    MGDL_FIGURE = "2305"  # ("%.0f", 2305, 24577, R.string.msg0000810, R.string.msg0020976),
    MEAL_FIGURE = "2306"  # ("%.0f", 2306, -1, -1, -1),
    TYPE_FIGURE = "2307"  # ("%.0f", 2307, -1, -1, -1),
    SAMPLE_LOCATION_FIGURE = "2308"  # ("%.0f", 2308, -1, -1, -1),
    HIGH_LOW_DETECTION_FIGURE = "2309"  # ("%.0f", 2309, -1, -1, -1),
    STEPS_FIGURE = "513"  # ("%1$,3.0f", 513, 61536, R.string.msg0000833, R.string.msg0020991),
    TIGHTLY_STEPS = "514"  # ("%1$,3.0f", 514, 61536, R.string.msg0000833, R.string.msg0020991),
    STAIR_UP_STEPS = "518"  # ("%1$,3.0f", 518, -1, R.string.msg0000833, R.string.msg0020991),
    BRISK_STEPS = "516"  # ("%1$,3.0f", 516, -1, R.string.msg0000833, R.string.msg0020991),
    KCAL_WALKING = "545"  # ("%1$,3.0f", 545, 16387, R.string.msg0000824, R.string.msg0020988),
    KCAL_ACTIVITY = "546"  # ("%1$,3.0f", 546, 16387, R.string.msg0000824, R.string.msg0020988),
    KCAL_FAT_BURNED = "579"  # ("%.1f", 579, 8192, R.string.msg0000852, R.string.msg0020990),
    KCAL_ALLDAY = "548"  # ("%1$,3.0f", 548, -1, R.string.msg0000824, R.string.msg0020988),
    KM_FIGURE = "3"  # ("%.1f", 3, 4099, R.string.msg0000801, R.string.msg0020992),
    KM_DISTANCE = "576"  # ("%.1f", 576, 4099, R.string.msg0000801, R.string.msg0020992),
    TIME_SLEEP_START = "1025"  # ("%d", 1025, 0, -1, -1),
    TIME_SLEEP_ONSET = "1026"  # ("%d", 1026, 0, -1, -1),
    TIME_SLEEP_WAKEUP = "1027"  # ("%d", 1027, -1, -1, -1),
    TIME_SLEEPING = "1028"  # ("%d", 1028, 61488, R.string.msg0000866, R.string.msg0000866),
    TIME_SLEEPING_EFFICIENCY = "1029"  # ("%.1f", 1029, 61584, R.string.msg0000817, R.string.msg0000817),
    TIME_SLEEP_AROUSAL = "1030"  # ("%d", 1030, 61504, R.string.msg0000867, R.string.msg0000867),
    TEMPERATURE_BASAL = "1281"  # ("%.2f", 1281, 12288, R.string.msg0000823, R.string.msg0020972),
    FAHRENHEIT_TEMPERATURE_BASAL = "1281"  # ("%.2f", 1281, 12304, R.string.msg0000829, R.string.msg0020996),
    THERMOMETER_TEMPERATURE = "4866"  # ("%.1f", 4866, -1, R.string.msg0000823, R.string.msg0020972),
    FAHRENHEIT_THERMOMETER_TEMPERATURE = "4866"  # ("%.1f", 4866, -1, R.string.msg0000829, R.string.msg0020996),
    THERMOMETER_MEASUREMENT_MODE_PREDICTED = "4869"  # ("%.0f", 4869, -1, -1, -1),
    THERMOMETER_MEASUREMENT_MODE_MEASURED = "4870"  # ("%.0f", 4870, -1, -1, -1),
    MENSTRUATION_RECORD = "61442"  # ("%.0f", 61442, -1, -1, -1),
    MILE_FIGURE = "576"  # ("%.1f", 576, 4112, R.string.msg0000802, R.string.msg0020997),
    KCAL_DAY = "544"  # ("%1$,3.0f", 544, 16387, R.string.msg0000824, R.string.msg0020988),
    KCAL_FIGURE = "3"  # ("%1$,3.0f", 3, 16387, R.string.msg0000824, R.string.msg0020988),
    MMHG_MEAN_ARTERIAL_PRESSURE_FIGURE = "16"  # ("%1$,3.0f", 16, 20496, R.string.msg0000808, R.string.msg0020959),
    KPA_MEAN_ARTERIAL_PRESSURE_FIGURE = "16"  # ("%.1f", 16, 20483, R.string.msg0000809, R.string.msg0020993),
    AFIB_DETECT_FIGURE = "35"  # ("%.1f", 35, -1, -1, -1),
    AFIB_MODE_FIGURE = "39"  # ("%.1f", 39, -1, -1, -1),
    ECG_BPM_FIGURE = "4143"  # ("%1$,3.0f", 4143, 61600, R.string.msg0000815, R.string.msg0020960),
    SPO2_OXYGEN_SATURATION = "1537"  # ("%.0f", 1537, 61584, R.string.msg0000817, R.string.msg0000817),
    SPO2_PULSE_RATE = "1538"  # ("%.0f", 1538, -1, R.string.msg0000815, R.string.msg0020960),
    THERMOMETER_TEMPERATURE_TYPE = "4871"  # ("%.0f", 4871, -1, -1, -1)


class DeviceCategory(enum.StrEnum):
    BPM = "0"
    SCALE = "1"
    # ACTIVITY = "2"
    # THERMOMETER = "3"
    # PULSE_OXIMETER = "4"


def _coerce_dataclass_fields(self) -> None:
    type_hints = get_type_hints(type(self))
    for field in fields(self):
        attr = getattr(self, field.name)
        field_type = type_hints[field.name]
        if field_type == datetime.tzinfo:
            if not isinstance(attr, datetime.tzinfo):
                object.__setattr__(self, field.name, pytz.timezone(attr))

        else:
            object.__setattr__(self, field.name, field_type(attr))


@dataclass(frozen=True, kw_only=False)
class BodyIndexListItem:
    value: int
    subtype: int
    scale: int
    measurementId: int

    def __post_init__(self) -> None:
        _coerce_dataclass_fields(self)


########################################################################################################################


@dataclass(frozen=True, kw_only=True)
class BPMeasurement:
    systolic: int
    diastolic: int
    pulse: int
    measurementDate: int
    timeZone: datetime.tzinfo
    irregularHB: bool = False
    movementDetect: bool = False
    cuffWrapDetect: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        _coerce_dataclass_fields(self)


@dataclass(frozen=True, kw_only=True)
class WeightMeasurement:
    weight: float
    measurementDate: int
    timeZone: datetime.tzinfo
    bmiValue: float = -1.0
    bodyFatPercentage: float = -1.0
    restingMetabolism: float = -1.0
    skeletalMusclePercentage: float = -1.0
    visceralFatLevel: float = -1.0
    metabolicAge: int = -1
    notes: str = ""

    def __post_init__(self) -> None:
        _coerce_dataclass_fields(self)


MeasurementTypes = T.Union[BPMeasurement, WeightMeasurement]

########################################################################################################################


def ble_mac_to_serial(mac: str) -> str:
    # e.g. 11:22:33:44:55:66 to 665544feff332211
    values = mac.split(":")
    serial = "".join(values[5:2:-1] + ["fe", "ff"] + values[2::-1])
    return serial.lower()


def serial_to_mac(serial: str) -> str:
    # e.g. 665544feff332211 to 11:22:33:44:55:66
    values = [serial[i : i + 2] for i in range(0, len(serial), 2)]  # noqa: F203
    return ":".join(values[5:2:-1] + values[2::-1])


def convert_weight_to_kg(weight: T.Union[int, float], unit: int) -> float:
    if unit == WeightUnit.G:
        return weight / 1000

    if unit == WeightUnit.LB:
        return weight * 0.45359237

    if unit == WeightUnit.ST:
        return weight * 6.35029318

    return weight


@T.overload
def convert_data_util(value: int, scale: int, _type: type[int]) -> int: ...  # noqa: F704


@T.overload
def convert_data_util(value: int, scale: int, _type: type[float] = float) -> float: ...  # noqa: F704


def convert_data_util(value: int, scale: int, _type: T.Callable[[Decimal], T.Any] = float) -> T.Any:
    if scale < 0:
        factor = Decimal("0.1") ** (-scale)

    else:
        factor = Decimal(10) ** scale

    return _type(Decimal(value) * factor)


########################################################################################################################


@dataclass(frozen=True, kw_only=True)
class OmronDevice:
    name: str
    macaddr: str
    category: DeviceCategory
    user: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.category, DeviceCategory):
            try:
                object.__setattr__(self, "category", DeviceCategory.__members__[self.category.upper()])

            except KeyError as exc:
                object.__setattr__(self, "enabled", False)
                raise ValueError(f"Device '{self.name}' has invalid category: '{self.category}'") from exc

    @property
    def serial(self) -> str:
        return ble_mac_to_serial(self.macaddr)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["category"] = self.category.name
        return result


########################################################################################################################


def _http_add_checksum(request: httpx.Request) -> None:
    if request.method in ["POST", "DELETE"] and request.content:
        request.headers["Checksum"] = hashlib.sha256(request.content).hexdigest()


########################################################################################################################


class OmronConnect(ABC):
    @abstractmethod
    def login(self, email: str, password: str, country: str) -> T.Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def refresh_oauth2(self, refresh_token: str, **kwargs: T.Any) -> T.Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_user(self) -> T.Dict[str, T.Any]:
        raise NotImplementedError

    @abstractmethod
    def get_registered_devices(self, days: T.Optional[int] = 30) -> T.Optional[list[OmronDevice]]:
        """Get registered devices.

        Args:
            days: For API v1, limits results to devices active in last N days (supports pagination).
                  For API v2, this parameter is ignored (returns all active devices).
                  Use None to fetch all historical devices (may be slow for API v1).
        """

        raise NotImplementedError

    @abstractmethod
    def get_measurements(
        self, device: OmronDevice, searchDateFrom: int = 0, searchDateTo: int = 0
    ) -> T.List[MeasurementTypes]:
        raise NotImplementedError


########################################################################################################################


class OmronConnect1(OmronConnect):
    _OGSC_APP_VERSION = "010.003.00001"
    _OGSC_SDK_VERSION = "000.101"

    _USER_AGENT = f"OmronConnect/{_OGSC_APP_VERSION}.001 CFNetwork/1335.0.3.4 Darwin/21.6.0)"

    def __init__(self, server: str, country: str):
        self._server = server
        self._country = country
        self._headers: T.Dict[str, str] = {}

        # Get AppID/AppKey from region_grouped_by_app.json based on server URL
        credentials = get_credentials_for_server(server)
        if not credentials:
            raise ValueError(f"No API v1 credentials found for server: {server}")

        app_id, app_key = credentials

        # Set _APP_URL using the selected app_id
        self._APP_URL = f"/apps/{app_id}/server-code"

        # Wrap transport with LogTransport for debug logging with credential redaction
        self._client = httpx.Client(
            transport=HttpxLogTransport(httpx.HTTPTransport()),
            headers={
                "user-agent": OmronConnect1._USER_AGENT,
                "X-OGSC-SDK-Version": OmronConnect1._OGSC_SDK_VERSION,
                "X-OGSC-App-Version": OmronConnect1._OGSC_APP_VERSION,
                "X-Kii-AppID": app_id,
                "X-Kii-AppKey": app_key,
            },
        )

        # pylint: disable=protected-access
        setattr(httpx._decoders.GZipDecoder, "decode", _patched_gzip_decode)

    def login(self, email: str, password: str, country: str = "") -> T.Optional[str]:
        authData = {
            "username": email,
            "password": password,
        }
        r = self._client.post(f"{self._server}/oauth2/token", json=authData, headers=self._headers)
        r.raise_for_status()

        resp = r.json()
        try:
            access_token = resp["access_token"]
            refresh_token = resp["refresh_token"]
            self._headers["authorization"] = f"Bearer {access_token}"
            return refresh_token

        except KeyError:
            L.error(f"login() -> {r}: '{r.text}'")

        return None

    def refresh_oauth2(self, refresh_token: str, **kwargs: T.Any) -> T.Optional[str]:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        r = self._client.post(f"{self._server}/oauth2/token", json=data, headers=self._headers)
        r.raise_for_status()

        resp = r.json()
        try:
            access_token = resp["access_token"]
            refresh_token = resp["refresh_token"]
            self._headers["authorization"] = f"Bearer {access_token}"
            return refresh_token

        except KeyError:
            L.error(f"refresh_oauth2() -> {r}: '{r.text}'")

        return None

    def get_user(self) -> T.Dict[str, T.Any]:
        r = self._client.get(f"{self._server}{self._APP_URL}/users/me", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def get_registered_devices(self, days: T.Optional[int] = 30) -> T.Optional[list[OmronDevice]]:
        """Fetch registered devices with pagination support.

        Args:
            days: Limit to devices active in last N days. None = all devices (may be slow).
        """

        syncList = []

        lastSyncDate = 0 if days is None else int((U.utcnow() - datetime.timedelta(days=days)).timestamp() * 1000)
        payload = {
            "countOnlyFlag": 0,
            "lastSyncDate": lastSyncDate,
        }

        while True:
            r = self._client.post(
                f"{self._server}{self._APP_URL}/versions/current/synchronizeDeviceConfData",
                headers=self._headers,
                json=payload,
            )
            r.raise_for_status()
            resp = r.json()

            returnedValue = resp.get("returnedValue", {})
            syncList.extend(returnedValue.get("syncList", []))

            nextPaginationKey = int(returnedValue.get("nextPaginationKey", 0))
            if not nextPaginationKey or lastSyncDate == nextPaginationKey:
                break

            payload["lastSyncDate"] = nextPaginationKey

            L.info(f"Fetching next page (pagination key: {nextPaginationKey})")

        def unique_devices(sync_list):
            devices = {}

            for sync in sync_list:
                for cat in sync.get("deviceCategoryList", []):
                    for model in cat.get("deviceModelList", []):
                        for dev in model.get("deviceSerialIDList", []):
                            try:
                                key = f"{dev['deviceSerialID']}:{dev['userNumberInDevice']}"
                                devices.setdefault(
                                    key,
                                    {
                                        "deviceCategory": cat["deviceCategory"],
                                        "deviceModel": model["deviceModel"],
                                        "deviceSerialID": dev["deviceSerialID"],
                                        "userNumberInDevice": dev["userNumberInDevice"],
                                    },
                                )

                            except KeyError:
                                L.debug("Skipping device with missing required fields")
                                continue

            return devices

        devices = unique_devices(syncList)
        result: list[OmronDevice] = []
        for device in devices.values():
            try:
                category = DeviceCategory(device["deviceCategory"])

            except (ValueError, KeyError):
                L.debug(f"Skipping device with unsupported category: {device.get('deviceModel', 'unknown')}")
                continue

            ocDev = OmronDevice(
                category=category,
                name=f"{device['deviceModel']}:{device['userNumberInDevice']}",
                macaddr=serial_to_mac(device["deviceSerialID"]),
                user=int(device["userNumberInDevice"]),
            )
            result.append(ocDev)

        return result

    # utc timestamps
    def get_measurements(
        self, device: OmronDevice, searchDateFrom: int = 0, searchDateTo: int = 0
    ) -> T.List[MeasurementTypes]:
        data = {
            "containCorrectedDataFlag": 1,
            "containAllDataTypeFlag": 1,
            "deviceCategory": device.category,
            "deviceSerialID": device.serial,
            "userNumberInDevice": int(device.user),
            "searchDateFrom": searchDateFrom if searchDateFrom >= 0 else 0,
            "searchDateTo": int(U.utcnow().timestamp() * 1000) if searchDateTo <= 0 else searchDateTo,
            # "deviceModel": "OSG",
        }

        r = self._client.post(
            f"{self._server}{self._APP_URL}/versions/current/measureData", json=data, headers=self._headers
        )
        r.raise_for_status()

        resp = r.json()
        L.debug(resp)

        returnedValue = resp.get("returnedValue")
        if not returnedValue:
            L.error(f"get_measurements() -> {r}: '{r.text}'")
            return []

        if isinstance(returnedValue, list):
            returnedValue = returnedValue[0]

        if isinstance(returnedValue, dict) and "errorCode" in returnedValue:
            L.error(f"get_measurements() -> {returnedValue}")
            return []

        if _debugSaveResponse:
            fname = f".debug/{data['searchDateTo']}_{device.category.name}_{device.serial}_{device.user}.json"
            U.json_save(fname, returnedValue)

        measurements: T.List[MeasurementTypes] = []
        devCat = DeviceCategory(returnedValue["deviceCategory"])
        deviceModelList = returnedValue["deviceModelList"]
        if deviceModelList is None:
            return measurements

        for devModel in deviceModelList:
            measurements.extend(self._process_device_model(devModel, device, devCat))

        return measurements

    def _process_device_model(
        self, devModel: T.Dict[str, T.Any], device: OmronDevice, devCat: DeviceCategory
    ) -> T.List[MeasurementTypes]:
        measurements: T.List[MeasurementTypes] = []
        deviceModel = devModel["deviceModel"]
        deviceSerialIDList = devModel["deviceSerialIDList"]
        for dev in deviceSerialIDList:
            deviceSerialID = dev["deviceSerialID"]
            user = dev["userNumberInDevice"]
            L.debug(f" - deviceModel: {deviceModel} category: {devCat.name} serial: {deviceSerialID} user: {user}")

            if deviceSerialID != device.serial:
                continue

            if device.category == DeviceCategory.BPM:
                measurements.extend(self._process_bpm_measurements(dev))

            elif device.category == DeviceCategory.SCALE:
                measurements.extend(self._process_scale_measurements(dev))

            break

        return measurements

    def _process_bpm_measurements(self, dev: T.Dict[str, T.Any]) -> T.List[BPMeasurement]:
        measurements: T.List[BPMeasurement] = []
        for m in dev["measureList"]:
            bodyIndexList = {k: BodyIndexListItem(*v) for k, v in m["bodyIndexList"].items()}
            systolic = convert_data_util(
                bodyIndexList[ValueType.MMHG_MAX_FIGURE].value,
                bodyIndexList[ValueType.MMHG_MAX_FIGURE].scale,
                int,
            )
            diastolic = convert_data_util(
                bodyIndexList[ValueType.MMHG_MIN_FIGURE].value,
                bodyIndexList[ValueType.MMHG_MIN_FIGURE].scale,
                int,
            )
            pulse = convert_data_util(
                bodyIndexList[ValueType.BPM_FIGURE].value,
                bodyIndexList[ValueType.BPM_FIGURE].scale,
                int,
            )
            bodymotion = bodyIndexList[ValueType.BODY_MOTION_FLAG_FIGURE].value
            irregHB = bodyIndexList[ValueType.ARRHYTHMIA_FLAG_FIGURE].value
            cuffWrapGuid = bodyIndexList[ValueType.KEEP_UP_CHECK_FIGURE].value
            timeZone = pytz.timezone(m["timeZone"])

            bp = BPMeasurement(
                systolic=systolic,
                diastolic=diastolic,
                pulse=pulse,
                measurementDate=m["measureDateTo"],
                timeZone=timeZone,
                irregularHB=irregHB != 0,
                movementDetect=bodymotion != 0,
                cuffWrapDetect=cuffWrapGuid != 0,
            )
            measurements.append(bp)

        return measurements

    def _process_scale_measurements(self, dev: T.Dict[str, T.Any]) -> T.List[WeightMeasurement]:
        measurements: T.List[WeightMeasurement] = []
        for m in dev["measureList"]:
            bodyIndexList = {k: BodyIndexListItem(*v) for k, v in m["bodyIndexList"].items()}
            weight_entry = bodyIndexList[ValueType.KG_FIGURE]
            weight = convert_data_util(weight_entry.value, weight_entry.scale)
            weightUnit = weight_entry.subtype
            weight = convert_weight_to_kg(weight, weightUnit)
            bodyFatPercentage = convert_data_util(
                bodyIndexList[ValueType.BODY_FAT_PER_FIGURE].value,
                bodyIndexList[ValueType.BODY_FAT_PER_FIGURE].scale,
            )
            skeletalMusclePercentage = convert_data_util(
                bodyIndexList[ValueType.RATE_SKELETAL_MUSCLE_FIGURE].value,
                bodyIndexList[ValueType.RATE_SKELETAL_MUSCLE_FIGURE].scale,
            )
            basal_met = convert_data_util(
                bodyIndexList[ValueType.BASAL_METABOLISM_FIGURE].value,
                bodyIndexList[ValueType.BASAL_METABOLISM_FIGURE].scale,
            )
            metabolic_age = convert_data_util(
                bodyIndexList[ValueType.BIOLOGICAL_AGE_FIGURE].value,
                bodyIndexList[ValueType.BIOLOGICAL_AGE_FIGURE].scale,
                int,
            )
            visceral_fat_rating = convert_data_util(
                bodyIndexList[ValueType.VISCERAL_FAT_FIGURE].value,
                bodyIndexList[ValueType.VISCERAL_FAT_FIGURE].scale,
            )
            bmi = convert_data_util(
                bodyIndexList[ValueType.BMI_FIGURE].value,
                bodyIndexList[ValueType.BMI_FIGURE].scale,
            )
            timeZone = pytz.timezone(m["timeZone"])

            wm = WeightMeasurement(
                weight=weight,
                measurementDate=m["measureDateTo"],
                timeZone=timeZone,
                bmiValue=bmi,
                bodyFatPercentage=bodyFatPercentage,
                restingMetabolism=basal_met,
                skeletalMusclePercentage=skeletalMusclePercentage,
                visceralFatLevel=visceral_fat_rating,
                metabolicAge=metabolic_age,
            )
            measurements.append(wm)

        return measurements


class OmronConnect2(OmronConnect):
    _APP_NAME = "OCM"
    _APP_VERSION = "8.2.1"
    _USER_AGENT = (
        f"OMRON connect/{_APP_VERSION} (com.omronhealthcare.omronconnect; build:24; iOS 18.7.2) Alamofire/5.9.1"
    )

    # monkey-patch httpx so checksum(req.content) works with omron servers.
    # pylint: disable=protected-access
    httpx._content.json_dumps = lambda obj, **kw: json.dumps(obj, **{**kw, "separators": (",", ":")})

    def __init__(self, server: str, country: str):
        self._server = server
        self._country = country
        self._headers: T.Dict[str, str] = {}
        self._email: str = ""

        # Wrap transport with LogTransport for debug logging with credential redaction (always)
        # Keep checksum event hook (required for API v2)
        self._client = httpx.Client(
            transport=HttpxLogTransport(httpx.HTTPTransport()),
            event_hooks={"request": [_http_add_checksum]},
            headers={
                "user-agent": OmronConnect2._USER_AGENT,
            },
        )
        # some oi-api.ohiomron.xxx/app request require /v2
        self._v2 = "/v2" if "/app" in server else ""

    def login(self, email: str, password: str, country: str) -> T.Optional[str]:
        data = {
            "emailAddress": email,
            "password": password,
            "country": country,
            "app": self._APP_NAME,
        }
        r = self._client.post(f"{self._server}/login", json=data)
        r.raise_for_status()

        resp = r.json()
        try:
            accessToken = resp["accessToken"]
            refreshToken = resp["refreshToken"]
            self._headers["authorization"] = f"{accessToken}"
            self._email = email
            self._country = country
            return refreshToken

        except KeyError:
            L.error(f"login() -> {r}: '{r.text}'")

        return None

    def refresh_oauth2(self, refresh_token: str, **kwargs: T.Any) -> T.Optional[str]:
        data = {
            "app": self._APP_NAME,
            "emailAddress": kwargs.get("email", self._email),
            "refreshToken": refresh_token,
        }

        r = self._client.post(f"{self._server}/login", json=data, headers=self._headers)
        r.raise_for_status()

        resp = r.json()
        try:
            accessToken = resp["accessToken"]
            refreshToken = resp["refreshToken"]
            self._headers["authorization"] = f"{accessToken}"
            return refreshToken

        except KeyError:
            L.error(f"refresh_oauth2() -> {r}: '{r.text}'")

        return None

    def get_user(self) -> T.Dict[str, T.Any]:
        r = self._client.get(f"{self._server}/user?app={self._APP_NAME}", headers=self._headers)
        r.raise_for_status()
        resp = r.json()

        return resp["data"]

    def get_registered_devices(self, days: T.Optional[int] = 30) -> T.Optional[list[OmronDevice]]:
        """Fetch registered devices (API v2 returns all active devices, days parameter is ignored)."""

        r = self._client.get(f"{self._server}{self._v2}/init-user?app={self._APP_NAME}", headers=self._headers)
        r.raise_for_status()
        resp = r.json()
        device_list = resp.get("data", {}).get("deviceList", [])

        result: list[OmronDevice] = []
        for device in device_list:
            attrs = device.get("attributes", {})

            if not attrs.get("isActive", 0):
                continue

            macAddress = attrs.get("macAddress", "").strip()
            if not macAddress:
                L.debug(f"Skipping device without MAC address: {attrs.get('name', 'unknown')}")
                continue

            try:
                deviceCategory = attrs["deviceCategory"]
                category = DeviceCategory(str(deviceCategory))

            except (ValueError, KeyError):
                L.debug(f"Skipping device with unsupported category: {attrs.get('name', 'unknown')}")
                continue

            # Create OmronDevice
            deviceModel = attrs.get("deviceModel", attrs.get("identifier", "Unknown"))
            userNumberInDevice = int(attrs.get("userNumberInDevice", 1))

            ocDev = OmronDevice(
                category=category,
                name=f"{deviceModel}:{userNumberInDevice}",
                macaddr=macAddress,
                user=userNumberInDevice,
            )
            result.append(ocDev)

        return result

    def get_bp_measurements(
        self, nextpaginationKey: int = 0, lastSyncedTime: int = 0, phoneIdentifier: str = ""
    ) -> T.List[T.Dict[str, T.Any]]:
        _lastSyncedTime = "" if lastSyncedTime <= 0 else lastSyncedTime
        r = self._client.get(
            f"{self._server}{self._v2}/sync/bp?nextpaginationKey={nextpaginationKey}"
            f"&lastSyncedTime={_lastSyncedTime}&phoneIdentifier={phoneIdentifier}",
            headers=self._headers,
        )
        r.raise_for_status()
        resp = r.json()

        if _debugSaveResponse:
            fname = f".debug/{lastSyncedTime}_bpm_v2.json"
            U.json_save(fname, resp)

        return resp["data"]

    def get_weighins(
        self, nextpaginationKey: int = 0, lastSyncedTime: int = 0, phoneIdentifier: str = ""
    ) -> T.List[T.Dict[str, T.Any]]:
        _lastSyncedTime = "" if lastSyncedTime <= 0 else lastSyncedTime
        r = self._client.get(
            f"{self._server}{self._v2}/sync/weight?nextpaginationKey={nextpaginationKey}"
            f"&lastSyncedTime={_lastSyncedTime}&phoneIdentifier={phoneIdentifier}",
            headers=self._headers,
        )
        r.raise_for_status()
        resp = r.json()

        if _debugSaveResponse:
            fname = f".debug/{lastSyncedTime}_weight_v2.json"
            U.json_save(fname, resp)

        return resp["data"]

    def get_measurements(
        self, device: OmronDevice, searchDateFrom: int = 0, searchDateTo: int = 0
    ) -> T.List[MeasurementTypes]:
        user = int(device.user)

        def filter_measurements(data: T.List[T.Dict[str, T.Any]]) -> T.List[MeasurementTypes]:
            r: T.List[MeasurementTypes] = []
            for m in data:
                userNumberInDevice = int(m["userNumberInDevice"])
                if user >= 0 and userNumberInDevice != user:
                    L.debug(f"skipping user: {user} != {userNumberInDevice}")
                    continue

                measurementDate = int(m["measurementDate"])
                if 0 < searchDateTo < measurementDate:
                    L.debug(f"skipping date: {measurementDate} > {searchDateTo}")
                    continue

                if int(m["isManualEntry"]):
                    L.debug("skipping manual entry")
                    continue

                if device.category == DeviceCategory.BPM:
                    # timezone(timedelta(seconds=int(m["timeZone"])))

                    bpm = BPMeasurement(
                        systolic=m["systolic"],
                        diastolic=m["diastolic"],
                        pulse=m["pulse"],
                        measurementDate=measurementDate,
                        timeZone=pytz.FixedOffset(int(m["timeZone"]) // 60),
                        irregularHB=int(m["irregularHB"]) != 0,
                        movementDetect=int(m["movementDetect"]) != 0,
                        cuffWrapDetect=int(m["cuffWrapDetect"]) != 0,
                        notes=m.get("notes", ""),
                    )
                    r.append(bpm)

                elif device.category == DeviceCategory.SCALE:
                    weight = float(m["weight"])
                    weightInLbs = float(m["weightInLbs"])
                    if weight <= 0 < weightInLbs:
                        weight = weightInLbs * 0.453592

                    # metabolicAge not observed in v2 API responses
                    wm = WeightMeasurement(
                        weight=weight,
                        measurementDate=measurementDate,
                        timeZone=pytz.FixedOffset(int(m["timeZone"]) // 60),
                        bmiValue=m["bmiValue"],
                        bodyFatPercentage=m["bodyFatPercentage"],
                        restingMetabolism=m["restingMetabolism"],
                        skeletalMusclePercentage=m["skeletalMusclePercentage"],
                        visceralFatLevel=m["visceralFatLevel"],
                        notes=m.get("notes", ""),
                    )
                    r.append(wm)

            return r

        data = None
        if device.category == DeviceCategory.BPM:
            data = self.get_bp_measurements(lastSyncedTime=searchDateFrom)

        elif device.category == DeviceCategory.SCALE:
            data = self.get_weighins(lastSyncedTime=searchDateFrom)

        return filter_measurements(data) if data else []


########################################################################################################################


def get_omron_connect(server: str, country: str) -> OmronConnect:
    if re.search(r"data-([a-z]{2})\.omronconnect\.com", server):
        return OmronConnect1(server, country)

    return OmronConnect2(server, country)


def try_servers(
    servers: T.List[str], country: str, operation: T.Callable[[OmronConnect], T.Any]
) -> T.Tuple[OmronConnect, T.Any]:
    for server in servers:
        try:
            oc = get_omron_connect(server, country)
            result = operation(oc)
            return oc, result

        except (httpx.ConnectError, httpx.TimeoutException, HTTPStatusError):
            continue

    raise ConnectionError("All servers failed")


class OmronClient:
    """
    Simplified OmronConnect client with country-based initialization.

    Handles server lookup and fallback internally. Delegates to OmronConnect1/2
    instances based on regional server compatibility.
    """

    def __init__(self, country: str):
        """
        Initialize client for a specific country.

        Args:
            country: ISO country code (e.g. 'US', 'JP', 'DE')

        Raises:
            ValueError: No servers available for country
        """

        self.country = country.upper()
        servers = get_servers_for_country_code(self.country)
        if not servers:
            raise ValueError(f"No servers available for country: {self.country}")

        self.servers: T.List[str] = servers
        self._active_client: T.Optional[OmronConnect] = None

    def login(self, email: str, password: str) -> T.Optional[str]:
        """
        Login with automatic server fallback.

        Returns:
            Refresh token if successful, None otherwise
        """

        def login_op(oc_instance: OmronConnect) -> T.Optional[str]:
            return oc_instance.login(email, password, self.country)

        self._active_client, refresh_token = try_servers(self.servers, self.country, login_op)
        return refresh_token

    def refresh_oauth2(self, refresh_token: str, **kwargs: T.Any) -> T.Optional[str]:
        """
        Refresh OAuth token with automatic server fallback.

        Returns:
            New refresh token if successful, None otherwise
        """

        def refresh_op(oc_instance: OmronConnect) -> T.Optional[str]:
            return oc_instance.refresh_oauth2(refresh_token, **kwargs)

        self._active_client, new_token = try_servers(self.servers, self.country, refresh_op)
        return new_token

    def get_measurements(
        self, device: OmronDevice, searchDateFrom: int = 0, searchDateTo: int = 0
    ) -> T.List[MeasurementTypes]:
        """Delegate to active client."""

        if not self._active_client:
            raise RuntimeError("Not connected - call login() or refresh_oauth2() first")

        return self._active_client.get_measurements(device, searchDateFrom, searchDateTo)

    def get_user(self) -> T.Dict[str, T.Any]:
        """Delegate to active client."""

        if not self._active_client:
            raise RuntimeError("Not connected - call login() or refresh_oauth2() first")

        return self._active_client.get_user()

    def get_registered_devices(self, days: T.Optional[int]) -> T.Optional[list[OmronDevice]]:
        """Delegate to active client."""

        if not self._active_client:
            raise RuntimeError("Not connected - call login() or refresh_oauth2() first")

        return self._active_client.get_registered_devices(days=days)


########################################################################################################################

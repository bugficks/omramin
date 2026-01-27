########################################################################################################################

import typing as T  # isort: split

import collections.abc
import dataclasses
import json
import pathlib
import re
import types
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from difflib import SequenceMatcher
from functools import reduce

import json5
from dateutil.parser import parse as dateutil_parse

########################################################################################################################

KeyType = T.TypeVar("KeyType")
ValueType = T.TypeVar("ValueType")

########################################################################################################################

# match case insensitive UUIDs with or without dashes
RX_UUID = re.compile(r"([0-9a-f]{32}|[0-9a-f-]{36})\Z", re.I)
RX_MACADDR = re.compile(r"^([0-9a-f]{2}[:-]){5}([0-9a-f]{2})$", re.I)

########################################################################################################################


# https://stackoverflow.com/a/17388505
def strsimilar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def sum_dict_value(d: T.Dict[T.Any, T.Any], key: T.Any) -> T.Any:
    return reduce(lambda a, b: a + b, map(lambda o: o[key], d))


########################################################################################################################


def is_valid_macaddr(macaddr: str) -> bool:
    return bool(RX_MACADDR.match(macaddr))


########################################################################################################################


class DataclassBase:
    # e.g.
    #   class SomeEnum(DataclassBase, StrEnum):
    #     ...
    # @dataclass(kw_only=True)
    # class SomeClass(DataclassBase):
    #     ...

    def to_dict(self) -> T.Dict[T.Any, T.Any]:
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)

        if isinstance(self, types.SimpleNamespace):
            return self.__dict__.copy()

        raise TypeError(f"'{self}' is not a dataclass or SimpleNamespace")

    def to_json(self) -> str:
        return json_beautify(self.to_dict())

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    @classmethod
    def from_dict(cls, obj):
        """Ignore extra keys/fields when creating dataclass from dict"""

        fieldnames = [f.name for f in dataclasses.fields(cls)]
        # https://stackoverflow.com/a/55096964
        return cls(**{k: v for k, v in obj.items() if k in fieldnames})


class SimpleNamespaceEx(types.SimpleNamespace, DataclassBase):
    pass


########################################################################################################################


# https://stackoverflow.com/a/51286749
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: T.Any) -> T.Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore[call-overload,arg-type]

        if isinstance(o, types.SimpleNamespace):
            return o.__dict__

        if isinstance(o, tzinfo):
            return str(o)

        return super().default(o)


def json_beautify(obj: T.Any) -> str:
    return json.dumps(obj, indent=4, sort_keys=True, cls=EnhancedJSONEncoder)


def json_print(obj) -> None:
    print(json_beautify(obj))


def json_save(fname: T.Union[pathlib.Path, str], obj: T.Any) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=False, cls=EnhancedJSONEncoder)


def json_load(fname: T.Union[pathlib.Path, str], object_hook: T.Optional[T.Callable] = None) -> T.Any:
    with open(fname, "r", encoding="utf-8") as f:
        return json5.load(f, object_hook=object_hook)
        # return json.load(f, object_hook=lambda d: types.SimpleNamespace(**d))


def json_load_file(f: T.IO, object_hook: T.Optional[T.Callable] = None) -> T.Any:
    f.seek(0)
    return json5.load(f, object_hook=object_hook)


def json_save_file(f: T.IO, obj: T.Any) -> None:
    f.seek(0)
    f.truncate()
    json.dump(obj, f, indent=4, sort_keys=False, cls=EnhancedJSONEncoder)


########################################################################################################################


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def utcfromtimestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def utcdatetimefromstr(dt: str) -> datetime:
    return dateutil_parse(dt).astimezone(timezone.utc)


def utcdatefromstr(dt: str) -> date:
    return utcdatetimefromstr(dt).date()


def utctimefromstr(dt: str) -> time:
    return utcdatetimefromstr(dt).time()


def utcfromtimestamp_isoformat(timestamp: float, timespec: str = "seconds") -> str:
    return utcfromtimestamp(timestamp).isoformat(timespec=timespec).replace("+00:00", "Z")


# https://codeigo.com/python/remove-seconds-from-datetime/
def minuteround(dt: datetime) -> datetime:
    # Round to the nearest minute. If second<30 set it to zero and leave minutes
    # unchanges. Otherwise set seconds to zero and increase minutes by 1.
    return dt.replace(second=0, microsecond=0, hour=dt.hour) + timedelta(minutes=dt.second // 30)


# https://stackoverflow.com/a/1060330
def daterange(start_date: date, end_date: date) -> T.Generator[date, None, None]:
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def datefromdatetime(dt: datetime) -> date:
    return datetime.combine(dt.date(), datetime.min.time()).date()


########################################################################################################################


# https://stackoverflow.com/a/3233356
def deep_update(d: T.Dict[KeyType, T.Any], u: T.Dict[KeyType, T.Any], *, existing=True) -> T.Dict[KeyType, T.Any]:
    r = d.copy()
    for k, v in u.items():
        if existing and k not in r:
            continue
        # pylint: disable-next=no-member
        if isinstance(v, collections.abc.Mapping) and isinstance(v, dict):
            r[k] = deep_update(r.get(k, type(r)()), v, existing=existing)

        else:
            r[k] = v

    return r


# https://stackoverflow.com/a/43228384
def deep_merge(d: T.Dict[KeyType, T.Any], u: T.Dict[KeyType, T.Any], *, existing=True) -> T.Dict[KeyType, T.Any]:
    """Return a new dictionary by merging two dictionaries recursively."""

    r = deepcopy(d)

    for k, v in u.items():
        if existing and k not in d:
            continue
        # pylint: disable-next=no-member
        if isinstance(v, collections.abc.Mapping) and isinstance(v, dict):
            r[k] = deep_merge(r.get(k, type(r)()), v, existing=existing)

        else:
            r[k] = deepcopy(u[k])

    return r


########################################################################################################################

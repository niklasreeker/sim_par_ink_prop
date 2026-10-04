"""Interpret TIA UTC times and the working CSV's German local times safely."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


LOCAL_TIMEZONE = "Europe/Berlin"
UTC_TIME_COLUMN = "UTC Time"
LOCAL_TIME_COLUMN = "Deutsche Zeit"
OFFSET_COLUMN = "UTC Offset"


@lru_cache(maxsize=1)
def german_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(LOCAL_TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            "Zeitzonendaten für Europe/Berlin fehlen. Bitte Python aus der "
            "Projektumgebung verwenden oder das Paket tzdata installieren."
        ) from exc


def iso_date(value: str) -> str:
    value = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    for pattern in ("%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Ungültiges Messdatum: {value}")


def format_time(timestamp: datetime) -> str:
    precision = "milliseconds" if timestamp.microsecond % 1000 == 0 else "microseconds"
    return timestamp.replace(tzinfo=None).time().isoformat(timespec=precision)


def format_offset(timestamp: datetime) -> str:
    value = timestamp.strftime("%z")
    return value[:3] + ":" + value[3:]


def timestamp_utc(date_value: str, time_value: str, *, offset: str | None = None) -> datetime:
    """An offset marks local input and disambiguates the repeated autumn hour."""
    naive = datetime.fromisoformat(iso_date(date_value) + "T" + str(time_value).strip())
    if naive.tzinfo is not None:
        raise ValueError("Uhrzeit und Zeitzonenoffset müssen in getrennten Spalten stehen.")
    if offset is None:
        return naive.replace(tzinfo=timezone.utc)
    offset = str(offset).strip()
    if not re.fullmatch(r"[+-]\d{2}:\d{2}", offset):
        raise ValueError(f"Ungültiger UTC Offset für deutsche Ortszeit: {offset}")
    aware = datetime.fromisoformat(naive.isoformat() + offset)
    utc = aware.astimezone(timezone.utc)
    local = utc.astimezone(german_timezone())
    if local.replace(tzinfo=None) != naive or local.utcoffset() != aware.utcoffset():
        raise ValueError("Deutsche Uhrzeit und UTC Offset passen nicht zu Europe/Berlin.")
    return utc


def resolve_time_column(columns, requested: str = UTC_TIME_COLUMN) -> str:
    if requested == UTC_TIME_COLUMN and requested not in columns and LOCAL_TIME_COLUMN in columns:
        return LOCAL_TIME_COLUMN
    if requested not in columns:
        raise ValueError(f"Messzeit-Spalte fehlt: {requested}")
    return requested


def measurement_timestamps_utc(frame, date_column: str = "Date", time_column: str = UTC_TIME_COLUMN):
    """Return aware UTC timestamps for calculations; CSV display stays local.

    pandas is imported only for evaluations. The importer needs no pandas.
    Invalid individual values become NaT for the caller's quality/error handling.
    """
    import pandas as pd

    selected = resolve_time_column(frame.columns, time_column)
    if date_column not in frame.columns:
        raise ValueError(f"Messdatum-Spalte fehlt: {date_column}")
    local_input = selected == LOCAL_TIME_COLUMN
    if local_input and OFFSET_COLUMN not in frame.columns:
        raise ValueError(f"Deutsche Messzeiten benötigen die Spalte {OFFSET_COLUMN}.")
    offsets = frame[OFFSET_COLUMN] if local_input else [None] * len(frame)
    values = []
    for date_value, time_value, offset in zip(frame[date_column], frame[selected], offsets):
        try:
            values.append(timestamp_utc(date_value, time_value, offset=offset))
        except (ValueError, TypeError, OverflowError):
            values.append(pd.NaT)
    return pd.Series(pd.to_datetime(values, utc=True), index=frame.index)

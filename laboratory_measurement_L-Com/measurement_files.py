"""Shared discovery of the accumulated measurement CSV and legacy exports."""

from __future__ import annotations

import csv
import re
from pathlib import Path


MASTER_NAME = re.compile(r"Messdaten_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv", re.IGNORECASE)


def find_master_file(folder: Path) -> Path | None:
    """Return the single working CSV; never silently choose between archives."""
    files = sorted(p for p in Path(folder).glob("*.csv") if MASTER_NAME.fullmatch(p.name))
    if len(files) > 1:
        raise ValueError(
            "Mehrere gemeinsame Messdateien gefunden: "
            + ", ".join(p.name for p in files)
            + ". Bitte nur die aktuelle Messdaten-Datei in diesem Ordner behalten."
        )
    return files[0] if files else None


def is_measurement_csv(path: Path) -> bool:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            lines = (line for line in handle if line.strip() and not line.lstrip().startswith(("/", "#")))
            header = next(csv.reader(lines), [])
        return {"ProbeNr", "Rho_M", "C_M", "m_SL120"}.issubset(c.strip() for c in header)
    except (OSError, UnicodeError, csv.Error):
        return False


def discover_measurement_files(folder: Path, *, prefer_master: bool = True) -> list[Path]:
    """Use the working CSV by default, avoiding overlapping TIA exports."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    if prefer_master:
        master = find_master_file(folder)
        if master is not None:
            return [master]
    return sorted(
        (p for p in folder.glob("*.csv") if is_measurement_csv(p)),
        key=lambda p: (p.name.casefold(), p.name),
    )

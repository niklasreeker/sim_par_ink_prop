#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TIA-Ringpuffer exportieren, korrigieren und in einer Messdatei sammeln.

Ohne Argumente: interaktive Auswahl zum Erstellen/Erweitern.
--update: alle vorhandenen Roh-Exporte ab Kennfeld_v2 (25).csv importieren.
--files DATEI ...: nur diese Exporte importieren (beim ersten Mal plus Version 25).
--dry-run: Import prüfen, ohne Dateien zu verändern.
Die Roh-Exporte bleiben unverändert; _korrigiert-Dateien werden nicht importiert.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from measurement_files import find_master_file


DEFAULT_FOLDER = Path(__file__).resolve().parent
EXPORT_NAME = re.compile(r"Kennfeld_v2 \((\d+)\)\.csv", re.IGNORECASE)
FIRST_VERSION = 25
REQUIRED_COLUMNS = {
    "SeqNo", "Date", "UTC Time", "Nr", "ProbeNr", "m_SL120", "m_Wasser",
    "m_IPA", "m_PG", "m_MG", "Rho_M", "C_M", "T_M",
}
TEXT_COLUMNS = {"Date", "UTC Time"}
MASS_COLUMNS = {"m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"}


@dataclass
class ImportResult:
    path: Path
    total: int
    added: int
    duplicates: int
    corrections: dict[str, int]
    changed: bool
    gaps: list[tuple[int, int]]


def export_version(path: Path) -> int:
    match = EXPORT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Kein TIA-Rohexport: {path.name}")
    return int(match[1])


def find_exports(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.glob("*.csv")
         if EXPORT_NAME.fullmatch(p.name) and export_version(p) >= FIRST_VERSION),
        key=lambda p: (export_version(p), p.name),
    )


def measurement_time(row: dict[str, str]) -> datetime:
    date_text = row["Date"]
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", date_text):
        date_text = datetime.strptime(date_text, "%d.%m.%Y").strftime("%Y-%m-%d")
    return datetime.fromisoformat(date_text + "T" + row["UTC Time"])


def measurement_key(row: dict[str, str]) -> tuple[int, datetime]:
    # SeqNo continues past the ring-buffer wrap. A PLC reset may reuse it;
    # the timestamp distinguishes those genuinely new measurements.
    return int(Decimal(row["SeqNo"])), measurement_time(row)


def row_signature(row: dict[str, str], columns: list[str]) -> tuple:
    # Whitespace and equivalent scientific notation are not data changes.
    # TIA also exports NaN/+INF in unusable sensor statistics. Preserve those
    # records and compare NaN through a stable token (NaN != NaN numerically).
    def numeric_token(value: str):
        number = Decimal(value)
        return "NaN" if number.is_nan() else number

    return tuple(
        measurement_time(row) if column == "Date" else
        None if column == "UTC Time" else numeric_token(row[column])
        for column in columns
    )


def correct_row(row: dict[str, str]) -> set[str]:
    fixes = set()
    seq, timestamp = measurement_key(row)
    probe = Decimal(row["ProbeNr"])
    # Restrict historical corrections to the documented experiment, even if
    # SeqNo or sample numbers are reused after a future PLC reset.
    if (132 <= seq <= 142 and probe == 3
            and timestamp.date().isoformat() == "2026-08-21"
            and abs(Decimal(row["m_PG"]) - Decimal("24.60")) > Decimal("0.001")):
        row["m_PG"] = "2.460000E+1"
        fixes.add("PG")
    if timestamp.date().isoformat() == "2026-09-16" and probe == 11:
        row["ProbeNr"] = "12"
        fixes.add("ProbeNr")
    if probe == 300:
        composition = {"m_SL120": "8.000000E+2", "m_Wasser": "8.000000E+3", "m_MG": "2.000000E+1"}
        if any(abs(Decimal(row[c]) - Decimal(value)) >= Decimal("0.001")
               for c, value in composition.items()):
            row.update(composition)
            fixes.add("Probe300")
    return fixes


def read_measurements(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Validate the entire export before an existing working CSV is changed."""
    rows = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            lines = (line for line in handle if line.strip() and not line.lstrip().startswith(("/", "#")))
            reader = csv.reader(lines, strict=True)
            columns = [c.strip() for c in next(reader, [])]
            missing = REQUIRED_COLUMNS - set(columns)
            if missing or len(columns) != len(set(columns)):
                raise ValueError(f"{path.name}: ungültige CSV-Spalten; fehlend: {sorted(missing)}")
            for number, values in enumerate(reader, start=2):
                if len(values) != len(columns):
                    raise ValueError(f"{path.name}, Datensatz {number}: unvollständige CSV-Zeile.")
                row = dict(zip(columns, (v.strip() for v in values)))
                try:
                    for column in columns:
                        if column not in TEXT_COLUMNS:
                            value = Decimal(row[column])
                            if column in MASS_COLUMNS and not value.is_finite():
                                raise ValueError(f"{column} ist keine endliche Einwaage")
                    for column in ("SeqNo", "Nr", "ProbeNr"):
                        value = Decimal(row[column])
                        if not value.is_finite() or value < 0 or value != value.to_integral_value():
                            raise ValueError(f"{column} ist keine nichtnegative Ganzzahl")
                    timestamp = measurement_time(row)
                    if timestamp.tzinfo is not None:
                        raise ValueError("UTC Time muss wie beim TIA-Export ohne Zeitzonenangabe sein")
                except (ValueError, InvalidOperation) as exc:
                    raise ValueError(f"{path.name}, Datensatz {number}: {exc}") from exc
                rows.append(row)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"{path.name}: CSV nicht lesbar ({exc})") from exc
    if not rows:
        raise ValueError(f"{path.name}: enthält keine Messungen.")
    return columns, rows


@contextmanager
def import_lock(folder: Path):
    lock = folder / ".messdaten_import.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(
            "Ein Messdaten-Import läuft bereits. Falls ein vorheriger Lauf abgestürzt ist, "
            "die Datei .messdaten_import.lock nach dem Schließen des anderen Programms entfernen."
        ) from exc
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def save_measurements(folder: Path, previous: Path | None, columns: list[str],
                      rows: list[dict[str, str]], now: datetime) -> Path:
    output = folder / f"Messdaten_{now:%Y-%m-%d_%H-%M-%S}.csv"
    if output.exists() and output != previous:
        raise ValueError(f"Zieldatei existiert bereits: {output.name}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=folder, prefix=".messdaten_", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomically replace the contents first. If renaming fails (e.g. an
        # open Excel file), the complete data still exists at the old path.
        os.replace(temporary, previous or output)
        if previous is not None and previous != output:
            try:
                previous.rename(output)
            except OSError as exc:
                raise OSError(
                    f"Messdaten gespeichert in {previous.name}, aber Umbenennung fehlgeschlagen. "
                    "Der Dateiname blieb unverändert; die Messungen sind vollständig gespeichert."
                ) from exc
        return output
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def update_measurements(folder: Path = DEFAULT_FOLDER, sources: list[Path] | None = None,
                        *, dry_run: bool = False, now: datetime | None = None) -> ImportResult:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Messdatenordner nicht gefunden: {folder}")
    with import_lock(folder):
        previous = find_master_file(folder)
        exports = find_exports(folder) if sources is None else [Path(p).resolve() for p in sources]
        for path in exports:
            if path.parent != folder or export_version(path) < FIRST_VERSION:
                raise ValueError(f"Nur TIA-Rohexporte ab Version 25 aus {folder} importieren: {path}")
        if previous is None:
            base = next((p for p in find_exports(folder) if export_version(p) == FIRST_VERSION), None)
            if base is None:
                raise FileNotFoundError("Für den ersten Import fehlt Kennfeld_v2 (25).csv mit den ältesten Messungen.")
            exports.insert(0, base)
        exports = sorted(set(exports), key=export_version)
        if not exports:
            raise FileNotFoundError("Keine Kennfeld_v2 (XX).csv ab Version 25 zum Importieren gefunden.")

        merged = {}
        signatures = {}
        corrections = {name: set() for name in ("PG", "ProbeNr", "Probe300")}
        columns = None
        added = duplicates = 0
        corrected_existing = False
        inputs = ([previous] if previous else []) + exports
        for path in inputs:
            header, rows = read_measurements(path)
            if columns is None:
                columns = header
            elif set(header) != set(columns):
                raise ValueError(f"{path.name}: CSV-Spalten passen nicht zur gemeinsamen Messdatei.")
            for row in rows:
                key = measurement_key(row)
                fixes = correct_row(row)
                for name in fixes:
                    corrections[name].add(key)
                if path == previous and fixes:
                    corrected_existing = True
                signature = row_signature(row, columns)
                if key in merged:
                    if signatures[key] != signature:
                        raise ValueError(
                            f"Widersprüchliche Messwerte in {path.name}: SeqNo {key[0]}, "
                            f"{key[1].isoformat(sep=' ')}. Import abgebrochen; Werte zuerst prüfen."
                        )
                    if path != previous:
                        duplicates += 1
                    continue
                merged[key] = row
                signatures[key] = signature
                if path != previous:
                    added += 1

        rows = [merged[key] for key in sorted(merged, key=lambda key: (key[1], key[0]))]
        sequences = sorted({key[0] for key in merged})
        gaps = [(left + 1, right - 1) for left, right in zip(sequences, sequences[1:]) if right > left + 1]
        changed = previous is None or added > 0 or corrected_existing
        timestamp = now or datetime.now().astimezone()
        output = folder / f"Messdaten_{timestamp:%Y-%m-%d_%H-%M-%S}.csv" if changed else previous
        if changed and not dry_run:
            output = save_measurements(folder, previous, columns, rows, timestamp)
        return ImportResult(output, len(rows), added, duplicates,
                            {name: len(keys) for name, keys in corrections.items()}, changed, gaps)


def print_result(result: ImportResult, dry_run: bool) -> None:
    print(f"\n{'Vorschau' if dry_run else 'Messdatei'}: {result.path.name}")
    print(f"Messpunkte insgesamt: {result.total}; neu: {result.added}; bereits vorhanden: {result.duplicates}")
    if any(result.corrections.values()):
        print("Bekannte Korrekturen beim Einlesen: " + ", ".join(f"{k}: {v}" for k, v in result.corrections.items()))
    if not result.changed:
        print("Keine neuen Messungen. Datei und Änderungszeit bleiben unverändert.")
    if dry_run:
        print("Es wurde keine CSV-Datei verändert.")
    if result.gaps:
        shown = ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in result.gaps[:10])
        print(f"Hinweis: Nicht vorhandene SeqNo: {shown}. Eventuell nie exportierte Messungen fehlen.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="Messdatenordner (Standard: Skriptordner).")
    parser.add_argument("--update", action="store_true", help="Alle Roh-Exporte ohne Menü importieren.")
    parser.add_argument("--files", type=Path, nargs="+", help="Bestimmte Roh-Exporte importieren.")
    parser.add_argument("--dry-run", action="store_true", help="Import nur prüfen, keine CSV schreiben.")
    args = parser.parse_args(argv)
    folder = args.folder.expanduser().resolve()
    try:
        sources = None
        if args.files:
            sources = [p if p.is_absolute() else folder / p for p in args.files]
        elif not args.update and not args.dry_run:
            current = find_master_file(folder)
            print("MESSDATEN SAMMELN UND KORRIGIEREN")
            print(f"Ordner: {folder}")
            print(f"Aktuelle Messdatei: {current.name}" if current else "Noch keine gemeinsame Messdatei vorhanden; Basis ist Version 25.")
            print("\n[1] Messdatei erweitern" if current else "\n[1] Messdatei erstmals erstellen")
            print("[0] Beenden")
            action = input("Auswahl [1]: ")
            # The action prompt and source menu both allow cancellation.
            if action.strip() == "0":
                return 0
            while action.strip() not in ("", "1"):
                action = input("Bitte 1 (fortfahren) oder 0 (beenden) eingeben [1]: ")
                if action.strip() == "0":
                    return 0
            exports = find_exports(folder)
            if not exports:
                raise FileNotFoundError("Keine TIA-Rohexporte ab Version 25 gefunden.")
            print("\nVorhandene TIA-Rohexporte:")
            for index, path in enumerate(exports, 1):
                print(f"  {index}) {path.name}")
            while True:
                choice = input("Alle importieren [Enter/A], Dateinummer(n), z.B. 3,4, oder 0 zum Beenden: ").strip()
                if choice == "0":
                    return 0
                if not choice or choice.upper() == "A":
                    break
                try:
                    indices = [int(value.strip()) for value in choice.split(",")]
                    if not all(1 <= i <= len(exports) for i in indices):
                        raise ValueError
                    sources = [exports[i - 1] for i in indices]
                    break
                except ValueError:
                    print("Bitte angezeigte Nummern, A oder 0 eingeben.")
        result = update_measurements(folder, sources, dry_run=args.dry_run)
        print_result(result, args.dry_run)
        return 0
    except (OSError, ValueError) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nImport abgebrochen.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

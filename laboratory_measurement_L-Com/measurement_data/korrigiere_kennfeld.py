#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatische, wiederholbar ausführbare Korrektur der Kennfeld-Messdaten.

Enthaltene Datenkorrekturen:

1. Probe 3, SeqNo 132 bis 142: fehlende PG-Zugabe von 24,60 g.
2. Probe 11 vom 09.09.2026, ab SeqNo 1013: fehlende Wasserzugabe von
   29,78 g. Da die CSV kumulierte Einwaagen enthält, werden auch alle
   nachfolgenden Zeilen dieser Probe um 29,78 g erhöht.

Die Erkennung ist idempotent: Bereits korrigierte Dateien werden bei einem
erneuten Lauf nicht ein zweites Mal verändert.
"""

import re
import sys
from pathlib import Path

TARGET_PG = 24.60
PG_PROBE = 3
PG_SEQ_RANGE = range(132, 143)  # SeqNo 132 bis 142 inklusive

WATER_PROBE = 11
WATER_DATE = "2026-09-09"
WATER_START_SEQ = 1013
WATER_DELTA = 29.78
# Der Wert am Beginn des betroffenen Abschnitts dient als eindeutiger Marker,
# damit die kumulierte Zugabe niemals doppelt addiert wird.
WATER_ORIGINAL_AT_START = 1000.40
WATER_CORRECTED_AT_START = WATER_ORIGINAL_AT_START + WATER_DELTA

FLOAT_TOLERANCE = 1e-3
REQUIRED_COLUMNS = {"SeqNo", "Date", "ProbeNr", "m_PG", "m_Wasser"}


def _parse_header(line):
    """Liefert Spaltenindizes oder None, wenn es kein Kennfeld-Format ist."""
    columns = [column.strip() for column in line.strip().split(",")]
    if not REQUIRED_COLUMNS.issubset(columns):
        return None
    return {column: columns.index(column) for column in REQUIRED_COLUMNS}


def _parse_row(line, columns):
    """Liest die für die Korrekturen benötigten Werte einer Datenzeile."""
    parts = line.split(",")
    if len(parts) <= max(columns.values()):
        return None
    try:
        return {
            "parts": parts,
            "seq": int(parts[columns["SeqNo"]].strip()),
            "date": parts[columns["Date"]].strip(),
            "probe": int(float(parts[columns["ProbeNr"]].strip())),
            "m_pg": float(parts[columns["m_PG"]].strip()),
            "m_wasser": float(parts[columns["m_Wasser"]].strip()),
        }
    except ValueError:
        # Überschrift, //END oder eine unvollständige Zeile
        return None


def _format_plc_float(value):
    """Formatiert eine Zahl wie die wissenschaftlichen Werte der PLC-CSV."""
    value_text = f"{value:.6E}"
    value_text = re.sub(r"E([+-])0+(\d+)$", r"E\1\2", value_text)
    return value_text.rjust(15)


def _analyze(lines, columns):
    pg_rows = []
    water_rows = []
    water_start = None

    for line in lines[1:]:
        row = _parse_row(line, columns)
        if row is None:
            continue

        if row["probe"] == PG_PROBE and row["seq"] in PG_SEQ_RANGE:
            pg_rows.append(row)

        is_water_campaign = (
            row["probe"] == WATER_PROBE
            and row["date"] == WATER_DATE
            and row["seq"] >= WATER_START_SEQ
        )
        if is_water_campaign:
            water_rows.append(row)
            if row["seq"] == WATER_START_SEQ:
                water_start = row["m_wasser"]

    pg_needs = sum(
        abs(row["m_pg"] - TARGET_PG) >= FLOAT_TOLERANCE for row in pg_rows
    )

    if water_start is None:
        water_state = "not_applicable"
    elif abs(water_start - WATER_ORIGINAL_AT_START) < FLOAT_TOLERANCE:
        water_state = "needs_correction"
    elif abs(water_start - WATER_CORRECTED_AT_START) < FLOAT_TOLERANCE:
        water_state = "already_corrected"
    else:
        water_state = "unknown"

    return {
        "pg_rows": pg_rows,
        "pg_needs": pg_needs,
        "water_rows": water_rows,
        "water_start": water_start,
        "water_state": water_state,
    }


def check_file_status(file_path):
    """
    Prüft, ob die Datei korrigiert werden muss, bereits korrigiert ist
    oder nicht das erwartete Format aufweist.
    Rückgabe: ('needs_correction' | 'already_corrected' | 'not_applicable', details)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return "error", str(e)

    if not lines:
        return "not_applicable", "Datei ist leer"

    columns = _parse_header(lines[0])
    if columns is None:
        return "not_applicable", "Kein passendes Kennfeld-CSV-Format"

    analysis = _analyze(lines, columns)
    applicable = bool(analysis["pg_rows"] or analysis["water_rows"])
    if not applicable:
        return "not_applicable", "Keine bekannte Korrekturstelle vorhanden"

    details = []
    if analysis["pg_rows"]:
        details.append(
            f"PG: {analysis['pg_needs']} von {len(analysis['pg_rows'])} Zeilen offen"
        )
    if analysis["water_state"] == "needs_correction":
        details.append(
            f"Wasser: {len(analysis['water_rows'])} Zeilen um 29,78 g erhöhen"
        )
    elif analysis["water_state"] == "already_corrected":
        details.append("Wasser: bereits korrigiert")
    elif analysis["water_state"] == "unknown":
        details.append(
            "Wasser: unerwarteter Startwert "
            f"{analysis['water_start']:.6f} g; keine automatische Änderung"
        )

    if analysis["pg_needs"] or analysis["water_state"] == "needs_correction":
        return "needs_correction", "; ".join(details)
    if analysis["water_state"] == "unknown":
        return "unknown", "; ".join(details)
    return "already_corrected", "; ".join(details)


def correct_file(input_path, output_path=None):
    """
    Führt die Korrektur an der angegebenen CSV-Datei durch.
    Format-treue Ersetzung: Nur Spalte m_PG (Index 8) wird angepasst.
    """
    input_p = Path(input_path)
    if output_path is None:
        stem = input_p.stem
        if stem.endswith("_korrigiert"):
            output_p = input_p
        else:
            output_p = input_p.with_name(f"{stem}_korrigiert.csv")
    else:
        output_p = Path(output_path)

    with open(input_p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Datei ist leer")
    columns = _parse_header(lines[0])
    if columns is None:
        raise ValueError("Kein passendes Kennfeld-CSV-Format")

    analysis = _analyze(lines, columns)
    correct_water = analysis["water_state"] == "needs_correction"

    corrected_lines = [lines[0]]
    change_count = 0

    for line in lines[1:]:
        row = _parse_row(line, columns)
        if row is not None:
            parts = row["parts"]

            if (row["probe"] == PG_PROBE
                    and row["seq"] in PG_SEQ_RANGE
                    and abs(row["m_pg"] - TARGET_PG) >= FLOAT_TOLERANCE):
                parts[columns["m_PG"]] = _format_plc_float(TARGET_PG)
                change_count += 1

            if (correct_water
                    and row["probe"] == WATER_PROBE
                    and row["date"] == WATER_DATE
                    and row["seq"] >= WATER_START_SEQ):
                parts[columns["m_Wasser"]] = _format_plc_float(
                    row["m_wasser"] + WATER_DELTA
                )
                change_count += 1

            line = ",".join(parts)
        corrected_lines.append(line)

    with open(output_p, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    return output_p, change_count


def list_csv_files(folder_path):
    """Listet alle CSV-Dateien im Ordner sortiert auf."""
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"), key=lambda p: p.name.lower())
    return csv_files


def main():
    script_dir = Path(__file__).resolve().parent
    print("=" * 70)
    print("   KENNFELD-CSV KORREKTUR-SKRIPT")
    print("=" * 70)
    print(f"Arbeitsverzeichnis: {script_dir}")
    print()

    csv_files = list_csv_files(script_dir)

    if not csv_files:
        print("Keine .csv Dateien im Verzeichnis gefunden!")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(0)

    print("Verfügbare CSV-Dateien:")
    print("-" * 70)

    file_statuses = []
    for idx, fpath in enumerate(csv_files, start=1):
        status, details = check_file_status(fpath)
        file_statuses.append((fpath, status, details))

        if status == "needs_correction":
            symbol = "[! KORREKTUR NÖTIG]"
        elif status == "already_corrected":
            symbol = "[✓ BEREITS KORRIGIERT]"
        else:
            symbol = "[–]"

        print(f" [{idx:2d}] {fpath.name}")
        print(f"      Status: {symbol} ({details})")

    print("-" * 70)
    print("  [A] Alle unkorrigierten Dateien auf einmal korrigieren")
    print("  [0] Beenden")
    print("-" * 70)

    while True:
        choice = input("\nBitte Nummer der zu korrigierenden Datei eingeben (oder 'A' / '0'): ").strip()

        if choice == "0":
            print("Vorgang beendet.")
            return

        if choice.upper() == "A":
            to_fix = [f for f, st, _ in file_statuses if st == "needs_correction"]
            if not to_fix:
                print("Keine Dateien gefunden, die eine Korrektur benötigen.")
            else:
                print(f"Korrigiere {len(to_fix)} Datei(en)...")
                for fpath in to_fix:
                    out_path, cnt = correct_file(fpath)
                    print(f"  -> {cnt} Zeilen korrigiert in: '{out_path.name}'")
                print("\nAlle anstehenden Dateien wurden erfolgreich korrigiert!")
            break

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(csv_files):
                selected_file, status, details = file_statuses[idx - 1]
                print(f"\nAusgewählt: '{selected_file.name}'")

                if status == "already_corrected":
                    re_run = input("Diese Datei scheint bereits korrigiert zu sein. Trotzdem erneut korrigieren? (j/N): ").strip().lower()
                    if re_run != "j":
                        continue

                out_path, cnt = correct_file(selected_file)
                print(f"Erfolg! {cnt} Zeilen wurden korrigiert.")
                print(f"Gespeichert unter: '{out_path.name}'")
                break
            else:
                print(f"Ungültige Nummer. Bitte zwischen 1 und {len(csv_files)} wählen.")
        else:
            print("Ungültige Eingabe. Bitte eine Zahl, 'A' oder '0' eingeben.")

    print("\nFertig.")


if __name__ == "__main__":
    main()

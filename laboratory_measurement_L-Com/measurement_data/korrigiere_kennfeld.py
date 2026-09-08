#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skript zur automatischen Korrektur der Kennfeld-Messdaten.
Fehler: In Versuch 3 (ProbeNr 3) bei SeqNo 132 bis 142 fehlte der Eintrag
        der PG-Zugabe (24.60 g PG).
Korrektur: Setzt m_PG in den betroffenen Zeilen auf '   2.460000E+1'.
"""

import os
import sys
from pathlib import Path

TARGET_PG_VAL = "   2.460000E+1"
SEQ_RANGE = range(132, 143)  # SeqNo 132 bis 142 inklusive


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

    header = lines[0].strip().split(",")
    if len(header) < 9 or "SeqNo" not in header[0] or "m_PG" not in header[8]:
        return "not_applicable", "Kein passendes Kennfeld-CSV-Format"

    found_seqs = []
    already_fixed_count = 0
    needs_fix_count = 0

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) >= 9:
            seq_str = parts[0].strip()
            if seq_str.isdigit():
                seq_num = int(seq_str)
                if seq_num in SEQ_RANGE:
                    found_seqs.append(seq_num)
                    m_pg_val = parts[8].strip()
                    try:
                        val_float = float(m_pg_val)
                        if abs(val_float - 24.60) < 1e-3:
                            already_fixed_count += 1
                        elif abs(val_float - 0.0) < 1e-3:
                            needs_fix_count += 1
                    except ValueError:
                        pass

    if not found_seqs:
        return "not_applicable", "Zeilen für SeqNo 132–142 nicht vorhanden"

    if already_fixed_count == len(SEQ_RANGE):
        return "already_corrected", "Bereits korrigiert (m_PG = 24.6g)"

    if needs_fix_count > 0:
        return "needs_correction", f"{needs_fix_count} Zeilen müssen korrigiert werden"

    return "unknown", "Unklarer Status"


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

    corrected_lines = []
    change_count = 0

    for line in lines:
        parts = line.split(",")
        if len(parts) >= 9:
            seq_str = parts[0].strip()
            probe_str = parts[4].strip() if len(parts) > 4 else ""
            if seq_str.isdigit():
                seq_num = int(seq_str)
                # Betrifft SeqNo 132 bis 142 (ProbeNr 3)
                if seq_num in SEQ_RANGE and (probe_str == "3" or probe_str == "3.0" or not probe_str):
                    parts[8] = TARGET_PG_VAL
                    line = ",".join(parts)
                    change_count += 1
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
    print("   KENNFELD-CSV KORREKTUR-SKRIPT (PG-Zugabe 24,60 g)")
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

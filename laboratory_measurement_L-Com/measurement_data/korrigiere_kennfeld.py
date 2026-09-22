#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatisiertes Korrekturskript für Kennfeld-Messdaten.

Korrekturen:
1. Versuch 3 (SeqNo 132 bis 142):
   m_PG = 24.60 g (PG-Zugabe vor IPA-Zugabe).
2. Probe 11 am 16.09.2026:
   ProbeNr wird von 11 auf 12 korrigiert (09.09. bleibt Probe 11).
"""

import sys
from pathlib import Path

TARGET_PG_VAL = "   2.460000E+1"
SEQ_RANGE_PG = range(132, 143)  # SeqNo 132 bis 142 inklusive


def check_file_status(file_path):
    """
    Prüft, ob eine CSV-Datei Korrekturbedarf hat.
    Rückgabe: (status, details, pg_needs_fix, probe_needs_fix)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        return "error", str(e), 0, 0

    if not lines:
        return "not_applicable", "Datei ist leer", 0, 0

    header = lines[0].strip().split(",")
    if len(header) < 9 or "SeqNo" not in header[0] or "m_PG" not in header[8]:
        return "not_applicable", "Kein passendes Kennfeld-CSV-Format", 0, 0

    pg_needs_fix = 0
    pg_fixed = 0
    probe_needs_fix = 0
    probe_fixed = 0

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) >= 9:
            seq_str = parts[0].strip()
            date_str = parts[1].strip()
            probe_str = parts[4].strip()

            # 1. PG-Prüfung (SeqNo 132..142)
            if seq_str.isdigit() and int(seq_str) in SEQ_RANGE_PG:
                try:
                    val_pg = float(parts[8].strip())
                    if abs(val_pg - 24.60) < 1e-3:
                        pg_fixed += 1
                    elif abs(val_pg - 0.0) < 1e-3:
                        pg_needs_fix += 1
                except ValueError:
                    pass

            # 2. ProbeNr-Prüfung (11 -> 12 am 16.09.)
            if "09-16" in date_str or "16.09" in date_str:
                if probe_str in ("11", "11.0"):
                    probe_needs_fix += 1
                elif probe_str in ("12", "12.0"):
                    probe_fixed += 1

    issues = []
    if pg_needs_fix > 0:
        issues.append(f"PG-Zugabe fehlt ({pg_needs_fix} Zeilen)")
    if probe_needs_fix > 0:
        issues.append(f"Probe 11 statt 12 am 16.09. ({probe_needs_fix} Zeilen)")

    if issues:
        return "needs_correction", " & ".join(issues), pg_needs_fix, probe_needs_fix

    already = []
    if pg_fixed == len(SEQ_RANGE_PG):
        already.append("PG = 24.6g korrigiert")
    if probe_fixed > 0:
        already.append(f"Probe 12 ({probe_fixed} Zeilen)")

    if already:
        return "already_corrected", f"Bereits korrigiert ({', '.join(already)})", 0, 0

    return "ok", "Keine bekannten Fehler vorhanden", 0, 0


def correct_file(input_path, output_path=None):
    """
    Führt beide Korrekturen zeichengenau durch und speichert die Datei ab.
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
    pg_fixes = 0
    probe_fixes = 0

    for line in lines:
        parts = line.split(",")
        if len(parts) >= 9:
            seq_str = parts[0].strip()
            date_str = parts[1].strip()
            probe_str = parts[4].strip()

            # 1. Korrektur: PG 24.60 g für SeqNo 132 bis 142
            if seq_str.isdigit() and int(seq_str) in SEQ_RANGE_PG:
                if probe_str in ("3", "3.0", ""):
                    try:
                        val_pg = float(parts[8].strip())
                        if abs(val_pg - 24.60) > 1e-3:
                            parts[8] = TARGET_PG_VAL
                            pg_fixes += 1
                    except ValueError:
                        pass

            # 2. Korrektur: ProbeNr 11 -> 12 am 16.09.
            if "09-16" in date_str or "16.09" in date_str:
                if probe_str in ("11", "11.0"):
                    width = len(parts[4])
                    parts[4] = f"{12:>{width}}"
                    probe_fixes += 1

            line = ",".join(parts)
        corrected_lines.append(line)

    with open(output_p, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    return output_p, pg_fixes, probe_fixes


def list_csv_files(folder_path):
    """Listet alle CSV-Dateien im angegebenen Ordner auf."""
    folder = Path(folder_path)
    return sorted(folder.glob("*.csv"), key=lambda p: p.name.lower())


def main():
    # Ermittelt den Ordner der Skriptdatei (unabhängig vom aktuellen Terminal-Pfad)
    script_dir = Path(__file__).resolve().parent

    print("=" * 75)
    print("   KENNFELD-CSV KORREKTUR-SKRIPT")
    print("   1) PG-Zugabe 24,60 g (SeqNo 132–142)")
    print("   2) Probe 11 -> Probe 12 (Messung am 16.09.)")
    print("=" * 75)
    print(f"Arbeitsverzeichnis: {script_dir}\n")

    csv_files = list_csv_files(script_dir)

    if not csv_files:
        print("Keine .csv Dateien im Verzeichnis gefunden!")
        print(f"Bitte stelle sicher, dass dieses Skript im Ordner mit den CSV-Dateien liegt.")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(0)

    print("Gefundene CSV-Dateien:")
    print("-" * 75)

    file_statuses = []
    for idx, fpath in enumerate(csv_files, start=1):
        status, details, pg_cnt, pr_cnt = check_file_status(fpath)
        file_statuses.append((fpath, status, details, pg_cnt, pr_cnt))

        if status == "needs_correction":
            symbol = "[! KORREKTUR NÖTIG]"
        elif status == "already_corrected":
            symbol = "[✓ BEREITS KORRIGIERT]"
        else:
            symbol = "[–]"

        print(f" [{idx:2d}] {fpath.name}")
        print(f"      Status: {symbol} ({details})")

    print("-" * 75)
    print("  [A] Alle unkorrigierten Dateien auf einmal korrigieren")
    print("  [0] Beenden")
    print("-" * 75)

    while True:
        choice = input("\nBitte Nummer der gewünschten Datei eingeben (oder 'A' / '0'): ").strip()

        if choice == "0":
            print("Vorgang beendet.")
            return

        if choice.upper() == "A":
            to_fix = [item for item in file_statuses if item[1] == "needs_correction"]
            if not to_fix:
                print("Keine Dateien gefunden, die eine Korrektur benötigen.")
            else:
                print(f"\nKorrigiere {len(to_fix)} Datei(en)...")
                for fpath, _, _, _, _ in to_fix:
                    out_path, pg_cnt, pr_cnt = correct_file(fpath)
                    print(f"  -> '{out_path.name}': {pg_cnt} PG-Zeilen & {pr_cnt} ProbeNr-Zeilen korrigiert.")
                print("\nAlle anstehenden Dateien wurden erfolgreich korrigiert!")
            break

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(csv_files):
                selected_file, status, details, _, _ = file_statuses[idx - 1]
                print(f"\nAusgewählt: '{selected_file.name}'")

                if status == "already_corrected":
                    re_run = input("Diese Datei scheint bereits korrigiert zu sein. Trotzdem erneut ausführen? (j/N): ").strip().lower()
                    if re_run != "j":
                        continue

                out_path, pg_cnt, pr_cnt = correct_file(selected_file)
                print(f"Erfolg! Gespeichert als: '{out_path.name}'")
                print(f"Details: {pg_cnt} PG-Zeilen und {pr_cnt} ProbeNr-Zeilen korrigiert.")
                break
            else:
                print(f"Ungültige Nummer. Bitte eine Zahl zwischen 1 und {len(csv_files)} eingeben.")
        else:
            print("Ungültige Eingabe. Bitte Zahl, 'A' oder '0' eingeben.")

    print("\nFertig.")


if __name__ == "__main__":
    main()
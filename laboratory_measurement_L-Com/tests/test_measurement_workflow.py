"""Regression checks for ring-buffer imports and measurement-file discovery."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "measurement_data"))

import messdaten_zusammenfuehren as importer
from measurement_files import discover_measurement_files, find_master_file


HEADER = "SeqNo,Date,UTC Time,Nr,ProbeNr,m_SL120,m_Wasser,m_IPA,m_PG,m_MG,Rho_M,C_M,T_M,Per_M,Per_S".split(",")
NOW = datetime(2026, 10, 3, 15, 0, 0)


def measurement(seq: int, *, date="2026-10-01", probe="304", **values):
    row = dict.fromkeys(HEADER, "0")
    row.update(SeqNo=str(seq), Date=date, Nr=str(seq), ProbeNr=probe,
               Rho_M="1008.2", C_M="1475.1", T_M="24.0", m_Wasser="9000",
               **{"UTC Time": (datetime(2026, 1, 1) + timedelta(seconds=seq)).strftime("%H:%M:%S.000")})
    row.update(values)
    return row


def write_export(folder, version, rows, *, header=HEADER, suffix=""):
    path = folder / f"Kennfeld_v2 ({version}){suffix}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
        handle.write("//END\n")
    return path


class MeasurementWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".measurement_test_")
        self.addCleanup(self.cleanup_temporary)
        self.folder = Path(self.temporary.name)

    def cleanup_temporary(self):
        # Resolve and constrain the recursively removed test directory.
        assert self.folder.resolve().is_relative_to(Path(__file__).resolve().parent)
        self.temporary.cleanup()

    def masters(self):
        return list(self.folder.glob("Messdaten_*.csv"))

    def test_initial_history_ring_wrap_update_and_repeat(self):
        base = write_export(self.folder, 25, [measurement(i) for i in range(1, 5)])
        # Physical ring-buffer order differs from chronological order.
        latest = write_export(self.folder, 32, [measurement(i) for i in (6, 7, 4, 5)])
        write_export(self.folder, 32, [measurement(99)], suffix="_korrigiert")
        originals = {path: path.read_bytes() for path in (base, latest)}
        result = importer.update_measurements(self.folder, [latest], now=NOW)
        self.assertEqual((result.total, result.added, result.duplicates), (7, 7, 1))
        self.assertEqual(result.path.name, "Messdaten_2026-10-03_15-00-00.csv")
        _, rows = importer.read_measurements(result.path)
        self.assertEqual([int(row["SeqNo"]) for row in rows], list(range(1, 8)))
        self.assertEqual(result.gaps, [])

        old_path = result.path
        old_bytes = old_path.read_bytes()
        old_mtime = old_path.stat().st_mtime_ns
        repeat = importer.update_measurements(self.folder, now=NOW + timedelta(minutes=1))
        self.assertFalse(repeat.changed)
        self.assertEqual(repeat.path, old_path)
        self.assertEqual(repeat.added, 0)
        self.assertEqual(old_path.read_bytes(), old_bytes)
        self.assertEqual(old_path.stat().st_mtime_ns, old_mtime)

        newest = write_export(self.folder, 43, [measurement(i) for i in (8, 9, 6, 7)])
        update = importer.update_measurements(self.folder, [newest], now=NOW + timedelta(minutes=2))
        self.assertEqual((update.total, update.added, update.duplicates), (9, 2, 2))
        self.assertEqual(self.masters(), [update.path])
        self.assertFalse(old_path.exists())
        for path, contents in originals.items():
            self.assertEqual(path.read_bytes(), contents)
        # The working CSV must retain history even after old exports are removed.
        base.unlink()
        latest.unlink()
        final = write_export(self.folder, 44, [measurement(10)])
        retained = importer.update_measurements(self.folder, [final], now=NOW + timedelta(minutes=3))
        self.assertEqual(retained.total, 10)

    def test_corrections_are_idempotent_and_limited_to_historical_dates(self):
        rows = [measurement(132, date="2026-08-21", probe="3"),
                measurement(200, date="2026-09-09", probe="11"),
                measurement(201, date="2026-09-16", probe="11"),
                measurement(202, probe="300"),
                measurement(132, date="2027-08-21", probe="3"),
                measurement(203, date="2027-09-16", probe="11")]
        write_export(self.folder, 25, rows)
        result = importer.update_measurements(self.folder, now=NOW)
        self.assertEqual(result.corrections, {"PG": 1, "ProbeNr": 1, "Probe300": 1})
        _, corrected = importer.read_measurements(result.path)
        by_date_seq = {(r["Date"], r["SeqNo"]): r for r in corrected}
        self.assertEqual(by_date_seq["2026-08-21", "132"]["m_PG"], "2.460000E+1")
        self.assertEqual(by_date_seq["2027-08-21", "132"]["m_PG"], "0")
        self.assertEqual(by_date_seq["2026-09-09", "200"]["ProbeNr"], "11")
        self.assertEqual(by_date_seq["2026-09-16", "201"]["ProbeNr"], "12")
        self.assertEqual(by_date_seq["2027-09-16", "203"]["ProbeNr"], "11")
        probe300 = by_date_seq["2026-10-01", "202"]
        self.assertEqual([probe300[c] for c in ("m_SL120", "m_Wasser", "m_MG")],
                         ["8.000000E+2", "8.000000E+3", "2.000000E+1"])
        for row in corrected:
            self.assertEqual(importer.correct_row(row), set())

    def test_numeric_equivalence_and_invalid_sensor_statistics(self):
        write_export(self.folder, 25, [measurement(1, Per_M="+INF", Per_S="NaN")])
        write_export(self.folder, 27, [measurement(1, Rho_M="1.008200E+3", Per_M="Infinity", Per_S="nan")])
        result = importer.update_measurements(self.folder, now=NOW)
        self.assertEqual((result.total, result.duplicates), (1, 1))
        self.assertFalse(importer.update_measurements(self.folder, now=NOW).changed)

    def test_seqno_reset_is_a_new_measurement(self):
        write_export(self.folder, 25, [measurement(1), measurement(2)])
        write_export(self.folder, 43, [measurement(1, date="2026-10-02")])
        result = importer.update_measurements(self.folder, now=NOW)
        self.assertEqual(result.total, 3)

    def test_conflict_or_truncated_export_never_changes_master(self):
        write_export(self.folder, 25, [measurement(1)])
        previous = importer.update_measurements(self.folder, now=NOW).path
        contents = previous.read_bytes()
        bad = write_export(self.folder, 27, [measurement(1, Rho_M="999"), measurement(2)])
        with self.assertRaisesRegex(ValueError, "Widersprüchliche"):
            importer.update_measurements(self.folder, [bad], now=NOW + timedelta(minutes=1))
        self.assertEqual(previous.read_bytes(), contents)
        bad.write_text(",".join(HEADER) + "\n1,2026-10-01,00:00:01.000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unvollständige"):
            importer.update_measurements(self.folder, [bad])
        self.assertEqual(previous.read_bytes(), contents)
        self.assertEqual(self.masters(), [previous])
        self.assertFalse((self.folder / ".messdaten_import.lock").exists())

    def test_mismatched_schema_never_changes_master(self):
        write_export(self.folder, 25, [measurement(1)])
        previous = importer.update_measurements(self.folder, now=NOW).path
        contents = previous.read_bytes()
        header = HEADER + ["Extra"]
        bad = write_export(self.folder, 27, [dict(measurement(2), Extra="1")], header=header)
        with self.assertRaisesRegex(ValueError, "CSV-Spalten passen nicht"):
            importer.update_measurements(self.folder, [bad])
        self.assertEqual(previous.read_bytes(), contents)

    def test_nonfinite_identifiers_are_rejected(self):
        write_export(self.folder, 25, [measurement(1, SeqNo="+INF")])
        with self.assertRaisesRegex(ValueError, "SeqNo"):
            importer.update_measurements(self.folder)
        self.assertEqual(self.masters(), [])

    def test_missing_base_or_duplicate_masters_fails(self):
        write_export(self.folder, 43, [measurement(1)])
        with self.assertRaisesRegex(FileNotFoundError, "ältesten"):
            importer.update_measurements(self.folder)
        write_export(self.folder, 25, [measurement(1)])
        previous = importer.update_measurements(self.folder, now=NOW).path
        other = self.folder / "Messdaten_2026-10-03_16-00-00.csv"
        other.write_bytes(previous.read_bytes())
        with self.assertRaisesRegex(ValueError, "Mehrere gemeinsame"):
            importer.update_measurements(self.folder)

    def test_dry_run_discovery_gaps_and_reordered_columns(self):
        base = write_export(self.folder, 25, [measurement(1)])
        write_export(self.folder, 27, [measurement(3)], header=HEADER[::-1])
        preview = importer.update_measurements(self.folder, dry_run=True, now=NOW)
        self.assertEqual(preview.gaps, [(2, 2)])
        self.assertEqual(self.masters(), [])
        result = importer.update_measurements(self.folder, now=NOW)
        self.assertEqual(discover_measurement_files(self.folder), [result.path])
        self.assertIn(base, discover_measurement_files(self.folder, prefer_master=False))
        self.assertEqual(find_master_file(self.folder), result.path)

    def test_locked_import_and_failed_write_preserve_master(self):
        write_export(self.folder, 25, [measurement(1)])
        previous = importer.update_measurements(self.folder, now=NOW).path
        write_export(self.folder, 27, [measurement(2)])
        contents = previous.read_bytes()
        lock = self.folder / ".messdaten_import.lock"
        lock.touch()
        with self.assertRaisesRegex(ValueError, "läuft bereits"):
            importer.update_measurements(self.folder)
        lock.unlink()
        with patch.object(importer.os, "replace", side_effect=PermissionError("open in Excel")):
            with self.assertRaises(PermissionError):
                importer.update_measurements(self.folder, now=NOW + timedelta(minutes=1))
        self.assertEqual(previous.read_bytes(), contents)
        self.assertEqual(list(self.folder.glob("*.tmp")), [])
        self.assertEqual(self.masters(), [previous])


if __name__ == "__main__":
    unittest.main()

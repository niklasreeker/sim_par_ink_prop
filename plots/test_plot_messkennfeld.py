"""Check field selection and parity with the calibration implementations."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import plot_messkennfeld as plot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "laboratory_measurement_L-Com"))
import residual_calibration_field as idw_reference
import residual_calibration_field_gaussian_kernel_regression as gaussian_reference


def model_payload(method):
    gaussian = method == plot.GAUSSIAN_METHOD
    interpolation = {"method": method, "scale": [1.0, 2.0, 3.0, 1.0]}
    if gaussian:
        interpolation.update(bandwidth=0.3, node_selection="all",
                             quality_weighting="fixed_inverse_squared_standard_error",
                             quality_floor_fraction=0.25, quality_absolute_floor=1e-9)
    else:
        interpolation.update(power=2.0, neighbors=4)
    return {
        "schema": "ink-residual-calibration-field", "schema_version": 4 if gaussian else 3,
        "composition_axes": ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"],
        "interpolation": interpolation,
        "nodes": [dict(Al_wt_pct=1.7 + i * .1, IPA_wt_pct=2.8 + i * .3,
                       PG_wt_pct=3.4 + i * .2, MG_wt_pct=0.0,
                       A_Rho_kg_m3=-i * .2, A_C_m_s=i * .5,
                       A_Rho_SE_kg_m3=None if i == 0 else .02 * i,
                       A_C_SE_m_s=.1 * (i + 1)) for i in range(6)],
    }


class PlotFieldTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.directory.name).resolve()
        self.assertTrue(self.root.is_relative_to(REPO))
        self.addCleanup(self.directory.cleanup)

    def save_model(self, method, name="field", repo=None, strength=None):
        root = repo or self.root
        path = root / "laboratory_measurement_L-Com" / "results" / name / "calibration_field.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model_payload(method)
        if strength is not None:
            payload["calibration_strength"] = strength
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    def test_both_methods_match_reference_at_nodes_and_between_nodes(self):
        for method, reference in [(plot.IDW_METHOD, idw_reference.idw_residual),
                                  (plot.GAUSSIAN_METHOD, gaussian_reference.gaussian_residual)]:
            path, payload = self.save_model(method, method)
            field = plot.CalibrationField(path, REPO)
            for al, ipa, pg, mg in [(1.7, 2.8, 3.4, 0.0), (1.85, 3.25, 3.7, 0.0),
                                    (2.1, 4.0, 4.2, .25), (10.0, 20.0, 30.0, 0.0)]:
                with self.subTest(method=method, al=al):
                    values, uncertainties, outside = field.evaluate(al, ipa, pg, mg)
                    expected = reference(payload, al, ipa, pg, mg=mg)
                    np.testing.assert_allclose(values, [expected["A_Rho_kg_m3"], expected["A_C_m_s"]],
                                               rtol=1e-13, atol=1e-13)
                    np.testing.assert_allclose(uncertainties,
                        [expected["A_Uncertainty_Rho_kg_m3"], expected["A_Uncertainty_C_m_s"]],
                        rtol=1e-13, atol=1e-13)
                    self.assertEqual(outside, expected["Outside_Axes"])

    def test_interactive_menu_lists_both_methods_and_retries_invalid_choices(self):
        self.save_model(plot.IDW_METHOD, "idw")
        self.save_model(plot.GAUSSIAN_METHOD, "gaussian")
        fields = plot.discover_calibration_fields(self.root)
        self.assertEqual(len(fields), 2)
        args = plot.parser().parse_args([])
        output = io.StringIO()
        with patch("builtins.input", side_effect=["0", "99", "2"]), contextlib.redirect_stdout(output):
            selected = plot.field_path(args, self.root, interactive=True)
        self.assertEqual(selected, fields[1][0])
        self.assertIn("IDW", output.getvalue())
        self.assertIn("Gaussian Kernel Regression, h=0.3", output.getvalue())

    def test_explicit_directory_and_noninteractive_default_do_not_prompt(self):
        path, _ = self.save_model(plot.GAUSSIAN_METHOD)
        args = plot.parser().parse_args(["--calibration-field", str(path.parent)])
        with patch("builtins.input", side_effect=AssertionError("Unexpected input prompt")):
            self.assertEqual(plot.field_path(args, self.root, interactive=True), path)
            default_path = self.root / plot.FIELD_RELATIVE
            default_path.parent.mkdir(parents=True, exist_ok=True)
            default_path.write_text("{}", encoding="utf-8")
            args = plot.parser().parse_args([])
            self.assertEqual(plot.field_path(args, self.root, interactive=False), default_path)

    def test_manual_path_and_no_discovered_fields(self):
        path = self.root / "calibration_field.json"
        path.write_text(json.dumps(model_payload(plot.GAUSSIAN_METHOD)), encoding="utf-8")
        args = plot.parser().parse_args([])
        with patch("builtins.input", side_effect=["", str(path.parent)]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(plot.field_path(args, self.root, interactive=True), path)

    def test_invalid_methods_and_gaussian_parameters_are_rejected(self):
        path, payload = self.save_model(plot.GAUSSIAN_METHOD)
        for change in [{"method": "unknown"}, {"bandwidth": 0}, {"node_selection": "nearest"}]:
            with self.subTest(change=change):
                invalid = json.loads(json.dumps(payload))
                invalid["interpolation"].update(change)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    plot.CalibrationField(path, REPO)

    def test_plot_workflow_saves_method_metadata_and_figures(self):
        for method in (plot.IDW_METHOD, plot.GAUSSIAN_METHOD):
            with self.subTest(method=method):
                path, _ = self.save_model(method, method, strength=0.5)
                output = self.root / (method + "_plots")
                with contextlib.redirect_stdout(io.StringIO()):
                    result = plot.main([
                        "--model", "both", "--al-min", "1.7", "--al-max", "2.1",
                        "--ipa-min", "2.8", "--ipa-max", "4.2", "--temperature", "23",
                        "--lines", "2", "--points", "3", "--no-show", "--repo-root", str(REPO),
                        "--calibration-field", str(path), "--output-dir", str(output),
                    ])
                self.assertEqual(result, 0)
                runs = list(output.iterdir())
                self.assertEqual(len(runs), 1)
                run = runs[0]
                self.assertTrue(run.is_dir())
                self.assertTrue(run.name.startswith("vergleich_"))
                self.assertIn("alpha0.5_T23C", run.name)
                self.assertIn("gaussian_h0.3" if method == plot.GAUSSIAN_METHOD else "idw_k4_p2", run.name)
                metadata = json.loads((run / "einstellungen.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["calibration_interpolation"]["method"], method)
                self.assertEqual(metadata["calibration_field"], str(path))
                self.assertNotIn("calibration_strength", metadata["settings"])
                self.assertEqual(metadata["calibration_strength"], 0.5)
                self.assertEqual(metadata["output_directory"], str(run))
                self.assertEqual(len(list(run.glob("*.png"))), 3)
                self.assertEqual(len(list(run.glob("*.svg"))), 3)
                self.assertIn("Gespeicherte Korrekturstärke α = 0.5",
                              (run / "messkennfeld_hybrid.svg").read_text(encoding="utf-8"))
                with (run / "kennfeldwerte.csv").open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    if row["Model"] == "hybrid":
                        self.assertAlmostEqual(float(row["A_Rho_kg_m3"]),
                                               0.5 * float(row["A_Rho_Field_kg_m3"]))
                        self.assertAlmostEqual(float(row["A_C_m_s"]),
                                               0.5 * float(row["A_C_Field_m_s"]))

    def test_strength_endpoints_and_half_correction_for_both_properties(self):
        calculator = Mock()
        calculator.density.return_value = 1.020
        calculator.sound_velocity.return_value = 1500.0
        field = Mock()
        field.evaluate.return_value = ([4.0, 6.0], [2.0, 3.0], [])
        for strength in (0.0, 0.5, 1.0):
            with self.subTest(strength=strength):
                field.calibration_strength = strength
                settings = plot.Settings("both", 1.7, 2.1, 2.8, 4.2, 23.0,
                                         lines=2, points=3)
                data, rows, *_ = plot.calculate_lines(calculator, field, settings)
                for model_id, correction in [("physics", 0.0), ("hybrid", strength)]:
                    for line in data[model_id]:
                        np.testing.assert_allclose(line["density"], 1020.0 + correction * 4.0)
                        np.testing.assert_allclose(line["sound"], 1500.0 + correction * 6.0)
                for row in rows:
                    if row["Model"] == "hybrid":
                        self.assertEqual(row["Calibration_Strength"], strength)
                        self.assertEqual(row["A_Rho_Uncertainty_kg_m3"], strength * 2.0)
                        self.assertEqual(row["A_C_Uncertainty_m_s"], strength * 3.0)

    def test_plot_has_no_strength_option_or_interactive_prompt(self):
        base = ["--model", "hybrid", "--al-min", "1.7", "--al-max", "2.1",
                "--ipa-min", "2.8", "--ipa-max", "4.2"]
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            settings = plot.configure(plot.parser().parse_args(base + ["--temperature", "23"]))
            self.assertFalse(hasattr(settings, "calibration_strength"))
        with patch("builtins.input", side_effect=["23"]), contextlib.redirect_stdout(io.StringIO()):
            settings = plot.configure(plot.parser().parse_args(base))
            self.assertFalse(hasattr(settings, "calibration_strength"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            plot.parser().parse_args(["--calibration-strength", "0.5"])
        path, _ = self.save_model(plot.GAUSSIAN_METHOD)
        self.assertEqual(plot.CalibrationField(path, REPO).calibration_strength, 1.0)

    def test_descriptive_directories_preserve_previous_runs(self):
        path, _ = self.save_model(plot.GAUSSIAN_METHOD, strength=0.5)
        field = plot.CalibrationField(path, REPO)
        settings = plot.Settings("hybrid", 1.7, 2.1, 2.8, 4.2, 23.0)
        base = self.root / "messkennfeld"
        first = plot.create_run_directory(base, settings, field, "20261009_120000")
        marker = first / "existing.txt"
        marker.write_text("preserved", encoding="utf-8")
        second = plot.create_run_directory(base, settings, field, "20261009_120000")
        self.assertNotEqual(first, second)
        self.assertTrue(first.name.startswith("hybrid_gaussian_h0.3_alpha0.5_T23C_"))
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
        physics = plot.Settings("physics", 1.7, 2.1, 2.8, 4.2, 23.0)
        physics_run = plot.create_run_directory(base, physics, None, "20261009_120000")
        self.assertTrue(physics_run.name.startswith("physik_T23C_"))
        self.assertNotIn("alpha", physics_run.name)

    def test_saved_strength_is_applied_exactly_once(self):
        path, payload = self.save_model(plot.GAUSSIAN_METHOD)
        payload["calibration_strength"] = 0.5
        path.write_text(json.dumps(payload), encoding="utf-8")
        field = plot.CalibrationField(path, REPO)
        settings = plot.Settings("hybrid", 1.7, 2.1, 2.8, 4.2, 23.0, lines=2, points=3)
        self.assertEqual(field.calibration_strength, 0.5)
        calculator = Mock()
        calculator.density.return_value = 1.020
        calculator.sound_velocity.return_value = 1500.0
        _, rows, *_ = plot.calculate_lines(calculator, field, settings)
        for row in rows:
            expected = gaussian_reference.gaussian_residual(payload, row["Al_wt_pct"],
                row["IPA_wt_pct"], row["PG_wt_pct"], mg=row["MG_wt_pct"])
            self.assertAlmostEqual(row["Density_kg_m3"], 1020.0 + expected["A_Rho_kg_m3"])
            self.assertAlmostEqual(row["Sound_m_s"], 1500.0 + expected["A_C_m_s"])


if __name__ == "__main__":
    unittest.main()

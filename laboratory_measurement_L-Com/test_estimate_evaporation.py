from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("estimate_evaporation.py")
SPEC = importlib.util.spec_from_file_location("estimate_evaporation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EstimateEvaporationTests(unittest.TestCase):
    def base_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Source_File": ["test.csv", "test.csv", "test.csv"],
            "Source_Row": [2, 3, 4],
            "ProbeNr": [3, 3, 3],
            "Nr": [1, 2, 3],
            "Date": ["2026-08-21"] * 3,
            "UTC Time": ["10:00:00", "11:00:00", "12:00:00"],
            "m_SL120": [10.0, 10.0, 12.0],
            "m_Wasser": [90.0, 90.0, 90.0],
            "m_IPA": [0.0, 0.0, 0.0],
            "m_PG": [0.0, 0.0, 0.0],
            "m_MG": [0.0, 0.0, 0.0],
            # 1 g representative sample, then 0.5 g evaporation per hour.
            # Before row 3, 2 g SL120 were additionally dosed.
            "m_vorher": [100.0, 98.5, 99.0],
            "m_nachher": [99.0, 97.5, 98.0],
        })

    def test_interval_balance_separates_sampling_addition_and_evaporation(self):
        result = MODULE.estimate_evaporation(self.base_frame(), MODULE.Settings())
        np.testing.assert_allclose(result["Sample_Removed_g"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(result["Mass_Added_g"], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(result["Evaporation_Step_g"], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(result["Evaporation_Cumulative_g"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            result["Evaporation_Rate_g_h"].dropna(), [0.5, 0.5]
        )

    def test_missing_mass_data_is_not_fabricated(self):
        frame = self.base_frame()
        frame[["m_vorher", "m_nachher"]] = 0.0
        result = MODULE.estimate_evaporation(frame, MODULE.Settings())
        self.assertFalse(result["Mass_Data_Available"].any())
        self.assertTrue(result["Evaporation_Step_g"].isna().all())
        self.assertEqual(set(result["Evaporation_Status"]), {"missing_mass_data"})

    def test_invalid_ipa_share_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.estimate_evaporation(
                self.base_frame(), MODULE.Settings(ipa_share=1.1)
            )


if __name__ == "__main__":
    unittest.main()

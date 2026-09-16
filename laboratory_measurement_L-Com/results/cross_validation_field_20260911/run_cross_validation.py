from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "laboratory_measurement_L-Com"
SOURCE = LAB / "measurement_data" / "Kennfeld_v2 (21)_korrigiert.csv"
MODULE_PATH = LAB / "residual_calibration_field.py"
OUT = Path(__file__).resolve().parent
PROBES = tuple(range(3, 12))


def load_module():
    spec = importlib.util.spec_from_file_location("residual_calibration_field_cv", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metric(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(x):
        return {k: np.nan for k in ("bias", "mae", "rmse", "p50_abs", "p90_abs", "p95_abs", "max_abs")}
    absolute = np.abs(x)
    return {
        "bias": float(np.mean(x)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(x**2))),
        "p50_abs": float(np.quantile(absolute, 0.50)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def make_model(module, nodes: pd.DataFrame) -> dict:
    centre, scale = module.node_scaler(nodes)
    return {
        "schema": "ink-residual-calibration-field",
        "schema_version": 3,
        "composition_axes": module.FIELD_AXES,
        "interpolation": {
            "method": "scaled_inverse_distance_weighting",
            "power": 2.0,
            "neighbors": 4,
            "centre": centre.tolist(),
            "scale": scale.tolist(),
        },
        "nodes": nodes.replace({np.nan: None}).to_dict(orient="records"),
    }


def run_scenario(module, simulated: pd.DataFrame, scenario: str, kind: str,
                 train_probes: tuple[int, ...], test_probes: tuple[int, ...]):
    train = simulated.loc[simulated["ProbeNr"].astype(int).isin(train_probes)].copy()
    test = simulated.loc[simulated["ProbeNr"].astype(int).isin(test_probes)].copy()
    nodes = module.build_nodes(train)
    model = make_model(module, nodes)

    predictions = []
    for index, row in test.iterrows():
        if row["Simulation_Status"] != "ok":
            continue
        correction = module.idw_residual(
            model,
            float(row["Al_wt_pct_eff"]),
            float(row["IPA_wt_pct_eff"]),
            float(row["PG_wt_pct_eff"]),
            float(row["MG_wt_pct_eff"]),
        )
        predictions.append({
            "index": index,
            "A_Rho_Field_kg_m3": correction["A_Rho_kg_m3"],
            "A_C_Field_m_s": correction["A_C_m_s"],
            "A_Rho_Uncertainty_kg_m3": correction["A_Uncertainty_Rho_kg_m3"],
            "A_C_Uncertainty_m_s": correction["A_Uncertainty_C_m_s"],
            "Calibration_Distance": correction["Nearest_Normalized_Distance"],
            "Calibration_Extrapolation": correction["Outside_Bounding_Box"],
            "Calibration_Outside_Axes": ",".join(correction["Outside_Axes"]),
        })
    predicted = pd.DataFrame(predictions).set_index("index")
    test = test.join(predicted)
    test["Rho_Hybrid_kg_m3"] = test["Rho_Physics_kg_m3"] + test["A_Rho_Field_kg_m3"]
    test["C_Hybrid_m_s"] = test["C_Physics_m_s"] + test["A_C_Field_m_s"]
    required = ["Rho_M", "C_M", "Rho_Physics_kg_m3", "C_Physics_m_s",
                "Rho_Hybrid_kg_m3", "C_Hybrid_m_s"]
    test["Selected_For_Evaluation"] = (
        test["Selected_For_Calibration"]
        & test["Simulation_Status"].eq("ok")
        & np.isfinite(test[required]).all(axis=1)
    )
    test["Training_Overlap"] = False
    test["Scenario"] = scenario
    test["Scenario_Kind"] = kind
    test["Train_Probes"] = ",".join(map(str, train_probes))
    test["Test_Probes"] = ",".join(map(str, test_probes))
    test["Rho_Error_Hybrid"] = test["Rho_Hybrid_kg_m3"] - test["Rho_M"]
    test["C_Error_Hybrid"] = test["C_Hybrid_m_s"] - test["C_M"]
    test["Rho_Error_Physics"] = test["Rho_Physics_kg_m3"] - test["Rho_M"]
    test["C_Error_Physics"] = test["C_Physics_m_s"] - test["C_M"]

    phases = module.phase_evaluation_summary(test)
    phases.insert(0, "Scenario", scenario)
    phases.insert(1, "Scenario_Kind", kind)
    phases.insert(2, "Train_Probes", ",".join(map(str, train_probes)))
    phases.insert(3, "Test_Probes", ",".join(map(str, test_probes)))

    selected = test.loc[test["Selected_For_Evaluation"]].copy()
    usable_phases = phases.loc[phases["N_Selected"].gt(0)].copy()
    record = {
        "Scenario": scenario,
        "Scenario_Kind": kind,
        "Train_Probes": ",".join(map(str, train_probes)),
        "Test_Probes": ",".join(map(str, test_probes)),
        "N_Train_Probes": len(train_probes),
        "N_Test_Probes": len(test_probes),
        "N_Nodes": len(nodes),
        "N_Test_Rows": len(selected),
        "N_Test_Phases": len(usable_phases),
        "Extrapolated_Row_Fraction": float(selected["Calibration_Extrapolation"].mean()),
        "Mean_Calibration_Distance": float(selected["Calibration_Distance"].mean()),
        "Max_Calibration_Distance": float(selected["Calibration_Distance"].max()),
    }
    for prefix in ("Rho", "C"):
        row_hybrid = metric(selected[f"{prefix}_Error_Hybrid"])
        row_physics = metric(selected[f"{prefix}_Error_Physics"])
        phase_hybrid = metric(usable_phases[f"{prefix}_Mean_Error"])
        phase_physics = metric(
            usable_phases[f"{prefix}_Mean_Physics"] - usable_phases[f"{prefix}_Mean_Measured"]
        )
        for key, value in row_hybrid.items():
            record[f"{prefix}_Row_{key}"] = value
        record[f"{prefix}_Row_Physics_RMSE"] = row_physics["rmse"]
        record[f"{prefix}_Row_RMSE_Improvement_pct"] = (
            100.0 * (1.0 - row_hybrid["rmse"] / row_physics["rmse"])
            if row_physics["rmse"] > 0 else np.nan
        )
        for key, value in phase_hybrid.items():
            record[f"{prefix}_Phase_{key}"] = value
        record[f"{prefix}_Phase_Physics_RMSE"] = phase_physics["rmse"]
        record[f"{prefix}_Phase_RMSE_Improvement_pct"] = (
            100.0 * (1.0 - phase_hybrid["rmse"] / phase_physics["rmse"])
            if phase_physics["rmse"] > 0 else np.nan
        )
    return record, test, phases, nodes


def main():
    module = load_module()
    raw = module.filter_samples(module.load_measurements([SOURCE]), [str(p) for p in PROBES])
    prepared = module.add_nominal_component_masses(
        module.add_timestamps_and_phases(raw, "Date", "UTC Time", 60.0)
    )
    prepared = module.select_calibration_rows(prepared, "auto", 5, 0.30, 0.60)
    protocols, _ = module.load_evaporation_protocol(None)
    corrected, evaporation_summaries = module.apply_evaporation(prepared, protocols)
    calculator = module.load_calculator(ROOT / "ink_calculator.py", ROOT / "tables_parameters")
    simulated = module.simulate_rows(corrected, calculator)

    scenarios = []
    for held_out in PROBES:
        train = tuple(p for p in PROBES if p != held_out)
        scenarios.append((f"LOPO_test_{held_out}", "Leave-one-probe-out", train, (held_out,)))
    scenarios.extend([
        ("Current_8_9_10", "Structured", (8, 9, 10), (3, 4, 5, 6, 7, 11)),
        ("Recent_8_9_10_11", "Structured", (8, 9, 10, 11), (3, 4, 5, 6, 7)),
        ("Early_3_4_5_6_7", "Structured", (3, 4, 5, 6, 7), (8, 9, 10, 11)),
        ("Composition_rich", "Structured", (3, 4, 8, 9, 10, 11), (5, 6, 7)),
        ("Alternating_A", "Structured", (3, 5, 7, 9, 11), (4, 6, 8, 10)),
        ("Alternating_B", "Structured", (4, 6, 8, 10), (3, 5, 7, 9, 11)),
    ])
    for train_probe in PROBES:
        for test_probe in PROBES:
            if train_probe != test_probe:
                scenarios.append((f"Pair_{train_probe}_to_{test_probe}", "Pairwise",
                                  (train_probe,), (test_probe,)))

    summaries, measurement_frames, phase_frames, node_frames = [], [], [], [],
    for number, (name, kind, train, test) in enumerate(scenarios, start=1):
        print(f"[{number:02d}/{len(scenarios)}] {name}: train={train}, test={test}")
        summary, measurements, phases, nodes = run_scenario(
            module, simulated, name, kind, train, test
        )
        summaries.append(summary)
        if kind != "Pairwise":
            measurement_frames.append(measurements.loc[measurements["Selected_For_Evaluation"]])
            phase_frames.append(phases.loc[phases["N_Selected"].gt(0)])
        nodes = nodes.copy()
        nodes.insert(0, "Scenario", name)
        nodes.insert(1, "Scenario_Kind", kind)
        if kind != "Pairwise":
            node_frames.append(nodes)

    summary = pd.DataFrame(summaries)
    measurements = pd.concat(measurement_frames, ignore_index=True)
    phases = pd.concat(phase_frames, ignore_index=True)
    nodes = pd.concat(node_frames, ignore_index=True)
    evaporation = pd.DataFrame([item.__dict__ for item in evaporation_summaries])

    summary.to_csv(OUT / "scenario_summary.csv", index=False)
    measurements.to_csv(OUT / "measurement_predictions.csv", index=False)
    phases.to_csv(OUT / "phase_predictions.csv", index=False)
    nodes.to_csv(OUT / "calibration_nodes_by_scenario.csv", index=False)
    evaporation.to_csv(OUT / "evaporation_summary.csv", index=False)

    pairwise = summary.loc[summary["Scenario_Kind"].eq("Pairwise")].copy()
    for prefix in ("Rho", "C"):
        matrix = pairwise.pivot(index="Test_Probes", columns="Train_Probes",
                                values=f"{prefix}_Phase_rmse")
        matrix.to_csv(OUT / f"pairwise_{prefix.lower()}_phase_rmse_matrix.csv")

    lopo_measurements = measurements.loc[measurements["Scenario_Kind"].eq("Leave-one-probe-out")]
    lopo_phases = phases.loc[phases["Scenario_Kind"].eq("Leave-one-probe-out")]
    empirical = {}
    for prefix, unit in (("Rho", "kg/m3"), ("C", "m/s")):
        row_stats = metric(lopo_measurements[f"{prefix}_Error_Hybrid"])
        phase_stats = metric(lopo_phases[f"{prefix}_Mean_Error"])
        empirical[prefix] = {"unit": unit, "measurement_rows": row_stats, "phase_means": phase_stats}
    payload = {
        "method": "Leave-one-probe-out is the primary independent validation; structured and pairwise fields are sensitivity analyses.",
        "probes": list(PROBES),
        "source": str(SOURCE),
        "quality_selection": "auto",
        "idw_power": 2.0,
        "idw_neighbors": 4,
        "evaporation": "Built-in probe/date protocol, applied row by row before simulation and residual interpolation.",
        "empirical_error": empirical,
        "n_scenarios": len(scenarios),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    workbook_measurement_columns = [
        "Scenario", "ProbeNr", "Phase", "Measurement_Time_UTC", "Selection_Method",
        "Al_wt_pct_eff", "IPA_wt_pct_eff", "PG_wt_pct_eff", "MG_wt_pct_eff",
        "T_M", "Total_Evaporation_Loss_g", "Rho_M", "Rho_Physics_kg_m3",
        "Rho_Hybrid_kg_m3", "Rho_Error_Hybrid", "C_M", "C_Physics_m_s",
        "C_Hybrid_m_s", "C_Error_Hybrid", "Calibration_Distance",
        "Calibration_Extrapolation",
    ]
    workbook_phase_columns = [
        "Scenario", "ProbeNr", "Phase_ID", "N_Selected", "N_Extrapolated",
        "Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct", "Temperature_Mean_C",
        "Total_Evaporation_Loss_Mean_g", "Rho_Mean_Measured", "Rho_Mean_Physics",
        "Rho_Mean_Hybrid", "Rho_Mean_Error", "C_Mean_Measured", "C_Mean_Physics",
        "C_Mean_Hybrid", "C_Mean_Error",
    ]

    def records(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    workbook_data = {
        "analysis": payload,
        "scenario_summary": records(summary),
        "lopo_measurements": records(
            lopo_measurements[workbook_measurement_columns]
        ),
        "lopo_phases": records(lopo_phases[workbook_phase_columns]),
        "structured_phases": records(
            phases.loc[phases["Scenario_Kind"].eq("Structured"), workbook_phase_columns]
        ),
        "rho_pairwise_matrix": {
            "probes": list(PROBES),
            "values": pairwise.pivot(index="Test_Probes", columns="Train_Probes",
                                     values="Rho_Phase_rmse").reindex(
                                         index=[str(p) for p in PROBES],
                                         columns=[str(p) for p in PROBES]
                                     ).astype(object).where(lambda x: pd.notna(x), None).values.tolist(),
        },
        "c_pairwise_matrix": {
            "probes": list(PROBES),
            "values": pairwise.pivot(index="Test_Probes", columns="Train_Probes",
                                     values="C_Phase_rmse").reindex(
                                         index=[str(p) for p in PROBES],
                                         columns=[str(p) for p in PROBES]
                                     ).astype(object).where(lambda x: pd.notna(x), None).values.tolist(),
        },
    }
    (OUT / "workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outDir = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(outDir, "workbook_data.json"), "utf8"));
const workbook = Workbook.create();
const font = "Arial";
const navy = "#173D61";
const blue = "#2E75B6";
const lightBlue = "#DCE6F1";
const paleBlue = "#EEF4F8";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const green = "#E2F0D9";
const gray = "#666666";

function applyBase(sheet) {
  sheet.showGridLines = false;
  const used = sheet.getUsedRange();
  if (used) {
    used.format.font = { name: font, size: 10, color: "#1F2933" };
    used.format.verticalAlignment = "center";
  }
}

function title(sheet, text, endCol = "J") {
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format.font = { name: font, size: 15, bold: true, color: navy };
  sheet.getRange(`A3:${endCol}3`).format.borders = {
    bottom: { style: "thin", color: blue },
  };
}

function section(sheet, address, text, endCol) {
  const row = address.match(/\d+/)[0];
  sheet.getRange(address).values = [[text]];
  const band = sheet.getRange(`${address.split(/\d/)[0]}${row}:${endCol}${row}`);
  band.format.fill = lightBlue;
  band.format.font = { name: font, size: 10, bold: true, color: navy };
  band.format.borders = { preset: "outside", style: "thin", color: "#9FBAD0" };
}

function header(range) {
  range.format.fill = navy;
  range.format.font = { name: font, size: 10, bold: true, color: "#FFFFFF" };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.borders = {
    insideVertical: { style: "thin", color: "#FFFFFF" },
    bottom: { style: "thin", color: navy },
  };
}

function addTable(sheet, startRow, headers, rows, name) {
  const endCol = columnName(headers.length);
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).values = [headers];
  if (rows.length) sheet.getRange(`A${startRow + 1}`).write(rows);
  const table = sheet.tables.add(`A${startRow}:${endCol}${startRow + rows.length}`, true, name);
  table.style = "TableStyleMedium2";
  table.showBandedColumns = false;
  table.showFilterButton = true;
  header(sheet.getRange(`A${startRow}:${endCol}${startRow}`));
  return { endCol, endRow: startRow + rows.length };
}

function columnName(count) {
  let n = count;
  let s = "";
  while (n > 0) {
    n--;
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26);
  }
  return s;
}

function recordsRows(records, keys) {
  return records.map(record => keys.map(key => record[key] ?? null));
}

function stat(values, key) {
  const x = values.map(v => Number(v[key])).filter(Number.isFinite);
  const bias = x.reduce((a, b) => a + b, 0) / x.length;
  const mae = x.reduce((a, b) => a + Math.abs(b), 0) / x.length;
  const rmse = Math.sqrt(x.reduce((a, b) => a + b * b, 0) / x.length);
  return { bias, mae, rmse };
}

const overview = workbook.worksheets.add("Übersicht");
applyBase(overview);
title(overview, "Cross-Validation des Residual-Kalibrierfelds", "T");
overview.getRange("A4").values = [["Proben 3–11, verdunstungskorrigiert, automatische Qualitätsauswahl"]];
overview.getRange("A4").format.font = { name: font, size: 10, italic: true, color: gray };

section(overview, "A6", "Erwartbarer Fehler einer neuen, unabhängigen Probe", "I");
overview.getRange("A7:I7").values = [[
  "Messgröße", "Ebene", "Bias", "MAE", "RMSE", "90 % |Fehler| ≤", "95 % |Fehler| ≤", "Maximum", "Einheit"
]];
header(overview.getRange("A7:I7"));
const emp = data.analysis.empirical_error;
overview.getRange("A8").write([
  ["Dichte", "Einzelmesspunkte", emp.Rho.measurement_rows.bias, emp.Rho.measurement_rows.mae, emp.Rho.measurement_rows.rmse, emp.Rho.measurement_rows.p90_abs, emp.Rho.measurement_rows.p95_abs, emp.Rho.measurement_rows.max_abs, "kg/m³"],
  ["Dichte", "Phasenmittel", emp.Rho.phase_means.bias, emp.Rho.phase_means.mae, emp.Rho.phase_means.rmse, emp.Rho.phase_means.p90_abs, emp.Rho.phase_means.p95_abs, emp.Rho.phase_means.max_abs, "kg/m³"],
  ["Schallgeschwindigkeit", "Einzelmesspunkte", emp.C.measurement_rows.bias, emp.C.measurement_rows.mae, emp.C.measurement_rows.rmse, emp.C.measurement_rows.p90_abs, emp.C.measurement_rows.p95_abs, emp.C.measurement_rows.max_abs, "m/s"],
  ["Schallgeschwindigkeit", "Phasenmittel", emp.C.phase_means.bias, emp.C.phase_means.mae, emp.C.phase_means.rmse, emp.C.phase_means.p90_abs, emp.C.phase_means.p95_abs, emp.C.phase_means.max_abs, "m/s"],
]);
overview.getRange("C8:H11").format.numberFormat = "0.000";

const summaries = data.scenario_summary;
const current = summaries.find(r => r.Scenario === "Current_8_9_10");
const recent = summaries.find(r => r.Scenario === "Recent_8_9_10_11");
const current11Phases = data.structured_phases.filter(r => r.Scenario === "Current_8_9_10" && Number(r.ProbeNr) === 11);
const current11Rho = stat(current11Phases, "Rho_Mean_Error");
const current11C = stat(current11Phases, "C_Mean_Error");
const current11Ext = current11Phases.reduce((a, r) => a + Number(r.N_Extrapolated), 0) /
  current11Phases.reduce((a, r) => a + Number(r.N_Selected), 0);

section(overview, "A13", "Vergleich wichtiger Feldkonfigurationen (Phasenmittel)", "H");
overview.getRange("A14:H14").values = [[
  "Konfiguration", "Kalibrierproben", "Testproben", "Dichte-RMSE", "Schall-RMSE", "Extrapolierte Zeilen", "Knoten", "Einordnung"
]];
header(overview.getRange("A14:H14"));
overview.getRange("A15").write([
  ["LOPO-Schätzung für Vollfeld", "je 8 von 9", "je 1 ausgelassen", emp.Rho.phase_means.rmse, emp.C.phase_means.rmse, null, null, "Beste Schätzung für neue Probe innerhalb ähnlicher Zusammensetzungen"],
  ["Aktuelles Feld", current.Train_Probes, current.Test_Probes, current.Rho_Phase_rmse, current.C_Phase_rmse, current.Extrapolated_Row_Fraction, current.N_Nodes, "Nur drei Proben; schwache Abdeckung"],
  ["Aktuelles Feld → Probe 11", "8,9,10", "11", current11Rho.rmse, current11C.rmse, current11Ext, 23, "Probe 11 liegt überwiegend außerhalb der Feldgrenzen"],
  ["Erweitertes aktuelles Feld", recent.Train_Probes, recent.Test_Probes, recent.Rho_Phase_rmse, recent.C_Phase_rmse, recent.Extrapolated_Row_Fraction, recent.N_Nodes, "Probe 11 verbessert besonders die Schallübertragung"],
]);
overview.getRange("D15:E18").format.numberFormat = "0.000";
overview.getRange("F15:F18").format.numberFormat = "0.0%";

section(overview, "A20", "Schlussfolgerungen", "I");
overview.getRange("A21:I25").values = [
  ["1", "Das vollständige Feld aus Proben 3–11 sollte für künftige Messungen verwendet werden.", null, null, null, null, null, null, null],
  ["2", "Als empirische 95-%-Grenze sind derzeit etwa ±0,85 kg/m³ und ±2,04 m/s für Phasenmittel anzusetzen.", null, null, null, null, null, null, null],
  ["3", "Das Feld 8/9/10 extrapoliert 67 % seiner externen Testzeilen; für Probe 11 sind es 88 %.", null, null, null, null, null, null, null],
  ["4", "Die interne IDW-Knotenunsicherheit ist nicht als Vorhersageintervall kalibriert; die empirischen LOPO-Grenzen sind belastbarer.", null, null, null, null, null, null, null],
  ["5", "Neue Proben außerhalb der untersuchten Zusammensetzungs- und Prozessbereiche benötigen zusätzliche Validierung.", null, null, null, null, null, null, null],
];
overview.getRange("A21:A25").format.font = { name: font, bold: true, color: navy };
overview.getRange("B21:I25").format.wrapText = true;
overview.getRange("A21:I25").format.fill = paleBlue;
overview.getRange("A21:I25").format.borders = { preset: "outside", style: "thin", color: "#B8CAD8" };
for (let row = 21; row <= 25; row++) overview.mergeCells(`B${row}:I${row}`);
overview.getRange("A1:T26").format.rowHeight = 19;
overview.getRange("A2").format.rowHeight = 28;
overview.getRange("A6:I6").format.rowHeight = 22;
overview.getRange("A13:H13").format.rowHeight = 22;
overview.getRange("A20:I20").format.rowHeight = 22;
overview.getRange("A21:I25").format.rowHeight = 32;
overview.getRange("A:A").format.columnWidth = 20;
overview.getRange("B:C").format.columnWidth = 18;
overview.getRange("D:G").format.columnWidth = 15;
overview.getRange("H:H").format.columnWidth = 52;
overview.getRange("I:I").format.columnWidth = 16;
overview.getRange("A15:H15").format.fill = green;
overview.getRange("A16:H17").format.fill = red;

const lopoSheet = workbook.worksheets.add("LOPO je Probe");
applyBase(lopoSheet);
title(lopoSheet, "Leave-one-probe-out nach Testprobe", "V");
const lopo = summaries.filter(r => r.Scenario_Kind === "Leave-one-probe-out");
const lopoKeys = ["Test_Probes", "N_Test_Phases", "N_Test_Rows", "Extrapolated_Row_Fraction",
  "Rho_Phase_bias", "Rho_Phase_mae", "Rho_Phase_rmse", "Rho_Phase_Physics_RMSE", "Rho_Phase_RMSE_Improvement_pct",
  "C_Phase_bias", "C_Phase_mae", "C_Phase_rmse", "C_Phase_Physics_RMSE", "C_Phase_RMSE_Improvement_pct"];
const lopoHeaders = ["Testprobe", "Phasen", "Messpunkte", "Extrapolation", "Dichte Bias", "Dichte MAE", "Dichte RMSE", "Dichte Physik RMSE", "Dichte Verbesserung", "Schall Bias", "Schall MAE", "Schall RMSE", "Schall Physik RMSE", "Schall Verbesserung"];
const lopoRows = recordsRows(lopo, lopoKeys);
for (const row of lopoRows) { row[8] /= 100; row[13] /= 100; }
addTable(lopoSheet, 5, lopoHeaders, lopoRows, "LopoTable");
lopoSheet.getRange("D6:D14").format.numberFormat = "0.0%";
lopoSheet.getRange("E6:H14").format.numberFormat = "0.000";
lopoSheet.getRange("I6:I14").format.numberFormat = "0.0%";
lopoSheet.getRange("J6:M14").format.numberFormat = "0.000";
lopoSheet.getRange("N6:N14").format.numberFormat = "0.0%";
lopoSheet.freezePanes.freezeRows(5);
lopoSheet.getRange("A:N").format.columnWidth = 15;
lopoSheet.getRange("A:A").format.columnWidth = 11;
const rhoChart = lopoSheet.charts.add("bar", [lopoSheet.getRange("A5:A14"), lopoSheet.getRange("G5:G14")]);
rhoChart.title = "Dichte-RMSE je ausgelassener Probe (kg/m³)";
rhoChart.titleTextStyle.typeface = font;
rhoChart.titleTextStyle.fontSize = 12;
rhoChart.hasLegend = false;
rhoChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
rhoChart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: font } };
rhoChart.setPosition("P5", "V15");
const cChart = lopoSheet.charts.add("bar", [lopoSheet.getRange("A5:A14"), lopoSheet.getRange("L5:L14")]);
cChart.title = "Schall-RMSE je ausgelassener Probe (m/s)";
cChart.titleTextStyle.typeface = font;
cChart.titleTextStyle.fontSize = 12;
cChart.hasLegend = false;
cChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 9 } };
cChart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: font } };
cChart.setPosition("P17", "V27");

const comboSheet = workbook.worksheets.add("Kombinationen");
applyBase(comboSheet);
title(comboSheet, "Vergleich der Mehrproben-Kalibrierfelder", "R");
const combo = summaries.filter(r => r.Scenario_Kind !== "Pairwise");
const comboKeys = ["Scenario", "Train_Probes", "Test_Probes", "N_Nodes", "N_Test_Phases", "N_Test_Rows",
  "Extrapolated_Row_Fraction", "Mean_Calibration_Distance", "Rho_Phase_bias", "Rho_Phase_mae", "Rho_Phase_rmse",
  "Rho_Phase_Physics_RMSE", "Rho_Phase_RMSE_Improvement_pct", "C_Phase_bias", "C_Phase_mae", "C_Phase_rmse",
  "C_Phase_Physics_RMSE", "C_Phase_RMSE_Improvement_pct"];
const comboHeaders = ["Szenario", "Kalibrierproben", "Testproben", "Knoten", "Phasen", "Messpunkte", "Extrapolation",
  "mittlere Distanz", "Dichte Bias", "Dichte MAE", "Dichte RMSE", "Dichte Physik RMSE", "Dichte Verbesserung",
  "Schall Bias", "Schall MAE", "Schall RMSE", "Schall Physik RMSE", "Schall Verbesserung"];
const comboRows = recordsRows(combo, comboKeys);
for (const row of comboRows) { row[12] /= 100; row[17] /= 100; }
addTable(comboSheet, 5, comboHeaders, comboRows, "CombinationTable");
comboSheet.getRange(`G6:G${5 + combo.length}`).format.numberFormat = "0.0%";
comboSheet.getRange(`H6:L${5 + combo.length}`).format.numberFormat = "0.000";
comboSheet.getRange(`M6:M${5 + combo.length}`).format.numberFormat = "0.0%";
comboSheet.getRange(`N6:Q${5 + combo.length}`).format.numberFormat = "0.000";
comboSheet.getRange(`R6:R${5 + combo.length}`).format.numberFormat = "0.0%";
comboSheet.freezePanes.freezeRows(5);
comboSheet.freezePanes.freezeColumns(1);
comboSheet.getRange("A:R").format.columnWidth = 15;
comboSheet.getRange("A:A").format.columnWidth = 22;
comboSheet.getRange("B:C").format.columnWidth = 23;

function makeMatrixSheet(name, titleText, matrix, tableName, unit) {
  const sheet = workbook.worksheets.add(name);
  applyBase(sheet);
  title(sheet, titleText, "K");
  sheet.getRange("A4").values = [["Zeilen = Testprobe, Spalten = einzelne Kalibrierprobe; Werte = Phasenmittel-RMSE"]];
  sheet.getRange("A4").format.font = { name: font, italic: true, color: gray };
  const probes = matrix.probes;
  const rows = probes.map((probe, i) => [probe, ...matrix.values[i]]);
  addTable(sheet, 6, ["Test \\ Training", ...probes.map(String)], rows, tableName);
  sheet.getRange("B7:J15").format.numberFormat = "0.000";
  sheet.getRange("B7:J15").conditionalFormats.add("colorScale", {
    colors: ["#63BE7B", "#FFEB84", "#F8696B"], thresholds: ["min", { type: "percentile", value: 50 }, "max"]
  });
  sheet.mergeCells("A17:J18");
  sheet.getRange("A17").values = [[`Einheit: ${unit}. Niedrige Werte zeigen ähnliche Residualkorrekturen; hohe Werte oder Extrapolation begrenzen die Übertragbarkeit.`]];
  sheet.getRange("A17:J18").format.wrapText = true;
  sheet.getRange("A:K").format.columnWidth = 14;
  sheet.getRange("A:A").format.columnWidth = 18;
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(1);
  return sheet;
}
makeMatrixSheet("Paarweise Dichte", "Paarweise Übertragbarkeit der Dichte", data.rho_pairwise_matrix, "PairRhoTable", "kg/m³");
makeMatrixSheet("Paarweise Schall", "Paarweise Übertragbarkeit der Schallgeschwindigkeit", data.c_pairwise_matrix, "PairCTable", "m/s");

const phaseSheet = workbook.worksheets.add("Phasen LOPO");
applyBase(phaseSheet);
title(phaseSheet, "Phasenmittel der unabhängigen LOPO-Tests", "S");
const phaseKeys = ["ProbeNr", "Phase_ID", "N_Selected", "N_Extrapolated", "Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct",
  "Temperature_Mean_C", "Total_Evaporation_Loss_Mean_g", "Rho_Mean_Measured", "Rho_Mean_Physics", "Rho_Mean_Hybrid", "Rho_Mean_Error",
  "C_Mean_Measured", "C_Mean_Physics", "C_Mean_Hybrid", "C_Mean_Error"];
const phaseHeaders = ["Probe", "Phase", "n gewählt", "n extrapoliert", "Al wt-%", "IPA wt-%", "PG wt-%", "MG wt-%", "Temperatur °C",
  "Verdunstung g", "Dichte gemessen", "Dichte Physik", "Dichte Hybrid", "Dichte Fehler", "Schall gemessen", "Schall Physik", "Schall Hybrid", "Schall Fehler"];
addTable(phaseSheet, 5, phaseHeaders, recordsRows(data.lopo_phases.filter(r => r.Scenario.startsWith("LOPO_test_")), phaseKeys), "PhaseTable");
const phaseEnd = 5 + data.lopo_phases.filter(r => r.Scenario.startsWith("LOPO_test_")).length;
phaseSheet.getRange(`E6:R${phaseEnd}`).format.numberFormat = "0.000";
phaseSheet.getRange("A:R").format.columnWidth = 14;
phaseSheet.getRange("B:B").format.columnWidth = 13;
phaseSheet.freezePanes.freezeRows(5);
phaseSheet.freezePanes.freezeColumns(2);

const pointSheet = workbook.worksheets.add("Messpunkte LOPO");
applyBase(pointSheet);
title(pointSheet, "Ausgewählte Messpunkte der unabhängigen LOPO-Tests", "U");
const pointKeys = ["ProbeNr", "Phase", "Measurement_Time_UTC", "Selection_Method", "Al_wt_pct_eff", "IPA_wt_pct_eff", "PG_wt_pct_eff", "MG_wt_pct_eff", "T_M",
  "Total_Evaporation_Loss_g", "Rho_M", "Rho_Physics_kg_m3", "Rho_Hybrid_kg_m3", "Rho_Error_Hybrid", "C_M", "C_Physics_m_s", "C_Hybrid_m_s", "C_Error_Hybrid",
  "Calibration_Distance", "Calibration_Extrapolation"];
const pointHeaders = ["Probe", "Phase", "Zeit UTC", "Auswahl", "Al wt-%", "IPA wt-%", "PG wt-%", "MG wt-%", "Temperatur °C", "Verdunstung g",
  "Dichte gemessen", "Dichte Physik", "Dichte Hybrid", "Dichte Fehler", "Schall gemessen", "Schall Physik", "Schall Hybrid", "Schall Fehler", "Distanz", "Extrapolation"];
const lopoPoints = data.lopo_measurements.filter(r => r.Scenario.startsWith("LOPO_test_"));
addTable(pointSheet, 5, pointHeaders, recordsRows(lopoPoints, pointKeys), "PointTable");
const pointEnd = 5 + lopoPoints.length;
pointSheet.getRange(`E6:S${pointEnd}`).format.numberFormat = "0.000";
pointSheet.getRange("A:T").format.columnWidth = 14;
pointSheet.getRange("C:C").format.columnWidth = 25;
pointSheet.getRange("D:D").format.columnWidth = 29;
pointSheet.freezePanes.freezeRows(5);
pointSheet.freezePanes.freezeColumns(2);

const methodSheet = workbook.worksheets.add("Methodik");
applyBase(methodSheet);
title(methodSheet, "Methodik und Grenzen", "J");
section(methodSheet, "A5", "Auswertungsdesign", "J");
methodSheet.getRange("A6:B13").values = [
  ["Primäre Validierung", "Leave-one-probe-out: Eine komplette Probe wird ausgeschlossen, das Feld aus den übrigen acht aufgebaut und nur auf der ausgelassenen Probe getestet."],
  ["Weitere Kombinationen", "Aktuelles Feld 8/9/10, erweitertes Feld 8/9/10/11, frühe Proben, zusammensetzungsreiches Feld und zwei alternierende Teilungen."],
  ["Paarvergleich", "72 gerichtete Felder aus jeweils einer Trainingsprobe und einer anderen Testprobe. Diese zeigen Übertragbarkeit, sind aber wegen geringer Feldabdeckung keine bevorzugte Prognose."],
  ["Qualitätsauswahl", "Automatisch; identisch zur Standardeinstellung des Skripts."],
  ["Verdunstung", "Für jede Probe und jeden Messzeitpunkt aus dem eingebauten Protokoll integriert. Gesamtverlust: 66 % IPA und 34 % Wasser."],
  ["Interpolation", "Skalierte inverse Distanzgewichtung, Potenz 2, vier Nachbarknoten."],
  ["Fehlervorzeichen", "Hybridvorhersage minus Messwert."],
  ["Hauptgewichtung", "Jede Rezeptphase erhält gleiches Gewicht; lange Aufzeichnungen dominieren die Hauptkennzahl nicht."],
];
section(methodSheet, "A15", "Interpretation und Grenzen", "J");
methodSheet.getRange("A16:B22").values = [
  ["95-%-Grenze", "Empirisches 95-%-Quantil der absoluten LOPO-Fehler. Es ist keine formale Garantie und basiert nur auf neun Proben."],
  ["Extrapolation", "Bounding-Box-Prüfung der Zusammensetzungsachsen. Innerhalb der Box kann die lokale Knotendichte trotzdem gering sein."],
  ["Interne Unsicherheit", "Die im IDW-Feld berechnete Knotenunsicherheit deckt in LOPO nur etwa 26 % der Dichte- und 53 % der Schallfehler ab. Sie ist daher kein kalibriertes Vorhersageintervall."],
  ["Proben 5–7", "Nur wenige Rezeptphasen; ihre Einzelkennzahlen sind statistisch weniger stabil."],
  ["Prozessübertragbarkeit", "Die Aussage gilt für ähnliche Geräte-, Temperatur-, Rühr- und Verdunstungsbedingungen wie in den vorhandenen Messungen."],
  ["Empfehlung", "Vollfeld aus 3–11 aufbauen und jede neue unabhängige Probe anschließend als weitere externe Validierung dokumentieren."],
  ["Quelldatei", data.analysis.source],
];
methodSheet.getRange("A6:A22").format.font = { name: font, bold: true, color: navy };
methodSheet.getRange("B6:B22").format.wrapText = true;
methodSheet.getRange("A:A").format.columnWidth = 24;
methodSheet.getRange("B:B").format.columnWidth = 92;
methodSheet.getRange("A6:B22").format.rowHeight = 34;
methodSheet.getRange("A5:J5").format.rowHeight = 22;
methodSheet.getRange("A15:J15").format.rowHeight = 22;
methodSheet.tabColor = "#7F8C8D";

overview.tabColor = navy;
lopoSheet.tabColor = blue;
comboSheet.tabColor = blue;

workbook.recalculate();
await fs.mkdir(path.join(outDir, "previews"), { recursive: true });
for (const sheet of workbook.worksheets.items) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 1, format: "png" });
  const safe = sheet.name.replace(/[^A-Za-z0-9_-]/g, "_");
  await fs.writeFile(path.join(outDir, "previews", `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const overviewCheck = await workbook.inspect({ kind: "table", sheetId: "Übersicht", range: "A6:I25", include: "values,formulas", tableMaxRows: 25, tableMaxCols: 9 });
console.log(overviewCheck.ndjson);
const errorCheck = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errorCheck.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outDir, "Kalibrierfeld_Cross_Validation_Proben_3_bis_11.xlsx"));
console.log(path.join(outDir, "Kalibrierfeld_Cross_Validation_Proben_3_bis_11.xlsx"));

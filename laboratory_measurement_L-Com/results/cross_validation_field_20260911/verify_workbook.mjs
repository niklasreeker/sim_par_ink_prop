import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const dir = path.dirname(fileURLToPath(import.meta.url));
const file = path.join(dir, "Kalibrierfeld_Cross_Validation_Proben_3_bis_11.xlsx");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 5000 });
console.log(sheets.ndjson);
const drawings = await workbook.inspect({ kind: "drawing", sheetId: "LOPO je Probe", maxChars: 5000 });
console.log(drawings.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "saved workbook formula error scan",
});
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: "Übersicht", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(dir, "previews", "saved_workbook_overview.png"), new Uint8Array(await preview.arrayBuffer()));

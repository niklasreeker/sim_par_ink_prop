#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tankmassenverlust der Proben 305, 308 und 309.

Zeitnull: jeweils direkt nach der ersten Tankprobe.
Die erste Entnahme liegt damit außerhalb des Bilanzzeitraums.

Berechnung:
    Verlust(t) = Tank(0) - Tank(t) - Probenmasse_seit_Start(t)

Mittlere Verlustrate:
    bereinigter Endverlust / gesamte Messdauer

Quelle:
Notion -> Studium -> Masterarbeit -> Sensor an der Anlage
Abschnitte Probe 305, 308 und 309, Stand 14.10.2026.

Bei Probe 305 werden vier vergleichbare Wägungen im inneren
Kreislauf verwendet. Bei 308 und 309 zusätzlich Zwischenwerte.

Abgezogen werden die gewogenen Probenmassen. Verbleibende
Waagensprünge und Änderungen des Leitungsinventars bleiben enthalten.
Der bereinigte Tankverlust ist daher nicht unabhängig als reine
Verdunstung nachgewiesen.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import webbrowser

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Datenformat:
# (Uhrzeit, Tankmasse in g, kumulierte Probenmasse seit Zeitnull in g,
#  Bemerkung)
#
# Die erste Tankprobe wurde vor dem Startwert bereits entnommen.
# Ihre Masse wird deshalb nicht erneut abgezogen.

DATA = {
    305: [
        ("10:58", 9005, 0, "Nach P1; erste Probe 34,2 g bereits entnommen"),
        ("12:15", 7915, 95, "Nach P2; Probe um 11:58: 95 g"),
        ("13:37", 7470, 155, "Nach P3; Probe um 13:27: 60 g"),
        ("15:54", 6255, 285, "Nach P4; Probe 130 g"),
    ],

    308: [
        ("11:20", 5120, 0, "Nach P1; erste Probe 105 g bereits entnommen"),
        ("11:57", 4775, 0, "Zwischenwägung"),
        ("12:11", 4660, 0, "Zwischenwägung"),
        ("13:09", 4170, 0, "Zwischenwägung"),
        ("13:13", 4135, 0, "Vor P2"),
        ("13:13", 3960, 150, "Nach P2; Probe 150 g"),
        ("13:17", 3925, 150, "Zwischenwägung"),
        ("13:41", 3690, 150, "Zwischenwägung"),
        ("14:17", 3380, 150, "Vor P3"),
        ("14:17", 3270, 220, "Nach P3; Probe 70 g"),
        ("15:08", 2850, 220, "Zwischenwägung"),
        ("15:30", 2660, 220, "Zwischenwägung"),
        ("15:49", 2525, 220, "Vor P4; Protokollangabe '2525h' als g"),
        ("15:49", 2460, 261, "Nach P4; Probe 41 g"),
        ("16:45", 2100, 261, "Zwischenwägung"),
        ("17:11", 1930, 261, "Letzte dokumentierte Wägung"),
    ],

    309: [
        ("11:43", 6370, 0, "Nach P1; erste Probe 70 g bereits entnommen"),
        ("12:16", 6130, 0, "Zwischenwägung"),
        ("12:54", 5800, 0, "Zwischenwägung"),
        ("13:26", 5545, 0, "Vor P2"),
        ("13:26", 5435, 90, "Nach P2; Probe 90 g"),
        ("14:04", 5125, 90, "Zwischenwägung"),
        ("14:39", 4840, 90, "Zwischenwägung"),
        ("15:02", 4640, 90, "Vor P3"),
        ("15:02", 4550, 140, "Nach P3; Probe 50 g"),
        ("15:32", 4290, 140, "Zwischenwägung"),
        ("15:47", 4160, 140, "Vor P4"),
        ("15:47", 4030, 260, "Nach P4; Probe 120 g"),
        ("16:23", 3790, 260, "Letzte Wägung; Pumpenänderungen ab 15:51"),
    ],
}


STYLES = {
    305: {
        "color": "#2166AC",
        "marker": "o",
        "linestyle": "-",
        "zorder": 4,
    },
    308: {
        "color": "#D2691E",
        "marker": "s",
        "linestyle": "--",
        "zorder": 3,
    },
    309: {
        "color": "#238B45",
        "marker": "^",
        "linestyle": "-.",
        "zorder": 2,
    },
}


def evaluate(probe, measurements):
    """Berechnet bereinigten Verlust und mittlere Verlustrate."""

    t0 = datetime.strptime(measurements[0][0], "%H:%M")
    m0 = measurements[0][1]

    rows = []

    for clock, tank_g, cumulative_sample_g, note in measurements:
        timestamp = datetime.strptime(clock, "%H:%M")
        elapsed_h = (timestamp - t0).total_seconds() / 3600

        # Vorher-/Nachherwerte dürfen dieselbe dokumentierte Minute haben.
        # Ihre Reihenfolge bleibt wie im Protokoll erhalten.
        if rows:
            if elapsed_h < rows[-1]["zeit_h"]:
                raise ValueError(
                    f"Probe {probe}: Zeitpunkte nicht chronologisch."
                )

            if cumulative_sample_g < rows[-1]["proben_seit_start_g"]:
                raise ValueError(
                    f"Probe {probe}: kumulierte Probenmasse nimmt ab."
                )

        raw_loss = m0 - tank_g
        adjusted_loss = raw_loss - cumulative_sample_g

        rows.append({
            "probe": probe,
            "uhrzeit": clock,
            "zeit_h": elapsed_h,
            "tank_g": tank_g,
            "proben_seit_start_g": cumulative_sample_g,
            "tankverlust_roh_g": raw_loss,
            "tankverlust_bereinigt_g": adjusted_loss,
            "bemerkung": note,
        })

    if rows[0]["proben_seit_start_g"] != 0:
        raise ValueError(
            f"Probe {probe}: Probenmasse am Start muss null sein."
        )

    duration_h = rows[-1]["zeit_h"]

    if duration_h <= 0:
        raise ValueError(
            f"Probe {probe}: Messdauer muss positiv sein."
        )

    final_loss_g = rows[-1]["tankverlust_bereinigt_g"]
    average_rate_g_h = final_loss_g / duration_h

    summary = {
        "probe": probe,
        "start_nach_p1": measurements[0][0],
        "ende": measurements[-1][0],
        "dauer_h": duration_h,
        "tankverlust_roh_g": rows[-1]["tankverlust_roh_g"],
        "proben_abzug_g": rows[-1]["proben_seit_start_g"],
        "tankverlust_bereinigt_g": final_loss_g,
        "mittlerer_verlust_g_h": average_rate_g_h,
    }

    return rows, summary


def write_csv(path, rows):
    """Speichert eine CSV mit Semikolon und UTF-8 für Excel."""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Tankmassenverlust: Probe 305, 308 und 309"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Ausgabeordner; standardmäßig der Ordner des Skripts",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Gespeicherte Grafik im Bildbetrachter öffnen",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(11.8, 7.2))

    all_rows = []
    summaries = []

    for probe, measurements in DATA.items():
        rows, summary = evaluate(probe, measurements)

        all_rows.extend(rows)
        summaries.append(summary)

        times = [row["zeit_h"] for row in rows]
        losses = [row["tankverlust_bereinigt_g"] for row in rows]

        rate_text = (
            f"{summary['mittlerer_verlust_g_h']:.1f}"
            .replace(".", ",")
        )

        ax.plot(
            times,
            losses,
            label=f"Probe {probe}  ·  Ø {rate_text} g/h",
            linewidth=2.1,
            markersize=5.6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            **STYLES[probe],
        )

    ax.set_xlabel(
        "Zeit seit erster Tankprobe [h]",
        labelpad=10,
    )

    ax.set_ylabel(
        "Kumulativer Massenverlust ohne Probenentnahmen [g]",
        labelpad=10,
    )

    ax.set_xlim(-0.06, 6.1)
    ax.set_ylim(-35, 3150)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(500))

    # Deutsche Tausendertrennzeichen auf der y-Achse.
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position:
                f"{value:,.0f}".replace(",", ".")
        )
    )

    ax.grid(axis="both", alpha=0.20)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#dddddd",
        fontsize=11.5,
        labelspacing=0.7,
    )

    fig.text(
        0.115,
        0.955,
        "Tankmassenverlust im Vergleich",
        fontsize=19,
        weight="bold",
        ha="left",
    )

    fig.text(
        0.115,
        0.912,
        "Probe 305, 308 und 309  ·  "
        "gewogene Probenentnahmen abgezogen",
        fontsize=11.5,
        color="#555555",
        ha="left",
    )

    fig.text(
        0.115,
        0.085,
        "Zeitnull: jeweils direkt nach P1.  "
        "Ø = bereinigter Endverlust / gesamte Messdauer.",
        fontsize=9.5,
        color="#555555",
    )

    fig.text(
        0.115,
        0.052,
        "305: vier Wägungen im inneren Kreislauf; "
        "308/309: zusätzliche Zwischenwerte. "
        "Linien verbinden Messpunkte.",
        fontsize=9.2,
        color="#555555",
    )

    fig.text(
        0.115,
        0.021,
        "Verbleibende Waagensprünge und Inventaränderungen "
        "sind enthalten; keine unabhängig belegte reine Verdunstung.",
        fontsize=9.2,
        color="#555555",
    )

    fig.subplots_adjust(
        left=0.115,
        right=0.97,
        bottom=0.18,
        top=0.86,
    )

    filename = "Tankmassenverlust_305_308_309"

    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            args.output_dir / f"{filename}.{extension}",
            dpi=220,
            facecolor="white",
        )

    plt.close(fig)

    write_csv(
        args.output_dir / "Tankmassenverlust_Messpunkte.csv",
        all_rows,
    )

    write_csv(
        args.output_dir / "Tankmassenverlust_Verlustraten.csv",
        summaries,
    )

    print("\nBereinigte Tankmassenverluste:\n")

    for summary in summaries:
        print(
            f"Probe {summary['probe']}: "
            f"{summary['start_nach_p1']} bis {summary['ende']} | "
            f"{summary['tankverlust_bereinigt_g']:.1f} g "
            f"in {summary['dauer_h']:.4f} h | "
            f"Ø {summary['mittlerer_verlust_g_h']:.2f} g/h"
        )

    png_path = args.output_dir / f"{filename}.png"

    print(f"\nDiagramm gespeichert: {png_path}")

    if args.show:
        webbrowser.open(png_path.resolve().as_uri())


if __name__ == "__main__":
    main()
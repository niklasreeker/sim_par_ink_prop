import matplotlib.pyplot as plt
import numpy as np


def run():
    # Datenbasis aus deinen Vorgaben
    # Struktur: 'Label': Wert

    data_density = {
        'Signalhub\n(1,72 - 1,82 % Al)': 0.001279,
        'Auflösung\n(Biegeschwinger)': 0.0001,
        'Auflösung\n(Coriolis)': 0.0005,
        'Temp.-Fehler\n(±0,1 °C)': 0.000033
    }

    data_ri = {
        'Signalhub\n(1,72 - 1,82 % Al)': 0.000015,
        'Auflösung\n(Refraktometer)': 0.0002,
        'Temp.-Fehler\n(±0,1 °C)': 0.000015
    }

    data_sound = {
        'Signalhub\n(1,72 - 1,82 % Al)': 0.323173,
        'Auflösung\n(Ultraschall)': 0.01,
        'Temp.-Fehler\n(±0,1 °C)': 0.150518
    }

    data_visc = {
        'Signalhub\n(1,72 - 1,82 % Al)': 0.003496,
        'Auflösung\n(Viskosimeter)': 0.013,
        'Temp.-Fehler\n(±0,1 °C)': 0.001933
    }

    # Plot-Konfiguration
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vergleich: Nutzsignal vs. Messunsicherheiten', fontsize=16, fontweight='bold')

    # Hilfsfunktion zum Zeichnen der einzelnen Subplots
    def plot_bar_chart(ax, title, data_dict, unit):
        labels = list(data_dict.keys())
        values = list(data_dict.values())

        # Farbkodierung: Signal = Grün, Auflösung = Blau, Temp-Fehler = Rot
        colors = []
        for label in labels:
            if 'Signalhub' in label:
                colors.append('#2ca02c')  # Grün
            elif 'Auflösung' in label:
                colors.append('#1f77b4')  # Blau
            else:
                colors.append('#d62728')  # Rot

        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(f'Messwertänderung [{unit}]', fontsize=11)
        ax.tick_params(axis='x', labelsize=10)

        # Y-Achse etwas höher machen, damit die Zahlen darüber passen
        max_val = max(values)
        ax.set_ylim(0, max_val * 1.25)

        # Zahlenwerte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            # Formatierung: Nicht zu viele Nullen, aber genau genug
            formatted_height = f"{height:.6f}".rstrip('0').rstrip('.')
            ax.text(bar.get_x() + bar.get_width() / 2., height + (max_val * 0.03),
                    formatted_height, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Zeichne die 4 Diagramme
    plot_bar_chart(axs[0, 0], 'Dichtemessung', data_density, 'g/cm³')
    plot_bar_chart(axs[0, 1], 'Brechungsindex', data_ri, 'nD')
    plot_bar_chart(axs[1, 0], 'Schallgeschwindigkeit', data_sound, 'm/s')
    plot_bar_chart(axs[1, 1], 'Viskosität', data_visc, 'mPa·s')

    # Layout optimieren (ohne extra Platz für die Legende)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    run()
# Tunnel-PINN mit variabler Geometrie

Masterarbeit 2026. Ein hybrides physikinformiertes neuronales Netz (PINN) ersetzt die FEM-Rechnung
eines Tunnels mit Betonschale im Boden. Eingaben: Steifemodul E, Wichte γ, Reibungswinkel φ,
Tiefe der Tunnelachse und Schalendicke. Ausgaben: Verschiebungsfeld, Schnittgrößen der Schale (M, N)
und Mohr-Coulomb-Ausnutzung im Boden.

Trainiert wurde mit 1080 FEniCSx-Rechnungen (Mohr-Coulomb) über 15 000 Schritte mit Physikgewicht
w_phys = 0,5 (Modell `PINNs/results/geo_15k_w05/geo_15k_w05_model.pt`).

| Kennzahl (15k, w 0,5) | Wert |
|---|---|
| Validierung (216 Läufe), RMSE | 0,090 mm (rel. 8,5 %) |
| Test (30 unabhängige Läufe), RMSE | 0,087 mm (rel. 11,3 %) |
| Abfragezeit PINN vs. FEM | 25 ms vs. 2,9 s |

## Ordnerstruktur

```
PINNs/
  tunnel_pinn/      Modell, Physik-Verluste, Training, Nachlauf (M, N, Mohr-Coulomb)
  examples/         Skripte: Training, Vorhersage, Test, Grenzdiagramme, Benchmark, PLAXIS-Vergleich
  results/          trainiertes Modell (geo_15k_w05) und Auswertungen
fenics-soil/
  src/              FEM-Skripte (FEniCSx, Mohr-Coulomb)
  fem_data_geo/     1080 FEM-Rechnungen (Trainings- und Validierungsdaten, .npz)
  fem_data_geo_test/ 34 unabhängige Testrechnungen
```

## Installation

```
pip install -r requirements.txt
```

Alle Befehle werden im Ordner `PINNs` ausgeführt.

## Vorhersage mit dem trainierten Modell

```
python examples/tunnel_predict.py --E 150 --gamma 20 --phi 30 --depth 20 --t 0.3 --ring --mc --out results/abfragen/E150_H20_t03
```

Gültiger Bereich: E 50–500 MPa, γ 18–22 kN/m³, φ 25–35°, Tiefe 10–35 m, t 0,2–0,5 m.

## Weitere Skripte

| Zweck | Befehl |
|---|---|
| Training (ca. 3 h CPU) | `python examples/tunnel_datanet_geo.py --out_prefix results/geo_15k_w05/geo_15k_w05` |
| Test gegen unabhängige FEM-Läufe | `python examples/tunnel_test_geo.py --post --out_dir results/test_geo_15k` |
| Grenzwert-Diagramme | `python examples/tunnel_grenzdiagramme.py` |
| Zeitvergleich PINN–FEM | `python examples/tunnel_benchmark.py --fem_dir ../fenics-soil/results/benchmark_fem` |
| Vergleich mit PLAXIS | `python examples/tunnel_plaxis_vergleich.py --plaxis "results/plaxis_vergleich/<Datei>.TXT" --E 150 --gamma 20 --phi 30 --depth 20 --t 0.3` |

Die FEM-Daten erzeugt man unter Linux mit FEniCSx (`fenics-soil/run_dataset_geo.sh`).

Hinweis: Ergebnisse dienen der Veranschaulichung und ersetzen keine Bemessung.

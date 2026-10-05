#!/bin/bash
# TEST-Datensatz fuer das Geo-PINN: FEM-Rechnungen, die das Netz NIE gesehen hat.
# Die Kombinationen liegen ZWISCHEN den Werten des Trainingsrasters (run_dataset_geo.sh),
# gleichmaessig verteilt (Latin Hypercube, Seed 2026). Die letzten 4 Zeilen liegen bewusst
# knapp AUSSERHALB der trainierten Bereiche (Extrapolation: E 40 / 550 MPa, Tiefe 38 m, t 0,6 m).
# Ergebnis: fem_data_geo_test/outputs/run_XXXX.npz + fem_data_geo_test/manifest.csv
# Auswertung danach im Ordner PINNs:
#   python examples/tunnel_test_geo.py --test_dir ..\fenics-soil\fem_data_geo_test --post
#
# Eigene Kombinationen einfach unten in der Liste ergaenzen (je Zeile: E gamma phi Tiefe t).
# Vorhandene run_XXXX.npz werden uebersprungen.
set -euo pipefail

export PETSC_ARCH=linux-gnu-real64-32
export PYTHONPATH=/dolfinx-env/lib/python3.12/site-packages:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages:/usr/local/lib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/fem_data_geo_test}"

# ---------- Testkombinationen: E[MPa] gamma[kN/m3] phi[Grad] Tiefe[m] t[m] ----------
COMBOS="
105 21.25 30.5 16 0.28
280 19.75 34.0 26 0.34
430 20.5 27.0 22 0.35
345 18.5 34.5 16 0.32
355 19.75 30.5 27 0.24
390 21.25 27.0 24 0.46
190 18.75 26.5 12 0.48
230 20.5 26.5 23 0.29
305 20.25 28.0 21 0.41
320 20.25 29.0 32 0.25
405 19.0 26.0 29 0.31
460 21.0 28.5 31 0.37
415 19.5 32.5 17 0.45
155 19.0 29.5 23 0.27
75 18.75 27.5 13 0.35
215 21.5 25.5 21 0.44
70 19.25 31.5 13 0.26
245 20.75 28.0 28 0.41
95 21.5 29.0 26 0.47
270 18.25 32.5 12 0.41
255 21.5 32.0 18 0.33
375 21.75 30.5 18 0.23
170 19.5 30.5 16 0.38
445 20.25 31.0 27 0.43
330 20.25 31.5 22 0.28
130 18.5 33.0 34 0.36
475 18.75 29.0 33 0.42
160 21.0 33.0 31 0.31
120 20.75 34.0 19 0.23
200 19.25 33.5 32 0.39
40 20 30 20 0.3
550 20 30 20 0.3
150 20 30 38 0.3
150 20 30 20 0.6
"
# -------------------------------------------------------------------------------------

TOTAL=$(echo "$COMBOS" | grep -c '[0-9]')
echo "Test-Datensatz: $TOTAL Rechnungen -> $OUTDIR"
i=1
while read -r E G P D T; do
  [ -z "${E:-}" ] && continue
  f=$(printf "%s/outputs/run_%04d.npz" "$OUTDIR" "$i")
  if [ -f "$f" ]; then
    echo "-- run $i vorhanden, uebersprungen"
  else
    echo "── run $i / $TOTAL : E=$E MPa  gamma=$G  phi=$P  depth=$D m  t=$T m ──"
    python3 "$SCRIPT_DIR/src/mohr_coulomb_dataexport_geo.py"       --run_id "$i" --E "$E" --gamma "$G" --phi "$P" --depth "$D" --t_liner "$T" --outdir "$OUTDIR"
  fi
  i=$((i+1))
done <<< "$COMBOS"
echo "fertig: $TOTAL Rechnungen in $OUTDIR"

#!/bin/bash
# Erzeugt den FEM-Datensatz mit VARIABLER GEOMETRIE fuer das PINN
# (PINNs/tunnel_pinn/tunnel_geo.py) mit src/mohr_coulomb_dataexport_geo.py.
# Ergebnis: fem_data_geo/outputs/run_XXXX.npz + fem_data_geo/manifest.csv
#
# Die Listen unten legen das Raster fest -- einfach anpassen.
#   DEPTHS : Tiefe der Tunnelachse unter Gelaendeoberkante [m]
#            (50 m Gebiet: 10 = 10 m unter GOK, 35 = 15 m ueber der Sohle)
#   TS     : Schalendicke [m]
#   E / GAMMAS / PHIS : Bodenkennwerte
# Anzahl Rechnungen = Produkt der Listenlaengen (wird vor dem Start ausgegeben).
#
# Bereits vorhandene run_XXXX.npz werden uebersprungen -> ein abgebrochener Lauf kann
# einfach neu gestartet werden. WICHTIG: Listen dann NICHT aendern, sonst passen die
# run_ids nicht mehr zu den Parametern (neuen Ordner mit OUTDIR verwenden).
set -euo pipefail

export PETSC_ARCH=linux-gnu-real64-32
export PYTHONPATH=/dolfinx-env/lib/python3.12/site-packages:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages:/usr/local/lib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/fem_data_geo}"

# ---------- Raster (anpassen) ----------
DEPTHS="10 15 20 25 30 35"       # 5-m-Schritte; 10-m-Schritte waeren z. B. "10 20 30"
TS="0.2 0.3 0.4 0.5"
ES="50 100 150 300 500"          # 100 zusaetzlich: im weichen Bereich aendert sich u am staerksten
GAMMAS="18 20 22"
PHIS="25 30 35"
# ----------------------------------------

n() { echo $#; }
TOTAL=$(( $(n $DEPTHS) * $(n $TS) * $(n $ES) * $(n $GAMMAS) * $(n $PHIS) ))
echo "Geometrie-Datensatz: $TOTAL Rechnungen -> $OUTDIR"

i=1
for D in $DEPTHS; do
  for T in $TS; do
    for E in $ES; do
      for g in $GAMMAS; do
        for phi in $PHIS; do
          f=$(printf "%s/outputs/run_%04d.npz" "$OUTDIR" "$i")
          if [ -f "$f" ]; then
            echo "-- run $i vorhanden, uebersprungen"
          else
            echo "── run $i / $TOTAL : depth=$D m  t=$T m  E=$E MPa  gamma=$g  phi=$phi ──"
            python3 "$SCRIPT_DIR/src/mohr_coulomb_dataexport_geo.py" \
              --run_id "$i" --depth "$D" --t_liner "$T" --E "$E" --gamma "$g" --phi "$phi" \
              --outdir "$OUTDIR"
          fi
          i=$((i+1))
        done
      done
    done
  done
done
echo "fertig: $TOTAL Rechnungen in $OUTDIR"

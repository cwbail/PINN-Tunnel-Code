#!/bin/bash
# ZEITMESSUNG der vollstaendigen FEM-Rechnung (mohr_coulomb.py) fuer den Vergleich FEM <-> PINN.
# Jede Rechnung macht alles, was man in der Praxis braucht: Netz, Loesen, Spannungen,
# Mohr-Coulomb-Ausnutzung, ParaView-Export (XDMF) und Schnittgroessen M/N.
# Die Kombinationen sind die ersten N Testrechnungen aus run_dataset_geo_test.sh,
# dadurch kann das PINN spaeter mit genau denselben Faellen verglichen werden.
#
# Aufruf (im Ordner fenics-soil, NUR wenn sonst nichts rechnet!):
#   bash run_benchmark_fem.sh          # 10 Rechnungen
#   N=30 bash run_benchmark_fem.sh     # 30 Rechnungen
#
# Ergebnis: results/benchmark_fem/
#   lauf_XX/            ParaView-Dateien, Schnittgroessen, zeiten.json je Rechnung
#   benchmark_fem.csv   eine Zeile je Rechnung (Wanduhrzeit, CPU-Zeit, Speicher, Abschnitte)
#   system.txt          Rechner (CPU, Kerne, Arbeitsspeicher) -- fuer die Arbeit angeben
# Danach den ganzen Ordner results/benchmark_fem nach Windows kopieren und im Ordner PINNs:
#   python examples/tunnel_benchmark.py --fem_dir ..\fenics-soil\results\benchmark_fem
set -euo pipefail

export PETSC_ARCH=linux-gnu-real64-32
export PYTHONPATH=/dolfinx-env/lib/python3.12/site-packages:/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/lib/python3.12/dist-packages:/usr/local/lib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results/benchmark_fem}"
N="${N:-30}"
# Beton wie in den Trainingsdaten (mohr_coulomb_dataexport_geo.py): 34 GPa, nu 0.2
E_LINER=34000
NU_LINER=0.2

# ---------- Kombinationen: E[MPa] gamma[kN/m3] phi[Grad] Tiefe[m] t[m] (= Testdaten 1..30) ----------
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
"

mkdir -p "$OUTDIR"
CSV="$OUTDIR/benchmark_fem.csv"

# Rechnerdaten festhalten
{
  echo "Datum: $(date)"
  echo "Kerne (nproc): $(nproc)"
  lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|MHz" || true
  grep -E "MemTotal" /proc/meminfo 2>/dev/null || true
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-nicht gesetzt}"
} > "$OUTDIR/system.txt"
cat "$OUTDIR/system.txt"

echo "lauf,E,gamma,phi,depth,t,wanduhr_s,cpu_s,speicher_MB,knoten,netz_s,loesen_s,spannungen_mc_s,paraview_s,schnittgroessen_s,im_skript_s" > "$CSV"

i=0
T_START=$(date +%s.%N)
while read -r E G P D T; do
  [ -z "${E:-}" ] && continue
  i=$((i+1)); [ "$i" -gt "$N" ] && break
  RUN=$(printf "lauf_%02d" "$i")
  echo "=== [$i/$N] E=$E gamma=$G phi=$P Tiefe=$D t=$T ==="
  t0=$(date +%s.%N)
  python3 "$SCRIPT_DIR/src/mohr_coulomb.py" --E "$E" --gamma "$G" --phi "$P" --depth "$D" --t_liner "$T" \
      --E_liner "$E_LINER" --nu_liner "$NU_LINER" --outdir "$OUTDIR/$RUN" > "$OUTDIR/$RUN.log" 2>&1
  t1=$(date +%s.%N)
  WALL=$(python3 -c "print(round($t1-$t0, 3))")
  # Abschnittszeiten aus zeiten.json in eine CSV-Zeile
  python3 - "$OUTDIR/$RUN/zeiten.json" "$RUN" "$WALL" >> "$CSV" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); z = d["zeiten_s"]
print(",".join(str(v) for v in [sys.argv[2], d["E"], d["gamma"], d["phi"], d["depth"], d["t_liner"],
      sys.argv[3], d["cpu_zeit_s"], d["max_arbeitsspeicher_MB"], d["knoten"],
      z.get("netz"), z.get("aufbau_und_loesen"), z.get("spannungen_mohr_coulomb"),
      z.get("paraview_export"), z.get("schnittgroessen_und_ausgabe"), z.get("gesamt_im_skript")]))
PY
  echo "    Wanduhr ${WALL} s   (Details: $OUTDIR/$RUN/zeiten.json)"
done <<< "$COMBOS"
T_END=$(date +%s.%N)

python3 - "$CSV" "$T_START" "$T_END" <<'PY'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
w = [float(r["wanduhr_s"]) for r in rows]
print()
print(f"FERTIG: {len(rows)} FEM-Rechnungen in {float(sys.argv[3])-float(sys.argv[2]):.1f} s")
print(f"  Wanduhr je Rechnung: Mittel {st.mean(w):.2f} s, Median {st.median(w):.2f} s, min {min(w):.2f}, max {max(w):.2f}")
for k in ["netz_s", "loesen_s", "spannungen_mc_s", "paraview_s", "schnittgroessen_s"]:
    print(f"  {k:18s} Mittel {st.mean(float(r[k]) for r in rows):.2f} s")
print(f"  Speicher max {max(float(r['speicher_MB']) for r in rows):.0f} MB")
PY
echo "Tabelle: $CSV"

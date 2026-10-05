"""Zeit- und Rechenleistungsvergleich PINN <-> FEM.

Misst fuer das Geo-PINN dieselben Abfragen, die eine FEM-Rechnung liefert:
  Feld      Verschiebung an allen FEM-Knoten
  Ring      Schnittgroessen M(theta), N(theta) der Schale
  MC        Mohr-Coulomb-Ausnutzung eta an allen Bodenknoten
Die Kombinationen sind dieselben wie in fenics-soil/run_benchmark_fem.sh (= Testrechnungen 1..n),
die Knoten kommen aus den Testdaten (fem_data_geo_test).

Zusaetzlich:
  Massenabfrage   max|u| am Tunnelrand fuer viele Parametersaetze auf einmal
  Grenzdiagramme  Zeit fuer das komplette Raster aus tunnel_grenzdiagramme.py
  Break-even      ab wie vielen Abfragen sich das PINN inkl. FEM-Daten + Training lohnt

WICHTIG: nur laufen lassen, wenn sonst nichts rechnet (kein Training!), sonst sind die Zeiten verfaelscht.

Aufruf (Ordner PINNs):
  python examples/tunnel_benchmark.py                                     # nur PINN messen
  python examples/tunnel_benchmark.py --fem_dir ..\\fenics-soil\\results\\benchmark_fem   # mit FEM-Vergleich
Optionen fuer den Break-even, falls bekannt:
  --t_daten_h   Gesamtzeit der 1080 FEM-Rechnungen fuer die Trainingsdaten [h]
  --t_train_h   Trainingszeit [h] (sonst aus Erstell-/Aenderungszeit von log.txt und Modelldatei)
"""
import argparse, csv, json, os, platform, statistics as st, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunnel_pinn.tunnel_geo import load_model_geo, _load_runs_geo, predict_rows_geo
from tunnel_pinn.tunnel_post import ring_forces_pinn, mc_pinn, umax_wall

GRID_GAMMAS, GRID_PHIS, GRID_TS, GRID_NH, GRID_NE = 5, 5, 4, 51, 80   # Raster von tunnel_grenzdiagramme.py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/geo_15k_w05/geo_15k_w05_model.pt")
    p.add_argument("--test_dir", default=str(Path(__file__).resolve().parents[2]/"fenics-soil"/"fem_data_geo_test"),
                   help="Knoten der Testrechnungen (gleiche Kombinationen wie run_benchmark_fem.sh)")
    p.add_argument("--fem_dir", default="", help="Ordner results/benchmark_fem aus fenics-soil (mit benchmark_fem.csv)")
    p.add_argument("--fem_data", default=str(Path(__file__).resolve().parents[2]/"fenics-soil"/"fem_data_geo"),
                   help="Trainingsdaten (nur fuer Anzahl und Speichergroesse)")
    p.add_argument("--n", type=int, default=10, help="Anzahl Kombinationen (wie N im FEM-Skript)")
    p.add_argument("--rep", type=int, default=5, help="Wiederholungen je Messung (Median)")
    p.add_argument("--n_masse", type=int, default=10000, help="Parametersaetze fuer die Massenabfrage")
    p.add_argument("--no_grid", action="store_true", help="Grenzdiagramm-Raster nicht messen")
    p.add_argument("--t_daten_h", type=float, default=None, help="Zeit fuer alle FEM-Trainingsrechnungen [h]")
    p.add_argument("--t_train_h", type=float, default=None, help="Trainingszeit [h]")
    p.add_argument("--threads", type=int, default=None, help="torch-Threads (Standard: alle)")
    p.add_argument("--out_dir", default="results/benchmark")
    return p.parse_args()


def stoppuhr(fn, rep):
    "Median der Laufzeit von fn() in Sekunden (ein Aufwaermlauf vorher)."
    fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return st.median(ts)


def ordner_mb(path):
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/1e6 if p.exists() else float("nan")


def trainingszeit_h(model_path):
    "Erstellzeit log.txt (Start) bis Aenderungszeit Modelldatei (Ende); nur unter Windows sinnvoll."
    log = Path(model_path).parent/"log.txt"
    if not log.exists() or os.name != "nt":
        return None
    return (Path(model_path).stat().st_mtime-log.stat().st_ctime)/3600


def systeminfo():
    info = {"system": platform.platform(), "prozessor": platform.processor(),
            "kerne_logisch": os.cpu_count(), "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    try:
        import psutil
        info["arbeitsspeicher_GB"] = round(psutil.virtual_memory().total/1e9, 1)
    except ImportError:
        pass
    return info


def speicher_mb():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return round(getattr(m, "peak_wset", m.rss)/1e6, 1)
    except ImportError:
        return float("nan")


def fem_lesen(fem_dir):
    f = Path(fem_dir)/"benchmark_fem.csv"
    if not f.exists():
        return None, ""
    rows = [{k: (float(v) if k != "lauf" else v) for k, v in r.items()} for r in csv.DictReader(open(f, encoding="utf-8"))]
    sysf = Path(fem_dir)/"system.txt"
    return rows, (sysf.read_text(encoding="utf-8", errors="replace") if sysf.exists() else "")


if __name__ == "__main__":
    a = parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)

    t0 = time.perf_counter()
    model, cfg = load_model_geo(a.model)
    t_laden = time.perf_counter()-t0
    Ri, cx = cfg.radius, cfg.width/2
    runs = sorted(_load_runs_geo(a.test_dir), key=lambda r: r["run_id"])[:a.n]
    print(f"Modell {a.model} (geladen in {t_laden:.2f} s), {len(runs)} Kombinationen, je {a.rep} Wiederholungen")

    # ---------------- 1) Einzelabfragen wie eine FEM-Rechnung ----------------
    rows = []
    for vr in runs:
        xy = torch.tensor(vr["xy"], dtype=torch.float32)
        par = torch.tensor(np.tile([vr["E"], vr["gamma"], vr["phi"], vr["depth"], vr["t"]], (len(xy), 1)), dtype=torch.float32)
        p_q = dict(E=vr["E"], gamma=vr["gamma"], phi=vr["phi"], depth=vr["depth"], t=vr["t"])
        soil = np.hypot(vr["xy"][:, 0]-cx, vr["xy"][:, 1]+vr["depth"]) >= Ri+vr["t"]

        def feld():
            with torch.no_grad():
                predict_rows_geo(model, xy, par)
        ring = lambda: ring_forces_pinn(model, p_q, cx, -vr["depth"], Ri, vr["t"], cfg.E_liner, cfg.nu_liner, cfg.nu)
        mc = lambda: mc_pinn(model, vr["xy"][soil], p_q, cfg.nu, 2.0)

        r = dict(lauf=vr["run_id"], E=vr["E"], gamma=vr["gamma"], phi=vr["phi"], depth=vr["depth"], t=vr["t"],
                 knoten=len(xy), feld_s=stoppuhr(feld, a.rep), ring_s=stoppuhr(ring, a.rep), mc_s=stoppuhr(mc, a.rep))
        r["gesamt_s"] = r["feld_s"]+r["ring_s"]+r["mc_s"]
        rows.append(r)
        print(f"  Lauf {r['lauf']:3d}: Feld {r['feld_s']*1e3:7.1f} ms  Ring {r['ring_s']*1e3:7.1f} ms  "
              f"MC {r['mc_s']*1e3:7.1f} ms  = {r['gesamt_s']*1e3:7.1f} ms  ({r['knoten']} Knoten)")
    with open(out/"benchmark_pinn.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    T_PINN = st.median(r["gesamt_s"] for r in rows)
    mem_einzel = speicher_mb()                 # Spitze nach den Einzelabfragen (vor der Massenabfrage)

    # ---------------- 2) Massenabfrage ----------------
    rng = np.random.default_rng(0)
    lo = np.array([cfg.E_min, cfg.gamma_min, cfg.phi_min, cfg.depth_min, cfg.t_min])
    hi = np.array([cfg.E_max, cfg.gamma_max, cfg.phi_max, cfg.depth_max, cfg.t_max])
    P = lo+(hi-lo)*rng.random((a.n_masse, 5))
    t_masse = stoppuhr(lambda: umax_wall(model, P, Ri, cx), max(1, a.rep//2))
    print(f"Massenabfrage: {a.n_masse} Parametersaetze max|u| Rand in {t_masse:.2f} s "
          f"= {t_masse/a.n_masse*1e3:.3f} ms je Satz")

    # ---------------- 3) Grenzdiagramm-Raster ----------------
    n_grid = GRID_GAMMAS*GRID_PHIS*GRID_TS*GRID_NH*GRID_NE
    t_grid = None
    if not a.no_grid:
        Pg = lo+(hi-lo)*rng.random((n_grid, 5))                  # gleiche Anzahl wie das echte Raster
        t0 = time.perf_counter(); umax_wall(model, Pg, Ri, cx); t_grid = time.perf_counter()-t0
        print(f"Grenzdiagramm-Raster: {n_grid:,} Parametersaetze in {t_grid:.1f} s")
    mem = speicher_mb()

    # ---------------- 4) Vergleich mit FEM ----------------
    fem, fem_sys = fem_lesen(a.fem_dir) if a.fem_dir else (None, "")
    res = dict(modell=a.model, modell_kB=round(Path(a.model).stat().st_size/1e3, 1), laden_s=round(t_laden, 3),
               pinn_median_s=dict(feld=st.median(r["feld_s"] for r in rows), ring=st.median(r["ring_s"] for r in rows),
                                  mc=st.median(r["mc_s"] for r in rows), gesamt=T_PINN),
               masse_ms_je_satz=t_masse/a.n_masse*1e3, grid_saetze=n_grid, grid_s=t_grid,
               pinn_speicher_MB=mem, pinn_speicher_einzel_MB=mem_einzel, trainingsdaten_MB=round(ordner_mb(a.fem_data), 1),
               trainingsdaten_rechnungen=len(list(Path(a.fem_data, "outputs").glob("run_*.npz"))) if Path(a.fem_data).exists() else None,
               system_pinn=systeminfo())
    t_train_h = a.t_train_h if a.t_train_h is not None else trainingszeit_h(a.model)
    res["training_h"] = t_train_h

    L = []
    L.append("ZEITVERGLEICH PINN <-> FEM")
    L.append("=" * 60)
    L.append(f"PINN-Rechner: {res['system_pinn']}")
    L.append(f"Modell {a.model}  ({res['modell_kB']} kB, laden {t_laden:.2f} s)")
    L.append("")
    L.append(f"PINN je Abfrage (Median ueber {len(rows)} Kombinationen, je {a.rep} Wiederholungen):")
    for k in ["feld", "ring", "mc", "gesamt"]:
        L.append(f"  {k:7s} {res['pinn_median_s'][k]*1e3:9.1f} ms")
    L.append(f"Massenabfrage max|u| Rand: {res['masse_ms_je_satz']:.3f} ms je Parametersatz")
    if t_grid:
        L.append(f"Grenzdiagramm-Raster: {n_grid:,} Parametersaetze in {t_grid:.1f} s")
    L.append(f"Arbeitsspeicher PINN (Spitze, ganzer Prozess): Einzelabfragen {mem_einzel} MB, "
             f"mit Grenzdiagramm-Raster {mem} MB")
    L.append(f"Trainingsdaten: {res['trainingsdaten_rechnungen']} Rechnungen, {res['trainingsdaten_MB']} MB  |  Modell {res['modell_kB']} kB")
    L.append(f"Trainingszeit: {t_train_h:.2f} h" if t_train_h else "Trainingszeit: unbekannt (--t_train_h angeben)")

    if fem:
        fem = [f for f in fem if int(f["lauf"].split("_")[1]) <= len(rows)] or fem
        T_FEM = st.median(f["wanduhr_s"] for f in fem)
        res["fem_median_s"] = {k: st.median(f[k] for f in fem) for k in
                               ["wanduhr_s", "cpu_s", "netz_s", "loesen_s", "spannungen_mc_s", "paraview_s", "schnittgroessen_s"]}
        res["fem_speicher_MB"] = max(f["speicher_MB"] for f in fem)
        T_daten = a.t_daten_h*3600 if a.t_daten_h else (res["trainingsdaten_rechnungen"] or 1080)*T_FEM
        T_train = (t_train_h or 0)*3600
        N_star = (T_daten+T_train)/(T_FEM-T_PINN)
        res.update(T_FEM=T_FEM, faktor=T_FEM/T_PINN, T_daten_s=T_daten, T_train_s=T_train, break_even=N_star,
                   grid_fem_h=n_grid*T_FEM/3600)
        L.append("")
        L.append("FEM (vollstaendige Rechnung inkl. ParaView-Export, Median):")
        L.append(fem_sys.strip())
        for k, v in res["fem_median_s"].items():
            L.append(f"  {k:18s} {v:9.2f} s")
        L.append(f"  Arbeitsspeicher max {res['fem_speicher_MB']:.0f} MB")
        L.append("")
        L.append(f"BESCHLEUNIGUNG je Abfrage: FEM {T_FEM:.1f} s / PINN {T_PINN*1e3:.0f} ms = Faktor {T_FEM/T_PINN:,.0f}")
        L.append(f"Vorleistung PINN: FEM-Daten {T_daten/3600:.1f} h"
                 + (" (geschaetzt: Anzahl x FEM-Zeit, eher zu hoch, da ohne Export)" if not a.t_daten_h else "")
                 + f" + Training {T_train/3600:.1f} h")
        L.append(f"BREAK-EVEN: ab ca. {N_star:,.0f} Abfragen ist das PINN insgesamt schneller")
        L.append(f"Grenzdiagramme ({n_grid:,} Saetze): PINN {t_grid or float('nan'):.0f} s  <->  FEM {n_grid*T_FEM/3600:,.0f} h "
                 f"= {n_grid*T_FEM/86400:,.0f} Tage")

        # --- Bild 1: Zeit je Abfrage (log) ---
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        fm = res["fem_median_s"]
        parts_f = [("Netz", fm["netz_s"]), ("Aufbau + Loesen", fm["loesen_s"]), ("Spannungen + MC", fm["spannungen_mc_s"]),
                   ("ParaView-Export", fm["paraview_s"]), ("Schnittgroessen", fm["schnittgroessen_s"]),
                   ("Start/Importe", max(fm["wanduhr_s"]-sum(v for _, v in [("", fm[k]) for k in
                    ["netz_s", "loesen_s", "spannungen_mc_s", "paraview_s", "schnittgroessen_s"]]), 0))]
        parts_p = [("Feld", res["pinn_median_s"]["feld"]), ("Ring M/N", res["pinn_median_s"]["ring"]),
                   ("Mohr-Coulomb", res["pinn_median_s"]["mc"])]
        parts_f.append(("GESAMT", T_FEM)); parts_p.append(("GESAMT", T_PINN))
        labels = [f"FEM: {n_}" for n_, _ in parts_f]+[f"PINN: {n_}" for n_, _ in parts_p]
        vals = [v for _, v in parts_f]+[v for _, v in parts_p]
        colors = ["#c0504d"]*(len(parts_f)-1)+["#7f1d1d"]+["#4f81bd"]*(len(parts_p)-1)+["#1f3a68"]
        yy = np.arange(len(vals))[::-1]
        ax[0].barh(yy, vals, color=colors)
        for y_, v in zip(yy, vals):
            ax[0].text(v*1.15, y_, f"{v:.2f} s" if v >= 1 else f"{v*1e3:.1f} ms", va="center", fontsize=8)
        ax[0].set_yticks(yy, labels, fontsize=8); ax[0].set_xscale("log")
        ax[0].set_xlim(min(v for v in vals if v > 0)*0.3, max(vals)*8)
        ax[0].set_xlabel("Zeit je Abfrage [s] (logarithmisch)")
        ax[0].set_title(f"Eine Abfrage: FEM {T_FEM:.1f} s, PINN {T_PINN*1e3:.0f} ms (Faktor {T_FEM/T_PINN:,.0f})", fontsize=10)
        ax[0].grid(axis="x", which="both", alpha=.25)
        # --- Bild 2: Break-even ---
        N = np.logspace(0, 7, 300)
        ax[1].loglog(N, N*T_FEM/3600, color="C3", lw=2, label="FEM: N x T_FEM")
        ax[1].loglog(N, (T_daten+T_train+N*T_PINN)/3600, color="C0", lw=2, label="PINN: FEM-Daten + Training + N x T_PINN")
        ax[1].axvline(N_star, color="0.4", ls="--"); ax[1].text(N_star*1.15, (T_daten+T_train)/3600*0.15,
                                                             f"Break-even\nN = {N_star:,.0f}", fontsize=9)
        ax[1].axvline(n_grid, color="C2", ls=":"); ax[1].text(n_grid*0.85, N[0]*T_FEM/3600*3,
                                                         f"Grenzdiagramme\n{n_grid:,} Saetze\nFEM: {n_grid*T_FEM/86400:,.0f} Tage", ha="right", fontsize=9, color="C2")
        ax[1].set_xlabel("Anzahl Abfragen N"); ax[1].set_ylabel("Gesamtzeit [h]")
        ax[1].set_title("Gesamtaufwand inkl. Vorleistung"); ax[1].legend(fontsize=8, loc="upper left"); ax[1].grid(which="both", alpha=.25)
        fig.tight_layout(); fig.savefig(out/"benchmark_vergleich.png", dpi=160); plt.close(fig)
        L.append("")
        L.append(f"Bild: {out/'benchmark_vergleich.png'}")
    else:
        L.append("")
        L.append("Kein FEM-Vergleich: --fem_dir mit benchmark_fem.csv angeben (aus fenics-soil/run_benchmark_fem.sh).")

    txt = "\n".join(L)
    (out/"benchmark_zusammenfassung.txt").write_text(txt, encoding="utf-8")
    (out/"benchmark.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print()
    print(txt)

"""PLAXIS 2D <-> PINN: Verschiebungen vergleichen (ParaView-Datei + Bilder + Kennwerte).

Eingabe: PLAXIS-Export "Table of total displacements" als .txt (Tabulator, Dezimalkomma),
Spalten  Soil element | Node | Local number | X [m] | Y [m] | u_x [m] | u_y [m] | |u| [m].
Jeder Knoten steht dort mehrfach (einmal je Element) -> wird zu einem Wert je Knoten gemittelt.

Koordinaten: das PLAXIS-Gebiet wird automatisch so verschoben, dass es wie beim PINN bei
x = 0 ... Breite liegt (Tunnelachse x = 25 m). y bleibt (Gelaendeoberkante y = 0).

Aufruf (Ordner PINNs), Parameter = die in PLAXIS verwendeten Werte:
  python examples/tunnel_plaxis_vergleich.py --plaxis "results/<datei>.TXT" ^
         --E 150 --gamma 20 --phi 30 --depth 20 --t 0.3 --out_dir results/plaxis_vergleich

Ausgabe in --out_dir:
  plaxis_vergleich.vtu   ParaView: u_plaxis, u_pinn, Differenz (Vektoren in m, Betraege in mm)
  vergleich_feld.png     |u| PLAXIS | |u| PINN | Differenz
  vergleich_linien.png   Setzungsmulde an der Oberflaeche, Tunnelrand, Lotrechte durch die Achse
  vergleich.txt          Kennwerte (RMSE, Maxima, First/Ulme/Sohle)

Hinweis Geometrie: in PLAXIS ist die Schale meist ein Plattenelement (ohne Dicke) am
Ausbruchrand, der Boden reicht dort bis r = R_i. Im PINN liegt die Schale als Ring
R_i ... R_o = R_i + t, der Boden beginnt erst bei R_o. Kennwerte werden daher getrennt
fuer alle Knoten und fuer Knoten mit r >= R_o (in beiden Modellen Boden) angegeben.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunnel_pinn.tunnel_geo import load_model_geo, predict_geo
from tunnel_pinn.tunnel_data import write_vtu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plaxis", required=True, help="PLAXIS-Export (Table of total displacements, .txt)")
    p.add_argument("--model", default="results/geo_15k_w05/geo_15k_w05_model.pt")
    p.add_argument("--E", type=float, required=True, help="E-Modul Boden [MPa]")
    p.add_argument("--gamma", type=float, required=True, help="Wichte [kN/m3]")
    p.add_argument("--phi", type=float, required=True, help="Reibungswinkel [Grad]")
    p.add_argument("--depth", type=float, required=True, help="Tiefe der Tunnelachse unter GOK [m]")
    p.add_argument("--t", type=float, required=True, help="Schalendicke [m]")
    p.add_argument("--out_dir", default="results/plaxis_vergleich")
    return p.parse_args()


def read_plaxis(path):
    "PLAXIS-Tabelle -> DataFrame mit einer Zeile je Knoten (x, y, ux, uy in m)."
    d = pd.read_csv(path, sep="\t", decimal=",", encoding="utf-8-sig", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"X [m]": "x", "Y [m]": "y", "u_x [m]": "ux", "u_y [m]": "uy"})
    return d.groupby("Node")[["x", "y", "ux", "uy"]].mean().reset_index()


def stats(e_mm, ref_mm):
    "RMSE [mm] und relativer Fehler [%] (Norm Differenz / Norm Referenz)."
    return np.sqrt((e_mm**2).sum(1).mean()), np.linalg.norm(e_mm)/np.linalg.norm(ref_mm)*100


if __name__ == "__main__":
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model_geo(a.model)
    Ri, Ro = cfg.radius, cfg.radius+a.t

    # ---------------- PLAXIS einlesen und ins PINN-Koordinatensystem schieben ----------------
    d = read_plaxis(a.plaxis)
    dx = -d["x"].min()                                   # linker Rand -> x = 0
    d["x"] += dx
    width = d["x"].max()
    cx, cy = width/2, -a.depth
    xy = d[["x", "y"]].to_numpy(np.float64)
    u_px = d[["ux", "uy"]].to_numpy(np.float64)
    r = np.hypot(xy[:, 0]-cx, xy[:, 1]-cy)
    print(f"PLAXIS: {len(d)} Knoten, Gebiet {width:.1f} x {-d['y'].min():.1f} m, verschoben um dx = {dx:+.2f} m")
    print(f"        kleinster Abstand zur Achse ({cx:.1f} | {cy:.1f}): {r.min():.3f} m  (R_i im PINN = {Ri} m)")
    if abs(r.min()-Ri) > 0.05:
        print("  WARNUNG: Tunnelrand passt nicht zu R_i -- stimmt --depth bzw. die Lage der Achse?")
    if abs(width-cfg.width) > 0.5 or abs(-d["y"].min()-cfg.total_depth) > 0.5:
        print(f"  WARNUNG: Gebietsgroesse weicht vom PINN ab ({cfg.width} x {cfg.total_depth} m)")
    for nm, v, lo, hi in [("E", a.E, cfg.E_min, cfg.E_max), ("gamma", a.gamma, cfg.gamma_min, cfg.gamma_max),
                          ("phi", a.phi, cfg.phi_min, cfg.phi_max), ("depth", a.depth, cfg.depth_min, cfg.depth_max),
                          ("t", a.t, cfg.t_min, cfg.t_max)]:
        if not lo <= v <= hi:
            print(f"  WARNUNG: {nm} = {v} ausserhalb des trainierten Bereichs [{lo}, {hi}] (Extrapolation)")

    # ---------------- PINN an denselben Knoten ----------------
    u_pn = predict_geo(model, xy, a.E, a.gamma, a.phi, a.depth, a.t).astype(np.float64)
    diff = u_pn-u_px
    mm = 1e3
    soil = r >= Ro

    # ---------------- ParaView ----------------
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    tcx, tcy = xy[tri.triangles, 0].mean(1), xy[tri.triangles, 1].mean(1)
    keep = np.hypot(tcx-cx, tcy-cy) >= Ri
    write_vtu(out/"plaxis_vergleich.vtu", xy, tri.triangles[keep], {
        "u_plaxis_m": u_px, "u_pinn_m": u_pn, "diff_pinn_minus_plaxis_m": diff,
        "mag_plaxis_mm": np.linalg.norm(u_px, axis=1)*mm, "mag_pinn_mm": np.linalg.norm(u_pn, axis=1)*mm,
        "diff_mm": np.linalg.norm(diff, axis=1)*mm,
        "uy_plaxis_mm": u_px[:, 1]*mm, "uy_pinn_mm": u_pn[:, 1]*mm, "diff_uy_mm": diff[:, 1]*mm,
        "ux_plaxis_mm": u_px[:, 0]*mm, "ux_pinn_mm": u_pn[:, 0]*mm, "diff_ux_mm": diff[:, 0]*mm,
        "boden_in_beiden": soil.astype(np.float32)})

    # ---------------- Feldbilder ----------------
    trm = mtri.Triangulation(xy[:, 0], xy[:, 1], tri.triangles); trm.set_mask(~keep)
    mag_px, mag_pn = np.linalg.norm(u_px, axis=1)*mm, np.linalg.norm(u_pn, axis=1)*mm
    vmax = max(mag_px.max(), mag_pn.max())
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k, (val, tit, cmap, lim) in enumerate([
            (mag_px, "|u| PLAXIS [mm]", "viridis", (0, vmax)), (mag_pn, "|u| PINN [mm]", "viridis", (0, vmax)),
            (np.linalg.norm(diff, axis=1)*mm, "|u_PINN - u_PLAXIS| [mm]", "magma", None)]):
        levels = 40 if lim is None else np.linspace(lim[0], lim[1], 41)     # PLAXIS und PINN mit gleicher Farbskala
        tc = ax[k].tricontourf(trm, val, levels, cmap=cmap)
        fig.colorbar(tc, ax=ax[k], shrink=0.85); ax[k].set_title(tit); ax[k].set_aspect("equal")
        ax[k].add_patch(plt.Circle((cx, cy), Ro, fill=False, color="w", lw=0.8))
        ax[k].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    fig.suptitle(f"PLAXIS vs PINN   E={a.E:g} MPa, gamma={a.gamma:g}, phi={a.phi:g}, Tiefe={a.depth:g} m, t={a.t:g} m")
    fig.tight_layout(); fig.savefig(out/"vergleich_feld.png", dpi=150); plt.close(fig)

    # ---------------- Linien: Oberflaeche, Tunnelrand, Lotrechte ----------------
    interp = [mtri.LinearTriInterpolator(trm, u_px[:, i]) for i in range(2)]
    def px_at(pts):
        return np.column_stack([np.asarray(f(pts[:, 0], pts[:, 1]).filled(np.nan)) for f in interp])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # Oberflaeche (y = 0, minimal darunter wegen Interpolation)
    xs = np.linspace(0, width, 301); ps = np.column_stack([xs, np.full_like(xs, -1e-3)])
    ax[0].plot(xs, px_at(ps)[:, 1]*mm, "C3", lw=2, label="PLAXIS")
    ax[0].plot(xs, predict_geo(model, ps, a.E, a.gamma, a.phi, a.depth, a.t)[:, 1]*mm, "C0--", lw=2, label="PINN")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("u_y [mm] (+ = Hebung)"); ax[0].set_title("Gelaendeoberflaeche y = 0")
    # Tunnelrand: Bodenseite r = R_o (in beiden Modellen Boden)
    th = np.linspace(0, 2*np.pi, 361); pr = np.column_stack([cx+(Ro+1e-3)*np.cos(th), cy+(Ro+1e-3)*np.sin(th)])
    upx_r, upn_r = px_at(pr), predict_geo(model, pr, a.E, a.gamma, a.phi, a.depth, a.t)
    ur = lambda u: u[:, 0]*np.cos(th)+u[:, 1]*np.sin(th)
    ax[1].plot(np.degrees(th), ur(upx_r)*mm, "C3", lw=2, label="PLAXIS")
    ax[1].plot(np.degrees(th), ur(upn_r)*mm, "C0--", lw=2, label="PINN")
    ax[1].set_xticks([0, 90, 180, 270, 360], ["0\nUlme re.", "90\nFirst", "180\nUlme li.", "270\nSohle", "360"])
    ax[1].set_ylabel("radiale Verschiebung u_r [mm] (+ = nach aussen)"); ax[1].set_title(f"Tunnelrand Bodenseite r = R_o = {Ro:.2f} m")
    # Lotrechte durch die Achse
    ys = np.linspace(-cfg.total_depth, 0, 401); ys = ys[np.abs(ys-cy) >= Ro]
    pv = np.column_stack([np.full_like(ys, cx), ys])
    ax[2].plot(px_at(pv)[:, 1]*mm, ys, "C3.", ms=3, label="PLAXIS")
    ax[2].plot(predict_geo(model, pv, a.E, a.gamma, a.phi, a.depth, a.t)[:, 1]*mm, ys, "C0.", ms=3, label="PINN")
    ax[2].set_xlabel("u_y [mm]"); ax[2].set_ylabel("y [m]"); ax[2].set_title(f"Lotrechte x = {cx:g} m (Tunnelachse)")
    for x_ in ax:
        x_.grid(alpha=.3); x_.legend()
    fig.tight_layout(); fig.savefig(out/"vergleich_linien.png", dpi=150); plt.close(fig)

    # ---------------- Kennwerte ----------------
    rm_all, rel_all = stats(diff*mm, u_px*mm)
    rm_s, rel_s = stats(diff[soil]*mm, u_px[soil]*mm)
    surf = ps[:, 1]
    sp, sn = px_at(ps)[:, 1]*mm, predict_geo(model, ps, a.E, a.gamma, a.phi, a.depth, a.t)[:, 1]*mm
    L = [f"PLAXIS: {a.plaxis}", f"PINN:   {a.model}",
         f"Parameter: E={a.E} MPa  gamma={a.gamma} kN/m3  phi={a.phi} Grad  Tiefe={a.depth} m  t={a.t} m",
         f"Knoten: {len(d)} (davon {soil.sum()} mit r >= R_o = {Ro:.2f} m)", "",
         f"Alle Knoten:         RMSE {rm_all:.3f} mm   rel. Fehler {rel_all:.1f} %",
         f"Boden r >= R_o:      RMSE {rm_s:.3f} mm   rel. Fehler {rel_s:.1f} %",
         f"max|u|  PLAXIS {mag_px.max():.3f} mm   PINN {mag_pn.max():.3f} mm",
         f"Oberflaeche u_y bei x={cx:g}:  PLAXIS {sp[150]:+.3f} mm   PINN {sn[150]:+.3f} mm", "",
         "Tunnelrand (Bodenseite r = R_o), radial u_r [mm], + = nach aussen:"]
    for deg, nm in [(0, "Ulme re."), (90, "First"), (180, "Ulme li."), (270, "Sohle")]:
        L.append(f"  {nm:9s} PLAXIS {ur(upx_r)[deg]*mm:+7.3f}   PINN {ur(upn_r)[deg]*mm:+7.3f}")
    L += ["", f"Dateien: {out/'plaxis_vergleich.vtu'}, {out/'vergleich_feld.png'}, {out/'vergleich_linien.png'}"]
    (out/"vergleich.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

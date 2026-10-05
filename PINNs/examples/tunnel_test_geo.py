"""
TEST eines fertig trainierten Geo-Modells gegen FEM-Rechnungen, die das Netz NIE gesehen hat
(eigener Ordner, z. B. fenics-soil/fem_data_geo_test) -- 29.09.2026.

Fuer jede FEM-Rechnung im Test-Ordner wird verglichen:
  1. Verschiebungsfeld an allen FEM-Knoten: RMSE gesamt, in der Schale, im Boden, relativer Fehler
  2. max. Verschiebung am Tunnelrand (r = R_i) und die Einstufung gegen den Grenzwert (--limit)
  3. Schnittgroessen der Schale: M (Ringkinematik) und N (PINN: Ringgleichgewicht mit Erddruck + Schub;
     FEM: Ringkinematik aus den FEM-Verschiebungen)                          [mit --post]
  4. Mohr-Coulomb-Ausnutzung eta im Umkreis R_o + 3 m (c = --c)            [mit --post]
Kombinationen ausserhalb der trainierten Bereiche werden als EXTRAPOLATION markiert.

Ergebnisse im Ordner --out_dir:
  test_pro_lauf.csv        alle Kennwerte je Testrechnung
  test_zusammenfassung.txt Mittelwerte, getrennt nach innerhalb/ausserhalb der trainierten Bereiche
  test_paritaet.png        PINN gegen FEM: max. Verschiebung am Tunnelrand (und M, N mit --post)
  test_fehler.png          RMSE je Testrechnung ueber E, Tiefe und Schalendicke
  test_feld_<id>.png       Feldvergleich FEM / PINN / Fehler fuer die schlechtesten Rechnungen

Aufruf (im Ordner PINNs):
  python examples/tunnel_test_geo.py --test_dir ..\\fenics-soil\\fem_data_geo_test --post
Probelauf mit den 216 Validierungsrechnungen des Trainingsdatensatzes:
  python examples/tunnel_test_geo.py --test_dir ..\\fenics-soil\\fem_data_geo --only_val --out_dir results/test_probe
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from tunnel_pinn.tunnel_geo import load_model_geo, load_split_geo, _load_runs_geo, predict_rows_geo
from tunnel_pinn.tunnel_post import (umax_wall, ring_forces, ring_forces_pinn, mc_fem, mc_pinn)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/geo_15k_w05/geo_15k_w05_model.pt")
    p.add_argument("--test_dir", default=str(Path(__file__).resolve().parents[2]/"fenics-soil"/"fem_data_geo_test"),
                   help="Ordner mit manifest.csv und outputs/run_XXXX.npz (wie fem_data_geo)")
    p.add_argument("--only_val", action="store_true",
                   help="nur die Validierungsrechnungen verwenden (Probelauf mit dem Trainingsdatensatz)")
    p.add_argument("--limit", type=float, default=10.0, help="Grenzwert max. Verschiebung am Tunnelrand [mm]")
    p.add_argument("--post", action="store_true", help="auch Schnittgroessen (M, N) und Mohr-Coulomb vergleichen")
    p.add_argument("--c", type=float, default=2.0, help="Kohaesion [kPa] fuer den Mohr-Coulomb-Vergleich")
    p.add_argument("--n_bilder", type=int, default=3, help="Feldvergleich fuer die n schlechtesten Rechnungen")
    p.add_argument("--out_dir", default="results/test_geo")
    return p.parse_args()


def fem_ring_u(vr, cx, Ri):
    "Lineare Interpolation der FEM-Knotenverschiebungen im Schalenring (fuer die Ringkinematik)."
    xy, u = vr["xy"], vr["u"]; cy = -vr["depth"]; Ro = Ri+vr["t"]
    r = np.hypot(xy[:, 0]-cx, xy[:, 1]-cy); k = (r > Ri-0.05) & (r < Ro+0.05)
    tri = mtri.Triangulation(xy[k, 0], xy[k, 1])
    g = np.hypot(xy[k][tri.triangles, 0].mean(1)-cx, xy[k][tri.triangles, 1].mean(1)-cy)
    tri.set_mask((g < Ri) | (g > Ro))
    ix, iy = mtri.LinearTriInterpolator(tri, u[k, 0]), mtri.LinearTriInterpolator(tri, u[k, 1])
    return lambda pts: np.column_stack([np.asarray(ix(pts[:, 0], pts[:, 1])), np.asarray(iy(pts[:, 0], pts[:, 1]))])


def in_range(vr, cfg):
    "Liegt die Rechnung im trainierten Bereich? Rueckgabe (bool, Text der Ueberschreitungen)."
    checks = [("E", vr["E"], cfg.E_min, cfg.E_max), ("gamma", vr["gamma"], cfg.gamma_min, cfg.gamma_max),
              ("phi", vr["phi"], cfg.phi_min, cfg.phi_max), ("depth", vr["depth"], cfg.depth_min, cfg.depth_max),
              ("t", vr["t"], cfg.t_min, cfg.t_max)]
    out = [f"{n}={v:g}" for n, v, lo, hi in checks if not (lo-1e-9 <= v <= hi+1e-9)]
    return len(out) == 0, " ".join(out)


def plot_field(vr, pr, cx, Ri, path):
    x, y = vr["xy"][:, 0], vr["xy"][:, 1]
    tri = mtri.Triangulation(x, y)
    tri.set_mask(((x[tri.triangles].mean(1)-cx)**2+(y[tri.triangles].mean(1)+vr["depth"])**2) < Ri**2)
    fm, pm = np.linalg.norm(vr["u"], axis=1)*1e3, np.linalg.norm(pr, axis=1)*1e3
    vmax = max(fm.max(), pm.max())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (f, t, vm) in zip(axs, [(fm, "FEM |u| [mm]", vmax), (pm, "PINN |u| [mm]", vmax), (np.abs(pm-fm), "|Fehler| [mm]", None)]):
        tc = ax.tricontourf(tri, f, levels=30, cmap="viridis", vmax=vm); ax.set_aspect("equal"); plt.colorbar(tc, ax=ax); ax.set_title(t)
    fig.suptitle(f"Testrechnung {vr['run_id']}: E={vr['E']:g} MPa, γ={vr['gamma']:g}, φ'={vr['phi']:g}°, "
                 f"Tiefe={vr['depth']:g} m, t={vr['t']:g} m")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    a = parse_args()
    model, cfg = load_model_geo(a.model)
    Ri, cx = cfg.radius, cfg.width/2
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.only_val:
        _, _, _, info = load_split_geo(a.test_dir, 0.2, 0)
        runs = info["val_runs"]
    else:
        runs = _load_runs_geo(a.test_dir)
    print(f"Modell: {a.model}\nTestdaten: {a.test_dir}  ({len(runs)} Rechnungen{', nur Validierung' if a.only_val else ''})")

    rows, preds = [], {}
    for n, vr in enumerate(runs, 1):
        xy = torch.tensor(vr["xy"], dtype=torch.float32)
        par = torch.tensor(np.tile([vr["E"], vr["gamma"], vr["phi"], vr["depth"], vr["t"]], (len(xy), 1)), dtype=torch.float32)
        with torch.no_grad():
            pr = predict_rows_geo(model, xy, par).numpy()
        preds[vr["run_id"]] = pr
        e = pr-vr["u"]
        rad = np.hypot(vr["xy"][:, 0]-cx, vr["xy"][:, 1]+vr["depth"]); Ro = Ri+vr["t"]
        lin, wall = rad < Ro, rad < Ri+0.005
        ok_range, why = in_range(vr, cfg)
        u_fem = np.linalg.norm(vr["u"][wall], axis=1).max()*1e3
        u_pinn = float(umax_wall(model, np.array([[vr["E"], vr["gamma"], vr["phi"], vr["depth"], vr["t"]]]), Ri, cx)[0])
        row = dict(run_id=vr["run_id"], E=vr["E"], gamma=vr["gamma"], phi=vr["phi"], depth=vr["depth"], t=vr["t"],
                   im_bereich=ok_range, ausserhalb=why,
                   rmse=np.sqrt((e**2).mean())*1e3, rmse_schale=np.sqrt((e[lin]**2).mean())*1e3,
                   rmse_boden=np.sqrt((e[~lin]**2).mean())*1e3, rel=np.linalg.norm(e)/np.linalg.norm(vr["u"])*100,
                   umax_fem=u_fem, umax_pinn=u_pinn,
                   grenz_fem=u_fem <= a.limit, grenz_pinn=u_pinn <= a.limit)
        if a.post:
            p_q = dict(E=vr["E"], gamma=vr["gamma"], phi=vr["phi"], depth=vr["depth"], t=vr["t"])
            rf = ring_forces(fem_ring_u(vr, cx, Ri), cx, -vr["depth"], Ri, vr["t"], cfg.E_liner, cfg.nu_liner)
            rp = ring_forces_pinn(model, p_q, cx, -vr["depth"], Ri, vr["t"], cfg.E_liner, cfg.nu_liner, cfg.nu)
            g, eta_f = mc_fem(vr["xy"], vr["u"], cx, -vr["depth"], Ro, vr["E"], cfg.nu, vr["gamma"], vr["phi"], a.c)
            near = np.hypot(g[:, 0]-cx, g[:, 1]+vr["depth"]) < Ro+3.0
            eta_p = mc_pinn(model, g[near], p_q, cfg.nu, a.c)["eta"]
            row.update(M_fem_max=np.abs(rf["M"]).max(), M_pinn_max=np.abs(rp["M"]).max(),
                       M_rmse=np.sqrt(((rf["M"]-rp["M"])**2).mean()),
                       N_fem_min=rf["N"].min(), N_pinn_min=rp["N"].min(), N_rmse=np.sqrt(((rf["N"]-rp["N"])**2).mean()),
                       eta_fem_max=eta_f[near].max(), eta_pinn_max=eta_p.max(), eta_mae=np.abs(eta_f[near]-eta_p).mean())
        rows.append(row)
        print(f"  [{n:3d}/{len(runs)}] Lauf {vr['run_id']:4d}  RMSE {row['rmse']:.3f} mm  max|u| Rand FEM {u_fem:5.2f} / PINN {u_pinn:5.2f} mm"
              + ("" if ok_range else f"   EXTRAPOLATION ({why})"))

    # ---------------- CSV ----------------
    keys = list(rows[0].keys())
    with open(out/"test_pro_lauf.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys)+"\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], (float, np.floating)) else str(r[k]) for k in keys)+"\n")

    # ---------------- Zusammenfassung ----------------
    def summary(sel, name):
        if not sel:
            return [f"{name}: keine Rechnungen"]
        m = lambda k: np.mean([r[k] for r in sel])
        wrong = sum(r["grenz_fem"] != r["grenz_pinn"] for r in sel)
        L = [f"{name}: {len(sel)} Rechnungen",
             f"  Verschiebungsfeld   RMSE {m('rmse'):.3f} mm (Schale {m('rmse_schale'):.3f}, Boden {m('rmse_boden'):.3f}), rel. Fehler {m('rel'):.1f} %",
             f"  max|u| Tunnelrand   mittl. Abweichung {np.mean([abs(r['umax_pinn']-r['umax_fem']) for r in sel]):.2f} mm, "
             f"max {np.max([abs(r['umax_pinn']-r['umax_fem']) for r in sel]):.2f} mm",
             f"  Grenzwert {a.limit:g} mm   {wrong} von {len(sel)} falsch eingestuft"]
        if a.post:
            L += [f"  Moment M            RMSE {m('M_rmse'):.1f} kNm/m (|M|max FEM im Mittel {m('M_fem_max'):.1f}, rel {m('M_rmse')/m('M_fem_max')*100:.0f} %)",
                  f"  Normalkraft N       RMSE {m('N_rmse'):.0f} kN/m (|N|max FEM im Mittel {-m('N_fem_min'):.0f}, rel {m('N_rmse')/(-m('N_fem_min'))*100:.0f} %)",
                  f"  Mohr-Coulomb eta    mittl. Abweichung {m('eta_mae'):.3f} (3 m um die Schale, c = {a.c:g} kPa)"]
        return L
    inner = [r for r in rows if r["im_bereich"]]; outer = [r for r in rows if not r["im_bereich"]]
    lines = [f"TEST Modell {a.model}", f"Testdaten {a.test_dir}" + (" (nur Validierung)" if a.only_val else ""), ""] \
        + summary(inner, "Innerhalb der trainierten Bereiche") + [""] + summary(outer, "Ausserhalb (Extrapolation)")
    (out/"test_zusammenfassung.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n"+"\n".join(lines))

    # ---------------- Paritaetsdiagramm ----------------
    panels = [("umax_fem", "umax_pinn", "max. Verschiebung am Tunnelrand [mm]")]
    if a.post:
        panels += [("M_fem_max", "M_pinn_max", "|M|max [kNm/m]"), ("N_fem_min", "N_pinn_min", "N min (Druck) [kN/m]")]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.4*len(panels), 5.2), squeeze=False)
    for ax, (kf, kp, lab) in zip(axs[0], panels):
        xf = np.array([r[kf] for r in rows]); xp = np.array([r[kp] for r in rows]); ir = np.array([r["im_bereich"] for r in rows])
        lo, hi = min(xf.min(), xp.min()), max(xf.max(), xp.max()); pad = 0.05*(hi-lo)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="0.5", lw=1)
        ax.scatter(xf[ir], xp[ir], s=22, color="#2c6784", label="im trainierten Bereich")
        if (~ir).any():
            ax.scatter(xf[~ir], xp[~ir], s=30, marker="^", color="#b04a3a", label="Extrapolation")
        if kf == "umax_fem":
            ax.axvline(a.limit, color="k", ls=":", lw=1); ax.axhline(a.limit, color="k", ls=":", lw=1)
        ax.set_xlabel(f"FEM: {lab}"); ax.set_ylabel(f"PINN: {lab}"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Test: PINN gegen FEM (auf der Diagonalen = exakt gleich)"); fig.tight_layout()
    fig.savefig(out/"test_paritaet.png", dpi=140); plt.close(fig)

    # ---------------- Fehler ueber den Parametern ----------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, k, lab in zip(axs, ["E", "depth", "t"], ["E [MPa]", "Tiefe H [m]", "Schalendicke t [m]"]):
        ax.scatter([r[k] for r in rows], [r["rmse"] for r in rows], s=22,
                   c=["#2c6784" if r["im_bereich"] else "#b04a3a" for r in rows])
        ax.set_xlabel(lab); ax.grid(alpha=0.3)
    axs[0].set_ylabel("RMSE Verschiebungsfeld [mm]"); axs[0].set_xscale("log")
    fig.suptitle("Test: Fehler je Rechnung (blau = im trainierten Bereich, rot = Extrapolation)"); fig.tight_layout()
    fig.savefig(out/"test_fehler.png", dpi=140); plt.close(fig)

    # ---------------- Feldbilder der schlechtesten Rechnungen ----------------
    for r in sorted(rows, key=lambda r: -r["rmse"])[:a.n_bilder]:
        vr = next(v for v in runs if v["run_id"] == r["run_id"])
        plot_field(vr, preds[r["run_id"]], cx, Ri, out/f"test_feld_{r['run_id']:04d}.png")
    print(f"\ngeschrieben: {out}")

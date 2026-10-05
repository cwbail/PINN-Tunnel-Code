"""
HYBRIDES Tunnel-PINN mit VARIABLER GEOMETRIE -- Startskript (Weg B, 28.09.2026).

Wie examples/tunnel_datanet_2D_torch.py (Einstellung E), aber das Netz bekommt zusaetzlich
die Tiefe der Tunnelachse und die Schalendicke als Eingang:
    u = NN(x, y, E, gamma, phi, depth, t)
Nach dem Training kann jede Kombination innerhalb der Bereiche abgefragt werden
(tunnel_geo.predict_geo).

FEM-Daten: erzeugt mit fenics-soil/src/mohr_coulomb_dataexport_geo.py
(fenics-soil/run_dataset_geo.sh), Standardordner fenics-soil/fem_data_geo.
Zum Testen mit den ALTEN Daten (alle Tiefe 25 m, t = 0,2 m):
  python examples/tunnel_datanet_geo.py --data_dir ..\\fenics-soil\\fem_data --n_steps 200

Aufruf:
  python examples/tunnel_datanet_geo.py --n_steps 8000 --out_prefix results/geo/geo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from tunnel_pinn.tunnel_geo import (GeoConfig, train_hybrid_geo, save_model_geo, predict_rows_geo)

DEFAULT_DATA = str(Path(__file__).resolve().parents[2]/"fenics-soil"/"fem_data_geo")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=15000)    # Standardmodell results/geo_15k_w05 (val 0.099 mm)
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA)
    p.add_argument("--w_data", type=float, default=1.0)
    p.add_argument("--w_phys", type=float, default=0.5, help="Gleichgewicht Boden") # 0.5: mehr Physik fuer Faelle ohne passende FEM-Rechnung (0.1 trifft die FEM-Daten genauer: val 0.135 statt 0.242 mm)
    p.add_argument("--w_liner_phys", type=float, default=None, help="Gleichgewicht Schale (Standard: = w_phys)")
    p.add_argument("--w_iface_disp", type=float, default=1.0, help="Uebergang u_Schale = u_Boden an R_o")
    p.add_argument("--liner_norm", choices=["sig", "stiff"], default="stiff")
    # Geometrie-Bereiche (Tiefe = Tunnelachse unter Gelaendeoberkante)
    p.add_argument("--depth_min", type=float, default=10.0)
    p.add_argument("--depth_max", type=float, default=35.0)
    p.add_argument("--t_min", type=float, default=0.2)
    p.add_argument("--t_max", type=float, default=0.5)
    # nur fuer alte Daten ohne Geometriefelder
    p.add_argument("--default_depth", type=float, default=25.0)
    p.add_argument("--default_t", type=float, default=0.2)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--n_train_runs", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_prefix", type=str, default="tunnel_geo")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = GeoConfig(
        radius=2.5, width=50.0, total_depth=50.0, nu=0.30,
        depth=25.0, t_liner=0.2,        # werden hier nicht benutzt (Geometrie kommt pro Punkt)
        depth_min=args.depth_min, depth_max=args.depth_max, t_min=args.t_min, t_max=args.t_max,
        E_min=50., E_max=500., gamma_min=18., gamma_max=22., phi_min=25., phi_max=35.,   # aus den Daten ueberschrieben
        E_liner=34000., nu_liner=0.2,
        hidden=128, n_layers=4, hidden_liner=64, n_layers_liner=4,
        n_steps=args.n_steps, lr=1e-3,
        log_every=max(1, args.n_steps//20),
        device=args.device,
    )

    model, history, info = train_hybrid_geo(
        cfg, args.data_dir, w_data=args.w_data, w_phys=args.w_phys, w_liner_phys=args.w_liner_phys,
        w_iface_disp=args.w_iface_disp, liner_norm=args.liner_norm,
        val_fraction=args.val_fraction, split_seed=args.split_seed, n_train_runs=args.n_train_runs,
        default_depth=args.default_depth, default_t=args.default_t)
    save_model_geo(model, cfg, f"{args.out_prefix}_model.pt")

    # Spalten: Schritt, Gesamtloss, Datenloss, Physik (Boden+Schale), RMSE Train, RMSE Val, Uebergang
    hist = np.array(history)
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(hist[:, 0], hist[:, 1], "k", lw=1.6, label="total")
    plt.semilogy(hist[:, 0], hist[:, 2], label="data")
    plt.semilogy(hist[:, 0], hist[:, 3], label="physics (soil + liner)")
    plt.semilogy(hist[:, 0], np.maximum(hist[:, 6], 1e-12), label="interface u")
    plt.xlabel("step"); plt.ylabel("loss"); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.title("Geometry-parametric hybrid PINN - loss terms")
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_losses.png", dpi=150)

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist[:, 0], hist[:, 4], label="train RMSE"); plt.plot(hist[:, 0], hist[:, 5], label="validation RMSE")
    plt.xlabel("step"); plt.ylabel("RMSE [mm]"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Geometry-parametric hybrid PINN - fit vs generalisation")
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_rmse.png", dpi=150)

    # ----- Fehler pro Validierungslauf, getrennt nach Schale und Boden -----
    cx = cfg.width/2
    print("\nPro Validierungslauf [mm]:  Lauf  E  gamma  phi  depth  t | max|u| FEM  PINN | RMSE gesamt  Schale  Boden")
    for vr in info["val_runs"]:
        xy = torch.tensor(vr["xy"])
        par = torch.tensor(np.tile([vr["E"], vr["gamma"], vr["phi"], vr["depth"], vr["t"]], (len(xy), 1)), dtype=torch.float32)
        with torch.no_grad():
            pr = predict_rows_geo(model, xy, par).numpy()
        e = pr-vr["u"]
        r = np.hypot(vr["xy"][:, 0]-cx, vr["xy"][:, 1]+vr["depth"])
        lin = r < cfg.radius+vr["t"]
        rm = lambda k: np.sqrt((e[k]**2).mean())*1e3 if k.any() else float("nan")
        print(f"  {vr['run_id']:>4} {vr['E']:>5.0f} {vr['gamma']:>4.0f} {vr['phi']:>4.0f} {vr['depth']:>5.1f} {vr['t']:>4.2f} | "
              f"{np.linalg.norm(vr['u'], axis=1).max()*1e3:>7.2f} {np.linalg.norm(pr, axis=1).max()*1e3:>7.2f} | "
              f"{rm(np.ones_like(lin)):>7.3f} {rm(lin):>7.3f} {rm(~lin):>7.3f}")

    # ----- Feldvergleich fuer den ersten Validierungslauf -----
    vr = info["val_runs"][0]
    xy = torch.tensor(vr["xy"])
    par = torch.tensor(np.tile([vr["E"], vr["gamma"], vr["phi"], vr["depth"], vr["t"]], (len(xy), 1)), dtype=torch.float32)
    with torch.no_grad():
        pr = predict_rows_geo(model, xy, par).numpy()
    x, y = vr["xy"][:, 0], vr["xy"][:, 1]
    tri = mtri.Triangulation(x, y)
    tcx, tcy = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    tri.set_mask(((tcx-cx)**2+(tcy+vr["depth"])**2) < cfg.radius**2)      # Hohlraum dieses Laufs ausblenden
    fem_mag = np.linalg.norm(vr["u"], axis=1)*1e3; pinn_mag = np.linalg.norm(pr, axis=1)*1e3
    vmax = max(fem_mag.max(), pinn_mag.max())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (fld, ttl, vm) in zip(axs, [(fem_mag, "FEM |u| [mm]", vmax), (pinn_mag, "PINN |u| [mm]", vmax),
                                        (np.abs(pinn_mag-fem_mag), "|error| [mm]", None)]):
        tc = ax.tricontourf(tri, fld, levels=30, cmap="viridis", vmax=vm)
        ax.set_aspect("equal"); plt.colorbar(tc, ax=ax); ax.set_title(ttl)
    fig.suptitle(f"Held-out run {vr['run_id']}: E={vr['E']} MPa, gamma={vr['gamma']}, phi={vr['phi']}, "
                 f"depth={vr['depth']} m, t={vr['t']} m")
    plt.tight_layout(); plt.savefig(f"{args.out_prefix}_heldout.png", dpi=150)

    print(f"\nfinal RMSE  train={hist[-1, 4]:.4f} mm   validation={hist[-1, 5]:.4f} mm")
    print(f"saved {args.out_prefix}_model.pt, _losses.png, _rmse.png, _heldout.png")

"""
Ein TRAINIERTES Tunnelmodell abfragen: Eingaben (E, gamma, phi) rein -> Ausgaben
(Verschiebung) raus, ohne FEM und ohne erneutes Training.

Zuerst ein Modell trainieren + speichern (schreibt <prefix>_model.pt):
    python examples/tunnel_datanet_2D_torch.py --n_steps 8000 --out_prefix results/datanet_8k/datanet_8k

Danach abfragen. Zwei Betriebsarten:

  (a) Einzelpunkt -- Verschiebung an (x, y) ausgeben:
      python examples/tunnel_predict.py --model results/datanet_8k/datanet_8k_model.pt \
             --E 220 --gamma 19.5 --phi 28 --point 25 -30

  (b) Gesamtes Feld -- ParaView-.vtu + PNG fuer diese Parameter schreiben:
      python examples/tunnel_predict.py --model results/datanet_8k/datanet_8k_model.pt \
             --E 220 --gamma 19.5 --phi 28 --out query_E220

E in MPa, gamma in kN/m^3, phi in Grad. Die Werte muessen innerhalb der Bereiche
liegen, auf denen das Modell trainiert wurde (sonst wird eine Warnung ausgegeben --
Extrapolation ist unzuverlaessig).

STANDARD: ohne --model wird results/geo_15k_w05/geo_15k_w05_model.pt verwendet
(Geo-Modell, 15000 Schritte, w_phys 0.5), z.B.:
      python examples/tunnel_predict.py --E 150 --gamma 20 --phi 30 --depth 20 --t 0.3 --ring --mc --out results\abfragen\E150_h20_t03

MODELLE MIT VARIABLER GEOMETRIE (examples/tunnel_datanet_geo.py) werden automatisch
erkannt. Dann braucht es zusaetzlich --depth (Tiefe der Tunnelachse unter GOK [m]) und
--t (Schalendicke [m]):
      python examples/tunnel_predict.py --model results/geo_lauf1/geo_lauf1_model.pt
             --E 120 --gamma 20 --phi 30 --depth 18 --t 0.35 --point 25 -15

NACHLAUF (tunnel_pinn/tunnel_post.py), zusaetzlich zu Punkt oder Feld:
  --ring        Momenten- und Normalkraftbild der Schale (<out>_ring.png + <out>_ring.csv).
                M aus der Ringkinematik (wie FEM mohr_coulomb.py), N aus dem Gleichgewicht
                N = -p R_o mit dem Erddruck p aus dem Bodennetz.
  --mc          Mohr-Coulomb-Ausnutzung im Boden (<out>_mc.png + <out>_mc.vtu),
                Kohaesion mit --c in kPa (Standard 2 kPa wie im FEM-Modell).
  Beispiel:  ... --depth 18 --t 0.35 --ring --mc --c 5 --out abfrage_18m
"""

import argparse
import sys
from pathlib import Path

# Projektwurzel in den Suchpfad legen, damit "import tunnel_pinn..." funktioniert
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from tunnel_pinn.tunnel_data import load_model, predict, export_prediction_vtu, write_vtu
from tunnel_pinn.tunnel_geo import load_model_geo, predict_geo
from tunnel_pinn.tunnel_post import ring_forces_pinn, plot_ring_forces, save_ring_csv, mc_pinn, plot_mc, soil_triangulation


def parse_args():
    "Definiert alle Kommandozeilenoptionen und liest sie ein."
    # description=__doc__ zeigt den Modul-Docstring oben bei --help an
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="results/geo_15k_w05/geo_15k_w05_model.pt",
                   help="path to a saved <prefix>_model.pt (Standard: Geo-Modell, 15000 Schritte, w_phys 0.5)")
    p.add_argument("--E", type=float, required=True, help="ground Young's modulus [MPa]")
    p.add_argument("--gamma", type=float, required=True, help="unit weight [kN/m^3]")
    p.add_argument("--phi", type=float, required=True, help="friction angle [deg]")
    p.add_argument("--depth", type=float, default=None, help="GEO models only: depth of the tunnel axis below the surface [m]")
    p.add_argument("--t", type=float, default=None, help="GEO models only: liner thickness [m]")
    p.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"),
                   help="query a single location (metres); prints displacement and exits")
    p.add_argument("--out", type=str, default="tunnel_query", help="output prefix for the field export (.vtu + .png)")
    p.add_argument("--grid", type=int, default=220, help="field-export grid resolution per axis")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ring", action="store_true", help="liner bending moment / normal force diagram (+ csv)")
    p.add_argument("--mc", action="store_true", help="Mohr-Coulomb utilisation eta in the soil (png + vtu)")
    p.add_argument("--c", type=float, default=2.0, help="soil cohesion [kPa] for --mc (not a network input)")
    return p.parse_args()


def _range_check(cfg, E, gamma, phi, depth=None, t=None):
    """Warnt, wenn ein abgefragter Parameter ausserhalb des trainierten Bereichs liegt.
    Das Programm laeuft trotzdem weiter -- das Ergebnis ist dann aber Extrapolation."""
    checks = [("E", E, cfg.E_min, cfg.E_max),
              ("gamma", gamma, cfg.gamma_min, cfg.gamma_max),
              ("phi", phi, cfg.phi_min, cfg.phi_max)]
    if depth is not None:
        checks += [("depth", depth, cfg.depth_min, cfg.depth_max), ("t", t, cfg.t_min, cfg.t_max)]
    for name, val, lo, hi in checks:
        if not (lo <= val <= hi):
            print(f"  WARNING: {name}={val} is OUTSIDE the trained range [{lo}, {hi}] -- extrapolation, unreliable.")


# Dieser Block laeuft nur, wenn die Datei direkt gestartet wird (nicht beim Importieren)
if __name__ == "__main__":
    args = parse_args()
    # Geo-Modell (variable Tiefe/Schalendicke) oder Modell mit fester Geometrie?
    is_geo = bool(torch.load(args.model, map_location="cpu", weights_only=False).get("geo", False))
    if is_geo:
        if args.depth is None or args.t is None:
            sys.exit("Dieses Modell hat variable Geometrie: bitte --depth (Tiefe Tunnelachse [m]) und --t (Schalendicke [m]) angeben.")
        model, cfg = load_model_geo(args.model, device=args.device)
        # gleiche Aufrufform wie predict() fuer feste Geometrie
        predict = lambda m, xy, E, g, ph: predict_geo(m, xy, E, g, ph, args.depth, args.t)
    else:
        # load_model liefert Modell UND die gespeicherte Konfiguration (Geometrie + Parameterbereiche)
        model, cfg = load_model(args.model, device=args.device)
    print(f"loaded {args.model}" + ("  [variable Geometrie]" if is_geo else ""))
    print(f"trained ranges:  E[{cfg.E_min},{cfg.E_max}] MPa   gamma[{cfg.gamma_min},{cfg.gamma_max}]   phi[{cfg.phi_min},{cfg.phi_max}] deg"
          + (f"   depth[{cfg.depth_min},{cfg.depth_max}] m   t[{cfg.t_min},{cfg.t_max}] m" if is_geo else ""))
    print(f"query:  E={args.E} MPa  gamma={args.gamma}  phi={args.phi} deg"
          + (f"  depth={args.depth} m  t={args.t} m" if is_geo else ""))
    _range_check(cfg, args.E, args.gamma, args.phi, args.depth if is_geo else None, args.t)

    # ---- Nachlauf: Schnittgroessen der Schale und Mohr-Coulomb-Ausnutzung ----
    xc, yc = cfg.center(); Ri = cfg.radius
    t_q = args.t if is_geo else cfg.t_liner                 # Schalendicke dieser Abfrage
    if is_geo:
        yc = -args.depth                                    # Tunnelmittelpunkt aus der abgefragten Tiefe
    par_q = dict(E=args.E, gamma=args.gamma, phi=args.phi)
    if is_geo:
        par_q.update(depth=args.depth, t=args.t)
    geo_txt = f", depth={args.depth} m, t={args.t} m" if is_geo else ""
    if args.ring:
        rs = ring_forces_pinn(model, par_q, xc, yc, Ri, t_q, cfg.E_liner, cfg.nu_liner, cfg.nu)
        plot_ring_forces(rs, xc, yc, f"{args.out}_ring.png",
                         f"PINN-Schnittgrößen: E={args.E} MPa, γ={args.gamma}, φ'={args.phi}°{geo_txt}\n"
                         f"M aus Ringkinematik, N = −p·R_o aus dem Erddruck (Gleichgewicht)")
        save_ring_csv(rs, f"{args.out}_ring.csv")
        print(f"\nring: M {rs['M'].min():.1f} .. {rs['M'].max():.1f} kNm/m   N {rs['N'].min():.0f} .. {rs['N'].max():.0f} kN/m")
        print(f"wrote {args.out}_ring.png and {args.out}_ring.csv")
    if args.mc:
        xs_ = np.linspace(0, cfg.width, args.grid); ys_ = np.linspace(-cfg.total_depth, 0, args.grid)
        XX_, YY_ = np.meshgrid(xs_, ys_)
        pm = np.column_stack([XX_.ravel(), YY_.ravel()]).astype(np.float32)
        pm = pm[(pm[:, 0]-xc)**2+(pm[:, 1]-yc)**2 >= (Ri+t_q)**2]          # nur Boden
        mc = mc_pinn(model, pm, par_q, cfg.nu, args.c)
        plot_mc(pm, mc["eta"], xc, yc, Ri+t_q, f"{args.out}_mc.png",
                f"Mohr-Coulomb-Ausnutzung η (PINN): E={args.E} MPa, γ={args.gamma}, φ'={args.phi}°, c={args.c} kPa{geo_txt}")
        trm = soil_triangulation(pm, xc, yc, Ri+t_q)
        write_vtu(f"{args.out}_mc.vtu", pm, trm.triangles[~trm.mask],
                  {"eta_MC": mc["eta"], "sigma_xx_total_MPa": mc["sxx"], "sigma_yy_total_MPa": mc["syy"], "tau_xy_MPa": mc["sxy"]})
        print(f"\nMohr-Coulomb: eta max = {mc['eta'].max():.2f}   Anteil eta > 1: {(mc['eta'] > 1).mean()*100:.2f} % der Bodenpunkte")
        print(f"wrote {args.out}_mc.png and {args.out}_mc.vtu")

    # ---- (a) Einzelpunkt ----
    if args.point is not None:
        # predict erwartet eine Liste von Punkten; [0] holt das Ergebnis des einen Punktes
        u = predict(model, [args.point], args.E, args.gamma, args.phi)[0]
        print(f"\ndisplacement at (x={args.point[0]}, y={args.point[1]}) m:")
        print(f"  u_x = {u[0]*1e3:+.4f} mm")      # *1e3: Meter -> Millimeter
        print(f"  u_y = {u[1]*1e3:+.4f} mm   ({'heave/up' if u[1] > 0 else 'settlement/down'})")
        print(f"  |u| = {np.hypot(*u)*1e3:.4f} mm")     # np.hypot = sqrt(ux^2+uy^2)
        sys.exit(0)                             # fertig -- Feldexport wird uebersprungen

    # ---- (b) gesamtes Feld ueber das Gebiet ----
    xs = np.linspace(0, cfg.width, args.grid)
    ys = np.linspace(-cfg.total_depth, 0, args.grid)
    XX, YY = np.meshgrid(xs, ys)                # regelmaessiges Auswertungsgitter
    # ravel() macht aus den 2-D-Gittern lange Vektoren, column_stack fuegt sie zu (N,2)
    pts = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)
    keep = (pts[:, 0]-xc)**2 + (pts[:, 1]-yc)**2 >= Ri**2      # den ausgebrochenen Hohlraum verwerfen
    pts = pts[keep]

    u = predict(model, pts, args.E, args.gamma, args.phi)
    mag_mm = np.linalg.norm(u, axis=1)*1e3      # Betrag der Verschiebung je Punkt [mm]
    print(f"\nfield: max|u| = {mag_mm.max():.4f} mm  over {pts.shape[0]:,} points")

    # ParaView-Export (nur PINN-Feld; fuer eine beliebige Abfrage gibt es keine FEM-Referenz)
    if is_geo:      # gleiche .vtu-Felder wie export_prediction_vtu, Hohlraum aus der abgefragten Tiefe
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
        tcx = pts[tri.triangles, 0].mean(1); tcy = pts[tri.triangles, 1].mean(1)
        vtu = write_vtu(f"{args.out}.vtu", pts, tri.triangles[((tcx-xc)**2+(tcy-yc)**2) >= Ri**2],
                        {"u_pinn_m": u, "mag_pinn_mm": mag_mm})
    else:
        vtu = export_prediction_vtu(model, pts, (args.E, args.gamma, args.phi), f"{args.out}.vtu")

    # schnelle PNG-Uebersicht: u_x, u_y und |u| nebeneinander
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (fld, ttl, cm) in zip(axs, [(u[:, 0]*1e3, "u_x [mm]", "RdBu"),     # RdBu = zweifarbig, gut fuer Vorzeichen
                                        (u[:, 1]*1e3, "u_y [mm]", "RdBu"),
                                        (mag_mm, "|u| [mm]", "viridis")]):     # viridis fuer reine Betraege
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=fld, s=3, cmap=cm)
        ax.set_aspect("equal"); plt.colorbar(sc, ax=ax); ax.set_title(ttl)
    fig.suptitle(f"PINN prediction:  E={args.E} MPa, gamma={args.gamma}, phi={args.phi}{geo_txt}  (no FEM, no retraining)")
    plt.tight_layout(); plt.savefig(f"{args.out}.png", dpi=150)
    print(f"wrote {args.out}.vtu (ParaView) and {args.out}.png")

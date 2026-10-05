"""
GRENZWERT-DIAGRAMME aus dem Geo-PINN -- 28.09.2026.

Frage: Welche Parameterkombinationen halten eine zulaessige Verformung am Tunnelrand ein?
Kriterium: maximale Verschiebung |u| an der Tunnelinnenseite (r = R_i), das ist praktisch
immer die Sohlhebung. Grenzwert mit --limit (Standard 10 mm).

Alle fuenf Parameter sind beruecksichtigt: E, gamma, phi', Tiefe H, Schalendicke t.
Zuerst wird fuer ein Raster aus gamma, phi', t und H das erforderliche E_min bestimmt
(kleinstes E, bei dem max|u| <= Grenzwert); alle Diagramme und die Tabelle kommen daraus.

Erzeugt im Ordner --out_dir:
  A_Emin_Matrix.png       Bemessungsdiagramm 3 x 3: Zeilen gamma = 18/20/22, Spalten
                          phi' = 25/30/35; je Feld E_min ueber der Tiefe, Kurven je Schalendicke
  B_Isolinien.png         Isolinienkarte der Verschiebung ueber E und Tiefe, ein Feld je
                          Schalendicke, fuer phi' = --phi_b und gamma = --gamma_b (sichere Seite)
  C_Beiwerte.png          Naeherung E_min ~ E_min,ref(H, t) * k_gamma * k_phi mit
                          Referenz gamma = 20, phi' = 30; Beiwerte je Schalendicke
  Emin_Tabelle.csv        E_min fuer alle Kombinationen (gamma 18..22 in 1er-, phi' 25..35 in
                          2,5-Grad-, t in 0,1-m-, H in 1-m-Schritten)
Mit --fem_data werden in der Isolinienkarte die FEM-Rechnungen als Kontrollpunkte eingezeichnet.

Aufruf (im Ordner PINNs):
  python examples/tunnel_grenzdiagramme.py
  python examples/tunnel_grenzdiagramme.py --limit 8 --out_dir results/grenzdiagramme_8mm
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
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

from tunnel_pinn.tunnel_geo import load_model_geo, _load_runs_geo
from tunnel_pinn.tunnel_post import umax_wall, e_min      # gemeinsam mit webapp/app.py

N_THETA = 72                                        # Punkte auf dem Tunnelrand (5-Grad-Raster)
E_TICKS = [50, 70, 100, 150, 200, 300, 500]
E_TICKS_FINE = [50, 55, 60, 65, 70, 80, 90, 100, 110, 120, 135, 150, 175, 200, 250, 300, 400, 500]
GAMMAS = [18.0, 19.0, 20.0, 21.0, 22.0]
PHIS = [25.0, 27.5, 30.0, 32.5, 35.0]
TS = [0.2, 0.3, 0.4, 0.5]
REF_G, REF_P = 20.0, 30.0                           # Referenzboden fuer die Beiwerte


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/geo_15k_w05/geo_15k_w05_model.pt")
    p.add_argument("--limit", type=float, default=10.0, help="zulaessige max. Verschiebung am Tunnelrand [mm]")
    p.add_argument("--gamma_b", type=float, default=22.0, help="gamma fuer die Isolinienkarte (22 = sichere Seite)")
    p.add_argument("--phi_b", type=float, default=35.0, help="phi' fuer die Isolinienkarte (35 = sichere Seite)")
    p.add_argument("--fem_data", default=str(Path(__file__).resolve().parents[2]/"fenics-soil"/"fem_data_geo"),
                   help="FEM-Datensatz fuer die Kontrollpunkte ('' = keine)")
    p.add_argument("--out_dir", default="results/grenzdiagramme")
    return p.parse_args()


def e_axis(ax, which="y", ticks=E_TICKS):
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(NullLocator())


def emin_table(model, cfg, Es, Hs, limit):
    "EM[g, p, t, h] = E_min fuer GAMMAS x PHIS x TS x Hs."
    Ri, cx = cfg.radius, cfg.width/2
    EM = np.full((len(GAMMAS), len(PHIS), len(TS), len(Hs)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, ph in enumerate(PHIS):
            P = np.array([[E, g, ph, H, t] for t in TS for H in Hs for E in Es])
            U = umax_wall(model, P, Ri, cx).reshape(len(TS), len(Hs), len(Es))
            for k in range(len(TS)):
                EM[i, j, k] = [e_min(U[k, h], Es, limit) for h in range(len(Hs))]
    return EM


if __name__ == "__main__":
    a = parse_args()
    model, cfg = load_model_geo(a.model)
    Ri, cx = cfg.radius, cfg.width/2
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    E_lo, E_hi = cfg.E_min, cfg.E_max
    Es = np.geomspace(E_lo, E_hi, 80)
    Hs = np.linspace(cfg.depth_min, cfg.depth_max, 51)          # 0,5-m-Schritte
    cols = plt.cm.viridis(np.linspace(0.08, 0.85, len(TS)))
    print("berechne E_min fuer alle Kombinationen ...")
    EM = emin_table(model, cfg, Es, Hs, a.limit)
    none = EM <= E_lo*1.001                                     # keine Anforderung
    EMp = np.where(none, np.nan, EM)                            # fuer Linien/Beiwerte

    # ---------------- Tabelle ----------------
    with open(out/"Emin_Tabelle.csv", "w", encoding="utf-8") as f:
        f.write(f"# erforderliches E_min [MPa] fuer max|u| am Tunnelrand <= {a.limit:g} mm (PINN, Modell {a.model})\n")
        f.write(f"# {E_lo:g} = keine Anforderung im trainierten Bereich (jeder Boden ab {E_lo:g} MPa reicht); "
                f"leer = auch {E_hi:g} MPa reicht nicht\n")
        f.write("gamma_kNm3,phi_deg,t_m,depth_m,E_min_MPa\n")
        for i, g in enumerate(GAMMAS):
            for j, ph in enumerate(PHIS):
                for k, t in enumerate(TS):
                    for h, H in enumerate(Hs):
                        if abs(H-round(H)) < 1e-6:
                            v = EM[i, j, k, h]
                            f.write(f"{g:g},{ph:g},{t:g},{H:g},{'' if np.isnan(v) else f'{v:.0f}'}\n")

    # ---------------- A: Matrix 3 x 3 ----------------
    gi = [GAMMAS.index(g) for g in (18.0, 20.0, 22.0)]; pj = [PHIS.index(p) for p in (25.0, 30.0, 35.0)]
    top_val = np.nanmax(EMp[np.ix_(gi, pj)]) if np.isfinite(EMp[np.ix_(gi, pj)]).any() else E_hi
    top = next((tk for tk in E_TICKS_FINE if tk >= 1.1*top_val), E_hi)
    fig, axs = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=True)
    for r, i in enumerate(gi):
        for c, j in enumerate(pj):
            ax = axs[r, c]
            frei = []
            for k, (t, col) in enumerate(zip(TS, cols)):
                y = EMp[i, j, k]
                ax.plot(Hs, y, color=col, lw=2.2, label=f"t = {t:.1f} m")
                if np.all(np.isnan(y)):
                    frei.append(f"{t:.1f}".replace(".", ","))
            if frei:
                ax.text(0.97, 0.96, "keine Anforderung für t = " + " / ".join(frei) + " m", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8.5, color="0.35",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8"))
            ax.axhspan(E_lo, E_lo*1.02, color="0.85", zorder=0)
            ax.set_yscale("log"); ax.set_ylim(E_lo, top); ax.set_xlim(cfg.depth_min-0.5, cfg.depth_max+0.5)
            e_axis(ax, ticks=E_TICKS_FINE); ax.grid(alpha=0.3, which="both")
            ax.set_title(f"γ = {GAMMAS[i]:g} kN/m³,  φ' = {PHIS[j]:g}°", fontsize=11.5)
            if r == 2: ax.set_xlabel("Tiefe der Tunnelachse H [m]")
            if c == 0: ax.set_ylabel("erforderlicher E-Modul E$_{min}$ [MPa]")
            if r == 0 and c == 0: ax.legend(title="Schalendicke", loc="upper left", fontsize=9)
    fig.suptitle(f"Bemessungsdiagramm: erforderlicher E-Modul des Bodens für max. Verschiebung am Tunnelrand ≤ {a.limit:g} mm\n"
                 "Zeilen: Wichte γ   ·   Spalten: Reibungswinkel φ'   ·   Kurven: Schalendicke t", fontsize=13.5)
    fig.text(0.5, 0.005, "Oberhalb einer Kurve (steiferer Boden) ist der Grenzwert eingehalten, unterhalb nicht. Keine Kurve = jeder Boden ab "
             f"{E_lo:g} MPa reicht. Zwischenwerte von γ und φ': Emin_Tabelle.csv oder Beiwerte (C_Beiwerte.png).", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(out/"A_Emin_Matrix.png", dpi=140); plt.close(fig)

    # ---------------- C: Beiwerte ----------------
    ig, ip = GAMMAS.index(REF_G), PHIS.index(REF_P)
    ref = EMp[ig, ip]                                           # (t, h)
    kg = np.array([[np.nanmean(EMp[i, ip, k]/ref[k]) for i in range(len(GAMMAS))] for k in range(len(TS))])
    kp = np.array([[np.nanmean(EMp[ig, j, k]/ref[k]) for j in range(len(PHIS))] for k in range(len(TS))])
    pred = ref[None, None]*kg.T[:, None, :, None]*kp.T[None, :, :, None]
    err = np.abs(pred-EMp)/EMp
    e_mean, e_max = np.nanmean(err)*100, np.nanmax(err)*100
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    for k, (t, col) in enumerate(zip(TS, cols)):
        if np.isfinite(ref[k]).any():
            axs[0].plot(Hs, ref[k], color=col, lw=2.2, label=f"t = {t:.1f} m")
        else:
            axs[0].plot([], [], color=col, lw=2.2, label=f"t = {t:.1f} m  (keine Anforderung)")
        if np.isfinite(kg[k]).all():
            axs[1].plot(GAMMAS, kg[k], "o-", color=col, lw=2, label=f"t = {t:.1f} m")
            axs[2].plot(PHIS, kp[k], "o-", color=col, lw=2, label=f"t = {t:.1f} m")
    axs[0].set_yscale("log"); e_axis(axs[0], ticks=E_TICKS_FINE); axs[0].set_ylim(E_lo, top)
    axs[0].set_title(f"1) Grunddiagramm E$_{{min,ref}}$ (γ = {REF_G:g}, φ' = {REF_P:g}°)"); axs[0].set_xlabel("Tiefe H [m]")
    axs[0].set_ylabel("E$_{min,ref}$ [MPa]"); axs[0].legend(fontsize=8.5, loc="upper left")
    axs[1].set_title("2) Beiwert k$_γ$"); axs[1].set_xlabel("Wichte γ [kN/m³]"); axs[1].axhline(1, color="0.6", lw=0.8)
    axs[2].set_title("3) Beiwert k$_φ$"); axs[2].set_xlabel("Reibungswinkel φ' [°]"); axs[2].axhline(1, color="0.6", lw=0.8)
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    axs[1].legend(fontsize=8.5); axs[1].set_ylabel("Faktor [-]")
    fig.suptitle(f"Beiwert-Verfahren (Näherung):  E$_{{min}}$ ≈ E$_{{min,ref}}$(H, t) · k$_γ$(γ, t) · k$_φ$(φ', t)   "
                 f"für max|u| am Tunnelrand ≤ {a.limit:g} mm", fontsize=13)
    fehlend = [f"{t:.1f}".replace(".", ",") for k, t in enumerate(TS) if not np.isfinite(kg[k]).all()]
    txt = (f"Genauigkeit gegenüber der direkten PINN-Auswertung (Emin_Tabelle.csv): im Mittel {e_mean:.1f} %, "
           f"höchstens {e_max:.1f} %. Für genaue Werte die Tabelle oder die Matrix A benutzen.")
    if fehlend:
        txt += (f"\nFür t ={' / '.join(fehlend)} m hat der Referenzboden keine Anforderung, dort gibt es keine Beiwerte. "
                "Bei ungünstigem γ oder φ' kann trotzdem eine Anforderung entstehen: dafür Matrix A oder Tabelle benutzen.")
    fig.text(0.5, -0.01, txt, ha="center", va="top", fontsize=9.5)
    fig.tight_layout(); fig.savefig(out/"C_Beiwerte.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------------- B: Isolinienkarte ----------------
    runs = _load_runs_geo(a.fem_data) if a.fem_data else []
    for r in runs:
        rad = np.hypot(r["xy"][:, 0]-cx, r["xy"][:, 1]+r["depth"])
        r["umax"] = np.linalg.norm(r["u"][rad < Ri+0.005], axis=1).max()*1e3
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.4), sharey=True)
    for ax, t in zip(axs, TS):
        P = np.array([[E, a.gamma_b, a.phi_b, H, t] for H in Hs for E in Es])
        U = umax_wall(model, P, Ri, cx).reshape(len(Hs), len(Es))
        cs = ax.contourf(Es, Hs, U, levels=np.arange(0, 21, 1.0), cmap="YlOrRd", extend="max")
        cl = ax.contour(Es, Hs, U, levels=[a.limit], colors="k", linewidths=2.4)
        ax.clabel(cl, fmt=lambda v: f"{v:g} mm", fontsize=9)
        for r in [r for r in runs if abs(r["gamma"]-a.gamma_b) < 1e-6 and abs(r["phi"]-a.phi_b) < 1e-6 and abs(r["t"]-t) < 1e-6]:
            ax.plot(r["E"], r["depth"], "o", ms=6, mec="k", mfc="k" if r["umax"] <= a.limit else "white", mew=1.3, zorder=5)
        ax.set_xscale("log"); e_axis(ax, "x"); ax.set_xlim(E_lo, E_hi); ax.set_ylim(cfg.depth_max, cfg.depth_min)
        ax.set_title(f"Schalendicke t = {t:.1f} m", fontsize=12); ax.set_xlabel("E-Modul des Bodens [MPa]")
    axs[0].set_ylabel("Tiefe der Tunnelachse H [m]")
    fig.colorbar(cs, ax=axs, label="max. Verschiebung am Tunnelrand [mm]", fraction=0.025, pad=0.02)
    fig.suptitle(f"Isolinienkarte: max. Verschiebung am Tunnelrand (φ' = {a.phi_b:g}°, γ = {a.gamma_b:g} kN/m³); "
                 f"schwarze Linie = {a.limit:g} mm, links davon überschritten", fontsize=13, y=1.02)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", lw=2.4, label=f"PINN: Grenze {a.limit:g} mm")]
    if runs:
        handles += [Line2D([], [], ls="", marker="o", ms=7, mec="k", mfc="k", mew=1.3,
                           label=f"FEM-Rechnung (γ = {a.gamma_b:g}, φ' = {a.phi_b:g}°): max|u| ≤ {a.limit:g} mm, eingehalten"),
                    Line2D([], [], ls="", marker="o", ms=7, mec="k", mfc="white", mew=1.3,
                           label=f"FEM-Rechnung: max|u| > {a.limit:g} mm, überschritten")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=len(handles), fontsize=10,
               title="Kontrolle: Die Punkte sind echte FEM-Rechnungen. Stimmt das PINN, liegen weiße Punkte links und schwarze rechts der Linie.",
               title_fontsize=10, frameon=True)
    fig.savefig(out/"B_Isolinien.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"Beiwert-Naeherung: mittl. Fehler {e_mean:.1f} %, max {e_max:.1f} %")
    print("k_gamma je t:", {t: np.round(kg[k], 3).tolist() for k, t in enumerate(TS)})
    print("k_phi   je t:", {t: np.round(kp[k], 3).tolist() for k, t in enumerate(TS)})
    print(f"geschrieben: {out}/A_Emin_Matrix.png, B_Isolinien.png, C_Beiwerte.png, Emin_Tabelle.csv")

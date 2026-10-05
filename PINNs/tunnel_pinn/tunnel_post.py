"""
NACHLAUF (Post-Processing) aus dem Verschiebungsfeld -- begonnen 28.09.2026.

1. Schnittgroessen der Schale N(theta), M(theta)
   Genau wie im FEM-Skript fenics-soil/src/mohr_coulomb.py (Abschnitt 11): Verschiebung
   am Mittelkreis R_mid = R_i + t/2 auswerten, in radial w / tangential v zerlegen,
   linearisierte Ringkinematik (Euler-Bernoulli):
       eps_theta = (w + v') / R_mid          dkappa = -(w + w'') / R_mid^2
       N = EA * eps_theta   (+ = Zug)        M = -EI * dkappa = EI (w + w'') / R_mid^2
   VORZEICHEN M (seit 01.10.2026, wie PLAXIS): M > 0 = Zug an der INNENseite der Schale.
   Beispiel Ovalisierung (Ulmen nach aussen): Ulme M < 0 (Zug aussen), First/Sohle M > 0.
   (mohr_coulomb.py gibt M mit umgekehrtem Vorzeichen aus: dort + = Zugfaser aussen.)
   mit E' = E_c/(1-nu_c^2) (ebener Verzerrungszustand), EA = E' t, EI = E' t^3/12.
   Ableitungen nach theta spektral ueber eine tiefpassgefilterte FFT (N_HARM Harmonische).
   Die Funktion bekommt nur eine Verschiebungsfunktion u(xy) -> funktioniert fuer PINN
   UND fuer FEM-Verschiebungen (Vergleich mit derselben Routine).

   NORMALKRAFT BEIM PINN: N aus der Ringkinematik ist beim PINN NICHT brauchbar (RMSE ~900 kN/m
   bei |N| ~1200): EA ~ 1e7 kN/m macht schon 0,05 mm Fehler in der Ringdehnung zu Hunderten kN/m.
   Deshalb kommt N aus dem GLEICHGEWICHT des Rings mit den Bodenspannungen an r = R_o (aus dem
   Bodennetz, Hooke + K0-Primaerspannung):
       sigma_rr (Normalspannung)  und  tau_rt (Schubspannung Boden/Schale)
   Gleichgewicht am Ringelement (Mittellinie R = R_mid, Last an R_o, Q = M'/R):
       tangential:  dN/dtheta = M'/R - R_o * tau_rt          (M nach obiger Konvention)
       radial:      N = R_o * sigma_rr - M''/R
   Verwendet wird die TANGENTIALE Bedingung (Anteile n >= 1, ueber FFT integriert), der
   Mittelwert von N aus der radialen. Pruefung 01.10.2026 gegen 30 FEniCS-Testrechnungen:
       Kesselformel N = -p R_o (bis 30.09.2026):  RMSE 170 kN/m, FALSCHE FORM (First/Ulme vertauscht)
       radial mit M'':                           RMSE 137 kN/m
       tangential (jetzt):                        RMSE  47 kN/m  (~4 % von |N|max)
   Grund: die Kesselformel nimmt den oertlichen Erddruck; die Normalkraft an der Ulme wird
   aber von der Last auf First/Sohle erzeugt (und umgekehrt) -- das liefert erst das Gleichgewicht.
   Das Moment M bleibt aus der Ringkinematik.

2. Mohr-Coulomb-Ausnutzung im Boden (wie mohr_coulomb.py, Abschnitt 9)
       eta = (s1 - s3) / (2 c cos(phi) - (s1 + s3) sin(phi))     (Zug positiv)
   mit GESAMTspannungen: sigma = sigma0 (K0 nach Jaky) + sigma(Delta u) (Hooke, EVZ),
   sigma_zz = lambda (eps_xx + eps_yy) + K0 gamma y. Hauptspannungen 3-D sortiert.
   eta <= 1: elastisch zulaessig, eta > 1: Fliessgrenze ueberschritten.
   Die Kohaesion c ist KEIN Netzeingang (beeinflusst die elastische Verschiebung nicht)
   und wird bei der Auswertung frei angegeben.

Einheiten: Laengen m, Spannungen/Moduln MPa, N in kN/m, M in kNm/m.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
import matplotlib.patches as mpatches

from .tunnel_base import lame, sigma0, k0_jaky, _grad

N_THETA = 360       # Winkelaufloesung am Ring (1-Grad-Raster), wie FEM
N_HARM = 24         # Fourier-Cutoff, wie FEM


# ----------------------------------------------------------------------
# 1. Schnittgroessen der Schale
# ----------------------------------------------------------------------

def spectral_derivs(f, n_harm=N_HARM):
    "1./2. Ableitung von f(theta), theta in [0, 2pi) periodisch, ueber tiefpassgefilterte FFT (wie FEM)."
    F = np.fft.rfft(f)
    k = np.arange(F.size)
    F[n_harm:] = 0.0
    return np.fft.irfft(F*(1j*k), n=f.size), np.fft.irfft(F*(-(k**2)), n=f.size)


def ring_forces(u_fn, cx, cy, Ri, t, E_c, nu_c, n_theta=N_THETA, n_harm=N_HARM):
    """N(theta) [kN/m] und M(theta) [kNm/m] aus einer Verschiebungsfunktion.
    u_fn(xy (n,2) in m) -> u (n,2) in m.  E_c in MPa.
    Rueckgabe: dict(theta [rad], N, M, w, v [m], R_mid)."""
    R_mid = Ri+0.5*t
    theta = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
    pts = np.column_stack([cx+R_mid*np.cos(theta), cy+R_mid*np.sin(theta)])
    u = np.asarray(u_fn(pts), dtype=np.float64)
    w = u[:, 0]*np.cos(theta)+u[:, 1]*np.sin(theta)            # radial, + nach aussen
    v = -u[:, 0]*np.sin(theta)+u[:, 1]*np.cos(theta)           # tangential, + in theta-Richtung
    dv, _ = spectral_derivs(v, n_harm)
    _, d2w = spectral_derivs(w, n_harm)
    Ep = E_c*1e3/(1.0-nu_c**2)                                  # MPa -> kN/m^2, EVZ
    EA, EI = Ep*t, Ep*t**3/12.0                                 # kN/m, kNm^2/m
    N = EA*(w+dv)/R_mid
    M = EI*(w+d2w)/R_mid**2                                    # + = Zug innen (wie PLAXIS)
    return dict(theta=theta, N=N, M=M, w=w, v=v, R_mid=R_mid)


def ring_forces_pinn(model, params, cx, cy, Ri, t, E_c, nu_c, nu_soil, n_theta=N_THETA, n_harm=N_HARM, dr=0.02):
    """Schnittgroessen aus dem PINN: M aus der Ringkinematik (wie FEM), N aus dem
    tangentialen Ringgleichgewicht mit sigma_rr und tau_rt aus dem Bodennetz knapp ausserhalb
    der Schale (r = R_o + dr), siehe Kopf der Datei. params: dict E, gamma, phi (+ depth, t).
    Rueckgabe wie ring_forces, zusaetzlich p (Erddruck, Druck +) und tau [kN/m^2] sowie zur
    Information N_kin (Kinematik) und N_kessel (alte Kesselformel -p R_o)."""
    names = list(params)
    def u_fn(pts):
        dev = next(model.parameters()).device
        x = torch.tensor(np.asarray(pts, np.float32), device=dev)
        p = {k: torch.full((x.shape[0], 1), float(v), device=dev) for k, v in params.items()}
        with torch.no_grad():
            return model(x, p).cpu().numpy()
    res = ring_forces(u_fn, cx, cy, Ri, t, E_c, nu_c, n_theta, n_harm)
    Ro = Ri+t
    th = res["theta"]
    xy = np.column_stack([cx+(Ro+dr)*np.cos(th), cy+(Ro+dr)*np.sin(th)])
    st = mc_pinn(model, xy, params, nu_soil, c_kPa=1.0)          # nur die Spannungen werden gebraucht
    c, s = np.cos(th), np.sin(th)
    srr = (st["sxx"]*c*c+st["syy"]*s*s+2*st["sxy"]*s*c)*1e3                # kN/m^2, Zug +
    trt = ((st["syy"]-st["sxx"])*s*c+st["sxy"]*(c*c-s*s))*1e3             # kN/m^2, Schub r-theta
    R = res["R_mid"]
    dM, _ = spectral_derivs(res["M"], n_harm)
    # tangential: dN/dtheta = g(theta)  ->  N_n = g_n / (i n) fuer n >= 1 (FFT, tiefpassgefiltert)
    G = np.fft.rfft(dM/R-Ro*trt); G[n_harm:] = 0.0
    k = np.arange(G.size); Nh = np.zeros_like(G)
    Nh[1:] = G[1:]/(1j*k[1:])
    Nh[0] = np.fft.rfft(Ro*srr)[0]                                 # Mittelwert aus radial (M'' hat Mittel 0)
    res["N_kin"] = res["N"]
    res["N_kessel"] = Ro*srr
    res["N"] = np.fft.irfft(Nh, n=th.size)
    res["p"] = -srr
    res["tau"] = trt
    return res


def _ring_diagram(ax, theta, values, R_ref, cx, cy, scale, title, unit_label, draw_sign=1.0):
    """Wie ring_diagram in mohr_coulomb.py: graue Ringkontur + radial ausgelenktes,
    vorzeichen-eingefaerbtes Band (rot = +, blau = -), Max/Min beschriftet.
    draw_sign = -1 zeichnet positive Werte nach innen (M: Band liegt so immer auf der Zugseite)."""
    n = theta.size
    ax.plot(R_ref*np.cos(theta)+cx, R_ref*np.sin(theta)+cy, color="0.4", lw=1.0)
    r_out = R_ref+draw_sign*values*scale
    sign = np.where(values >= 0, 1, -1)
    change = np.where(np.diff(sign) != 0)[0]+1
    shift = change[0] if len(change) else 0
    th_r, r_r, s_r = np.roll(theta, -shift), np.roll(r_out, -shift), np.roll(sign, -shift)
    for seg in np.split(np.arange(n), np.where(np.diff(s_r) != 0)[0]+1):
        if len(seg) < 2:
            continue
        th = th_r[seg]; color = "crimson" if s_r[seg[0]] > 0 else "royalblue"
        if len(seg) == n:
            ox, oy = r_r[seg]*np.cos(th)+cx, r_r[seg]*np.sin(th)+cy
            ix, iy = R_ref*np.cos(th)+cx, R_ref*np.sin(th)+cy
            verts = list(zip(ox, oy))+[(ox[0], oy[0])]+list(zip(ix[::-1], iy[::-1]))+[(ix[-1], iy[-1])]
            codes = [Path.MOVETO]+[Path.LINETO]*(len(ox)-1)+[Path.CLOSEPOLY]+[Path.MOVETO]+[Path.LINETO]*(len(ix)-1)+[Path.CLOSEPOLY]
            ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor=color, alpha=0.65, lw=0.8))
        else:
            ax.fill(np.concatenate([R_ref*np.cos(th)+cx, (r_r[seg]*np.cos(th)+cx)[::-1]]),
                    np.concatenate([R_ref*np.sin(th)+cy, (r_r[seg]*np.sin(th)+cy)[::-1]]),
                    facecolor=color, edgecolor=color, alpha=0.65, lw=0.8)
    R_label = 1.65*R_ref
    for i, lbl in [(int(np.argmax(values)), "max"), (int(np.argmin(values)), "min")]:
        xo, yo = r_out[i]*np.cos(theta[i])+cx, r_out[i]*np.sin(theta[i])+cy
        ax.plot(xo, yo, marker="o", ms=4, color="black", zorder=5)
        ax.annotate(f"{lbl}: {values[i]:.0f} {unit_label}\nθ={np.degrees(theta[i]):.0f}°", xy=(xo, yo),
                    xytext=(R_label*np.cos(theta[i])+cx, R_label*np.sin(theta[i])+cy), fontsize=8,
                    ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"), zorder=6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(cx-1.9*R_ref, cx+1.9*R_ref); ax.set_ylim(cy-1.9*R_ref, cy+1.9*R_ref)
    ax.set_title(f"{title}\n(rot = +, blau = –  |  Skala: {1/scale:.3g} {unit_label} pro m Auslenkung)")


def plot_ring_forces(res, cx, cy, path, suptitle):
    "Momenten- und Normalkraftbild nebeneinander (wie schnittgroessen_ring.png der FEM)."
    R = res["R_mid"]
    sM = 0.5*R/(np.max(np.abs(res["M"]))+1e-30); sN = 0.5*R/(np.max(np.abs(res["N"]))+1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _ring_diagram(axes[0], res["theta"], res["M"], R, cx, cy, sM, "Biegemoment  M(θ)  (+ = Zug innen, Band auf Zugseite)",
                  "kNm/m", draw_sign=-1.0)
    _ring_diagram(axes[1], res["theta"], res["N"], R, cx, cy, sN, "Normalkraft  N(θ)", "kN/m")
    fig.suptitle(suptitle); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def save_ring_csv(res, path):
    np.savetxt(path, np.column_stack([np.degrees(res["theta"]), res["N"], res["M"]]),
               header="theta_deg,N_kN_per_m,M_kNm_per_m", delimiter=",", comments="")
    return path


# ----------------------------------------------------------------------
# 2. Mohr-Coulomb-Ausnutzung im Boden
# ----------------------------------------------------------------------

def mc_eta(sxx, syy, sxy, szz, c, phi_deg):
    """Ausnutzungsgrad eta aus GESAMTspannungen (Zug +, MPa), wie mohr_coulomb.py.
    c in MPa. Arrays gleicher Form; Rueckgabe eta (Nenner ~0 -> 0 wie FEM)."""
    s_avg = 0.5*(sxx+syy); s_rad = np.sqrt((0.5*(sxx-syy))**2+sxy**2)
    pr = np.sort(np.stack([s_avg+s_rad, s_avg-s_rad, szz], axis=-1), axis=-1)
    s3, s1 = pr[..., 0], pr[..., 2]
    ph = np.radians(phi_deg)
    den = 2.0*c*np.cos(ph)-(s1+s3)*np.sin(ph)
    return np.where(np.abs(den) > 1e-6, (s1-s3)/np.where(np.abs(den) > 1e-6, den, 1.0), 0.0)


def total_stresses(exx, eyy, exy, x, y, E, nu, gamma, phi_deg):
    """Gesamtspannungen im Boden [MPa]: Hooke (EVZ) aus den Dehnungen + K0-Primaerzustand."""
    lam, mu = lame(E, nu)
    tr = exx+eyy
    sxx = lam*tr+2*mu*exx; syy = lam*tr+2*mu*eyy; sxy = 2*mu*exy; szz = lam*tr
    syy0 = gamma*y/1000.; sxx0 = k0_jaky(phi_deg)*syy0            # wie tunnel_base.sigma0
    return sxx+sxx0, syy+syy0, sxy, szz+sxx0                       # sigma_zz,0 = K0 gamma y (wie FEM)


def mc_pinn(model, xy, params, nu, c_kPa, chunk=20000):
    """eta an Bodenpunkten xy (n,2) aus dem Bodennetz. params: dict name -> float
    (E, gamma, phi und bei Geo-Modellen depth, t). Rueckgabe dict(eta, sxx, syy, sxy, szz)."""
    dev = next(model.parameters()).device
    out = {k: [] for k in ["eta", "sxx", "syy", "sxy", "szz"]}
    for a in range(0, len(xy), chunk):
        x = torch.tensor(np.asarray(xy[a:a+chunk], np.float32), device=dev).requires_grad_(True)
        p = {k: torch.full((x.shape[0], 1), float(v), device=dev) for k, v in params.items()}
        u = model.forward_soil(x, p)
        dux, duy = _grad(u[:, 0:1], x), _grad(u[:, 1:2], x)
        exx = dux[:, 0].detach().cpu().numpy(); eyy = duy[:, 1].detach().cpu().numpy()
        exy = 0.5*(dux[:, 1]+duy[:, 0]).detach().cpu().numpy()
        X = xy[a:a+chunk]
        sxx, syy, sxy, szz = total_stresses(exx, eyy, exy, X[:, 0], X[:, 1], params["E"], nu, params["gamma"], params["phi"])
        eta = mc_eta(sxx, syy, sxy, szz, c_kPa/1000., params["phi"])
        for k, v in zip(out, [eta, sxx, syy, sxy, szz]):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def soil_triangulation(xy, cx, cy, Ro):
    "Dreiecksnetz ueber Punkte, Dreiecke innerhalb r < R_o (Schale + Hohlraum) ausgeblendet."
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    gx = xy[tri.triangles, 0].mean(1); gy = xy[tri.triangles, 1].mean(1)
    tri.set_mask((gx-cx)**2+(gy-cy)**2 < Ro**2)
    return tri


def mc_fem(xy, u, cx, cy, Ro, E, nu, gamma, phi_deg, c_kPa):
    """eta aus einem FEM-Verschiebungsfeld (Knoten xy, u): Dehnungen je Dreieck aus der
    linearen Interpolation (konstant je Dreieck, wie DG0 in der FEM), ausgewertet am
    Dreiecksschwerpunkt. Rueckgabe (Schwerpunkte (m,2), eta (m,))."""
    tri = soil_triangulation(xy, cx, cy, Ro)
    T = tri.triangles[~tri.mask]
    P = xy[T]                                           # (m,3,2)
    U = u[T]                                            # (m,3,2)
    # Gradient einer linearen Funktion auf dem Dreieck: [[x1-x0, y1-y0],[x2-x0, y2-y0]] g = [f1-f0, f2-f0]
    A = np.stack([P[:, 1]-P[:, 0], P[:, 2]-P[:, 0]], axis=1)          # (m,2,2)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    A, P, U = A[ok], P[ok], U[ok]
    rhs = lambda k: np.stack([U[:, 1, k]-U[:, 0, k], U[:, 2, k]-U[:, 0, k]], axis=1)[..., None]   # (m,2,1)
    gux = np.linalg.solve(A, rhs(0))[..., 0]            # (du_x/dx, du_x/dy) je Dreieck
    guy = np.linalg.solve(A, rhs(1))[..., 0]
    exx, eyy, exy = gux[:, 0], guy[:, 1], 0.5*(gux[:, 1]+guy[:, 0])
    g = P.mean(1)
    sxx, syy, sxy, szz = total_stresses(exx, eyy, exy, g[:, 0], g[:, 1], E, nu, gamma, phi_deg)
    return g, mc_eta(sxx, syy, sxy, szz, c_kPa/1000., phi_deg)


def plot_mc(xy, eta, cx, cy, Ro, path, title, vmax=None):
    "Farbbild der Ausnutzung eta; Kontur eta = 1 schwarz (Fliessgrenze)."
    tri = soil_triangulation(xy, cx, cy, Ro)
    vmax = vmax or max(1.2, float(np.nanpercentile(eta, 99.5)))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (lo, hi, sub) in zip(axs, [((0, 50), (-50, 0), "ganzes Gebiet"),
                                       ((cx-4*Ro, cx+4*Ro), (cy-4*Ro, cy+4*Ro), "Ausschnitt Tunnel")]):
        tc = ax.tricontourf(tri, np.clip(eta, 0, vmax), levels=np.linspace(0, vmax, 25), cmap="viridis")
        if np.nanmax(eta) > 1:
            ax.tricontour(tri, eta, levels=[1.0], colors="k", linewidths=1.2)
        ax.add_patch(plt.Circle((cx, cy), Ro, color="0.6"))
        ax.set_xlim(*lo); ax.set_ylim(*hi); ax.set_aspect("equal"); ax.set_title(sub)
        plt.colorbar(tc, ax=ax, label="η  (> 1: Fließgrenze überschritten)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 3. Grenzwert-Auswertung: max. Verschiebung am Tunnelrand und erforderliches E
#    (benutzt von examples/tunnel_grenzdiagramme.py und webapp/app.py)
# ----------------------------------------------------------------------

def umax_wall(model, P, Ri, cx, n_theta=72, chunk=400_000):
    """max |u| [mm] an der Tunnelinnenseite (r = R_i) fuer viele Kombinationen.
    P: (k,5) Spalten E, gamma, phi, depth, t (Geo-Modell). 72 Punkte = 5-Grad-Raster."""
    from .tunnel_geo import predict_rows_geo
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    P = np.asarray(P, dtype=np.float64); k = len(P)
    xy = np.stack([cx+Ri*np.cos(th)[None, :].repeat(k, 0), -P[:, 3:4]+Ri*np.sin(th)[None, :]], -1).reshape(-1, 2)
    par = np.repeat(P, n_theta, 0)
    out = []
    with torch.no_grad():
        for a in range(0, len(xy), chunk):
            out.append(predict_rows_geo(model, torch.tensor(xy[a:a+chunk], dtype=torch.float32),
                                        torch.tensor(par[a:a+chunk], dtype=torch.float32)).numpy())
    u = np.concatenate(out).reshape(k, n_theta, 2)
    return np.linalg.norm(u, axis=2).max(1)*1e3


def e_min(row, Es, limit):
    """Kleinstes E, bei dem row(E) <= limit (row faellt mit E), log-interpoliert.
    Es[0] = schon das weichste E reicht (keine Anforderung), nan = auch das steifste reicht nicht."""
    ok = np.where(row <= limit)[0]
    if len(ok) == 0:
        return np.nan
    i = ok[0]
    if i == 0:
        return Es[0]
    f = (row[i-1]-limit)/(row[i-1]-row[i])
    return float(np.exp(np.log(Es[i-1])+f*(np.log(Es[i])-np.log(Es[i-1]))))

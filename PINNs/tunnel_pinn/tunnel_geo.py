"""
GEOMETRIE-PARAMETRISCHES hybrides Tunnel-PINN (Weg B, begonnen 28.09.2026).

Wie tunnel_data.train_hybrid (Einstellung E: Gleichgewicht Boden + Schale mit
verformungsbezogener Normierung + Verschiebungs-Uebergang an R_o), aber mit ZWEI
zusaetzlichen Netzeingaengen:
    depth  Tiefe der Tunnelachse unter der Gelaendeoberkante [m]  (Mittelpunkt y = -depth)
    t      Dicke der Betonschale [m]                              (R_o = R_i + t)
Das Netz wird damit zu u = NN(x, y, E, gamma, phi, depth, t).

Unterschied zum bisherigen Modell: Dort sind Tunnelmittelpunkt, R_i und R_o EINE feste
Zahl fuer das ganze Modell (register_buffer in TwoNetTunnelPINN). Hier traegt jeder
Punkt seine eigene Geometrie in params["depth"] und params["t"]; alles, was vom
Mittelpunkt oder von R_o abhaengt (Tunnel-Merkmale, Lage in der Schalendicke, Wahl
Boden-/Schalennetz, Punkteziehen, Uebergangsring, Normierung der Schale), wird pro
Punkt berechnet. Gebiet (50 x 50 m), Ausbruchsradius R_i und Beton bleiben fest.

Nicht enthalten: die Randterme des reinen PINN (Tunnelwand-Last, Aussenraender) --
sie haben im Hybrid geschadet (Versuchsreihe 27.09.2026) und sind hier weggelassen.

Das bestehende Modell (tunnel_data.py, tunnel_twonet.py) bleibt unveraendert.
FEM-Daten dazu: fenics-soil/src/mohr_coulomb_dataexport_geo.py (schreibt depth_m und
t_liner_m in jede .npz). Alte Daten ohne diese Felder werden mit default_depth /
default_t gelesen (zum Testen).
"""

import os
import csv
import math
from dataclasses import dataclass

import numpy as np
import torch

from .tunnel_base import TunnelConfig, lame, _boundary_radius
from .tunnel_twonet import TwoNetTunnelPINN, soil_physics_loss, liner_physics_loss


# ----------------------------------------------------------------------
# Konfiguration: TunnelConfig + Bereiche fuer Tiefe und Schalendicke
# ----------------------------------------------------------------------

@dataclass
class GeoConfig(TunnelConfig):
    depth_min: float = 10.0     # Tiefe der Tunnelachse [m] (10 m unter GOK)
    depth_max: float = 35.0     # ... bis 35 m (= 15 m ueber der Sohle bei 50 m Gebiet)
    t_min: float = 0.2          # Schalendicke [m]
    t_max: float = 0.5

    def param_ranges(self):
        "Wie TunnelConfig, plus depth und t -> das Netz bekommt 5 Parameter-Eingaenge."
        r = super().param_ranges()
        r["depth"] = (self.depth_min, self.depth_max)
        r["t"] = (self.t_min, self.t_max)
        return r


PAR_NAMES = ["E", "gamma", "phi", "depth", "t"]     # Spaltenreihenfolge in den Datenpaketen


def sample_raw_params_geo(n, cfg, device="cpu"):
    "n zufaellige (E, gamma, phi, depth, t), gleichverteilt in den Bereichen der Config; je (n,1)."
    p = {}
    for name, (lo, hi) in cfg.param_ranges().items():
        p[name] = lo+(hi-lo)*torch.rand(n, 1, device=device)
    return p


# ----------------------------------------------------------------------
# Modell: Geometrie pro Punkt statt fest
# ----------------------------------------------------------------------

class GeoTwoNetTunnelPINN(TwoNetTunnelPINN):
    """Zwei-Netz-Modell wie TwoNetTunnelPINN; Netze, _pnorm und Ausgabe-Skalierung werden
    geerbt (durch GeoConfig.param_ranges haben beide Netze automatisch 2 Eingaenge mehr).
    Ueberschrieben wird alles, was den Tunnelmittelpunkt oder R_o benutzt."""

    def __init__(self, cfg: GeoConfig, u_sd, device="cpu"):
        super().__init__(cfg, u_sd=u_sd, device=device)
        self.cx = float(cfg.width/2)        # Tunnelachse immer in Gebietsmitte (x)

    def _polar(self, x, params):  #wo liegt ein punkt relativ zu tunnelachse
        "(dx, dy, r, R_o) pro Punkt: Mittelpunkt (cx, -depth), R_o = R_i + t."
        dx = x[:, 0:1]-self.cx
        dy = x[:, 1:2]+params["depth"]      # y - (-depth)
        r = torch.sqrt(dx*dx+dy*dy).clamp_min(1e-9)
        Ro = self.Ri+params["t"]
        return dx, dy, r, Ro

    def _radial_features_geo(self, x, params):
        "Tunnel-Merkmale (wie _radial_features), aber mit Mittelpunkt und R_o des Punktes."
        dx, dy, r, Ro = self._polar(x, params)
        rho = Ro/r
        c, s = dx/r, dy/r
        c2, s2 = c*c-s*s, 2*c*s
        rho2 = rho*rho
        return torch.cat([rho, rho2, c2, s2, rho2*c2, rho2*s2], dim=1)

    def forward_soil(self, x, params):  #berechnet verschiebung für einen punkt
        xn = 2*(x-self.xmin)/(self.xmax-self.xmin)-1        # lage im gebiet [-1,1]
        inp = torch.cat([xn, self._radial_features_geo(x, params), self._pnorm(params)], dim=1)  #hängt alles zusammen (lage, lage rel. tunnel u. ver. parameter)
        return self.soil_net(inp)*self.u_sd #liefert2 dimensionslose dim auf 1. u_sd rechnet in meter um

    def forward_liner(self, x, params):
        dx, dy, r, Ro = self._polar(x, params)
        rho_n = 2*(r-self.Ri)/(Ro-self.Ri)-1               # [-1,1] ueber die JEWEILIGE Dicke
        inp = torch.cat([rho_n, dx/r, dy/r, self._pnorm(params)], dim=1)
        return self.liner_net(inp)*self.u_sd

    def forward(self, x, params): #liefert welches netz zuständig ist für ein bereich und somit die verformung delta x+y
        _, _, r, Ro = self._polar(x, params)
        return torch.where(r >= Ro, self.forward_soil(x, params), self.forward_liner(x, params))


# ----------------------------------------------------------------------
# Kollokationspunkte mit Geometrie pro Punkt
# ----------------------------------------------------------------------

def _xbounds(cfg, device):
    xmin = torch.tensor([0., -cfg.total_depth], dtype=torch.float32, device=device)
    xmax = torch.tensor([cfg.width, 0.], dtype=torch.float32, device=device)
    return xmin, xmax


def sample_soil_geo(n, cfg, device="cpu"):
    """n Bodenpunkte, jeder mit eigenem (E, gamma, phi, depth, t). Verteilung wie sample_soil:
    halb flaechengleich, halb logarithmisch zum Tunnel verdichtet, r von R_o bis zum Gebietsrand
    in Richtung theta. Rueckgabe (x (n,2), params). n = n _physic = 2000) """
    p = sample_raw_params_geo(n, cfg, device)
    xmin, xmax = _xbounds(cfg, device)
    cx, cy = cfg.width/2, -p["depth"][:, 0]   #koordianten von tunnelachse
    Ro = cfg.radius+p["t"][:, 0]        # außenkante tunnelradius
    theta = torch.rand(n, device=device)*2*math.pi    #zufälliger winkel
    rho = torch.rand(n, device=device)  ##zufälliger hilfszahl [0,1)
    R = _boundary_radius(theta, xmin, xmax, (cx, cy))   # Abstand bis Rand
    r_area = torch.sqrt(Ro*Ro+rho*(R*R-Ro*Ro))
    r_geo = Ro*(R/Ro)**rho
    half = n//2
    r = torch.cat([r_area[:half], r_geo[half:]])
    x = torch.stack([cx+r*torch.cos(theta), cy+r*torch.sin(theta)], dim=-1)
    return x, p


def sample_annulus_geo(n, cfg, device="cpu"):
    "n Punkte in der Schale R_i <= r < R_o (jeweils eigene Tiefe und Dicke). Rueckgabe (x, params)."
    p = sample_raw_params_geo(n, cfg, device)
    cx, cy = cfg.width/2, -p["depth"][:, 0]
    theta = torch.rand(n, device=device)*2*math.pi
    r = cfg.radius+p["t"][:, 0]*torch.rand(n, device=device)
    return torch.stack([cx+r*torch.cos(theta), cy+r*torch.sin(theta)], dim=-1), p


def sample_ring_geo(n, cfg, device="cpu"):
    "n Punkte genau auf r = R_o (Grenzflaeche Boden/Schale) mit eigener Geometrie."
    p = sample_raw_params_geo(n, cfg, device)
    cx, cy = cfg.width/2, -p["depth"][:, 0]
    theta = torch.rand(n, device=device)*2*math.pi
    r = cfg.radius+p["t"][:, 0]
    return torch.stack([cx+r*torch.cos(theta), cy+r*torch.sin(theta)], dim=-1), p


# ----------------------------------------------------------------------
# FEM-Daten mit Geometrie laden
# ----------------------------------------------------------------------

def _load_runs_geo(data_dir, default_depth=25.0, default_t=0.2):
    """Wie tunnel_data._load_runs, liest zusaetzlich depth_m und t_liner_m aus jeder .npz.
    Fehlen die Felder (alte Daten), werden default_depth / default_t eingesetzt."""
    runs = []
    with open(os.path.join(data_dir, "manifest.csv"), newline="") as f:
        for r in csv.DictReader(f):
            d = np.load(os.path.join(data_dir, r["file"].replace("/", os.sep)))
            runs.append(dict(
                run_id=int(r["run_id"]),
                xy=np.asarray(d["xy"], dtype=np.float32),
                u=np.asarray(d["u"], dtype=np.float32),
                E=float(d["E_MPa"]), gamma=float(d["gamma_kNm3"]), phi=float(d["phi_deg"]),
                depth=float(d["depth_m"]) if "depth_m" in d.files else float(default_depth),
                t=float(d["t_liner_m"]) if "t_liner_m" in d.files else float(default_t),
            ))
    if not runs:
        raise RuntimeError(f"No runs found in {data_dir}")
    return runs


def _pack_geo(runs, device):
    "Laeufe zu flachen Tensoren; par-Spalten = PAR_NAMES (E, gamma, phi, depth, t)."
    XY, PAR, U = [], [], []
    for rr in runs:
        n = rr["xy"].shape[0]
        XY.append(rr["xy"]); U.append(rr["u"])
        PAR.append(np.tile([rr[k] for k in PAR_NAMES], (n, 1)).astype(np.float32))
    t = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=device)
    return {"xy": t(XY), "par": t(PAR), "u": t(U)}


def load_split_geo(data_dir, val_fraction=0.2, seed=0, device="cpu", n_train_runs=None,
                   default_depth=25.0, default_t=0.2):
    """Laufweise Aufteilung wie tunnel_data.load_split. n_train_runs: nur so viele
    Trainingslaeufe (zufaellig, Seed = seed+1000). Liefert (train, val, param_ranges, info);
    param_ranges enthaelt nur E, gamma, phi (aus ALLEN Laeufen) -- depth und t kommen aus der Config."""
    runs = _load_runs_geo(data_dir, default_depth, default_t)
    rng = np.random.default_rng(seed)  #seed = 0-> immer die selbe mischung
    order = rng.permutation(len(runs))  #zahlen von 0-1079 (anzahl FEM Daten) zufällige reihenfolge
    n_val = max(1, int(round(val_fraction*len(runs))))
    val_pos = set(order[:n_val].tolist())
    train_runs = [runs[i] for i in range(len(runs)) if i not in val_pos]
    val_runs = [runs[i] for i in range(len(runs)) if i in val_pos]
    if n_train_runs is not None and n_train_runs < len(train_runs):
        pick = np.random.default_rng(seed+1000).permutation(len(train_runs))[:n_train_runs]
        train_runs = [train_runs[i] for i in sorted(pick)]
    rng_of = lambda k: (min(r[k] for r in runs), max(r[k] for r in runs))
    param_ranges = {k: rng_of(k) for k in ["E", "gamma", "phi"]}
    info = dict(n_runs=len(runs), n_train=len(train_runs), n_val=len(val_runs),
                train_ids=sorted(r["run_id"] for r in train_runs),
                val_ids=sorted(r["run_id"] for r in val_runs), val_runs=val_runs,
                depths=sorted({r["depth"] for r in runs}), ts=sorted({r["t"] for r in runs}))
    return _pack_geo(train_runs, device), _pack_geo(val_runs, device), param_ranges, info


def params_dict_geo(par_rows):
    "(n,5)-Tensor -> Dictionary {E, gamma, phi, depth, t}, je (n,1)."
    return {k: par_rows[:, i:i+1] for i, k in enumerate(PAR_NAMES)}


def predict_rows_geo(model, xy, par):
    """Verschiebung an xy mit punktweisen Parametern par (n,5). Wie tunnel_data._predict:
    jedes Netz nur auf seinen eigenen Punkten (Schale: r < R_o des jeweiligen Punktes)."""
    p = params_dict_geo(par)
    _, _, r, Ro = model._polar(xy, p)
    is_lin = (r < Ro).squeeze(1)
    out = torch.zeros_like(xy)
    if is_lin.any():
        out[is_lin] = model.forward_liner(xy[is_lin], {k: v[is_lin] for k, v in p.items()})
    soil = ~is_lin
    if soil.any():
        out[soil] = model.forward_soil(xy[soil], {k: v[soil] for k, v in p.items()})
    return out


def rmse_mm_geo(model, pack, chunk=200_000):
    """RMSE der Verschiebung ueber ein Datenpaket in mm. Blockweise (chunk Zeilen), damit
    auch Millionen FEM-Knoten nicht den Arbeitsspeicher sprengen."""
    se, n = 0.0, pack["xy"].shape[0]
    with torch.no_grad():
        for a in range(0, n, chunk):
            pred = predict_rows_geo(model, pack["xy"][a:a+chunk], pack["par"][a:a+chunk])
            se += float(((pred-pack["u"][a:a+chunk])**2).sum())
    return (se/(2*n))**0.5*1e3


@torch.no_grad()
def predict_geo(model, xy, E, gamma, phi, depth, t):
    """Abfrage fuer EINE Kombination (E [MPa], gamma [kN/m^3], phi [Grad], depth [m], t [m]).
    xy (N,2) in Metern -> Verschiebung (N,2) in Metern. Punkte im Hohlraum (r < R_i) liefern
    Werte ohne Bedeutung und sollten ausgeblendet werden."""
    dev = next(model.parameters()).device
    xy_t = torch.as_tensor(np.asarray(xy, np.float32).reshape(-1, 2), device=dev)
    par = torch.tensor([[E, gamma, phi, depth, t]], dtype=torch.float32, device=dev).repeat(xy_t.shape[0], 1)
    return predict_rows_geo(model, xy_t, par).cpu().numpy()


def save_model_geo(model, cfg, path):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "u_sd": model.u_sd, "geo": True}, path)
    return path


def load_model_geo(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GeoTwoNetTunnelPINN(ckpt["cfg"], u_sd=ckpt["u_sd"], device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["cfg"]


# ----------------------------------------------------------------------
# Training (Einstellung E, mit Geometrie pro Punkt)
# ----------------------------------------------------------------------

def train_hybrid_geo(cfg: GeoConfig, data_dir, w_data=1.0, w_phys=0.5, n_data=4000, n_phys=2000,
                     val_fraction=0.2, split_seed=0, w_liner_phys=None, w_iface_disp=1.0, n_iface=200,
                     n_liner=320, liner_norm="stiff", n_train_runs=None,
                     default_depth=25.0, default_t=0.2, verbose=True):  #werte wertevon von args. überschrieben. in Tunnel_datanet_geo ändern
    """Hybrides Training mit Tiefe und Schalendicke als Eingaengen. Liefert (model, history, info).
    Loss = w_data*Daten + w_phys*Gleichgewicht Boden + w_lin*Gleichgewicht Schale
           + w_iface_disp*(u_Schale - u_Boden an R_o)^2   (wie Einstellung E)."""
    torch.manual_seed(cfg.seed)
    device = cfg.device
    Ri = cfg.radius
    lam_c, mu_c = cfg.lame_liner()

    # --- Daten zuerst, damit die E/gamma/phi-Bereiche feststehen ---
    train, val, param_ranges, info = load_split_geo(data_dir, val_fraction, split_seed, device,
                                                    n_train_runs, default_depth, default_t)
    cfg.E_min, cfg.E_max = param_ranges["E"]
    cfg.gamma_min, cfg.gamma_max = param_ranges["gamma"]
    cfg.phi_min, cfg.phi_max = param_ranges["phi"]
    out_of_range = [v for v in info["depths"] if not cfg.depth_min <= v <= cfg.depth_max] + \
                   [v for v in info["ts"] if not cfg.t_min <= v <= cfg.t_max]
    if out_of_range:
        print(f"  WARNUNG: Daten ausserhalb der Geometrie-Bereiche der Config: {out_of_range}")

    # --- Referenzgroessen (für den tiefsten Tunnel = grösste Spannungen) ---
    sig_ref = cfg.gamma_max*cfg.depth_max/1000.                 # MPa
    G_mid = ((cfg.E_min+cfg.E_max)/2)/(2*(1+cfg.nu))
    u_sd = Ri*sig_ref/(2*G_mid)                                 # m
    len_soil = max(cfg.width, cfg.total_depth)
    w_lin = w_phys if w_liner_phys is None else w_liner_phys
    if verbose:
        print(f"data: {info['n_runs']} runs -> {info['n_train']} train, {info['n_val']} val (held-out)")
        print(f"  train points: {train['xy'].shape[0]:,}   val points: {val['xy'].shape[0]:,}")
        print(f"  held-out run ids: {info['val_ids']}")
        print(f"  depths in data: {info['depths']}   t in data: {info['ts']}")
        print(f"  ranges: E{param_ranges['E']} gamma{param_ranges['gamma']} phi{param_ranges['phi']} "
              f"depth({cfg.depth_min}, {cfg.depth_max}) t({cfg.t_min}, {cfg.t_max})")
        print(f"  sig_ref={sig_ref:.3f} MPa  u_sd={u_sd*1e3:.2f} mm  w_phys={w_phys} w_liner={w_lin} "
              f"w_iface_disp={w_iface_disp} liner_norm={liner_norm}")

    model = GeoTwoNetTunnelPINN(cfg, u_sd=u_sd, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.lr_decay**(1/max(cfg.n_steps, 1)))

    M = train["xy"].shape[0]
    history = []
    for i in range(cfg.n_steps+1):
        opt.zero_grad()

        # ---- Daten ----
        idx = torch.randint(0, M, (n_data,), device=device)
        pred = predict_rows_geo(model, train["xy"][idx], train["par"][idx])
        l_data = (((pred-train["u"][idx])/u_sd)**2).mean()

        # ---- Gleichgewicht Boden (jeder Punkt mit eigener Geometrie) ----
        x_soil, p_soil = sample_soil_geo(n_phys, cfg, device)
        l_ps = soil_physics_loss(model, x_soil, p_soil, cfg.nu, len_soil, sig_ref)

        # ---- Gleichgewicht Schale; Normierung mit der Dicke DES PUNKTES ----
        x_lin, p_lin = sample_annulus_geo(n_liner, cfg, device)
        if liner_norm == "stiff":   # Residuum/(lam_c+2mu_c) * t^2/u_sd
            l_pl = liner_physics_loss(model, x_lin, p_lin, lam_c, mu_c, p_lin["t"]**2, u_sd*(lam_c+2*mu_c))
        else:                       # Residuum * t/sig_ref
            l_pl = liner_physics_loss(model, x_lin, p_lin, lam_c, mu_c, p_lin["t"], sig_ref)

        # ---- Uebergang Schale/Boden an R_o des Punktes ----
        l_idisp = torch.zeros((), device=device)
        if w_iface_disp > 0:
            x_if, p_if = sample_ring_geo(n_iface, cfg, device)
            l_idisp = (((model.forward_soil(x_if, p_if)-model.forward_liner(x_if, p_if))/u_sd)**2).mean()

        loss = w_data*l_data + w_phys*l_ps + w_lin*l_pl + w_iface_disp*l_idisp  #w physic = w_lin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step(); sched.step()

        if i % cfg.log_every == 0:
            tr = rmse_mm_geo(model, train); va = rmse_mm_geo(model, val)
            history.append((i, loss.item(), l_data.item(), (l_ps+l_pl).item(), tr, va, l_idisp.item()))
            if verbose:
                print(f"[step {i}/{cfg.n_steps}] loss={loss.item():.3e}  data={l_data.item():.3e}  "
                      f"soil={l_ps.item():.3e}  liner={l_pl.item():.3e}  iface_u={l_idisp.item():.3e}  "
                      f"RMSE train={tr:.4f} mm  val={va:.4f} mm")

    return model, history, info

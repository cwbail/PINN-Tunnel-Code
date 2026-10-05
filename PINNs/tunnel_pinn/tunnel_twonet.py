"""
Zwei-Netz-PINN mit Grenzflaechenkopplung (PyTorch) fuer den betonausgekleideten Tunnel.

Motivation
----------
Der erste Ansatz war ein EINZELNES glattes Netz fuer Boden und Betonschale
zusammen, mit ortsabhaengigen Materialeigenschaften (dieses Modell wurde
inzwischen entfernt). Fuer eine steife Schale scheitert das: an der
Boden/Schale-Grenzflaeche (r=R_o) springt die Steifigkeit um
den Faktor ~550 (Beton ~33 GPa gegenueber Boden ~10-500 MPa). Die Kontinuitaet der
Spannungen zwingt dann die DEHNUNG dazu, ueber die Grenzflaeche hinweg um den
Faktor ~550 zu springen. Ein einzelnes C-unendlich-Netz (tanh) kann eine unstetige
Dehnung nicht darstellen -- es verschmiert sie. Dadurch wird die Ausbruchslast nie
von der Schale in den Boden uebertragen, und es entsteht keine Tunnelkonvergenz
(die Verschiebung faellt in eine unphysikalische, glatte globale Hebung zusammen).
Entfernt man die Schale (macht sie so weich wie den Boden), stellt sich sofort
wieder die korrekte, tunnelnahe Setzung ein -- das bestaetigt, dass die
Grenzflaeche das Problem ist und nicht die Physik.

Dieses Modul loest das so, wie es XPINN/cPINN und die FEM tun: ein EIGENES Netz
pro Materialgebiet, gekoppelt ueber einen expliziten Grenzflaechen-Loss.

    soil_net  : gueltig im Boden,        R_o <= r        (parametrisches Boden-E)
    liner_net : gueltig im Kreisring,    R_i <= r < R_o  (fester Beton)

Der Grenzflaechen-Loss bei r=R_o erzwingt
    Verschiebungskontinuitaet :  u_soil = u_liner
    Spannungskontinuitaet     :  sigma_soil . n = sigma_liner . n   (radiales n)
Die Dehnung darf dabei frei um das Steifigkeitsverhaeltnis springen, waehrend
Verschiebung und Randspannung stetig bleiben -- genau das C0-Verschiebungs- /
Dehnungssprung-Verhalten der FEM.

Alles Uebrige (K0-/Jaky-Formulierung der Ausbruchslast, Referenzzustand sigma0,
Ausbruchslast an der Tunnellaibung bei R_i, freie Oberflaeche / eingespannte Sohle /
Gleitlager an den Seiten, parametrische Konditionierung auf E/gamma/phi,
Ausgabeskalierung) ist unveraendert geblieben und nutzt die Helfer aus
tunnel_base.py weiter. Die vollstaendige Beschreibung der Physik steht im
Modul-Docstring von tunnel_base.py.

Aufruf:  python examples/tunnel_twonet_2D_torch.py --n_steps 20000
"""

import math

import torch
import torch.nn as nn

# Wiederverwendung aus dem Basis-Modul: Config, Materialgesetz, Referenzspannung,
# Ableitungshelfer und die Sampling-Funktionen fuer Laibung und Aussenraender
from .tunnel_base import (
    TunnelConfig, lame, sigma0, _grad,
    _boundary_radius, sample_tunnel_wall, sample_outer_edges, sample_raw_params,
    )


# ----------------------------------------------------------------------
# Modell: zwei tanh-MLPs (Boden + Schale) mit parametrischer Konditionierung
# ----------------------------------------------------------------------

class _MLP(nn.Module):
    """Glattes (tanh) voll verbundenes Netz -- muss C-unendlich sein, weil der
    Physik-Loss zweite Ableitungen braucht.

    PyTorch-Grundlagen: nn.Module ist die Basisklasse jedes Netzes. In __init__
    werden die Schichten angelegt, in forward() steht, wie die Eingabe durchlaeuft.
    """

    def __init__(self, in_dim, out_dim, hidden, n_layers, last_layer_scale=1e-2):
        super().__init__()
        sizes = [in_dim]+[hidden]*n_layers+[out_dim]    # [Eingang, hidden, ..., Ausgang]
        layers = []
        for i in range(len(sizes)-1):   #len: gibt die anzahl an objekten/zahlen
            layers.append(nn.Linear(sizes[i], sizes[i+1]))  # affine Abbildung y = W x + b
            if i < len(sizes)-2:                            # nach der letzten Schicht KEINE Aktivierung
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # Start mit nahezu null Ausgabe: dann sind auch die Anfangsdehnungen -- und damit
        # die riesigen Betonspannungen (E~34 GPa x Dehnung), die die spannungsbasierten
        # Laibungs-/Grenzflaechen-Losses sonst zu Beginn explodieren lassen -- nahe null.
        # torch.no_grad(): diese Aenderung der Gewichte ist keine Trainingsoperation und
        # soll nicht in den Gradientenverlauf aufgenommen werden.
        with torch.no_grad():
            layers[-1].weight.mul_(last_layer_scale)    # mul_ mit Unterstrich = an Ort und Stelle
            layers[-1].bias.mul_(last_layer_scale)

    def forward(self, x):
        return self.net(x)


class TwoNetTunnelPINN(nn.Module):
    """Zwei gekoppelte Netze: soil_net fuer den Boden, liner_net fuer den Beton-Kreisring.
    Beide sind auf die Bodenparameter (E, gamma, phi) konditioniert. forward() waehlt
    zur Auswertung pro Punkt das passende Netz; im Training werden forward_soil /
    forward_liner direkt auf gebietsspezifischen Punkten aufgerufen."""

    def __init__(self, cfg: TunnelConfig, u_sd, device="cpu"):
        super().__init__()
        self.param_ranges = cfg.param_ranges() #gibt an welche param sich ändern aus tunnel_base datei
        self.param_names = list(self.param_ranges.keys()) #name von der parametrische angaben
        n_p = len(self.param_names) #anzahl an bodenparameter was jedes netz zusätzlich als eingang bekommmt

        # Bodennetz: kartesische (x,y), normiert auf [-1,1], PLUS 6 tunnelzentrierte
        # radiale/Kirsch-Merkmale (siehe _radial_features), damit es das Abklingen der
        # cos2theta-Ovalisierung um den Hohlraum darstellen kann, + die Parameter
        self.n_radial = 6
        self.soil_net = _MLP(2+self.n_radial+n_p, 2, cfg.hidden, cfg.n_layers)
        # Schalennetz: polar ueber den (duennen) Kreisring -- (rho_n ueber die Dicke, cos, sin) + Parameter. 11,2, 128 (= 64*2), 4 (2 hidden)
        self.liner_net = _MLP(3+n_p, 2, cfg.hidden_liner, cfg.n_layers_liner)

        self.u_sd = float(u_sd)             #zahl aus train wird gein geladen als komma zahl für wenn in torch übergeben
        cx, cy = cfg.center()
        # register_buffer: gehoert zum Modell (wandert mit .to(device), landet im state_dict),
        # ist aber kein trainierbarer Parameter -- passend fuer feste Geometriewerte
        self.register_buffer("center", torch.tensor([cx, cy], dtype=torch.float32))
        self.Ri = float(cfg.radius)                     # Ausbruchsradius
        self.Ro = float(cfg.outer_liner_radius())       # Grenzflaeche Boden/Schale
        self.register_buffer("xmin", torch.tensor([0., -cfg.total_depth], dtype=torch.float32))
        self.register_buffer("xmax", torch.tensor([cfg.width, 0.], dtype=torch.float32))
        self.to(device)                     # Modell auf CPU oder GPU schieben

    def _pnorm(self, params):       #heir werden parametrische werte alle mit range -1 und 1 skaliert somit das nicht die größten zahlen zB E modul das ganze gewicht nimmt. so werden die alle gleich behandelt
        """Skaliert jeden Bodenparameter auf [-1,1]. Ohne diese Normierung wuerden E
        (~hunderte MPa) und phi (~30 Grad) voellig unterschiedlich stark auf das Netz
        wirken und das Training verschlechtern."""
        cols = []   #leere liste für die normierten spalten
        for name in self.param_names:   #name von parametrische angaben
            lo, hi = self.param_ranges[name]    
            mid, half = (lo+hi)/2, (hi-lo)/2    #mitte (also wert 0)
            cols.append((params[name]-mid)/half)    #auf (-1,1) abbilden
        return torch.cat(cols, dim=1)       # zu einem (n, n_params)-Tensor zusammenfuegen

    def _radial_features(self, x):
        """Tunnelzentrierte radiale/Kirsch-Basismerkmale. Die ausbruchsbedingte
        Verschiebung um einen Kreishohlraum ist (Kirsch/Michell) eine Kombination aus
        achsensymmetrischem ~1/r-Abklingen und cos2theta/sin2theta-Ovalisierungsmoden,
        die ein reines kartesisches MLP nur schwer bilden kann (spektraler Bias). Gibt
        man sie direkt als Eingaben mit, kann das Bodennetz die tunnelnahe Ovalisierung
        unmittelbar darstellen. Alle Merkmale sind fuer r>=R_o beschraenkt
        (rho=R_o/r liegt in (0,1])."""
        dx = x[:, 0:1]-self.center[0]       # Slicing 0:1 behaelt die Spaltenform (n,1)
        dy = x[:, 1:2]-self.center[1]
        r = torch.sqrt(dx*dx+dy*dy).clamp_min(1e-9)     # clamp_min verhindert div 0 für rho
        rho = self.Ro/r                                     # ~1/r radiales Abklingen, in (0,1]
        c, s = dx/r, dy/r                                   # cos(theta), sin(theta)
        c2, s2 = c*c-s*s, 2*c*s                             # cos2theta, sin2theta
        rho2 = rho*rho
        return torch.cat([rho, rho2, c2, s2, rho2*c2, rho2*s2], dim=1)

    def forward_soil(self, x, params):
        "Auswertung NUR des Bodennetzes (gueltig fuer r>=R_o)."
        xn = 2*(x-self.xmin)/(self.xmax-self.xmin)-1        # auf [-1,1]^2 normieren
        inp = torch.cat([xn, self._radial_features(x), self._pnorm(params)], dim=1)
        return self.soil_net(inp)*self.u_sd                 # zurueck auf physikalische Groesse

    def forward_liner(self, x, params):
        "Auswertung NUR des Schalennetzes (gueltig fuer R_i<=r<R_o)."
        dx = x[:, 0:1]-self.center[0]
        dy = x[:, 1:2]-self.center[1]
        r = torch.sqrt(dx*dx+dy*dy).clamp_min(1e-9)
        rho_n = 2*(r-self.Ri)/(self.Ro-self.Ri)-1           # [-1,1] ueber die Schalendicke
        # Polare Eingabe: Lage in der Dicke + Umfangsrichtung (cos, sin) -- genau die
        # Koordinaten, in denen das Tragverhalten des Rings beschrieben ist
        inp = torch.cat([rho_n, dx/r, dy/r, self._pnorm(params)], dim=1)    #mehrere tensoren in dim 1 (also nach rechts. 0 wäre nach unten) zusammenkleben
        return self.liner_net(inp)*self.u_sd   # netz wird ausgewertet und dann mal u_sd (constant)

    def forward(self, x, params):  #entscheided welches netz zuständig ist (soild/lining)
        "Auswertung: Boden fuer r>=R_o, Schale fuer r<R_o (den Hohlraum r<R_i sollte der Aufrufer ausmaskieren)."
        dx = x[:, 0:1]-self.center[0]
        dy = x[:, 1:2]-self.center[1]
        r = torch.sqrt(dx*dx+dy*dy) #muss berechnet werden für entscheidung ob boden/lining
        # Achtung: torch.where wertet BEIDE Zweige aus und waehlt dann elementweise --
        # das ist hier gewollt und unproblematisch, kostet aber die doppelte Rechenzeit.
        return torch.where(r >= self.Ro, self.forward_soil(x, params), self.forward_liner(x, params))   #wenn bereich boden r>Ro, dann forward soil berechnen. sonnst liner. ergebnis ist tensor mit u_x und u_y für jeden gitterpunkt


# ----------------------------------------------------------------------
# Geometrie-Sampling (gebietsspezifisch)
# ----------------------------------------------------------------------

def sample_soil(n_theta, n_rho, xmin, xmax, center, Ro, device="cpu"):
    """Kollokationspunkte im Boden: r von R_o bis zum Gebietsrand, FLAECHENGLEICH
    verteilt, damit das Fernfeld sauber abgedeckt ist -- das ist entscheidend, um das
    Gleichgewicht im Fernfeld zu erzwingen (eine geometrische Verdichtung am Tunnel
    unterabtastet das Fernfeld und laesst eine unphysikalische globale Hebung den
    Physik-Loss erfuellen). Die radialen Eingangsmerkmale des Bodennetzes liefern die
    Aufloesung nahe am Tunnel bereits mit, dichte tunnelnahe Punkte sind hier also
    nicht mehr noetig."""
    n = n_theta*n_rho #60**60 = 3600 verschiedene punkte. Tensor mit ergebnis für jedes punkt
    theta = torch.rand(n, device=device)*2*math.pi      # zufaelliger Winkel
    rho = torch.rand(n, device=device)                  # Hilfsvariable in [0,1)
    R = _boundary_radius(theta, xmin, xmax, center)     # Abstand bis zum Rechteckrand. gibt alles an boundary radius weiter. xmin usw hat sample soil selber bekommen.
    # MISCHUNG: halb flaechengleich (Fernfeldabdeckung, fuer das globale Gleichgewicht /
    # Kontrolle der Hebung) + halb geometrisch/log-uniform (Dichte am Tunnel, fuer die
    # Ovalisierung).
    r_area = torch.sqrt(Ro*Ro + rho*(R*R-Ro*Ro))             # flaechengleich
    r_geo = Ro*(R/Ro)**rho                                    # log-uniform (dichter an der Schale)
    half = n//2
    r = torch.cat([r_area[:half], r_geo[half:]])
    # Polar- zurueck in kartesische Koordinaten -> (n,2)
    return torch.stack([center[0]+r*torch.cos(theta), center[1]+r*torch.sin(theta)], dim=-1)


def sample_annulus(n_theta, n_rho, center, Ri, Ro, device="cpu"):
    "Kollokationspunkte in der Schale: r in [R_i, R_o] (duenner Kreisring, gleichverteilt in r)."
    n = n_theta*n_rho   #8*max nt und 20 = 8*60= 480
    theta = torch.rand(n, device=device)*2*math.pi
    r = Ri+(Ro-Ri)*torch.rand(n, device=device)
    return torch.stack([center[0]+r*torch.cos(theta), center[1]+r*torch.sin(theta)], dim=-1) #gibt liste von 480 punkte zufällig gezogen in der schale


def sample_ring(n, center, radius, device="cpu"):
    "Punkte auf dem Kreis r=radius (wird fuer die Grenzflaeche R_o verwendet)."
    theta = torch.rand(n, device=device)*2*math.pi
    return torch.stack([center[0]+radius*torch.cos(theta), center[1]+radius*torch.sin(theta)], dim=-1)


# ----------------------------------------------------------------------
# Loss-Funktionen
# ----------------------------------------------------------------------

    """Homogenes Navier-Residuum div(sigma)=0. Das rohe Residuum hat die Einheit eines
    Spannungsgradienten (MPa/m); wir normieren es mit sig_ref/length_scale (dem
    charakteristischen Spannungsgradienten des jeweiligen Gebiets). Dadurch wird es zum
    LOKALEN SPANNUNGSUNGLEICHGEWICHT relativ zu sig_ref -- dimensionslos und, ganz
    wichtig, in Boden und Beton gleich empfindlich. (Wuerde man stattdessen mit
    (lam+2mu) normieren, verschwaende ein realer Gleichgewichtsfehler in MPa-Groesse im
    steifen Beton; die Schale bliebe nahezu starr und wuerde sich um den Faktor ~100 zu
    wenig verformen.)

    `forward_fn` ist entweder model.forward_soil oder model.forward_liner -- in Python
    kann man Methoden wie normale Werte herumreichen.
    """
def _navier_loss(forward_fn, x, params, lam, mu, length_scale, sig_ref):
    x = x.clone().requires_grad_(True)      # Kopie, nach der autograd ableiten darf
    u = forward_fn(x, params)   #netz liefert verschiebung
    ux, uy = u[:, 0:1], u[:, 1:2]       #verformung trennen in tensor ux, uy
    dux, duy = _grad(ux, x), _grad(uy, x)   # erste Ableitungen. wird von pytorch abgeleitet. wie ux ändert wenn x bzw y wenig ändert
    ux_x, ux_y = dux[:, 0:1], dux[:, 1:2] #liefert dehnung
    uy_x, uy_y = duy[:, 0:1], duy[:, 1:2] #zweite Ableitungen- (wie dehnungen ort zu ort ändert). spannungen ändert sich auch (bestimmt ob im gleichgewicht)
    ux_xx = _grad(ux_x, x)[:, 0:1]; ux_yy = _grad(ux_y, x)[:, 1:2]; ux_xy = _grad(ux_x, x)[:, 1:2]
    uy_xx = _grad(uy_x, x)[:, 0:1]; uy_yy = _grad(uy_y, x)[:, 1:2]; uy_xy = _grad(uy_x, x)[:, 1:2]
    scale = length_scale/sig_ref            # macht das Residuum dimensionslos
    res_x = ((lam+2*mu)*ux_xx + mu*ux_yy + (lam+mu)*uy_xy)*scale
    res_y = ((lam+2*mu)*uy_yy + mu*uy_xx + (lam+mu)*ux_xy)*scale
    return (res_x**2).mean() + (res_y**2).mean()    # mittlerer quadratischer Fehler -> Skalar


def _strains_of(forward_fn, x, params):
    """Liefert (u, exx, eyy, exy) fuer forward_fn an den Punkten x (x wird intern
    kopiert und auf requires_grad gesetzt).

    exx = du_x/dx, eyy = du_y/dy, exy = 0.5*(du_x/dy + du_y/dx) -- die Komponenten des
    linearisierten Verzerrungstensors.
    """
    x = x.clone().requires_grad_(True)
    u = forward_fn(x, params)
    ux, uy = u[:, 0:1], u[:, 1:2]
    dux, duy = _grad(ux, x), _grad(uy, x)
    exx = dux[:, 0:1]; eyy = duy[:, 1:2]; exy = 0.5*(dux[:, 1:2]+duy[:, 0:1])
    return u, exx, eyy, exy


def _radial_stress(sxx, syy, sxy, cos, sin):
    """Liefert (sigma_rr, sigma_r_theta) aus den kartesischen Spannungen und der
    radialen Richtung (cos,sin) -- also die Drehung des Spannungstensors in
    Polarkoordinaten. sigma_rr ist die Normalspannung auf einer Flaeche mit radialer
    Normale, sigma_r_theta die zugehoerige Schubspannung."""
    s_rr = sxx*cos**2 + syy*sin**2 + 2*sxy*sin*cos
    s_rt = (syy-sxx)*sin*cos + sxy*(cos**2-sin**2)
    return s_rr, s_rt


def soil_physics_loss(model, x, params, nu, length_scale, sig_ref):
    "Gleichgewicht im Boden: Lame-Parameter folgen aus dem parametrischen E pro Punkt."
    lam, mu = lame(params["E"], nu)
    return _navier_loss(model.forward_soil, x, params, lam, mu, length_scale, sig_ref)


def liner_physics_loss(model, x, params, lam_c, mu_c, length_scale, sig_ref):
    "Gleichgewicht in der Betonschale: feste Lame-Parameter des Betons."
    return _navier_loss(model.forward_liner, x, params, lam_c, mu_c, length_scale, sig_ref)


def tunnel_wall_loss(model, x, params, lam_c, mu_c, sig_ref):
    "Ausbruchslast bei r=R_i auf dem Schalennetz: sigma(Delta u).n + sigma0.n = 0 (radial+Schub), mit Betonsteifigkeit."
    _, exx, eyy, exy = _strains_of(model.forward_liner, x, params)
    # Hookesches Gesetz (ebener Verzerrungszustand) mit Betonparametern
    sxx = lam_c*(exx+eyy)+2*mu_c*exx
    syy = lam_c*(exx+eyy)+2*mu_c*eyy
    sxy = 2*mu_c*exy
    dx = x[:, 0:1]-model.center[0]; dy = x[:, 1:2]-model.center[1]
    r = torch.sqrt(dx*dx+dy*dy).clamp_min(1e-9); cos, sin = dx/r, dy/r
    s_rr, s_rt = _radial_stress(sxx, syy, sxy, cos, sin)
    # geostatischer Ausgangszustand am selben Punkt, ebenfalls radial gedreht
    sxx0, syy0, sxy0 = sigma0(x[:, 0:1], x[:, 1:2], params["gamma"], params["phi"])
    s_rr0, s_rt0 = _radial_stress(sxx0, syy0, sxy0, cos, sin)
    # Gesamtspannung an der Ausbruchsflaeche soll verschwinden; /sig_ref macht es dimensionslos
    return (((s_rr+s_rr0)/sig_ref)**2).mean() + (((s_rt+s_rt0)/sig_ref)**2).mean()


def interface_loss(model, x, params, nu, lam_c, mu_c, sig_ref):
    """Grenzflaeche Boden/Schale bei r=R_o: Verschiebungskontinuitaet (normiert mit u_sd)
    + Kontinuitaet der radialen Randspannungen (normiert mit sig_ref). Die Dehnung darf
    dabei frei springen -- genau das ist der Punkt des Zwei-Netz-Ansatzes."""
    lam_s, mu_s = lame(params["E"], nu)     # Bodenparameter (pro Punkt)
    # dieselben Punkte, aber einmal mit dem Boden- und einmal mit dem Schalennetz ausgewertet
    u_s, exx_s, eyy_s, exy_s = _strains_of(model.forward_soil, x, params)
    u_l, exx_l, eyy_l, exy_l = _strains_of(model.forward_liner, x, params)

    # Hookesches Gesetz je Seite mit dem jeweils eigenen Material
    sxx_s = lam_s*(exx_s+eyy_s)+2*mu_s*exx_s; syy_s = lam_s*(exx_s+eyy_s)+2*mu_s*eyy_s; sxy_s = 2*mu_s*exy_s
    sxx_l = lam_c*(exx_l+eyy_l)+2*mu_c*exx_l; syy_l = lam_c*(exx_l+eyy_l)+2*mu_c*eyy_l; sxy_l = 2*mu_c*exy_l

    dx = x[:, 0:1]-model.center[0]; dy = x[:, 1:2]-model.center[1]
    r = torch.sqrt(dx*dx+dy*dy).clamp_min(1e-9); cos, sin = dx/r, dy/r
    srr_s, srt_s = _radial_stress(sxx_s, syy_s, sxy_s, cos, sin)
    srr_l, srt_l = _radial_stress(sxx_l, syy_l, sxy_l, cos, sin)

    disp = (((u_s-u_l)/model.u_sd)**2).mean()       # Verschiebungen muessen uebereinstimmen
    trac = (((srr_s-srr_l)/sig_ref)**2).mean() + (((srt_s-srt_l)/sig_ref)**2).mean()    # Randspannungen ebenso
    return disp+trac


def top_surface_loss(model, x, params, nu, sig_ref):
    "Freie Gelaendeoberkante (y=0): spannungsfrei (Bodennetz)."
    x = x.clone().requires_grad_(True)
    lam, mu = lame(params["E"], nu)
    u = model.forward_soil(x, params)
    dux, duy = _grad(u[:, 0:1], x), _grad(u[:, 1:2], x)
    exx, eyy, exy = dux[:, 0:1], duy[:, 1:2], 0.5*(dux[:, 1:2]+duy[:, 0:1])
    syy = lam*(exx+eyy)+2*mu*eyy    # Normalspannung senkrecht zur Oberflaeche
    sxy = 2*mu*exy                  # Schubspannung in der Oberflaeche
    return ((syy/sig_ref)**2).mean() + ((sxy/sig_ref)**2).mean()


def bottom_fixed_loss(model, x, params):
    "Voll eingespannte Sohle: Delta u = 0 (Bodennetz), normiert mit u_sd."
    u = model.forward_soil(x, params)
    return ((u/model.u_sd)**2).mean()


def roller_side_loss(model, x, params, nu, sig_ref):
    "Gleitlager an den Seiten (x=0/width): Delta u_x=0 (normiert mit u_sd) und sigma_xy=0 (Bodennetz)."
    x = x.clone().requires_grad_(True)
    _, mu = lame(params["E"], nu)   # nur der Schubmodul wird gebraucht
    u = model.forward_soil(x, params)
    ux = u[:, 0:1]
    dux, duy = _grad(ux, x), _grad(u[:, 1:2], x)
    exy = 0.5*(dux[:, 1:2]+duy[:, 0:1])
    sxy = 2*mu*exy
    return ((ux/model.u_sd)**2).mean() + ((sxy/sig_ref)**2).mean()


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(cfg: TunnelConfig, verbose=True):
    "Trainiert das Zwei-Netz-PINN mit Grenzflaechenkopplung. Liefert (model, history)."
    torch.manual_seed(cfg.seed)     # fixiert alle Zufallszahlen -> reproduzierbarer Lauf
    device = cfg.device

    # --- Geometrie als Tensoren bereitstellen ---
    center = torch.tensor(cfg.center(), dtype=torch.float32, device=device) #tunnel mitte als tensor
    xmin = torch.tensor([0., -cfg.total_depth], dtype=torch.float32, device=device)
    xmax = torch.tensor([cfg.width, 0.], dtype=torch.float32, device=device)
    Ri, Ro = cfg.radius, cfg.outer_liner_radius()
    lam_c, mu_c = cfg.lame_liner()      # feste Betonparameter

    # charakteristische Skalen (machen jeden Loss dimensionslos und O(1))
    sig_ref = abs(cfg.gamma_max*(-cfg.depth)/1000.)                 # MPa, Vertikalspannung in Tunneltiefe
    G_mid = ((cfg.E_min+cfg.E_max)/2)/(2*(1+cfg.nu))                # mittlerer Schubmodul
    u_sd = Ri*sig_ref/(2*G_mid)                                    # grobe Groessenordnung der Konvergenz für die weitere berechnung weil netzt können schlecht so kleine zahlen lernen. Formel 27.7 Kolymbas. ohne ausbauwiderstand
    # Laengenskala fuer die Normierung des Gleichgewichts (Spannungsgradient): das Residuum
    # wird mit sig_ref/length normiert und liest sich dadurch als physikalischer
    # Volumenkraftfehler. Der BODEN muss die GEBIETSskala nehmen -- das Gleichgewicht muss
    # global gelten, und eine tunnelnahe Skala (2*Ro) laesst ein sich aufsummierendes
    # Ungleichgewicht im Fernfeld winzig erscheinen, sodass eine unphysikalische globale
    # Hebung den Loss erfuellen kann. Die Radialspannung der Schale variiert dagegen
    # tatsaechlich ueber ihre (duenne) Dicke, deshalb behaelt sie t_liner.
    len_soil = max(cfg.width, cfg.total_depth)
    len_liner = cfg.t_liner

    model = TwoNetTunnelPINN(cfg, u_sd=u_sd, device=device)   #baut beide netzt auf mit _init_
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)     # Standard-Optimierer
    lr_gamma = cfg.lr_decay**(1/max(cfg.n_steps, 1))                # pro Schritt noetiger Faktor
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_gamma)

    nt, nr = cfg.n_interior

    def resample():
        """Zieht einen kompletten frischen Satz Kollokationspunkte fuer ALLE Loss-Terme
        plus die zugehoerigen zufaelligen Bodenparameter."""
        x_soil = sample_soil(nt, nr, xmin, xmax, center, Ro, device=device) #zufällige punkte im boden, wo später geprüft wird ob im gleichgewicht. wird in sample_soil weiter gegeben. ergbnis sind 3600 zufällig gezogene punkte mit polarkoordianten als ergebnis
        x_lin = sample_annulus(max(nt, 20), 8, center, Ri, Ro, device=device)
        x_wall = sample_tunnel_wall(cfg.n_tunnel, center, Ri, sampler="uniform", device=device)
        x_iface = sample_ring(cfg.n_interface, center, Ro, device=device)
        left, right, bottom, top = sample_outer_edges(cfg.n_edge, xmin, xmax, sampler="uniform", device=device)  #gibt tuple mit koordinaten: (n*4,2)
        x_sides = torch.cat([left, right], dim=0)  #tuple aus zwei seiten links dann rechts, in dim 0 (vert)
        P = lambda n: sample_raw_params(n, cfg, device)      # Kurzschreibweise (Lambda-Funktion)
        return (x_soil, x_lin, x_wall, x_iface, x_sides, bottom, top,
                P(x_soil.shape[0]), P(x_lin.shape[0]), P(cfg.n_tunnel), P(cfg.n_interface),
                P(2*cfg.n_edge), P(cfg.n_edge), P(cfg.n_edge))  #shape [0]-> nimme shape (3600,2), die erste (0) zahl davon P(3600), E 3600 werte, gamma 3600 werte usw. parameter zu x.soil (pos 1)
    #trainingsschleife-> wo programm lernt
    batch = resample()   #ziegt ersten punkt und parameter aus resample
    history = []    #leigt leere liste an wo loss notiert wird für plotten
    for i in range(cfg.n_steps+1):  #durchlaufen mit anzahl der schritte
        # nur alle resample_every Schritte neu ziehen -- konstantes Gradientensignal
        if cfg.resample_every and i > 0 and i % cfg.resample_every == 0:
            batch = resample()   #neue punkte erstellen bei i = 10, 20 ,30 usw
        # Tupel wieder auf die einzelnen Namen verteilen (Entpacken)
        (x_soil, x_lin, x_wall, x_iface, x_sides, x_bottom, x_top,
         p_soil, p_lin, p_wall, p_iface, p_sides, p_bottom, p_top) = batch

        optimiser.zero_grad()       # alte Gradienten loeschen (PyTorch summiert sonst auf)
        # --- alle Loss-Anteile auswerten ---
        l_soil = soil_physics_loss(model, x_soil, p_soil, cfg.nu, len_soil, sig_ref)
        l_lin = liner_physics_loss(model, x_lin, p_lin, lam_c, mu_c, len_liner, sig_ref)
        l_wall = tunnel_wall_loss(model, x_wall, p_wall, lam_c, mu_c, sig_ref)
        l_iface = interface_loss(model, x_iface, p_iface, cfg.nu, lam_c, mu_c, sig_ref)
        l_sides = roller_side_loss(model, x_sides, p_sides, cfg.nu, sig_ref)
        l_bottom = bottom_fixed_loss(model, x_bottom, p_bottom)
        l_top = top_surface_loss(model, x_top, p_top, cfg.nu, sig_ref)
        l_phys = l_soil+l_lin
        l_outer = l_sides+l_bottom+l_top
        # Das Schalengleichgewicht bekommt ein eigenes (typischerweise kleineres) Gewicht, damit
        # der steife Ring ovalisieren kann, statt ins triviale Starrkoerper-Minimum zu fallen.
        w_lin = cfg.w_liner_phys if cfg.w_liner_phys >= 0 else cfg.w_phys
        loss = (cfg.w_phys*l_soil + w_lin*l_lin + cfg.w_tunnel*l_wall
                + cfg.w_interface*l_iface + cfg.w_outer*l_outer)
        loss.backward()             # Rueckwaertsdurchlauf: Gradienten nach allen Gewichten
        # Gradientennorm begrenzen, um gelegentliche Ausreisser aus einem unguenstig gezogenen
        # Batch abzufangen. Bewusst grosszuegig (10, nicht ~1): zu enges Clipping bremst das
        # Netz so stark, dass es die grossen (mehrere zehn mm) Verschiebungen der echten
        # Loesung nie aufbauen kann. resample_every=10 entfernt die meisten Spitzen ohnehin.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimiser.step()            # Gewichte aktualisieren
        scheduler.step()            # Lernrate verkleinern

        # .item() holt den reinen Zahlenwert aus dem Tensor (ohne Gradientenverfolgung)
        history.append((i, loss.item(), l_phys.item(), l_wall.item(), l_iface.item(), l_outer.item()))
        if verbose and i % cfg.log_every == 0:
            print(f"[step {i}/{cfg.n_steps}] lr={scheduler.get_last_lr()[0]:.2e}  loss={loss.item():.4e}  "
                  f"soil={l_soil.item():.3e}  liner={l_lin.item():.3e}  wall={l_wall.item():.3e}  "
                  f"iface={l_iface.item():.3e}  outer={l_outer.item():.3e}")
    return model, history


@torch.no_grad()    # Dekorator: keine Gradienten aufzeichnen (schneller, weniger Speicher)
def evaluate_at_ground_params(model, x_batch, E, gamma, phi):
    "Fragt das trainierte Zwei-Netz-Modell an einem Punkte-Batch fuer EIN (E,gamma,phi) ab. Liefert Delta u."
    n = x_batch.shape[0]    #anzahl an punkte
    device = x_batch.device #wo die punkte liegen
    # torch.full((n,1), wert) = Spalte der Laenge n, komplett mit demselben Wert gefuellt
    params = {      #alle punkte die selben boden parameter
        "E": torch.full((n, 1), float(E), device=device),
        "gamma": torch.full((n, 1), float(gamma), device=device),
        "phi": torch.full((n, 1), float(phi), device=device),
        }
    return model(x_batch, params)

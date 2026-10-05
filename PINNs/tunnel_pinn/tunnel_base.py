"""
==========================================================================
BASIS-MODUL: Konfiguration, Physik-Grundlagen, Geometrie-Sampling
==========================================================================
Dieses Modul enthaelt (a) die zentrale Konfigurations-Dataclass
`TunnelConfig`, (b) die elastizitaetstheoretischen Hilfsfunktionen
(Lame-Parameter, K0/Jaky, geostatischer Spannungszustand sigma0), (c) das
Sampling der Kollokationspunkte und (d) den Ableitungshelfer `_grad`.

Hier steht KEIN Modell. Die beiden genutzten Varianten (tunnel_twonet.py =
reiner Vorwaertsloeser mit zwei Netzen, tunnel_data.py = hybrides Training
mit FEM-Daten) importieren ihre Konfiguration und ihre Physik-Helfer aus
DIESER Datei und bringen ihr Modell selbst mit.

Historie: die urspruengliche FBPINN-Gebietszerlegung (ueberlappende
Teilgebiete + Kosinus-Fensterung) wurde beim Wechsel auf das Zwei-Netz-
Modell aufgegeben und ist entfernt. Deshalb heisst diese Datei jetzt
tunnel_base.py und nicht mehr tunnel_fbpinn.py.

--------------------------------------------------------------------------
WO STELLE ICH MEINE PARAMETER EIN?
--------------------------------------------------------------------------
Alles liegt in der `TunnelConfig`-Dataclass weiter unten. Man erzeugt eine
Instanz, ueberschreibt die gewuenschten Felder und uebergibt sie an
`tunnel_twonet.train()` bzw. `tunnel_data.train_hybrid()`. Lauffaehige
Beispiele stehen in examples/.

Es gibt zwei Arten von Parametern:
  - FESTE Konstanten (Radius, Tiefe, Breite/total_depth, nu, Schale E/t/nu,
    Loss-Gewichte, Netz- und Trainings-Hyperparameter): ein Wert, der fuer
    diesen Trainingslauf fest eingestellt ist.
  - PARAMETRISCHE Eingaben (Bodensteifigkeit E, Wichte gamma,
    Reibungswinkel phi'): werden als *Bereiche* angegeben. Das Netz wird
    ueber zufaellig gezogene Werte aus diesem Bereich trainiert und wird
    dadurch zu einer Funktion von (x, y, E, gamma, phi). Nach dem Training
    kann also jeder Wert *innerhalb dieser Bereiche* ohne erneutes Training
    abgefragt werden -- siehe `tunnel_twonet.evaluate_at_ground_params()`.
    Abfragen weit ausserhalb der trainierten Bereiche sind Extrapolation
    und unzuverlaessig.

--------------------------------------------------------------------------
GEOMETRIE UND RANDBEDINGUNGEN (K0- / Ausbruchslast-Methode)
--------------------------------------------------------------------------
Das Gebiet ist ein Halbraum-Rechteck x in [0,width], y in [-total_depth,0],
wobei y=0 die reale Gelaendeoberkante ist. Der Tunnel liegt bei
(width/2, -depth). Das ist der uebliche geotechnische FEM-Aufbau, NICHT das
Kirsch-Problem im unendlichen Gebiet, und besitzt daher KEINE einfache
geschlossene Referenzloesung.

Der geostatische K0-Ausgangsspannungszustand (Jaky) fuer den *ungestoerten*
Boden (ohne Tunnel) lautet
    sigma_yy0(y) = gamma*y            (y<=0, also bereits Druck)
    sigma_xx0(y) = K0*sigma_yy0(y),   K0 = 1-sin(phi')
    sigma_xy0    = 0
Dieser Zustand ist per Konstruktion mit dem Eigengewicht im Gleichgewicht
(div(sigma0) = (0,-gamma)) und erzeugt daher UEBERALL NULL Verschiebung --
er ist der Referenzzustand und nichts, was das Netz lernen muesste.

Das Netz loest fuer Delta u, also die DURCH DEN AUSBRUCH VERURSACHTE
Verschiebung relativ zu diesem verschiebungsfreien Referenzzustand. Weil
sigma0 bereits selbst im Gleichgewicht ist, erfuellt Delta u im gesamten
Boden die schlichte HOMOGENE Navier-Cauchy-Gleichung (kein Volumenkraft-Term
noetig -- siehe `tunnel_twonet.soil_physics_loss()`). Die Stoerung wird an
der Tunnellaibung eingebracht: die Randbedingung an der Ausbruchsflaeche wird auf die
GESAMTspannung gestellt (elastische Antwort des Netzes + sigma0). Der
sigma0-Anteil bleibt dabei als negativer Zielwert stehen -- das ist die
klassische "Ausbruchslast", siehe `tunnel_twonet.tunnel_wall_loss()`.

Aeussere Randbedingungen:
  - oben (y=0, Gelaendeoberkante): frei/natuerlich -- spannungsfrei (fuer
    Delta u also sigma_yy=sigma_xy=0; die sigma0-Randspannung ist dort
    ohnehin exakt null, daher keine Korrektur noetig)
  - unten (y=-total_depth): voll eingespannt, Delta u=(0,0)
  - links/rechts (x=0, x=width): Gleitlager -- Delta u_x=0 (wesentlich)
    und sigma_xy=0 (natuerlich, reibungsfreies Gleiten)
Damit entfaellt der frueher verwendete (falsche) Ansatz, auf allen vier
Aussenraendern ein Kirsch-Verschiebungsfeld des unendlichen Gebiets
vorzugeben.

--------------------------------------------------------------------------
BETON-INNENSCHALE -- explizites Zwei-Material-Gebiet (Boden + Beton)
--------------------------------------------------------------------------
`radius` ist der AUSGEBROCHENE HOHLRAUMRADIUS R_i (das, was man
normalerweise "Tunnelradius" nennt, und was dem R_i des FEM-Referenzmodells
entspricht). Die Schale belegt den Kreisring von R_i bis R_o=R_i+t_liner als
ZWEITES MATERIALGEBIET. Der ausgeschlossene Hohlraum liegt bei R_i (der echte
Hohlraum); die Randbedingung an der Tunnellaibung (spannungsfreie
Ausbruchsflaeche) wird direkt dort gestellt, und zwar mit der Materialantwort
des Betons, als "Ausbruchslast" genau wie im unverkleideten Fall -- einen
separaten Steifigkeitsparameter fuer die Schale gibt es nicht, da ihre
Steifigkeit sich daraus ergibt, dass die Elastizitaetsgleichung in ihrem
eigenen Materialgebiet tatsaechlich geloest wird, genau wie im
FEM-Referenzmodell.

Wie die beiden Materialgebiete gekoppelt werden, steht in tunnel_twonet.py:
je ein eigenes Netz pro Gebiet plus ein expliziter Grenzflaechen-Loss bei
R_o (Verschiebungs- und Spannungskontinuitaet). Nicht erfasst ist das
Eigengewicht der Schale selbst (ein kleiner Sekundaereffekt, den das
FEM-Referenzmodell beruecksichtigt, dieses Modell nicht).
--------------------------------------------------------------------------
"""

import math
from dataclasses import dataclass   # dataclass = Klasse, die vor allem Felder haelt (spart __init__)

import torch                        # PyTorch: Tensoren + automatische Ableitung (autograd)


# ----------------------------------------------------------------------
# Konfiguration -- hier stellt man sein eigenes Tunnelproblem ein
# ----------------------------------------------------------------------

# Einheitenkonvention: Laengen in Metern, Spannungen/Moduln in MPa, Wichte in kN/m^3
# (Meter + MPa sind konsistent: 1 MPa = 1 MN/m^2 = 1000 kN/m^2)

@dataclass
class TunnelConfig:   #tunnel twonet 2D torch cfg. usw
    """Zentrale Sammelstelle aller Einstellungen eines Trainingslaufs.

    Als @dataclass genuegt es, die Felder aufzulisten -- Python erzeugt den
    Konstruktor automatisch. Man ueberschreibt nur, was man aendern will,
    z.B. TunnelConfig(radius=2.5, depth=25.0, n_steps=8000).
    """

    # --- Tunnelgeometrie (fest) ---
    radius: float = 4.0# AUSBRUCHSradius (Hohlraum) R_i [m]
    depth: float = 15.0# Tiefe der Tunnelachse [m] unter der Gelaendeoberkante (y=0)
    # width/total_depth stehen standardmaessig auf 2*depth, damit der Tunnel vertikal genau
    # zwischen freier Oberflaeche und eingespannter Sohle liegt (wie im FEM-Referenzmodell)
    # und ringsum symmetrischen Abstand hat -- das ist NICHT automatisch an depth gekoppelt
    # (es sind einfache Felder, keine abgeleiteten Groessen). Wenn man `depth` aendert, sollte
    # man diese Werte also mitaendern, z.B. TunnelConfig(depth=d, width=2*d, total_depth=2*d)
    width: float = 30.0# horizontale Gebietsausdehnung [m] (x in [0, width], Tunnel bei x=width/2)
    total_depth: float = 30.0# vertikale Gebietsausdehnung [m] (y in [-total_depth, 0])-> Sollte nicht 50m Sein?

    # --- Querdehnzahl des Bodens (fest -- nicht parametrisiert) ---
    nu: float = 0.25

    # --- PARAMETRISCHE Bereiche: ueber diese soll das Netz generalisieren ---
    # (Netzeingang wird zu (x, y, E, gamma, phi); nach dem Training kann jeder Wert
    # innerhalb dieser Bereiche per evaluate_at_ground_params() abgefragt werden, ohne
    # neu zu trainieren)
    E_min: float = 50.# Bereich des E-Moduls des Bodens [MPa] (steifer Boden / schwacher Fels). Pruefen
    E_max: float = 500.
    gamma_min: float = 18.# Bereich der Wichte des Bodens [kN/m^3]
    gamma_max: float = 22.
    phi_min: float = 25.# Bereich des effektiven Reibungswinkels [Grad] -- K0 folgt daraus ueber Jaky
    phi_max: float = 35.

    # --- Beton-Innenschale (fest -- explizites zweites Materialgebiet, R_i bis R_o) ---
    E_liner: float = 33000.# E-Modul der Betonschale [MPa] (C30/37, Ecm nach EC2)
    nu_liner: float = 0.2# Querdehnzahl des Betons
    t_liner: float = 0.5# Schalendicke [m]

    # --- Loss-Gewichte --- wofuer sind die? Sie balancieren die einzelnen Loss-Terme
    # gegeneinander: jeder Term ist bereits dimensionslos normiert, das Gewicht sagt,
    # wie stark der Optimierer ihn im Vergleich zu den anderen ernst nehmen soll.
    w_phys: float = 1.0# Gewicht des Physik-Residuums (Navier-Cauchy)
    w_tunnel: float = 1.0# Gewicht der Randbedingung an der Tunnellaibung
    w_outer: float = 1e2# Gewicht der aeusseren Raender zusammen (oben+unten+Seiten)

    # --- Bodennetz (Architektur des soil_net in tunnel_twonet.py) ---
    hidden: int = 64    # Neuronen pro verdeckter Schicht; evtl. verkleinern?
    n_layers: int = 2   # Anzahl verdeckter Schichten; evtl. erhoehen?

    # --- Schalennetz + Grenzflaeche: siehe tunnel_pinn/tunnel_twonet.py ---
    # Der Beton-Kreisring bekommt ein EIGENES Netz, damit der ~550-fache
    # Dehnungssprung an der Boden/Schale-Grenzflaeche (r=R_o) exakt dargestellt werden
    # kann, statt von einem einzigen glatten Netz verschmiert zu werden. Zusammen mit
    # hidden/n_layers oben beschreiben diese Felder das komplette Modell.
    hidden_liner: int = 32# Breite des Schalennetzes (der Kreisring ist klein -> kleines Netz genuegt)
    n_layers_liner: int = 2# Tiefe des Schalennetzes
    n_interface: int = 80# Kollokationspunkte auf der Grenzflaeche (r=R_o) pro Schritt
    w_interface: float = 1e2# Gewicht der Spannungs- + Verschiebungskontinuitaet an der Grenzflaeche
    w_liner_phys: float = -1.0# Gewicht des Schalen-Gleichgewichts; <0 bedeutet "nimm w_phys". NIEDRIGER
    # als w_phys setzen, damit der steife Betonring OVALISIEREN kann: eine reine Starrkoerper-
    # bewegung erfuellt das Schalengleichgewicht trivial (Dehnung null), ein hohes Gewicht auf
    # dem Schalen-Physikterm faengt den Optimierer also in genau diesem starren Minimum, und der
    # Ring biegt sich nicht so, wie es der echte (FEM-)Betonring tut.

    # --- Training ---

    n_steps: int = 20000            # Anzahl Optimierungsschritte (Iterationen)
    lr: float = 1e-3    # Lernrate des Adam-Optimierers; evtl. erhoehen
    lr_decay: float = 0.1# final_lr/lr ueber das Training (geometrischer Abfall) -- daempft spaetes Oszillieren
    n_interior: tuple = (30, 30)# (n_theta, n_rho) Kollokationspunkte im Inneren pro Schritt (deckt Boden UND Schale ab)
    n_edge: int = 20# Punkte pro Aussenrand (x4 Raender) pro Schritt
    n_tunnel: int = 60# Punkte auf der Tunnellaibung (bei R_i, der Ausbruchsflaeche) pro Schritt  #?
    resample_every: int = 10# alle N Schritte neue Kollokationspunkte + Parameterziehungen (>1 gibt dem
    # Optimierer mehrere aufeinanderfolgende Schritte mit konsistentem Gradientensignal pro Batch,
    # statt bei jedem einzelnen Schritt frisches Rauschen aus der stetigen Verteilung -- Resampling
    # bei jedem Schritt hat dauerhafte Loss-Spitzen erzeugt, die nie abgeklungen sind)
    log_every: int = 1000           # alle N Schritte eine Statuszeile ausgeben
    seed: int = 0                   # Zufallsstartwert -> reproduzierbare Laeufe
    device: str = "cpu"             # "cpu" oder "cuda" (GPU)

    def outer_liner_radius(self):
        "Grenzflaechenradius Boden/Schale R_o -- Grenze zwischen den beiden Materialgebieten"
        return self.radius+self.t_liner

    def center(self):
        "Tunnelmittelpunkt (x, y) im Gebietskoordinatensystem (y negativ = unter GOK)"
        return (self.width/2, -self.depth)

    def param_ranges(self):
        "Die parametrischen Bereiche als Dictionary -- in dieser Form erwartet sie die Netz-Normierung"
        return {
            "E": (self.E_min, self.E_max),
            "gamma": (self.gamma_min, self.gamma_max),
            "phi": (self.phi_min, self.phi_max),
            }

    def lame_liner(self):   # Materialphysik der Schale
        "Feste Lame-Parameter (lambda, mu) des Betonmaterials der Schale"
        return lame(self.E_liner, self.nu_liner)


def lame(E, nu):    # Hilfsfunktion fuer die Physik
    """Lame-Parameter (lambda, mu) aus E-Modul E und Querdehnzahl nu, ebener Verzerrungszustand.

    E darf ein einfacher Python-Float ODER ein Torch-Tensor sein: die Formeln sind
    reine Arithmetik und funktionieren fuer beides (bei Tensoren elementweise). Genau
    deshalb kann hier pro Punkt ein anderes E stehen (parametrischer Boden).
    mu ist der Schubmodul G, lambda der erste Lame-Parameter.
    """
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    return lam, mu


def _sin_deg(phi_deg):  # rechnet Grad in Bogenmass um
    """sin(phi_deg), phi_deg in GRAD -- funktioniert fuer Python-Floats und fuer Torch-Tensoren.

    Die Fallunterscheidung ist noetig, weil math.sin() keine Tensoren kann und
    torch.sin() keine reinen Python-Zahlen mit Gradientenverfolgung braucht.
    """
    if torch.is_tensor(phi_deg):
        return torch.sin(phi_deg*(math.pi/180))
    return math.sin(math.radians(phi_deg))


def k0_jaky(phi_deg):   # Berechnung des K0-Werts
    "Formel von Jaky (1944): Erdruhedruckbeiwert K0 fuer normalkonsolidierten Boden"
    return 1-_sin_deg(phi_deg)


def sigma0(x, y, gamma, phi_deg):
    """Geostatischer K0-Ausgangsspannungszustand (sigma_xx0, sigma_yy0, sigma_xy0) [MPa]
    des UNGESTOERTEN (tunnelfreien) Bodens im Punkt (x,y), y<=0 unterhalb der freien
    Oberflaeche y=0. Das ist der Referenzzustand, relativ zu dem die Netzausgabe Delta u
    die ausbruchsbedingte Verschiebung darstellt (siehe Modul-Docstring)."""
    sigma_yy0 = (gamma*y)/1000. # kN/m^2 (kPa) -> MPa; y<=0, daher bereits Druck (negativ)
    sigma_xx0 = k0_jaky(phi_deg)*sigma_yy0  # nutzt die k0_jaky-Funktion, Eingabe = phi_deg
    # Schubspannung ist im K0-Zustand exakt null. Sie muss aber die gleiche Form/den
    # gleichen Typ wie die anderen Komponenten haben, damit man weiterrechnen kann:
    sigma_xy0 = torch.zeros_like(sigma_yy0) if torch.is_tensor(sigma_yy0) else 0.
    return sigma_xx0, sigma_yy0, sigma_xy0


# ----------------------------------------------------------------------
# Geometrie: Sampling des Gebiets "Halbraum minus Kreis" (Tunnelquerschnitt)
# ----------------------------------------------------------------------

def _boundary_radius(theta, xmin, xmax, center):    #theta in sample soil definiert
    """Abstand vom Tunnelmittelpunkt bis zum Rechteckrand in Richtung theta
    (Strahl-Rechteck-Schnitt).

    Dadurch laesst sich das Rechteck-minus-Kreis-Gebiet in Polarkoordinaten
    beschreiben: fuer jede Richtung theta laeuft der Radius von der Tunnelwand bis
    genau zu diesem R(theta).
    """
    cos, sin = torch.cos(theta), torch.sin(theta)
    eps = 1e-12                             # verhindert Division durch null bei cos/sin = 0
    # Je nach Vorzeichen von cos/sin trifft der Strahl die rechte oder linke bzw. die
    # obere oder untere Rechteckkante; clamp haelt den Nenner vom Vorzeichenwechsel weg.
    tx = torch.where(cos >= 0, (xmax[0]-center[0])/torch.clamp(cos, min=eps),
                                (xmin[0]-center[0])/torch.clamp(cos, max=-eps))   #xmax [0]-> x achse- rechte seite des gebietes
    ty = torch.where(sin >= 0, (xmax[1]-center[1])/torch.clamp(sin, min=eps),
                                (xmin[1]-center[1])/torch.clamp(sin, max=-eps)) # xmax [1]-> y achse. max = GOK
    return torch.minimum(tx, ty)            # die naeher liegende Kante begrenzt den Strahl


def sample_tunnel_wall(n, center, radius, sampler="grid", device="cpu"):
    """n Punkte auf dem Kreis r=radius (Tunnellaibung). "grid" = gleichmaessig verteilte
    Winkel, sonst zufaellige Winkel. Rueckgabe: (n,2)-Tensor kartesischer Koordinaten."""
    if sampler == "grid":
        theta = torch.linspace(0, 2*math.pi, n, device=device)
    else:
        theta = torch.rand(n, device=device)*2*math.pi
    return torch.stack([center[0]+radius*torch.cos(theta), center[1]+radius*torch.sin(theta)], dim=-1)


def sample_outer_edges(n, xmin, xmax, sampler="grid", device="cpu"):
    """Punkte auf den vier Aussenraendern. Rueckgabe:
    [links(x=xmin0), rechts(x=xmax0), unten(y=xmin1), oben(y=xmax1)], jeweils (n,2).

    Trick der Schleife: `dim` ist die Achse, die auf dem Rand FESTGEHALTEN wird
    (0 = x fest -> senkrechte Kante, 1 = y fest -> waagrechte Kante), `other` ist die
    Achse, entlang der variiert wird.
    """
    pts = []
    for dim in range(2):    #range von 0-1
        other = 1-dim
        if sampler == "grid":   #erzeugt laufparameter t (also n werte entlange variable achse zwischen max und min). grid steht in def. sonnst zufallszahl
            t = torch.linspace(xmin[other], xmax[other], n, device=device)
        else:
            t = xmin[other]+(xmax[other]-xmin[other])*torch.rand(n, device=device)
        for val in [xmin[dim], xmax[dim]]:      # beide gegenueberliegenden Kanten
            p = torch.zeros(n, 2, device=device)
            p[:, dim] = val                     # feste Koordinate (die Kante selbst)
            p[:, other] = t                     # laufende Koordinate entlang der Kante
            pts.append(p)
    return pts# [links, rechts, unten, oben]


def sample_raw_params(n, cfg, device="cpu"):
    """Zieht n zufaellige (E, gamma, phi)-Tripel, gleichverteilt aus den Bereichen der
    Config -- EIN Tripel pro Kollokationspunkt. Dadurch sieht das Netz in jedem Schritt
    viele verschiedene Bodenparameter und lernt, ueber sie zu generalisieren.
    Rueckgabe: Dictionary Name -> Tensor der Form (n,1)."""
    E = cfg.E_min+(cfg.E_max-cfg.E_min)*torch.rand(n, 1, device=device)  #erzeugt zahlen mit streuung im bereich max zu min (torch.rand erzeugt n anzahl an zahlen 0-1)
    gamma = cfg.gamma_min+(cfg.gamma_max-cfg.gamma_min)*torch.rand(n, 1, device=device)
    phi = cfg.phi_min+(cfg.phi_max-cfg.phi_min)*torch.rand(n, 1, device=device)
    return {"E": E, "gamma": gamma, "phi": phi}


# ----------------------------------------------------------------------
# Ableitungen (autograd-Helfer, von tunnel_twonet.py genutzt)
# ----------------------------------------------------------------------

def _grad(y, x):
    """d(sum(y))/dx -- pro Datenpunkt korrekt, weil jede Ausgabezeile nur von der
    zugehoerigen Eingabezeile abhaengt.

    Das ist das Herzstueck jedes PINN: torch.autograd.grad bildet die Ableitung der
    Netzausgabe nach den EINGABEkoordinaten (nicht nach den Gewichten). Die Summe ueber
    alle Punkte ist nur ein Trick, um alle Punkte in EINEM Aufruf zu erledigen -- da die
    Punkte voneinander unabhaengig sind, liefert die Ableitung der Summe genau die
    Einzelableitungen. create_graph=True ist zwingend, damit man das Ergebnis NOCHMAL
    ableiten kann (2. Ableitungen fuer Navier-Cauchy) und damit der Loss spaeter nach den
    Gewichten differenzierbar bleibt.
    """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

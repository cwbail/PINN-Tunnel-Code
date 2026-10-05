"""
Datenladen + hybrides Training (Daten + Physik) fuer das Tunnel-PINN.

Das ist die DATENGETRIEBENE Variante: statt die Loesung allein aus der Physik zu
gewinnen (das macht der Vorwaertsloeser in tunnel_twonet.py, der unveraendert
bleibt), trainieren wir DASSELBE Zwei-Netz-Modell darauf, eine Menge von
FEM-Rechnungen nachzubilden (ueberwachtes Lernen mit mittlerem quadratischem
Fehler, MSE), und BEHALTEN dabei das Physik-Residuum (Gleichgewicht) als
Regularisierung zwischen und um die Daten herum.

Datenformat (erzeugt von mohr_coulomb_dataexport.py):
    fem_data/manifest.csv                 # eine Zeile pro Rechenlauf
    fem_data/outputs/run_XXXX.npz         # xy (N,2)[m], u (N,2)[m], E_MPa, nu, gamma_kNm3, phi_deg, ...

Die Aufteilung in Trainings- und Validierungsdaten erfolgt PRO LAUF (ganze
Parametersaetze werden zurueckgehalten). Der Validierungsfehler misst damit die
Generalisierung auf Bodenparameter, die das Netz nie gesehen hat -- und nicht nur
die Interpolation zwischen benachbarten Punkten eines Feldes, das es ohnehin schon
angepasst hat.
"""

import os
import csv                              # zum Lesen der manifest.csv
import numpy as np
import torch
import matplotlib.tri as mtri           # Dreiecksvernetzung fuer den ParaView-Export

from .tunnel_twonet import (TwoNetTunnelPINN, soil_physics_loss, liner_physics_loss,
                            sample_soil, sample_annulus, sample_raw_params,
                            # fuer full_physics=True: Rand-/Kopplungsterme des reinen PINN
                            tunnel_wall_loss, interface_loss, roller_side_loss,
                            bottom_fixed_loss, top_surface_loss,
                            sample_tunnel_wall, sample_ring, sample_outer_edges)


# ----------------------------------------------------------------------
# Laden + Aufteilung in Training/Validierung pro Rechenlauf
# ----------------------------------------------------------------------

def _load_runs(data_dir):
    """Liest manifest.csv und laedt die npz-Datei jedes Laufs. Liefert eine Liste von Dictionaries.

    Eine .npz-Datei ist ein Container mehrerer NumPy-Arrays; d["xy"] holt das
    jeweilige Array heraus.
    """
    manifest = os.path.join(data_dir, "manifest.csv")
    runs = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):     # DictReader: jede CSV-Zeile wird zum Dictionary
            # Pfadtrenner aus der CSV auf das Betriebssystem umstellen (Windows: \)
            d = np.load(os.path.join(data_dir, r["file"].replace("/", os.sep)))
            runs.append(dict(
                run_id=int(r["run_id"]),
                xy=np.asarray(d["xy"], dtype=np.float32),         # (N,2) Knotenkoordinaten [m]
                u=np.asarray(d["u"], dtype=np.float32),           # (N,2) FEM-Verschiebungen [m]
                E=float(d["E_MPa"]), gamma=float(d["gamma_kNm3"]), phi=float(d["phi_deg"]),
            ))
    if not runs:
        raise RuntimeError(f"No runs found in {manifest}")
    return runs


def _pack(runs, device):
    """Haengt mehrere Rechenlaeufe zu flachen Tensoren aneinander, aus denen sich im
    Training bequem Punkte ziehen lassen. Aus welchem Lauf ein Punkt stammt, spielt
    danach keine Rolle mehr -- zu jedem Punkt stehen seine eigenen Bodenparameter. werden als Tensoren exportiert"""
    XY, PAR, U = [], [], []
    for rr in runs:
        xy, u = rr["xy"], rr["u"]
        n = xy.shape[0]
        XY.append(xy)
        U.append(u)
        # np.tile wiederholt das Parametertripel fuer JEDEN Knoten dieses Laufs,
        # damit zu jedem Punkt die zugehoerigen Bodenparameter stehen
        PAR.append(np.tile([rr["E"], rr["gamma"], rr["phi"]], (n, 1)).astype(np.float32))
    # Kurzschreibweise: Listen aneinanderhaengen und als Torch-Tensor auf das Geraet legen
    t = lambda a: torch.tensor(np.concatenate(a, 0), device=device)
    return dict(xy=t(XY), par=t(PAR), u=t(U))


def load_split(data_dir, val_fraction=0.2, seed=0, device="cpu"):
    """Laedt alle Rechenlaeufe und teilt sie LAUFWEISE in Training/Validierung auf.
    Liefert (train_pack, val_pack, param_ranges, info)."""
    runs = _load_runs(data_dir)   #manifest lesen, pro lauf .npz laden
    rng = np.random.default_rng(seed)           # eigener Zufallsgenerator -> reproduzierbare Aufteilung
    order = rng.permutation(len(runs))          # zufaellige Reihenfolge der Laeufe
    n_val = max(1, int(round(val_fraction*len(runs))))
    val_pos = set(order[:n_val].tolist())       # die ersten n_val Positionen werden zurueckgehalten
    train_runs = [runs[i] for i in range(len(runs)) if i not in val_pos]
    val_runs = [runs[i] for i in range(len(runs)) if i in val_pos]

    # Parameterbereiche aus ALLEN Laeufen (damit die Normierung Training + Validierung abdeckt)
    E = [r["E"] for r in runs]; G = [r["gamma"] for r in runs]; PH = [r["phi"] for r in runs]
    param_ranges = {"E": (min(E), max(E)), "gamma": (min(G), max(G)), "phi": (min(PH), max(PH))}

    # info sammelt alles, was man spaeter zum Nachvollziehen/Plotten braucht
    info = dict(n_runs=len(runs), n_train=len(train_runs), n_val=len(val_runs),
                train_ids=sorted(r["run_id"] for r in train_runs),
                val_ids=sorted(r["run_id"] for r in val_runs),
                val_runs=val_runs)
    return (_pack(train_runs, device),
            _pack(val_runs, device),
            param_ranges, info)


def _params_dict(par_rows):
    """Wandelt einen (n,3)-Tensor [E,gamma,phi] in die Dictionary-Form um, die das
    Modell erwartet. Das Slicing 0:1 statt 0 erhaelt dabei die Spaltenform (n,1)."""
    return {"E": par_rows[:, 0:1], "gamma": par_rows[:, 1:2], "phi": par_rows[:, 2:3]}


def _predict(model, xy, par):
    """Verschiebung des Modells an den Punkten xy mit punktweisen Parametern par (n,3);
    waehlt fuer jeden Punkt das passende Netz (Boden oder Schale).

    Anders als model.forward() werden hier die beiden Netze NUR auf ihren jeweils
    eigenen Punkten ausgewertet (boolesche Maskierung), was Rechenzeit spart.
    """
    p = _params_dict(par)
    dx = xy[:, 0:1]-model.center[0]; dy = xy[:, 1:2]-model.center[1]
    r = torch.sqrt(dx*dx+dy*dy)
    is_lin = (r < model.Ro).squeeze(1)      # squeeze(1) macht aus (n,1) einen flachen (n,)-Maskenvektor
    out = torch.zeros_like(xy)              # Ergebnisfeld gleicher Form vorbereiten
    if is_lin.any():
        # xy[is_lin] waehlt nur die Punkte in der Schale; die Parameter werden gleich mitgefiltert
        out[is_lin] = model.forward_liner(xy[is_lin], {k: v[is_lin] for k, v in p.items()})
    soil = ~is_lin                          # ~ invertiert die boolesche Maske
    if soil.any():
        out[soil] = model.forward_soil(xy[soil], {k: v[soil] for k, v in p.items()})
    return out


@torch.no_grad()    # Dekorator: keine Gradienten aufzeichnen -- es wird nur ausgewertet
def rmse_mm(model, pack):
    "Mittlerer quadratischer Verschiebungsfehler (RMSE) ueber ein Datenpaket, in Millimetern."
    pred = _predict(model, pack["xy"], pack["par"])
    # *1e3: Meter -> Millimeter; .item() holt die reine Zahl aus dem Tensor
    return float(torch.sqrt(((pred-pack["u"])**2).mean()).item()*1e3)


# ----------------------------------------------------------------------
# Trainiertes Modell speichern / laden und fuer neue Eingaben abfragen
# ----------------------------------------------------------------------

def save_model(model, cfg, path):
    """Speichert ein trainiertes Modell, damit es spaeter ohne erneutes Training geladen
    und abgefragt werden kann. Gesichert werden die Netzgewichte (state_dict) + die
    Konfiguration (die Geometrie, Architektur und die trainierten Parameterbereiche
    enthaelt) + die Ausgabeskalierung."""
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "u_sd": model.u_sd}, path)
    return path


def load_model(path, device="cpu"):
    """Laedt ein mit save_model() gespeichertes Modell. Liefert (model, cfg).
    Das Modell steht im Auswertungsmodus (eval)."""
    # weights_only=False ist noetig, weil neben den Gewichten auch das cfg-Objekt gespeichert ist
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = TwoNetTunnelPINN(cfg, u_sd=ckpt["u_sd"], device=device)
    model.load_state_dict(ckpt["state_dict"])   # Gewichte in die frisch gebaute Architektur laden
    model.eval()                                # Auswertungsmodus (schaltet z.B. Dropout/BatchNorm ab)
    return model, cfg


@torch.no_grad()
def predict(model, xy, E, gamma, phi):
    """Eingaben rein -> Ausgaben raus. Wertet das trainierte Modell an den Punkten `xy`
    aus ((N,2), Meter) fuer EIN Bodenparameter-Tripel (E [MPa], gamma [kN/m^3],
    phi [Grad]). Liefert die Verschiebung (N,2) in METERN.
    Beispiel:  u = predict(model, [[25.0, -30.0]], E=220, gamma=19.5, phi=28)  # ein Punkt
    """
    # reshape(-1,2) erlaubt auch die Uebergabe einer einzelnen Punktliste;
    # next(model.parameters()).device fragt ab, auf welchem Geraet das Modell liegt
    xy_t = torch.as_tensor(np.asarray(xy, np.float32).reshape(-1, 2), device=next(model.parameters()).device)
    par = torch.tensor(np.tile([float(E), float(gamma), float(phi)], (xy_t.shape[0], 1)),
                       dtype=torch.float32, device=xy_t.device)
    # .cpu().numpy(): zurueck auf die CPU und in ein normales NumPy-Array umwandeln
    return _predict(model, xy_t, par).cpu().numpy()


# ----------------------------------------------------------------------
# ParaView-Export (.vtu) -- exakte Knotenwerte in ParaView ansehen
# ----------------------------------------------------------------------

def write_vtu(path, xy, triangles, point_data):
    """Schreibt ein 2-D-Dreiecksfeld in eine von ParaView lesbare .vtu-Datei
    (ASCII VTK XML, ohne zusaetzliche Bibliothek).
        xy         : (N,2) Knotenkoordinaten
        triangles  : (M,3) Konnektivitaet als Ganzzahlen (lineare Dreiecke)
        point_data : {Name: (N,) Skalar  oder  (N,2)/(N,3) Vektor}
    Die Datei laesst sich in ParaView oeffnen; ueber die Tabellenansicht oder
    'Hover Points On' liest man exakte Werte an exakten Stellen ab, faerbt nach
    beliebigen Feldern ein oder nutzt Warp By Vector."""
    xy = np.asarray(xy); triangles = np.asarray(triangles, dtype=np.int64)
    N, M = xy.shape[0], triangles.shape[0]      # Anzahl Knoten und Zellen
    pts = np.zeros((N, 3), np.float32); pts[:, :2] = xy     # VTK erwartet 3-D-Punkte -> z=0
    with open(path, "w") as f:
        # --- Dateikopf ---
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
                '  <UnstructuredGrid>\n'
                f'    <Piece NumberOfPoints="{N}" NumberOfCells="{M}">\n')
        # --- Knotenkoordinaten ---
        f.write('      <Points>\n'
                '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, pts, fmt="%.7g")
        f.write('        </DataArray>\n      </Points>\n')
        # --- Zellen: Konnektivitaet, Offsets, Zelltyp ---
        f.write('      <Cells>\n'
                '        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, triangles.reshape(1, -1), fmt="%d")
        f.write('        </DataArray>\n'
                '        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        # Offsets: jede Zelle hat 3 Knoten, also 3, 6, 9, ...
        np.savetxt(f, (np.arange(1, M+1)*3).reshape(1, -1), fmt="%d")
        f.write('        </DataArray>\n'
                '        <DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full((1, M), 5, dtype=np.int64), fmt="%d")   # 5 = VTK_TRIANGLE
        f.write('        </DataArray>\n      </Cells>\n')
        # --- die eigentlichen Feldwerte an den Knoten ---
        f.write('      <PointData>\n')
        for name, arr in point_data.items():
            arr = np.asarray(arr, np.float32)
            if arr.ndim == 1:                   # Skalarfeld
                f.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
                np.savetxt(f, arr.reshape(1, -1), fmt="%.7g")
            else:                               # Vektorfeld
                if arr.shape[1] == 2:                                 # Vektoren auf 3 Komponenten auffuellen
                    a3 = np.zeros((arr.shape[0], 3), np.float32); a3[:, :2] = arr; arr = a3
                f.write(f'        <DataArray type="Float32" Name="{name}" '
                        f'NumberOfComponents="{arr.shape[1]}" format="ascii">\n')
                np.savetxt(f, arr, fmt="%.7g")
            f.write('        </DataArray>\n')
        f.write('      </PointData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')


def export_prediction_vtu(model, xy, params, path, u_fem=None):
    """Wertet das trainierte Modell an den Punkten `xy` fuer EIN (E,gamma,phi)-Tripel aus
    und schreibt eine ParaView-.vtu. Wird u_fem (N,2) mitgegeben, werden zusaetzlich das
    FEM-Feld und der Fehler PINN-gegen-FEM geschrieben, fuer den direkten Vergleich. Die
    Verschiebungsvektoren stehen in METERN (passend zur FEM-XDMF); Betraege/Fehler gibt
    es zusaetzlich in mm, weil sich das leichter liest."""
    xy = np.asarray(xy, np.float32)
    par = torch.tensor(np.tile(list(params), (xy.shape[0], 1)), dtype=torch.float32)
    with torch.no_grad():                       # reine Auswertung, keine Gradienten noetig
        u_pinn = _predict(model, torch.tensor(xy), par).cpu().numpy()

    # Punkte triangulieren und die Dreiecke im ausgebrochenen Hohlraum verwerfen
    xc, yc, Ri = float(model.center[0]), float(model.center[1]), float(model.Ri)
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    cx = xy[tri.triangles, 0].mean(1); cy = xy[tri.triangles, 1].mean(1)    # Schwerpunkt jedes Dreiecks
    keep = ((cx-xc)**2 + (cy-yc)**2) >= Ri**2   # nur Dreiecke ausserhalb des Hohlraums behalten
    triangles = tri.triangles[keep]

    data = {"u_pinn_m": u_pinn, "mag_pinn_mm": np.linalg.norm(u_pinn, axis=1)*1e3}
    if u_fem is not None:
        u_fem = np.asarray(u_fem, np.float32)
        data["u_fem_m"] = u_fem
        data["mag_fem_mm"] = np.linalg.norm(u_fem, axis=1)*1e3
        data["error_mm"] = np.linalg.norm(u_pinn-u_fem, axis=1)*1e3
    write_vtu(path, xy, triangles, data)
    return path


# ----------------------------------------------------------------------
# Hybrides Training (Daten + Physik)
# ----------------------------------------------------------------------

def train_hybrid(cfg, data_dir, w_data=1.0, w_phys=0.6, n_data=4000, n_phys=2000,
                 val_fraction=0.2, split_seed=0, full_physics=True,
                 w_liner_phys=None, w_iface_disp=0.0, n_iface=200, liner_norm="sig", verbose=True):
    """Trainiert das Zwei-Netz-Modell so, dass es die FEM-Daten trifft (MSE) UND die
    Physik einhaelt (als Regularisierung). Liefert (model, history, info).

    w_data / w_phys gewichten die beiden Ziele gegeneinander, n_data / n_phys legen
    fest, wie viele Punkte pro Schritt frü den jeweiligen Term gezogen werden.

    full_physics=False: Physikterm = nur Gleichgewicht (Navier in Boden + Schale).
    full_physics=True:  zusaetzlich ALLE Rand- und Kopplungsterme des reinen PINN
        (Ausbruchslast an R_i, Grenzflaeche R_o, Seiten, Sohle, Oberflaeche). Ohne sie
        erfuellt bei Parametern ohne FEM-Daten (z. B. E=217) auch u=0 das Gleichgewicht;
        mit ihnen ist der Physikterm ein vollstaendiges Randwertproblem. Die relativen
        Gewichte der Randterme kommen aus cfg (w_tunnel, w_interface, w_outer) wie im
        reinen PINN; der gesamte Physikblock wird mit w_phys gewichtet.

    Uebernommen aus der Versuchsreihe vom 27.09.2026 (tunnel_data_experiment.py, Lauf E):
    w_liner_phys  eigenes Gewicht fuer das Gleichgewicht in der Schale (None = w_phys)
    w_iface_disp  Gewicht der Verschiebungs-Kontinuitaet u_Schale = u_Boden an R_o
                  (0 = aus); n_iface Punkte dafuer
    liner_norm    Normierung des Schalen-Gleichgewichts:
                  "sig"   Residuum * t_liner / sig_ref (wie Boden). Die Beton-Steifigkeit
                          (~37800 MPa) verstaerkt jede kleine Ungenauigkeit so stark, dass
                          die Schale starr bleibt.
                  "stiff" Residuum / (lam_c+2mu_c) * t_liner^2 / u_sd, also die Kruemmung
                          der Verschiebung relativ zu u_sd; Steifigkeit kuerzt sich heraus.
    Empfohlen (Lauf E): full_physics=False, w_phys=0.1, w_iface_disp=1, liner_norm="stiff".
    """
    torch.manual_seed(cfg.seed)     # fixiert alle Zufallszahlen -> reproduzierbarer Lauf
    device = cfg.device
    center_t = cfg.center()
    center = torch.tensor(center_t, dtype=torch.float32, device=device)
    xmin = torch.tensor([0., -cfg.total_depth], dtype=torch.float32, device=device)
    xmax = torch.tensor([cfg.width, 0.], dtype=torch.float32, device=device)
    Ri, Ro = cfg.radius, cfg.outer_liner_radius()
    lam_c, mu_c = cfg.lame_liner()

    # Skalen (identisch zu den Konventionen des Vorwaertsloesers)
    sig_ref = abs(cfg.gamma_max*(-cfg.depth)/1000.)     # Referenzspannung [MPa] in Tunneltiefe
    G_mid = ((cfg.E_min+cfg.E_max)/2)/(2*(1+cfg.nu))    # mittlerer Schubmodul
    u_sd = Ri*sig_ref/(2*G_mid)                         # Referenzgroesse der Verschiebung
    len_soil = max(cfg.width, cfg.total_depth)          # Laengenskala Boden = Gebietsgroesse
    len_liner = cfg.t_liner                             # Laengenskala Schale = Dicke

    # --- Daten laden ---
    train, val, param_ranges, info = load_split(data_dir, val_fraction, split_seed, device)
    # Die Parameternormierung des Modells an die tatsaechlichen Bereiche des Datensatzes anpassen
    cfg.E_min, cfg.E_max = param_ranges["E"]   #überschreibt die bereich mit denen der Daten aus load_split funktion
    cfg.gamma_min, cfg.gamma_max = param_ranges["gamma"]
    cfg.phi_min, cfg.phi_max = param_ranges["phi"]
    if verbose:
        print(f"data: {info['n_runs']} runs -> {info['n_train']} train, {info['n_val']} val (held-out params)")
        print(f"  train points: {train['xy'].shape[0]:,}   val points: {val['xy'].shape[0]:,}")
        print(f"  held-out run ids (validation): {info['val_ids']}")
        print(f"  param ranges: E{param_ranges['E']} gamma{param_ranges['gamma']} phi{param_ranges['phi']}")

    model = TwoNetTunnelPINN(cfg, u_sd=u_sd, device=device)  #hier wird _init_automatisch abgefragt
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)   # Gewichtsabfall (weight decay) einbauen? Oder ist die Lernrate klein genug?
    lr_gamma = cfg.lr_decay**(1/max(cfg.n_steps, 1))    
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_gamma)

    M = train["xy"].shape[0]        # Gesamtzahl der Trainingspunkte (alle Laeufe zusammen) läufe * trainungspunkte
    history = []
    for i in range(cfg.n_steps+1):
        opt.zero_grad()             # alte Gradienten loeschen (PyTorch summiert sonst auf)

        # ---- Datenterm (MSE): zufaelliger Mini-Batch aus den Trainingslaeufen ----
        idx = torch.randint(0, M, (n_data,), device=device)     # n_data zufaellige Zeilenindizes
        xb, pb, ub = train["xy"][idx], train["par"][idx], train["u"][idx]  #aus alle FEM knoten werden 4000 zufällig herausgegriffen. xp (ort), pb (parameter den boden). ub (verschiebung)
        pred = _predict(model, xb, pb)     #netzt räd u an n_data punkte
        l_data = (((pred-ub)/u_sd)**2).mean()       # /u_sd macht den Fehler dimensionslos. Abweichung netz- FEM, quadriert, gemittelt (MSE). u_sd ist grobe soll verschiebung nach kirsch

        # ---- Physikterm (Regularisierung): frische Kollokationspunkte, Parameter ----
        # stetig ueber den trainierten Bereich gezogen (hilft bei ungesehenen Parameterwerten) ----
        x_soil = sample_soil(int(n_phys**0.5)+1, int(n_phys**0.5)+1, xmin, xmax, center, Ro, device=device)  #frische zufallspunkte im boden
        x_lin = sample_annulus(40, 8, center, Ri, Ro, device=device)
        p_soil = sample_raw_params(x_soil.shape[0], cfg, device)
        p_lin = sample_raw_params(x_lin.shape[0], cfg, device)
        l_ps = soil_physics_loss(model, x_soil, p_soil, cfg.nu, len_soil, sig_ref)
        if liner_norm == "stiff":   # Residuum/(lam_c+2mu_c) * t^2/u_sd (verformungsbezogen)
            l_pl = liner_physics_loss(model, x_lin, p_lin, lam_c, mu_c, len_liner**2, u_sd*(lam_c+2*mu_c))
        else:                       # Residuum * t/sig_ref (spannungsbezogen, wie im Boden)
            l_pl = liner_physics_loss(model, x_lin, p_lin, lam_c, mu_c, len_liner, sig_ref)
        l_phys = l_ps+l_pl          # nur fuer Ausgabe/Plot (ungewichtet)

        # ---- Verschiebungs-Kontinuitaet Schale/Boden an R_o ----
        l_idisp = torch.zeros((), device=device)
        if w_iface_disp > 0:
            x_if = sample_ring(n_iface, center, Ro, device=device)
            p_if = sample_raw_params(x_if.shape[0], cfg, device)
            # dieselben Punkte einmal mit dem Boden-, einmal mit dem Schalennetz auswerten
            l_idisp = (((model.forward_soil(x_if, p_if)-model.forward_liner(x_if, p_if))/u_sd)**2).mean()

        # ---- optional: Rand- und Kopplungsterme (wie in tunnel_twonet.train) ----
        l_bc = torch.zeros((), device=device)      # 0, falls full_physics=False
        if full_physics:        #randbedingungen ( am auflager und tunnel face usw) dann = true
            x_wall = sample_tunnel_wall(cfg.n_tunnel, center, Ri, sampler="uniform", device=device)
            x_iface = sample_ring(cfg.n_interface, center, Ro, device=device)
            left, right, x_bottom, x_top = sample_outer_edges(cfg.n_edge, xmin, xmax, sampler="uniform", device=device)
            x_sides = torch.cat([left, right], dim=0)
            P = lambda n: sample_raw_params(n, cfg, device)
            l_wall = tunnel_wall_loss(model, x_wall, P(x_wall.shape[0]), lam_c, mu_c, sig_ref)
            l_iface = interface_loss(model, x_iface, P(x_iface.shape[0]), cfg.nu, lam_c, mu_c, sig_ref)
            l_outer = (roller_side_loss(model, x_sides, P(x_sides.shape[0]), cfg.nu, sig_ref)
                       + bottom_fixed_loss(model, x_bottom, P(x_bottom.shape[0]))
                       + top_surface_loss(model, x_top, P(x_top.shape[0]), cfg.nu, sig_ref))
            l_bc = cfg.w_tunnel*l_wall + cfg.w_interface*l_iface + cfg.w_outer*l_outer

        w_lin = w_phys if w_liner_phys is None else w_liner_phys
        loss = w_data*l_data + w_phys*(l_ps + l_bc) + w_lin*l_pl + w_iface_disp*l_idisp
        loss.backward()             # Rueckwaertsdurchlauf: Gradienten nach allen Gewichten
        # Gradientennorm begrenzen, damit einzelne Ausreisser das Training nicht aus der Bahn werfen
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step(); sched.step()    # Gewichte aktualisieren, Lernrate verkleinern

        if i % cfg.log_every == 0:
            # RMSE auf Trainings- UND Validierungsdaten: zeigt Anpassung gegen Generalisierung
            tr = rmse_mm(model, train); va = rmse_mm(model, val)
            # Spalte 6 = gewichtete Randterme (bei full_physics=False immer 0)
            history.append((i, loss.item(), l_data.item(), l_phys.item(), tr, va, l_bc.item()))
            if verbose:
                bc_txt = f"  bc={l_bc.item():.3e}" if full_physics else ""
                if_txt = f"  iface_u={l_idisp.item():.3e}" if w_iface_disp > 0 else ""
                print(f"[step {i}/{cfg.n_steps}] loss={loss.item():.3e}  data={l_data.item():.3e}  "
                      f"soil={l_ps.item():.3e}  liner={l_pl.item():.3e}{bc_txt}{if_txt}  "
                      f"RMSE train={tr:.4f} mm  val={va:.4f} mm")

    return model, history, info

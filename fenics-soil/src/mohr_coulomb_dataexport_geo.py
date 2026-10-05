"""
GEOMETRIE-VARIANTE (28.09.2026): Kopie von mohr_coulomb_dataexport.py mit zwei zusaetzlichen
Startoptionen --depth (Tiefe der Tunnelachse unter GOK [m]) und --t_liner (Schalendicke [m]).
Beide werden in jede .npz (depth_m, t_liner_m) und in manifest.csv geschrieben.
Standard-Ausgabeordner: fem_data_geo (getrennt vom alten Datensatz fem_data).
Fuer das PINN: PINNs/tunnel_pinn/tunnel_geo.py, Start: PINNs/examples/tunnel_datanet_geo.py

Tunnelbau – Ebener Verformungszustand (EVZ)  |  FEM mit FEniCSx + gmsh
=======================================================================

Geometrie : 50 m × 50 m Querschnitt
            Tunnelachse bei x = 25 m, y = –25 m  (Geländeoberfläche = y = 0)
            Tunneldurchmesser D = 5.0 m,  Betonschale t = 0.50 m

Boden     : sandiger Schluff
            E = 10 MPa,  ν = 0.30,  γ = 18 kN/m³
            c = 2 kPa,  φ = 27°,  ψ = 3°
            (φ/ψ für MC-Ausnutzungs-Nachweis,  KEIN Plastizitätsalgorithmus)

Schale    : C35/45 Beton nach EC2  (Ecm ≈ 34 GPa)
            E = 34 GPa,  ν = 0.20,  γ = 25 kN/m³,  t = 0.50 m

Modell    : Linear-elastisch,  K0-Verfahren  (Ruhedruck nach Jaky: K0 = 1–sin φ).
            Geostatische Primärspannungen σ₀ werden als Initialzustand vorgegeben.
            Δu = Verschiebungen NUR aus Tunneleinbau (Spannungsumlagerung),
            NICHT aus dem Bodeneigengewicht vor dem Tunnelbau.

Netz      : gmsh  → Boolean-Fragment  → Boden | Betonschale | freier Hohlraum
            P2-Dreieckselemente,  Verfeinerung nahe Tunnel (h_min = 0.15 m)

Output    : Verschiebungen u_x / u_y  (P2)
            Spannungen σ_xx, σ_yy, τ_xy, σ_zz  (DG0, EVZ)
            von-Mises-Spannung  (vollständiges 3-D im EVZ)
            Tangential- / Radialspannung in der Schale
            Mohr-Coulomb-Ausnutzung η im Boden
            → 9 XDMF-Dateien für ParaView

Abhängigkeiten:
    dolfinx >= 0.9,  gmsh,  basix,  petsc4py  (MUMPS empfohlen)

Ausführen:
    python tunnel_fem.py
    # oder mit MPI (serielle Ausführung empfohlen für Studienarbeit):
    mpirun -n 1 python tunnel_fem.py
    Laufen bringen: /home/fenics/project/run.sh
"""

import gmsh          # ← MUSS als allererstes stehen, vor dolfinx!
import sys
# sicherstellen dass sys.modules['gmsh'] das richtige Paket ist
assert hasattr(gmsh, 'initialize'), \
    "Falsches gmsh-Modul geladen! Prüfe Import-Reihenfolge."
import os
import argparse          # [DATA EXPORT] read per-run parameters from the command line
import csv               # [DATA EXPORT] write/append the manifest
import numpy as np
import basix.ufl
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem import (
    functionspace, Function, Constant,
    dirichletbc, locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
import ufl
from ufl import (
    TestFunction, TrialFunction,
    grad, inner, sym, Identity, tr,
    SpatialCoordinate, sqrt,
)
from petsc4py import PETSc

os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MATERIALPARAMETER
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# [DATA EXPORT VERSION]  This is a copy of mohr_coulomb.py, modified ONLY to:
#   (1) take the soil parameters from the command line (one FEM run per parameter set), and
#   (2) write the displacement field to fem_data/outputs/run_XXXX.npz + a manifest row,
#       for use as supervised (MSE) training data in the hybrid data+physics PINN.
# The physics/solve is UNCHANGED from the original. Geometry is still fixed here; when you
# later add variable geometry, add those as CLI args too -- the node coordinates are already
# stored per-run (see the export block after the solve), so variable meshes will just work.
#
# NOTE: displacement is LINEAR-ELASTIC, so it depends on E, nu, gamma and phi (via K0=1-sinφ)
#       -- but NOT on cohesion. Cohesion is recorded in the manifest for later (utilisation/
#       stress outputs), yet varying ONLY cohesion produces identical displacement fields, so
#       for a displacement dataset vary E / gamma / phi.
#
# Example single run:
#   python mohr_coulomb_dataexport.py --run_id 1 --E 100 --gamma 20 --phi 30
# Loop many runs from a shell / batch script, giving each a unique --run_id.
# ══════════════════════════════════════════════════════════════════════════════
_ap = argparse.ArgumentParser(description="FEM run -> displacement .npz for PINN training data")
_ap.add_argument("--run_id",   type=int,   required=True,  help="unique integer id for this run (-> run_XXXX.npz)")
_ap.add_argument("--E",        type=float, default=10.0,   help="soil Young's modulus [MPa]")
_ap.add_argument("--nu",       type=float, default=0.30,   help="soil Poisson's ratio [-]")
_ap.add_argument("--gamma",    type=float, default=18.0,   help="soil unit weight [kN/m^3]")
_ap.add_argument("--phi",      type=float, default=27.0,   help="soil friction angle [deg] (sets K0=1-sin(phi))")
_ap.add_argument("--cohesion", type=float, default=2.0,    help="soil cohesion [kPa] (recorded only; no effect on displacement)")
_ap.add_argument("--depth",    type=float, default=25.0,   help="depth of the tunnel axis below the surface [m] (GEO)")
_ap.add_argument("--t_liner",  type=float, default=0.2,    help="liner thickness [m] (GEO)")
_ap.add_argument("--outdir",   type=str,   default="fem_data_geo", help="output folder (created if missing)")
_ap.add_argument("--full_output", action="store_true",
                 help="also run the stress/Mohr-Coulomb/XDMF sections (slower). Off by default = displacement only.")
_ap.add_argument("--overwrite",  action="store_true", help="allow overwriting an existing run_XXXX.npz")
args = _ap.parse_args()

# ── Boden: sandiger Schluff (values from the command line) ────────────────────
E_s   = args.E * 1.0e6            # MPa    -> Pa
nu_s  = args.nu                   # [-]
gam_s = args.gamma * 1.0e3        # kN/m^3 -> N/m^3
c_s   = args.cohesion * 1.0e3     # kPa    -> Pa   (used only in the optional MC-utilisation section)
phi_s = np.radians(args.phi)      # deg    -> rad
psi_s = np.radians(3.0)           # Dilatanzwinkel [rad] (kept fixed)
K0_s  = 1.0 - np.sin(phi_s)       # Ruhedruckbeiwert nach Jaky: K0 = 1 – sin(φ)  [-]
print(f"[run {args.run_id:04d}]  E={args.E} MPa  nu={args.nu}  gamma={args.gamma} kN/m^3  "
      f"phi={args.phi} deg  cohesion={args.cohesion} kPa  ->  K0={K0_s:.4f}")

lam_s = E_s  * nu_s  / ((1 + nu_s)  * (1 - 2*nu_s))   # 1. Lamé-Konstante [Pa]
mu_s  = E_s  / (2*(1 + nu_s))                           # Schubmodul        [Pa]

# ── Beton: C35/45 nach EC2 ────────────────────────────────────────────────────
E_c,  nu_c,  gam_c = 34.0e9, 0.20, 25.0e3      # Pa  |  –  |  N/m³
lam_c = E_c  * nu_c  / ((1 + nu_c)  * (1 - 2*nu_c))
mu_c  = E_c  / (2*(1 + nu_c))

# ══════════════════════════════════════════════════════════════════════════════
# 2.  GEOMETRIEPARAMETER
# ══════════════════════════════════════════════════════════════════════════════

W, H  = 50.0, 50.0      # Querschnitt: Breite × Tiefe  [m]
xc    = W / 2            # Tunnelachse x = 25.0 m
yc    = -args.depth      # Tunnelachse y = -depth  (Oberfläche = 0)  [GEO: Startoption --depth]
R_i   = 2.5              # Innenradius Hohlraum  [m]  →  D = 5.0 m
t_lin = args.t_liner    # Wanddicke Betonschale [m]  [GEO: Startoption --t_liner]
R_o   = R_i + t_lin      # Außenradius Schale    [m]
# Plausibilitaet: Tunnel samt Schale muss mit Abstand im Gebiet liegen
if not (R_o + 1.0 <= args.depth <= H - R_o - 1.0):
    raise SystemExit(f"[GEO] depth={args.depth} m passt nicht: Schale (R_o={R_o} m) muss mind. 1 m "
                     f"Abstand zu Oberkante und Sohle haben ({R_o+1:.1f} <= depth <= {H-R_o-1:.1f}).")
print(f"[GEO] Tunnelachse bei y = {yc:.2f} m (depth {args.depth} m), Schalendicke t = {t_lin:.2f} m, R_o = {R_o:.2f} m")

# Physikalische Gruppen – interne Markierungen
SOIL,   LINING, VOID   = 1, 2, 3    #definiert Flächen farben für Paraview (überlagern)
BC_BOT, BC_LEFT, BC_RIGHT, BC_TOP = 10, 11, 12, 13

# ══════════════════════════════════════════════════════════════════════════════
# 3.  NETZGENERIERUNG  (gmsh)
# ══════════════════════════════════════════════════════════════════════════════

def build_mesh():
    """
    ->netzt wird mit gsmh erstellt
    Erstellt das 2-D-Netz mit drei Subdomänen:
      SOIL    – Boden (Rechteck minus Außenkreis)
      LINING  – Betonschale (Ringfläche Außen- minus Innenkreis)
      Hohlraum – kein Physical Group → nicht in DOLFINx-Mesh exportiert.
                Innenwand = freie Randbedingung (σ·n = 0, natürliche RB).

    Netzgröße: SizeMin = 0.15 m nahe Tunnel,  SizeMax = 3.0 m weit weg.
    P2-Elemente: Kreisgeometrie wird korrekt approximiert.
    """
    gmsh.initialize()
    gmsh.model.add("tunnel_EVZ") #ist der Name des Modells, wird in gmsh angezeigt
    gmsh.option.setNumber("General.Verbosity", 1)   #erbosity Level of information printed on the terminal and the message console (0: silent except for fatal errors, 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug) Default value: 5

    occ = gmsh.model.occ        #occ hilft definieren wie die Geometrie aufgebaut wird, z.B. Rechteck, Kreis, etc.

    # ── Geometrie aufbauen ────────────────────────────────────────────────────
    r_rect  = occ.addRectangle(0.0, -H,  0.0, W, H)
    d_outer = occ.addDisk(xc, yc, 0.0, R_o, R_o)   # Außenkreis (Schale+Hohlraum)
    d_inner = occ.addDisk(xc, yc, 0.0, R_i, R_i)   # Innenkreis (Hohlraum)

    # Boolean Fragment: teilt alle Flächen entlang der Schnittlinien.
    # Ergebnis: 3 Flächen  →  Boden | Schale | Hohlraum
    occ.fragment([(2, r_rect)], [(2, d_outer), (2, d_inner)])
    occ.synchronize()

    # ── Flächen klassifizieren (über Flächeninhalt) ───────────────────────────
    A_soil = W*H - np.pi*R_o**2            # ≈ 2 471.7 m²
    A_lin  = np.pi*(R_o**2 - R_i**2)      # ≈     8.6 m²
    # A_void = np.pi*R_i**2               # ≈    19.6 m²  (nicht vernetzt)
    TOL_A  = 5.0                            # m²-Toleranz für Identifikation

    s_soil, s_lin, s_void = [], [], []
    for _, tag in gmsh.model.getEntities(2):    #2 = Dimension der Entität (Flächen) (von KI)
        area = occ.getMass(2, tag)          
        if   abs(area - A_soil) < TOL_A:  s_soil.append(tag)        #flächeninhalt mit der vorher bestimmten sollwerten verglichen mit einerer toleranz, damit die flächen identifiziert werden können
        elif abs(area - A_lin)  < TOL_A:  s_lin.append(tag)
        else:                              s_void.append(tag)

    if not s_soil or not s_lin:
        raise RuntimeError(
            f"Flächen-Klassifikation fehlgeschlagen!\n"
            f"  Gefunden: {[occ.getMass(2,t) for _,t in gmsh.model.getEntities(2)]}\n"
            f"  Erwartet: ~{A_soil:.1f} m² (Boden)  und  ~{A_lin:.1f} m² (Schale)"
        )

    # Hohlraum-Flächen aus der Geometrie löschen, bevor das Netz generiert wird.
    # recursive=False: Randkurven (Tunnelinnenwand, r=R_i) bleiben erhalten
    # → Innenwand der Schale wird automatisch freier Rand (σ·n = 0, natürl. RB).
    for tag in s_void:
        occ.remove([(2, tag)], recursive=False)
    occ.synchronize()

    gmsh.model.addPhysicalGroup(2, s_soil, SOIL,   "Boden")
    gmsh.model.addPhysicalGroup(2, s_lin,  LINING, "Betonschale")

    # ── Randkurven identifizieren ─────────────────────────────────────────────
    EPS_C = 0.01        #ist toleranz für die Identifikation der Randkurven, damit die Randbedingungen später korrekt zugeordnet werden können
    c_bot, c_left, c_right, c_top = [], [], [], []
    for _, tag in gmsh.model.getEntities(1):
        xn, yn, _, xx, yx, _ = gmsh.model.getBoundingBox(1, tag)
        if   yx < -H + EPS_C:  c_bot.append(tag)
        elif xx <      EPS_C:  c_left.append(tag)
        elif xn >  W - EPS_C:  c_right.append(tag)
        elif yn >    - EPS_C:  c_top.append(tag)

    gmsh.model.addPhysicalGroup(1, c_bot,   BC_BOT,   "Sohle")  
    gmsh.model.addPhysicalGroup(1, c_left,  BC_LEFT,  "Links")
    gmsh.model.addPhysicalGroup(1, c_right, BC_RIGHT, "Rechts")
    gmsh.model.addPhysicalGroup(1, c_top,   BC_TOP,   "Gelaende")

    # ── Netzverfeinerung: fein nahe Tunnel, grob weit weg ────────────────────
    outer_c = set(c_bot + c_left + c_right + c_top)
    inner_c = [t for _, t in gmsh.model.getEntities(1) if t not in outer_c]

    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", inner_c)
    gmsh.model.mesh.field.setNumber( fd, "Sampling",   300)

    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField",  fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin",  0.15)   # 15 cm nahe Tunnel
    gmsh.model.mesh.field.setNumber(ft, "SizeMax",  3.0)    # 3 m weit weg
    gmsh.model.mesh.field.setNumber(ft, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMax",  15.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)

    gmsh.model.mesh.generate(2) #2 = Dimension des Netzes (2D)
    # P2-Elemente: Kreisrand wird durch quadrat. Kurvenelemente korrekt approx.
    # Falls Probleme → Zeile auskommentieren für P1-Netz (subparametric, OK)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")  #Siehe seite 117 von GMSH für weitere Möglichkeiten der Netzoptimierung

    return gmsh.model


# ── Netz erzeugen und in DOLFINx importieren ──────────────────────────────────
print("═" * 64)
print("  TUNNELBAU  –  2-D FEM (EVZ, linear-elastisch)")
print("═" * 64)
print("  Netz … ", end="", flush=True)

gm = build_mesh()
mesh_data = gmshio.model_to_mesh(gm, MPI.COMM_WORLD, 0, gdim=2)
domain, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
gmsh.finalize()

nc = domain.topology.index_map(domain.topology.dim).size_global
nn = domain.topology.index_map(0).size_global
print(f"OK  ({nc:,} Elemente  |  {nn:,} Knoten  |  P2)")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  FUNKTIONENRAUM  –  P2 vektoriell (Verschiebungen)
# ══════════════════════════════════════════════════════════════════════════════

cell_type = domain.topology.cell_name()   # "triangle"
Ve = basix.ufl.element("Lagrange", cell_type, 2, shape=(2,))
V  = functionspace(domain, Ve)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  RANDBEDINGUNGEN
# ══════════════════════════════════════════════════════════════════════════════

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

def dofs(sub_idx: int, bc_tag: int):
    """Gibt DOF-Indices für eine Unterkomponente an einem Rand zurück."""
    return locate_dofs_topological(
        V.sub(sub_idx), fdim, facet_tags.find(bc_tag)
    )

zero = PETSc.ScalarType(0.0) #Nullwert für Dirichlet-RB (Verschiebung = 0 an den definierten Rändern)
bcs = [
    # Sohle (y = –50 m): vollständig eingespannt
    dirichletbc(zero, dofs(0, BC_BOT),   V.sub(0)),   # ux = 0  def 0- keine verschiebung , V.sub definiert das auf X achse bezogene Unterfeld, dofs(0, BC_BOT) gibt die DOF-Indices für die x-Komponente an der Sohle zurück
    dirichletbc(zero, dofs(1, BC_BOT),   V.sub(1)),   # uy = 0 x = 0; y = 1 def 1- keine verschiebung, V.sub definiert das auf Y achse bezogene Unterfeld, dofs(1, BC_BOT) gibt die DOF-Indices für die y-Komponente an der Sohle zurück
    # Seitenwände: Gleitlager (kein horizontaler Ausfluss)
    dirichletbc(zero, dofs(0, BC_LEFT),  V.sub(0)),   # ux = 0
    dirichletbc(zero, dofs(0, BC_RIGHT), V.sub(0)),   # ux = 0
    # Geländeoberfläche (BC_TOP):  frei  →  σ·n = 0  (natürliche RB)
    # Tunnelinnenwand:              frei  →  σ·n = 0  (natürliche RB, Hohlraum)
]

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VARIATIONSPROBLEM  (schwache Form der linearen Elastizität, EVZ)
# ══════════════════════════════════════════════════════════════════════════════

def eps(u):
    """Symmetrischer Verzerrungstensor  ε = ½(∇u + ∇uᵀ)
    u ist das Verschiebungsfeld (wie viel sich jeder Punkt bewegt) Der Verzerrungstensor misst, wie stark sich das Material dehnt oder schert — nicht die absolute Bewegung, sondern die relative Formänderung. sym(grad(u)) sorgt dafür, dass nur der symmetrische Anteil bleibt (Rotation ignorieren)"""
    return sym(grad(u))

def sigma(u, lam, mu):
    """Cauchy-Spannungstensor (isotrope lineare Elastizität)    
       σ = λ·tr(ε)·I + 2μ·ε
       lam, mu können Python-Floats oder DG0-Functions sein.
       ist das Hook'sche Gesetz in Tensorform
    """
    return lam * tr(eps(u)) * Identity(2) + 2*mu*eps(u) #cauchy spannungstensor, tr(eps(u)) ist die Spur des Verzerrungstensors, Identity(2) ist der 2x2 Einheitsmatrix, eps(u) ist der Verzerrungstensor ->prüfen wahrscheinlich anderst machen

# Subdomain-Maß (Integrationsbereiche)
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags) #ufl: Unified Form Language, dx ist das Maß für die Integration über die Zellen, subdomain_data=cell_tags ermöglicht die Integration über verschiedene Subdomänen mit unterschiedlichen Materialparametern

du = TrialFunction(V)    # Ansatzfunktion (unbekannte Verschiebung) knoten j wird verschoben
v  = TestFunction(V)     # Testfunktion -> durch die änderung von du zu v wird die schwache Formulierung der Elastizitätsgleichung erstellt. a = k (steifigkeit). knoten i hat eine kraft die durch die verschiebung von knoten j verursacht wird. v ist die virtuelle Verschiebung, die in der virtuellen Arbeit verwendet wird, um die Wirkung der Kräfte zu messen. Ist die Reaktion zur verschiebung

# Bilinearform: subdomain-weise Integration mit verschiedenen Materialparametern
a = (
    inner(sigma(du, lam_s, mu_s), eps(v)) * dx(SOIL)      # Boden
  + inner(sigma(du, lam_c, mu_c), eps(v)) * dx(LINING)    # Betonschale
)   #innere virtuelle Arbeit (wie viel Wid.,pmerstand das Material gegen verformung leistet (werden separat integriert- nach assemblierung wird daraus die globale steifigkeitsmatrix K))

# Koordinaten (werden auch in Abschn. 8 für Tangential-/Radialspannungen genutzt)
xsp = SpatialCoordinate(domain)

# ── K0-Initialspannungsfeld (geostatisch, Ruhedruck nach Jaky) ───────────────
# y < 0 unter Geländeoberfläche → σ₀_yy = γ·y < 0 (Druckspannung ✓)
#                                  σ₀_xx = K0·γ·y  (Horizontaldruck   ✓)
# Gilt für die gesamte Ausgangsdomäne (Boden + Bereich, wo später Schale liegt).
sig0_ufl = ufl.as_matrix([
    [K0_s * gam_s * xsp[1], 0.0],
    [0.0,                   gam_s * xsp[1]]
])

# Körperkräfte
g_s = Constant(domain, np.array([0.0, -gam_s]))
g_c = Constant(domain, np.array([0.0, -gam_c]))

# Linearform: Eigengewicht MINUS Initialspannungen  →  K0-Verfahren
# Physikalisch: die Initialspannungen σ₀ sind bereits im Gleichgewicht mit dem
# Eigengewicht; nach Subtraktion bleibt nur die Spannungsumlagerung durch den
# Tunneleinbau als treibende Last übrig (Δu = Tunnel-induzierte Verformungen).
L = (inner(g_s, v) * dx(SOIL)       #definiert linear form für kräfte
   + inner(g_c, v) * dx(LINING)     #inner ist die vektor produkt g_s (x+y)- Berücksichtigung der Vektor richtungen
   - inner(sig0_ufl, eps(v)) * dx(SOIL))
   #- inner(sig0_ufl, eps(v)) * dx(LINING)) entfernt weil eigentlich schale ncith da, somit keine wirkung auf erde

# ══════════════════════════════ ════════════════════════════════════════════════
# 7.  LÖSUNG  –  direkte LU-Zerlegung (MUMPS)
# ══════════════════════════════════════════════════════════════════════════════

print("  Systemaufstellung & Lösung … ", end="", flush=True)    #systemaufstellung & lösen: in die konsole aus, damit sieht das solver startet. flush=true erzwingt sofortige ausgabe (wichtig, weil danach eine rechenintensive operation kommt)

problem = LinearProblem(
    a, L, bcs=bcs,      #a: Bilinearform (linke seite Stiefigkeitsmatrix); L- Linearform (rechte Seite: Lastvektor zb eigengewicht; bcs- dirichlet Randbedingungen aus Abschnitt 5 eingearbeitet)
    petsc_options_prefix="tunnel_solver",       #a ist was berechnet wird (auf seite der unebkannte) steifigkeit mal verformung = kraft
    petsc_options={     #konfiguration des linearen solvers
        "ksp_type":                  "preonly", #kein iterativer Krylov-löser, nur preconditioner selbst löst. Preconditioner ist für das erste iterativen prozess notwendig, aber hier wird direkt gelöst, deshalb preonly. 
                    #prüfen- wollen vielleicht krylov solver iterativ für berechnung
        "pc_type":                   "lu",  #direkte LU zerlegung (kein iterationsverfahren)
        "pc_factor_mat_solver_type": "mumps", #benutz MUMPS- hochperformanter parallelen Direkt Solver für sparse matrizen
    },
)
u = problem.solve() #löst das lineares system mit K* u = f. erfebnis u ist das verschiebungsfeld mit je 2 DOFs pro knoten (ux + uy)
u.name = "Verschiebung" # ist der namen für export (zB in paraview)
u.x.scatter_forward()

u_arr  = u.x.array.reshape(-1, 2)   #u.x.array ist ein 1D-array mit allen DOFs (zB [ux0, uy0, ux1, uy1, ...]), reshape(-1, 2) formt es in ein 2D-array um mit 2 Spalten (eine für ux und eine für uy)
ux_max = float(np.abs(u_arr[:, 0]).max())
uy_max = float(np.abs(u_arr[:, 1]).max())
print("OK")
print(f"  max|ux| = {ux_max*1e3:.4f} mm   |   max|uy| = {uy_max*1e3:.4f} mm")   #gibt die maximale verschiebugn in mm an (* 1e3 um von m in mm umzurechnen)

# ══════════════════════════════════════════════════════════════════════════════
# [DATA EXPORT]  write this run's displacement field + a manifest row
# ══════════════════════════════════════════════════════════════════════════════
# Coordinates of the P2 displacement DOFs (one (x,y) per node), matching u row-for-row.
# Stored PER RUN so that variable geometry (a different mesh each run) will work unchanged.
# NB: serial run assumed (mpirun -n 1); parallel would need gathering across ranks.
_coords = V.tabulate_dof_coordinates()[:, :2].copy()          # (N, 2)  metres
_u      = u.x.array.reshape(-1, 2).astype(np.float64).copy()  # (N, 2)  metres  (ux, uy)
assert _coords.shape[0] == _u.shape[0], \
    f"[DATA EXPORT] coord/displacement row mismatch: {_coords.shape} vs {_u.shape} (serial run?)"

_outdir  = os.path.abspath(args.outdir)
_outputs = os.path.join(_outdir, "outputs")
os.makedirs(_outputs, exist_ok=True)
_tag  = f"run_{args.run_id:04d}"
_npz  = os.path.join(_outputs, _tag + ".npz")
if os.path.exists(_npz) and not args.overwrite:
    raise SystemExit(f"[DATA EXPORT] {_npz} already exists. Use a new --run_id or pass --overwrite.")

# self-contained: the field AND the parameters that produced it, all in one file
np.savez_compressed(
    _npz,
    xy           = _coords.astype(np.float32),   # (N,2) [m]
    u            = _u,                            # (N,2) [m]  (ux, uy)
    E_MPa        = np.float64(args.E),
    nu           = np.float64(args.nu),
    gamma_kNm3   = np.float64(args.gamma),
    phi_deg      = np.float64(args.phi),
    cohesion_kPa = np.float64(args.cohesion),
    K0           = np.float64(K0_s),
    depth_m      = np.float64(args.depth),        # [GEO] Tiefe Tunnelachse
    t_liner_m    = np.float64(args.t_liner),      # [GEO] Schalendicke
    R_i_m        = np.float64(R_i),
)

# manifest.csv: upsert this run_id (re-running the same id replaces its row, no duplicates)
_manifest = os.path.join(_outdir, "manifest.csv")
_header = ["run_id", "file", "E_MPa", "nu", "gamma_kNm3", "phi_deg",
           "cohesion_kPa", "K0", "depth_m", "t_liner_m", "n_nodes", "max_ux_mm", "max_uy_mm"]
_row = {"run_id": args.run_id, "file": f"outputs/{_tag}.npz", "E_MPa": args.E, "nu": args.nu,
        "gamma_kNm3": args.gamma, "phi_deg": args.phi, "cohesion_kPa": args.cohesion,
        "K0": f"{K0_s:.4f}", "depth_m": args.depth, "t_liner_m": args.t_liner, "n_nodes": _coords.shape[0],
        "max_ux_mm": f"{ux_max*1e3:.4f}", "max_uy_mm": f"{uy_max*1e3:.4f}"}
_rows = {}
if os.path.exists(_manifest):
    with open(_manifest, newline="") as _f:
        for _r in csv.DictReader(_f):
            _rows[int(_r["run_id"])] = _r
_rows[args.run_id] = {k: str(_row[k]) for k in _header}
with open(_manifest, "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=_header)
    _w.writeheader()
    for _rid in sorted(_rows):
        _w.writerow(_rows[_rid])

print(f"[DATA EXPORT] wrote    {_npz}")
print(f"[DATA EXPORT] manifest {_manifest}")

if not args.full_output:
    # displacement-only mode (default): skip the stress / Mohr-Coulomb / XDMF sections below
    print(f"[run {args.run_id:04d}] done (displacement only).")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  SPANNUNGSAUSWERTUNG  –  DG0  (stückweise konstant, ein Wert je Element)
# ══════════════════════════════════════════════════════════════════════════════

print("  Spannungsauswertung … ", end="", flush=True)

# DG0-Raum für Spannungsfelder- Stückweise konstante pro zelle
DG0_elem = basix.ufl.element("DG", cell_type, 0)
DG0 = functionspace(domain, DG0_elem)

# Materialparameter als DG0-Felder (notwendig für σ(u, λ(x), μ(x)))
lam_f = Function(DG0);   mu_f = Function(DG0)
si = cell_tags.find(SOIL);    li = cell_tags.find(LINING);    vi = cell_tags.find(VOID)
lam_f.x.array[si] = lam_s;   lam_f.x.array[li] = lam_c
mu_f.x.array[si]  = mu_s;    mu_f.x.array[li]  = mu_c
lam_f.x.scatter_forward();   mu_f.x.scatter_forward()

# Spannungstensor (UFL-Ausdruck, mit ortsabhängigen Materialfeldern)
sig_expr = sigma(u, lam_f, mu_f)    # 2×2-UFL-Tensor
e_expr   = eps(u)

# In EVZ gilt:  εzz = 0  →  σzz = λ·(εxx + εyy)
szz_expr = lam_f * (e_expr[0, 0] + e_expr[1, 1])

def proj(expr, name: str) -> Function:
    """Interpoliert einen UFL-Ausdruck auf den DG0-Raum
       (Auswertung am Elementmittelpunkt → ein Wert je Zelle)."""
    f   = Function(DG0, name=name)
    pts = DG0.element.interpolation_points
    f.interpolate(fem.Expression(expr, pts))
    return f

# Kartesische Spannungskomponenten
sig_xx = proj(sig_expr[0, 0],  "sigma_xx")
sig_yy = proj(sig_expr[1, 1],  "sigma_yy")
sig_xy = proj(sig_expr[0, 1],  "tau_xy")
sig_zz = proj(szz_expr,        "sigma_zz")    # EVZ: Nebenspannung out-of-plane

# ── von-Mises-Spannung (vollständiges 3-D im EVZ) ────────────────────────────
# σ_vM = √( ½·[(σxx–σyy)² + (σyy–σzz)² + (σzz–σxx)² + 6τxy²] )
sxx = sig_expr[0, 0];  syy = sig_expr[1, 1]
sxy = sig_expr[0, 1];  szz = szz_expr

sig_vm = proj(
    sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2)),
    "sigma_vMises"
)

# ── Tangential- und Radialspannung (Zylinderkoordinaten um Tunnelachse) ───────
#
# Koordinaten relativ zum Tunnelmittelpunkt (xsp aus Abschn. 6 übernommen):
dr_x = xsp[0] - xc
dr_y = xsp[1] - yc
r_c  = sqrt(dr_x**2 + dr_y**2)     # Abstand von der Tunnelachse  [m]

# Einheitsvektoren (Zylinderbasis):
# e_r = (cosθ, sinθ)  mit  cosθ = dr_x/r,  sinθ = dr_y/r
cosT = dr_x / r_c
sinT = dr_y / r_c

# σ_θθ = σxx·sin²θ – 2τxy·sinθ·cosθ + σyy·cos²θ   (Tangentialspannung)
# σ_rr = σxx·cos²θ + 2τxy·sinθ·cosθ + σyy·sin²θ   (Radialspannung)
sig_tt = proj(
    sxx*sinT**2 - 2*sxy*sinT*cosT + syy*cosT**2,
    "sigma_tangential"
)
sig_rr = proj(
    sxx*cosT**2 + 2*sxy*sinT*cosT + syy*sinT**2,
    "sigma_radial"
)

# ── K0-Initialspannungen projizieren (DG0) → Gesamtspannungen bilden ─────────
# σ₀_xx = K0·γ·y,  σ₀_yy = γ·y,  σ₀_zz = K0·γ·y  (isotrop in horiz. Ebene)
sig0_xx_f = proj(K0_s * gam_s * xsp[1], "sigma0_xx")
sig0_yy_f = proj(       gam_s * xsp[1], "sigma0_yy")
sig0_zz_f = proj(K0_s * gam_s * xsp[1], "sigma0_zz")

# Gesamtspannung = Inkrementell (aus Δu) + Initial (σ₀) — nur im Boden!
# In der Schale: kein Initialspannungszustand (Neubeton → σ₀ = 0).
sxx_tot_arr = sig_xx.x.array.copy();  sxx_tot_arr[si] += sig0_xx_f.x.array[si]
syy_tot_arr = sig_yy.x.array.copy();  syy_tot_arr[si] += sig0_yy_f.x.array[si]
szz_tot_arr = sig_zz.x.array.copy();  szz_tot_arr[si] += sig0_zz_f.x.array[si]

sig_xx_tot = Function(DG0, name="sigma_xx_gesamt")
sig_yy_tot = Function(DG0, name="sigma_yy_gesamt")
sig_xx_tot.x.array[:] = sxx_tot_arr;  sig_xx_tot.x.scatter_forward()
sig_yy_tot.x.array[:] = syy_tot_arr;  sig_yy_tot.x.scatter_forward()

print("OK")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  MOHR-COULOMB-AUSNUTZUNG  (Boden, reine Nachrechnung – kein Return Mapping)
# ══════════════════════════════════════════════════════════════════════════════
#
# MC-Fließbedingung (Zugspannungen positiv):
#   f = σ₁ – σ₃ – 2c·cosφ + (σ₁+σ₃)·sinφ  ≤  0
#
# Ausnutzungsgrad:
#   η = (σ₁ – σ₃) / (2c·cosφ – (σ₁+σ₃)·sinφ)
#   η ≤ 1: im elastischen Bereich
#   η > 1: Fließgrenze überschritten → Modell ungültig, Plastizität erforderlich
#
# 3-D-Hauptspannungen im EVZ: σ1_2D, σ3_2D (in-plane) + σzz (out-of-plane)
# → alle drei sortieren, größte/kleinste in die MC-Bedingung einsetzen.

# MC-Nachweis mit GESAMTSPANNUNGEN (Inkrement + Geostatik)
sxx_a = sxx_tot_arr
syy_a = syy_tot_arr
szz_a = szz_tot_arr
sxy_a = sig_xy.x.array.copy()   # Schubspannung: keine Initialkomponente (τ₀=0)

mc_arr = np.zeros(len(sxx_a))
for i in si:
    s_avg = 0.5 * (sxx_a[i] + syy_a[i])
    s_rad = np.sqrt(((sxx_a[i] - syy_a[i]) * 0.5)**2 + sxy_a[i]**2)
    s1_ip = s_avg + s_rad    # in-plane: größte Hauptspannung
    s3_ip = s_avg - s_rad    # in-plane: kleinste Hauptspannung

    # 3-D: σzz kann Zwischen- oder Extremwert sein → aufsteigend sortieren
    principals = np.sort([s1_ip, s3_ip, szz_a[i]])
    s3, s1 = float(principals[0]), float(principals[2])   # min  |  max

    denom = 2.0*c_s*np.cos(phi_s) - (s1 + s3)*np.sin(phi_s)
    mc_arr[i] = (s1 - s3) / denom if abs(denom) > 1.0 else 0.0

mc_field = Function(DG0, name="MC_Ausnutzung")
mc_field.x.array[:] = mc_arr
mc_field.x.scatter_forward()

eta_max = float(mc_arr[si].max())

# ══════════════════════════════════════════════════════════════════════════════
# 10. VTK/XDMF-EXPORT  (je Feld eine XDMF-Datei → ParaView)
# ══════════════════════════════════════════════════════════════════════════════

print("  Ergebnisse schreiben … ", end="", flush=True)

def save_xdmf(fname: str, fn: Function):
    """Schreibt eine Funktion als XDMF-Datei (kompatibel mit ParaView)."""
    with io.XDMFFile(MPI.COMM_WORLD, f"results/{fname}.xdmf", "w") as f:
        f.write_mesh(domain)
        f.write_function(fn)

save_xdmf("disp",              u)           # Verschiebungsfeld Δu (nur Tunneleinbau)
save_xdmf("sigma_xx",          sig_xx)     # Inkrementelle Spannung (aus Δu)
save_xdmf("sigma_yy",          sig_yy)
save_xdmf("tau_xy",            sig_xy)
save_xdmf("sigma_zz",          sig_zz)
save_xdmf("sigma_xx_gesamt",   sig_xx_tot) # Gesamtspannung (Inkrement + K0-Geostatik)
save_xdmf("sigma_yy_gesamt",   sig_yy_tot)
save_xdmf("sigma_vMises",      sig_vm)
save_xdmf("sigma_tangential",  sig_tt)     # σ_θθ in der Schale
save_xdmf("sigma_radial",      sig_rr)     # σ_rr in der Schale
save_xdmf("MC_Ausnutzung",     mc_field)   # η mit Gesamtspannungen

# Materialzonen: 1=Boden, 2=Betonschale, 3=Hohlraum
# → In ParaView mit "Material_ID" einfärben, um alle drei Zonen zu unterscheiden
mat_field = Function(DG0, name="Material_ID")
mat_field.x.array[si] = float(SOIL)
mat_field.x.array[li] = float(LINING)
if len(vi) > 0:
    mat_field.x.array[vi] = float(VOID)
mat_field.x.scatter_forward()
save_xdmf("Material_ID", mat_field)

print("OK")

# ══════════════════════════════════════════════════════════════════════════════
# 11. ZUSAMMENFASSUNG
# ══════════════════════════════════════════════════════════════════════════════

# Schalen-Kennwerte (nur Lining-Elemente)
stt_lin = sig_tt.x.array[li]
svm_lin = sig_vm.x.array[li]

# Näherungsweise Normalkraft: N ≈ σ_θθ · t  (Membranannahme, Mittelwert)
N_mean      = float(np.mean(stt_lin)) * t_lin      # N/m
stt_max_abs = float(np.max(np.abs(stt_lin)))       # Pa

print()
print("═" * 64)
print("  ERGEBNISSE  –  ÜBERSICHT")
print("═" * 64)
print(f"  Geometrie  : {W:.0f}×{H:.0f} m | Achse ({xc:.0f}|{yc:.0f}) m "
      f"| D = {2*R_i:.0f} m | t = {t_lin:.2f} m")
print(f"  Netz       : {nc:,} Elemente  |  {nn:,} Knoten  (P2)")
print()
print("  ── Verschiebungen  Δu  (NUR aus Tunneleinbau, K0-Verfahren) ────────")
print(f"     max|ux|  =  {ux_max*1e3:9.4f} mm")
print(f"     max|uy|  =  {uy_max*1e3:9.4f} mm   ← dominante Setzungskomponente")
print(f"     K0 (Jaky) = {K0_s:.4f}   [1 – sin({np.degrees(phi_s):.1f}°)]")
print()
print("  ── Betonschale  ─────────────────────────────────────────────────────")
print(f"     |σ_θθ|_max  =  {stt_max_abs/1e3:8.1f}  kN/m²  (Tangentialspannung)")
print(f"     N_Mittel    ≈  {N_mean/1e3:8.1f}  kN/m   (Normalkraft, Membrananteil)")
print(f"     σ_vM,max    =  {float(np.max(svm_lin))/1e6:8.3f}  MPa")
print()
print("  ── Boden  –  Mohr-Coulomb  ──────────────────────────────────────────")
if eta_max > 1.0:
    print(f"     η_max = {eta_max:.3f}  ⚠  FLIESSGRENZE ÜBERSCHRITTEN!")
    print(f"             Linear-elastisches Modell nicht ausreichend.")
    print(f"             → Elastoplastisches Modell (MC-Return-Mapping) nötig.")
else:
    print(f"     η_max = {eta_max:.3f}  ✓  Boden bleibt im elastischen Bereich.")
print()
print("  ── Ergebnisdateien  (in ParaView öffnen)  ───────────────────────────")
for fn in ["disp", "sigma_xx", "sigma_yy", "sigma_xx_gesamt", "sigma_yy_gesamt",
           "tau_xy", "sigma_zz", "sigma_vMises",
           "sigma_tangential", "sigma_radial", "MC_Ausnutzung", "Material_ID"]:
    print(f"     results/{fn}.xdmf")
print()
print("  Tipp ParaView: File → Open → Mehrere .xdmf gleichzeitig laden,")
print("  dann 'Warp by Vector' (Faktor ~200) für Verformungsdarstellung.")
print("  Material_ID: 1=Boden | 2=Betonschale | 3=Hohlraum")
print("═" * 64)
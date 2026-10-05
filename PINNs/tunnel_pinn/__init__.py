"""
PyTorch-Implementierung des Tunnel-PINN.

Module:
    tunnel_base.py    -- Konfiguration (TunnelConfig), Physik-Helfer, Sampling, autograd
    tunnel_twonet.py  -- Zwei-Netz-Modell mit Grenzflaechenkopplung (reiner Vorwaertsloeser)
    tunnel_data.py    -- Laden der FEM-Daten, hybrides Training, Speichern/Abfragen, VTU-Export
"""

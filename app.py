# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Konstanten
# =========================
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K)
VMOLAR_STP_L_PER_MOL = 22.414  # L/mol bei 0°C, 1 bar (für "Normvolumen"-Darstellung)

# =========================
# Pumpenkennlinien (Einphasen) - Beispiel
# =========================
PUMPS = [
    {"id": "P1", "Qw": [0, 15, 30, 45, 60], "Hw": [55, 53, 48, 40, 28],
     "eta": [0.28, 0.52, 0.68, 0.66, 0.52], "Pw": [1.1, 3.9, 5.8, 6.2, 7.3]},
    {"id": "P2", "Qw": [0, 20, 40, 60, 80], "Hw": [48, 46, 40, 30, 18],
     "eta": [0.30, 0.60, 0.72, 0.68, 0.55], "Pw": [1.5, 4.2, 7.4, 8.5, 9.2]},
    {"id": "P3", "Qw": [0, 30, 60, 90, 120], "Hw": [42, 41, 36, 26, 14],
     "eta": [0.25, 0.55, 0.73, 0.70, 0.58], "Pw": [1.8, 6.0, 10.3, 11.0, 10.5]},
    {"id": "P4", "Qw": [0, 15, 30, 45, 60], "Hw": [70, 68, 62, 52, 40],
     "eta": [0.22, 0.48, 0.66, 0.65, 0.50], "Pw": [2.0, 5.5, 9.0, 10.5, 11.5]},
    {"id": "P5", "Qw": [0, 25, 50, 75, 100], "Hw": [46, 44, 38, 28, 16],
     "eta": [0.30, 0.62, 0.75, 0.72, 0.60], "Pw": [1.4, 4.5, 7.5, 8.0, 7.8]},
]

MEDIA = {
    "Wasser (20°C)": (998.0, 1.0),
    "Wasser (60°C)": (983.0, 0.47),
    "Glykol 30% (20°C)": (1040.0, 3.5),
    "Hydrauliköl ISO VG 32 (40°C)": (860.0, 32.0),
    "Hydrauliköl ISO VG 46 (40°C)": (870.0, 46.0),
    "Hydrauliköl ISO VG 68 (40°C)": (880.0, 68.0),
}
RHO_WATER_REF = 998.0  # Referenz für "Pw"-Skalierung

# =========================
# Mehrphasen-Pumpen (Kennlinien als ∆p in bar) - Beispiel
# Q in m³/h, p in bar
# =========================
MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "p_max_bar": 9,
        "GVF_max": 0.40,  # 40%
        "curves_p_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "p": [8.6, 8.5, 8.2, 7.6, 6.8, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "p": [8.4, 8.3, 8.0, 7.3, 6.3, 4.6]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "p": [8.2, 8.0, 7.6, 6.7, 5.6, 4.0]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "p": [7.8, 7.5, 6.9, 5.8, 4.6, 3.2]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.2, 2.7, 3.3, 4.0, 4.6, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.1, 2.6, 3.2, 3.9, 4.5, 4.9]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "P": [2.0, 2.5, 3.1, 3.8, 4.4, 4.8]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "P": [1.9, 2.4, 3.0, 3.6, 4.1, 4.5]},
        }
    },
    {
        "id": "MPH-100",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 50,
        "p_max_bar": 20,
        "GVF_max": 0.40,
        "curves_p_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "p": [18.8, 18.5, 17.8, 16.0, 13.5, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "p": [18.2, 18.0, 17.0, 15.2, 12.6, 9.2]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "p": [17.5, 17.2, 16.0, 14.0, 11.5, 8.2]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "p": [16.5, 16.0, 14.8, 12.8, 10.0, 7.0]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "P": [3.0, 4.2, 5.8, 7.5, 9.0, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "P": [2.9, 4.1, 5.7, 7.3, 8.8, 9.8]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.8, 4.0, 5.5, 7.1, 8.6, 9.5]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.6, 3.8, 5.2, 6.8, 8.2, 9.0]},
        }
    },
    {
        "id": "MPH-200",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 100,
        "p_max_bar": 30,
        "GVF_max": 0.40,
        "curves_p_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "p": [28.5, 28.0, 26.5, 23.5, 19.0, 14.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "p": [27.8, 27.0, 25.3, 22.0, 17.8, 13.0]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "p": [26.8, 26.0, 24.0, 20.8, 16.5, 12.0]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "p": [25.5, 24.6, 22.3, 19.0, 14.8, 10.5]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.5, 7.0, 9.5, 12.5, 15.5, 18.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.3, 6.8, 9.2, 12.1, 15.0, 17.4]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "P": [5.1, 6.5, 8.8, 11.6, 14.2, 16.5]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.8, 6.2, 8.3, 10.9, 13.5, 15.6]},
        }
    }
]

# =========================
# ATEX Datenbank (vereinfacht)
# =========================
ATEX_MOTORS = [
    {
        "id": "Standard Zone 2 (ec)",
        "marking": "II 3G Ex ec IIC T3 Gc",
        "zone_suitable": [2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "3G",
        "description": "Standardlösung für Zone 2. Wählbar, wenn T3 (200°C) ausreicht und Medientemperatur < 185°C ist."
    },
    {
        "id": "Zone 1 (eb)",
        "marking": "II 2G Ex eb IIC T3 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "2G",
        "description": "Standardlösung für Zone 1 ('Erhöhte Sicherheit'). Wählbar, wenn T3 ausreichend ist."
    },
    {
        "id": "Zone 1 (db eb) T4",
        "marking": "II 2G Ex db eb IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "description": "Für Zone 1 mit strengeren Temperaturanforderungen (T4). Motor druckfest (db), Anschlusskasten erhöhte Sicherheit (eb)."
    },
    {
        "id": "Zone 1 (db) T4",
        "marking": "II 2G Ex db IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "description": "Für Zone 1 (T4). Vollständig druckfeste Kapselung inkl. Anschlusskasten."
    }
]

# Henry-Parameter (Beispiel, empirisch grob)
# H(T) = A * exp(B*(1/T - 1/T0))
HENRY_CONSTANTS = {
    "Luft": {"A": 1100.0, "B": 1500},  # A bewusst etwas niedriger -> steilere Diagonalen wie in deiner Vorgabe
    "CO2": {"A": 29.4, "B": 2400},
    "O2": {"A": 1500.0, "B": 1500},
    "N2": {"A": 1650.0, "B": 1300},
    "CH4": {"A": 1400.0, "B": 1600},
}

# =========================
# Helper
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def linspace(start, stop, num):
    if num <= 0:
        return []
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]

def m3h_to_lmin(m3h):
    return m3h * 1000.0 / 60.0

def interp_clamped(x, xs, ys):
    """Lineare Interpolation mit Clamping auf Randwerte."""
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if abs(x1 - x0) < 1e-12:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================
# Viskositätskorrektur (HI-ähnlich, pragmatisch – stabil & nachvollziehbar)
# =========================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """
    Vereinfachte HI-ähnliche B-Kennzahl.
    Wichtig: B steigt mit Viskosität und sinkt mit zunehmendem Q/H (vereinfacht).
    """
    Q = max(Q_m3h, 1e-6)
    H = max(H_m, 1e-6)
    nu = max(nu_cSt, 1e-6)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    """
    Liefert (CQ, CH, Ceta) ~ (Förderstrom-, Förderhöhen-, Wirkungsgradfaktor)
    CQ, CH, Ceta = 1.0 bei B<=1 (kaum Effekte).
    """
    if B <= 1.0:
        return 1.0, 1.0, 1.0

    logB = math.log10(B)

    # CQ: Förderstrom sinkt moderat
    CQ = 1.0 - 0.08 * logB - 0.015 * (logB ** 2)
    CQ = clamp(CQ, 0.70, 1.0)

    # CH: Förderhöhe sinkt (stärker als CQ)
    CH = math.exp(-0.165 * (logB ** 2.2))
    CH = clamp(CH, 0.35, 1.0)

    # Ceta: Wirkungsgrad sinkt
    Ceta = 1.0 - 0.25 * logB - 0.05 * (logB ** 2)
    Ceta = clamp(Ceta, 0.10, 1.0)

    return CQ, CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt, iters=6):
    """
    Umrechnung eines geforderten viskosen Betriebspunktes (Q_vis,H_vis)
    auf einen "äquivalenten" Wasserpunkt (Q_w,H_w), damit Kennlinienvergleich
    auf Wasserkennlinien erfolgen kann.

    Idee (stabil & nachvollziehbar):
    - Für einen Wasserpunkt (Q_w,H_w) gelten Korrekturen CQ,CH:
        Q_vis ≈ Q_w * CQ
        H_vis ≈ H_w * CH
    - CQ,CH hängen aber über B von Q_w,H_w ab -> Iteration.
    """
    Qw = max(Q_vis, 1e-6)
    Hw = max(H_vis, 1e-6)

    last = None
    for _ in range(iters):
        B = compute_B_HI(Qw, Hw, nu_cSt)
        CQ, CH, Ceta = viscosity_correction_factors(B)

        Qw_new = Q_vis / max(CQ, 1e-6)
        Hw_new = H_vis / max(CH, 1e-6)

        # Dämpfung gegen Schwingen
        Qw = 0.6 * Qw + 0.4 * Qw_new
        Hw = 0.6 * Hw + 0.4 * Hw_new

        last = (B, CQ, CH, Ceta)

    B, CQ, CH, Ceta = last
    return {
        "Q_water": Qw,
        "H_water": Hw,
        "B": B,
        "CQ": CQ,
        "CH": CH,
        "Ceta": Ceta
    }

def generate_viscous_curve(pump, nu_cSt, rho):
    """
    Erzeuge viskose Kennlinien aus Wasserkennlinien.
    - Q_vis = Q_w * CQ(Q_w,H_w,nu)
    - H_vis = H_w * CH(...)
    - eta_vis = eta_w * Ceta(...)
    - P_vis: Skalierung von Pw über Ceta (=> viskos typ. höher)
      P_vis = Pw * (rho/RHO_WATER_REF) / Ceta
    """
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w, P_w in zip(pump["Qw"], pump["Hw"], pump["eta"], pump["Pw"]):
        B = compute_B_HI(Q_w, H_w, nu_cSt)
        CQ, CH, Ceta = viscosity_correction_factors(B)
        Q_vis.append(Q_w * CQ)
        H_vis.append(H_w * CH)
        eta_vis.append(clamp(eta_w * Ceta, 1e-6, 0.999))
        P_vis.append(P_w * (rho / RHO_WATER_REF) / max(Ceta, 1e-6))
    return Q_vis, H_vis, eta_vis, P_vis

def choose_best_pump(pumps, Q_water, H_water, allow_out_of_range=True):
    """
    Auswahlkriterium:
    - Vergleich auf Wasserkennlinie (Q_water/H_water)
    - Minimiert |H(Q)-H_water|, mit Strafterm wenn Q außerhalb Kennlinie.
    - Bei Gleichstand -> höhere η bevorzugen.
    """
    best = None
    for p in pumps:
        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        in_range = (qmin <= Q_water <= qmax)
        if (not in_range) and (not allow_out_of_range):
            continue

        Q_eval = clamp(Q_water, qmin, qmax)
        penalty = 0.0 if in_range else abs(Q_water - Q_eval) / max(qmax - qmin, 1e-9) * 10.0

        H_at = interp_clamped(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp_clamped(Q_eval, p["Qw"], p["eta"])

        score = abs(H_at - H_water) + penalty
        cand = {
            "id": p["id"], "pump": p, "in_range": in_range, "Q_eval": Q_eval,
            "H_at": H_at, "eta_at": eta_at, "errH": abs(H_at - H_water), "score": score
        }

        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    return best

# =========================
# Henry / Löslichkeit
# =========================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1100.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def solubility_mol_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """C [mol/L] = p_partial / H(T)."""
    p = max(p_bar_abs, 1e-6)
    H = henry_constant(gas, T_celsius)
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    return p_partial / max(H, 1e-12)

def solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Normvolumen-Darstellung (diagonal):
    V_N [cm³/L] = C [mol/L] * V_molar(STP) [L/mol] * 1000
    => linear ~ p, daher Diagonalen wie in deiner Vorgabe.
    """
    C = solubility_mol_per_L(gas, p_bar_abs, T_celsius, y_gas=y_gas)
    return C * VMOLAR_STP_L_PER_MOL * 1000.0

def solubility_curve_diagonal(gas, T_celsius, p_max, y_gas=1.0):
    ps = linspace(0.1, p_max, 160)
    ys = [solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) for p in ps]
    return ps, ys

def gvf_to_Vgas_per_Lliq(gvf_pct):
    """
    Setze V_liq = 1 L.
    GVF = V_gas / (V_gas + V_liq)
    => V_gas = GVF/(1-GVF) * V_liq
    """
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    return (gvf / (1.0 - gvf)) * 1.0  # [L_gas / L_liq] bei der Referenzbedingung, bei der GVF gemessen wurde

def free_gvf_from_total_and_dissolved(gvf_total_pct, dissolved_cm3N_per_L):
    """
    Freier GVF am Saugpunkt aus Gesamt-GVF und maximal löslichem Gas.
    Vereinfachte Gegenüberstellung in Normvolumen zur robusten Worst-Case-Schätzung.
    """
    Vliq = 1.0
    Vgas_total = gvf_to_Vgas_per_Lliq(gvf_total_pct) * Vliq  # L/L
    Vgas_diss_max = max(dissolved_cm3N_per_L, 0.0) / 1000.0 * Vliq
    Vgas_free = max(0.0, Vgas_total - Vgas_diss_max)
    gvf_free = Vgas_free / max(Vgas_free + Vliq, 1e-12)
    return 100.0 * gvf_free

# =========================
# Mehrphasen: Affinität (Q~n, Δp~n², P~n³)
# =========================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-9)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2)

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_target,
                               n_min=0.50, n_max=1.20, tol=1e-3, iters=60):
    """
    Suche n_ratio so, dass Δp_scaled(Q_req, n_ratio) ≈ dp_target.
    Gibt None zurück, wenn Ziel im Suchintervall nicht erreichbar ist.
    """
    f_min = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_min) - dp_target
    f_max = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_max) - dp_target
    if f_min * f_max > 0:
        return None

    lo, hi = n_min, n_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, mid) - dp_target
        if abs(f_mid) <= tol:
            return mid
        if f_min * f_mid <= 0:
            hi = mid
            f_max = f_mid
        else:
            lo = mid
            f_min = f_mid
    return 0.5 * (lo + hi)

def choose_gvf_curve_key_worstcase(curves_dict, gvf_free_req_pct):
    """Worst Case: nächsthöhere GVF-Kurve (ceiling)."""
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def choose_best_mph_pump(pumps, Q_req, dp_req, gvf_free_req_pct, dp_margin_pct=10.0):
    """
    Normlogisch:
    - Δp_req = p_d - p_s
    - Kennlinie in Δp(Q) für gegebenen (freien) GVF
    - Worst Case: nächsthöhere GVF-Kurve
    - Reserve: dp_target = dp_req*(1+dp_margin)
    - optional Drehzahl (Affinität)
    """
    dp_target = dp_req * (1.0 + dp_margin_pct / 100.0)
    best = None

    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        if Q_req > pump["Q_max_m3h"] * 1.05:
            continue

        gvf_key = choose_gvf_curve_key_worstcase(pump["curves_p_vs_Q"], gvf_free_req_pct)
        curve = pump["curves_p_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]

        Qmin, Qmax = min(curve["Q"]), max(curve["Q"])
        in_curve = (Qmin <= Q_req <= Qmax)

        candidates = []

        # A) Nenndrehzahl
        if in_curve:
            dp_avail_nom = interp_clamped(Q_req, curve["Q"], curve["p"])
            if dp_avail_nom >= dp_target - 1e-6:
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail_nom - dp_target) + abs(gvf_key - gvf_free_req_pct) * 0.20
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_avail_nom,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "mode": "Nenndrehzahl",
                    "dp_reserve": dp_avail_nom - dp_req,
                    "dp_target": dp_target,
                    "score": score
                })

        # B) Drehzahl-Anpassung (Bisektion auf dp_target)
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["p"], Q_req, dp_target)
        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if Qmin <= Q_base <= Qmax:
                dp_scaled = _dp_scaled_at_Q(curve["Q"], curve["p"], Q_req, n_ratio)
                P_base = interp_clamped(Q_base, power_curve["Q"], power_curve["P"])
                P_scaled = P_base * (n_ratio ** 3)
                score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.20
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_scaled,
                    "P_required": P_scaled,
                    "n_ratio": n_ratio,
                    "mode": f"Drehzahl {n_ratio*100:.1f}%",
                    "dp_reserve": dp_scaled - dp_req,
                    "dp_target": dp_target,
                    "score": score
                })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Pumpenauslegung", layout="wide")
st.title("Pumpenauslegungstool")

if "page" not in st.session_state:
    st.session_state.page = "pump"

with st.sidebar:
    st.header("📍 Navigation")
    col1, col2, col3 = st.columns(3)
    if col1.button("Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if col2.button("Mehrphasen", use_container_width=True):
        st.session_state.page = "mph"
    if col3.button("ATEX", use_container_width=True):
        st.session_state.page = "atex"
    st.info(f"**Aktiv:** {st.session_state.page}")

# =========================================================
# PAGE 1: Einphasen (Viskosität)
# =========================================================
if st.session_state.page == "pump":
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### ⚙️ Eingaben (Einphasen)")
        Q_vis_req = st.number_input("Qᵥ, Förderstrom [m³/h]", 0.1, 300.0, 40.0, 1.0)
        H_vis_req = st.number_input("Hᵥ, Förderhöhe [m]", 0.1, 300.0, 35.0, 1.0)

        mk = st.selectbox("Medium", list(MEDIA.keys()), 0)
        rho_def, nu_def = MEDIA[mk]
        rho = st.number_input("ρ [kg/m³]", 1.0, 2000.0, float(rho_def), 5.0)
        nu = st.number_input("ν [cSt]", 0.1, 1000.0, float(nu_def), 0.5)

        allow_out = st.checkbox("Auswahl außerhalb Kennlinie zulassen", True)
        reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)

    # Umrechnung viskos -> Wasser (iterativ)
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu, iters=6)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CQ = conv["CQ"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]

    # Wasserwerte am Auslegungspunkt (auf Wasserkennlinie)
    H_at = best["H_at"]
    eta_water = best["eta_at"]
    P_water_kW_op = interp_clamped(best["Q_eval"], p["Qw"], p["Pw"])

    # Viskos (abschließend am gleichen Wasservergleichspunkt)
    eta_vis = clamp(eta_water * Ceta, 1e-6, 0.999)
    P_vis_kW = P_water_kW_op * (rho / RHO_WATER_REF) / max(Ceta, 1e-6)
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    # viskose Kennlinienpunkte
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)

    with right:
        st.markdown("### 📊 Umrechnung viskos → Wasser (für Kennlinienvergleich)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qᵥ (gegeben)", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")
        c2.metric("Hᵥ (gegeben)", f"{H_vis_req:.2f} m")
        c3.metric("Q_w (äquivalent)", f"{Q_water:.2f} m³/h", f"via CQ={CQ:.3f}")
        c4.metric("H_w (äquivalent)", f"{H_water:.2f} m", f"via CH={CH:.3f}")

        st.info(
            f"{'✅' if B < 1.0 else '⚠️'} **B = {B:.2f}** | "
            f"CQ={CQ:.3f}, CH={CH:.3f}, Cη={Ceta:.3f} "
            f"({'geringe' if B < 1.0 else 'aktive'} Viskositätseffekte)"
        )

        st.divider()
        st.markdown("### ✅ Auslegungsergebnis (Einphasen)")
        st.success(f"**Gewählte Pumpe: {best['id']}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Betriebspunkt", f"{Q_vis_req:.1f} m³/h", f"{H_vis_req:.1f} m")
        col2.metric("Vergleichspunkt (Wasser)", f"{Q_water:.1f} m³/h", f"{H_water:.1f} m")
        col3.metric("η (viskos)", f"{eta_vis:.3f}", f"η_w={eta_water:.3f}")
        col4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW", f"P_w={P_water_kW_op:.2f} kW")
        col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

        if not best["in_range"]:
            st.warning(
                f"⚠️ Q_w außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
                f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
            )

        st.divider()
        st.markdown("### 📈 Kennlinien")
        tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

        # Q-H
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
            ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
            ax1.scatter([Q_water], [H_at], marker="^", s=150, edgecolors="black",
                        linewidths=2, label="BP (Wasser, eval)", zorder=5)
            ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=200, linewidths=3,
                        label="BP (viskos, req)", zorder=5)
            ax1.set_xlabel("Q [m³/h]")
            ax1.set_ylabel("H [m]")
            ax1.set_title("Q-H Kennlinien")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            st.pyplot(fig1, clear_figure=True)

        # Q-η
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(p["Qw"], p["eta"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
            ax2.plot(Q_vis_curve, eta_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
            ax2.scatter([Q_water], [eta_water], marker="^", s=150, edgecolors="black",
                        linewidths=2, label="η (Wasser, eval)", zorder=5)
            ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=200, linewidths=3,
                        label="η (viskos, req)", zorder=5)
            ax2.set_xlabel("Q [m³/h]")
            ax2.set_ylabel("η [-]")
            ax2.set_title("Q-η Kennlinien")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

        # Q-P
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(p["Qw"], p["Pw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
            ax3.plot(Q_vis_curve, P_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
            ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, edgecolors="black",
                        linewidths=2, label="BP (Wasser, eval)", zorder=5)
            ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, linewidths=3,
                        label="BP (viskos, req)", zorder=5)
            ax3.set_xlabel("Q [m³/h]")
            ax3.set_ylabel("P Welle [kW]")
            ax3.set_title("Q-P Kennlinien (Wellenleistung)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            st.pyplot(fig3, clear_figure=True)

        with st.expander("📘 Rechenweg (ausführlich) – Viskosität", expanded=False):
            st.markdown(f"""
**Ziel:** Auswahl auf Wasserkennlinie, obwohl das Medium viskos ist.

#### 1) Gegeben (viskos)
- Förderstrom: **Qᵥ = {Q_vis_req:.3f} m³/h**
- Förderhöhe: **Hᵥ = {H_vis_req:.3f} m**
- Viskosität: **ν = {nu:.3f} cSt**
- Dichte: **ρ = {rho:.1f} kg/m³**

#### 2) Korrekturfaktoren (HI-ähnlich)
Wir nutzen eine stabile Näherung mit einer Kennzahl **B**, aus der Faktoren abgeleitet werden:
- **B = f(Q, H, ν)** (steigt mit ν, sinkt mit Q/H)
- daraus:
  - **CQ** (Förderstromfaktor, typ. < 1)
  - **CH** (Förderhöhenfaktor, typ. < 1)
  - **Cη** (Wirkungsgradfaktor, typ. < 1)

Aktueller Iterations-Endwert:
- **B = {B:.3f}**
- **CQ = {CQ:.3f}**, **CH = {CH:.3f}**, **Cη = {Ceta:.3f}**

#### 3) Iterative Umrechnung viskos → Wasserpunkt (Kennlinienvergleich)
Gesucht: Wasserpunkt (**Q_w**, **H_w**) so, dass bei viskoser Korrektur gilt:

- **Qᵥ ≈ Q_w · CQ(Q_w,H_w,ν)**
- **Hᵥ ≈ H_w · CH(Q_w,H_w,ν)**

Wir iterieren (6 Schritte, gedämpft):
- Start: Q_w = Qᵥ, H_w = Hᵥ  
- Update:
  - Q_w ← Qᵥ / CQ  
  - H_w ← Hᵥ / CH  

Ergebnis:
- **Q_w = {Q_water:.3f} m³/h**
- **H_w = {H_water:.3f} m**

#### 4) Pumpenauswahl auf Wasserkennlinie
Für jede Pumpe:
- Interpolation der Förderhöhe **H(Q_w)**
- Fehler: **|H(Q_w) − H_w|**
- optional Strafterm, wenn Q_w außerhalb Kennlinie

Gewählt:
- **{best['id']}** (H(Q_w)={H_at:.2f} m, η_w={eta_water:.3f})

#### 5) Leistung & Motor (robust über Pw-Kennlinie)
Damit die **Q-P-Kennlinie** realistisch bleibt, nutzen wir die hinterlegte Wasser-Wellenleistung **P_w(Q)**
und skalieren nur über den Wirkungsgradfaktor:

- Wasser-Wellenleistung am Vergleichspunkt:  
  **P_w = {P_water_kW_op:.3f} kW**

- Viskos (vereinfachte Skalierung):  
  **Pᵥ ≈ P_w · (ρ/ρ_ref) / Cη**  
  mit ρ_ref={RHO_WATER_REF:.0f} kg/m³  
  ⇒ **Pᵥ = {P_vis_kW:.3f} kW**

- Motorreserve: +{reserve_pct}%  
  ⇒ IEC-Motorgröße: **{P_motor_kW:.2f} kW**
""")

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit (p,T) + freier GVF + ∆p-Kennlinien + Auswahl")

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### ⚙️ Eingaben")
        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)
        y_gas = st.slider("Gasanteil (Partialdruckfaktor) y_gas [-]", 0.0, 1.0, 1.0, 0.05)

        st.divider()
        st.markdown("### Betriebspunkt (Hydraulik)")
        Q_req = st.number_input("Volumenstrom Q [m³/h]", 0.1, 150.0, 15.0, 1.0)
        p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 100.0, 2.0, 0.1)
        p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 200.0, 7.0, 0.1)
        dp_req = max(0.0, (p_discharge - p_suction))

        st.divider()
        st.markdown("### Gasgehalt")
        gvf_in = st.slider("Gesamt-GVF_in [%] (am Saugpunkt)", 0, 40, 10, 1)
        dp_margin_pct = st.slider("Reserve auf Δp [%] (Auslegung)", 0, 30, 10)

        st.divider()
        st.markdown("### Plotoptionen")
        show_temp_band = st.checkbox("Löslichkeit bei T-10 / T / T+10", value=True)
        show_overlay = st.checkbox("3. Plot (Overlay) anzeigen", value=True)

    # ---- Löslichkeit (diagonal in Normvolumen) am Saugpunkt
    diss_cm3N_L_ps = solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    gvf_free = free_gvf_from_total_and_dissolved(gvf_in, diss_cm3N_L_ps)

    # ---- Pumpenauswahl
    best = choose_best_mph_pump(MPH_PUMPS, Q_req, dp_req, gvf_free, dp_margin_pct=dp_margin_pct)

    with right:
        # 2 Plots nebeneinander: Solubility diagonal + Δp-Q
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Löslichkeit vs Druck (diagonal, Normvolumen)
        if show_temp_band:
            temp_variants = [temperature - 10, temperature, temperature + 10]
            temp_variants = [t for t in temp_variants if -10 <= t <= 150]
        else:
            temp_variants = [temperature]

        p_max = max(14.0, p_discharge * 1.15)

        color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        for i, T in enumerate(temp_variants):
            pressures, sol = solubility_curve_diagonal(gas_medium, T, p_max=p_max, y_gas=y_gas)
            ax1.plot(pressures, sol, "--", linewidth=2,
                     color=color_cycle[i % len(color_cycle)],
                     label=f"{gas_medium} {T:.0f}°C (y={y_gas:.2f})")

        ax1.scatter([p_suction], [diss_cm3N_L_ps], s=180, marker="o",
                    edgecolors="black", linewidths=2, label="Saugpunkt", zorder=5)
        ax1.axvline(p_suction, linestyle=":", linewidth=2)
        ax1.axvline(p_discharge, linestyle=":", linewidth=2)

        ax1.set_xlabel("Druck p_abs [bar]")
        ax1.set_ylabel("Löslichkeit [cm³(N)/L] (diagonal)")
        ax1.set_title("Gaslöslichkeit (Henry → Normvolumen)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # Plot 2: Δp-Q Kennlinien
        Q_req_lmin = m3h_to_lmin(Q_req)

        if best:
            pump = best["pump"]
            curves = pump["curves_p_vs_Q"]

            gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

            for gvf_key in sorted(curves.keys()):
                curve = curves[gvf_key]
                Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
                lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
                alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.5
                ax2.plot(Q_lmin, curve["p"], "o-", linewidth=lw, alpha=alpha,
                         color=gvf_colors.get(gvf_key, "gray"),
                         label=f"{pump['id']} ({gvf_key}% GVF)")

            ax2.scatter([Q_req_lmin], [dp_req], s=180, marker="o",
                        edgecolors="black", linewidths=2, label="Betriebspunkt (Δp_req)", zorder=5)

            ax2.set_xlabel("Q [L/min]")
            ax2.set_ylabel("Δp [bar]")
            ax2.set_title(f"Mehrphasen-Kennlinien (Δp): {pump['id']}")
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)
            ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
            ax2.set_ylim(0, pump["p_max_bar"] * 1.1)
        else:
            ax2.text(0.5, 0.5, "❌ Keine geeignete Pumpe gefunden",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=14)
            ax2.set_xlabel("Q [L/min]")
            ax2.set_ylabel("Δp [bar]")
            ax2.set_title("Mehrphasen-Kennlinien")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # ---- Ergebnisse
        st.divider()
        st.markdown("### ✅ Ergebnisse (normlogisch)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Q", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
        c2.metric("Δp_req", f"{dp_req:.2f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
        c3.metric("GVF_in", f"{gvf_in:.0f} %")
        c4.metric("Löslichkeit @ p_s", f"{diss_cm3N_L_ps:.1f} cm³(N)/L", f"T={temperature:.0f}°C")
        c5.metric("GVF_free (Worst Case)", f"{gvf_free:.1f} %", "für Kennlinie")

        if gvf_free > 0.0:
            st.warning("⚠️ Freies Gas vorhanden (GVF_free > 0). Kennlinienwahl erfolgt Worst Case auf nächsthöhere GVF-Kurve.")
        else:
            st.info("ℹ️ Aus dieser Abschätzung ergibt sich kein freies Gas (alles im Löslichkeitslimit).")

        if best:
            st.markdown("### 🔧 Empfohlene Pumpe")
            st.success(
                f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | "
                f"Modus: **{best['mode']}** | Δp_target={best['dp_target']:.2f} bar"
            )

            dpres = best["dp_available"] - dp_req
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Δp verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve ggü. Δp_req: {dpres:+.2f} bar")
            c2.metric("Leistung", f"{best['P_required']:.2f} kW")
            c3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
            c4.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

            if best["dp_available"] + 1e-3 < dp_req:
                st.warning("⚠️ Δp_req wird minimal unterschritten (Numerik/Toleranz). Erhöhe Reserve oder wähle größere Pumpe.")
        else:
            st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")
            st.markdown("""
**Typische Gründe:**
- Δp_req zu hoch für alle Pumpen/Kennlinien
- Q zu hoch für Pumpengrößenbereich
- GVF_free über Pumpengrenze
            """)

        st.divider()
        st.markdown("### 📋 Verfügbare Mehrphasenpumpen")
        cols = st.columns(len(MPH_PUMPS))
        for i, pmp in enumerate(MPH_PUMPS):
            with cols[i]:
                selected = bool(best) and best["pump"]["id"] == pmp["id"]
                st.success(f"✅ **{pmp['id']}**" if selected else f"**{pmp['id']}**")
                st.caption(f"Typ: {pmp['type']}")
                st.caption(f"Q_max: {pmp['Q_max_m3h']} m³/h")
                st.caption(f"Δp_max: {pmp['p_max_bar']} bar")
                st.caption(f"GVF_max: {pmp['GVF_max']*100:.0f}%")

        # ---- 3. Plot (Overlay)
        if show_overlay:
            st.divider()
            st.markdown("### 📈 Overlay: Löslichkeit (diagonal) + freies Gasvolumen (bei p)")

            figO, axO = plt.subplots(figsize=(14, 5.5))

            for i, T in enumerate(temp_variants):
                ps, sol = solubility_curve_diagonal(gas_medium, T, p_max=p_max, y_gas=y_gas)
                axO.plot(ps, sol, "--", linewidth=2,
                         color=color_cycle[i % len(color_cycle)],
                         label=f"Löslichkeit {gas_medium} {T:.0f}°C (ref p_s)")

            gvf_levels = sorted(set([10, 15, 20, int(gvf_in)]))
            gvf_colors = {10: "#9bbcff", 15: "#6f9cff", 20: "#2f6bff", int(gvf_in): "#1f4de3"}

            for gvf_pct in gvf_levels:
                Vgas_total_ps_L_per_L = gvf_to_Vgas_per_Lliq(gvf_pct)  # L/L bei p_s
                free_curve = []
                for p_abs in ps:
                    Vgas_total_cm3_L = Vgas_total_ps_L_per_L * (p_suction / max(p_abs, 1e-6)) * 1000.0
                    sol_cm3N_L = solubility_cm3N_per_L(gas_medium, p_abs, temperature, y_gas=y_gas)
                    free_cm3_L = max(0.0, Vgas_total_cm3_L - sol_cm3N_L)
                    free_curve.append(free_cm3_L)

                axO.plot(ps, free_curve, linewidth=2.5,
                         color=gvf_colors.get(gvf_pct, "tab:blue"),
                         label=f"{gvf_pct}% GVF (freies Gasvolumen/L bei p)")

            axO.scatter([p_suction], [diss_cm3N_L_ps], s=160, marker="o",
                        edgecolors="black", linewidths=2, label="Saugpunkt (Löslichkeit)", zorder=5)
            axO.axvline(p_suction, linestyle=":", linewidth=2)
            axO.axvline(p_discharge, linestyle=":", linewidth=2)

            axO.set_xlabel("Druck p_abs [bar]")
            axO.set_ylabel("Löslichkeit [cm³(N)/L] / freies Gasvolumen [cm³/L] (bei p)")
            axO.set_title("Overlay: Löslichkeit (diagonal) vs. freies Gasvolumen (komprimiert)")
            axO.grid(True, alpha=0.3)
            axO.legend(ncols=2, fontsize=9)
            st.pyplot(figO, clear_figure=True)

        with st.expander("📘 Rechenweg (ausführlich) – Mehrphase", expanded=False):
            dp_target = dp_req * (1.0 + dp_margin_pct / 100.0)
            st.markdown(f"""
#### 1) Druckanforderung (normlogisch)
- **Δp_req = p_d − p_s**
- p_s = {p_suction:.2f} bar abs  
- p_d = {p_discharge:.2f} bar abs  
⇒ **Δp_req = {dp_req:.3f} bar**

Reserve:
- {dp_margin_pct:.0f}%  
⇒ **Δp_target = {dp_target:.3f} bar**

#### 2) Löslichkeit am Saugpunkt (Henry → Normvolumen)
- **C(p,T) [mol/L] = p_partial / H(T)**  
- **V_N [cm³/L] = C · V_molar(STP) · 1000**  (→ diagonal in p)

Am Saugpunkt:
⇒ **{diss_cm3N_L_ps:.2f} cm³(N)/L**

#### 3) Freies Gas (Worst Case)
- **GVF_in = {gvf_in:.1f}%**  
- **GVF_free = {gvf_free:.2f}%**

#### 4) Kennlinienwahl & Drehzahl
- Worst Case: nächsthöhere GVF-Kurve ≥ GVF_free  
- optional: Affinität (Q~n, Δp~n², P~n³) → Bisektion auf Δp_target

**Wichtig zu „Δp_req erreicht?“**  
Wenn du Reserve > 0% setzt, wird auf **Δp_target** dimensioniert.  
Δp_req kann dann “erreicht” sein, obwohl Δp_target knapp nicht erreicht wird.
            """)

# =========================================================
# PAGE 3: ATEX
# =========================================================
elif st.session_state.page == "atex":
    st.subheader("⚡ ATEX-Motorauslegung")
    st.caption("Auslegung nach RL 2014/34/EU (vereinfachte Logik)")

    col_in, col_res = st.columns([1, 2])

    with col_in:
        st.markdown("### 1) Prozessdaten")
        P_req_input = st.number_input("Erf. Wellenleistung Pumpe [kW]", min_value=0.1, value=5.5, step=0.5)
        T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

        st.divider()
        st.markdown("### 2) Zone")
        atmosphere = st.radio("Atmosphäre", ["G (Gas)", "D (Staub)"], index=0)
        if atmosphere == "G (Gas)":
            zone_select = st.selectbox("Ex-Zone (Gas)", [0, 1, 2], index=2)
        else:
            zone_select = st.selectbox("Ex-Zone (Staub)", [20, 21, 22], index=2)

    with col_res:
        st.markdown("### 📋 ATEX-Konformitätsprüfung")
        valid_config = True

        if atmosphere == "D (Staub)":
            st.error("❌ Staub-Ex: hierfür ist hier kein Motor-Datensatz hinterlegt.")
            valid_config = False
        elif zone_select == 0:
            st.error("❌ Zone 0: hierfür ist hier kein Motor-Datensatz hinterlegt.")
            valid_config = False
        else:
            st.success(f"✅ Zone {zone_select} (Gas) ist grundsätzlich abbildbar.")

        if valid_config:
            st.markdown("#### Temperatur-Check")
            t_margin = 15.0

            suitable_motors = []
            for m in ATEX_MOTORS:
                if zone_select in m["zone_suitable"]:
                    if (m["t_max_surface"] - t_margin) >= T_medium:
                        suitable_motors.append(m)

            if not suitable_motors:
                st.error(f"❌ Kein Motor verfügbar für T_medium = {T_medium:.1f}°C (mit 15K Abstand).")
            else:
                st.markdown("#### Leistungsdimensionierung")
                P_motor_min = P_req_input * 1.15
                P_iec = motor_iec(P_motor_min)

                a, b, c = st.columns(3)
                a.metric("P_Pumpe", f"{P_req_input:.2f} kW")
                b.metric("P_min (+15%)", f"{P_motor_min:.2f} kW")
                c.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

                st.divider()
                st.markdown("### 🔧 Verfügbare ATEX-Motoren")
                selection = st.radio(
                    "Wählen Sie einen Motortyp:",
                    options=suitable_motors,
                    format_func=lambda x: f"{x['marking']} ({x['id']})"
                )

                if selection:
                    st.info(f"ℹ️ **Warum dieser Motor?**\n\n{selection['description']}")
                    st.success("✅ Gültige Konfiguration gefunden")

                    with st.expander("Technische Details", expanded=True):
                        st.markdown(f"""
- **Leistung:** {P_iec:.2f} kW (inkl. Reserve)
- **Kennzeichnung:** `{selection['marking']}`
- **Max. Oberfläche:** {selection['t_max_surface']:.1f}°C ({selection['temp_class']})
- **Medientemperatur:** {T_medium:.1f}°C
                        """)
                        delta_t = selection["t_max_surface"] - T_medium
                        st.caption(f"Temperaturabstand: {delta_t:.1f} K (Anforderung: ≥ 15 K)")

    with st.expander("ℹ️ Definition der Ex-Zonen (Kurz)"):
        st.markdown("""
| Zone (Gas) | Beschreibung |
|---|---|
| Zone 0 | ständig/über lange Zeit |
| Zone 1 | gelegentlich |
| Zone 2 | selten/kurzzeitig |
        """)

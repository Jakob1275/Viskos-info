# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Konstanten
# =========================================================
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K) (ideales Gas)
# Referenzzustand für "Normvolumen" (üblich in Datenblättern): 0°C, 1.01325 bar
T_N_K = 273.15
P_N_BAR = 1.01325
V_MOLAR_N_L_PER_MOL = R_BAR_L * T_N_K / P_N_BAR  # ≈ 22.414 L/mol

# =========================================================
# Beispiel-Datenbanken
# =========================================================

# ---------------------------
# Einphasen-Pumpenkennlinien (Wasserbasis) + η + P
# ---------------------------
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

# ---------------------------
# Mehrphasen-Pumpenkennlinien (∆p in bar über Q in m³/h)
# Kurvenkeys: GVF in Prozent (0, 5, 10, 15, ...)
# ---------------------------
MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "dp_max_bar": 9,
        "GVF_max": 0.4,  # 40% freies Gas
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.6, 8.5, 8.2, 7.6, 6.8, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.4, 8.3, 8.0, 7.3, 6.3, 4.6]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.2, 8.0, 7.6, 6.7, 5.6, 4.0]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "dp": [7.8, 7.5, 6.9, 5.8, 4.6, 3.2]},
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
        "dp_max_bar": 20,
        "GVF_max": 0.4,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "dp": [18.8, 18.5, 17.8, 16.0, 13.5, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "dp": [18.2, 18.0, 17.0, 15.2, 12.6, 9.2]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "dp": [17.5, 17.2, 16.0, 14.0, 11.5, 8.2]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "dp": [16.5, 16.0, 14.8, 12.8, 10.0, 7.0]},
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
        "dp_max_bar": 30,
        "GVF_max": 0.4,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "dp": [28.5, 28.0, 26.5, 23.5, 19.0, 14.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "dp": [27.8, 27.0, 25.3, 22.0, 17.8, 13.0]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "dp": [26.8, 26.0, 24.0, 20.8, 16.5, 12.0]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "dp": [25.5, 24.6, 22.3, 19.0, 14.8, 10.5]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.5, 7.0, 9.5, 12.5, 15.5, 18.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.3, 6.8, 9.2, 12.1, 15.0, 17.4]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "P": [5.1, 6.5, 8.8, 11.6, 14.2, 16.5]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.8, 6.2, 8.3, 10.9, 13.5, 15.6]},
        }
    }
]

# ---------------------------
# ATEX-Motoren (vereinfachtes Beispiel)
# ---------------------------
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

# ---------------------------
# Henry-Parameter (Beispieldaten; Praxis: validierte Tabellen verwenden)
# H(T) = A * exp( B * (1/T - 1/T0) )
# ---------------------------
HENRY_CONSTANTS = {
    "Luft": {"A": 1400.0, "B": 1500},
    "CO2": {"A": 29.4, "B": 2400},
    "O2": {"A": 1500.0, "B": 1500},
    "N2": {"A": 1650.0, "B": 1300},
    "CH4": {"A": 1400.0, "B": 1600},
}

# =========================================================
# Helper
# =========================================================
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
    """Lineare Interpolation mit Clamping"""
    if len(xs) < 2:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            dx = xs[i] - xs[i - 1]
            if abs(dx) < 1e-12:
                return ys[i]
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / dx
    return ys[-1]

def motor_iec(P_kW):
    """IEC-Nennleistungen (grobe Stufen, Beispiel)"""
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================================================
# Viskositätskorrektur (HI/ISO-Logik, pragmatische Näherung)
# Hinweis: Für "voll normkonform" müssten HI-/ISO-Kennfelder/Charts umgesetzt werden.
# =========================================================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """
    B-Kennzahl (ähnlich HI). Interne Nutzung zur Ableitung von CH und Cη.
    Rechenweg nutzt Q in gpm und H in ft.
    """
    Q = max(Q_m3h, 1e-9)
    H = max(H_m, 1e-9)
    nu = max(nu_cSt, 1e-9)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    """
    Näherungsformel für:
    - CH: Förderhöhen-Korrekturfaktor
    - Cη: Wirkungsgrad-Korrekturfaktor
    """
    if B <= 1.0:
        return 1.0, 1.0

    # CH fällt mit steigender "B" (stärkere Viskositätseffekte)
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = clamp(CH, 0.30, 1.00)

    # η-Korrektur (pragmatisch)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = clamp(Ceta, 0.10, 1.00)

    return CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt):
    """
    Umrechnung "viskoser Betriebspunkt" → "äquivalenter Wasserbetriebspunkt"
    (vereinfachte HI-Logik):
    - Q bleibt gleich
    - H_wasser = H_vis / CH
    """
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / max(CH, 1e-9)}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """
    Rückrechnung Wasserkennlinie → viskose Kennlinie (für Plot):
    - H_vis = H_wasser * CH
    - η_vis = η_wasser * Cη
    """
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return Q_water, H_water * CH, max(1e-6, eta_water * Ceta)

def generate_viscous_curve(pump, nu_cSt, rho):
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        # Hydraulische Leistung: ρ g Q H; Q in m³/s = (m³/h)/3600
        P_hyd_W = rho * G * (Q_v / 3600.0) * H_v
        P_shaft_kW = (P_hyd_W / max(eta_v, 1e-6)) / 1000.0
        Q_vis.append(Q_v)
        H_vis.append(H_v)
        eta_vis.append(eta_v)
        P_vis.append(P_shaft_kW)
    return Q_vis, H_vis, eta_vis, P_vis

def choose_best_pump(pumps, Q_water, H_water, allow_out_of_range=True):
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
            "id": p["id"],
            "pump": p,
            "in_range": in_range,
            "Q_eval": Q_eval,
            "H_at": H_at,
            "eta_at": eta_at,
            "errH": abs(H_at - H_water),
            "score": score,
        }

        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    return best

# =========================================================
# Henry / Gaslöslichkeit (Normvolumen -> diagonal)
# =========================================================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1.0 / T_K - 1.0 / T0_K))  # bar·L/mol

def dissolved_mol_per_Lliq(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry: C [mol/L_liq] = p_partial / H(T)
    p_partial = y_gas * p_abs
    """
    p = max(p_bar_abs, 1e-9)
    y = clamp(y_gas, 0.0, 1.0)
    H = max(henry_constant(gas, T_celsius), 1e-12)
    return (y * p) / H

def solubility_cm3N_per_Lliq(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Löslichkeit als "Normvolumen" (cm³_N pro Liter Flüssigkeit),
    daher: V_N = n * V_molar_N (konstant) -> Ergebnis linear mit p -> diagonal.
    """
    C = dissolved_mol_per_Lliq(gas, p_bar_abs, T_celsius, y_gas=y_gas)  # mol/L
    V_N_L_per_L = C * V_MOLAR_N_L_PER_MOL  # L_N/L
    return 1000.0 * V_N_L_per_L  # cm³_N/L

def solubility_curve_vs_pressure(gas, T_celsius, p_max=30, y_gas=1.0):
    pressures = linspace(0.0, p_max, 160)
    sol = [solubility_cm3N_per_Lliq(gas, p, T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol

# =========================================================
# Freier GVF aus Gesamt-GVF_in und Löslichkeitslimit (molar konsistent)
# =========================================================
def gvf_free_from_total_gvf(gas, gvf_in_pct, p_suction_bar_abs, T_celsius, y_gas=1.0):
    """
    Normlogik:
    - Wir beziehen uns auf 1 Liter Flüssigkeit am Saugpunkt.
    - GVF_in beschreibt Gas-Volumenanteil im Gemisch am Saugpunkt (p_s, T).
    - Daraus -> Gasvolumen V_gas,s [L] und Gasmenge n_total [mol] (ideales Gas).
    - Henry liefert n_diss_max [mol/L_liq] = p_partial/H(T).
    - Freies Gas: n_free = max(0, n_total - n_diss_max).
    - Zurück zu Volumen am Saugpunkt: V_free = n_free * R*T / p_s.
    - GVF_free = V_free / (V_free + V_liq).
    """
    gvf = clamp(gvf_in_pct / 100.0, 0.0, 0.999999)
    T_K = T_celsius + 273.15
    p_s = max(p_suction_bar_abs, 1e-9)

    V_liq = 1.0  # L
    V_gas_s = gvf / (1.0 - gvf) * V_liq  # L (am Saugpunkt)

    # Gasmenge am Saugpunkt (ideales Gas): n = p*V/(R*T)
    n_total = p_s * V_gas_s / (R_BAR_L * T_K)  # mol

    # maximal lösbare Gasmenge pro L Flüssigkeit
    n_diss_max = dissolved_mol_per_Lliq(gas, p_s, T_celsius, y_gas=y_gas) * V_liq  # mol

    n_free = max(0.0, n_total - n_diss_max)

    V_free = n_free * R_BAR_L * T_K / p_s  # L am Saugpunkt
    gvf_free = V_free / (V_free + V_liq)
    return 100.0 * gvf_free, {
        "V_liq_L": V_liq,
        "V_gas_total_L": V_gas_s,
        "n_total_mol": n_total,
        "n_diss_max_mol": n_diss_max,
        "n_free_mol": n_free,
        "V_gas_free_L": V_free,
    }

# =========================================================
# Mehrphasen: Affinitätsgesetze + Drehzahlsuche (Bisektion)
# =========================================================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    """
    Affinität:
    - Q ~ n  -> Q_base = Q_req / n_ratio
    - dp ~ n² -> dp_scaled = dp_base(Q_base) * n_ratio²
    """
    Q_base = Q_req / max(n_ratio, 1e-12)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2), Q_base, dp_base

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_target,
                               n_min=0.5, n_max=1.1, tol=1e-3, iters=70):
    """
    Finde n_ratio in [n_min, n_max], so dass dp_scaled(Q_req, n_ratio) = dp_target
    """
    f_min = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_min)[0] - dp_target
    f_max = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_max)[0] - dp_target
    if f_min * f_max > 0:
        return None

    lo, hi = n_min, n_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, mid)[0] - dp_target
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
    """
    Worst Case: nächsthöhere verfügbare GVF-Kurve (ceiling).
    Beispiel: GVF_free=8.7% -> wähle 10% Kurve.
    """
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def choose_best_mph_pump_normbased(pumps, Q_req, dp_req, gvf_free_req_pct,
                                  dp_margin=0.10):
    """
    Normlogische Auswahl:
    - dp_req: erforderliche Druckerhöhung (∆p) zwischen Saug- und Druckseite (bar)
    - dp_margin: Reserve auf ∆p (z.B. 10%)
    - gvf_free_req_pct: freier GVF am Saugpunkt (worst case)
    - Drehzahl wird berücksichtigt (Affinität, Bisektion).
    """
    best = None
    dp_target = dp_req * (1.0 + dp_margin)

    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        # Sicherheitsannahme: bis 110% Drehzahl
        if Q_req > pump["Q_max_m3h"] * 1.10:
            continue

        gvf_key = choose_gvf_curve_key_worstcase(pump["curves_dp_vs_Q"], gvf_free_req_pct)
        curve = pump["curves_dp_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]

        Qmin, Qmax = min(curve["Q"]), max(curve["Q"])
        in_curve = (Qmin <= Q_req <= Qmax)

        candidates = []

        # A) Nenndrehzahl
        if in_curve:
            dp_avail_nom = interp_clamped(Q_req, curve["Q"], curve["dp"])
            if dp_avail_nom >= dp_target:
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail_nom - dp_req) + abs(gvf_key - gvf_free_req_pct) * 0.25
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_avail_nom,
                    "dp_target": dp_target,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "mode": "Nenndrehzahl",
                    "dp_reserve": dp_avail_nom - dp_req,
                    "score": score,
                    "Q_base": Q_req,
                    "dp_base": dp_avail_nom,
                })

        # B) Drehzahl-Anpassung
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["dp"], Q_req, dp_target)
        if n_ratio is not None:
            dp_scaled, Q_base, dp_base = _dp_scaled_at_Q(curve["Q"], curve["dp"], Q_req, n_ratio)
            if Qmin <= Q_base <= Qmax:
                # Leistung: P ~ n³ (auf Basispunkt Q_base)
                P_base = interp_clamped(Q_base, power_curve["Q"], power_curve["P"])
                P_scaled = P_base * (n_ratio ** 3)

                score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.25
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_scaled,
                    "dp_target": dp_target,
                    "P_required": P_scaled,
                    "n_ratio": n_ratio,
                    "mode": f"Drehzahl {n_ratio*100:.1f}%",
                    "dp_reserve": dp_scaled - dp_req,
                    "score": score,
                    "Q_base": Q_base,
                    "dp_base": dp_base,
                })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

# =========================================================
# Dritte Grafik: Overlay (Solubility vs. "Gasbedarf" bei GVF)
# =========================================================
def gas_ratio_cm3_per_Lliq_vs_pressure(gvf_pct, p_suction_bar_abs, T_celsius, p_list_bar_abs):
    """
    Kurve "Gasvolumen pro Liter Flüssigkeit" in cm³/L (bei Betriebs-p,T),
    wenn am Saugpunkt der GVF vorliegt und Gas isotherm komprimiert wird.

    Annahme:
    - Bezugsvolumen: 1 L Flüssigkeit
    - am Saugpunkt: V_gas,s = gvf/(1-gvf) * 1L
    - Molenstrom pro L Flüssigkeit: n = p_s*V_gas,s/(R*T)
    - bei anderem Druck p: V_gas(p)=n*R*T/p
    """
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    T_K = T_celsius + 273.15
    p_s = max(p_suction_bar_abs, 1e-9)

    V_liq = 1.0
    V_gas_s = gvf / (1.0 - gvf) * V_liq  # L
    n = p_s * V_gas_s / (R_BAR_L * T_K)  # mol

    out = []
    for p in p_list_bar_abs:
        p = max(p, 1e-9)
        V_gas_p = n * R_BAR_L * T_K / p
        out.append(1000.0 * (V_gas_p / V_liq))  # cm³/L
    return out

# =========================================================
# Streamlit App
# =========================================================
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
    st.subheader("🔄 Einphasen: Pumpenauswahl mit Viskositätskorrektur (HI/ISO-Logik)")

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben (Einphasen)")
        Q_vis_req = st.number_input("Qᵥ, Förderstrom [m³/h]", 0.1, 300.0, 40.0, 1.0)
        H_vis_req = st.number_input("Hᵥ, Förderhöhe [m]", 0.1, 300.0, 35.0, 1.0)

        mk = st.selectbox("Medium", list(MEDIA.keys()), 0)
        rho_def, nu_def = MEDIA[mk]
        rho = st.number_input("ρ [kg/m³]", 1.0, 2000.0, float(rho_def), 5.0)
        nu = st.number_input("ν [cSt]", 0.1, 1000.0, float(nu_def), 0.5)

        allow_out = st.checkbox("Auswahl außerhalb Kennlinie", True)
        reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)

    # 1) Umrechnung Betriebspunkt viskos -> Wasser
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    st.info(
        f"{'✅' if B < 1.0 else '⚠️'} "
        f"B = {B:.2f} | CH = {CH:.3f} | Cη = {Ceta:.3f}"
    )

    st.markdown("### 📊 Umrechnung viskos → Wasser (für Kennlinienvergleich)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q_w", f"{Q_water:.2f} m³/h", "Annahme: Q bleibt konstant")
    c2.metric("H_w", f"{H_water:.2f} m", f"H_w = Hᵥ / CH")
    c3.metric("B", f"{B:.2f}", "HI-Kennzahl (Näherung)")
    c4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}", "Korrekturfaktoren")

    # 2) Pumpe wählen
    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]
    eta_water = best["eta_at"]
    eta_vis = max(1e-6, eta_water * Ceta)

    # 3) Leistung
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_shaft_kW = (P_hyd_W / eta_vis) / 1000.0
    P_motor_kW = motor_iec(P_shaft_kW * (1.0 + reserve_pct / 100.0))

    # Ergebnis
    st.divider()
    st.markdown("### ✅ **Auslegungsergebnis (Einphasen)**")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Qᵥ", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")
    col2.metric("Hᵥ", f"{H_vis_req:.2f} m")
    col3.metric("ηᵥ", f"{eta_vis:.3f}")
    col4.metric("P_hyd", f"{P_hyd_W/1000.0:.2f} kW")
    col5.metric("P_Welle", f"{P_shaft_kW:.2f} kW")
    col6.metric("IEC Motor", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}% Reserve")

    if not best["in_range"]:
        st.warning(
            f"⚠️ Q_w außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
            f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
        )

    # Kennlinien
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])

    st.divider()
    st.markdown("### 📈 Kennlinien (Wasser vs. viskos)")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        ax1.scatter([Q_water], [H_water], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser)", zorder=5)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=200, linewidths=3,
                    label="BP (viskos)", zorder=5)
        ax1.set_xlabel("Q [m³/h]")
        ax1.set_ylabel("H [m]")
        ax1.set_title("Q-H Kennlinien")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        st.pyplot(fig1, clear_figure=True)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(p["Qw"], p["eta"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax2.plot(Q_vis_curve, eta_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        ax2.scatter([Q_water], [eta_water], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=200, linewidths=3,
                    label="η (viskos)", zorder=5)
        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η Kennlinien")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(p["Qw"], p["Pw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser)", zorder=5)
        ax3.scatter([Q_vis_req], [P_shaft_kW], marker="x", s=200, linewidths=3,
                    label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien (Wellenleistung)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    # Langer Rechenweg
    with st.expander("📘 Rechenweg & Normbezug (ausführlich)", expanded=True):
        st.markdown("""
### Ziel der Viskositätskorrektur
Pumpenkennlinien werden i. d. R. **auf Wasser** vermessen (Abnahme/Prüfung z. B. nach ISO 9906).
Bei viskosen Medien verschieben sich Kennlinie und Wirkungsgrad. Um trotzdem **eine Wasserkennlinie**
zur Auswahl nutzen zu können, wird der viskose Betriebspunkt auf einen **äquivalenten Wasserbetriebspunkt** abgebildet.

**Typischer Norm-/Standardbezug (Hinweis):**
- **DIN EN ISO 9906** (Abnahmeprüfung Kreiselpumpen auf Wasserbasis)
- **ANSI/HI** Viscosity Correction (Hydraulic Institute)
- **ISO/TR 17766** (Viscosity corrections – technische Richtlinie, je nach Ausgabe)

> Hinweis: In diesem Tool ist die HI-Logik **als pragmatische Näherung** umgesetzt (CH/Cη aus B-Kennzahl).
Für strikte Normkonformität müssten die offiziellen HI/ISO-Kennfelder/Charts implementiert werden.
        """)

        st.markdown("### 1) Gegebene Betriebsdaten (viskoses Medium)")
        st.markdown(f"""
- Volumenstrom: **Qᵥ = {Q_vis_req:.3f} m³/h**
- Förderhöhe: **Hᵥ = {H_vis_req:.3f} m**
- Kinematische Viskosität: **ν = {nu:.2f} cSt**
- Dichte: **ρ = {rho:.1f} kg/m³**
        """)

        st.markdown("### 2) B-Kennzahl (HI-ähnlich) berechnen")
        st.markdown("""
Die HI-Korrektur arbeitet typischerweise mit Kennzahlen, die Viskosität, Volumenstrom und Förderhöhe kombinieren.
Im Code wird eine gebräuchliche Form (mit interner Umrechnung in **gpm** und **ft**) genutzt.
        """)
        st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q^{0.25}\cdot H^{0.375}}")
        st.markdown(f"""
Ergebnis: **B = {B:.3f}**

Interpretation:
- **B < 1** → Viskositätseffekt meist gering
- **B ≥ 1** → deutliche Kennlinienverschiebung möglich
        """)

        st.markdown("### 3) Korrekturfaktoren bestimmen")
        st.markdown("""
- **CH** korrigiert die Förderhöhe (Head) von Wasser → viskos bzw. zurück.
- **Cη** korrigiert den Wirkungsgrad.

Im Tool:
- Förderhöhenkorrektur: **H_w = Hᵥ / CH**
- Wirkungsgradkorrektur: **ηᵥ = η_w · Cη**
        """)
        st.latex(r"H_\mathrm{w} = \frac{H_\nu}{C_H}")
        st.latex(r"\eta_\nu = \eta_\mathrm{w}\cdot C_\eta")

        st.markdown(f"""
Ergebnisse:
- **CH = {CH:.4f}**
- **Cη = {Ceta:.4f}**
        """)

        st.markdown("### 4) Abbildung auf Wasserkennlinie (Auswahlpunkt)")
        st.markdown(f"""
- **Q_w = Qᵥ = {Q_water:.3f} m³/h** (Annahme der Näherung: Q bleibt gleich)
- **H_w = Hᵥ / CH = {H_vis_req:.3f} / {CH:.4f} = {H_water:.3f} m**
        """)

        st.markdown("### 5) Pumpenauswahl auf Wasserkennlinie")
        st.markdown(f"""
Die Auswahl erfolgt durch Interpolation auf den Wasserkennlinien der Pumpen:
- Gesucht: Pumpe mit **H(Q_w)** nahe **H_w** (und optional hoher η)
- Gewählt: **{best['id']}**
- Interpolierter Punkt:
  - **H(Q_w) = {best['H_at']:.3f} m**
  - **η_w(Q_w) = {eta_water:.3f}**
        """)

        st.markdown("### 6) Rückrechnung auf viskosen Betriebspunkt (für Leistung)")
        H_vis_calc = H_water * CH
        st.markdown(f"""
- **Hᵥ,calc = H_w · CH = {H_water:.3f} · {CH:.4f} = {H_vis_calc:.3f} m**
- **ηᵥ,calc = η_w · Cη = {eta_water:.3f} · {Ceta:.4f} = {eta_vis:.3f}**
        """)

        st.markdown("### 7) Leistungsberechnung")
        st.markdown("Hydraulische Leistung:")
        st.latex(r"P_\mathrm{hyd} = \rho \cdot g \cdot Q \cdot H")
        st.markdown(f"""
- **Q = {Q_vis_req:.3f} m³/h = {Q_vis_req/3600.0:.6f} m³/s**
- **P_hyd = {P_hyd_W:.1f} W = {P_hyd_W/1000.0:.3f} kW**
        """)

        st.markdown("Wellenleistung (Schaftleistung) mit Wirkungsgrad:")
        st.latex(r"P_\mathrm{Welle} = \frac{P_\mathrm{hyd}}{\eta_\nu}")
        st.markdown(f"- **P_Welle = {P_shaft_kW:.3f} kW**")

        st.markdown("Motor-Nennleistung mit Reserve:")
        st.latex(r"P_\mathrm{Motor,min} = P_\mathrm{Welle}\cdot (1+\mathrm{Reserve})")
        st.markdown(f"""
- Reserve = **{reserve_pct}%**
- **P_Motor,min = {P_shaft_kW*(1+reserve_pct/100.0):.3f} kW**
- IEC-Stufe: **{P_motor_kW:.2f} kW**
        """)

# =========================================================
# PAGE 2: Mehrphase (normlogisch ∆p + freier GVF + Drehzahl)
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("🧪 Mehrphasen: Löslichkeit (p,T) + freier GVF + ∆p-Kennlinien + Auswahl (mit Drehzahl)")

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Medium / Gas")
        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)
        y_gas = st.slider("Partialdruckfaktor y_gas [-]", 0.0, 1.0, 1.0, 0.05)

        st.divider()
        st.subheader("Betriebspunkt (Hydraulik)")
        Q_req = st.number_input("Volumenstrom Q (Flüssigkeitsstrom) [m³/h]", 0.1, 150.0, 15.0, 1.0)
        p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 100.0, 2.0, 0.1)
        p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 200.0, 7.0, 0.1)

        st.divider()
        st.subheader("GVF & Reserve")
        gvf_in = st.slider("Gesamt-GVF_in am Saugpunkt [%]", 0, 40, 10, 1)
        dp_margin = st.slider("∆p-Reserve [%]", 0, 30, 10, 1)

        st.divider()
        st.subheader("Plot")
        show_temp_band = st.checkbox("T-10/T/T+10", value=True)

    # ∆p-Anforderung ohne zusätzliche Verlust-Box (wie gefordert)
    dp_req = max(0.0, p_discharge - p_suction)

    # Löslichkeit am Saugpunkt (Normvolumen -> diagonal in Plot)
    sol_cm3N_L_at_suction = solubility_cm3N_per_Lliq(gas_medium, p_suction, temperature, y_gas=y_gas)

    # Freier GVF aus molarer Bilanz
    gvf_free, gvf_dbg = gvf_free_from_total_gvf(gas_medium, gvf_in, p_suction, temperature, y_gas=y_gas)

    # Pumpenauswahl
    best = choose_best_mph_pump_normbased(
        MPH_PUMPS,
        Q_req=Q_req,
        dp_req=dp_req,
        gvf_free_req_pct=gvf_free,
        dp_margin=dp_margin / 100.0
    )

    # =========================================
    # Grafik 1+2: zwei nebeneinander
    # =========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Löslichkeit vs Druck (diagonal) ---
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "black"]
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure(gas_medium, T, p_max=max(15.0, p_discharge * 1.2), y_gas=y_gas)
        ax1.plot(pressures, sol, "--", linewidth=2, color=color_cycle[i % len(color_cycle)],
                 label=f"{gas_medium} {T:.0f}°C (y={y_gas:.2f})")

    ax1.scatter([p_suction], [sol_cm3N_L_at_suction], s=160, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt", zorder=5)
    ax1.axvline(p_suction, linewidth=1.5, linestyle=":", alpha=0.7)
    ax1.axvline(p_discharge, linewidth=1.5, linestyle=":", alpha=0.7)

    ax1.set_xlabel("p_abs [bar]")
    ax1.set_ylabel("Löslichkeit [cm³_N/L] (Normvolumen)")
    ax1.set_title("Gaslöslichkeit (Henry, als Normvolumen)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # --- Plot 2: ∆p-Q Pumpenkennlinien ---
    Q_req_lmin = m3h_to_lmin(Q_req)
    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]

        gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.5
            ax2.plot(Q_lmin, curve["dp"], "o-", linewidth=lw, alpha=alpha,
                     color=gvf_colors.get(gvf_key, "gray"),
                     label=f"{pump['id']} ({gvf_key}% GVF)")

        ax2.scatter([Q_req_lmin], [dp_req], s=160, marker="o",
                    edgecolors="black", linewidths=2, label="Betriebspunkt (∆p_req)", zorder=5)

        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title(f"Mehrphasen-Kennlinien (∆p): {pump['id']}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
        ax2.set_ylim(0, pump["dp_max_bar"] * 1.1)
    else:
        ax2.text(0.5, 0.5, "❌ Keine geeignete Pumpe gefunden",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title("Mehrphasen-Kennlinien")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # =========================================
    # Grafik 3: Overlay
    # =========================================
    st.markdown("### 📉 Overlay: Löslichkeit (diagonal) + Gasvolumen pro Liter Flüssigkeit (bei GVF)")
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    p_min = 0.1
    p_max = max(14.0, p_discharge * 1.2)
    p_grid = np.linspace(p_min, p_max, 240)

    # Löslichkeit (T-10/T/T+10)
    for i, T in enumerate(temp_variants):
        sol_line = [solubility_cm3N_per_Lliq(gas_medium, p, T, y_gas=y_gas) for p in p_grid]
        ax3.plot(p_grid, sol_line, linestyle="--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    # GVF-Linien (wie im Beispielbild + aktueller Wert)
    gvf_levels = sorted({10, 15, 20, int(gvf_in)})
    gvf_levels = [g for g in gvf_levels if 0 < g <= 40]
    gvf_style = {10: ("tab:blue", "-"), 15: ("tab:cyan", "-"), 20: ("tab:purple", "-")}
    for g in gvf_levels:
        y = gas_ratio_cm3_per_Lliq_vs_pressure(g, p_suction, temperature, p_grid)
        color, ls = gvf_style.get(g, ("tab:gray", "-"))
        ax3.plot(p_grid, y, linestyle=ls, linewidth=2.2, alpha=0.75,
                 color=color, label=f"{g}% GVF (Gasvolumen/L bei p)")

    ax3.axvline(p_suction, linestyle=":", linewidth=1.8, alpha=0.8)
    ax3.axvline(p_discharge, linestyle=":", linewidth=1.8, alpha=0.8)
    ax3.scatter([p_suction], [sol_cm3N_L_at_suction], s=90, edgecolors="black", linewidths=1.5,
                zorder=5, label="Saugpunkt (Löslichkeit)")

    ax3.set_xlabel("Druck [bar abs]")
    ax3.set_ylabel("Löslichkeit [cm³_N/L]  /  Gasvolumen pro Liter Flüssigkeit [cm³/L]")
    ax3.set_title("Overlay: Löslichkeit vs. Gasvolumen (isotherm komprimiert)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(p_min, p_max)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=9, ncol=2)
    st.pyplot(fig3, clear_figure=True)

    # =========================================
    # Ergebnisse
    # =========================================
    st.divider()
    st.markdown("### ✅ Ergebnisse (normlogisch)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("∆p_req", f"{dp_req:.2f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
    c3.metric("GVF_in", f"{gvf_in:.0f} %")
    c4.metric("Löslichkeit @Saugpunkt", f"{sol_cm3N_L_at_suction/1000.0:.4f} L_N/L", f"{sol_cm3N_L_at_suction:.1f} cm³_N/L")
    c5.metric("GVF_free (für Kennlinie)", f"{gvf_free:.1f} %")

    if gvf_free > 0.0:
        st.warning("⚠️ Freies Gas vorhanden (GVF_free > 0). Für Kennlinien wird freier GVF verwendet (Worst Case).")
    else:
        st.info("ℹ️ Aus dieser Abschätzung entsteht kein freies Gas (alles im Löslichkeitslimit).")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("∆p verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve ggü. ∆p_req: {best['dp_reserve']:.2f} bar")
        c2.metric("Leistung", f"{best['P_required']:.2f} kW", "P~n³")
        c3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        c4.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")
        st.markdown("""
**Typische Gründe:**
- ∆p_req (mit Reserve) zu hoch für alle Pumpen/Kennlinien
- Q außerhalb Pumpengrößenbereich
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
            st.caption(f"∆p_max: {pmp['dp_max_bar']} bar")
            st.caption(f"GVF_max: {pmp['GVF_max']*100:.0f}% (frei)")

    # Ausführlicher Rechenweg
    with st.expander("📘 Rechenweg & Normbezug (ausführlich)", expanded=True):
        st.markdown("""
### Ziel der Mehrphasen-Auslegung
Bei Mehrphasen-/Zweiphasenbetrieb ist entscheidend, **wie viel Gas wirklich frei vorliegt**.
Ein Teil kann am Saugpunkt in der Flüssigkeit gelöst sein (Henry), der Rest ist **freies Gas** und
wirkt stark auf die Pumpenkennlinie (∆p-Abfall).

**Typischer Norm-/Standardbezug (Hinweis):**
- **API 610 / ISO 13709**: Auslegung/Anforderungen für Kreiselpumpen (Rahmenwerk)
- Hersteller-Mehrphasenkennlinien (∆p-Q) sind maßgebend
- Gaslöslichkeit: Henry-Gesetz (Tabellenwerte), idealisiertes Verhalten; bei Gemischen über Partialdrücke

> Wichtig: Mehrphasenkennlinien sind stark herstellerspezifisch.
Dieses Tool bildet die Logik normnah ab (∆p, abs. Drücke, freier GVF, Reserve, Affinität), ersetzt aber
keine Herstellerprüfung (NPSH, Schlupf, Stufenanzahl, Temperatur, Kompressibilität, etc.).
        """)

        st.markdown("### 1) Gegeben")
        st.markdown(f"""
- Gas: **{gas_medium}**
- Temperatur: **T = {temperature:.1f} °C**
- Partialdruckfaktor: **y_gas = {y_gas:.2f}**
- Flüssigkeitsvolumenstrom: **Q = {Q_req:.3f} m³/h**
- Saugdruck: **p_s = {p_suction:.3f} bar abs**
- Druckseite: **p_d = {p_discharge:.3f} bar abs**
- Gesamt-GVF am Saugpunkt: **GVF_in = {gvf_in:.1f}%**
- ∆p-Reserve: **{dp_margin:.1f}%**
        """)

        st.markdown("### 2) ∆p-Anforderung")
        st.latex(r"\Delta p_\mathrm{req} = p_d - p_s")
        st.markdown(f"""
- **∆p_req = {p_discharge:.3f} − {p_suction:.3f} = {dp_req:.3f} bar**
- Ziel mit Reserve: **∆p_target = ∆p_req · (1+Reserve) = {dp_req:.3f} · (1+{dp_margin/100.0:.3f}) = {dp_req*(1+dp_margin/100.0):.3f} bar**
        """)

        st.markdown("### 3) Löslichkeit am Saugpunkt (Henry)")
        st.latex(r"C = \frac{p_\mathrm{partial}}{H(T)}\quad [\mathrm{mol/L}]")
        st.latex(r"p_\mathrm{partial} = y_\mathrm{gas}\cdot p_s")

        H_T = henry_constant(gas_medium, temperature)
        C = dissolved_mol_per_Lliq(gas_medium, p_suction, temperature, y_gas=y_gas)
        st.markdown(f"""
- Henry-Konstante: **H(T) = {H_T:.2f} bar·L/mol**
- Partialdruck: **p_partial = y·p_s = {y_gas:.2f}·{p_suction:.2f} = {y_gas*p_suction:.2f} bar**
- Gelöst max.: **C = p_partial/H(T) = {C:.6f} mol/L**
        """)

        st.markdown("Umrechnung in **Normvolumen** (cm³_N/L):")
        st.latex(r"V_N/L = C \cdot V_{m,N}")
        st.markdown(f"""
- Molares Normvolumen: **V_m,N = {V_MOLAR_N_L_PER_MOL:.3f} L/mol**
- Löslichkeit: **{sol_cm3N_L_at_suction:.2f} cm³_N/L**
        """)

        st.markdown("### 4) Gesamtgas am Saugpunkt aus GVF_in")
        st.latex(r"V_{gas,s} = \frac{GVF}{1-GVF}\cdot V_{liq}")
        st.markdown(f"""
- Bezugsvolumen: **1 L Flüssigkeit**
- **V_gas,s = {gvf_dbg['V_gas_total_L']:.4f} L pro 1 L Flüssigkeit**
        """)

        st.markdown("### 5) Molarbilanz: freies Gas bestimmen")
        st.latex(r"n_{total} = \frac{p_s\cdot V_{gas,s}}{R\cdot T}")
        st.markdown(f"""
- **n_total = {gvf_dbg['n_total_mol']:.6f} mol**
- **n_diss,max = {gvf_dbg['n_diss_max_mol']:.6f} mol**
- **n_free = max(0, n_total − n_diss,max) = {gvf_dbg['n_free_mol']:.6f} mol**
        """)

        st.latex(r"V_{free,s} = \frac{n_{free}\cdot R\cdot T}{p_s}")
        st.markdown(f"""
- **V_free,s = {gvf_dbg['V_gas_free_L']:.6f} L**
- **GVF_free = V_free,s/(V_free,s+1L) = {gvf_free:.2f}%**
        """)

        st.markdown("### 6) Kennlinienwahl (Worst Case)")
        if best:
            st.markdown(f"""
- GVF_free = **{gvf_free:.2f}%**
- Gewählte Kennlinie: **{best['gvf_curve']}% GVF** (nächsthöher, Worst Case)
            """)

        st.markdown("### 7) Drehzahl (Affinität) – Bisektion")
        st.latex(r"Q \sim n,\quad \Delta p \sim n^2,\quad P \sim n^3")
        if best:
            st.markdown(f"""
- Modus: **{best['mode']}**
- n_ratio = **{best['n_ratio']:.3f}**
- Q_base = **{best['Q_base']:.3f} m³/h**
- ∆p_base = **{best['dp_base']:.3f} bar**
- ∆p_scaled = **{best['dp_available']:.3f} bar**
- Leistung (skaliert): **{best['P_required']:.3f} kW**
            """)

# =========================================================
# PAGE 3: ATEX
# =========================================================
elif st.session_state.page == "atex":
    st.subheader("⚡ ATEX-Motorauslegung")
    st.caption("Auslegung nach RL 2014/34/EU (vereinfachte Logik)")

    col_in, col_res = st.columns([1, 2])

    with col_in:
        st.header("1) Prozessdaten")
        P_req_input = st.number_input("Erf. Wellenleistung Pumpe [kW]", min_value=0.1, value=5.5, step=0.5)
        T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

        st.divider()
        st.header("2) Zone")
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


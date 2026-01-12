# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Konstanten
# =========================
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K) ideales Gas

# Normbedingungen (für "diagonale" Darstellung in cm³N/L)
P_N_BAR = 1.01325
T_N_K = 273.15  # 0°C als Norm (passt zu typischen Ncm³-Angaben)

# Basisdrehzahl (Default), wenn nicht pumpenspezifisch angegeben
N0_RPM_DEFAULT = 2900

# Umrechnung: bar -> m Förderhöhe (Wasserähnlich)
BAR_TO_M_WATER = 1e5 / (998.0 * G)  # ~10.21 m/bar

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

# =========================
# Mehrphasen-Pumpen (Beispiel, überarbeitet: gekrümmter + näher am Foliencharakter)
# Kennlinien in Δp [bar] vs Q [m³/h], danach im Plot in H [m] umgerechnet.
# =========================
MPH_PUMPS = [
    {
        "id": "MPH-40",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 40,
        "dp_max_bar": 12,
        "GVF_max": 0.4,  # 40%
        "n0_rpm": 2900,
        # stärker gekrümmt, wie typische Kennlinien (nicht linear)
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [11.2, 11.0, 10.6, 10.0, 9.2, 7.6, 6.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [10.9, 10.7, 10.2, 9.6, 8.8, 7.2, 5.7]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [10.5, 10.2, 9.7, 9.0, 8.2, 6.6, 5.1]},
            15: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [9.9, 9.6, 9.0, 8.2, 7.4, 5.8, 4.5]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [9.1, 8.8, 8.2, 7.4, 6.6, 5.0, 3.9]},
        },
        "power_kW_vs_Q": {
            # Leistung moderat steigend; bei höherem GVF etwas niedriger (wie oft in Herstellerblättern dargestellt)
            0:  {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [3.0, 3.4, 3.9, 4.5, 5.1, 6.2, 7.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.9, 3.3, 3.8, 4.3, 4.9, 6.0, 6.8]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.8, 3.2, 3.6, 4.1, 4.7, 5.7, 6.4]},
            15: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.7, 3.0, 3.4, 3.9, 4.4, 5.3, 6.0]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.5, 2.8, 3.2, 3.6, 4.0, 4.8, 5.4]},
        }
    },
    {
        "id": "MPH-70",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 70,
        "dp_max_bar": 20,
        "GVF_max": 0.4,
        "n0_rpm": 2900,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 55, 70], "dp": [18.8, 18.5, 17.7, 16.2, 14.2, 11.5, 9.2]},
            5:  {"Q": [0, 10, 20, 30, 40, 55, 70], "dp": [18.2, 17.8, 17.0, 15.5, 13.6, 11.0, 8.8]},
            10: {"Q": [0, 10, 20, 30, 40, 55, 70], "dp": [17.4, 17.0, 16.2, 14.6, 12.8, 10.3, 8.1]},
            15: {"Q": [0, 10, 20, 30, 40, 55, 70], "dp": [16.2, 15.8, 14.9, 13.3, 11.5, 9.2, 7.2]},
            20: {"Q": [0, 10, 20, 30, 40, 55, 70], "dp": [14.9, 14.5, 13.6, 12.0, 10.3, 8.2, 6.5]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 55, 70], "P": [4.0, 4.8, 5.9, 7.2, 8.6, 10.4, 12.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 55, 70], "P": [3.9, 4.7, 5.7, 7.0, 8.4, 10.2, 11.7]},
            10: {"Q": [0, 10, 20, 30, 40, 55, 70], "P": [3.8, 4.5, 5.5, 6.7, 8.0, 9.7, 11.1]},
            15: {"Q": [0, 10, 20, 30, 40, 55, 70], "P": [3.6, 4.3, 5.2, 6.4, 7.5, 9.0, 10.3]},
            20: {"Q": [0, 10, 20, 30, 40, 55, 70], "P": [3.3, 4.0, 4.8, 5.9, 6.8, 8.2, 9.2]},
        }
    },
]

# =========================
# ATEX (Beispiel)
# =========================
ATEX_MOTORS = [
    {"id": "Standard Zone 2 (ec)", "marking": "II 3G Ex ec IIC T3 Gc", "zone_suitable": [2],
     "temp_class": "T3", "t_max_surface": 200.0, "category": "3G",
     "description": "Standardlösung für Zone 2."},
    {"id": "Zone 1 (eb)", "marking": "II 2G Ex eb IIC T3 Gb", "zone_suitable": [1, 2],
     "temp_class": "T3", "t_max_surface": 200.0, "category": "2G",
     "description": "Standardlösung für Zone 1 ('Erhöhte Sicherheit')."},
    {"id": "Zone 1 (db eb) T4", "marking": "II 2G Ex db eb IIC T4 Gb", "zone_suitable": [1, 2],
     "temp_class": "T4", "t_max_surface": 135.0, "category": "2G",
     "description": "Zone 1 mit strengeren Temperaturanforderungen (T4)."},
    {"id": "Zone 1 (db) T4", "marking": "II 2G Ex db IIC T4 Gb", "zone_suitable": [1, 2],
     "temp_class": "T4", "t_max_surface": 135.0, "category": "2G",
     "description": "Zone 1 (T4). Vollständig druckfeste Kapselung."},
]

# =========================
# Henry Parameter (nur Platzhalter/Beispielwerte)
# =========================
HENRY_CONSTANTS = {
    "Luft": {"A": 1400.0, "B": 1500},
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
    if len(xs) < 2:
        return ys[0] if ys else 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[-1]

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================
# Viskosität (HI-ähnlich, pragmatisch)
# =========================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    Q = max(Q_m3h, 1e-6)
    H = max(H_m, 1e-6)
    nu = max(nu_cSt, 1e-6)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    if B <= 1.0:
        return 1.0, 1.0
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = clamp(CH, 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = clamp(Ceta, 0.1, 1.0)
    return CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt):
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / max(CH, 1e-9)}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return Q_water, H_water * CH, max(1e-6, eta_water * Ceta)

def generate_viscous_curve(pump, nu_cSt, rho):
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        P_v = (rho * G * (Q_v / 3600.0) * H_v) / (1000.0 * max(eta_v, 1e-6))  # kW
        Q_vis.append(Q_v)
        H_vis.append(H_v)
        eta_vis.append(eta_v)
        P_vis.append(P_v)
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
        cand = {"id": p["id"], "pump": p, "in_range": in_range, "Q_eval": Q_eval,
                "H_at": H_at, "eta_at": eta_at, "errH": abs(H_at - H_water), "score": score}
        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    return best

# =========================
# Einphasen: optimale Drehzahl, damit Kennlinie Betriebspunkt trifft (Affinität)
# Q ~ n, H ~ n², P ~ n³
# f(n) = H0(Q_req/n) * n² - H_req = 0
# =========================
def _H_scaled_at_Q(Q_curve, H_curve, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-9)
    H_base = interp_clamped(Q_base, Q_curve, H_curve)
    return H_base * (n_ratio ** 2)

def find_speed_ratio_for_point(Q_curve, H_curve, Q_req, H_req, n_min=0.5, n_max=1.2, tol=1e-3, iters=80):
    f_min = _H_scaled_at_Q(Q_curve, H_curve, Q_req, n_min) - H_req
    f_max = _H_scaled_at_Q(Q_curve, H_curve, Q_req, n_max) - H_req
    if f_min * f_max > 0:
        return None  # kein Vorzeichenwechsel -> nicht sicher lösbar im Bereich
    lo, hi = n_min, n_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _H_scaled_at_Q(Q_curve, H_curve, Q_req, mid) - H_req
        if abs(f_mid) <= tol:
            return mid
        if f_min * f_mid <= 0:
            hi = mid
            f_max = f_mid
        else:
            lo = mid
            f_min = f_mid
    return 0.5 * (lo + hi)

# =========================
# Henry / Gaslöslichkeit (bei Betriebsbedingungen) + Normvolumen
# =========================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def gas_solubility_L_per_L_oper(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry: C [mol/L] = p_partial / H(T)
    Ideales Gas bei Betriebsbedingungen: V_molar_oper [L/mol] = R*T/p
    -> V_gas_oper / V_liq [L/L] = C * V_molar_oper
    """
    p = max(p_bar_abs, 1e-6)
    T_K = T_celsius + 273.15
    H = henry_constant(gas, T_celsius)  # bar·L/mol
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    C_mol_L = p_partial / max(H, 1e-12)
    V_molar_oper = R_BAR_L * T_K / p
    return C_mol_L * V_molar_oper

def oper_to_norm_volume_ratio(p_oper_bar_abs, T_oper_c):
    """
    Umrechnung operatives Gasvolumen -> Normvolumen:
    V_N = V_oper * (p_oper/p_N) * (T_N/T_oper)
    """
    T_oper_K = T_oper_c + 273.15
    return (p_oper_bar_abs / P_N_BAR) * (T_N_K / max(T_oper_K, 1e-9))

def gas_solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Solubility in Normvolumen cm³N/L:
    (V_gas_oper/L) * ratio_oper_to_norm * 1000
    """
    V_oper_L_per_L = gas_solubility_L_per_L_oper(gas, p_bar_abs, T_celsius, y_gas=y_gas)
    ratio = oper_to_norm_volume_ratio(p_bar_abs, T_celsius)
    return V_oper_L_per_L * ratio * 1000.0

def solubility_diagonal_curve_cm3N_per_L(gas, T_celsius, p_max=14, y_gas=1.0):
    pressures = linspace(0.1, p_max, 120)
    sol_cm3N_L = [gas_solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol_cm3N_L

# =========================
# Folien-Logik: 10% ↔ 100 cm³N/L
# =========================
def total_gas_cm3N_per_L_from_gvf_like(gvf_pct):
    """
    Mapping wie in der Folie:
    10% ≙ 100 cm³N/L  ->  1% ≙ 10 cm³N/L
    """
    return max(0.0, gvf_pct) * 10.0

# =========================
# Mehrphase: Affinität (Q~n, Δp~n², P~n³) korrekt über Q_base
# =========================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-9)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2)

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_req,
                               n_min=0.5, n_max=1.1, tol=1e-4, iters=80):
    f_min = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_min) - dp_req
    f_max = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_max) - dp_req
    if f_min * f_max > 0:
        return None
    lo, hi = n_min, n_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, mid) - dp_req
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
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def choose_best_mph_pump(pumps, Q_req, dp_req, gvf_free_req_pct, dp_margin=0.05):
    """
    Auswahl:
    - Kennlinien in Δp
    - Worst Case: nächsthöhere GVF-Kurve >= GVF_free
    - Prüfe bei n=1.0, sonst Drehzahl-Anpassung (wenn möglich)
    """
    best = None

    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        if Q_req > pump["Q_max_m3h"] * 1.1:
            continue

        n0 = pump.get("n0_rpm", N0_RPM_DEFAULT)

        gvf_key = choose_gvf_curve_key_worstcase(pump["curves_dp_vs_Q"], gvf_free_req_pct)
        curve = pump["curves_dp_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]

        Qmin, Qmax = min(curve["Q"]), max(curve["Q"])
        candidates = []

        # A) Nenndrehzahl
        if Qmin <= Q_req <= Qmax:
            dp_avail_nom = interp_clamped(Q_req, curve["Q"], curve["dp"])
            if dp_avail_nom >= dp_req * (1.0 + dp_margin):
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail_nom - dp_req) + abs(gvf_key - gvf_free_req_pct) * 0.25
                candidates.append({
                    "pump": pump, "gvf_curve": gvf_key, "dp_available": dp_avail_nom,
                    "P_required": P_nom, "n_ratio": 1.0, "n_abs_rpm": float(n0),
                    "mode": f"Nenndrehzahl {n0:.0f} rpm",
                    "dp_reserve": dp_avail_nom - dp_req, "score": score
                })

        # B) Drehzahl (Bisektion) -> Ziel: dp(Q_req) = dp_req
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["dp"], Q_req, dp_req)
        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if Qmin <= Q_base <= Qmax:
                dp_scaled = _dp_scaled_at_Q(curve["Q"], curve["dp"], Q_req, n_ratio)
                P_base = interp_clamped(Q_base, power_curve["Q"], power_curve["P"])
                P_scaled = P_base * (n_ratio ** 3)

                if dp_scaled + 1e-6 >= dp_req:
                    n_abs = n0 * n_ratio
                    score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.25
                    candidates.append({
                        "pump": pump, "gvf_curve": gvf_key, "dp_available": dp_scaled,
                        "P_required": P_scaled, "n_ratio": n_ratio, "n_abs_rpm": float(n_abs),
                        "mode": f"Drehzahl {n_abs:.0f} rpm",
                        "dp_reserve": dp_scaled - dp_req, "score": score
                    })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

# =========================
# Sidebar Inputs
# =========================
def sidebar_inputs_pump():
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

    # Bereich für n-Optimierung (Viskosität)
    st.divider()
    st.subheader("Drehzahl-Optimierung (Viskosität)")
    n_min = st.slider("n_min/n0", 0.40, 1.00, 0.60, 0.01)
    n_max = st.slider("n_max/n0", 1.00, 1.60, 1.20, 0.01)

    return Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct, float(n_min), float(n_max)

def sidebar_inputs_mph():
    st.divider()
    st.subheader("⚙️ Medium / Gas")
    gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
    temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)

    st.divider()
    st.subheader("Betriebspunkt (Hydraulik)")
    Q_req = st.number_input("Volumenstrom Flüssigkeit Q_liq [m³/h]", 0.1, 150.0, 30.0, 1.0)
    p_suction = st.number_input("p_abs (Saugseite) [bar abs]", 0.1, 200.0, 2.0, 0.1)
    p_discharge = st.number_input("p_abs (Druckseite) [bar abs]", 0.1, 400.0, 7.0, 0.1)

    dp_req = max(0.0, (p_discharge - p_suction))

    st.divider()
    st.subheader("Gasdefinition (Druckseite)")

    gas_mode = st.radio(
        "Wie ist das Gas vorgegeben?",
        [
            "GVF_out [%] (an Druckseite im Wasser gelöst)",
            "Recyclingstrom Q_rec [m³/h] (gesättigt an Druckseite)"
        ],
        index=0
    )

    gvf_out_pct = None
    Q_rec_m3h = None

    if gas_mode.startswith("GVF_out"):
        gvf_out_pct = st.slider("GVF_out [%] (gelöst bei p_abs Druckseite)", 0.0, 40.0, 10.0, 0.1)
    else:
        Q_rec_m3h = st.number_input("Recyclingstrom Q_rec [m³/h] (mit gelöstem Gas)", 0.0, 150.0, 3.0, 0.1)
        st.caption("Annahme: Recyclingstrom ist bei p_abs Druckseite,T gesättigt (max. löslich). Frischanteil ist entgast.")

    st.divider()
    st.subheader("Darstellung / Logik")
    show_temp_band = st.checkbox("Löslichkeit bei T-10/T/T+10", value=True)
    only_dissolved_allowed = st.checkbox("Nur gelöst zulässig (wie Folie: 'nicht möglich')", value=False)

    return (gas_medium, temperature, Q_req, p_suction, p_discharge, dp_req,
            gas_mode, gvf_out_pct, Q_rec_m3h, show_temp_band, only_dissolved_allowed)

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
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur (HI-ähnliche Näherung)")

    with st.sidebar:
        Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct, n_min_ratio, n_max_ratio = sidebar_inputs_pump()

    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    st.markdown("### 📊 Umrechnung viskos → Wasserkennlinie")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q_w", f"{Q_water:.2f} m³/h")
    c2.metric("H_w", f"{H_water:.2f} m")
    c3.metric("B", f"{B:.3f}")
    c4.metric("C_H / C_η", f"{CH:.3f} / {Ceta:.3f}")

    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]
    eta_water = best["eta_at"]

    # Betriebspunkt viskos (real)
    eta_vis = max(1e-6, eta_water * Ceta)
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / eta_vis) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    # Kennlinien viskos generieren (Nenndrehzahl)
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)

    # =========
    # Optimale Drehzahl: Kennlinie soll Betriebspunkt treffen
    # =========
    n_ratio_opt = find_speed_ratio_for_point(
        Q_vis_curve, H_vis_curve,
        Q_req=Q_vis_req,
        H_req=H_vis_req,
        n_min=n_min_ratio,
        n_max=n_max_ratio,
        tol=1e-3
    )
    n0 = N0_RPM_DEFAULT
    n_opt_rpm = None
    P_opt_kW = None
    saving_pct = None

    # Vergleichsleistung bei Nenndrehzahl (auf viskoser Kennlinie bei Q_req)
    # (wenn Q außerhalb, clampen -> Vergleich bleibt "realistisch konservativ")
    Qv_min, Qv_max = min(Q_vis_curve), max(Q_vis_curve)
    Q_for_nom = clamp(Q_vis_req, Qv_min, Qv_max)
    P_nom_kW_at_Q = interp_clamped(Q_for_nom, Q_vis_curve, P_vis_curve)

    if n_ratio_opt is not None:
        n_opt_rpm = n0 * n_ratio_opt
        # Leistung über Affinität: P(Q_req,n) = P0(Q_req/n) * n^3
        Q_base = Q_vis_req / n_ratio_opt
        P_base = interp_clamped(Q_base, Q_vis_curve, P_vis_curve)
        P_opt_kW = P_base * (n_ratio_opt ** 3)
        if P_nom_kW_at_Q > 1e-9:
            saving_pct = (P_nom_kW_at_Q - P_opt_kW) / P_nom_kW_at_Q * 100.0

    st.divider()
    st.markdown("### ✅ Ergebnis (Einphasen)")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h")
    col2.metric("H (viskos)", f"{H_vis_req:.2f} m")
    col3.metric("η (viskos)", f"{eta_vis:.3f}")
    col4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW")
    col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")
    col6.metric("Q", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")

    st.divider()
    st.markdown("### 🎯 Drehzahl-Optimierung (Betriebspunkt treffen)")
    if n_ratio_opt is None:
        st.warning("⚠️ Für den gewählten n-Bereich konnte keine Drehzahl gefunden werden, die die Kennlinie exakt durch den Betriebspunkt legt. "
                   "Passe n_min/n0 bzw. n_max/n0 an oder prüfe, ob der Betriebspunkt grundsätzlich im Kennfeld erreichbar ist.")
    else:
        a, b, c = st.columns(3)
        a.metric("n_opt", f"{n_opt_rpm:.0f} rpm", f"n/n0 = {n_ratio_opt:.3f}")
        b.metric("P bei Nenndrehzahl", f"{P_nom_kW_at_Q:.2f} kW", "auf visk. Kennlinie")
        b2 = f"{P_opt_kW:.2f} kW"
        c.metric("P bei n_opt", b2, (f"Einsparung {saving_pct:.1f}%" if saving_pct is not None else ""))

    if not best["in_range"]:
        st.warning(f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
                   f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h.")

    # Betriebspunkt auf der Kennlinie (viskos) für korrekte BP-Visualisierung (Nenndrehzahl)
    H_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, H_vis_curve)
    eta_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, eta_vis_curve)
    P_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, P_vis_curve)

    # Wasser-Punkt auf Kennlinie
    H_water_op = interp_clamped(Q_water, p["Qw"], p["Hw"])
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])

    # Optional: skaliertes Kennfeld bei n_opt für Plot
    Q_vis_curve_scaled = None
    H_vis_curve_scaled = None
    P_vis_curve_scaled = None
    eta_vis_curve_scaled = None
    if n_ratio_opt is not None:
        Q_vis_curve_scaled = [q * n_ratio_opt for q in Q_vis_curve]
        H_vis_curve_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
        P_vis_curve_scaled = [pp * (n_ratio_opt ** 3) for pp in P_vis_curve]
        # η bleibt bei Affinität näherungsweise gleich (wir lassen η-Kurve identisch und markieren nur den Punkt)
        eta_vis_curve_scaled = list(eta_vis_curve)

    st.divider()
    st.markdown("### 📈 Kennlinien")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos, n0)")

        if Q_vis_curve_scaled is not None:
            ax1.plot(Q_vis_curve_scaled, H_vis_curve_scaled, ":", linewidth=3.0, label=f"{p['id']} (viskos, n_opt)")

        # Wasser-Betriebspunkt (auf Kennlinie)
        ax1.scatter([Q_water], [H_water_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser auf Kennlinie)", zorder=6)

        # Betriebspunkt (nur EIN Marker, kein Sollpunkt-Kreis)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=220, linewidths=3,
                    label="BP (viskos)", zorder=7)

        ax1.set_xlabel("Q [m³/h]")
        ax1.set_ylabel("H [m]")
        ax1.set_title("Q-H Kennlinien")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        st.pyplot(fig1, clear_figure=True)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(p["Qw"], p["eta"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax2.plot(Q_vis_curve, eta_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos, n0)")

        ax2.scatter([Q_water], [eta_water], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="η (Wasser)", zorder=6)

        ax2.scatter([Q_vis_req], [eta_vis_on_curve], marker="x", s=220, linewidths=3,
                    label="η bei BP (viskos)", zorder=7)

        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η Kennlinien")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(p["Qw"], p["Pw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos, n0)")

        if Q_vis_curve_scaled is not None:
            ax3.plot(Q_vis_curve_scaled, P_vis_curve_scaled, ":", linewidth=3.0, label=f"{p['id']} (viskos, n_opt)")

        ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser)", zorder=6)

        ax3.scatter([Q_vis_req], [P_vis_on_curve], marker="x", s=220, linewidths=3,
                    label="P bei BP (viskos)", zorder=7)

        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    with st.expander("📘 Rechenweg (ausführlich)", expanded=False):
        st.markdown("""
### Schritt 1: HI-ähnliche Kennzahl B
Bei viskosen Medien steigt die innere Reibung. Dadurch sinken Förderhöhe und Wirkungsgrad, die Leistung steigt.
Für eine Auswahl auf Wasserkennlinien verwenden wir die Kennzahl **B**.
""")
        st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q_{gpm}^{0.25} \cdot H_{ft}^{0.375}}")
        st.markdown(f"- B = **{B:.3f}**")
        st.markdown("---")
        st.markdown("### Schritt 2: Korrekturfaktoren")
        st.latex(r"H_{vis} = H_w \cdot C_H \quad,\qquad \eta_{vis} = \eta_w \cdot C_\eta")
        st.markdown(f"- C_H = **{CH:.3f}**, C_η = **{Ceta:.3f}**")
        st.markdown("---")
        st.markdown("### Schritt 3: Umrechnung viskos → Wasserkennlinie")
        st.latex(r"H_w = \frac{H_{vis}}{C_H}")
        st.markdown(f"- Q_w = **{Q_water:.2f} m³/h**, H_w = **{H_water:.2f} m**")
        st.markdown("---")
        st.markdown("### Schritt 4: Leistung")
        st.latex(r"P_{hyd} = \rho g Q H \quad,\qquad P_{shaft} = \frac{P_{hyd}}{\eta_{vis}}")
        st.markdown(f"- P_hyd = **{P_hyd_W:.0f} W**, P_shaft = **{P_vis_kW:.2f} kW**")
        st.markdown("---")
        st.markdown("### Schritt 5: Drehzahl-Optimierung (Affinität)")
        st.markdown("Ziel: Kennlinie so skalieren, dass der Betriebspunkt getroffen wird.")
        st.latex(r"Q \sim n,\quad H \sim n^2,\quad P \sim n^3")
        st.latex(r"H(Q,n)=H_0\!\left(\frac{Q}{n/n_0}\right)\cdot\left(\frac{n}{n_0}\right)^2")
        if n_ratio_opt is None:
            st.markdown("- **Keine Lösung** im gewählten Bereich n_min…n_max.")
        else:
            st.markdown(f"- n/n0 = **{n_ratio_opt:.3f}**, n_opt = **{n_opt_rpm:.0f} rpm**")
            st.markdown(f"- P(n0) ≈ **{P_nom_kW_at_Q:.2f} kW**, P(n_opt) ≈ **{P_opt_kW:.2f} kW**")
            st.markdown(f"- Einsparung ≈ **{saving_pct:.1f}%**")

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Gasdefinition Druckseite + Löslichkeit + Förderhöhe-Kennlinien")

    with st.sidebar:
        (gas_medium, temperature, Q_req, p_suction, p_discharge, dp_req,
         gas_mode, gvf_out_pct, Q_rec_m3h, show_temp_band, only_dissolved_allowed) = sidebar_inputs_mph()

    y_gas = 1.0  # wie gewünscht

    # Förderhöhe-Anforderung aus Δp (nur für Darstellung/Markierung)
    H_req_m = dp_req * BAR_TO_M_WATER

    # Löslichkeit an Saug- und Druckseite (cm³N/L)
    sol_s_cm3N_L = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    sol_d_cm3N_L = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature, y_gas=y_gas)

    # Gesamtgas als Normvolumen cm³N/L (konstant, definiert durch Druckseite / Recycling)
    if gas_mode.startswith("GVF_out"):
        total_cm3N_L = total_gas_cm3N_per_L_from_gvf_like(gvf_out_pct)  # 10% -> 100 cm³N/L
        not_solvable_at_discharge = total_cm3N_L > (sol_d_cm3N_L + 1e-9)
    else:
        # Recycling: nur der Recyclinganteil trägt Gas (gesättigt bei p_d)
        Q_rec = max(0.0, min(float(Q_rec_m3h), float(Q_req)))
        frac = 0.0 if Q_req <= 1e-9 else (Q_rec / Q_req)
        total_cm3N_L = sol_d_cm3N_L * frac
        not_solvable_at_discharge = False

    # Freies Gas am Saugpunkt (wenn total > Löslichkeit bei p_s)
    free_s_cm3N_L = max(0.0, total_cm3N_L - sol_s_cm3N_L)

    # Für Pumpenkennlinien: GVF_free[%] grob aus Folienmapping (1% ~ 10 cm³N/L)
    gvf_free_req_pct = free_s_cm3N_L / 10.0

    not_solvable_at_suction = free_s_cm3N_L > 1e-9

    if only_dissolved_allowed and not_solvable_at_suction:
        best = None
    else:
        best = choose_best_mph_pump(MPH_PUMPS, Q_req, dp_req, gvf_free_req_pct)

    # =========================
    # Plot 1+2: Löslichkeit (cm³N/L) + Förderhöhe-Kennlinien (aus Δp)
    # =========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Löslichkeit diagonal (cm³N/L) vs p_abs
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    for i, T in enumerate(temp_variants):
        pressures, sol_cm3N = solubility_diagonal_curve_cm3N_per_L(gas_medium, T, p_max=14, y_gas=y_gas)
        ax1.plot(pressures, sol_cm3N, "--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    # Referenzlinien wie Folie: 10/15/20% -> 100/150/200 cm³N/L
    ax1.axhline(100, linewidth=1.5, linestyle=":", alpha=0.6)
    ax1.axhline(150, linewidth=1.5, linestyle=":", alpha=0.6)
    ax1.axhline(200, linewidth=1.5, linestyle=":", alpha=0.6)
    ax1.text(13.8, 100, "10% (100)", va="center", ha="right", fontsize=9)
    ax1.text(13.8, 150, "15% (150)", va="center", ha="right", fontsize=9)
    ax1.text(13.8, 200, "20% (200)", va="center", ha="right", fontsize=9)

    ax1.scatter([p_suction], [sol_s_cm3N_L], s=160, marker="o",
                edgecolors="black", linewidths=2, label="p_abs Saug (Löslichkeit)", zorder=6)
    ax1.scatter([p_discharge], [sol_d_cm3N_L], s=140, marker="s",
                edgecolors="black", linewidths=2, label="p_abs Druck (Löslichkeit)", zorder=6)

    # Actual total als Punkt bei Druckseite (Interpretation wie Folie: GVF_out an Druckseite gelöst)
    ax1.scatter([p_discharge], [total_cm3N_L], s=180, marker="X",
                edgecolors="black", linewidths=2, label="GVF_out (als cm³N/L)", zorder=7)

    ax1.set_xlabel("Druck p_abs [bar]")
    ax1.set_ylabel("Löslichkeit / Gasmenge [cm³N/L]")
    ax1.set_title("Gaslöslichkeit (Henry → Normvolumen) + Referenz (10/15/20%)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 220)
    ax1.legend(fontsize=9)

    # Plot 2: Förderhöhe-Q Kennlinien (aus Δp)
    Q_req_lmin = m3h_to_lmin(Q_req)
    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]
        gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            H_m = [dp * BAR_TO_M_WATER for dp in curve["dp"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.55
            ax2.plot(Q_lmin, H_m, "o-", linewidth=lw, alpha=alpha,
                     color=gvf_colors.get(gvf_key, "gray"),
                     label=f"{pump['id']} ({gvf_key}% GVF)")

        ax2.scatter([Q_req_lmin], [H_req_m], s=180, marker="x",
                    linewidths=3, label="Betriebspunkt (H_req)", zorder=7)

        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("H [m]")
        ax2.set_title(f"Mehrphasen-Kennlinien (Förderhöhe): {pump['id']}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
        ax2.set_ylim(0, pump["dp_max_bar"] * BAR_TO_M_WATER * 1.05)
    else:
        ax2.text(0.5, 0.5, "Keine Pumpenauswahl (siehe Hinweis)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("H [m]")
        ax2.set_title("Mehrphasen-Kennlinien (Förderhöhe)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # =========================
    # Ergebnisse
    # =========================
    st.divider()
    st.markdown("### ✅ Ergebnisse (Mehrphase)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q_liq", f"{Q_req:.1f} m³/h", f"{m3h_to_lmin(Q_req):.1f} L/min")
    c2.metric("H_req", f"{H_req_m:.1f} m", f"Δp={dp_req:.2f} bar")

    if gas_mode.startswith("GVF_out"):
        c3.metric("GVF_out", f"{gvf_out_pct:.2f} %", f"{total_cm3N_L:.1f} cm³N/L")
    else:
        c3.metric("Recyclingstrom", f"{float(Q_rec_m3h):.2f} m³/h", f"total={total_cm3N_L:.1f} cm³N/L")

    c4.metric("Löslichkeit p_abs Saug", f"{sol_s_cm3N_L:.1f} cm³N/L", "")
    c5.metric("freies Gas p_abs Saug", f"{free_s_cm3N_L:.1f} cm³N/L", f"GVF_free≈{gvf_free_req_pct:.2f}%")

    if not_solvable_at_discharge:
        st.error("❌ Nicht möglich: Der gewählte GVF_out ist bei p_abs Druckseite,T nicht vollständig löslich (wie im Beispiel: 15% nicht löslich).")
    if not_solvable_at_suction:
        st.warning("⚠️ Beim Druckabfall zur Saugseite entsteht freie Gasphase (Worst Case relevant).")
    else:
        st.info("ℹ️ Bei p_abs Saugseite,T bleibt alles gelöst (kein freies Gas).")

    if only_dissolved_allowed and not_solvable_at_suction:
        st.error("❌ Auswahl blockiert (Nur gelöst zulässig). Deaktiviere die Option, um Worst-Case (freies Gas) auszulegen.")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        a, b, c, d = st.columns(4)
        H_avail_m = best["dp_available"] * BAR_TO_M_WATER
        H_res_m = best["dp_reserve"] * BAR_TO_M_WATER

        a.metric("H verfügbar", f"{H_avail_m:.1f} m", f"Reserve: {H_res_m:.1f} m")
        b.metric("Leistung", f"{best['P_required']:.2f} kW")
        c.metric("Drehzahl n", f"{best.get('n_abs_rpm', N0_RPM_DEFAULT):.0f} rpm", f"n/n0={best['n_ratio']:.3f}")
        d.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

        if best["dp_available"] + 1e-6 < dp_req:
            st.error("❌ H_req wird nicht erreicht (sollte bei Auswahl nicht passieren).")
    else:
        if not (only_dissolved_allowed and not_solvable_at_suction):
            st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")

    # =========================
    # Rechenweg (Mehrphase) – wieder wie bei Viskosität
    # =========================
    with st.expander("📘 Rechenweg (Mehrphase, ausführlich)", expanded=False):
        st.markdown("### Schritt 1: Δp → Förderhöhe")
        st.latex(r"H = \frac{\Delta p}{\rho g}")
        st.markdown(f"- Δp = **{dp_req:.2f} bar** → H_req ≈ **{H_req_m:.1f} m** (wasserähnlich)")

        st.markdown("---")
        st.markdown("### Schritt 2: Henry-Löslichkeit → cm³N/L")
        st.latex(r"C=\frac{p}{H(T)}\quad,\quad V_{gas,oper}=\frac{RT}{p}\quad\Rightarrow\quad \frac{V_{gas,oper}}{V_{liq}}=C\cdot V_{gas,oper}")
        st.latex(r"V_{N}=V_{oper}\cdot\frac{p_{oper}}{p_N}\cdot\frac{T_N}{T_{oper}}")
        st.markdown(f"- Löslichkeit Saugseite: **{sol_s_cm3N_L:.1f} cm³N/L** bei p_abs={p_suction:.2f} bar, T={temperature:.1f}°C")
        st.markdown(f"- Löslichkeit Druckseite: **{sol_d_cm3N_L:.1f} cm³N/L** bei p_abs={p_discharge:.2f} bar, T={temperature:.1f}°C")

        st.markdown("---")
        st.markdown("### Schritt 3: Folien-Mapping GVF_out → Gesamtgasmenge (cm³N/L)")
        st.latex(r"10\% \leftrightarrow 100\,\mathrm{cm^3_N/L}\quad\Rightarrow\quad \mathrm{cm^3_N/L}=10\cdot GVF[\%]")
        st.markdown(f"- total = **{total_cm3N_L:.1f} cm³N/L**")

        st.markdown("---")
        st.markdown("### Schritt 4: Freies Gas an Saugseite (Worst Case)")
        st.latex(r"free = \max(0,\ total - sol(p_{saug}))")
        st.markdown(f"- free_s = **{free_s_cm3N_L:.1f} cm³N/L** → GVF_free ≈ **{gvf_free_req_pct:.2f}%**")

        st.markdown("---")
        st.markdown("### Schritt 5: Pumpenauswahl – Worst-Case-Kurve")
        st.markdown("Es wird die **nächsthöhere GVF-Kennlinie** verwendet (konservativ).")
        if best:
            st.markdown(f"- gewählte Kurve: **{best['gvf_curve']}% GVF**")
            st.markdown("---")
            st.markdown("### Schritt 6: Drehzahl-Anpassung (Affinität)")
            st.latex(r"Q \sim n,\quad \Delta p \sim n^2,\quad P \sim n^3")
            st.markdown(f"- n/n0 = **{best['n_ratio']:.3f}**, n = **{best['n_abs_rpm']:.0f} rpm**")
            st.markdown(f"- P ≈ **{best['P_required']:.2f} kW**")
        else:
            st.markdown("- keine Auswahl (siehe Hinweise oberhalb).")

# =========================================================
# PAGE 3: ATEX
# =========================================================
elif st.session_state.page == "atex":
    st.subheader("⚡ ATEX-Motorauslegung (vereinfachte Logik)")
    col_in, col_res = st.columns([1, 2])

    with col_in:
        st.header("1) Prozessdaten")
        P_req_input = st.number_input("Erf. Wellenleistung Pumpe [kW]", min_value=0.1, value=5.5, step=0.5)
        T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

        st.divider()
        st.header("2) Zone")
        atmosphere = st.radio("Atmosphäre", ["G (Gas)", "D (Staub)"], index=0)
        zone_select = st.selectbox("Ex-Zone (Gas)", [0, 1, 2], index=2) if atmosphere == "G (Gas)" else st.selectbox("Ex-Zone (Staub)", [20, 21, 22], index=2)

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

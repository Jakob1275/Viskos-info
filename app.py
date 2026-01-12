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
T_N_K = 273.15  # 0°C als Norm

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
# Mehrphasen-Pumpen: Kennlinien als Δp in bar (Beispiel)
# =========================
MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "dp_max_bar": 9,
        "GVF_max": 0.4,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.6, 8.5, 8.2, 7.6, 6.8, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.4, 8.3, 8.0, 7.3, 6.3, 4.6]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "dp": [8.2, 8.0, 7.6, 6.7, 5.6, 4.0]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "dp": [7.8, 7.5, 6.9, 5.8, 4.6, 3.2]},
            20: {"Q": [0, 5, 10, 15, 20, 25], "dp": [7.2, 6.9, 6.2, 5.2, 4.0, 2.7]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.2, 2.7, 3.3, 4.0, 4.6, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.1, 2.6, 3.2, 3.9, 4.5, 4.9]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "P": [2.0, 2.5, 3.1, 3.8, 4.4, 4.8]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "P": [1.9, 2.4, 3.0, 3.6, 4.1, 4.5]},
            20: {"Q": [0, 5, 10, 15, 20, 25], "P": [1.8, 2.2, 2.7, 3.2, 3.7, 4.0]},
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
            20: {"Q": [0, 10, 20, 30, 40, 50], "dp": [15.3, 14.8, 13.5, 11.6, 9.0, 6.2]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "P": [3.0, 4.2, 5.8, 7.5, 9.0, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "P": [2.9, 4.1, 5.7, 7.3, 8.8, 9.8]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.8, 4.0, 5.5, 7.1, 8.6, 9.5]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.6, 3.8, 5.2, 6.8, 8.2, 9.0]},
            20: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.4, 3.5, 4.8, 6.2, 7.4, 8.0]},
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
            20: {"Q": [0, 20, 40, 60, 80, 100], "dp": [23.8, 22.8, 20.4, 17.4, 13.2, 9.2]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.5, 7.0, 9.5, 12.5, 15.5, 18.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.3, 6.8, 9.2, 12.1, 15.0, 17.4]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "P": [5.1, 6.5, 8.8, 11.6, 14.2, 16.5]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.8, 6.2, 8.3, 10.9, 13.5, 15.6]},
            20: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.4, 5.6, 7.5, 9.7, 11.8, 13.3]},
        }
    }
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
# Henry Parameter (Platzhalterwerte)
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
        return ys[0]
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

def bar_from_head_m(H_m, rho):
    # p_abs [bar] aus Förderhöhe (hydraulische Druckhöhe)
    return max(0.0001, (rho * G * max(H_m, 0.0)) / 1e5)

def head_m_from_bar(dp_bar, rho):
    return (dp_bar * 1e5) / (rho * G)

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
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / max(CH, 1e-12)}

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
# Einphasen: Drehzahl so, dass Kennlinie Sollpunkt trifft (Affinität, pragmatisch)
# =========================
def _H_scaled_at_Q_from_curve(curve_Q, curve_H, Q_req, n_ratio):
    # Q_base = Q_req / n_ratio; H_scaled = H(Q_base)*n_ratio^2
    Q_base = Q_req / max(n_ratio, 1e-9)
    H_base = interp_clamped(Q_base, curve_Q, curve_H)
    return H_base * (n_ratio ** 2)

def find_speed_ratio_for_head(curve_Q, curve_H, Q_req, H_req, n_min=0.3, n_max=1.2, tol=1e-4, iters=80):
    f_min = _H_scaled_at_Q_from_curve(curve_Q, curve_H, Q_req, n_min) - H_req
    f_max = _H_scaled_at_Q_from_curve(curve_Q, curve_H, Q_req, n_max) - H_req
    if f_min * f_max > 0:
        return None
    lo, hi = n_min, n_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _H_scaled_at_Q_from_curve(curve_Q, curve_H, Q_req, mid) - H_req
        if abs(f_mid) <= tol:
            return mid
        if f_min * f_mid <= 0:
            hi = mid
            f_max = f_mid
        else:
            lo = mid
            f_min = f_mid
    return 0.5 * (lo + hi)

def scaled_power_from_curve(curve_Q, curve_P, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-9)
    P_base = interp_clamped(Q_base, curve_Q, curve_P)
    return P_base * (n_ratio ** 3)

# =========================
# Henry / Gaslöslichkeit (bei Betriebsbedingungen) + Normvolumen
# =========================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def gas_solubility_L_per_L_oper(gas, p_bar_abs, T_celsius, y_gas=1.0):
    p = max(p_bar_abs, 1e-6)
    T_K = T_celsius + 273.15
    H = henry_constant(gas, T_celsius)  # bar·L/mol
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    C_mol_L = p_partial / max(H, 1e-12)
    V_molar_oper = R_BAR_L * T_K / p
    return C_mol_L * V_molar_oper  # L_gas_oper / L_liq

def oper_to_norm_volume_ratio(p_oper_bar_abs, T_oper_c):
    T_oper_K = T_oper_c + 273.15
    return (p_oper_bar_abs / P_N_BAR) * (T_N_K / max(T_oper_K, 1e-9))

def gas_solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    V_oper_L_per_L = gas_solubility_L_per_L_oper(gas, p_bar_abs, T_celsius, y_gas=y_gas)
    ratio = oper_to_norm_volume_ratio(p_bar_abs, T_celsius)
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L

def solubility_diagonal_curve_cm3N_per_L(gas, T_celsius, p_max=14, y_gas=1.0):
    pressures = linspace(0.1, p_max, 140)
    sol_cm3N_L = [gas_solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol_cm3N_L

def gvf_to_Vgas_oper_L_per_L(gvf_pct):
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    return gvf / (1.0 - gvf) * 1.0  # bei Vliq=1L

def Vgas_oper_L_per_L_to_gvf(Vgas_oper_L_per_L):
    Vliq = 1.0
    Vgas = max(0.0, Vgas_oper_L_per_L) * Vliq
    return 100.0 * (Vgas / (Vgas + Vliq))

# =========================
# Gasdefinition Druckseite -> Rückrechnung Saugseite (in Normvolumen pro Liter)
# =========================
def total_gas_norm_from_gvf_out_dissolved(gvf_out_pct, p_d_bar_abs, T_c, rho_liq):
    """
    GVF_out (%) ist als "bei p_d gelöst" gemeint.
    -> Wir interpretieren das als Gasvolumen (operativ) pro Liter Flüssigkeit am Druckpunkt.
    -> Umrechnung in Normvolumen pro Liter (L_N/L).
    """
    Vgas_oper_L_per_L = gvf_to_Vgas_oper_L_per_L(gvf_out_pct)
    ratio = oper_to_norm_volume_ratio(p_d_bar_abs, T_c)
    Vgas_norm_L_per_L = Vgas_oper_L_per_L * ratio
    return Vgas_norm_L_per_L  # L_N / L_liq

def total_gas_norm_from_recycle(Q_rec_m3h, Q_liq_m3h, gas, p_d_bar_abs, T_c, y_gas=1.0):
    """
    Recyclingstrom Q_rec ist "gesättigt an der Druckseite".
    -> gesättigte Löslichkeit (Normvolumen pro Liter) bei p_d:
       sol_cm3N/L -> sol_LN/L = sol/1000
    -> Gas-Normstrom = sol_LN/L * Q_rec(L/h)
    -> auf Hauptstrom Q_liq umgelegt: total_norm_L_per_L = GasNormFlow / Q_liq(L/h)
    """
    Q_liq_m3h = max(Q_liq_m3h, 1e-9)
    sol_cm3N_L = gas_solubility_cm3N_per_L(gas, p_d_bar_abs, T_c, y_gas=y_gas)
    sol_LN_per_L = sol_cm3N_L / 1000.0

    Q_rec_Lph = max(Q_rec_m3h, 0.0) * 1000.0
    Q_liq_Lph = Q_liq_m3h * 1000.0

    gas_norm_flow_Lph = sol_LN_per_L * Q_rec_Lph
    total_norm_L_per_L = gas_norm_flow_Lph / Q_liq_Lph
    return total_norm_L_per_L

def free_gvf_at_suction_from_total_norm(total_gas_norm_L_per_L, gas, p_s_bar_abs, T_c, y_gas=1.0):
    """
    total_gas_norm_L_per_L bleibt konstant.
    Am Saugpunkt ist nur solubility_norm möglich (Kapazität).
    Überschuss wird frei (Normvolumen), dann in operatives Volumen und GVF umgerechnet.
    """
    sol_cm3N_L_s = gas_solubility_cm3N_per_L(gas, p_s_bar_abs, T_c, y_gas=y_gas)
    sol_LN_per_L_s = sol_cm3N_L_s / 1000.0

    free_norm_L_per_L = max(0.0, total_gas_norm_L_per_L - sol_LN_per_L_s)

    # zurück in operatives Volumen am Saugpunkt:
    ratio_oper_to_norm = oper_to_norm_volume_ratio(p_s_bar_abs, T_c)
    free_oper_L_per_L = free_norm_L_per_L / max(ratio_oper_to_norm, 1e-12)

    gvf_free_pct = Vgas_oper_L_per_L_to_gvf(free_oper_L_per_L)

    return {
        "sol_cm3N_L_s": sol_cm3N_L_s,
        "sol_LN_per_L_s": sol_LN_per_L_s,
        "free_norm_L_per_L": free_norm_L_per_L,
        "free_oper_L_per_L": free_oper_L_per_L,
        "gvf_free_pct": gvf_free_pct
    }

# =========================
# Mehrphase: Affinität (Q~n, Δp~n², P~n³) über Q_base
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
    best = None
    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        if Q_req > pump["Q_max_m3h"] * 1.1:
            continue

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
                    "P_required": P_nom, "n_ratio": 1.0, "mode": "Nenndrehzahl",
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
                    score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.25
                    candidates.append({
                        "pump": pump, "gvf_curve": gvf_key, "dp_available": dp_scaled,
                        "P_required": P_scaled, "n_ratio": n_ratio,
                        "mode": "Drehzahl angepasst",
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

    n0_rpm = st.number_input("Nenndrehzahl n₀ [1/min]", 500, 6000, 2900, 50)

    Q_vis_req = st.number_input("Qᵥ (Soll) [m³/h]", 0.1, 300.0, 40.0, 1.0)
    H_vis_req = st.number_input("Hᵥ (Soll) [m]", 0.1, 300.0, 35.0, 1.0)

    mk = st.selectbox("Medium", list(MEDIA.keys()), 0)
    rho_def, nu_def = MEDIA[mk]
    rho = st.number_input("ρ [kg/m³]", 1.0, 2000.0, float(rho_def), 5.0)
    nu = st.number_input("ν [cSt]", 0.1, 1000.0, float(nu_def), 0.5)

    allow_out = st.checkbox("Auswahl außerhalb Kennlinie", True)
    reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)

    return n0_rpm, Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct

def sidebar_inputs_mph():
    st.divider()
    st.subheader("⚙️ Medium / Gas / Drehzahl")
    gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
    temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)

    rho_liq = st.number_input("ρ Flüssigkeit [kg/m³]", 800.0, 1300.0, 998.0, 1.0)
    n0_rpm = st.number_input("Nenndrehzahl n₀ [1/min]", 500, 6000, 2900, 50)

    st.divider()
    st.subheader("Betriebspunkt (Hydraulik)")
    Q_req = st.number_input("Volumenstrom Flüssigkeit Q_liq [m³/h]", 0.1, 150.0, 15.0, 1.0)

    st.caption("Druck als **Förderhöhe (absolut)**: H_abs = p_abs / (ρ g)")
    Hs_abs = st.number_input("Saugseite H_s,abs [m]", 0.1, 5000.0, 20.0, 1.0)
    Hd_abs = st.number_input("Druckseite H_d,abs [m]", 0.1, 5000.0, 70.0, 1.0)

    dp_head_m = max(0.0, Hd_abs - Hs_abs)

    # für Henry brauchen wir p_abs in bar:
    p_suction_bar_abs = bar_from_head_m(Hs_abs, rho_liq)
    p_discharge_bar_abs = bar_from_head_m(Hd_abs, rho_liq)

    st.divider()
    st.subheader("Gasdefinition (Druckseite)")

    mode = st.radio(
        "Wie ist das Gas vorgegeben?",
        ["GVF_out [%] (an Druckseite im Wasser gelöst)",
         "Recyclingstrom Q_rec [m³/h] (gesättigt an Druckseite)"],
        index=0
    )

    gvf_out_pct = None
    Q_rec_m3h = None

    if mode.startswith("GVF_out"):
        gvf_out_pct = st.slider("GVF_out [%] (gelöst bei H_d)", 0.0, 40.0, 10.0, 0.1)
    else:
        Q_rec_m3h = st.number_input("Q_rec [m³/h]", 0.0, 200.0, 3.0, 0.1)

    st.divider()
    st.subheader("Darstellung")
    show_temp_band = st.checkbox("Löslichkeit bei T-10/T/T+10", value=True)

    return {
        "gas_medium": gas_medium,
        "temperature": temperature,
        "rho_liq": rho_liq,
        "n0_rpm": n0_rpm,
        "Q_req": Q_req,
        "Hs_abs": Hs_abs,
        "Hd_abs": Hd_abs,
        "dp_head_m": dp_head_m,
        "p_suction_bar_abs": p_suction_bar_abs,
        "p_discharge_bar_abs": p_discharge_bar_abs,
        "mode": mode,
        "gvf_out_pct": gvf_out_pct,
        "Q_rec_m3h": Q_rec_m3h,
        "show_temp_band": show_temp_band
    }

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
        n0_rpm, Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct = sidebar_inputs_pump()

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

    # Betriebspunkt viskos (Sollpunkt)
    eta_vis_soll = max(1e-6, eta_water * Ceta)
    P_hyd_W_soll = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW_soll = (P_hyd_W_soll / eta_vis_soll) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW_soll * (1.0 + reserve_pct / 100.0))

    # Kennlinien viskos generieren (bei n0)
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)

    # Betriebspunkt auf Kennlinien (rechts auf Kennlinie): gleicher Q, aber H aus Kennlinie
    H_water_on_curve = interp_clamped(Q_water, p["Qw"], p["Hw"])
    H_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, H_vis_curve)

    eta_water_on_curve = interp_clamped(Q_water, p["Qw"], p["eta"])
    eta_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, eta_vis_curve)

    P_water_on_curve = interp_clamped(Q_water, p["Qw"], p["Pw"])
    P_vis_on_curve = interp_clamped(Q_vis_req, Q_vis_curve, P_vis_curve)

    # Drehzahl/Leistung so, dass viskose Kennlinie den Sollpunkt trifft (pragmatisch via Affinität)
    n_ratio_opt = find_speed_ratio_for_head(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min=0.3, n_max=1.2)
    if n_ratio_opt is not None:
        n_opt_rpm = n0_rpm * n_ratio_opt
        P_opt_kW = scaled_power_from_curve(Q_vis_curve, P_vis_curve, Q_vis_req, n_ratio_opt)
        # Energieeinsparung relativ zu "auf Kennlinie" bei n0 (bei gleichem Q)
        base = max(P_vis_on_curve, 1e-9)
        saving_pct = 100.0 * max(0.0, (base - P_opt_kW) / base)
    else:
        n_opt_rpm = None
        P_opt_kW = None
        saving_pct = None

    st.divider()
    st.markdown("### ✅ Ergebnis (Einphasen)")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    # Q(viskos) NICHT als Delta-Zeile (kein L/min als Delta) -> nur Wert
    col1.metric("Qᵥ (Soll)", f"{Q_vis_req:.2f} m³/h")
    col2.metric("Hᵥ (Soll)", f"{H_vis_req:.2f} m")
    col3.metric("ηᵥ (Soll)", f"{eta_vis_soll:.3f}")
    col4.metric("P Welle (Soll)", f"{P_vis_kW_soll:.2f} kW")
    col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(
            f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
            f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
        )

    st.markdown("### 🌀 Drehzahl-Optimierung (damit Kennlinie den Sollpunkt trifft)")
    if n_opt_rpm is None:
        st.info("Keine eindeutige Drehzahllösung im Suchbereich (0.3…1.2·n₀).")
    else:
        cA, cB, cC, cD = st.columns(4)
        cA.metric("n₀", f"{n0_rpm:d} 1/min")
        cB.metric("n_opt", f"{int(round(n_opt_rpm))} 1/min", f"{(n_ratio_opt*100):.1f}% von n₀")
        cC.metric("P (auf Kennlinie bei n₀)", f"{P_vis_on_curve:.2f} kW")
        cD.metric("P_opt", f"{P_opt_kW:.2f} kW", f"−{saving_pct:.1f}%")

    # Kennlinienplots
    st.divider()
    st.markdown("### 📈 Kennlinien (Einphasen)")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        # Betriebspunkt "auf Kennlinie" (rechts auf Kennlinie)
        ax1.scatter([Q_water], [H_water_on_curve], marker="^", s=160, edgecolors="black",
                    linewidths=2, label="BP (Wasser auf Kennlinie)", zorder=5)
        ax1.scatter([Q_vis_req], [H_vis_on_curve], marker="x", s=220, linewidths=3,
                    label="BP (viskos auf Kennlinie)", zorder=6)
        # Sollpunkt (viskos)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="o", s=140, edgecolors="black",
                    linewidths=2, label="Sollpunkt (viskos)", zorder=7)

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
        ax2.scatter([Q_water], [eta_water_on_curve], marker="^", s=160, edgecolors="black",
                    linewidths=2, label="BP η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis_on_curve], marker="x", s=220, linewidths=3,
                    label="BP η (viskos auf Kennlinie)", zorder=6)
        ax2.scatter([Q_vis_req], [eta_vis_soll], marker="o", s=140, edgecolors="black",
                    linewidths=2, label="Soll η (viskos)", zorder=7)
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
        ax3.scatter([Q_water], [P_water_on_curve], marker="^", s=160, edgecolors="black",
                    linewidths=2, label="BP (Wasser)", zorder=5)
        ax3.scatter([Q_vis_req], [P_vis_on_curve], marker="x", s=220, linewidths=3,
                    label="BP (viskos auf Kennlinie)", zorder=6)
        ax3.scatter([Q_vis_req], [P_vis_kW_soll], marker="o", s=140, edgecolors="black",
                    linewidths=2, label="Sollpunkt (viskos)", zorder=7)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Gasdefinition Druckseite + Löslichkeit + ΔH-Kennlinien + Overlay")

    with st.sidebar:
        inp = sidebar_inputs_mph()

    gas_medium = inp["gas_medium"]
    temperature = inp["temperature"]
    rho_liq = inp["rho_liq"]
    n0_rpm = inp["n0_rpm"]
    Q_req = inp["Q_req"]
    Hs_abs = inp["Hs_abs"]
    Hd_abs = inp["Hd_abs"]
    dp_head_m = inp["dp_head_m"]
    p_suction = inp["p_suction_bar_abs"]
    p_discharge = inp["p_discharge_bar_abs"]
    mode = inp["mode"]
    gvf_out_pct = inp["gvf_out_pct"]
    Q_rec_m3h = inp["Q_rec_m3h"]
    show_temp_band = inp["show_temp_band"]

    y_gas = 1.0

    # Totalgas (Norm/L) aus Druckseiten-Definition
    if mode.startswith("GVF_out"):
        total_norm_L_per_L = total_gas_norm_from_gvf_out_dissolved(gvf_out_pct, p_discharge, temperature, rho_liq)
    else:
        total_norm_L_per_L = total_gas_norm_from_recycle(Q_rec_m3h, Q_req, gas_medium, p_discharge, temperature, y_gas=y_gas)

    # Saugseite: freie Gasphase aus total_norm - solubility_norm
    suction_state = free_gvf_at_suction_from_total_norm(total_norm_L_per_L, gas_medium, p_suction, temperature, y_gas=y_gas)
    gvf_free = suction_state["gvf_free_pct"]

    # Mehrphasenpumpe auswählen: Kennlinien liegen in Δp -> wir nutzen ΔH (Förderhöhe) überall in UI,
    # rechnen aber intern Δp [bar] als dp_req_bar:
    dp_req_bar = (dp_head_m * rho_liq * G) / 1e5

    best = choose_best_mph_pump(MPH_PUMPS, Q_req, dp_req_bar, gvf_free)

    # =========================
    # Plot 1+2: Löslichkeit (nur Förderhöhe markieren) + Kennlinien als ΔH
    # =========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Löslichkeit vs Förderhöhe (H_abs) statt p_abs
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]

    # x-Achse: H_abs in m (0.. entsprechend 14 bar)
    p_max_bar = 14.0
    H_max = head_m_from_bar(p_max_bar, rho_liq)

    for i, T in enumerate(temp_variants):
        pressures, sol_cm3N = solubility_diagonal_curve_cm3N_per_L(gas_medium, T, p_max=p_max_bar, y_gas=y_gas)
        Hs = [head_m_from_bar(p, rho_liq) for p in pressures]
        ax1.plot(Hs, sol_cm3N, "--", linewidth=2, color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C (cm³N/L)")

    # Markiere NUR Förderhöhenpunkte (Saug/Druck) auf Löslichkeitskennlinie (bei T)
    sol_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    sol_d = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature, y_gas=y_gas)

    ax1.scatter([Hs_abs], [sol_s], s=160, marker="o", edgecolors="black", linewidths=2,
                label="Saugseite (H_s,abs)", zorder=5)
    ax1.scatter([Hd_abs], [sol_d], s=160, marker="s", edgecolors="black", linewidths=2,
                label="Druckseite (H_d,abs)", zorder=6)

    ax1.set_xlabel("Förderhöhe H_abs [m]")
    ax1.set_ylabel("Löslichkeit [cm³N/L]")
    ax1.set_title("Gaslöslichkeit (Henry → Normvolumen)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, H_max)
    ax1.legend(fontsize=9)

    # Kennlinien (ΔH statt Δp)
    Q_req_lmin = m3h_to_lmin(Q_req)

    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]
        gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            H_curve = [head_m_from_bar(dp, rho_liq) for dp in curve["dp"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.55
            ax2.plot(Q_lmin, H_curve, "o-", linewidth=lw, alpha=alpha,
                     color=gvf_colors.get(gvf_key, "gray"),
                     label=f"{pump['id']} ({gvf_key}% GVF)")

        ax2.scatter([Q_req_lmin], [dp_head_m], s=180, marker="o",
                    edgecolors="black", linewidths=2, label="Betriebspunkt (ΔH_req)", zorder=5)

        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("ΔH [m]")
        ax2.set_title(f"Mehrphasen-Kennlinien (ΔH): {pump['id']}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Keine Pumpenauswahl möglich", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("ΔH [m]")
        ax2.set_title("Mehrphasen-Kennlinien (ΔH)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # =========================
    # Overlay wie Vorlage: Löslichkeit (Kapazität) + Linien "freie Gasmenge" (abnehmend) aus Druckseiten-Definition
    # (in L_N/min bezogen auf Q_liq) über H_abs (statt p_abs)
    # =========================
    st.markdown("## 📉 Overlay wie Vorlage: Pumpenkennlinien über Löslichkeit")

    def free_gas_LminN_over_head(total_norm_L_per_L, gas, T_c, Q_liq_m3h, rho_liq, H_min=0.0, H_max=0.0, n=160):
        Hs = linspace(max(0.0, H_min), max(H_min + 1e-6, H_max), n)
        Q_liq_Lmin = m3h_to_lmin(Q_liq_m3h)
        ys = []
        for H_abs in Hs:
            p_abs = bar_from_head_m(H_abs, rho_liq)
            sol_cm3N = gas_solubility_cm3N_per_L(gas, p_abs, T_c, y_gas=y_gas)
            sol_LN_per_L = sol_cm3N / 1000.0
            free_norm_L_per_L = max(0.0, total_norm_L_per_L - sol_LN_per_L)
            ys.append(free_norm_L_per_L * Q_liq_Lmin)  # L_N/min
        return Hs, ys

    fig3, ax3 = plt.subplots(figsize=(14, 6))

    # Löslichkeit (Kapazität) als L_N/min (auf Q_liq bezogen), damit es wirklich "überlagert" wird
    # Kapazität_LN/min = (sol_LN/L) * Q_liq(L/min)
    Q_liq_Lmin = m3h_to_lmin(Q_req)

    for i, T in enumerate(temp_variants):
        pressures, sol_cm3N = solubility_diagonal_curve_cm3N_per_L(gas_medium, T, p_max=p_max_bar, y_gas=y_gas)
        Hs_curve = [head_m_from_bar(p, rho_liq) for p in pressures]
        sol_LN_per_L = [v / 1000.0 for v in sol_cm3N]
        cap_LN_min = [s * Q_liq_Lmin for s in sol_LN_per_L]
        ax3.plot(Hs_curve, cap_LN_min, "--", linewidth=2, color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C (Kapazität)")

    # Freies Gas (aus Druckseiten-Definition) als "Kennlinie" (abnehmend)
    H_min = 0.0
    H_max_overlay = head_m_from_bar(p_max_bar, rho_liq)

    Hs_line, free_line = free_gas_LminN_over_head(total_norm_L_per_L, gas_medium, temperature, Q_req, rho_liq,
                                                  H_min=H_min, H_max=H_max_overlay, n=180)
    ax3.plot(Hs_line, free_line, "-", linewidth=3.0, color="tab:orange", label="freie Gasmenge (tatsächlich)")

    # Referenzen 10/15/20% als Druckseiten-GVF_out-gelöst (nur wenn GVF_out-Mode aktiv)
    if mode.startswith("GVF_out"):
        for gvf_ref, c in [(10.0, "tab:blue"), (15.0, "tab:green"), (20.0, "tab:red")]:
            total_ref = total_gas_norm_from_gvf_out_dissolved(gvf_ref, p_discharge, temperature, rho_liq)
            Hs_ref, free_ref = free_gas_LminN_over_head(total_ref, gas_medium, temperature, Q_req, rho_liq,
                                                       H_min=H_min, H_max=H_max_overlay, n=180)
            ax3.plot(Hs_ref, free_ref, "-", linewidth=2.2, color=c, label=f"{gvf_ref:.1f}% (Referenz)")

    # Markiere NUR Förderhöhe am Saugpunkt als Orientierung (Kapazität am Saugpunkt)
    p_s = bar_from_head_m(Hs_abs, rho_liq)
    sol_s_cm3N = gas_solubility_cm3N_per_L(gas_medium, p_s, temperature, y_gas=y_gas)
    sol_s_LN_per_L = sol_s_cm3N / 1000.0
    cap_s = sol_s_LN_per_L * Q_liq_Lmin
    ax3.scatter([Hs_abs], [cap_s], s=180, marker="o", edgecolors="black", linewidths=2,
                label="Saugseite (Kapazität)", zorder=6)

    ax3.set_xlabel("Förderhöhe H_abs [m]")
    ax3.set_ylabel("L_N/min (Kapazität / freies Gas)")
    ax3.set_title("Overlay: Löslichkeit (Kapazität) + freie Gasmenge (wie Vorlage)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, H_max_overlay)
    ax3.set_ylim(0, 180)
    ax3.legend(fontsize=9, ncol=2)
    st.pyplot(fig3, clear_figure=True)

    # =========================
    # Ergebnisse Mehrphase
    # =========================
    st.divider()
    st.markdown("### ✅ Ergebnisse (Mehrphase)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q_liq", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("ΔH_req", f"{dp_head_m:.2f} m", f"H_d={Hd_abs:.1f} | H_s={Hs_abs:.1f}")
    if mode.startswith("GVF_out"):
        c3.metric("Gasdefinition", f"GVF_out={gvf_out_pct:.2f} %", "gelöst bei H_d")
    else:
        c3.metric("Gasdefinition", f"Q_rec={Q_rec_m3h:.2f} m³/h", "gesättigt bei H_d")

    c4.metric("total Gas (Norm)", f"{total_norm_L_per_L:.4f} L_N/L", "konstant")
    c5.metric("GVF_free @ Saug", f"{gvf_free:.2f} %", "für Kennlinie")

    if best:
        pump = best["pump"]
        n_abs = int(round(n0_rpm * best["n_ratio"]))
        H_avail_m = head_m_from_bar(best["dp_available"], rho_liq)
        H_reserve_m = head_m_from_bar(best["dp_reserve"], rho_liq)
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{pump['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        a, b, c, d = st.columns(4)
        a.metric("ΔH verfügbar", f"{H_avail_m:.2f} m", f"Reserve: {H_reserve_m:.2f} m")
        b.metric("Leistung", f"{best['P_required']:.2f} kW")
        c.metric("Drehzahl n", f"{n_abs:d} 1/min", f"n/n₀={best['n_ratio']:.3f}")
        d.metric("GVF_max Pumpe", f"{pump['GVF_max']*100:.0f}%")
    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")

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

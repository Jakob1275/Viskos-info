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
# keys in curves: GVF in Prozent (0,5,10,15 ...)
# =========================
MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "dp_max_bar": 9,
        "GVF_max": 0.4,  # 40%
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
    # Vereinfachung: Q bleibt gleich, H wird auf Wasserkennlinie "hochgerechnet"
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / CH}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    # Realer Betrieb: H sinkt (CH < 1), eta sinkt (Ceta < 1)
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

def gvf_total_to_Vgas_oper_L_per_L(gvf_pct):
    """
    Setze V_liq = 1 L.
    GVF = V_gas / (V_gas + V_liq) -> V_gas = GVF/(1-GVF) * V_liq
    """
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    return gvf / (1.0 - gvf) * 1.0

def free_gvf_from_total_and_dissolved_oper(gvf_total_pct, dissolved_oper_L_per_L):
    """
    gelöstes Maximum (operativ) wird abgezogen:
    Vgas_free_oper = max(0, Vgas_total_oper - Vgas_diss_max_oper)
    und in freies GVF zurückgerechnet.
    """
    Vliq = 1.0
    Vgas_total = gvf_total_to_Vgas_oper_L_per_L(gvf_total_pct)
    Vgas_diss = max(dissolved_oper_L_per_L, 0.0) * Vliq
    Vgas_free = max(0.0, Vgas_total - Vgas_diss)

    gvf_free = Vgas_free / (Vgas_free + Vliq)
    return 100.0 * gvf_free, Vgas_total, Vgas_diss, Vgas_free

def gvf_from_airflow_Nm3h(Q_air_Nm3h, Q_liq_m3h, p_suction_bar_abs, T_celsius):
    """
    Umrechnung Nm³/h Luft (Norm) -> GVF_in am Saugpunkt.

    1) V_N,air [m³/h] gegeben
    2) in operatives Gasvolumen bei p_s, T:
       V_oper = V_N * (p_N/p_oper) * (T_oper/T_N)
    3) GVF = V_oper / (V_oper + Q_liq)
    """
    Q_liq = max(Q_liq_m3h, 1e-9)

    T_oper_K = T_celsius + 273.15
    V_oper_m3h = Q_air_Nm3h * (P_N_BAR / max(p_suction_bar_abs, 1e-9)) * (T_oper_K / T_N_K)

    gvf = V_oper_m3h / (V_oper_m3h + Q_liq)
    return 100.0 * gvf, V_oper_m3h

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

                # WICHTIG: als "gültig" akzeptieren wir, wenn dp_scaled >= dp_req (numerische Toleranz)
                if dp_scaled + 1e-6 >= dp_req:
                    score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.25
                    candidates.append({
                        "pump": pump, "gvf_curve": gvf_key, "dp_available": dp_scaled,
                        "P_required": P_scaled, "n_ratio": n_ratio,
                        "mode": f"Drehzahl {n_ratio*100:.1f}%",
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

    return Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct

def sidebar_inputs_mph():
    st.divider()
    st.subheader("⚙️ Medium / Gas")
    gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
    temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)

    st.divider()
    st.subheader("Betriebspunkt (Hydraulik)")
    Q_req = st.number_input("Volumenstrom Flüssigkeit Q_liq [m³/h]", 0.1, 150.0, 15.0, 1.0)
    p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 200.0, 2.0, 0.1)
    p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 400.0, 7.0, 0.1)

    dp_req = max(0.0, (p_discharge - p_suction))  # ohne Δp_loss (wie gewünscht)

    st.divider()
    st.subheader("Gasanteil am Saugpunkt (Input wählbar)")
    mode = st.radio("Eingabeart", ["GVF_in [%]", "Q_air_N [Nm³/h]"], index=0)

    gvf_in_pct = None
    Q_air_Nm3h = None
    V_air_oper_m3h = None

    if mode == "GVF_in [%]":
        gvf_in_pct = st.slider("Gesamt-GVF_in [%] (am Saugpunkt)", 0.0, 40.0, 10.0, 0.1)
    else:
        Q_air_Nm3h = st.number_input("Luftstrom Q_air,N [Nm³/h]", 0.0, 200.0, 3.0, 0.1)
        gvf_in_pct, V_air_oper_m3h = gvf_from_airflow_Nm3h(Q_air_Nm3h, Q_req, p_suction, temperature)
        st.caption(f"→ daraus berechnet: GVF_in ≈ {gvf_in_pct:.2f}% (bei p_s,T)")

    st.divider()
    st.subheader("Darstellung / Logik")
    show_temp_band = st.checkbox("Löslichkeit bei T-10/T/T+10", value=True)
    only_dissolved_allowed = st.checkbox("Nur gelöst zulässig (wie Folie: 'nicht möglich')", value=False)

    return (gas_medium, temperature, Q_req, p_suction, p_discharge, dp_req,
            mode, gvf_in_pct, Q_air_Nm3h, V_air_oper_m3h, show_temp_band, only_dissolved_allowed)

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
        Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct = sidebar_inputs_pump()

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

    st.divider()
    st.markdown("### ✅ Ergebnis (Einphasen)")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")
    col2.metric("H (viskos)", f"{H_vis_req:.2f} m")
    col3.metric("η (viskos)", f"{eta_vis:.3f}")
    col4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW")
    col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
                   f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h.")

    # Kennlinien viskos generieren
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)

    # Wasser-Punkt auf Kennlinie
    H_water_op = interp_clamped(Q_water, p["Qw"], p["Hw"])
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])

    st.divider()
    st.markdown("### 📈 Kennlinien")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        ax1.scatter([Q_water], [H_water_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser auf Kennlinie)", zorder=5)
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
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, linewidths=3,
                    label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    with st.expander("📘 Rechenweg (ausführlich)", expanded=False):
        st.markdown("""
### Schritt 1: HI-ähnliche Kennzahl B (Warum überhaupt?)
Bei viskosen Medien (z. B. Öl, Glykol) steigt die innere Reibung. Dadurch:
- sinken Förderhöhe und Förderstrom bei gleicher Drehzahl,
- sinkt der Wirkungsgrad,
- steigt die erforderliche Wellenleistung.

Damit wir trotzdem eine Pumpe aus **Wasserkennlinien** auswählen können, wird zuerst bewertet,
**wie stark** Viskositätseffekte sind. Das macht die Kennzahl **B**.

""")
        st.markdown("**Formel (HI-ähnlich, pragmatisch):**")
        st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q_{gpm}^{0.25} \cdot H_{ft}^{0.375}}")
        st.markdown("""
- \(\nu\) in **cSt**
- \(Q_{gpm}\) ist der Förderstrom in **gpm**
- \(H_{ft}\) ist die Förderhöhe in **ft**

**Interpretation:**
- **B < 1** → Viskositätseffekte klein → Wasserkennlinie passt fast
- **B ≥ 1** → Effekte relevant → Korrekturfaktoren werden angewendet
""")
        st.markdown("**Ergebnis in deinem Betriebspunkt:**")
        st.markdown(f"- B = **{B:.3f}**")

        st.markdown("---")
        st.markdown("""
### Schritt 2: Korrekturfaktoren aus B
Aus B werden (vereinfacht) zwei Faktoren abgeleitet:

- \(C_H\): reduziert die **Förderhöhe** (typisch < 1 bei viskos)
- \(C_\eta\): reduziert den **Wirkungsgrad** (typisch < 1 bei viskos)

""")
        st.latex(r"C_H = \text{f}(B)\quad,\qquad C_\eta = \text{f}(B)")
        st.markdown(f"- \(C_H\) = **{CH:.3f}**  → H_vis = H_w · C_H")
        st.markdown(f"- \(C_\\eta\) = **{Ceta:.3f}** → η_vis = η_w · C_η")

        st.markdown("---")
        st.markdown("""
### Schritt 3: Umrechnung viskos → Wasserkennlinie (für die Auswahl)
Wir kennen den gewünschten viskosen Betriebspunkt \((Q_v, H_v)\).
Für die Auswahl auf Wasserkennlinie wird auf einen äquivalenten Wasserpunkt umgerechnet:

- \(Q_w = Q_v\) (hier pragmatisch unverändert)
- \(H_w = \frac{H_v}{C_H}\)  (weil bei viskos die Höhe sinkt)

""")
        st.latex(r"Q_w = Q_v")
        st.latex(r"H_w = \frac{H_v}{C_H}")
        st.markdown(f"- \(Q_w\) = **{Q_water:.2f} m³/h**")
        st.markdown(f"- \(H_w\) = **{H_water:.2f} m**")

        st.markdown("---")
        st.markdown("""
### Schritt 4: Auswahl + Leistung
1) Wähle die Pumpe, deren Wasserkennlinie bei \(Q_w\) möglichst nahe an \(H_w\) liegt.  
2) Rechne den Betriebspunkt wieder zurück auf viskos:
- \(H_v = H_w \cdot C_H\)
- \(\eta_v = \eta_w \cdot C_\eta\)

3) Wellenleistung:
""")
        st.latex(r"P_{hyd} = \rho g \, Q \, H")
        st.latex(r"P_{shaft} = \frac{P_{hyd}}{\eta_v}")

        st.markdown(f"- \(P_{{hyd}}\) = **{P_hyd_W:.0f} W**")
        st.markdown(f"- \(P_{{shaft}}\) = **{P_vis_kW:.2f} kW**")
        st.markdown(f"- Motor (IEC, +Reserve) = **{P_motor_kW:.2f} kW**")

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit (p,T) + GVF + Δp-Kennlinien + Auswahl")

    with st.sidebar:
        (gas_medium, temperature, Q_req, p_suction, p_discharge, dp_req,
         mode, gvf_in, Q_air_Nm3h, V_air_oper_m3h, show_temp_band, only_dissolved_allowed) = sidebar_inputs_mph()

    # y_gas fix = 1.0 (wie du wolltest)
    y_gas = 1.0

    # 1) Löslichkeit am Saugpunkt (operativ) und als Normvolumen (für diagonale Darstellung)
    dissolved_oper_L_per_L = gas_solubility_L_per_L_oper(gas_medium, p_suction, temperature, y_gas=y_gas)
    dissolved_cm3N_L = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)

    # 2) GVF_in -> freies Gas (Worst Case)
    gvf_free, Vgas_total_oper, Vgas_diss_oper, Vgas_free_oper = free_gvf_from_total_and_dissolved_oper(gvf_in, dissolved_oper_L_per_L)

    # "wie Folie": wenn nur gelöst zulässig und GVF_in > löslich -> nicht möglich
    # Dazu vergleichen wir Gasmenge aus GVF_in (operativ) gegen gelöstes Maximum (operativ).
    not_solvable = Vgas_total_oper > (Vgas_diss_oper + 1e-12)

    if only_dissolved_allowed and not_solvable:
        st.error("❌ Nicht möglich im Sinne der Folie: Der eingestellte Gasanteil ist am Saugpunkt nicht vollständig löslich.")
        st.info("Wenn du trotzdem weiter auslegen willst (Worst Case über freie Gasphase), deaktiviere 'Nur gelöst zulässig'.")
        # Trotzdem die Grafiken zeigen, aber keine Pumpenauswahl.
        best = None
    else:
        best = choose_best_mph_pump(MPH_PUMPS, Q_req, dp_req, gvf_free)

    # =========================
    # Plot 1+2 (oben): diagonale Löslichkeit + Δp-Kennlinien
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
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C (cm³N/L)")

    ax1.scatter([p_suction], [dissolved_cm3N_L], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt (Löslichkeit)", zorder=5)
    ax1.set_xlabel("p_abs [bar]")
    ax1.set_ylabel("Löslichkeit [cm³N/L]")
    ax1.set_title("Gaslöslichkeit (Henry → Normvolumen, diagonal)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Plot 2: Δp-Q Kennlinien
    Q_req_lmin = m3h_to_lmin(Q_req)
    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]
        gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.55
            ax2.plot(Q_lmin, curve["dp"], "o-", linewidth=lw, alpha=alpha,
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
        ax2.set_ylim(0, pump["dp_max_bar"] * 1.1)
    else:
        ax2.text(0.5, 0.5, "Keine Pumpenauswahl (siehe Hinweis)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("Δp [bar]")
        ax2.set_title("Mehrphasen-Kennlinien (Δp)")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # =========================
    # Plot 3 (Overlay wie Vorlage):
    # "Pumpenkennlinien (Gasvolumenstrom) über Löslichkeit (diagonal)"
    # =========================
    st.markdown("## 📉 Overlay: Pumpenkennlinien (Gasvolumenstrom) über Löslichkeit (diagonal)")

    fig3, ax3 = plt.subplots(figsize=(14, 6))

    # Diagonale Löslichkeit (cm³N/L) für T-10/T/T+10
    for i, T in enumerate(temp_variants):
        pressures, sol_cm3N = solubility_diagonal_curve_cm3N_per_L(gas_medium, T, p_max=14, y_gas=y_gas)
        ax3.plot(pressures, sol_cm3N, "--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    # Marker am Saugpunkt (Löslichkeit)
    ax3.scatter([p_suction], [dissolved_cm3N_L], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt (Löslichkeit)", zorder=5)

    # Pumpen-"Kennlinien" als Gasvolumenstrom (bei p): y-Achse in "L/min Gas (Norm)" bzw cm³N/L Skala
    # Idee wie Folie: "Scale: 100 Ncm³/Liter is 10%"
    #
    # Wir zeigen für:
    # - tatsächlicher GVF_in (aus Slider oder aus Nm³/h berechnet)
    # - Referenzen 10/15/20%
    #
    # Dazu brauchen wir Gasstrom als Funktion von Druck: Q_gas,N(p) bei konstantem GVF_in und konstantem Q_liq.
    #
    # Schritt:
    #   Vgas_total_oper/L = GVF/(1-GVF)  (bei Betrieb)
    #   davon gelöst max (oper) = dissolved_oper_L_per_L
    #   freies Gas oper/L = max(0, total - dissolved)
    #   -> in Normvolumen umrechnen: * ratio_oper_to_norm
    #
    # Um eine "Kennlinie" zu erzeugen, vary p_abs entlang x-Achse und berechne freies Gas pro Liter (Norm) und
    # daraus Gasvolumenstrom (Norm) = (freies Gas pro Liter) * Q_liq
    #
    # Anschließend plotten wir Gasstrom in L/min_N (passt zur Vorlage-Optik).

    def free_gas_flow_curve_LminN(gvf_in_pct, T_c, Q_liq_m3h, p_min=0.1, p_max=14, n=120):
        ps = linspace(p_min, p_max, n)
        Q_liq_Lmin = m3h_to_lmin(Q_liq_m3h)

        ys = []
        for p_abs in ps:
            dissolved_oper = gas_solubility_L_per_L_oper(gas_medium, p_abs, T_c, y_gas=y_gas)
            gvf_free_pct_local, Vtot_oper, Vdiss_oper, Vfree_oper = free_gvf_from_total_and_dissolved_oper(gvf_in_pct, dissolved_oper)

            # freies Gas pro Liter Flüssigkeit (oper) = Vfree_oper (da Vliq=1L)
            ratio = oper_to_norm_volume_ratio(p_abs, T_c)
            Vfree_norm_L_per_L = Vfree_oper * ratio  # L_N / L_liq

            # Gasstrom Norm [L_N/min] = (L_N/L) * Q_liq [L/min]
            Qgas_LminN = Vfree_norm_L_per_L * Q_liq_Lmin
            ys.append(Qgas_LminN)
        return ps, ys

    # tatsächlicher GVF_in (aus gewählter Eingabeart)
    gvf_actual = float(gvf_in)

    # Referenzen: 10/15/20 plus actual (wenn nicht nah dran)
    ref_list = [10.0, 15.0, 20.0]
    if all(abs(gvf_actual - r) > 0.2 for r in ref_list):
        ref_list = [gvf_actual] + ref_list
    else:
        # wenn nahezu identisch, zeigen wir trotzdem "actual" gesondert
        ref_list = [gvf_actual] + ref_list

    # Linien zeichnen
    line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, gvf_ref in enumerate(ref_list):
        ps, qgas = free_gas_flow_curve_LminN(gvf_ref, temperature, Q_req, p_min=0.1, p_max=14, n=140)
        label = f"{gvf_ref:.1f}% GVF (Gasstrom als Pumpenkennlinie)"
        lw = 3.0 if abs(gvf_ref - gvf_actual) < 1e-6 else 2.0
        ax3.plot(ps, qgas, "-", linewidth=lw, color=line_colors[i % len(line_colors)], label=label)

    ax3.set_xlabel("Druck [bar abs]")
    ax3.set_ylabel("Löslichkeit [cm³N/L]  /  Gasvolumenstrom [L_N/min]")
    ax3.set_title("Overlay wie Vorlage: Pumpenkennlinien (Gasstrom) über Löslichkeit")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 180)  # wie deine Vorlage (Skala)
    ax3.legend(fontsize=9, ncol=2)

    st.pyplot(fig3, clear_figure=True)

    # =========================
    # Ergebnisse
    # =========================
    st.divider()
    st.markdown("### ✅ Ergebnisse (Mehrphase)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q_liq", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("Δp_req", f"{dp_req:.3f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
    if mode == "Q_air_N [Nm³/h]":
        c3.metric("Q_air,N", f"{Q_air_Nm3h:.2f} Nm³/h", f"GVF_in≈{gvf_in:.2f}%")
    else:
        c3.metric("GVF_in", f"{gvf_in:.2f} %")
    c4.metric("Löslichkeit (Saugseite)", f"{dissolved_oper_L_per_L:.6f} L/L (oper)", f"{dissolved_cm3N_L:.2f} cm³N/L")
    c5.metric("GVF_free (für Kennlinie)", f"{gvf_free:.2f} %")

    if not_solvable:
        st.warning("⚠️ Der eingestellte Gasanteil ist am Saugpunkt NICHT vollständig löslich → freie Gasphase entsteht (Worst Case relevant).")
    else:
        st.info("ℹ️ Der eingestellte Gasanteil liegt im Löslichkeitslimit (am Saugpunkt).")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Δp verfügbar", f"{best['dp_available']:.3f} bar", f"Reserve: {best['dp_reserve']:.3f} bar")
        c2.metric("Leistung", f"{best['P_required']:.2f} kW")
        c3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        c4.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

        # Nur noch warnen, wenn wirklich unterschritten (Toleranz)
        if best["dp_available"] + 1e-6 < dp_req:
            st.error("❌ Δp_req wird nicht erreicht (sollte bei Auswahl nicht passieren).")
    else:
        if not (only_dissolved_allowed and not_solvable):
            st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")

    with st.expander("📘 Rechenweg (ausführlich)", expanded=False):
        st.markdown(f"""
## Schritt 1: Δp-Anforderung (ohne Zusatzverluste)
Da **Δp_loss entfernt** ist, gilt:

""")
        st.latex(r"\Delta p_{req} = p_d - p_s")
        st.markdown(f"- p_s = {p_suction:.2f} bar abs  \n- p_d = {p_discharge:.2f} bar abs  \n⇒ **Δp_req = {dp_req:.3f} bar**")

        st.markdown("---")
        st.markdown("## Schritt 2: Löslichkeit am Saugpunkt (Henry + ideales Gas)")
        st.markdown("Wir berechnen das maximal lösliche Gasvolumen bezogen auf **1 Liter Flüssigkeit** am Saugpunkt.")

        st.markdown("1) Henry (Konzentration):")
        st.latex(r"C = \frac{p_{partial}}{H(T)}")

        st.markdown("2) Ideales Gas (Molarvolumen bei Betriebsbedingungen):")
        st.latex(r"V_{molar,oper} = \frac{RT}{p}")

        st.markdown("3) Gasvolumen pro Liter Flüssigkeit (operativ):")
        st.latex(r"\frac{V_{gas,oper}}{V_{liq}} = C \cdot V_{molar,oper}")

        st.markdown(f"Ergebnis am Saugpunkt:\n- **{dissolved_oper_L_per_L:.6f} L/L (operativ)**")

        st.markdown("Für die **diagonale Darstellung** wird in **Normvolumen** umgerechnet:")
        st.latex(r"V_N = V_{oper}\cdot\frac{p_{oper}}{p_N}\cdot\frac{T_N}{T_{oper}}")
        st.markdown(f"⇒ **Löslichkeit = {dissolved_cm3N_L:.2f} cm³N/L**")

        st.markdown("---")
        st.markdown("## Schritt 3: GVF_in (aus % oder aus Nm³/h) und freies Gas")
        if mode == "Q_air_N [Nm³/h]":
            st.markdown("Du gibst einen Luftstrom in **Nm³/h** ein. Daraus wird der GVF am Saugpunkt berechnet:")
            st.latex(r"V_{oper} = V_N \cdot \frac{p_N}{p_{oper}}\cdot\frac{T_{oper}}{T_N}")
            st.latex(r"GVF = \frac{V_{oper}}{V_{oper}+Q_{liq}}")
            st.markdown(f"- Eingabe: Q_air,N = **{Q_air_Nm3h:.2f} Nm³/h**")
            st.markdown(f"- Umgerechnet: GVF_in ≈ **{gvf_in:.2f}%** (bei p_s,T)")

        st.markdown("Anschließend (Worst Case) wird geprüft, ob alles löslich ist:")
        st.latex(r"V_{gas,total} = \frac{GVF}{1-GVF}\cdot V_{liq}")
        st.latex(r"V_{gas,free} = \max(0, V_{gas,total}-V_{gas,diss,max})")
        st.latex(r"GVF_{free} = \frac{V_{gas,free}}{V_{gas,free}+V_{liq}}")

        st.markdown(f"- V_gas,total (oper/L) = **{Vgas_total_oper:.6f}**")
        st.markdown(f"- V_gas,diss,max (oper/L) = **{Vgas_diss_oper:.6f}**")
        st.markdown(f"- V_gas,free (oper/L) = **{Vgas_free_oper:.6f}**")
        st.markdown(f"⇒ **GVF_free = {gvf_free:.2f}%**")

        st.markdown("---")
        st.markdown("## Schritt 4: Kennlinienwahl und Drehzahl (Affinität)")
        st.markdown("""
- Kennlinienwahl **Worst Case**: nächsthöhere GVF-Kurve ≥ GVF_free  
- Vergleich gegen Δp(Q)  
- Optional Drehzahl-Anpassung:
""")
        st.latex(r"Q \sim n,\quad \Delta p \sim n^2,\quad P \sim n^3")
        st.markdown("Die Bisektion sucht n/n0, so dass Δp(Q_req) = Δp_req erfüllt wird (sofern möglich).")

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

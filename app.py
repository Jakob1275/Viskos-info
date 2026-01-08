# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Konstanten
# =========================================================
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K)

# =========================================================
# Pumpenkennlinien (Einphasen) - Beispiel
# =========================================================
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

# =========================================================
# Mehrphasenpumpen (∆p-Kennlinien in bar)
# =========================================================
MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "p_max_bar": 9,
        "GVF_max": 0.4,  # 40%
        "curves_p_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "p": [8.6, 8.5, 8.2, 7.6, 6.8, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "p": [8.4, 8.3, 8.0, 7.3, 6.3, 4.6]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "p": [8.2, 8.0, 7.6, 6.7, 5.6, 4.0]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "p": [7.8, 7.5, 6.9, 5.8, 4.6, 3.2]},
            20: {"Q": [0, 5, 10, 15, 20, 25], "p": [7.3, 7.0, 6.2, 5.0, 3.9, 2.7]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.2, 2.7, 3.3, 4.0, 4.6, 5.0]},
            5:  {"Q": [0, 5, 10, 15, 20, 25], "P": [2.1, 2.6, 3.2, 3.9, 4.5, 4.9]},
            10: {"Q": [0, 5, 10, 15, 20, 25], "P": [2.0, 2.5, 3.1, 3.8, 4.4, 4.8]},
            15: {"Q": [0, 5, 10, 15, 20, 25], "P": [1.9, 2.4, 3.0, 3.6, 4.1, 4.5]},
            20: {"Q": [0, 5, 10, 15, 20, 25], "P": [1.8, 2.2, 2.8, 3.3, 3.7, 4.0]},
        }
    },
    {
        "id": "MPH-100",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 50,
        "p_max_bar": 20,
        "GVF_max": 0.4,
        "curves_p_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "p": [18.8, 18.5, 17.8, 16.0, 13.5, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "p": [18.2, 18.0, 17.0, 15.2, 12.6, 9.2]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "p": [17.5, 17.2, 16.0, 14.0, 11.5, 8.2]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "p": [16.5, 16.0, 14.8, 12.8, 10.0, 7.0]},
            20: {"Q": [0, 10, 20, 30, 40, 50], "p": [15.0, 14.6, 13.4, 11.5, 8.8, 6.0]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50], "P": [3.0, 4.2, 5.8, 7.5, 9.0, 10.0]},
            5:  {"Q": [0, 10, 20, 30, 40, 50], "P": [2.9, 4.1, 5.7, 7.3, 8.8, 9.8]},
            10: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.8, 4.0, 5.5, 7.1, 8.6, 9.5]},
            15: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.6, 3.8, 5.2, 6.8, 8.2, 9.0]},
            20: {"Q": [0, 10, 20, 30, 40, 50], "P": [2.4, 3.5, 4.8, 6.2, 7.5, 8.2]},
        }
    },
    {
        "id": "MPH-200",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 100,
        "p_max_bar": 30,
        "GVF_max": 0.4,
        "curves_p_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "p": [28.5, 28.0, 26.5, 23.5, 19.0, 14.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "p": [27.8, 27.0, 25.3, 22.0, 17.8, 13.0]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "p": [26.8, 26.0, 24.0, 20.8, 16.5, 12.0]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "p": [25.5, 24.6, 22.3, 19.0, 14.8, 10.5]},
            20: {"Q": [0, 20, 40, 60, 80, 100], "p": [23.5, 22.8, 20.5, 17.5, 13.0, 9.0]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.5, 7.0, 9.5, 12.5, 15.5, 18.0]},
            5:  {"Q": [0, 20, 40, 60, 80, 100], "P": [5.3, 6.8, 9.2, 12.1, 15.0, 17.4]},
            10: {"Q": [0, 20, 40, 60, 80, 100], "P": [5.1, 6.5, 8.8, 11.6, 14.2, 16.5]},
            15: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.8, 6.2, 8.3, 10.9, 13.5, 15.6]},
            20: {"Q": [0, 20, 40, 60, 80, 100], "P": [4.4, 5.7, 7.5, 9.8, 12.0, 13.8]},
        }
    }
]

# =========================================================
# ATEX-Datenbank
# =========================================================
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

# =========================================================
# Henry-Parameter (vereinfachte Beispielwerte)
# =========================================================
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

# =========================================================
# Viskositätskorrektur (HI-ähnliche pragmatische Näherung)
# =========================================================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    Q = max(Q_m3h, 1e-9)
    H = max(H_m, 1e-9)
    nu = max(nu_cSt, 1e-9)
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
    """
    Umrechnung: viskos -> Wasserkennlinie (vereinfachte HI-Logik)
    Q bleibt (hier) gleich, H_w = H_vis / CH
    """
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / max(CH, 1e-9)}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """
    Umrechnung: Wasser -> viskos für Kennlinien-Visualisierung
    H_vis = H_water * CH, eta_vis = eta_water * Ceta
    """
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return Q_water, H_water * CH, max(1e-9, eta_water * Ceta)

def shaft_power_kW(rho, Q_m3h, H_m, eta):
    """Physikalische Wellenleistung: P = rho*g*Q*H/eta"""
    Q = max(Q_m3h, 0.0) / 3600.0  # m³/s
    eta_eff = max(eta, 1e-9)
    return (rho * G * Q * H_m) / (1000.0 * eta_eff)

def generate_viscous_curve(pump, nu_cSt, rho):
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        P_v = shaft_power_kW(rho, Q_v, H_v, eta_v)
        Q_vis.append(Q_v)
        H_vis.append(H_v)
        eta_vis.append(eta_v)
        P_vis.append(P_v)
    return Q_vis, H_vis, eta_vis, P_vis

def generate_water_power_curve(pump, rho):
    """Wasser-Power aus Physik (nicht aus Dummy Pw)"""
    P = []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        P.append(shaft_power_kW(rho, Q_w, H_w, eta_w))
    return P

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
            "id": p["id"], "pump": p, "in_range": in_range, "Q_eval": Q_eval,
            "H_at": H_at, "eta_at": eta_at, "errH": abs(H_at - H_water), "score": score
        }

        if best is None or score < best["score"] - 1e-12:
            best = cand
        elif abs(score - best["score"]) <= 1e-12 and eta_at > best["eta_at"]:
            best = cand
    return best

# =========================================================
# Henry / Gaslöslichkeit
# =========================================================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def gas_solubility_L_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry: C [mol/L] = p_partial / H(T)
    Ideales Gas: V_molar [L/mol] = R*T/p_abs
    => V_gas/L_liquid = C*V_molar  (bei Betriebs-p,T)
    """
    p = max(p_bar_abs, 1e-9)
    T_K = T_celsius + 273.15
    H = henry_constant(gas, T_celsius)  # bar·L/mol
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    C_mol_L = p_partial / max(H, 1e-12)
    V_molar = R_BAR_L * T_K / p
    return C_mol_L * V_molar

def solubility_cm3_oper_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """Gasvolumen (Betriebs-p,T) in cm³/L"""
    return 1000.0 * gas_solubility_L_per_L(gas, p_bar_abs, T_celsius, y_gas=y_gas)

def solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Für die 'diagonalen' Linien wie in deiner Vorlage:
    Umrechnung des gelösten Gasvolumens auf Normalbedingungen (N):
    V_N = V_oper * (p_oper / p_N) * (T_N / T_oper)
    """
    pN = 1.01325
    TN = 273.15
    p = max(p_bar_abs, 1e-9)
    T = T_celsius + 273.15
    V_oper_L_per_L = gas_solubility_L_per_L(gas, p, T_celsius, y_gas=y_gas)
    V_N_L_per_L = V_oper_L_per_L * (p / pN) * (TN / T)
    return 1000.0 * V_N_L_per_L  # cm³N / L

def solubility_curve_vs_pressure_cm3N(gas, T_celsius, p_max=14, y_gas=1.0):
    pressures = linspace(0.0, p_max, 200)
    sol = [solubility_cm3N_per_L(gas, max(1e-6, p), T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol

def gvf_from_Vgas_Vliq(Vgas, Vliq):
    Vgas = max(Vgas, 0.0)
    Vliq = max(Vliq, 1e-12)
    return Vgas / (Vgas + Vliq)

def free_gvf_from_total_and_dissolved(gvf_total_pct, dissolved_L_per_L):
    """
    Setze V_liq = 1 L.
    Aus GVF_total: V_gas_total = gvf/(1-gvf)
    gelöst max: V_gas_diss = dissolved_L_per_L
    freies Gas: max(0, V_gas_total - V_gas_diss)
    """
    gvf = clamp(gvf_total_pct / 100.0, 0.0, 0.999999)
    Vliq = 1.0
    Vgas_total = gvf / (1.0 - gvf) * Vliq
    Vgas_diss_max = max(dissolved_L_per_L, 0.0) * Vliq
    Vgas_free = max(0.0, Vgas_total - Vgas_diss_max)
    return 100.0 * gvf_from_Vgas_Vliq(Vgas_free, Vliq)

# =========================================================
# Mehrphasen: Affinitätsgesetze
# =========================================================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-12)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2)

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_req,
                               n_min=0.5, n_max=1.2, tol=1e-4, iters=80):
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

def choose_best_mph_pump_normbased(pumps, Q_req, dp_req, gvf_free_req_pct, dp_margin=0.00):
    """
    dp_req in bar (∆p)
    gvf_free_req_pct in %
    dp_margin: Reservefaktor (z.B. 0.05=5%) – hier standardmäßig 0, weil du exakt sehen willst,
               ob dp_req erreicht wird.
    """
    best = None

    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        if Q_req > pump["Q_max_m3h"] * 1.1:
            continue

        gvf_key = choose_gvf_curve_key_worstcase(pump["curves_p_vs_Q"], gvf_free_req_pct)
        curve = pump["curves_p_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]

        Qmin, Qmax = min(curve["Q"]), max(curve["Q"])

        candidates = []

        # A) Nenndrehzahl
        if Qmin <= Q_req <= Qmax:
            dp_avail = interp_clamped(Q_req, curve["Q"], curve["p"])
            if dp_avail >= dp_req * (1.0 + dp_margin):
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail - dp_req) + abs(gvf_key - gvf_free_req_pct) * 0.2
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_avail,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "mode": "Nenndrehzahl",
                    "dp_reserve": dp_avail - dp_req,
                    "score": score
                })

        # B) Drehzahl-Anpassung
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["p"], Q_req, dp_req)
        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if Qmin <= Q_base <= Qmax:
                dp_scaled = _dp_scaled_at_Q(curve["Q"], curve["p"], Q_req, n_ratio)
                P_base = interp_clamped(Q_base, power_curve["Q"], power_curve["P"])
                P_scaled = P_base * (n_ratio ** 3)
                score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.2
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_scaled,
                    "P_required": P_scaled,
                    "n_ratio": n_ratio,
                    "mode": f"Drehzahl {n_ratio*100:.1f}%",
                    "dp_reserve": dp_scaled - dp_req,
                    "score": score
                })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

# =========================================================
# UI: Sidebar-Navigation (zentral, garantiert in Sidebar)
# =========================================================
def sidebar_navigation():
    st.sidebar.header("📍 Navigation")
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if c2.button("Mehrphasen", use_container_width=True):
        st.session_state.page = "mph"
    if c3.button("ATEX", use_container_width=True):
        st.session_state.page = "atex"

    st.sidebar.info(f"**Aktiv:** {st.session_state.page}")

# =========================================================
# UI: Sidebar Inputs pro Seite (nur st.sidebar nutzen!)
# =========================================================
def sidebar_inputs_pump():
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Eingaben (Einphasen)")
    Q_vis_req = st.sidebar.number_input("Qᵥ, Förderstrom [m³/h]", 0.1, 300.0, 40.0, 1.0)
    H_vis_req = st.sidebar.number_input("Hᵥ, Förderhöhe [m]", 0.1, 300.0, 35.0, 1.0)

    mk = st.sidebar.selectbox("Medium", list(MEDIA.keys()), 0)
    rho_def, nu_def = MEDIA[mk]
    rho = st.sidebar.number_input("ρ [kg/m³]", 1.0, 2000.0, float(rho_def), 5.0)
    nu = st.sidebar.number_input("ν [cSt]", 0.1, 1000.0, float(nu_def), 0.5)

    allow_out = st.sidebar.checkbox("Auswahl außerhalb Kennlinie", True)
    reserve_pct = st.sidebar.slider("Motorreserve [%]", 0, 30, 15)

    return Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct

def sidebar_inputs_mph():
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Medium / Gas")
    gas_medium = st.sidebar.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
    temperature = st.sidebar.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)

    st.sidebar.divider()
    st.sidebar.subheader("Betriebspunkt (Hydraulik)")
    Q_req = st.sidebar.number_input("Volumenstrom Q [m³/h]", 0.1, 150.0, 15.0, 1.0)
    p_suction = st.sidebar.number_input("Saugdruck p_s [bar abs]", 0.1, 100.0, 2.0, 0.1)
    p_discharge = st.sidebar.number_input("Druckseite p_d [bar abs]", 0.1, 200.0, 7.0, 0.1)

    st.sidebar.divider()
    st.sidebar.subheader("GVF")
    gvf_in = st.sidebar.slider("Gesamt-GVF_in [%] (am Saugpunkt)", 0, 40, 10, 1)

    st.sidebar.divider()
    st.sidebar.subheader("Plot")
    show_temp_band = st.sidebar.checkbox("Löslichkeit bei T-10/T/T+10", value=True)

    return gas_medium, temperature, Q_req, p_suction, p_discharge, gvf_in, show_temp_band

# =========================================================
# App Setup
# =========================================================
st.set_page_config(page_title="Pumpenauslegung", layout="wide")
st.title("Pumpenauslegungstool")

if "page" not in st.session_state:
    st.session_state.page = "pump"

# Sidebar zentral
sidebar_navigation()

# =========================================================
# PAGE 1: Einphasen (Viskosität)
# =========================================================
if st.session_state.page == "pump":
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur")

    Q_vis_req, H_vis_req, rho, nu, allow_out, reserve_pct = sidebar_inputs_pump()

    # Umrechnung viskos -> Wasser
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    st.info(f"{'✅' if B < 1.0 else '⚠️'} B = {B:.2f} "
            f"{'< 1.0 → geringe Viskositätseffekte' if B < 1.0 else '≥ 1.0 → Viskositätskorrektur aktiv'}")

    st.markdown("### 📊 Umrechnung viskos → Wasser")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q_Wasser", f"{Q_water:.2f} m³/h")
    c2.metric("H_Wasser", f"{H_water:.2f} m", f"+{H_water - H_vis_req:.1f} m")
    c3.metric("B-Zahl", f"{B:.2f}")
    c4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")

    # Auswahl Pumpe auf Wasserkennlinie
    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]

    # Betriebspunkt Wirkungsgrad (Wasser) aus Kennlinie
    eta_water_op = best["eta_at"]
    eta_vis_op = max(1e-9, eta_water_op * Ceta)

    # Leistung am Betriebspunkt (viskos) physikalisch korrekt
    P_vis_kW = shaft_power_kW(rho, Q_vis_req, H_vis_req, eta_vis_op)
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    st.markdown("### ✅ **AUSLEGUNGSERGEBNIS (Einphasen)**")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} l/min")
    col2.metric("H (viskos)", f"{H_vis_req:.2f} m")
    col3.metric("η (viskos)", f"{eta_vis_op:.3f}")
    col4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW")
    col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
                   f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h.")

    # Kennlinien (viskos + Wasser)
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)
    rho_water_ref = 998.0  # Referenz für Wasser-Plot (passt zu MEDIA Default)
    P_water_curve = generate_water_power_curve(p, rho_water_ref)

    # Betriebspunkt P(wasser) (für Marker) ebenfalls physikalisch
    H_at_water = interp_clamped(Q_water, p["Qw"], p["Hw"])
    eta_at_water = interp_clamped(Q_water, p["Qw"], p["eta"])
    P_water_kW_op = shaft_power_kW(rho_water_ref, Q_water, H_at_water, eta_at_water)

    st.divider()
    st.markdown("### 📈 Kennlinien")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P (physikalisch)"])

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
        ax2.scatter([Q_water], [eta_water_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis_op], marker="x", s=200, linewidths=3,
                    label="η (viskos)", zorder=5)
        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η Kennlinien")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(p["Qw"], P_water_curve, "o-", linewidth=2, label=f"{p['id']} (Wasser, ρgQH/η)")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos, ρgQH/η)")
        ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, edgecolors="black",
                    linewidths=2, label="BP (Wasser)", zorder=5)
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, linewidths=3,
                    label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien (physikalisch konsistent)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    with st.expander("📘 Rechenweg – Schritt 1: HI-ähnliche Kennzahl B", expanded=False):

        st.markdown("""
### Konsequenz für die weitere Berechnung (warum wir überhaupt korrigieren)

Die Pumpenkennlinie in der Datenbank ist eine **Wasserkennlinie**.  
Das bedeutet: **Q-H, Wirkungsgrad η und Leistung P** gelten für ein Medium mit sehr niedriger Viskosität
(typisch Wasser bei ca. 20 °C).

Sobald das Medium deutlich **viskoser** ist (z. B. Öl, Glykol, zähes Prozessmedium),
ändert sich das Strömungsbild in der Pumpe:

- Reibungsverluste steigen,
- Umlenkverluste steigen,
- interne Leckage-/Spaltverluste wirken stärker,
- der Wirkungsgrad sinkt,
- die erreichbare Förderhöhe sinkt (bei gleicher Drehzahl),
- die Wellenleistung steigt (Motor kann überlastet werden).

Damit wir eine Pumpe **nicht zu optimistisch** auswählen, rechnen wir den viskosen Betrieb
in einen **äquivalenten Betriebspunkt auf der Wasserkennlinie** um und korrigieren danach zurück.

Die Kennzahl **B** ist dabei das zentrale Maß dafür,
**wie stark** diese Viskositätseinflüsse im relevanten Betriebspunkt sind.

---
""")

        st.markdown("""
### Bestimmung der Korrekturfaktoren (aus B)

Aus der Kennzahl **B** leiten wir Korrekturfaktoren ab.
Diese Korrekturfaktoren sind dimensionslos (reine Faktoren) und wirken wie „Skalierer“
für Kennlinienwerte.

Wir unterscheiden dabei:

- **C_Q**: Wie stark verschiebt sich der Förderstrom (hier in deiner Implementierung nicht aktiv genutzt,
  weil du Qᵥ → Q_w vereinfacht gleichsetzt).
- **C_H / CH**: Wie stark fällt die Förderhöhe ab (Förderhöhe im viskosen Betrieb ist kleiner).
- **C_η / Ceta**: Wie stark fällt der Wirkungsgrad ab (η_vis < η_w).

---
""")

        st.latex(r"""
C_Q \quad \text{(Korrekturfaktor für den Förderstrom)}
""")

        st.latex(r"""
C_H \quad \text{(Korrekturfaktor für die Förderhöhe)}
""")

        st.latex(r"""
C_\eta \quad \text{(Korrekturfaktor für den Wirkungsgrad)}
""")

        st.markdown("""
### Was bedeuten diese Faktoren im Code?

In deinem Code werden diese Faktoren so verwendet:

1) **Umrechnung viskos → Wasserpunkt (für die Pumpenauswahl)**

Du willst eine Pumpe auf der Wasserkennlinie finden, die im viskosen Betrieb
deinen Zielpunkt erreicht.

Dafür setzt du:

- Q_w = Qᵥ (Vereinfachung)
- H_w = Hᵥ / C_H

Warum Division?

Wenn eine Pumpe im viskosen Betrieb nur noch

- H_vis = H_w * C_H

liefert, musst du auf der Wasserkennlinie eine höhere Förderhöhe ansetzen,
damit nach Abfall wieder Hᵥ getroffen wird.

Damit suchst du die passende Pumpe auf der Wasserkennlinie mit dem Zielwert H_w.

---
""")

        st.latex(r"""
Q_w = Q_v
""")

        st.latex(r"""
H_w = \frac{H_v}{C_H}
""")

        st.markdown("""
2) **Wirkungsgrad-Korrektur für die Leistungsberechnung**

Die Leistung hängt stark vom Wirkungsgrad ab.
Deshalb korrigierst du den Wirkungsgrad:

- η_vis = η_w * C_η

Damit wird die berechnete Wellenleistung realistisch erhöht.

---
""")

        st.latex(r"""
\eta_{vis} = \eta_w \cdot C_\eta
""")

        st.markdown("""
### Anwendung auf die Wasserkennlinie (konkret)

Nachdem die Pumpe gewählt ist, nutzt du die Wasserkennlinie bei Q_w,
um die Grundwerte abzulesen (Interpolation):

- H_at(Q_w)   aus Q-H
- η_w(Q_w)    aus Q-η

Dann werden daraus viskose Werte gemacht:

- Förderhöhe im viskosen Betrieb (für die viskose Kennlinie):
  H_vis(Q) = H_w(Q) * C_H

- Wirkungsgrad im viskosen Betrieb:
  η_vis(Q) = η_w(Q) * C_η

- daraus folgt Wellenleistung:
  P_shaft = (ρ g Q H_vis) / η_vis

---
""")

        st.latex(r"""
P_{hyd} = \rho g Q H
""")

        st.latex(r"""
P_{shaft} = \frac{P_{hyd}}{\eta_{vis}}
""")

        st.markdown("""
### Ziel der Korrektur (warum das wichtig ist)

Dieses Vorgehen stellt sicher, dass:

- die Pumpenauswahl **nicht zu optimistisch** erfolgt,
- der Betriebspunkt **realistisch** getroffen wird,
- die **Wellenleistung nicht unterschätzt** wird (Motor-/Überlastschutz),
- und die Auslegung insgesamt **betriebssicher** bleibt.

Kurz:  
Die Kennzahl **B** ist die **Brücke zwischen idealer Wasserkennlinie und realem Anlagenbetrieb**
mit viskosem Medium.

---
""")



# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit (p,T) + freier GVF + ∆p-Kennlinien + Auswahl")

    gas_medium, temperature, Q_req, p_suction, p_discharge, gvf_in, show_temp_band = sidebar_inputs_mph()

    # y_gas bewusst entfernt, fix = 1.0 (wie gefordert)
    y_gas = 1.0

    # ∆p-Anforderung (ohne Zusatzverluste)
    dp_req = max(0.0, (p_discharge - p_suction))

    # Löslichkeit am Saugpunkt (für freie Gasabschätzung)
    dissolved_L_per_L = gas_solubility_L_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    dissolved_cm3_oper_L = 1000.0 * dissolved_L_per_L
    dissolved_cm3N_L = solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)

    # freier GVF (Worst-Case für Kennlinienwahl)
    gvf_free = free_gvf_from_total_and_dissolved(gvf_in, dissolved_L_per_L)

    # Auswahl
    best = choose_best_mph_pump_normbased(MPH_PUMPS, Q_req, dp_req, gvf_free, dp_margin=0.0)

    # -----------------------------------------------------
    # Plot 1 + Plot 2 (nebeneinander) + Plot 3 Overlay darunter
    # -----------------------------------------------------
    fig_top, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Löslichkeit vs Druck (cm³N/L) => "diagonal" wie Vorlage
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure_cm3N(gas_medium, T, p_max=14, y_gas=y_gas)
        ax1.plot(pressures, sol, "--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C (cm³N/L)")

    ax1.scatter([p_suction], [dissolved_cm3N_L], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt", zorder=5)

    ax1.set_xlabel("p_abs [bar]")
    ax1.set_ylabel("Löslichkeit [cm³N/L]")
    ax1.set_title("Gaslöslichkeit (Henry → Normalvolumen, diagonal)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Plot 2: ∆p-Q Kennlinien (Auswahlpumpe)
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
                    edgecolors="black", linewidths=2, label="Betriebspunkt (∆p_req)", zorder=5)

        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title(f"Mehrphasen-Kennlinien (∆p): {pump['id']}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
        ax2.set_ylim(0, pump["p_max_bar"] * 1.1)
    else:
        ax2.text(0.5, 0.5, "❌ Keine geeignete Pumpe gefunden",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title("Mehrphasen-Kennlinien")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_top, clear_figure=True)

    # -----------------------------------------------------
    # Plot 3: Overlay wie Vorlage
    # - x: Druck [bar abs]
    # - y: Löslichkeit [cm³N/L] (diagonal, gestrichelt)
    #      Pumpenkennlinien als Gasvolumenstrom [L/min] über Druck (solid)
    # -----------------------------------------------------
    st.markdown("### 📉 Overlay: Pumpenkennlinien (Gasvolumenstrom) über Löslichkeit (diagonal)")

    fig3, ax3 = plt.subplots(figsize=(14, 6))

    # Löslichkeit (diagonal) – gleiche Linien wie Plot 1 (aber in einem gemeinsamen Overlay)
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure_cm3N(gas_medium, T, p_max=14, y_gas=y_gas)
        ax3.plot(pressures, sol, "--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    ax3.scatter([p_suction], [dissolved_cm3N_L], s=160, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt (Löslichkeit)", zorder=6)

    # Pumpenkennlinien im Overlay: p_abs(Q)=p_s + ∆p(Q), y=Q_gas(Q)=GVF*Q_total
    # -> "Pumpenkennlinien über den Löslichkeiten" wie in deiner Vorlage
    if best:
        pump = best["pump"]
        curves = pump["curves_p_vs_Q"]
        gvf_plot_set = [10, 15, 20]  # orientiert an deiner Vorlage

        for gvf_pct in gvf_plot_set:
            gvf_key = choose_gvf_curve_key_worstcase(curves, gvf_pct)
            curve = curves[gvf_key]

            Q_grid = np.linspace(min(curve["Q"]), max(curve["Q"]), 120)  # m³/h
            dp_vals = np.array([interp_clamped(float(q), curve["Q"], curve["p"]) for q in Q_grid])  # bar
            p_abs_vals = p_suction + dp_vals  # bar abs

            Q_total_lmin = m3h_to_lmin(Q_grid)  # L/min
            Q_gas_lmin = Q_total_lmin * (gvf_pct / 100.0)

            ax3.plot(p_abs_vals, Q_gas_lmin, linewidth=2.5, alpha=0.95,
                     label=f"{gvf_pct}% GVF (Pumpenkennlinie als Q_gas)")

    ax3.set_xlabel("Druck [bar abs]")
    ax3.set_ylabel("Löslichkeit [cm³N/L] / Gasvolumenstrom [L/min]")
    ax3.set_title("Overlay wie Vorlage: Pumpenkennlinien über Löslichkeit")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 180)
    ax3.legend(fontsize=9, ncol=2)

    st.pyplot(fig3, clear_figure=True)

    # -----------------------------------------------------
    # Ergebnisse
    # -----------------------------------------------------
    st.divider()
    st.markdown("### ✅ Ergebnisse (normlogisch)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("∆p_req", f"{dp_req:.2f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
    c3.metric("GVF_in", f"{gvf_in:.0f} %")
    c4.metric("Löslichkeit (Saugseite)", f"{dissolved_L_per_L:.4f} L/L", f"{dissolved_cm3N_L:.1f} cm³N/L")
    c5.metric("GVF_free (für Kennlinie)", f"{gvf_free:.1f} %")

    if gvf_free > 0.0:
        st.warning("⚠️ Freies Gas vorhanden (GVF_free > 0). Kennlinienwahl erfolgt Worst Case auf nächsthöhere GVF-Kurve.")
    else:
        st.info("ℹ️ Kein freies Gas aus dieser Abschätzung (alles im Löslichkeitslimit).")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        tol = 1e-6
        ok = (best["dp_available"] + tol) >= dp_req

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("∆p verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve: {best['dp_reserve']:.2f} bar")
        c2.metric("Leistung", f"{best['P_required']:.2f} kW")
        c3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        c4.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

        if not ok:
            st.error("❌ ∆p_req wird nicht erreicht (prüfe Kennlinie / GVF / Drehzahlbereich).")
    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")

    st.divider()
    st.markdown("### 📋 Verfügbare Mehrphasenpumpen")
    cols = st.columns(len(MPH_PUMPS))
    for i, pmp in enumerate(MPH_PUMPS):
        with cols[i]:
            selected = bool(best) and best["pump"]["id"] == pmp["id"]
            st.success(f"✅ **{pmp['id']}**" if selected else f"**{pmp['id']}**")
            st.caption(f"Typ: {pmp['type']}")
            st.caption(f"Q_max: {pmp['Q_max_m3h']} m³/h")
            st.caption(f"∆p_max: {pmp['p_max_bar']} bar")
            st.caption(f"GVF_max: {pmp['GVF_max']*100:.0f}%")

    with st.expander("📘 Rechenweg (ausführlich)", expanded=False):
        st.markdown(f"""
## Schritt 1: ∆p-Anforderung (ohne Zusatzverluste)
Da du **Δp_loss entfernt** haben wolltest, gilt:

\\[
\\Delta p_{{req}} = p_d - p_s
\\]

- p_s = {p_suction:.2f} bar abs  
- p_d = {p_discharge:.2f} bar abs  

⇒ **∆p_req = {dp_req:.3f} bar**

---

## Schritt 2: Löslichkeit am Saugpunkt (Henry + ideales Gas)
Wir bestimmen das maximal lösliche Gasvolumen bezogen auf 1 Liter Flüssigkeit am Saugpunkt.

1) Henry:
\\[
C = \\frac{{p_{{partial}}}}{{H(T)}}
\\]

2) Ideales Gas (Betriebsbedingungen):
\\[
V_{{molar}} = \\frac{{RT}}{{p}}
\\]

3) Gasvolumen pro Liter Flüssigkeit:
\\[
\\frac{{V_{{gas}}}}{{V_{{liq}}}} = C \\cdot V_{{molar}}
\\]

Ergebnis am Saugpunkt (p_s, T):
- V_gas,gelöst / V_liq = **{dissolved_L_per_L:.6f} L/L** (bei Betriebsbedingungen)

Zusätzlich für die **diagonale Darstellung** in der Grafik: Umrechnung auf Normalvolumen:
\\[
V_N = V_{{oper}} \\cdot \\frac{{p_{{oper}}}}{{p_N}} \\cdot \\frac{{T_N}}{{T_{{oper}}}}
\\]

⇒ **Löslichkeit (cm³N/L) am Saugpunkt = {dissolved_cm3N_L:.2f}**

---

## Schritt 3: Freier GVF (Worst Case für Kennlinienwahl)
Gegeben GVF_in am Saugpunkt. Wir setzen 1 L Flüssigkeit:

- Gesamt-Gasvolumen (aus GVF):
\\[
V_{{gas,total}} = \\frac{{GVF}}{{1-GVF}} \\cdot V_{{liq}}
\\]

- Gelöst max:
\\[
V_{{gas,diss,max}} = (V_{{gas}}/V_{{liq}}) \\cdot V_{{liq}}
\\]

- Frei:
\\[
V_{{gas,free}} = \\max(0, V_{{gas,total}} - V_{{gas,diss,max}})
\\]

- Freier GVF:
\\[
GVF_{{free}} = \\frac{{V_{{gas,free}}}}{{V_{{gas,free}} + V_{{liq}}}}
\\]

⇒ **GVF_free = {gvf_free:.2f}%**

---

## Schritt 4: Kennlinienwahl und Drehzahl
- Kennlinienwahl **Worst Case**: nächsthöhere GVF-Kurve ≥ GVF_free  
- Vergleich gegen ∆p(Q)  
- Optional Drehzahl-Anpassung (Affinität):
\\[
Q \\sim n, \\quad \\Delta p \\sim n^2, \\quad P \\sim n^3
\\]

Die Bisektion sucht n/n0, so dass ∆p(Q_req) = ∆p_req erfüllt wird (sofern möglich).
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

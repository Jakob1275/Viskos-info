# app.py
# Pumpenauslegungstool (Einphasen + Mehrphasen + ATEX)
# - Einphasen: Viskositätskorrektur (HI-ähnliche Näherung)
# - Mehrphasen: Henry-Löslichkeit + freier GVF (am Saugpunkt) + ∆p-Kennlinien + Pumpenauswahl inkl. Drehzahl (Affinität)
# - Mehrphasen: 3. Grafik "Overlay" (Sättigungskennlinien diagonal + GVF-Kennlinien wie im Beispielbild)

import math
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# Konstanten
# =========================================================
G = 9.80665  # m/s²

# ideales Gas:
R_BAR_L = 0.08314462618  # bar·L/(mol·K)
P_STP_BAR = 1.01325
T_STP_K = 273.15
V_MOLAR_STP_L_PER_MOL = 22.414  # L/mol (STP, 0°C, 1 atm)

# =========================================================
# Daten: Einphasen-Pumpen (Beispiele)
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
# Daten: Mehrphasen-Pumpen (Beispiele) - Kennlinien als ∆p [bar]
# keys: GVF in Prozent (0,5,10,15,...)
# =========================================================
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

# =========================================================
# ATEX-Datenbank (Beispiel)
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
# Henry-Konstanten (vereinfachte Parameter)
# H(T) = A * exp(B*(1/T - 1/T0))
# H in bar·L/mol
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
    """Lineare Interpolation (xs aufsteigend)"""
    if len(xs) < 2:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================================================
# Viskositätskorrektur (HI-ähnliche Näherung, pragmatisch)
# =========================================================
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
        cand = {
            "id": p["id"], "pump": p, "in_range": in_range, "Q_eval": Q_eval,
            "H_at": H_at, "eta_at": eta_at, "errH": abs(H_at - H_water), "score": score
        }

        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    return best

# =========================================================
# Henry / Gaslöslichkeit
# =========================================================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1.0 / T_K - 1.0 / T0_K))

def dissolved_mol_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry: C [mol/L] = p_partial / H(T)
    """
    p = max(p_bar_abs, 1e-9)
    y = clamp(y_gas, 0.0, 1.0)
    H = max(henry_constant(gas, T_celsius), 1e-12)
    return (y * p) / H

def solubility_Lstp_per_Lliq(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    GELÖSTES GAS als STP-Volumen pro Liter Flüssigkeit:
    V_STP/L_liq = C * V_molar_STP  => proportional zu p  => Diagonale Linien im Plot
    """
    C = dissolved_mol_per_L(gas, p_bar_abs, T_celsius, y_gas=y_gas)
    return C * V_MOLAR_STP_L_PER_MOL

def solubility_cm3stp_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    return 1000.0 * solubility_Lstp_per_Lliq(gas, p_bar_abs, T_celsius, y_gas=y_gas)

def solubility_curve_vs_pressure(gas, T_celsius, p_max=30, y_gas=1.0):
    ps = linspace(0.0, p_max, 120)
    sol = [solubility_cm3stp_per_L(gas, max(p, 1e-6), T_celsius, y_gas=y_gas) for p in ps]
    return ps, sol

# --- GVF & Gasstrom
def gvf_to_Vgas_per_Vliq(gvf_pct):
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    return gvf / (1.0 - gvf)  # Vgas/Vliq

def Vgas_per_Vliq_to_gvf_pct(Vgas_per_Vliq):
    V = max(Vgas_per_Vliq, 0.0)
    return 100.0 * (V / (1.0 + V))

def free_gvf_at_suction_actual(gas, T_celsius, gvf_total_pct, y_gas, p_suction_bar_abs):
    """
    Freies Gas am Saugpunkt in "tatsächlicher" Volumenbasis:
    - Gesamtgasvolumen aus GVF_total (tatsächlich) -> Vgas_total/Vliq
    - maximal lösbares Gas in tatsächlichem Volumen am Saugpunkt:
        V_diss_actual/L = C * V_molar(p_s,T) = (y*p/H) * (R*T/p) = y*R*T/H  (p kürzt sich)
    => physikalisch: tatsächlicher Sättigungs-GVF hängt (idealisiert) stark von T und Gasart ab, kaum von p
    """
    T_K = T_celsius + 273.15
    H = max(henry_constant(gas, T_celsius), 1e-12)
    y = clamp(y_gas, 0.0, 1.0)

    Vgas_total_per_Vliq = gvf_to_Vgas_per_Vliq(gvf_total_pct)
    Vgas_diss_actual_per_Vliq = y * (R_BAR_L * T_K) / H  # L_gas(actual at suction)/L_liq

    Vgas_free = max(0.0, Vgas_total_per_Vliq - Vgas_diss_actual_per_Vliq)
    return Vgas_per_Vliq_to_gvf_pct(Vgas_free), Vgas_diss_actual_per_Vliq

def gas_flow_from_gvf(Q_liq_Lmin, gvf_pct):
    """Gasvolumenstrom (tatsächlich) aus GVF und Liquid-Flow"""
    Vgas_per_Vliq = gvf_to_Vgas_per_Vliq(gvf_pct)
    return Vgas_per_Vliq * max(Q_liq_Lmin, 0.0)

def gasflow_actual_to_stp(Q_gas_Lmin_actual, p_bar_abs, T_celsius):
    """
    Ideal Gas: V_stp = V * (p/p_stp) * (T_stp/T)
    """
    T_K = T_celsius + 273.15
    return Q_gas_Lmin_actual * (max(p_bar_abs, 1e-9) / P_STP_BAR) * (T_STP_K / max(T_K, 1e-9))

# =========================================================
# Mehrphasen: Affinitätsgesetze (korrekt über Q_base)
# =========================================================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    Q_base = Q_req / max(n_ratio, 1e-9)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2)

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_target,
                               n_min=0.5, n_max=1.1, tol=1e-3, iters=60):
    """
    Finde n_ratio so, dass dp_scaled(Q_req,n_ratio) = dp_target.
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
    """
    Worst-Case: nächsthöhere GVF-Kurve (ceiling) ≥ gvf_free.
    """
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def choose_best_mph_pump_normbased(pumps, Q_req, dp_req, gvf_free_req_pct,
                                  dp_margin=0.10, allow_speed=True,
                                  n_min=0.5, n_max=1.1):
    """
    Auswahl-Logik:
    - dp_req ist geforderte Druckerhöhung ∆p (aus p_d - p_s + Verluste)
    - dp_target = dp_req*(1+dp_margin) als Auslegungsreserve
    - Kennlinien werden worst-case über gvf_free_req_pct gewählt
    - optional: Drehzahl-Anpassung (Affinität) für dp_target
    """
    dp_target = dp_req * (1.0 + dp_margin)
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
        in_curve = (Qmin <= Q_req <= Qmax)

        candidates = []

        # A) Nenndrehzahl
        if in_curve:
            dp_avail_nom = interp_clamped(Q_req, curve["Q"], curve["dp"])
            if dp_avail_nom >= dp_target:
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail_nom - dp_target) + abs(gvf_key - gvf_free_req_pct) * 0.25
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

        # B) Drehzahl-Anpassung
        if allow_speed:
            n_ratio = find_speed_ratio_bisection(curve["Q"], curve["dp"], Q_req, dp_target,
                                                 n_min=n_min, n_max=n_max)
            if n_ratio is not None:
                Q_base = Q_req / n_ratio
                if Qmin <= Q_base <= Qmax:
                    dp_scaled = _dp_scaled_at_Q(curve["Q"], curve["dp"], Q_req, n_ratio)  # ~ dp_target
                    P_base = interp_clamped(Q_base, power_curve["Q"], power_curve["P"])
                    P_scaled = P_base * (n_ratio ** 3)
                    score = abs(1.0 - n_ratio) * 6.0 + abs(gvf_key - gvf_free_req_pct) * 0.25
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

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Pumpenauslegung", layout="wide")
st.title("Pumpenauslegungstool")

if "page" not in st.session_state:
    st.session_state.page = "pump"

with st.sidebar:
    st.header("📍 Navigation")
    c1, c2, c3 = st.columns(3)
    if c1.button("Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if c2.button("Mehrphasen", use_container_width=True):
        st.session_state.page = "mph"
    if c3.button("ATEX", use_container_width=True):
        st.session_state.page = "atex"
    st.info(f"**Aktiv:** {st.session_state.page}")

# =========================================================
# PAGE 1: Einphasen
# =========================================================
if st.session_state.page == "pump":
    st.subheader("🔄 Einphasen: Pumpenauswahl mit Viskositätskorrektur")

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

    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    st.info(
        f"{'✅' if B < 1.0 else '⚠️'} B = {B:.2f} "
        f"{'< 1.0 → geringe Viskositätseffekte' if B < 1.0 else '≥ 1.0 → Viskositätskorrektur aktiv'}"
    )

    st.markdown("### 📊 Umrechnung viskos → Wasser")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Q_Wasser", f"{Q_water:.2f} m³/h")
    a2.metric("H_Wasser", f"{H_water:.2f} m", f"+{H_water - H_vis_req:.1f} m")
    a3.metric("B-Zahl", f"{B:.2f}")
    a4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")

    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]
    eta_water = best["eta_at"]
    eta_vis = max(1e-6, eta_water * Ceta)

    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / eta_vis) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    st.markdown("### ✅ **AUSLEGUNGSERGEBNIS (Einphasen)**")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")
    b2.metric("H (viskos)", f"{H_vis_req:.2f} m")
    b3.metric("η (viskos)", f"{eta_vis:.3f}")
    b4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW")
    b5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(
            f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
            f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
        )

    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])

    st.divider()
    st.markdown("### 📈 Kennlinien")
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
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, linewidths=3,
                    label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    with st.expander("📘 Rechenweg (Kurz)", expanded=False):
        st.markdown(f"""
**Gegeben (viskos):**
- Qᵥ = {Q_vis_req:.2f} m³/h
- Hᵥ = {H_vis_req:.2f} m
- ν = {nu:.2f} cSt
- ρ = {rho:.1f} kg/m³

**Korrektur:**
- B = {B:.2f}
- CH = {CH:.3f}
- Cη = {Ceta:.3f}

**Umrechnung auf Wasserkennlinie:**
- Q_w = Qᵥ = {Q_water:.2f} m³/h
- H_w = Hᵥ / CH = {H_water:.2f} m

**Leistung:**
- P_hyd = ρ g Q H = {P_hyd_W:.0f} W
- P_Welle ≈ {P_vis_kW:.2f} kW
- IEC Motor (+{reserve_pct}%): {P_motor_kW:.2f} kW
""")

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit (p,T) + freier GVF + ∆p-Kennlinien + Auswahl")

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Medium / Gas")
        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)
        y_gas = st.slider("Gasanteil (Partialdruckfaktor) y_gas [-]", 0.0, 1.0, 1.0, 0.05)

        st.divider()
        st.subheader("Betriebspunkt (Hydraulik, normlogisch)")
        Q_req = st.number_input("Flüssigkeits-Volumenstrom Q_liq [m³/h]", 0.1, 150.0, 15.0, 1.0)

        # Normlogik: ∆p-Anforderung aus p_d - p_s + Verlusten (typisch in Auslegungen)
        p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 100.0, 2.0, 0.1)
        p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 200.0, 7.0, 0.1)
        dp_losses = st.number_input("Zusätzliche Verluste Δp_loss [bar]", 0.0, 50.0, 0.5, 0.1)
        dp_req = max(0.0, (p_discharge - p_suction) + dp_losses)

        st.divider()
        st.subheader("Gasanteile (Eingabe)")
        gvf_in = st.slider("Gesamt-GVF_in [%] (am Saugpunkt)", 0, 40, 10, 1)

        st.divider()
        st.subheader("Reserve / Drehzahl")
        dp_margin = st.slider("∆p-Reserve [%]", 0, 30, 10, 1) / 100.0
        allow_speed = st.checkbox("Drehzahl-Anpassung zulassen", value=True)
        n_min = st.slider("n_min [%]", 30, 80, 50, 1) / 100.0
        n_max = st.slider("n_max [%]", 100, 130, 110, 1) / 100.0

        st.divider()
        st.subheader("Plots")
        show_temp_band = st.checkbox("T-10/T/T+10 zeigen", value=True)

    Q_req_lmin = m3h_to_lmin(Q_req)

    # --- freier GVF am Saugpunkt (tatsächlich) + gelöstes Gas (tatsächliches Volumenverhältnis)
    gvf_free, Vgas_diss_actual_per_Vliq = free_gvf_at_suction_actual(
        gas_medium, temperature, gvf_in, y_gas, p_suction
    )

    # --- Pumpenauswahl (auf gvf_free basierend, ∆p mit Reserve, inkl. Drehzahl)
    best = choose_best_mph_pump_normbased(
        MPH_PUMPS, Q_req, dp_req, gvf_free,
        dp_margin=dp_margin,
        allow_speed=allow_speed,
        n_min=n_min, n_max=n_max
    )

    # =====================================================
    # Grafik 1+2: (1) Löslichkeit vs p_abs (DIAGONAL, STP) + (2) ∆p-Q Kennlinien (inkl. Drehzahl-Kurve)
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Löslichkeit vs p_abs (STP-Volumen pro L Flüssigkeit) => diagonal
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "black"]
    pmax_plot = max(10.0, p_discharge * 1.2)

    for i, T in enumerate(temp_variants):
        ps, sol = solubility_curve_vs_pressure(gas_medium, T, p_max=pmax_plot, y_gas=y_gas)
        ax1.plot(ps, sol, "--", linewidth=2, color=color_cycle[i % len(color_cycle)],
                 label=f"{gas_medium} bei {T:.0f}°C (y={y_gas:.2f})")

    # Markierungen: Saugpunkt & Druckseite
    sol_suction = solubility_cm3stp_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    sol_discharge = solubility_cm3stp_per_L(gas_medium, p_discharge, temperature, y_gas=y_gas)

    ax1.scatter([p_suction], [sol_suction], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt", zorder=5)
    ax1.scatter([p_discharge], [sol_discharge], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Druckseite", zorder=5)

    ax1.set_xlabel("p_abs [bar]")
    ax1.set_ylabel("Löslichkeit [cm³/L] (auf STP bezogen)")
    ax1.set_title("Gaslöslichkeit (Henry, STP-Volumen) – diagonal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # --- Plot 2: ∆p-Q Kennlinien (Mehrphasen)
    Q_req_lmin = m3h_to_lmin(Q_req)
    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]
        gvf_colors = {0: "black", 5: "tab:green", 10: "tab:blue", 15: "tab:red", 20: "tab:purple"}

        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.45
            ax2.plot(Q_lmin, curve["dp"], "o-", linewidth=lw, alpha=alpha,
                     color=gvf_colors.get(gvf_key, "gray"),
                     label=f"{pump['id']} ({gvf_key}% GVF)")

        # Betriebspunkt: dp_req + dp_target
        dp_target = best["dp_target"]
        ax2.scatter([Q_req_lmin], [dp_req], s=180, marker="o",
                    edgecolors="black", linewidths=2, label="Betriebspunkt (∆p_req)", zorder=6)
        ax2.scatter([Q_req_lmin], [dp_target], s=180, marker="o",
                    edgecolors="black", linewidths=2, label="Ziel (∆p_req + Reserve)", zorder=6)

        # Drehzahl-skalierte Kennlinie (nur ausgewählte GVF-Kurve)
        if abs(best["n_ratio"] - 1.0) > 1e-3:
            n = best["n_ratio"]
            gvf_key = best["gvf_curve"]
            base_curve = curves[gvf_key]
            Q_scaled = [m3h_to_lmin(q * n) for q in base_curve["Q"]]
            dp_scaled = [dp * (n ** 2) for dp in base_curve["dp"]]
            ax2.plot(Q_scaled, dp_scaled, "--", linewidth=3.0,
                     label=f"skalierte Kurve (n={n*100:.1f}%)")

        ax2.set_xlabel("Q_liq [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title(f"Mehrphasen-Kennlinien (∆p): {pump['id']}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
        ax2.set_ylim(0, pump["dp_max_bar"] * 1.15)
    else:
        ax2.text(0.5, 0.5, "❌ Keine geeignete Pumpe gefunden",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q_liq [L/min]")
        ax2.set_ylabel("∆p [bar]")
        ax2.set_title("Mehrphasen-Kennlinien")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # =====================================================
    # Grafik 3: Overlay wie im Beispielbild
    # - x: Druck [bar] (hier: p_abs auf der Druckseite ~ p_s + ∆p)
    # - y: Gasvolumenstrom [L/min] (auf STP bezogen)
    # - diagonal: Sättigungs-/Löslichkeitslinien (STP) -> linear mit p
    # - fallend: GVF-Pumpenkennlinien (aus ∆p-Q + GVF -> Q_gas) -> wie im Bild
    # =====================================================
    st.divider()
    st.markdown("### 📉 Overlay: Sättigungskennlinien (diagonal) + GVF-Kennlinien (überlagert)")

    fig3, ax3 = plt.subplots(figsize=(13, 6))

    # Druckachse (abs): sinnvoller Bereich
    p_abs_min = 0.0
    p_abs_max = max(14.0, p_discharge * 1.2)

    # Diagonale Sättigungslinien: gelöstes Gas als Gas-Volumenstrom (STP) bei Q_req
    # y = solubility(L_STP/L_liq) * Q_liq(L/min)
    for i, T in enumerate(temp_variants):
        ps = linspace(0.0, p_abs_max, 120)
        y_sol_flow = []
        for p_abs in ps:
            sol_Lstp_per_L = solubility_Lstp_per_Lliq(gas_medium, max(p_abs, 1e-6), T, y_gas=y_gas)
            y_sol_flow.append(sol_Lstp_per_L * Q_req_lmin)
        ax3.plot(ps, y_sol_flow, linestyle="--", linewidth=2,
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    # GVF-Kennlinien der ausgewählten Pumpe: x=p_abs_discharge, y=Q_gas_STP
    if best:
        pump = best["pump"]
        curves = pump["curves_dp_vs_Q"]

        # Umrechnung Gasstrom actual -> STP am Saugpunkt (konstant für alle Kurven)
        conv_to_stp = (p_suction / P_STP_BAR) * (T_STP_K / (temperature + 273.15))

        # Im Beispielbild sind 10/15/20% häufig – wir plotten alle vorhandenen >0
        for gvf_key in sorted([k for k in curves.keys() if k > 0]):
            curve = curves[gvf_key]
            xs = []
            ys = []
            for Q_m3h, dp_bar in zip(curve["Q"], curve["dp"]):
                Q_liq_lmin = m3h_to_lmin(Q_m3h)
                # Gasstrom aus GVF (tatsächlich am Saugpunkt)
                Q_gas_actual = gas_flow_from_gvf(Q_liq_lmin, gvf_key)
                Q_gas_stp = Q_gas_actual * conv_to_stp
                # x-Achse: absolute Druckseite ~ p_s + ∆p
                p_abs_dis = p_suction + dp_bar
                xs.append(p_abs_dis)
                ys.append(Q_gas_stp)
            ax3.plot(xs, ys, linewidth=2.5, label=f"{gvf_key}% {gas_medium}")

        # Betriebspunkt (gesamt Gasstrom aus gvf_in) auf STP bezogen
        Q_gas_in_actual = gas_flow_from_gvf(Q_req_lmin, gvf_in)
        Q_gas_in_stp = Q_gas_in_actual * conv_to_stp
        ax3.scatter([p_discharge], [Q_gas_in_stp], s=140, edgecolors="black", linewidths=2,
                    label="Betriebspunkt (Gasstrom)", zorder=6)

    ax3.set_xlabel("Druck [bar abs]")
    ax3.set_ylabel("gelöstes Gas (L_STP/min) / Gasvolumenstrom (L_STP/min)")
    ax3.set_title("Überlagerung: Löslichkeit (diagonal) + GVF-Kennlinien (wie im Beispiel)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(p_abs_min, p_abs_max)
    # y-Limit automatisch, aber mit etwas Puffer
    ax3.set_ylim(0, max(160.0, Q_req_lmin * 0.8))
    ax3.legend(ncol=2, fontsize=9)

    st.pyplot(fig3, clear_figure=True)

    # =====================================================
    # Ergebnisse
    # =====================================================
    st.divider()
    st.markdown("### ✅ Ergebnisse (Mehrphase)")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Q_liq", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    r2.metric("p_s / p_d", f"{p_suction:.2f} / {p_discharge:.2f} bar abs")
    r3.metric("∆p_req", f"{dp_req:.2f} bar", f"+loss={dp_losses:.2f}")
    r4.metric("GVF_in", f"{gvf_in:.0f} %")
    r5.metric("GVF_free (tats.)", f"{gvf_free:.1f} %")
    r6.metric("gelöst max (tats.)", f"{Vgas_diss_actual_per_Vliq:.3f} L/L")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("∆p verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve ggü. ∆p_req: {best['dp_reserve']:.2f} bar")
        e2.metric("Ziel ∆p", f"{best['dp_target']:.2f} bar", f"Reserve: {dp_margin*100:.0f}%")
        e3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        e4.metric("Leistung", f"{best['P_required']:.2f} kW")

        # Plausi-Hinweis
        if best["dp_available"] + 1e-6 < best["dp_target"]:
            st.warning("⚠️ Auswahl liegt knapp unter dem Ziel ∆p (Toleranz/Interpolation). Prüfe Herstellerdaten.")
    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")
        st.markdown("""
**Typische Gründe:**
- ∆p_req zu hoch für alle Pumpen/Kennlinien
- Q_liq zu hoch für Pumpengrößenbereich
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
            st.caption(f"GVF_max: {pmp['GVF_max']*100:.0f}%")

    with st.expander("📘 Rechenweg (Kurz)", expanded=False):
        st.markdown(f"""
**1) ∆p-Anforderung (Auslegung)**
- ∆p_req = (p_d − p_s) + ∆p_loss  
= ({p_discharge:.2f} − {p_suction:.2f}) + {dp_losses:.2f}  
= **{dp_req:.2f} bar**
- Ziel: ∆p_target = ∆p_req × (1 + Reserve) = **{(dp_req*(1+dp_margin)):.2f} bar**

**2) Löslichkeit (Plot diagonal)**
- Gezeigt als **STP-Volumen** (cm³/L bzw. L_STP/L) ⇒ proportional zu p_abs ⇒ diagonal

**3) Freier GVF (Auswahlbasis)**
- GVF_in ist am Saugpunkt (tatsächliche Volumenbasis)
- maximal gelöst (tats.) ≈ y·R·T/H
- GVF_free wird daraus abgeschätzt (idealisiert)

**4) Pumpenauswahl**
- Worst-Case: nächsthöhere GVF-Kurve ≥ GVF_free
- Vergleich gegen ∆p_target
- Optional: Drehzahl über Affinität (Q~n, ∆p~n², P~n³)
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


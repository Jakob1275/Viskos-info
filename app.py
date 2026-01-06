# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Konstanten
# =========================================================
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K) (ideales Gas, R)

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
# Mehrphasen-Pumpen (Kennlinien als ∆p in bar) - Beispiel
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
        "GVF_max": 0.4,
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
        "GVF_max": 0.4,
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
    "CO2":  {"A": 29.4,   "B": 2400},
    "O2":   {"A": 1500.0, "B": 1500},
    "N2":   {"A": 1650.0, "B": 1300},
    "CH4":  {"A": 1400.0, "B": 1600},
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
# Viskositätskorrektur (HI-ähnliche Näherung)
# =========================================================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """
    B-Zahl als (vereinfachter) Indikator für Viskositätseinfluss
    (in Anlehnung an Hydraulic Institute / pragmatische Näherung)
    """
    Q = max(Q_m3h, 1e-6)
    H = max(H_m, 1e-6)
    nu = max(nu_cSt, 1e-6)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    """
    Liefert:
      CH   ~ Korrekturfaktor Förderhöhe (H_vis = H_w * CH)
      Ceta ~ Korrekturfaktor Wirkungsgrad (eta_vis = eta_w * Ceta)
    """
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
    Wir wählen im Tool:
      - Q bleibt identisch (Q_w = Q_vis)
      - H_w wird hochgerechnet: H_w = H_vis / CH
    """
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / max(CH, 1e-12)}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """
    Aus Wasserkennlinie (Q_w,H_w,eta_w) -> viskose Abschätzung:
      H_vis = H_w * CH
      eta_vis = eta_w * Ceta
    """
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
    """
    Wählt Pumpe, deren Wasserkennlinie bei Q am nächsten zu H_water liegt.
    (Optional außerhalb Q-Bereich, dann mit Penalty.)
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

# =========================================================
# Henry / Gaslöslichkeit
# =========================================================
def henry_constant(gas, T_celsius):
    """
    H(T) in bar·L/mol (vereinfachte Exponentialform)
    """
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def dissolved_moles_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry: C [mol/L] = p_partial / H(T)
    """
    p = max(p_bar_abs, 1e-9)
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    H = max(henry_constant(gas, T_celsius), 1e-12)
    return p_partial / H  # mol/L

def dissolved_gas_cm3_per_L_ref(gas, p_bar_abs, T_celsius, p_ref_bar_abs, y_gas=1.0):
    """
    "Diagonal"-Darstellung wie in deiner Vorlage:
      - Henry liefert C ~ p
      - Wir rechnen dieses gelöste Gas auf ein Referenzvolumen um:
            V_ref = C * R*T / p_ref   [L/L]
        => cm³/L = 1000 * V_ref

    Wenn p_ref konstant ist (hier: p_ref = p_suction), entsteht eine DIAGONALE Linie (linear in p).
    """
    p_ref = max(p_ref_bar_abs, 1e-9)
    C = dissolved_moles_per_L(gas, p_bar_abs, T_celsius, y_gas=y_gas)  # mol/L
    T_K = T_celsius + 273.15
    V_ref_L_per_L = C * R_BAR_L * T_K / p_ref
    return 1000.0 * V_ref_L_per_L

def solubility_curve_vs_pressure_cm3_ref(gas, T_celsius, p_max, p_ref_bar_abs, y_gas=1.0):
    pressures = linspace(0.0, p_max, 160)
    sol = [
        dissolved_gas_cm3_per_L_ref(
            gas, max(1e-6, p), T_celsius, p_ref_bar_abs=p_ref_bar_abs, y_gas=y_gas
        )
        for p in pressures
    ]
    return pressures, sol

def total_gas_moles_from_gvf_at_suction(gvf_total_pct, p_suction_bar_abs, T_celsius):
    """
    Wir interpretieren GVF_in als Volumenanteil am Saugpunkt.
    Setze V_liq = 1 L
    Dann: V_gas_s = gvf/(1-gvf) * V_liq   [L]
    Gasmenge am Saugpunkt (ideal): n_total = p_s * V_gas_s / (R*T)  [mol/L_liq]
    """
    gvf = clamp(gvf_total_pct / 100.0, 0.0, 0.999999)
    V_liq = 1.0  # L
    V_gas_s = gvf / (1.0 - gvf) * V_liq  # L gas per L liquid at suction
    T_K = T_celsius + 273.15
    p_s = max(p_suction_bar_abs, 1e-9)
    n_total = p_s * V_gas_s / (R_BAR_L * T_K)  # mol/L_liq
    return n_total, V_gas_s

def free_gas_volume_cm3_per_L_at_pressure(
    gvf_total_pct,
    gas,
    p_suction_bar_abs,
    p_bar_abs,
    T_celsius,
    y_gas=1.0
):
    """
    Freies Gas als Funktion des Drucks p:
      1) total moles aus GVF_in am Saugpunkt: n_total
      2) gelöst (Kapazität) bei p: n_diss(p) = p_partial/H(T)
      3) n_free = max(0, n_total - n_diss)
      4) V_free(p) = n_free * R*T / p   [L/L_liq]
      5) cm³/L = 1000 * V_free(p)
    """
    p = max(p_bar_abs, 1e-9)
    T_K = T_celsius + 273.15
    n_total, _Vgas_s = total_gas_moles_from_gvf_at_suction(gvf_total_pct, p_suction_bar_abs, T_celsius)
    n_diss = dissolved_moles_per_L(gas, p, T_celsius, y_gas=y_gas)
    n_free = max(0.0, n_total - n_diss)
    V_free_L_per_L = n_free * R_BAR_L * T_K / p
    return 1000.0 * V_free_L_per_L, n_total, n_diss, n_free

def free_gvf_at_suction(gvf_total_pct, dissolved_cm3N_per_L, p_suction_bar_abs, T_celsius):
    """
    Beibehaltung der ursprünglichen "Normlogik" für GVF_free:
    - Wir rechnen total-Gasmenge aus GVF_in am Saugpunkt (als moles)
    - und vergleichen gegen gelöste Kapazität am Saugpunkt (moles)
    Hier nutzen wir dissolved_cm3N_per_L, um moles zu rekonstruieren.

    dissolved_cm3N_per_L: Gasvolumen bei Normbedingungen pro L Flüssigkeit.
    -> Umrechnung nach mol: n = V / V_molar(N)
    (V_molar(N) ~ 22.414 L/mol bei 0°C, 1.01325 bar; hier nehmen wir 22.414 vereinfachend)
    """
    V_molar_N = 22.414  # L/mol (vereinfachend)
    V_diss_N_L_per_L = (dissolved_cm3N_per_L / 1000.0)  # L/L
    n_diss_s = V_diss_N_L_per_L / V_molar_N  # mol/L

    n_total, Vgas_s = total_gas_moles_from_gvf_at_suction(gvf_total_pct, p_suction_bar_abs, T_celsius)
    n_free_s = max(0.0, n_total - n_diss_s)

    # Freier GVF am Saugpunkt (volumetrisch) aus n_free_s:
    T_K = T_celsius + 273.15
    p_s = max(p_suction_bar_abs, 1e-9)
    V_free_s_L_per_L = n_free_s * R_BAR_L * T_K / p_s  # L/L
    V_liq = 1.0
    gvf_free = V_free_s_L_per_L / (V_free_s_L_per_L + V_liq)
    return 100.0 * gvf_free, {
        "n_total": n_total,
        "n_diss_s": n_diss_s,
        "n_free_s": n_free_s,
        "Vgas_total_s_L_per_L": Vgas_s,
        "Vgas_free_s_L_per_L": V_free_s_L_per_L
    }

def dissolved_gas_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Für die GVF_free-Rechnung (Normlogik):
      n = p/H  [mol/L]
      V_N = n * V_molar_N  [L/L]
    """
    V_molar_N = 22.414  # L/mol
    n = dissolved_moles_per_L(gas, p_bar_abs, T_celsius, y_gas=y_gas)
    return 1000.0 * n * V_molar_N  # cm³N/L

# =========================================================
# Mehrphasen: Affinitätsgesetze (Drehzahl)
# =========================================================
def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    """
    Q ~ n, dp ~ n²
    dp(Q_req, n) = dp_base(Q_req/n) * n²
    """
    Q_base = Q_req / max(n_ratio, 1e-9)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2)

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_req,
                               n_min=0.5, n_max=1.35, tol=1e-3, iters=70):
    """
    Sucht n_ratio so, dass dp_scaled(Q_req, n_ratio) = dp_req
    (Bisection; nur wenn Vorzeichenwechsel existiert)
    """
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
    """Worst Case: nächsthöhere GVF-Kurve (ceiling)"""
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def choose_best_mph_pump_normbased(pumps, Q_req, dp_req, gvf_free_req_pct,
                                  dp_margin=0.10):
    """
    Auswahl Mehrphasenpumpe:
      - gvf_free_req_pct in %
      - dp_req in bar (∆p)
      - dp_margin Reserve auf ∆p (z.B. 10%)
      - berücksichtigt optional Drehzahl über Affinitäten (Q~n, ∆p~n², P~n³)
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
        in_curve = (Qmin <= Q_req <= Qmax)

        candidates = []

        # A) Nenndrehzahl
        if in_curve:
            dp_avail_nom = interp_clamped(Q_req, curve["Q"], curve["p"])
            if dp_avail_nom >= dp_req * (1.0 + dp_margin):
                P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail_nom - dp_req) + abs(gvf_key - gvf_free_req_pct) * 0.25
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_avail_nom,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "mode": "Nenndrehzahl",
                    "dp_reserve": dp_avail_nom - dp_req,
                    "score": score
                })

        # B) Drehzahl-Anpassung (Bisektion)
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["p"], Q_req, dp_req)
        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if Qmin <= Q_base <= Qmax:
                dp_scaled = _dp_scaled_at_Q(curve["Q"], curve["p"], Q_req, n_ratio)
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
                    "score": score
                })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

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
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur")

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

    # Umrechnung viskos -> Wasser
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    st.info(
        f"{'✅' if B < 1.0 else '⚠️'} "
        f"B = {B:.2f}  |  "
        f"{'B < 1.0 → geringe Viskositätseffekte' if B < 1.0 else 'B ≥ 1.0 → Viskositätskorrektur aktiv'}"
    )

    st.markdown("### 📊 Umrechnung viskos → Wasser (Auslegung auf Wasserkennlinie)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q_Wasser", f"{Q_water:.2f} m³/h")
    c2.metric("H_Wasser", f"{H_water:.2f} m", f"+{H_water - H_vis_req:.1f} m")
    c3.metric("B-Zahl", f"{B:.2f}")
    c4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")

    # Auswahl (WICHTIG: PUMPS verwenden -> verhindert NameError)
    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]
    eta_water = best["eta_at"]
    eta_vis = max(1e-6, eta_water * Ceta)

    # Leistung viskos
    Q_m3s = Q_vis_req / 3600.0
    P_hyd_W = rho * G * Q_m3s * H_vis_req  # W
    P_vis_kW = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    st.markdown("### ✅ **AUSLEGUNGSERGEBNIS (Einphasen)**")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} l/min")
    col2.metric("H (viskos)", f"{H_vis_req:.2f} m")
    col3.metric("η (viskos)", f"{eta_vis:.3f}")
    col4.metric("P Welle (viskos)", f"{P_vis_kW:.2f} kW")
    col5.metric("IEC-Motor (+Reserve)", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(
            f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
            f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
        )

    # Kennlinien viskos
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

    with st.expander("📘 Rechenweg (ausführlich) – Viskosität", expanded=False):
        st.markdown(f"""
### 1) Gegeben (viskos)
- Förderstrom: **Qᵥ = {Q_vis_req:.3f} m³/h**
- Förderhöhe: **Hᵥ = {H_vis_req:.3f} m**
- Dichte: **ρ = {rho:.1f} kg/m³**
- kinem. Viskosität: **ν = {nu:.2f} cSt**

### 2) Bildung der B-Zahl (Näherung)
Wir nutzen eine HI-ähnliche Kennzahl:
- Umrechnung:  
  Q in gpm: **Q_gpm = Q·4.40287**  
  H in ft: **H_ft = H·3.28084**
- Formel (Näherung):  
  **B = 16.5 · √ν / (Q_gpm^0.25 · H_ft^0.375)**

Ergebnis:
- **B = {B:.3f}**

### 3) Korrekturfaktoren
Aus B werden Faktoren gebildet:
- **CH**: Förderhöhenfaktor (H sinkt viskos)  
- **Cη**: Wirkungsgradfaktor (η sinkt viskos)

Ergebnis:
- **CH = {CH:.3f}**
- **Cη = {Ceta:.3f}**

Interpretation:
- viskose Förderhöhe: **Hᵥ ≈ H_w · CH**
- viskoser Wirkungsgrad: **ηᵥ ≈ η_w · Cη**

### 4) Umrechnung auf Wasserkennlinie (Auslegungsschritt)
Wir rechnen den gewünschten viskosen Betriebspunkt auf die Wasserkennlinie um:
- **Q_w = Qᵥ = {Q_water:.3f} m³/h**
- **H_w = Hᵥ / CH = {H_water:.3f} m**

Damit wählen wir die Pumpe auf Basis der Wasserkennlinie (damit du echte Herstellerkennlinien nutzen kannst).

### 5) Pumpenauswahl auf Wasserkennlinie
- Wir interpolieren H_w(Q_w) für jede Pumpe und minimieren |H_at − H_w|  
- Falls Q_w außerhalb Kennlinie: optional Penalty (wenn aktiviert)

Gewählt:
- **Pumpe = {best['id']}**
- Interpoliert bei Q_eval={best['Q_eval']:.3f} m³/h:
  - **H_at = {best['H_at']:.3f} m**
  - **η_w = {best['eta_at']:.3f}**

### 6) Zurückrechnung auf viskos (für Leistung)
- **ηᵥ = η_w · Cη = {eta_vis:.4f}**

Hydraulische Leistung:
- Q in m³/s: **Q = {Q_m3s:.6f}**
- **P_hyd = ρ·g·Q·H = {P_hyd_W:,.0f} W**

Wellenleistung:
- **P_Welle = P_hyd / ηᵥ = {P_vis_kW:.3f} kW**

Motor:
- Reserve: **{reserve_pct}%**
- IEC Stufe: **{P_motor_kW:.2f} kW**
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
        st.subheader("Betriebspunkt (Hydraulik)")
        Q_req = st.number_input("Volumenstrom Q [m³/h]", 0.1, 150.0, 15.0, 1.0)

        # Normlogik: ∆p-Auslegung (ohne Δp_loss – wie gewünscht)
        p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 100.0, 2.0, 0.1)
        p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 200.0, 7.0, 0.1)
        dp_req = max(0.0, (p_discharge - p_suction))

        # KEINE eigene Sidebar-Sektion "Gasanteile" mehr – nur der Slider:
        gvf_in = st.slider("Gesamt-GVF_in [%] (am Saugpunkt)", 0, 40, 10, 1)

        st.divider()
        st.subheader("Plot")
        show_temp_band = st.checkbox("Löslichkeit bei T-10/T/T+10", value=True)

    # --- Löslichkeit am Saugpunkt (für DIAGONALE Darstellung: p_ref = p_suction)
    p_ref = p_suction  # Referenzdruck für "diagonale" Löslichkeit
    dissolved_suction_cm3_ref = dissolved_gas_cm3_per_L_ref(
        gas_medium, p_suction, temperature, p_ref_bar_abs=p_ref, y_gas=y_gas
    )

    # --- Für GVF_free (Normlogik)
    dissolved_suction_cm3N = dissolved_gas_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)
    gvf_free, gvf_dbg = free_gvf_at_suction(gvf_in, dissolved_suction_cm3N, p_suction, temperature)

    # --- Pumpe wählen
    best = choose_best_mph_pump_normbased(MPH_PUMPS, Q_req, dp_req, gvf_free)

    # =====================================================
    # Plot 1+2 (nebeneinander)
    # =====================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---------- Plot 1: DIAGONALE Löslichkeit vs Druck (cm³/L bezogen auf p_ref=p_s)
    if show_temp_band:
        temp_variants = [temperature - 10, temperature, temperature + 10]
        temp_variants = [t for t in temp_variants if -10 <= t <= 150]
    else:
        temp_variants = [temperature]

    p_max_plot = max(14.0, p_discharge * 1.2, p_suction * 1.2)
    p_max_plot = min(30.0, p_max_plot)

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "black"]
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure_cm3_ref(
            gas_medium, T, p_max=p_max_plot, p_ref_bar_abs=p_ref, y_gas=y_gas
        )
        ax1.plot(
            pressures, sol,
            "--", linewidth=2,
            color=color_cycle[i % len(color_cycle)],
            label=f"{gas_medium} bei {T:.0f}°C (ref p_s, y={y_gas:.2f})"
        )

    ax1.scatter([p_suction], [dissolved_suction_cm3_ref], s=180, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt", zorder=5)

    ax1.set_xlabel("p_abs [bar]")
    ax1.set_ylabel("Löslichkeit [cm³/L] (bezogen auf p_ref = p_s)")
    ax1.set_title("Gaslöslichkeit – diagonal wie Vorlage (Henry, Referenz p_s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # ---------- Plot 2: ∆p-Q Kennlinien
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
    st.pyplot(fig, clear_figure=True)

    # =====================================================
    # Plot 3 (Overlay) – wie in deiner Vorlage
    #   - DIAGONALE Löslichkeit (ref p_s)
    #   - Freies Gasvolumen pro L Flüssigkeit (bei p) für GVF 10/15/20 (und optional GVF_in)
    # =====================================================
    st.markdown("### 📉 Overlay: Löslichkeit (diagonal) + freies Gasvolumen (bei p)")

    p_overlay_max = max(14.0, p_discharge * 2.0)
    p_overlay_max = min(30.0, p_overlay_max)

    pressures_overlay = np.linspace(0.0, p_overlay_max, 200)

    fig_o, axo = plt.subplots(figsize=(14, 6))

    # Löslichkeit diagonal (ref p_s) – gestrichelt
    for i, T in enumerate(temp_variants):
        pr, sol = solubility_curve_vs_pressure_cm3_ref(
            gas_medium, T, p_max=p_overlay_max, p_ref_bar_abs=p_ref, y_gas=y_gas
        )
        axo.plot(pr, sol, "--", linewidth=2,
                 color=color_cycle[i % len(color_cycle)],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C (ref p_s)")

    # Freies Gasvolumen-Kurven (bei p) – solide Linien (starten ab p_s, damit es nicht "explodiert")
    gvf_plot = [10, 15, 20]
    if gvf_in not in gvf_plot:
        gvf_plot = sorted(gvf_plot + [gvf_in])

    # Farbwahl (ähnlich Vorlage: Blautöne)
    gvf_line_colors = {
        10: "#9bbcf2",
        15: "#5e93e8",
        20: "#1f5edb",
    }

    y_max_collect = []
    for gvf_pct in gvf_plot:
        yvals = []
        for p_now in pressures_overlay:
            if p_now < p_suction:
                yvals.append(np.nan)  # Kurve beginnt erst ab Saugdruck
                continue
            v_free_cm3L, n_total, n_diss, n_free = free_gas_volume_cm3_per_L_at_pressure(
                gvf_pct, gas_medium, p_suction, p_now, temperature, y_gas=y_gas
            )
            yvals.append(v_free_cm3L)
            if not np.isnan(v_free_cm3L):
                y_max_collect.append(v_free_cm3L)

        color = gvf_line_colors.get(gvf_pct, "tab:blue")
        lw = 3.0 if gvf_pct == gvf_in else 2.0
        alpha = 1.0 if gvf_pct == gvf_in else 0.85
        axo.plot(pressures_overlay, yvals, linewidth=lw, alpha=alpha, color=color,
                 label=f"{gvf_pct:.0f}% GVF (freies Gasvolumen/L bei p)")

    # Marker am Saugpunkt (Löslichkeit)
    axo.scatter([p_suction], [dissolved_suction_cm3_ref], s=140, marker="o",
                edgecolors="black", linewidths=2, label="Saugpunkt (Löslichkeit)", zorder=5)

    axo.set_xlabel("Druck [bar abs]")
    axo.set_ylabel("Löslichkeit [cm³/L] (ref p_s) / freies Gasvolumen [cm³/L] (bei p)")
    axo.set_title("Overlay: Löslichkeit (diagonal) vs. freies Gasvolumen (komprimiert)")

    axo.grid(True, alpha=0.3)
    axo.set_xlim(0, p_overlay_max)

    if len(y_max_collect) > 0:
        y_top = max(max(y_max_collect) * 1.15, max([dissolved_suction_cm3_ref * 1.15, 180.0]))
        # damit es nicht ausufert
        y_top = min(y_top, 1200.0)
        axo.set_ylim(0, y_top)
    else:
        axo.set_ylim(0, max(dissolved_suction_cm3_ref * 1.2, 180.0))

    axo.legend(fontsize=9, ncol=2)
    st.pyplot(fig_o, clear_figure=True)

    # =====================================================
    # Ergebnisse + ausführlicher Rechenweg
    # =====================================================
    st.divider()
    st.markdown("### ✅ Ergebnisse (normlogisch)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("∆p_req", f"{dp_req:.2f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
    c3.metric("GVF_in (Saugpunkt)", f"{gvf_in:.0f} %")
    c4.metric("Löslichkeit (Saugpunkt)", f"{dissolved_suction_cm3_ref:.1f} cm³/L", "diagonal (ref p_s)")
    c5.metric("GVF_free (für Kennlinie)", f"{gvf_free:.1f} %")

    if gvf_free > 0.0:
        st.warning("⚠️ Freies Gas vorhanden (GVF_free > 0). Kennlinienauswahl erfolgt Worst Case auf nächsthöhere GVF-Kurve.")
    else:
        st.info("ℹ️ Kein freies Gas aus dieser Abschätzung am Saugpunkt (alles im Löslichkeitslimit).")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("∆p verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve: {best['dp_reserve']:.2f} bar")
        c2.metric("Leistung", f"{best['P_required']:.2f} kW")
        c3.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        c4.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

        if best["dp_available"] < dp_req:
            st.error("❌ ∆p_req wird nicht erreicht (sollte bei Auswahl nicht passieren).")
    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")
        st.markdown("""
**Typische Gründe:**
- ∆p_req zu hoch für alle Pumpen/Kennlinien
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
            st.caption(f"∆p_max: {pmp['p_max_bar']} bar")
            st.caption(f"GVF_max: {pmp['GVF_max']*100:.0f}%")

    with st.expander("📘 Rechenweg (ausführlich) – Mehrphase", expanded=False):
        # zusätzliche Zwischenwerte für Transparenz
        H_T = henry_constant(gas_medium, temperature)
        C_s = dissolved_moles_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)

        n_total = gvf_dbg["n_total"]
        n_diss_s = gvf_dbg["n_diss_s"]
        n_free_s = gvf_dbg["n_free_s"]
        Vgas_total_s = gvf_dbg["Vgas_total_s_L_per_L"]
        Vgas_free_s = gvf_dbg["Vgas_free_s_L_per_L"]

        st.markdown(f"""
### 1) ∆p-Anforderung (normlogisch)
Wir dimensionieren Mehrphasenpumpen typischerweise über **∆p** (Druckerhöhung):

- **∆p_req = p_d − p_s**
- Eingaben:
  - p_s = **{p_suction:.2f} bar abs**
  - p_d = **{p_discharge:.2f} bar abs**
- Ergebnis:
  - **∆p_req = {dp_req:.2f} bar**

---

### 2) Henry-Konstante und gelöste Kapazität (am Saugpunkt)
Henry (vereinfacht):
- **H(T) = A · exp(B · (1/T − 1/T0))**
- Ergebnis:
  - **H(T={temperature:.1f}°C) = {H_T:.2f} bar·L/mol**

Gelöste Konzentration (Kapazität) bei p_s:
- Partialdruck: **p_partial = y_gas · p_s = {y_gas:.2f} · {p_suction:.2f}**
- **C_s = p_partial / H(T) = {C_s:.6f} mol/L**

Für die **diagonale Darstellung** (wie Vorlage) rechnen wir diese Kapazität auf ein Referenzvolumen bei **p_ref = p_s** um:
- **V_ref/L = C · (R·T)/p_ref**
- **Löslichkeit am Saugpunkt (ref p_s)**:
  - **{dissolved_suction_cm3_ref:.2f} cm³/L**

---

### 3) Gasmenge aus GVF_in (am Saugpunkt)
Interpretation: **GVF_in gilt volumetrisch am Saugpunkt**.

Setze V_liq = 1 L:
- **V_gas,s = gvf/(1−gvf) · V_liq**
- bei GVF_in = {gvf_in:.0f}%:
  - **V_gas,s = {Vgas_total_s:.5f} L Gas pro L Flüssigkeit**

Umrechnung in Stoffmenge (ideal, bei p_s):
- **n_total = p_s · V_gas,s / (R·T)**
- Ergebnis:
  - **n_total = {n_total:.6f} mol/L_liq**

---

### 4) Freies Gas am Saugpunkt (GVF_free)
Für die Kennlinienauswahl benötigen wir **freies Gas** (nicht gelöst).

Wir nutzen eine **Norm-Logik**:
- gelöste Normkapazität am Saugpunkt: **{dissolved_suction_cm3N:.2f} cm³N/L**
- Umrechnung nach mol (V_molar,N≈22.414 L/mol):
  - **n_diss,s ≈ {n_diss_s:.6f} mol/L**
- Freie Stoffmenge:
  - **n_free,s = max(0, n_total − n_diss,s) = {n_free_s:.6f} mol/L**

Rückrechnung in freies Gasvolumen am Saugpunkt:
- **V_free,s = n_free,s · (R·T)/p_s = {Vgas_free_s:.6f} L/L**
- daraus **GVF_free**:
  - **GVF_free = {gvf_free:.2f}%**

---

### 5) Kennlinienwahl (Worst Case)
- Wir wählen die **nächsthöhere** hinterlegte GVF-Kurve:
  - GVF_free = {gvf_free:.2f}%  → Kurve **{best['gvf_curve'] if best else '—'}%** (falls Auswahl möglich)

---

### 6) Vergleich gegen ∆p-Kennlinie und Drehzahl
Nenndrehzahl:
- ∆p_avail(Q_req) aus Kennlinie → Muss ≥ ∆p_req·(1+Reserve) sein

Optional Drehzahl (Affinität):
- **Q ~ n**
- **∆p ~ n²**
- **P ~ n³**
Wir suchen n/n0 via Bisektion, so dass:
- **∆p_scaled(Q_req, n) = ∆p_req**

Ergebnis (falls ausgewählt):
- Modus: **{best['mode'] if best else '—'}**
- n/n0: **{best['n_ratio'] if best else 0:.3f}**
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

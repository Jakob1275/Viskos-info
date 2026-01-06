# app.py
# ------------------------------------------------------------
# Pumpenauslegungstool (Einphasen + Mehrphasen + ATEX)
# - Einphasen: Viskositätskorrektur (HI/ISO-Logik, pragmatische Näherung)
# - Mehrphasen: Henry-Löslichkeit (als "Sättigung" in cm³N/L -> diagonal) +
#               freier GVF am Saugpunkt + Δp-Kennlinien + Drehzahl (Affinität)
# - ATEX: vereinfachte Motorauswahl
#
# WICHTIG:
# - Mehrphasen-"Sättigungskennlinien" sind hier bewusst als cm³N/L (Normvolumen) geplottet,
#   damit sie (Henry: C ~ p) DIAGONAL verlaufen – wie in deiner Vorgabe.
# - Der Overlay-Plot zeigt zusätzlich das komprimierte freie Gasvolumen pro Liter Flüssigkeit
#   (bei gegebenem GVF am Saugpunkt), das mit steigendem Druck ~ 1/p abnimmt – wie im Beispielbild.
# ------------------------------------------------------------

import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Konstanten
# ---------------------------
G = 9.80665  # m/s²
R_BAR_L = 0.08314462618  # bar·L/(mol·K) ideales Gas
P_N_BAR = 1.0  # "Normaldruck" für Normvolumen (vereinfachend 1.0 bar)
T_N_K = 273.15  # "Normtemperatur" (0°C) für Normvolumen
V_MOLAR_N_L_PER_MOL = R_BAR_L * T_N_K / P_N_BAR  # [L/mol] bei (P_N, T_N)

# Sicherheitsmargen (Engineering-typisch; Werte können je nach Standard/Projekt variieren)
DP_MARGIN = 0.10  # 10% Δp-Reserve
N_RATIO_MIN = 0.50
N_RATIO_MAX = 1.10

# ---------------------------
# Pumpenkennlinien (Einphasen) - Beispiel
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
# Mehrphasen-Pumpen (Kennlinien als Δp in bar) - Beispiel
# keys in curves: GVF in Prozent (0,5,10,15 ...)
# ---------------------------
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

# ---------------------------
# ATEX-Datenbank (vereinfachtes Beispiel)
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

# Henry-Konstanten (Demo-Parameter)
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
    """Lineare Interpolation mit Clamping; xs aufsteigend erwartet."""
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
    """Nächste IEC-Stufe (vereinfachte Liste)."""
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================================================
# Einphasen: Viskositätskorrektur (HI/ISO-Logik – Näherung)
# =========================================================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """
    B-Kennzahl in Anlehnung an HI-Ansätze (pragmatische Näherung).
    Intern Umrechnung in gpm/ft wie in vielen HI-Darstellungen.
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
    - CH: Förderhöhenfaktor (H_vis = H_w * CH)
    - Ceta: Wirkungsgradfaktor (eta_vis = eta_w * Ceta)
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
    Umrechnung von viskosem Betriebspunkt auf äquivalenten Wasserbetriebspunkt.
    Logik:
      H_w = H_vis / CH
      Q_w = Q_vis (in dieser Näherung konstant)
    """
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / CH}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """Rückrechnung von Wasserkennlinie auf viskos (für Kurvendarstellung)."""
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return Q_water, H_water * CH, max(1e-6, eta_water * Ceta)

def generate_viscous_curve(pump, nu_cSt, rho):
    """Erzeugt viskose Kennlinie aus Wasserkennlinie (H, eta, P)."""
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
    """Wählt Pumpe, die bei Q_water H am besten trifft (mit optionaler Range-Strafe)."""
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
# Mehrphasen: Henry-Löslichkeit (diagonal als cm³N/L) + freier GVF + Affinität
# =========================================================
def henry_constant(gas, T_celsius):
    """
    Temperaturabhängige Henry-Konstante (vereinfachte Parameterform).
    H in [bar·L/mol]
    """
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def dissolved_gas_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    "Sättigungskurve" als Normvolumen (cm³N/L):
      Henry: C = p_partial/H  [mol/L]  -> linear in p
      Normvolumen: V_N = C * V_molar_N  [L_N/L]
      -> cm³N/L = 1000 * V_N
    Dadurch verlaufen die Sättigungskurven DIAGONAL (wie in deiner Vorgabe).
    """
    p = max(p_bar_abs, 1e-9)
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    H = max(henry_constant(gas, T_celsius), 1e-12)
    C_mol_L = p_partial / H  # mol/L
    Vn_L_per_L = C_mol_L * V_MOLAR_N_L_PER_MOL
    return 1000.0 * Vn_L_per_L  # cm³N/L

def solubility_curve_vs_pressure_cm3N(gas, T_celsius, p_max, y_gas=1.0):
    pressures = linspace(0.0, p_max, 160)
    sol = [dissolved_gas_cm3N_per_L(gas, max(1e-6, p), T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol

def gvf_to_Vgas_per_Lliq_at_same_p(gvf_pct):
    """
    Bei gegebenem GVF (Volumenanteil Gas am gleichen Druck):
      GVF = Vgas/(Vgas+Vliq)  =>  Vgas/Vliq = GVF/(1-GVF)
    Liefert Vgas [L] pro 1 L Flüssigkeit (dimensionslos L/L).
    """
    gvf = clamp(gvf_pct / 100.0, 0.0, 0.999999)
    return gvf / (1.0 - gvf)

def free_gvf_at_suction(gvf_total_in_pct, dissolved_cm3N_per_L, p_suction_bar, T_celsius):
    """
    Abschätzung "freier GVF am Saugpunkt".

    Idee (Engineering-Näherung):
    1) Aus GVF_in (volumetrisch am Saugdruck p_s) berechne Vgas_in (L/L) am Saugpunkt.
    2) Wandle dieses Vgas_in in Normvolumen um (cm³N/L), um es mit der Löslichkeit (cm³N/L) zu vergleichen:
         Vgas_N = Vgas_in * (p_s / P_N) * (T_N / T)
       (Ideales Gas: n ~ pV/T; Normbezug über p und T)
    3) Freies Gas in Normvolumen:
         Vfree_N = max(0, Vgas_N - Vdiss_N)
    4) Wandle Vfree_N zurück in reales Gasvolumen am Saugpunkt:
         Vfree = (Vfree_N/1000) * (P_N/p_s) * (T/T_N)
    5) Freier GVF am Saugpunkt:
         GVF_free = Vfree/(Vfree+1 L)

    Ergebnis in %.
    """
    T_K = T_celsius + 273.15
    Vgas_in_L_per_L = gvf_to_Vgas_per_Lliq_at_same_p(gvf_total_in_pct)  # L/L am p_s
    # in Normvolumen (cm³N/L)
    VgasN_cm3N_per_L = 1000.0 * Vgas_in_L_per_L * (p_suction_bar / P_N_BAR) * (T_N_K / T_K)
    VfreeN_cm3N_per_L = max(0.0, VgasN_cm3N_per_L - max(0.0, dissolved_cm3N_per_L))

    # zurück zum Gasvolumen am Saugpunkt (L/L)
    Vfree_L_per_L = (VfreeN_cm3N_per_L / 1000.0) * (P_N_BAR / max(p_suction_bar, 1e-9)) * (T_K / T_N_K)
    gvf_free = Vfree_L_per_L / (Vfree_L_per_L + 1.0)
    return 100.0 * gvf_free, {
        "Vgas_in_L_per_L": Vgas_in_L_per_L,
        "VgasN_cm3N_per_L": VgasN_cm3N_per_L,
        "VfreeN_cm3N_per_L": VfreeN_cm3N_per_L,
        "Vfree_L_per_L": Vfree_L_per_L
    }

def choose_gvf_curve_key_worstcase(curves_dict, gvf_free_req_pct):
    """
    Worst-Case-Kennlinienwahl: nächste höhere verfügbare GVF-Kurve (ceiling).
    Wenn gvf_free_req größer als höchste Kurve -> höchste Kurve.
    """
    keys = sorted(curves_dict.keys())
    for k in keys:
        if k >= gvf_free_req_pct:
            return k
    return keys[-1]

def _dp_scaled_at_Q(curve_Q, curve_dp, Q_req, n_ratio):
    """
    Affinität:
      Q ~ n,  Δp ~ n²
    Bei gewünschtem Q_req und n_ratio:
      Q_base = Q_req/n_ratio (Punkt auf Basiskurve)
      Δp_scaled(Q_req) = Δp_base(Q_base) * n_ratio²
    """
    Q_base = Q_req / max(n_ratio, 1e-12)
    dp_base = interp_clamped(Q_base, curve_Q, curve_dp)
    return dp_base * (n_ratio ** 2), Q_base, dp_base

def find_speed_ratio_bisection(curve_Q, curve_dp, Q_req, dp_target,
                               n_min=N_RATIO_MIN, n_max=N_RATIO_MAX,
                               tol=1e-3, iters=70):
    """
    Sucht n_ratio, sodass Δp_scaled(Q_req, n_ratio) ≈ dp_target.
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

def choose_best_mph_pump(pumps, Q_req, dp_req, gvf_free_req_pct, dp_margin=DP_MARGIN):
    """
    Normlogik/Engineering-Logik:
    - Auslegung auf Δp (Differenzdruck)
    - Kennlinienauswahl über freien GVF (Worst-Case: nächste höhere GVF-Kurve)
    - Mindestreserve Δp: Δp_avail >= Δp_req*(1+Margin)
    - Optionale Drehzahlanpassung via Affinität (Q~n, Δp~n², P~n³)
    """
    best = None

    for pump in pumps:
        if gvf_free_req_pct > pump["GVF_max"] * 100.0:
            continue
        if Q_req > pump["Q_max_m3h"] * N_RATIO_MAX:
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
                score = abs(dp_avail - dp_req) + abs(gvf_key - gvf_free_req_pct) * 0.25
                candidates.append({
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "dp_available": dp_avail,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "mode": "Nenndrehzahl",
                    "dp_reserve": dp_avail - dp_req,
                    "score": score,
                    "Q_base": Q_req,
                    "dp_base": dp_avail
                })

        # B) Drehzahl (Ziel: dp_req*(1+margin) treffen, nicht nur dp_req)
        dp_target = dp_req * (1.0 + dp_margin)
        n_ratio = find_speed_ratio_bisection(curve["Q"], curve["p"], Q_req, dp_target)
        if n_ratio is not None:
            dp_scaled, Q_base, dp_base = _dp_scaled_at_Q(curve["Q"], curve["p"], Q_req, n_ratio)
            if Qmin <= Q_base <= Qmax:
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
                    "score": score,
                    "Q_base": Q_base,
                    "dp_base": dp_base
                })

        for cand in candidates:
            if best is None or cand["score"] < best["score"]:
                best = cand

    return best

def gas_volume_curve_from_gvf_at_suction(gvf_pct, p_suction_bar, pressures_bar, T_isothermal=True):
    """
    Kurve: Gasvolumen pro Liter Flüssigkeit bei Druck p (cm³/L, reales Volumen bei p),
    ausgehend von einem GVF-Wert AM SAUGPUNKT p_s.

    Annahme: Isotherme Kompression des freien Gases (Boyle: V ~ 1/p).
    - Am Saugpunkt: Vgas_s/Lliq = GVF/(1-GVF)  [L/L]
    - Bei p: Vgas(p) = Vgas_s * (p_s/p)
    - Ausgabe in cm³/L (reales Volumen bei p)
    """
    Vgas_s_L_per_L = gvf_to_Vgas_per_Lliq_at_same_p(gvf_pct)
    out = []
    for p in pressures_bar:
        p_eff = max(p, 1e-9)
        Vgas_L_per_L = Vgas_s_L_per_L * (p_suction_bar / p_eff)
        out.append(1000.0 * Vgas_L_per_L)  # cm³/L
    return out

# =========================================================
# Streamlit Setup (muss vor st.*-Ausgaben stehen)
# =========================================================
st.set_page_config(page_title="Pumpenauslegung", layout="wide")
st.title("Pumpenauslegungstool")

if "page" not in st.session_state:
    st.session_state.page = "pump"

# =========================================================
# Sidebar Navigation (nur Navigation hier; keine "Gasanteile"-Sektion)
# =========================================================
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
# PAGE 1: Einphasen (Viskosität)
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
    Q_water, H_water = conv["Q_water"], conv["H_water"]
    B, CH, Ceta = conv["B"], conv["CH"], conv["Ceta"]

    st.info(
        f"{'✅' if B < 1.0 else '⚠️'} B = {B:.2f} "
        f"{'< 1.0 → geringe Viskositätseffekte' if B < 1.0 else '≥ 1.0 → Viskositätskorrektur relevant'}"
    )

    st.markdown("### 📊 Umrechnung viskos → äquivalenter Wasserbetriebspunkt")
    a, b, c, d = st.columns(4)
    a.metric("Q_w", f"{Q_water:.2f} m³/h", f"{m3h_to_lmin(Q_water):.1f} L/min")
    b.metric("H_w", f"{H_water:.2f} m", f"ΔH={H_water - H_vis_req:+.1f} m")
    c.metric("B-Zahl", f"{B:.2f}")
    d.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")

    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out_of_range=allow_out)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()

    p = best["pump"]
    eta_water = best["eta_at"]
    eta_vis = max(1e-6, eta_water * Ceta)

    # Leistung am viskosen Betriebspunkt
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / eta_vis) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    st.markdown("### ✅ Ergebnis (Einphasen)")
    st.success(f"**Gewählte Pumpe: {best['id']}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Qᵥ", f"{Q_vis_req:.2f} m³/h", f"{m3h_to_lmin(Q_vis_req):.1f} L/min")
    col2.metric("Hᵥ", f"{H_vis_req:.2f} m")
    col3.metric("Q_w", f"{Q_water:.2f} m³/h")
    col4.metric("H_w", f"{H_water:.2f} m")
    col5.metric("P_Welle", f"{P_vis_kW:.2f} kW")
    col6.metric("IEC-Motor", f"{P_motor_kW:.2f} kW", f"+{reserve_pct}%")

    if not best["in_range"]:
        st.warning(
            f"⚠️ Q_w außerhalb Kennlinie ({min(p['Qw'])}…{max(p['Qw'])} m³/h). "
            f"Bewertung bei Q_eval={best['Q_eval']:.2f} m³/h."
        )

    # Kurven
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])

    st.divider()
    st.markdown("### 📈 Kennlinien (Wasser vs. viskos)")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, label=f"{p['id']} (viskos)")
        ax1.scatter([Q_water], [H_water], marker="^", s=150, edgecolors="black", linewidths=2, label="BP (Wasser)", zorder=5)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=200, linewidths=3, label="BP (viskos)", zorder=5)
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
        ax2.scatter([Q_water], [eta_water], marker="^", s=150, edgecolors="black", linewidths=2, label="η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=200, linewidths=3, label="η (viskos)", zorder=5)
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
        ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, edgecolors="black", linewidths=2, label="BP (Wasser)", zorder=5)
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, linewidths=3, label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P Kennlinien")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    # ---- Ausführlicher Rechenweg (Einphasen)
    with st.expander("📘 Rechenweg (ausführlich): Viskositätskorrektur & Auslegung", expanded=False):
        st.markdown(f"""
### 1) Eingaben (viskoses Medium)
- Volumenstrom: **Qᵥ = {Q_vis_req:.3f} m³/h**  
- Förderhöhe: **Hᵥ = {H_vis_req:.3f} m**  
- Dichte: **ρ = {rho:.1f} kg/m³**  
- Kinematische Viskosität: **ν = {nu:.3f} cSt**  

> Ziel: Auswahl einer Pumpe anhand der **Wasserkennlinie** (Herstellerangaben sind i. d. R. auf Wasser nach ISO 9906),
> aber korrigiert für Viskositätseinflüsse (HI/ISO-Logik).

---

### 2) Bildung einer Viskositätskennzahl B (HI-nahe Näherung)
Wir berechnen eine Kennzahl **B**, die die Stärke der Viskositätseinflüsse charakterisiert.

- Umrechnung:  
  Q in gpm, H in ft  
- Näherungsformel im Code:
""")
        st.latex(r"B = \frac{16.5\cdot \sqrt{\nu}}{Q^{0.25}\cdot H^{0.375}}")
        st.markdown(f"""
Mit den Zahlen ergibt sich:
- **B = {B:.3f}**

Interpretation:
- **B < 1**: Viskositätseinfluss klein, Wasserkennlinie oft ausreichend.
- **B ≥ 1**: Viskositätskorrektur relevant (H sinkt, η sinkt, Leistungsbedarf steigt).

---

### 3) Korrekturfaktoren bestimmen (CH und Cη)
Aus B werden Korrekturfaktoren ermittelt:
- **CH = {CH:.4f}** (Förderhöhenfaktor)  
- **Cη = {Ceta:.4f}** (Wirkungsgradfaktor)

Bedeutung:
- Rückrechnung auf Wasser: **H_w = Hᵥ / CH**  
- Rückrechnung auf viskos: **Hᵥ ≈ H_w · CH**  
- Wirkungsgrad: **ηᵥ ≈ η_w · Cη**

---

### 4) Umrechnung des Betriebspunktes auf Wasserbasis
- **Q_w = Qᵥ = {Q_water:.3f} m³/h** (in dieser Näherung konstant)  
- **H_w = Hᵥ / CH = {H_vis_req:.3f} / {CH:.4f} = {H_water:.3f} m**

> Damit suchen wir die beste Pumpe auf der Wasserkennlinie bei (Q_w, H_w).

---

### 5) Pumpenauswahl auf Wasserkennlinie
Auswahlkriterium:
- Interpoliere **H(Q_w)** und **η(Q_w)** der Pumpen
- Minimierung der Abweichung **|H(Q_w) - H_w|** (mit optionaler Range-Strafe)

Ergebnis:
- Gewählte Pumpe: **{best["id"]}**
- Interpolierte Werte am Auswertungspunkt:
  - **H_at = {best["H_at"]:.3f} m**
  - **η_w = {eta_water:.4f}**

---

### 6) Rückrechnung auf viskosen Betriebspunkt (für η und Leistungsbedarf)
- **ηᵥ = η_w · Cη = {eta_water:.4f} · {Ceta:.4f} = {eta_vis:.4f}**

---

### 7) Leistungsrechnung
Hydraulische Leistung:
""")
        st.latex(r"P_{hyd} = \rho \cdot g \cdot Q \cdot H")
        st.markdown(f"""
Mit Q in m³/s:
- Q = {Q_vis_req:.3f}/3600 = **{Q_vis_req/3600.0:.6f} m³/s**
- **P_hyd = {P_hyd_W:.1f} W**

Wellenleistung (vereinfachend, ohne zusätzliche mechanische Verluste):
- **P_Welle = P_hyd / ηᵥ = {P_vis_kW:.3f} kW**

Motorauslegung (Reserve {reserve_pct}%):
- **P_motor_min = {P_vis_kW:.3f} · (1+{reserve_pct}/100) = {P_vis_kW*(1+reserve_pct/100):.3f} kW**
- IEC-Stufe: **{P_motor_kW:.2f} kW**

---

### 8) Norm-/Standardbezug (Hinweis)
- Herstellerkennlinien und Abnahme typischerweise auf Wasser: **DIN EN ISO 9906**
- Viskositätskorrekturen in der Praxis häufig nach HI/ISO-Ansätzen (z. B. **ANSI/HI** bzw. **ISO/TR 17766** als Leitlinie)

> Hinweis: Diese Implementierung ist bewusst **pragmatisch** (Engineering-Tool). Für Vertragsauslegung/Abnahme:
> Hersteller-/HI-Korrekturtabellen bzw. geprüfte Korrekturfaktoren verwenden.
""")

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit (p,T) + freier GVF + Δp-Kennlinien + Auswahl")

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben (Mehrphasen)")

        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", -10.0, 150.0, 20.0, 1.0)

        # Partialdruckfaktor (optional, aber ohne extra Sektion)
        y_gas = st.slider("Gasanteil y_gas (Partialdruckfaktor) [-]", 0.0, 1.0, 1.0, 0.05)

        st.divider()
        st.subheader("Betriebspunkt (Drücke, Δp)")
        Q_req = st.number_input("Volumenstrom Q [m³/h]", 0.1, 150.0, 15.0, 1.0)
        p_suction = st.number_input("Saugdruck p_s [bar abs]", 0.1, 200.0, 2.0, 0.1)
        p_discharge = st.number_input("Druckseite p_d [bar abs]", 0.1, 300.0, 7.0, 0.1)

        # Δp_loss wurde auf Wunsch ENTFERNT
        dp_req = max(0.0, (p_discharge - p_suction))

        st.divider()
        st.subheader("GVF am Saugpunkt")
        gvf_in = st.slider("Gesamt-GVF_in am Saugpunkt [%]", 0, 40, 10, 1)

        st.divider()
        st.subheader("Plot")
        show_temp_band = st.checkbox("Sättigung bei T-10/T/T+10", value=True)

    # ---------------------------
    # 1) Sättigung (diagonal) als cm³N/L am Saugpunkt
    # ---------------------------
    dissolved_suction_cm3N_L = dissolved_gas_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=y_gas)

    # ---------------------------
    # 2) Freier GVF am Saugpunkt (aus GVF_in und Löslichkeit)
    # ---------------------------
    gvf_free_pct, gvf_dbg = free_gvf_at_suction(
        gvf_total_in_pct=gvf_in,
        dissolved_cm3N_per_L=dissolved_suction_cm3N_L,
        p_suction_bar=p_suction,
        T_celsius=temperature
    )

    # ---------------------------
    # 3) Pumpenauswahl (Δp + freier GVF + Drehzahl)
    # ---------------------------
    best = choose_best_mph_pump(MPH_PUMPS, Q_req, dp_req, gvf_free_pct, dp_margin=DP_MARGIN)

    # ---------------------------
    # PLOTS: 1) Sättigung diagonal  2) Δp-Kennlinien  3) Overlay wie Vorlage
    # ---------------------------
    st.markdown("### 📈 Plots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Sättigung (diagonal) vs Druck
    if show_temp_band:
        temp_variants = [temperature - 10.0, temperature, temperature + 10.0]
        temp_variants = [t for t in temp_variants if -10.0 <= t <= 150.0]
    else:
        temp_variants = [temperature]

    p_max_plot = max(14.0, p_discharge * 1.2, 10.0)

    color_cycle = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "black"]
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure_cm3N(gas_medium, T, p_max=p_max_plot, y_gas=y_gas)
        ax1.plot(
            pressures, sol, "--", linewidth=2.2,
            color=color_cycle[i % len(color_cycle)],
            label=f"Sättigung {gas_medium} {T:.0f}°C (y={y_gas:.2f})"
        )

    # Saugpunkt markieren
    ax1.scatter([p_suction], [dissolved_suction_cm3N_L], s=180, marker="o",
                edgecolors="black", linewidths=2, zorder=5, label="Saugpunkt (Sättigung)")

    ax1.set_xlabel("Druck [bar abs]")
    ax1.set_ylabel("Sättigung / Löslichkeit [cm³N/L]")
    ax1.set_title("Sättigungskurven (Henry) – diagonal (Normvolumen)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, p_max_plot)

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
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.45
            ax2.plot(Q_lmin, curve["p"], "o-", linewidth=lw, alpha=alpha,
                     color=gvf_colors.get(gvf_key, "gray"),
                     label=f"{pump['id']} ({gvf_key}% GVF)")
        # Betriebspunkt
        ax2.scatter([Q_req_lmin], [dp_req], s=180, marker="o",
                    edgecolors="black", linewidths=2, zorder=5, label="Betriebspunkt (Δp_req)")

        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("Δp [bar]")
        ax2.set_title(f"Mehrphasen-Kennlinien (Δp): {pump['id']}")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.25))
        ax2.set_ylim(0, pump["p_max_bar"] * 1.15)
    else:
        ax2.text(0.5, 0.5, "❌ Keine geeignete Pumpe gefunden",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel("Q [L/min]")
        ax2.set_ylabel("Δp [bar]")
        ax2.set_title("Mehrphasen-Kennlinien")
        ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Plot 3: Overlay wie Vorlage (Sättigung diagonal + Gasvolumen pro Liter bei GVF)
    st.markdown("### 🧩 Overlay: Sättigung (diagonal) + Gasvolumen pro Liter Flüssigkeit (bei GVF)")

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Sättigungslinien (diagonal)
    for i, T in enumerate(temp_variants):
        pressures, sol = solubility_curve_vs_pressure_cm3N(gas_medium, T, p_max=p_max_plot, y_gas=y_gas)
        ax3.plot(
            pressures, sol,
            linestyle="--", linewidth=2.0,
            color=color_cycle[i % len(color_cycle)],
            label=f"Sättigung {gas_medium} {T:.0f}°C"
        )

    # GVF-Gasvolumenlinien (reales Gasvolumen am jeweiligen Druck, ausgehend vom Saugpunkt)
    overlay_gvfs = [10, 15, 20]
    pressures_overlay = linspace(max(0.5, 0.05 * p_max_plot), p_max_plot, 200)
    for gvf_pct in overlay_gvfs:
        Vgas_cm3_per_L = gas_volume_curve_from_gvf_at_suction(gvf_pct, p_suction, pressures_overlay)
        ax3.plot(
            pressures_overlay, Vgas_cm3_per_L,
            linewidth=2.2,
            label=f"{gvf_pct}% GVF (Gasvolumen/L bei p)"
        )

    # Markiere Saug- und Druckseite
    ax3.axvline(p_suction, linestyle=":", linewidth=2.0)
    ax3.axvline(p_discharge, linestyle=":", linewidth=2.0)

    ax3.scatter([p_suction], [dissolved_suction_cm3N_L], s=140,
                edgecolors="black", linewidths=2, zorder=6, label="Saugpunkt (Sättigung)")

    ax3.set_xlabel("Druck [bar abs]")
    ax3.set_ylabel("Sättigung [cm³N/L]  /  Gasvolumen [cm³/L] (bei p)")
    ax3.set_title("Overlay: Löslichkeit vs. Gasvolumen (wie Vorgabe)")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=9, ncol=2)
    ax3.set_xlim(0, p_max_plot)

    st.pyplot(fig3, clear_figure=True)

    # ---------------------------
    # Ergebnisse
    # ---------------------------
    st.divider()
    st.markdown("### ✅ Ergebnisse (Mehrphasen, normlogisch)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q", f"{Q_req:.1f} m³/h", f"{Q_req_lmin:.1f} L/min")
    c2.metric("Δp_req", f"{dp_req:.2f} bar", f"p_d={p_discharge:.2f} | p_s={p_suction:.2f}")
    c3.metric("GVF_in (Saug)", f"{gvf_in:.0f} %")
    c4.metric("Sättigung am Saugpunkt", f"{dissolved_suction_cm3N_L:.1f} cm³N/L")
    c5.metric("GVF_free (Saug)", f"{gvf_free_pct:.1f} %")

    if gvf_free_pct > 0.0:
        st.warning("⚠️ Freies Gas vorhanden (GVF_free > 0). Pumpenkennlinien werden auf freien GVF bezogen (Worst Case).")
    else:
        st.info("ℹ️ Nach dieser Abschätzung liegt kein freies Gas am Saugpunkt vor (alles im Löslichkeitslimit).")

    if best:
        st.markdown("### 🔧 Empfohlene Pumpe")
        st.success(
            f"**{best['pump']['id']}** | Kurve: **{best['gvf_curve']}% GVF** | Modus: **{best['mode']}**"
        )

        a, b, c, d = st.columns(4)
        a.metric("Δp verfügbar", f"{best['dp_available']:.2f} bar", f"Reserve: {best['dp_reserve']:.2f} bar")
        b.metric("Leistung", f"{best['P_required']:.2f} kW")
        c.metric("Drehzahl n/n0", f"{best['n_ratio']:.3f}", f"{best['n_ratio']*100:.1f}%")
        d.metric("GVF_max Pumpe", f"{best['pump']['GVF_max']*100:.0f}%")

    else:
        st.error("❌ Keine geeignete Mehrphasenpumpe gefunden.")
        st.markdown("""
**Typische Gründe:**
- Δp_req (inkl. Reserve) zu hoch für alle Kennlinien
- Q zu hoch für Pumpenbereich
- GVF_free über Pumpengrenze
        """)

    # ---------------------------
    # Ausführlicher Rechenweg (Mehrphasen)
    # ---------------------------
    with st.expander("📘 Rechenweg (ausführlich): Sättigung, freier GVF, Δp-Auswahl & Drehzahl", expanded=False):
        dp_target = dp_req * (1.0 + DP_MARGIN)
        st.markdown(f"""
### 1) Eingaben
- Gas: **{gas_medium}**
- Temperatur: **T = {temperature:.1f} °C**
- Partialdruckfaktor: **y_gas = {y_gas:.2f}**
- Saugdruck: **p_s = {p_suction:.2f} bar abs**
- Druckseite: **p_d = {p_discharge:.2f} bar abs**
- Volumenstrom: **Q = {Q_req:.2f} m³/h** ({Q_req_lmin:.1f} L/min)
- Gesamt-GVF am Saugpunkt: **GVF_in = {gvf_in:.0f}%**

Auslegung auf Differenzdruck:
- **Δp_req = p_d − p_s = {p_discharge:.2f} − {p_suction:.2f} = {dp_req:.2f} bar**
- Engineering-Reserve (Margin {DP_MARGIN*100:.0f}%):
  - **Δp_target = Δp_req·(1+Margin) = {dp_req:.2f}·{(1+DP_MARGIN):.2f} = {dp_target:.2f} bar**

---

### 2) Sättigung/Löslichkeit (Henry) – bewusst als Normvolumen (cm³N/L) → diagonal
Henry:
""")
        st.latex(r"C=\frac{p_{\mathrm{partial}}}{H(T)}")
        st.markdown(f"""
- Partialdruck: **p_partial = y_gas · p**
- Henry-Konstante (temperaturabhängig): **H(T)**

Umrechnung in **Normvolumen**:
""")
        st.latex(r"V_{N} = C \cdot V_{m,N}")
        st.markdown(f"""
mit **V_m,N = R·T_N/P_N = {V_MOLAR_N_L_PER_MOL:.3f} L/mol** bei (T_N=0°C, P_N=1 bar).

Damit:
- **Sättigung am Saugpunkt = {dissolved_suction_cm3N_L:.2f} cm³N/L**

Interpretation:
- Das ist die maximale Gasmenge (als Normvolumen), die bei p_s und T als *gelöst* angenommen werden kann.
- Alles darüber ist in dieser Näherung **freies Gas**.

---

### 3) Umrechnung GVF_in → Gasvolumen am Saugpunkt
Definition:
""")
        st.latex(r"GVF=\frac{V_{gas}}{V_{gas}+V_{liq}} \;\;\Rightarrow\;\; \frac{V_{gas}}{V_{liq}}=\frac{GVF}{1-GVF}")
        st.markdown(f"""
Für **GVF_in = {gvf_in:.0f}%** ergibt sich am Saugpunkt:
- **V_gas,in / V_liq = {gvf_dbg["Vgas_in_L_per_L"]:.4f} L/L** (reales Volumen bei p_s)

Um vergleichbar zur Sättigung zu sein, rechnen wir in Normvolumen um (ideales Gas):
""")
        st.latex(r"V_{gas,N} = V_{gas} \cdot \frac{p_s}{P_N}\cdot \frac{T_N}{T}")
        st.markdown(f"""
Numerisch:
- **V_gas,N = {gvf_dbg["VgasN_cm3N_per_L"]:.2f} cm³N/L**

---

### 4) Freies Gas am Saugpunkt
- Gelöst max.: **V_diss,N = {dissolved_suction_cm3N_L:.2f} cm³N/L**
- Total (aus GVF_in): **V_gas,N = {gvf_dbg["VgasN_cm3N_per_L"]:.2f} cm³N/L**

Freies Gas (Normvolumen):
- **V_free,N = max(0, V_gas,N − V_diss,N) = {gvf_dbg["VfreeN_cm3N_per_L"]:.2f} cm³N/L**

Zurück in reales Gasvolumen am Saugpunkt:
""")
        st.latex(r"V_{free} = V_{free,N}\cdot \frac{P_N}{p_s}\cdot \frac{T}{T_N}")
        st.markdown(f"""
- **V_free = {gvf_dbg["Vfree_L_per_L"]:.4f} L/L**

Freier GVF:
- **GVF_free = V_free/(V_free+1) = {gvf_free_pct:.2f}%**

---

### 5) Kennlinienwahl (Worst Case)
- Wir wählen die **nächsthöhere** verfügbare GVF-Kurve **≥ GVF_free**:
  - Beispiel: GVF_free=8.7% ⇒ Kennlinie **10%** (Worst Case)

Damit wird die Pumpenauswahl konservativer, ohne dass du manuell Sicherheitsaufschläge rechnen musst.

---

### 6) Δp-Abgleich und Drehzahl (Affinitätsgesetze)
Für Kreiselpumpen gilt näherungsweise:
""")
        st.latex(r"Q \sim n,\quad \Delta p \sim n^2,\quad P \sim n^3")
        st.markdown(f"""
**Fall A: Nenndrehzahl**
- Prüfe Δp(Q) auf der gewählten GVF-Kurve
- Bedingung: **Δp_avail ≥ Δp_target** (inkl. Reserve)

**Fall B: Drehzahlanpassung**
Wir suchen n_ratio im Bereich [{N_RATIO_MIN:.2f}, {N_RATIO_MAX:.2f}] so, dass:
- **Δp_scaled(Q_req) ≈ Δp_target**

Skalierung:
- Q_base = Q_req / n_ratio  (Punkt auf Basiskennlinie)
- Δp_scaled = Δp_base(Q_base) · n_ratio²
- P_scaled = P_base(Q_base) · n_ratio³

Die Lösung wird per **Bisektion** gefunden (robust für monotone Kennlinien).

---

### 7) Norm-/Standardbezug (Hinweis)
- Auslegung und Terminologie (Δp, Duty Point, Margin) ist üblich in Engineering-Standards (z. B. **API 610 / ISO 13709** als Referenzrahmen im Öl/Gas-Umfeld).
- Zweiphasen-/Mehrphasenkennlinien sind **herstellerspezifisch**; dieses Tool wählt konservativ über Worst-Case-GVF und Δp-Margen.

> Für verbindliche Auslegung: Herstellerkurven, NPSH-Checks, Gas-/Flüssigkeitseigenschaften und Betriebsbereiche detailliert berücksichtigen.
""")

    # verfügbare Mehrphasenpumpen
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
            t_margin = 15.0  # konservativer Abstand

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

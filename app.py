import math
import warnings
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="Pumpenauslegung", layout="wide", page_icon="🔧")

# =========================
# Debug
# =========================
DEBUG = True

def show_error(e: Exception, where: str = ""):
    st.error(f"❌ Fehler {('in ' + where) if where else ''}: {e}")
    if DEBUG:
        import traceback
        st.code(traceback.format_exc())

# =========================
# Konstanten
# =========================
G = 9.80665
R_BAR_L = 0.08314462618
P_N_BAR = 1.01325
T_N_K = 273.15
N0_RPM_DEFAULT = 2900
BAR_TO_PA = 1e5

# Mehrphase: feste Saugseite (Unterdruck, damit Luft eingesogen werden kann)
P_SUCTION_FIXED_BAR_ABS = 0.80  # bar(abs) -> Unterdruck gegenüber 1.013 bar

# Luft-Zusammensetzung (vereinfachte Hauptanteile)
AIR_COMPONENTS = [
    ("Stickstoff aus Gasgemisch", 0.79),
    ("Sauerstoff aus Gasgemisch", 0.21),
]

# =========================
# Medien / Gase
# =========================
MEDIA = {
    "Wasser (20°C)": {"rho": 998.0, "nu": 1.0, "p_vapor": 0.0234},
    "Wasser (60°C)": {"rho": 983.0, "nu": 0.47, "p_vapor": 0.1992},
    "Glykol 30% (20°C)": {"rho": 1040.0, "nu": 3.5, "p_vapor": 0.01},
    "Hydrauliköl ISO VG 32 (40°C)": {"rho": 860.0, "nu": 32.0, "p_vapor": 1e-5},
    "Rohöl (API 30)": {"rho": 876.0, "nu": 10.0, "p_vapor": 0.05},
}

HENRY_CONSTANTS = {
    "Luft": {"A": 1300.0, "B": 1300, "MW": 28.97},
    "Methan (CH4)": {"A": 1400.0, "B": 1600, "MW": 16.04},
    "Ethan (C2H6)": {"A": 800.0, "B": 1800, "MW": 30.07},
    "Propan (C3H8)": {"A": 500.0, "B": 2000, "MW": 44.10},
    "CO2": {"A": 29.4, "B": 2400, "MW": 44.01},
    "H2S": {"A": 10.0, "B": 2100, "MW": 34.08},
    # Pseudo-Einträge für Luftkomponenten aus Gemisch (nur für Plot/Logik)
    "Stickstoff aus Gasgemisch": {"A": 1600.0, "B": 1300, "MW": 28.0},
    "Sauerstoff aus Gasgemisch": {"A": 1200.0, "B": 1300, "MW": 32.0},
}

# sehr einfache Z-Approximation (nur Stabilität/Trend, kein EOS-Anspruch)
REAL_GAS_FACTORS = {
    "Luft": lambda p, T: max(0.85, 1.0 - 0.00008 * p),
    "Methan (CH4)": lambda p, T: max(0.80, 1.0 - 0.00015 * p),
    "CO2": lambda p, T: max(0.70, 0.90 + 0.00006 * (T - 273.15)),
    # Gemischkomponenten -> 1.0 als robuste Näherung
    "Stickstoff aus Gasgemisch": lambda p, T: 1.0,
    "Sauerstoff aus Gasgemisch": lambda p, T: 1.0,
}

# =========================
# Pumpendaten
# =========================
PUMPS = [
    {
        "id": "P1",
        "Qw": [0, 10, 20, 30, 40, 50],
        "Hw": [30, 29, 27, 24, 20, 15],
        "eta": [0.35, 0.55, 0.65, 0.62, 0.55, 0.45],
        "Pw": [1.2, 2.8, 4.2, 5.5, 6.5, 7.2],  # Achtung: kann von H/η abweichen (Mess-/Motorwerte etc.)
        "NPSHr": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "max_viscosity": 500,
        "max_density": 1200,
    },
    {
        "id": "P2",
        "Qw": [0, 8, 16, 24, 32, 40],
        "Hw": [36, 35, 33, 30, 26, 20],
        "eta": [0.30, 0.48, 0.60, 0.63, 0.58, 0.50],
        "Pw": [1.4, 2.7, 3.9, 5.0, 6.1, 7.4],
        "NPSHr": [1.1, 1.3, 1.6, 2.1, 2.7, 3.2],
        "max_viscosity": 600,
        "max_density": 1200,
    },
    {
        "id": "P3",
        "Qw": [0, 12, 24, 36, 48, 60],
        "Hw": [26, 25, 23, 20, 16, 12],
        "eta": [0.33, 0.52, 0.64, 0.66, 0.60, 0.48],
        "Pw": [1.1, 2.6, 4.0, 5.6, 7.0, 8.6],
        "NPSHr": [0.9, 1.1, 1.4, 1.8, 2.3, 2.9],
        "max_viscosity": 450,
        "max_density": 1200,
    },
]

MPH_PUMPS = [
    {
        "id": "MPH-40",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 40,
        "dp_max_bar": 12,
        "GVF_max": 0.4,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 2.5,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [11.2, 11.0, 10.6, 10.0, 9.2, 7.6, 6.0]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [10.5, 10.2, 9.7, 9.0, 8.2, 6.6, 5.1]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [9.1, 8.8, 8.2, 7.4, 6.6, 5.0, 3.9]},
            30: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [7.5, 7.2, 6.8, 6.2, 5.5, 4.2, 3.2]},
            40: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [5.5, 5.3, 5.0, 4.6, 4.0, 3.0, 2.2]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [3.0, 3.4, 3.9, 4.5, 5.1, 6.2, 7.0]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.8, 3.2, 3.6, 4.1, 4.7, 5.7, 6.4]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.5, 2.8, 3.2, 3.6, 4.0, 4.8, 5.4]},
            30: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.2, 2.5, 2.8, 3.2, 3.5, 4.2, 4.8]},
            40: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [1.8, 2.0, 2.3, 2.6, 2.9, 3.5, 4.0]},
        },
    },
    {
        "id": "MPH-25",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "dp_max_bar": 14,
        "GVF_max": 0.35,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 2.0,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 4, 8, 12, 16, 20, 25], "dp": [13.5, 13.2, 12.6, 11.6, 10.4, 9.0, 7.5]},
            10: {"Q": [0, 4, 8, 12, 16, 20, 25], "dp": [12.7, 12.3, 11.6, 10.6, 9.4, 8.1, 6.8]},
            20: {"Q": [0, 4, 8, 12, 16, 20, 25], "dp": [11.0, 10.6, 9.9, 9.0, 8.0, 6.8, 5.6]},
            30: {"Q": [0, 4, 8, 12, 16, 20, 25], "dp": [9.1, 8.8, 8.2, 7.4, 6.6, 5.6, 4.7]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 4, 8, 12, 16, 20, 25], "P": [2.2, 2.5, 2.9, 3.4, 3.9, 4.5, 5.1]},
            10: {"Q": [0, 4, 8, 12, 16, 20, 25], "P": [2.0, 2.3, 2.7, 3.1, 3.6, 4.1, 4.6]},
            20: {"Q": [0, 4, 8, 12, 16, 20, 25], "P": [1.8, 2.1, 2.4, 2.8, 3.2, 3.6, 4.1]},
            30: {"Q": [0, 4, 8, 12, 16, 20, 25], "P": [1.6, 1.9, 2.2, 2.5, 2.9, 3.3, 3.7]},
        },
    },
    {
        "id": "MPH-60",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 60,
        "dp_max_bar": 9,
        "GVF_max": 0.50,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 3.0,
        "curves_dp_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [9.0, 8.8, 8.4, 7.8, 7.0, 6.1, 5.2]},
            10: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [8.4, 8.1, 7.7, 7.1, 6.3, 5.5, 4.7]},
            20: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [7.4, 7.1, 6.7, 6.1, 5.5, 4.8, 4.1]},
            30: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [6.1, 5.9, 5.6, 5.1, 4.6, 4.0, 3.4]},
            40: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [4.6, 4.4, 4.2, 3.9, 3.5, 3.1, 2.6]},
            50: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [3.5, 3.4, 3.2, 3.0, 2.7, 2.4, 2.0]},
        },
        "power_kW_vs_Q": {
            0:  {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [3.5, 3.9, 4.4, 5.0, 5.8, 6.6, 7.4]},
            10: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [3.2, 3.6, 4.0, 4.6, 5.3, 6.0, 6.7]},
            20: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [2.9, 3.2, 3.6, 4.1, 4.7, 5.3, 5.9]},
            30: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [2.5, 2.8, 3.1, 3.6, 4.1, 4.7, 5.2]},
            40: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [2.1, 2.4, 2.7, 3.1, 3.6, 4.1, 4.6]},
            50: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [1.8, 2.0, 2.3, 2.7, 3.1, 3.5, 3.9]},
        },
    },
]

ATEX_MOTORS = [
    {
        "id": "Standard Zone 2 (Ex ec)",
        "marking": "II 3G Ex ec IIC T3 Gc",
        "zone_suitable": [2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "3G",
        "efficiency_class": "IE3",
    },
]

# =========================
# Utilities
# =========================
def safe_clamp(x, a, b):
    try:
        return max(a, min(b, x))
    except Exception:
        return a

def safe_interp(x, xp, fp):
    try:
        xp = list(xp)
        fp = list(fp)
        if len(xp) != len(fp) or len(xp) < 2:
            return fp[0] if fp else 0.0
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                if xp[i + 1] == xp[i]:
                    return fp[i]
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])
        return fp[-1]
    except Exception:
        return fp[-1] if fp else 0.0

def m3h_to_lmin(m3h):
    return float(m3h) * 1000.0 / 60.0

def gas_flow_required_norm_lmin(Q_liq_m3h, C_target_cm3N_L):
    """
    Ziel-Gasmenge (Ncm³/L) -> Gasvolumenstrom bei Normbedingungen [L/min].
    """
    return float(Q_liq_m3h) * float(C_target_cm3N_L) / 60.0

def oper_to_norm_ratio(p_bar_abs, T_celsius, gas):
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p_bar_abs, T_celsius), 0.5)
    return (float(p_bar_abs) / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)

def gas_flow_oper_lmin_from_gvf(Q_liq_m3h, gvf_pct):
    gvf_frac = safe_clamp(float(gvf_pct) / 100.0, 0.0, 0.99)
    Q_gas_m3h = float(Q_liq_m3h) * (gvf_frac / max(1.0 - gvf_frac, 1e-9))
    return m3h_to_lmin(Q_gas_m3h)

def cm3N_L_from_gvf_pct_at_suction(gvf_pct, p_suction_bar_abs, T_celsius, gas):
    gvf_frac = safe_clamp(float(gvf_pct) / 100.0, 0.0, 0.99)
    Vgas_oper_L_per_L = gvf_frac / max(1.0 - gvf_frac, 1e-9)
    ratio = oper_to_norm_ratio(p_suction_bar_abs, T_celsius, gas)
    Vn_L_per_L = Vgas_oper_L_per_L * ratio
    return Vn_L_per_L * 1000.0

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================
# Wasser-Guard (wichtig!)
# =========================
WATER_NU_CST = 1.0
WATER_EPS = 0.15  # ~15% Toleranz, damit "Wasser" nicht fälschlich korrigiert wird

def is_effectively_water(nu_cSt: float) -> bool:
    return float(nu_cSt) <= (WATER_NU_CST + WATER_EPS)

# =========================
# HI Viskosität (robust)
# =========================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    Q = max(float(Q_m3h), 1e-6)
    H = max(float(H_m), 1e-6)
    nu = max(float(nu_cSt), 1e-6)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    if B <= 1.0:
        return 1.0, 1.0
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = safe_clamp(CH, 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = safe_clamp(Ceta, 0.1, 1.0)
    return CH, Ceta

def viscous_to_water_point(Q_vis_m3h, H_vis_m, nu_cSt):
    """
    Rückrechnung von viskosem Betriebspunkt auf äquivalenten Wasser-Betriebspunkt.
    H_vis = H_water \cdot CH  =>  H_water = H_vis / CH
    Bei höherer Viskosität ist H_vis < H_water, daher ist CH < 1.
    Wichtig: Bei "Wasser" darf KEINE HI-Korrektur aktiv werden, sonst bekommst du
    unterschiedliche Q-P/Q-H-Kurven für Wasser vs. viskos.
    """
    B = compute_B_HI(Q_vis_m3h, H_vis_m, nu_cSt)

    if is_effectively_water(nu_cSt):
        CH, Ceta = 1.0, 1.0
    else:
        CH, Ceta = viscosity_correction_factors(B)

    Q_water = float(Q_vis_m3h)  # CQ ~ 1 (stabil)
    # KORREKTUR: H_water = H_vis / CH, da H_vis = H_water * CH
    H_water = float(H_vis_m) / max(CH, 1e-9)
    return {"Q_water": Q_water, "H_water": H_water, "B": B, "CH": CH, "Ceta": Ceta}

def generate_viscous_curve(pump, nu_cSt, rho, use_consistent_power=True):
    """
    Generiert viskose Kennlinien aus Wasserkennlinien.
    
    Option A (use_consistent_power=True, empfohlen):
    - Konsistente Berechnung: P wird direkt aus H_vis und η_vis berechnet
    - P_vis = (ρ g Q H_vis) / η_vis
    
    Option B (use_consistent_power=False, legacy):
    - Wasser-Leistungskurve basiert auf Pw-Datensatz (realistisch inkl. Verluste)
    - Viskose Leistung wird als Skalierung von Pw abgeleitet (damit Offsets/Verluste erhalten bleiben)
    - H_vis und eta_vis werden wie bisher über (CH, Ceta) korrigiert
    """
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    etaw = np.array(pump["eta"], dtype=float)
    Pw_ref = np.array(pump["Pw"], dtype=float)  # kW, Referenz (Wasser)

    H_vis, eta_vis, P_vis = [], [], []

    for i, (q, h, e) in enumerate(zip(Qw, Hw, etaw)):
        # HI-Korrektur (bei Wasser -> CH=Ceta=1, wenn du Water-Guard nutzt)
        B = compute_B_HI(q if q > 0 else 1e-6, max(h, 1e-6), nu_cSt)
        if is_effectively_water(nu_cSt):
            CH, Ceta = 1.0, 1.0
        else:
            CH, Ceta = viscosity_correction_factors(B)

        # KORREKTUR: Bei höherer Viskosität SINKT die Förderhöhe -> Multiplikation mit CH!
        hv = float(h) * max(float(CH), 1e-9)
        ev = safe_clamp(float(e) * float(Ceta), 0.05, 0.95)

        if use_consistent_power:
            # Konsistente Berechnung: P_vis direkt aus H_vis und η_vis
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            pv = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0
        else:
            # Legacy: Skalierung von Pw_ref (bewahrt Verluste/Offsets)
            # Theoretische Wasserleistung aus H&η nur für Skalierungsfaktor
            P_hyd_water_W = rho * G * (float(q) / 3600.0) * float(h)
            P_water_theory = (P_hyd_water_W / max(float(e), 1e-9)) / 1000.0
            
            # theoretische viskose Leistung (nur für Skalierungsfaktor)
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            P_vis_theory = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0

            # Skalierung der realistischen Wasser-Pw auf "viskos"
            if P_water_theory > 1e-6:
                scale = float(P_vis_theory) / float(P_water_theory)
            else:
                # Bei Q≈0 ist Theorie unbrauchbar -> konservativ nur über Effizienzfaktor skalieren
                scale = 1.0 / max(float(Ceta), 1e-9)

            pv = float(Pw_ref[i]) * float(scale)

        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)

    return Qw.tolist(), H_vis, eta_vis, P_vis

def water_power_curve_from_H_eta(pump, rho):
    """
    Konsistente Wasser-P-Kurve aus H(Q) und η(Q).
    Damit ist Q-P mathematisch konsistent und nicht abhängig von pump["Pw"].
    """
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    etaw = np.array(pump["eta"], dtype=float)

    P = []
    for q, h, e in zip(Qw, Hw, etaw):
        P_hyd = rho * G * (q / 3600.0) * h
        P.append((P_hyd / max(e, 1e-9)) / 1000.0)
    return Qw.tolist(), P

# =========================
# Root / Drehzahl
# =========================
def bisect_root(f, a, b, it=70, tol=1e-6):
    fa = f(a)
    fb = f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        return None
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if abs(fm) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)

def find_best_ratio_by_scan(dp_at_ratio_fn, dp_req, n_min, n_max, steps=60, prefer_above=True):
    dp_req = float(dp_req)
    n_min = float(n_min)
    n_max = float(n_max)
    if n_max <= n_min:
        return None

    values = []
    for i in range(steps + 1):
        nr = n_min + (n_max - n_min) * (i / steps)
        dp = dp_at_ratio_fn(nr)
        if np.isfinite(dp):
            values.append((nr, dp))

    if not values:
        return None

    if prefer_above:
        above = [(nr, dp) for nr, dp in values if dp >= dp_req]
        if above:
            return min(above, key=lambda x: x[1] - dp_req)[0]

    return min(values, key=lambda x: abs(x[1] - dp_req))[0]

def find_speed_ratio(Q_curve, H_curve, Q_req, H_req, n_min=0.5, n_max=1.2):
    Q_curve = list(map(float, Q_curve))
    H_curve = list(map(float, H_curve))
    Q_req = float(Q_req)
    H_req = float(H_req)

    def f(nr):
        if nr <= 0:
            return 1e9
        Q_base = Q_req / nr
        H_base = safe_interp(Q_base, Q_curve, H_curve)
        return (H_base * (nr ** 2)) - H_req

    return bisect_root(f, float(n_min), float(n_max), it=80, tol=1e-5)

# =========================
# Gas / Löslichkeit / GVF
# =========================
def henry_constant(gas, T_celsius):
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K = float(T_celsius) + 273.15
    T0_K = 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def real_gas_factor(gas, p_bar, T_celsius):
    T_K = float(T_celsius) + 273.15
    if gas in REAL_GAS_FACTORS:
        return float(REAL_GAS_FACTORS[gas](float(p_bar), T_K))
    return 1.0

def gas_solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Henry-basiert, Ausgabe: cm³N/L (Normvolumen pro Liter Flüssigkeit)
    Hinweis: Vereinfachtes Modell (Trend/Robustheit).
    """
    p = max(float(p_bar_abs), 1e-6)
    T_K = float(T_celsius) + 273.15
    H = max(henry_constant(gas, T_celsius), 1e-12)
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)
    p_partial = safe_clamp(float(y_gas), 0.0, 1.0) * p

    # Henry: C = p/H (mol/L)
    C_mol_L = p_partial / H

    # Gasvolumen operativ (L/L) bei p,T
    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    # oper -> normal
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L

def free_gas_gvf_pct_at_suction_from_cm3N_L(free_cm3N_L, p_suction_bar_abs, T_celsius, gas):
    """
    Freies Gas (Norm cm³N/L) -> GVF% (operativ) an Saugseite
    """
    free_cm3N_L = max(float(free_cm3N_L), 0.0)
    p = max(float(p_suction_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vn_L_per_L = free_cm3N_L / 1000.0
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)  # oper -> normal
    Vgas_oper_L_per_L = Vn_L_per_L / max(ratio, 1e-12)

    gvf = (Vgas_oper_L_per_L / (1.0 + Vgas_oper_L_per_L)) * 100.0
    return safe_clamp(gvf, 0.0, 99.0)

def solubility_diagonal_curve(gas, T_celsius, y_gas=1.0, p_min=0.2, p_max=14.0, n=140):
    ps = np.linspace(p_min, p_max, n)
    sol = [gas_solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) for p in ps]
    return ps, np.array(sol)

def pressure_required_for_C_target(gas, T_celsius, C_target_cm3N_L, y_gas=1.0, p_min=0.2, p_max=200.0):
    """
    Finde p_abs, so dass C_sat(p,T) ≈ C_target (bisection).
    Gibt None zurück, wenn Ziel außerhalb des Suchbereichs liegt.
    """
    C_target = float(C_target_cm3N_L)
    if C_target <= 0:
        return p_min

    def f(p):
        return gas_solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) - C_target

    a, b = float(p_min), float(p_max)
    fa, fb = f(a), f(b)

    if fa >= 0:
        return a
    if fb < 0:
        return None

    lo, hi = a, b
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if abs(fm) < 1e-3:
            return mid
        if fm >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def pressure_required_for_air_components(T_celsius, C_total_cm3N_L, p_min=0.2, p_max=200.0):
    """
    Luft wird als N2/O2 betrachtet. Ziel ist vollständiges Lösen ALLER Komponenten.
    Dazu wird je Komponente der erforderliche Druck berechnet und p_req = max(p_req_i) genommen.
    """
    targets = {}
    p_reqs = []

    for gas_i, y in AIR_COMPONENTS:
        C_i = float(C_total_cm3N_L) * float(y)
        targets[gas_i] = C_i
        p_i = pressure_required_for_C_target(gas_i, T_celsius, C_i, y_gas=y, p_min=p_min, p_max=p_max)
        if p_i is None:
            return None, targets
        p_reqs.append(p_i)

    return max(p_reqs) if p_reqs else None, targets

# =========================
# Pump selection (Einphase)
# =========================
def choose_best_pump(pumps, Q_req, H_req, nu_cSt, rho, allow_out_of_range=True):
    best = None
    for p in pumps:
        try:
            if nu_cSt > p.get("max_viscosity", 500):
                continue
            if rho > p.get("max_density", 1200):
                continue

            qmin, qmax = min(p["Qw"]), max(p["Qw"])
            in_range = (qmin <= Q_req <= qmax)
            if (not in_range) and (not allow_out_of_range):
                continue

            Q_eval = safe_clamp(Q_req, qmin, qmax)
            H_at = safe_interp(Q_eval, p["Qw"], p["Hw"])
            eta_at = safe_interp(Q_eval, p["Qw"], p["eta"])

            score = abs(H_at - H_req)
            penalty = 0.0 if in_range else abs(Q_req - Q_eval) / max((qmax - qmin), 1e-9) * 10.0

            cand = {
                "id": p["id"],
                "pump": p,
                "in_range": in_range,
                "Q_eval": Q_eval,
                "H_at": H_at,
                "eta_at": eta_at,
                "score": score + penalty
            }
            if best is None or cand["score"] < best["score"]:
                best = cand
        except Exception:
            continue
    return best

# =========================
# Mehrphase: Interpolation über GVF-Kurven + Drehzahl
# =========================
def _interp_between_gvf_keys(pump, gvf_pct):
    keys = sorted(pump["curves_dp_vs_Q"].keys())
    if gvf_pct <= keys[0]:
        return keys[0], keys[0], 0.0
    if gvf_pct >= keys[-1]:
        return keys[-1], keys[-1], 0.0
    lo = max([k for k in keys if k <= gvf_pct])
    hi = min([k for k in keys if k >= gvf_pct])
    if hi == lo:
        return lo, hi, 0.0
    w = (gvf_pct - lo) / (hi - lo)
    return lo, hi, w

def nearest_gvf_key(pump, gvf_pct):
    keys = sorted(pump["curves_dp_vs_Q"].keys())
    if not keys:
        return gvf_pct
    return min(keys, key=lambda k: abs(k - gvf_pct))

def _dp_at_Q_gvf(pump, Q_m3h, gvf_pct):
    lo, hi, w = _interp_between_gvf_keys(pump, gvf_pct)
    c_lo = pump["curves_dp_vs_Q"][lo]
    c_hi = pump["curves_dp_vs_Q"][hi]
    dp_lo = safe_interp(Q_m3h, c_lo["Q"], c_lo["dp"])
    dp_hi = safe_interp(Q_m3h, c_hi["Q"], c_hi["dp"])
    return (1 - w) * dp_lo + w * dp_hi, lo, hi, w

def _P_at_Q_gvf(pump, Q_m3h, gvf_pct):
    lo, hi, w = _interp_between_gvf_keys(pump, gvf_pct)
    p_lo = pump["power_kW_vs_Q"][lo]
    p_hi = pump["power_kW_vs_Q"][hi]
    P_lo = safe_interp(Q_m3h, p_lo["Q"], p_lo["P"])
    P_hi = safe_interp(Q_m3h, p_hi["Q"], p_hi["P"])
    return (1 - w) * P_lo + w * P_hi, lo, hi, w

def choose_best_mph_pump(pumps, Q_req_m3h, dp_req_bar, gvf_free_pct, nu_cSt, rho_liq,
                        n_min_ratio=0.5, n_max_ratio=1.2,
                        w_power=0.5, w_eta=0.3, w_gas=0.2,
                        C_target_cm3N_L=0.0, p_suction_bar_abs=1.0, T_celsius=20.0, gas_medium="Luft"):
    """
    Wählt beste Pumpe bei vorgegebenem Q und dp.
    Interpolation zwischen GVF-Kurven (auch 8/9/11% möglich).
    """
    best = None

    Q_req = float(Q_req_m3h)
    dp_req = float(dp_req_bar)
    gvf_free_pct = float(gvf_free_pct)

    for pump in pumps:
        try:
            if gvf_free_pct > pump["GVF_max"] * 100.0:
                continue
            if nu_cSt > pump.get("max_viscosity", 500):
                continue
            if rho_liq > pump.get("max_density", 1200):
                continue

            dp_nom, _, _, _ = _dp_at_Q_gvf(pump, Q_req, gvf_free_pct)

            def dp_at_ratio(nr):
                if nr <= 0:
                    return 0.0
                Q_base = Q_req / nr
                dp_base, _, _, _ = _dp_at_Q_gvf(pump, Q_base, gvf_free_pct)
                return dp_base * (nr ** 2)

            def f(nr):
                return dp_at_ratio(nr) - dp_req

            n_ratio = bisect_root(f, n_min_ratio, n_max_ratio, it=80, tol=1e-4)
            if n_ratio is None:
                n_ratio = find_best_ratio_by_scan(dp_at_ratio, dp_req, n_min_ratio, n_max_ratio, steps=60, prefer_above=True)

            candidates = []

            if dp_nom >= dp_req:
                P_nom, _, _, _ = _P_at_Q_gvf(pump, Q_req, gvf_free_pct)
                candidates.append({
                    "pump": pump,
                    "gvf_key": gvf_free_pct,
                    "dp_avail": dp_nom,
                    "P_req": P_nom,
                    "n_ratio": 1.0,
                    "n_rpm": pump["n0_rpm"],
                    "mode": "Nenndrehzahl",
                    "Q_m3h": Q_req,
                })

            if n_ratio is not None:
                Q_base = Q_req / n_ratio
                dp_scaled = dp_at_ratio(n_ratio)
                if dp_scaled >= dp_req:
                    P_base, _, _, _ = _P_at_Q_gvf(pump, Q_base, gvf_free_pct)
                    P_scaled = P_base * (n_ratio ** 3)
                    candidates.append({
                        "pump": pump,
                        "gvf_key": gvf_free_pct,
                        "dp_avail": dp_scaled,
                        "P_req": P_scaled,
                        "n_ratio": n_ratio,
                        "n_rpm": pump["n0_rpm"] * n_ratio,
                        "mode": "Drehzahl angepasst",
                        "Q_m3h": Q_req,
                    })

            ratio_norm = oper_to_norm_ratio(p_suction_bar_abs, T_celsius, gas_medium)
            Q_gas_req_norm_lmin = gas_flow_required_norm_lmin(Q_req, C_target_cm3N_L)

            for cand in candidates:
                dp_surplus = max(0.0, cand["dp_avail"] - dp_req)

                P_spec = cand["P_req"] / max(Q_req, 1e-6)
                P_hyd_kW = (cand["dp_avail"] * BAR_TO_PA) * (Q_req / 3600.0) / 1000.0
                eta_est = safe_clamp(P_hyd_kW / max(cand["P_req"], 1e-9), 0.0, 1.0)
                eta_term = 1.0 - eta_est

                Q_gas_oper_lmin = gas_flow_oper_lmin_from_gvf(Q_req, gvf_free_pct)
                Q_gas_norm_lmin = Q_gas_oper_lmin * ratio_norm
                gas_err = abs(Q_gas_norm_lmin - Q_gas_req_norm_lmin) / max(Q_gas_req_norm_lmin, 1e-6)

                score = (
                    float(w_power) * P_spec +
                    float(w_eta) * eta_term +
                    float(w_gas) * gas_err +
                    0.05 * dp_surplus +
                    0.10 * abs(cand["n_ratio"] - 1.0)
                )
                cand["score"] = score
                cand["eta_est"] = eta_est
                cand["gas_err"] = gas_err
                if best is None or score < best["score"]:
                    best = cand

        except Exception:
            continue

    return best

def choose_best_mph_pump_autoQ(pumps, dp_req_bar, gvf_free_pct, nu_cSt, rho_liq,
                              n_min_ratio=0.5, n_max_ratio=1.2,
                              w_power=0.5, w_eta=0.3, w_gas=0.2,
                              C_target_cm3N_L=0.0, p_suction_bar_abs=1.0, T_celsius=20.0, gas_medium="Luft"):
    """
    Q ist nicht Eingabe: es werden Kandidaten-Q aus Kennlinien geprüft.
    """
    best = None
    dp_req = float(dp_req_bar)
    gvf_free_pct = float(gvf_free_pct)

    for pump in pumps:
        try:
            if gvf_free_pct > pump["GVF_max"] * 100.0:
                continue
            if nu_cSt > pump.get("max_viscosity", 500):
                continue
            if rho_liq > pump.get("max_density", 1200):
                continue

            base_keys = sorted(pump["curves_dp_vs_Q"].keys())
            any_curve = pump["curves_dp_vs_Q"][base_keys[0]]
            Q_candidates = list(map(float, any_curve["Q"]))
            Q_candidates = [q for q in Q_candidates if q > 0]

            for Q_req in Q_candidates:
                cand = choose_best_mph_pump(
                    [pump], Q_req_m3h=Q_req, dp_req_bar=dp_req, gvf_free_pct=gvf_free_pct,
                    nu_cSt=nu_cSt, rho_liq=rho_liq, n_min_ratio=n_min_ratio, n_max_ratio=n_max_ratio,
                    w_power=w_power, w_eta=w_eta, w_gas=w_gas,
                    C_target_cm3N_L=C_target_cm3N_L, p_suction_bar_abs=p_suction_bar_abs,
                    T_celsius=T_celsius, gas_medium=gas_medium
                )
                if cand is None:
                    continue

                q_rel = Q_req / max(pump["Q_max_m3h"], 1e-9)
                edge_penalty = 0.0
                if q_rel < 0.2 or q_rel > 0.9:
                    edge_penalty = 0.8

                cand_score = cand["score"] + edge_penalty
                cand["score2"] = cand_score

                if best is None or cand_score < best["score2"]:
                    best = cand

        except Exception:
            continue

    return best

# =========================
# Pages
# =========================
def run_single_phase_pump():
    try:
        st.header("Einphasenpumpen mit Viskositätskorrektur")

        with st.expander("Eingaben – aufklappen", expanded=True):
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                st.subheader("Betriebspunkt (viskos)")
                Q_vis_req = st.number_input("Förderstrom Q_vis [m³/h]", min_value=0.1, value=30.0, step=1.0)
                H_vis_req = st.number_input("Förderhöhe H_vis [m]", min_value=0.1, value=20.0, step=1.0)
            with colB:
                st.subheader("Medium")
                medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
                rho = st.number_input("Dichte ρ [kg/m³]", min_value=1.0, value=float(MEDIA[medium]["rho"]), step=5.0)
                nu = st.number_input("Kinematische Viskosität ν [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.5)
            with colC:
                st.subheader("Optionen")
                allow_out = st.checkbox("Auswahl außerhalb Kennlinie zulassen", value=True)
                reserve_pct = st.slider("Motorreserve [%]", 0, 30, 10)
                n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
                n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)
                use_consistent_power = st.checkbox("Konsistente Leistungsberechnung (P aus H & η)", value=True,
                                                   help="Empfohlen: Berechnet P direkt aus H und η statt Skalierung von Pw")
        
        # NPSH / Kavitationsprüfung entfernt (auf Wunsch)

        conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
        Q_water = conv["Q_water"]
        H_water = conv["H_water"]
        B, CH, Ceta = conv["B"], conv["CH"], conv["Ceta"]

        best = choose_best_pump(PUMPS, Q_water, H_water, nu, rho, allow_out_of_range=allow_out)
        if not best:
            st.error("Keine geeignete Pumpe gefunden.")
            return

        pump = best["pump"]
        eta_water = float(best["eta_at"])

        eta_vis = safe_clamp(eta_water * Ceta, 0.05, 0.95)
        P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
        P_vis_kW = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
        P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

        Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(pump, nu, rho, use_consistent_power)
        
        # Wasser-Leistungskurve mit gleicher Methode berechnen (damit bei Wasser identisch mit viskos)
        if use_consistent_power:
            Q_water_curve, P_water_curve = water_power_curve_from_H_eta(pump, rho)
        else:
            Q_water_curve = pump["Qw"]
            P_water_curve = pump["Pw"]
        
        n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)

        n_opt_rpm = None
        P_opt_kW = None
        saving_pct = None
        P_throttle_kW = None  # Baseline ohne VFD (n0 + "Drossel"/Verlust)

        # Baseline ohne VFD: Pumpe bei n0 liefert bei Q_vis_req eine bestimmte Förderhöhe H_pump_n0
        # (wenn H_pump_n0 > H_vis_req, wird die "zu viel" Förderhöhe als Verlust vernichtet,
        # die Pumpe muss sie aber trotzdem aufbauen -> Leistung basiert auf H_pump_n0)
        H_pump_n0 = safe_interp(Q_vis_req, pump["Qw"], pump["Hw"])
        eta_pump_n0 = safe_interp(Q_vis_req, pump["Qw"], pump["eta"])

        # Viskositätswirkung auf η (bei Wasser ist Ceta=1 durch Wasser-Guard)
        eta_pump_n0_vis = safe_clamp(float(eta_pump_n0) * float(Ceta), 0.05, 0.95)

        P_hyd_throttle_W = rho * G * (Q_vis_req / 3600.0) * float(H_pump_n0)
        P_throttle_kW = (P_hyd_throttle_W / max(eta_pump_n0_vis, 1e-9)) / 1000.0

        # VFD-Fall: gleiche Anforderung Q_vis_req & H_vis_req, η am "Basis"-Punkt (Q_base) aus viskoser Kurve
        if n_ratio_opt is not None:
            n_opt_rpm = N0_RPM_DEFAULT * n_ratio_opt

            Q_base = Q_vis_req / n_ratio_opt
            eta_base = safe_interp(Q_base, Q_vis_curve, eta_vis_curve)
            eta_vfd = safe_clamp(float(eta_base), 0.05, 0.95)

            P_hyd_vfd_W = rho * G * (Q_vis_req / 3600.0) * float(H_vis_req)
            P_opt_kW = (P_hyd_vfd_W / max(eta_vfd, 1e-9)) / 1000.0

            saving_pct = ((P_throttle_kW - P_opt_kW) / P_throttle_kW * 100.0) if P_throttle_kW > 0 else 0.0

        st.subheader("Ergebnisse")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Gewählte Pumpe", best["id"])
        with c2:
            st.metric("Q_vis", f"{Q_vis_req:.2f} m³/h")
            st.metric("H_vis", f"{H_vis_req:.2f} m")
        with c3:
            st.metric("Q_wasser (Äquivalent)", f"{Q_water:.2f} m³/h")
            st.metric("H_wasser (Äquivalent)", f"{H_water:.2f} m")
        with c4:
            st.metric("Wellenleistung (viskos)", f"{P_vis_kW:.2f} kW")
            st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")
            st.metric("Leistung n0 (bei Q, H_n0)", f"{P_throttle_kW:.2f} kW")
        with c5:
            if n_opt_rpm is not None and saving_pct is not None:
                st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
                st.metric("η_vis", f"{eta_vis:.3f}")
                st.metric("Energieeinsparung", f"{saving_pct:.1f}%")
            else:
                st.info("Keine optimale Drehzahl im gewählten Bereich gefunden.")

        if not best["in_range"]:
            st.warning(f"Betriebspunkt außerhalb Wasserkennlinie! Bewertung bei Q={best['Q_eval']:.1f} m³/h")
            
        
        # NPSH-Prüfung entfernt (auf Wunsch)

        st.subheader("Kennlinien")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # --- Q-H ---
        ax1.plot(pump["Qw"], pump["Hw"], "o-", label="Wasser (n0)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", label="Viskos (n0)")
        if n_ratio_opt is not None:
            Q_scaled = [q * n_ratio_opt for q in pump["Qw"]]
            H_scaled = [h * (n_ratio_opt ** 2) for h in pump["Hw"]]
            ax1.plot(Q_scaled, H_scaled, ":", label=f"Wasser (n≈{n_opt_rpm:.0f} rpm)")
            Q_vis_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            H_vis_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
            ax1.plot(Q_vis_scaled, H_vis_scaled, "-.", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=90, label="BP (viskos)")
        ax1.scatter([Q_water], [best["H_at"]], marker="^", s=90, label="BP (Wasser-Äquiv.)")
        ax1.set_xlabel("Q [m³/h]")
        ax1.set_ylabel("H [m]")
        ax1.set_title("Q-H")
        ax1.grid(True)
        ax1.legend()

        # --- Q-η ---
        ax2.plot(pump["Qw"], pump["eta"], "o-", label="Wasser (n0)")
        ax2.plot(Q_vis_curve, eta_vis_curve, "s--", label="Viskos (n0)")
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=90, label="η (viskos)")
        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η")
        ax2.grid(True)
        ax2.legend()

        # --- Q-P (konsistent!) ---
        method_label = "aus H & η" if use_consistent_power else "Pw Referenz"
        ax3.plot(Q_water_curve, P_water_curve, "o-", label=f"Wasser ({method_label})")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", label=f"Viskos ({method_label})")

        if n_ratio_opt is not None:
            # Affinität: P ~ n^3, Q ~ n
            P_scaled = [p * (n_ratio_opt ** 3) for p in P_vis_curve]
            Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            ax3.plot(Q_scaled, P_scaled, ":", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")

        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=90, label="BP (viskos)")

        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        title_method = "konsistent aus H & η" if use_consistent_power else "aus Pw-Datensatz"
        ax3.set_title(f"Q-P ({title_method})")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Detaillierter Rechenweg"):
            st.markdown("### 0) Überblick")
            st.markdown("Ziel: Aus dem viskosen Betriebspunkt auf den äquivalenten Wasserpunkt zurückrechnen, Pumpe auswählen und Leistung bestimmen.")

            st.markdown("---")
            st.markdown("### 1) Eingabedaten")
            st.latex(r"Q_{vis} = " + f"{Q_vis_req:.2f}" + r"\;\mathrm{m^3/h}")
            st.latex(r"H_{vis} = " + f"{H_vis_req:.2f}" + r"\;\mathrm{m}")
            st.latex(r"\nu = " + f"{nu:.2f}" + r"\;\mathrm{cSt}")
            st.latex(r"\rho = " + f"{rho:.0f}" + r"\;\mathrm{kg/m^3}")

            st.markdown("---")
            st.markdown("### 2) Einheitenumrechnung (für HI‑Methode)")
            st.latex(r"Q_{gpm} = Q_{vis}\cdot 4.40287")
            st.latex(r"H_{ft} = H_{vis}\cdot 3.28084")
            Q_gpm_calc = Q_vis_req * 4.40287
            H_ft_calc = H_vis_req * 3.28084
            st.markdown(f"- Berechnet: **Q_gpm = {Q_gpm_calc:.2f}**, **H_ft = {H_ft_calc:.2f}**")

            st.markdown("---")
            st.markdown("### 3) HI‑Viskositätsparameter $B$")
            st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q_{gpm}^{0.25}\cdot H_{ft}^{0.375}}")
            st.markdown(f"- Ergebnis: **B = {B:.3f}**")

            st.info("Wasser‑Guard aktiv: Für ν≈1 cSt wird CH=Cη=1 gesetzt." if is_effectively_water(nu) else "HI‑Korrektur aktiv (ν > ~1.15 cSt).")

            st.markdown("---")
            st.markdown("### 4) Korrekturfaktoren")
            st.latex(r"C_H=\exp\left(-0.165\cdot (\log_{10}(B))^{2.2}\right)")
            st.latex(r"C_\eta = 1 - 0.25\log_{10}(B) - 0.05(\log_{10}(B))^2")
            st.markdown(f"- **C_H = {CH:.3f}**, **C_η = {Ceta:.3f}**")
            st.markdown("- Hinweis: $C_H<1$ bedeutet geringere Förderhöhe bei höherer Viskosität.")

            st.markdown("---")
            st.markdown("### 5) Rückrechnung auf Wasser‑Äquivalent")
            st.latex(r"H_{vis} = H_{water}\cdot C_H \Rightarrow H_{water} = \frac{H_{vis}}{C_H}")
            st.latex(r"Q_w \approx Q_{vis} \; (C_Q \approx 1)")
            st.markdown(f"- Ergebnis: **Q_w = {Q_water:.2f} m³/h**, **H_w = {H_water:.2f} m**")

            st.markdown("---")
            st.markdown("### 6) Wirkungsgrad und Wellenleistung (viskos)")
            st.latex(r"\eta_{vis} = \eta_{water}\cdot C_\eta")
            st.latex(r"P_{hyd}=\rho g Q H \quad,\quad P_{Welle}=\frac{P_{hyd}}{\eta}")
            st.markdown(f"- \(\eta_{{vis}} = {eta_vis:.3f}\)")
            st.markdown(f"- \(P_{{hyd}} = {P_hyd_W:.0f}\,W\) → **P_welle = {P_vis_kW:.2f} kW**")
            st.markdown(f"- Motorreserve {reserve_pct}% → **IEC = {P_motor_kW:.2f} kW**")

            st.markdown("---")
            st.markdown("### 7) Drehzahlvariation (Affinitätsgesetze)")
            st.latex(r"H(Q,n)=H(Q/n,n_0)\cdot (n/n_0)^2")
            st.latex(r"P(n)=P(n_0)\cdot (n/n_0)^3")
            if n_ratio_opt is not None and P_opt_kW is not None:
                st.markdown(f"- Drehzahlfaktor: **n/n0 = {n_ratio_opt:.3f}** → **n = {n_opt_rpm:.0f} rpm**")
                st.markdown(f"- **P_opt = {P_opt_kW:.2f} kW**, Einsparung **{saving_pct:.1f}%**")
            else:
                st.markdown("- Keine gültige optimale Drehzahl im Bereich gefunden.")

    except Exception as e:
        show_error(e, "Einphasenpumpen")

def run_multi_phase_pump():
    """
    Mehrphase:
    - Eingaben: C_ziel [Ncm³/L] (Default 100), Gas, Medium, Temperatur
    - Saugseite FIX: p_s = P_SUCTION_FIXED_BAR_ABS (Unterdruck)
    - p_req (Austritt) so, dass ALLE Einzelgase vollständig gelöst sind (bei Luft: N2+O2)
    - dp_req = p_req - p_s => H_req (physikalisch mit rho)
    - GVF_s aus FREIEM Gas an Saugseite (gelöst zählt nicht zur GVF)
    - Q ist NICHT vorgegeben: Betriebspunkt-Q wird aus Kennlinien als optimum bestimmt
    - GVF-Kurven werden interpoliert (8/9/11% möglich)
    - Rechenweg ergänzt
    """
    try:
        st.header("Mehrphasenpumpen-Auslegung (Q automatisch, Unterdruck Saugseite, vollständige Lösung)")

        with st.expander("Eingaben – aufklappen", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                st.subheader("Gasmenge (Norm)")
                C_ziel = st.number_input(
                    "Ziel-Gasmenge C_ziel [Ncm³/L] (Normvolumen je Liter Flüssigkeit)",
                    min_value=0.0, value=100.0, step=10.0
                )
                st.caption("100 Ncm³/L = 0.000100 Nm³/L (Norm)")

            with c2:
                st.subheader("Gas / Medium / Temperatur")
                gas_medium = st.selectbox("Gas", list(HENRY_CONSTANTS.keys()), index=0)
                liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)
                temperature = st.number_input("Temperatur T [°C]", min_value=-10.0, value=20.0, step=1.0)

            with c3:
                st.subheader("Optionen")
                safety_factor = st.slider("Sicherheitsfaktor auf GVF_s [%]", 0, 20, 10)
                use_cziel_as_gvf = st.checkbox(
                    "GVF-Kennlinie aus C_ziel (Normvolumen-%)",
                    value=False,
                    help="Wenn aktiv: C_ziel [Ncm³/L] wird physikalisch in GVF_s an der Saugseite umgerechnet. Kann mit GVF-Interpolation kombiniert werden."
                )
                use_interpolated_gvf = st.checkbox(
                    "GVF interpolieren (zwischen Kennlinien)",
                    value=True,
                    help="Wenn aktiv: GVF darf zwischen den Kurven liegen (z.B. 12,3%)."
                )
                if use_cziel_as_gvf and use_interpolated_gvf:
                    st.caption("Kombi aktiv: GVF wird aus C_ziel berechnet und zwischen Kennlinien interpoliert.")
                st.markdown("**Optimierung (gewichtete Kombination)**")
                w_power = st.slider("Gewicht Energie (P)", 0.0, 1.0, 0.5, 0.05)
                w_eta = st.slider("Gewicht Wirkungsgrad (η)", 0.0, 1.0, 0.3, 0.05)
                w_gas = st.slider("Gewicht Luftmenge", 0.0, 1.0, 0.2, 0.05)
                show_temp_band = st.checkbox("Temperaturband im Löslichkeitsdiagramm", value=True)
                show_ref_targets = st.checkbox("Referenzlinien (50/100/150 Ncm³/L)", value=True)

        rho_liq = float(MEDIA[liquid_medium]["rho"])
        nu_liq = float(MEDIA[liquid_medium]["nu"])
        p_suction = float(P_SUCTION_FIXED_BAR_ABS)

        # Umrechnungsfaktor bar -> m (physikalisch mit rho)
        BAR_TO_M_LIQ = (BAR_TO_PA) / (rho_liq * G)  # m pro bar

        # 1) p_req bestimmen: vollständige Lösung (optimaler Punkt)
        targets = None
        if gas_medium == "Luft":
            p_req, targets = pressure_required_for_air_components(temperature, C_ziel, p_min=0.2, p_max=200.0)
        else:
            p_req = pressure_required_for_C_target(gas_medium, temperature, C_ziel, y_gas=1.0, p_min=0.2, p_max=200.0)
            targets = {gas_medium: C_ziel}

        if p_req is None:
            dp_req = None
            H_req_m = None
        else:
            dp_req = max(0.0, float(p_req) - p_suction)
            H_req_m = dp_req * BAR_TO_M_LIQ

        # 2) Freies Gas an Saugseite (für GVF_s)
        dissolved_s = 0.0
        free_s = 0.0

        if gas_medium == "Luft":
            for g, y in AIR_COMPONENTS:
                C_i = targets.get(g, float(C_ziel) * float(y))
                C_sat_i = gas_solubility_cm3N_per_L(g, p_suction, temperature, y_gas=y)
                dissolved_s += min(C_i, C_sat_i)
                free_s += max(0.0, C_i - C_sat_i)
            gvf_ref_gas = "Luft"
        else:
            C_sat_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature, y_gas=1.0)
            dissolved_s = min(float(C_ziel), C_sat_s)
            free_s = max(0.0, float(C_ziel) - C_sat_s)
            gvf_ref_gas = gas_medium

        if use_cziel_as_gvf:
            # Physikalisch: Normvolumen -> GVF_s an der Saugseite (p,T,Z)
            gvf_s_pct = free_gas_gvf_pct_at_suction_from_cm3N_L(
                C_ziel, p_suction, temperature, gvf_ref_gas
            )
        else:
            gvf_s_pct = free_gas_gvf_pct_at_suction_from_cm3N_L(free_s, p_suction, temperature, gvf_ref_gas)
        gvf_s_pct_safe = gvf_s_pct * (1.0 + safety_factor / 100.0)
        if use_interpolated_gvf:
            gvf_curve_pct = gvf_s_pct_safe
        else:
            gvf_curve_pct = nearest_gvf_key(MPH_PUMPS[0], gvf_s_pct_safe) if MPH_PUMPS else gvf_s_pct_safe

        frac_diss_s = (dissolved_s / C_ziel * 100.0) if C_ziel > 0 else 0.0
        frac_free_s = (free_s / C_ziel * 100.0) if C_ziel > 0 else 0.0

        # 3) Pumpenauswahl: Q automatisch
        best_pump = None
        w_sum = max(float(w_power + w_eta + w_gas), 1e-9)
        w_power_n = float(w_power) / w_sum
        w_eta_n = float(w_eta) / w_sum
        w_gas_n = float(w_gas) / w_sum

        if dp_req is not None:
            best_pump = choose_best_mph_pump_autoQ(
                MPH_PUMPS,
                dp_req_bar=dp_req,
                gvf_free_pct=gvf_curve_pct,
                nu_cSt=nu_liq,
                rho_liq=rho_liq,
                w_power=w_power_n,
                w_eta=w_eta_n,
                w_gas=w_gas_n,
                C_target_cm3N_L=C_ziel,
                p_suction_bar_abs=p_suction,
                T_celsius=temperature,
                gas_medium=gas_medium
            )

        # =========================
        # Ergebnisse
        # =========================
        st.subheader("Ergebnisse")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Saugdruck (fix, Unterdruck)", f"{p_suction:.2f} bar(abs)")
            st.caption("Fix gesetzt, damit Luft eingesogen werden kann.")
        with r2:
            st.metric("Gelöst @ p_s", f"{frac_diss_s:.1f}%")
            st.metric("Frei @ p_s", f"{frac_free_s:.1f}%")
        with r3:
            st.metric("GVF_s (frei)", f"{gvf_s_pct:.2f}%")
            st.metric("GVF_s (+Sicherheit)", f"{gvf_s_pct_safe:.2f}%")
        with r4:
            if p_req is None:
                st.warning("p_req nicht erreichbar (0.2…200 bar) – Ziel zu hoch im Modell.")
            else:
                st.metric("p_req (Austritt, alle gelöst)", f"{p_req:.2f} bar(abs)")
                st.metric("Förderhöhe H_req", f"{H_req_m:.1f} m")

        if best_pump and dp_req is not None:
            c_oper_ncm3l = cm3N_L_from_gvf_pct_at_suction(gvf_curve_pct, p_suction, temperature, gvf_ref_gas)
            gvf_src = "aus C_ziel" if use_cziel_as_gvf else "physikalisch"
            gvf_display = f"{gvf_curve_pct:.1f}%" if use_interpolated_gvf else f"{gvf_curve_pct:.0f}%"
            gvf_mode = "interpoliert" if use_interpolated_gvf else "diskret"
            st.success(
                f"✅ Empfohlene Pumpe: {best_pump['pump']['id']}  |  GVF-Kennlinie {gvf_display} ({gvf_mode}, {gvf_src})"
            )
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Betriebspunkt Q", f"{best_pump['Q_m3h']:.2f} m³/h")
            with p2:
                st.metric("Δp verfügbar", f"{best_pump['dp_avail']:.2f} bar")
            with p3:
                st.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
            with p4:
                st.metric("Drehzahl / Modus", f"{best_pump['n_rpm']:.0f} rpm | {best_pump['mode']}")
            st.metric("Gasgehalt am Betriebspunkt", f"{c_oper_ncm3l:.1f} Ncm³/L")
            if "eta_est" in best_pump:
                st.caption(
                    f"Score‑Details: η_est={best_pump['eta_est']:.2f} | Gas‑Abweichung={best_pump['gas_err']*100:.1f}%"
                )
        else:
            st.info("Keine geeignete Mehrphasenpumpe gefunden (oder p_req nicht bestimmbar).")

        # =========================
        # Diagramme
        # =========================
        st.subheader("Diagramme")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

        # --- Löslichkeit ---
        if gas_medium == "Luft":
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve("Luft", T, y_gas=1.0)
                        ax1.plot(p_arr, sol_arr, "-", alpha=0.7, label=f"Luft (Gemisch) @ {T:.0f}°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve("Luft", temperature, y_gas=1.0)
                ax1.plot(p_arr, sol_arr, "-", label=f"Luft (Gemisch) @ {temperature:.0f}°C")

            for g, y in AIR_COMPONENTS:
                if show_temp_band:
                    for T in [temperature - 10, temperature, temperature + 10]:
                        if -10 <= T <= 150:
                            p_arr, sol_arr = solubility_diagonal_curve(g, T, y_gas=y)
                            ax1.plot(p_arr, sol_arr, "--", alpha=0.35, label="_nolegend_")
                else:
                    p_arr, sol_arr = solubility_diagonal_curve(g, temperature, y_gas=y)
                    ax1.plot(p_arr, sol_arr, "--", alpha=0.35, label="_nolegend_")

            ax1.axhline(C_ziel, linestyle=":", alpha=0.9, label="C_ziel (Luft)")
            ax1.text(13.8, C_ziel, "C_ziel (Luft)", va="center", ha="right", fontsize=8)

            Csat_s_mix = gas_solubility_cm3N_per_L("Luft", p_suction, temperature, y_gas=1.0)
            ax1.scatter([p_suction], [Csat_s_mix], s=60, label="C_sat @ p_s (Luft)")

            if p_req is not None:
                ax1.scatter([p_req], [C_ziel], s=110, marker="^", label="p_req (Luft)")
                ax1.axvline(p_req, linestyle=":", alpha=0.6, label="p_req")

        else:
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve(gas_medium, T, y_gas=1.0)
                        ax1.plot(p_arr, sol_arr, "--", label=f"{gas_medium} @ {T:.0f}°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve(gas_medium, temperature, y_gas=1.0)
                ax1.plot(p_arr, sol_arr, "--", label=f"{gas_medium} @ {temperature:.0f}°C")

            ax1.axhline(C_ziel, linestyle=":", alpha=0.8)
            ax1.text(13.8, C_ziel, "C_ziel", va="center", ha="right", fontsize=9)

            if p_req is not None:
                ax1.scatter([p_req], [C_ziel], s=110, marker="^", label="p_req")
                ax1.axvline(p_req, linestyle=":", alpha=0.6, label="p_req")

        if show_ref_targets:
            for Cref in [50.0, 100.0, 150.0]:
                ax1.axhline(Cref, linestyle=":", alpha=0.25)
                ax1.text(13.8, Cref, f"{Cref:.0f} Ncm³/L", va="center", ha="right", fontsize=8)

        ax1.set_xlabel("Absolutdruck [bar]")
        ax1.set_ylabel("Gasgehalt [Ncm³/L]")
        ax1.set_title("Löslichkeit & Ziel: vollständige Lösung")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlim(0, 14)

        # --- Mehrphasen-Kennlinien ---
        if best_pump and dp_req is not None:
            pump = best_pump["pump"]
            Q_sel = float(best_pump["Q_m3h"])
            Q_lmin_sel = m3h_to_lmin(Q_sel)
            H_req_plot = dp_req * BAR_TO_M_LIQ
            H_avail_plot = best_pump["dp_avail"] * BAR_TO_M_LIQ
            gvf_sel = float(gvf_curve_pct)
            n_ratio_sel = float(best_pump.get("n_ratio", 1.0))

            max_Q_lmin = 0.0
            max_H = 0.0

            for gvf_key in sorted(pump["curves_dp_vs_Q"].keys()):
                curve = pump["curves_dp_vs_Q"][gvf_key]
                Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
                H_m = [dp * BAR_TO_M_LIQ for dp in curve["dp"]]
                max_Q_lmin = max(max_Q_lmin, max(Q_lmin))
                max_H = max(max_H, max(H_m))
                ax2.plot(Q_lmin, H_m, "--", alpha=0.5, label=f"{gvf_key}% GVF")

            if use_interpolated_gvf:
                # Interpolierte GVF-Kurve (BP liegt exakt darauf)
                base_keys = sorted(pump["curves_dp_vs_Q"].keys())
                base_curve = pump["curves_dp_vs_Q"][base_keys[0]]
                Q_interp = list(map(float, base_curve["Q"]))
                dp_interp = [_dp_at_Q_gvf(pump, q, gvf_sel)[0] for q in Q_interp]
                Q_interp_scaled = [q * n_ratio_sel for q in Q_interp]
                H_interp = [dp * (n_ratio_sel ** 2) * BAR_TO_M_LIQ for dp in dp_interp]
                ax2.plot(
                    [m3h_to_lmin(q) for q in Q_interp_scaled],
                    H_interp,
                    "-",
                    linewidth=2.5,
                    label=f"GVF≈{gvf_sel:.1f}% (interpoliert, n={n_ratio_sel:.2f}·n0)"
                )
            else:
                # Ausgewählte diskrete GVF-Kurve (BP liegt exakt darauf)
                sel_curve = pump["curves_dp_vs_Q"].get(gvf_sel)
                if sel_curve:
                    Q_sel_curve = [q * n_ratio_sel for q in sel_curve["Q"]]
                    H_sel_curve = [dp * (n_ratio_sel ** 2) * BAR_TO_M_LIQ for dp in sel_curve["dp"]]
                    ax2.plot(
                        [m3h_to_lmin(q) for q in Q_sel_curve],
                        H_sel_curve,
                        "-",
                        linewidth=2.5,
                        label=f"GVF {gvf_sel:.0f}% (ausgewählt, n={n_ratio_sel:.2f}·n0)"
                    )

            ax2.scatter(Q_lmin_sel, H_avail_plot, s=110, marker="x", label="Betriebspunkt (auf Kennlinie)")
            ax2.scatter(Q_lmin_sel, H_req_plot, s=70, marker="o", facecolors="none", edgecolors="black", label="Anforderung (Δp_req)")
            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Förderhöhe [m]")
            ax2.set_title(f"Mehrphasen-Kennlinien: {pump['id']}")
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlim(0, max_Q_lmin * 1.1 if max_Q_lmin > 0 else 10)
            ax2.set_ylim(0, max_H * 1.1 if max_H > 0 else 10)
        else:
            ax2.text(0.5, 0.5, "Keine geeignete Pumpe / kein p_req", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Förderhöhe [m]")
            ax2.set_title("Mehrphasen-Kennlinien")
            ax2.grid(True)

        # --- Solubility + Pump curve vs pressure ---
        if gas_medium == "Luft":
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve("Luft", T, y_gas=1.0)
                        ax3.plot(p_arr, sol_arr, "-", alpha=0.7, label=f"Luft (Gemisch) @ {T:.0f}°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve("Luft", temperature, y_gas=1.0)
                ax3.plot(p_arr, sol_arr, "-", label=f"Luft (Gemisch) @ {temperature:.0f}°C")

            for g, y in AIR_COMPONENTS:
                if show_temp_band:
                    for T in [temperature - 10, temperature, temperature + 10]:
                        if -10 <= T <= 150:
                            p_arr, sol_arr = solubility_diagonal_curve(g, T, y_gas=y)
                            ax3.plot(p_arr, sol_arr, "--", alpha=0.6, label=f"{g} (y={y:.2f}) @ {T:.0f}°C")
                else:
                    p_arr, sol_arr = solubility_diagonal_curve(g, temperature, y_gas=y)
                    ax3.plot(p_arr, sol_arr, "--", label=f"{g} (y={y:.2f})")
        else:
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve(gas_medium, T, y_gas=1.0)
                        ax3.plot(p_arr, sol_arr, "--", label=f"{gas_medium} @ {T:.0f}°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve(gas_medium, temperature, y_gas=1.0)
                ax3.plot(p_arr, sol_arr, "--", label=f"{gas_medium} @ {temperature:.0f}°C")

        ax3.set_xlabel("Absolutdruck [bar]")
        ax3.set_ylabel("Löslichkeit [Ncm³/L] / Gasvolumenstrom [L/min]")
        ax3.set_title("Löslichkeit + Pumpenkennlinie (gemeinsame y-Achse)")
        ax3.grid(True)
        ax3.set_xlim(0, 14)

        if best_pump and dp_req is not None:
            pump = best_pump["pump"]
            Q_sel = float(best_pump["Q_m3h"])
            gvf_plot = float(gvf_curve_pct)
            gvf_frac = safe_clamp(gvf_plot / 100.0, 0.0, 0.99)
            n_ratio_sel = float(best_pump.get("n_ratio", 1.0))

            if use_interpolated_gvf:
                base_keys = sorted(pump["curves_dp_vs_Q"].keys())
                base_curve = pump["curves_dp_vs_Q"][base_keys[0]]
                Q_curve = list(map(float, base_curve["Q"]))
                dp_curve = [_dp_at_Q_gvf(pump, q, gvf_plot)[0] for q in Q_curve]
            else:
                sel_curve = pump["curves_dp_vs_Q"].get(gvf_plot)
                Q_curve = list(map(float, sel_curve["Q"])) if sel_curve else []
                dp_curve = list(map(float, sel_curve["dp"])) if sel_curve else []

            if Q_curve and dp_curve:
                Q_curve_scaled = [q * n_ratio_sel for q in Q_curve]
                dp_curve_scaled = [dp * (n_ratio_sel ** 2) for dp in dp_curve]
                p_abs = [p_suction + dp for dp in dp_curve_scaled]
                Q_gas_m3h = [q * (gvf_frac / (1.0 - gvf_frac)) for q in Q_curve_scaled]
                Q_gas_lmin = [m3h_to_lmin(qg) for qg in Q_gas_m3h]
                ax3.plot(p_abs, Q_gas_lmin, "-", linewidth=2.5, color="tab:red",
                         label=f"Gasvolumenstrom (GVF {gvf_plot:.1f}%, n={n_ratio_sel:.2f}·n0)")

                ax3.scatter(
                    [p_suction + best_pump["dp_avail"]],
                    [m3h_to_lmin(Q_sel * (gvf_frac / (1.0 - gvf_frac)))],
                    s=80,
                    color="tab:red",
                    marker="x",
                    label="Betriebspunkt (Gas)"
                )

        ax3.legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # Rechenweg (Mehrphase)
        # =========================
        with st.expander("Detaillierter Rechenweg (Mehrphase)"):
            st.markdown("### 0) Überblick")
            st.markdown(
                "Ziel: Aus **C_ziel**, Gas/Medium und Temperatur den erforderlichen Druck bestimmen, "
                "GVF_s an der Saugseite ableiten, eine passende GVF‑Kennlinie wählen und anschließend "
                "die energetisch optimale Pumpe (gewichtete Kombination) bestimmen."
            )

            st.markdown("---")
            st.markdown("### 1) Eingaben & Basisgrößen")
            st.markdown(f"- **p_s (fix):** {p_suction:.2f} bar(abs)")
            st.markdown(f"- **C_ziel:** {C_ziel:.1f} Ncm³/L  (= {C_ziel/1e6:.6f} Nm³/L)")
            st.markdown(f"- **Gas:** {gas_medium} | **Medium:** {liquid_medium} | **T:** {temperature:.1f} °C")
            st.markdown(f"- **Sicherheitsfaktor GVF_s:** {safety_factor:.0f}%")
            st.markdown(f"- **Umrechnung bar→m:** \(H=\\Delta p/(\\rho g)\) ⇒ 1 bar = {BAR_TO_M_LIQ:.2f} m (bei ρ={rho_liq:.0f} kg/m³)")

            st.markdown("---")
            st.markdown("### 2) Löslichkeit & Zielbedingung")
            st.latex(r"C_{sat}(p,T)=\text{Henry-Modell} \rightarrow \mathrm{Ncm^3/L}")
            st.markdown("**Ziel:** vollständige Lösung des Gases in der Flüssigkeit.")
            if gas_medium == "Luft":
                st.markdown("- Luft wird als **N₂ (79%) + O₂ (21%)** modelliert.")
                for g, y in AIR_COMPONENTS:
                    st.markdown(f"  - Ziel {g}:")
                    st.latex(rf"C_i = y\cdot C_{{ziel}} = {targets[g]:.1f}\;\mathrm{{Ncm^3/L}}")
                st.latex(r"p_{req}=\max_i\{p_{req,i}\}\quad\text{mit}\quad C_{sat,i}(p_{req,i},T)\ge C_i")
            else:
                st.latex(r"C_{sat}(p_{req},T)\ge C_{ziel}")
            if p_req is None:
                st.warning("p_req konnte im Bereich 0.2…200 bar nicht gefunden werden (Ziel zu hoch im Modell).")
            else:
                st.markdown(f"- Ergebnis: **p_req = {p_req:.2f} bar(abs)**")

            st.markdown("---")
            st.markdown("### 3) Druckhub & Förderhöhe")
            if dp_req is None:
                st.markdown("- Δp/H_req nicht berechenbar ohne p_req.")
            else:
                st.latex(r"\Delta p = p_{req}-p_s \quad;\quad H_{req}=\frac{\Delta p\cdot 10^5}{\rho g}")
                st.markdown(f"- Δp = {dp_req:.3f} bar → **H_req = {H_req_m:.2f} m**")

            st.markdown("---")
            st.markdown("### 4) Freies Gas an der Saugseite → GVF_s")
            st.latex(r"C_{free,s}=\max(0, C_{ziel}-C_{sat}(p_s,T)) \quad(\text{bzw. Summe über Komponenten})")
            st.markdown(f"- @p_s: gelöst = {dissolved_s:.1f} Ncm³/L, frei = {free_s:.1f} Ncm³/L")
            st.latex(r"GVF_s=\frac{V_{gas,op}}{1+V_{gas,op}}\cdot 100\%")
            st.markdown(f"- GVF_s = {gvf_s_pct:.2f}% → mit Sicherheit: **{gvf_s_pct_safe:.2f}%**")

            st.markdown("---")
            st.markdown("### 5) GVF‑Kennlinie & Gasgehalt am Betriebspunkt")
            gvf_src = "aus C_ziel" if use_cziel_as_gvf else "physikalisch (GVF_s)"
            gvf_mode = "interpoliert" if use_interpolated_gvf else "diskret"
            st.markdown(
                f"- Quelle: **{gvf_src}**, Modus: **{gvf_mode}** → GVF‑Kennlinie = **{gvf_curve_pct:.2f}%**"
            )
            st.latex(r"C_{op,N} \approx \frac{GVF}{1-GVF}\cdot \frac{p}{p_N}\cdot \frac{T_N}{T}\cdot \frac{1}{Z}\;\;[\mathrm{Ncm^3/L}]")
            if best_pump and dp_req is not None:
                c_oper_ncm3l = cm3N_L_from_gvf_pct_at_suction(gvf_curve_pct, p_suction, temperature, gvf_ref_gas)
                st.markdown(f"- Gasgehalt am Betriebspunkt: **{c_oper_ncm3l:.1f} Ncm³/L**")

            st.markdown("---")
            st.markdown("### 6) Pumpenauswahl & Affinitätsgesetze")
            st.latex(r"\Delta p(Q,n)=\Delta p(Q/n,n_0)\cdot (n/n_0)^2")
            st.latex(r"P(n)=P(n_0)\cdot (n/n_0)^3")
            if best_pump:
                st.markdown(
                    f"- Auswahl: **{best_pump['pump']['id']}**, "
                    f"Q={best_pump['Q_m3h']:.2f} m³/h, "
                    f"Δp_avail={best_pump['dp_avail']:.2f} bar, "
                    f"P={best_pump['P_req']:.2f} kW, "
                    f"n={best_pump['n_rpm']:.0f} rpm ({best_pump['mode']})"
                )
                if dp_req is not None:
                    st.markdown(f"- Vergleich: Δp_req={dp_req:.2f} bar ↔ Δp_avail={best_pump['dp_avail']:.2f} bar")
            else:
                st.markdown("- Keine geeignete Pumpe im Datensatz gefunden (oder p_req nicht bestimmbar).")

            st.markdown("---")
            st.markdown("### 7) Optimierungsziel (gewichtete Kombination)")
            st.markdown(
                "Die Auswahl minimiert eine gewichtete Zielfunktion aus **Energie**, **Wirkungsgrad** "
                "und **Gasmenge**, zusätzlich mit kleinen Strafanteilen für Drucküberschuss und "
                "Drehzahlabweichung."
            )
            st.latex(r"\text{Score}=w_P\cdot P_{spec} + w_\eta\cdot(1-\eta_{est}) + w_g\cdot\varepsilon_{gas}")

    except Exception as e:
        show_error(e, "Mehrphasenpumpen")

def run_atex_selection():
    try:
        st.header("ATEX-Motorauslegung")

        with st.expander("ATEX-Eingaben – aufklappen", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
                T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)
            with c2:
                atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"], index=0)
                if atmosphere == "Gas":
                    zone = st.selectbox("Zone", [0, 1, 2], index=2)
                else:
                    zone = st.selectbox("Zone", [20, 21, 22], index=2)
            with c3:
                t_margin = st.number_input("Temperaturabstand [K] (konservativ)", min_value=0.0, value=15.0, step=1.0)
                reserve_factor = st.number_input("Leistungsaufschlag [-] (z.B. 1.15)", min_value=1.0, value=1.15, step=0.01)

        st.subheader("Ergebnisse")

        if atmosphere == "Staub":
            st.error("❌ Staub-Ex: Keine Motor-Datensätze hinterlegt.")
            return
        if zone == 0:
            st.error("❌ Zone 0: Keine Motor-Datensätze hinterlegt.")
            return

        suitable = [m for m in ATEX_MOTORS if (zone in m["zone_suitable"])]
        if not suitable:
            st.error("❌ Kein passender Motor-Datensatz für die gewählte Zone vorhanden.")
            return

        suitable = [m for m in suitable if (m["t_max_surface"] - t_margin) >= T_medium]
        if not suitable:
            st.error(f"❌ Kein Motor verfügbar für T_medium = {T_medium:.1f}°C (mit {t_margin:.0f} K Abstand).")
            return

        P_motor_min = P_req * float(reserve_factor)
        P_iec = motor_iec(P_motor_min)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wellenleistung", f"{P_req:.2f} kW")
        with c2:
            st.metric(f"Mindestleistung (+{(reserve_factor-1)*100:.0f}%)", f"{P_motor_min:.2f} kW")
        with c3:
            st.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

        st.subheader("Verfügbare ATEX-Motoren")
        selected = st.radio(
            "Motortyp wählen:",
            options=suitable,
            format_func=lambda x: f"{x['marking']} ({x['id']})"
        )
        st.success("✅ Gültige Konfiguration gefunden")

        with st.expander("Detaillierter Rechenweg (ATEX)"):
            st.markdown("### 1) Eingaben")
            st.markdown(f"- **Wellenleistung:** {P_req:.2f} kW")
            st.markdown(f"- **Medientemperatur:** {T_medium:.1f} °C")
            st.markdown(f"- **Atmosphäre / Zone:** {atmosphere} / Zone {zone}")
            st.markdown(f"- **Temperaturabstand ΔT:** {t_margin:.0f} K")
            st.markdown(f"- **Leistungsaufschlag:** {reserve_factor:.2f} (-)")

            st.markdown("### 2) Leistungsanforderung (Motorgröße)")
            st.latex(r"P_{motor,min}=f_{res}\cdot P_{welle}")
            st.markdown(f"- \(P_{{motor,min}} = {reserve_factor:.2f}\cdot {P_req:.2f} = {P_motor_min:.2f}\,kW\)")
            st.markdown(f"- **IEC-Stufe (nächstgrößer):** {P_iec:.2f} kW")

            st.markdown("### 3) ATEX-Zulässigkeit (Zone)")
            st.markdown("- Filter: nur Motoren mit Freigabe für die gewählte Zone.")
            st.markdown(f"- Ergebnis: {len(suitable)} Motortyp(en) nach Zonenfilter.")

            st.markdown("### 4) Temperaturkriterium (Oberflächentemperatur)")
            st.latex(r"T_{surface,max}-T_{medium}\ge \Delta T")
            st.markdown(
                f"- Kriterium: \(T_{{surface,max}} - {T_medium:.1f} \ge {t_margin:.0f}\,K\)"
            )
            st.markdown(
                f"- Ergebnis: **{selected['t_max_surface']:.0f}°C** (Motor) \u2192 Abstand = "
                f"**{selected['t_max_surface'] - T_medium:.1f} K**"
            )

            st.markdown("### 5) Auswahl")
            st.markdown(f"- **Kennzeichnung:** {selected['marking']}")
            st.markdown(f"- **Motortyp-ID:** {selected['id']}")

        if st.button("ATEX-Dokumentation exportieren"):
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>ATEX-Auslegung - {selected['id']}</title>
<style>
body {{ font-family: Arial; max-width: 900px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
</style></head><body>
<h1>ATEX-Motorauslegung</h1>
<p>Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
<table>
<tr><th>Parameter</th><th>Wert</th></tr>
<tr><td>Wellenleistung</td><td>{P_req:.2f} kW</td></tr>
<tr><td>Mindestleistung</td><td>{P_motor_min:.2f} kW</td></tr>
<tr><td>IEC Motorgröße</td><td>{P_iec:.2f} kW</td></tr>
<tr><td>Atmosphäre</td><td>{atmosphere}</td></tr>
<tr><td>Zone</td><td>{zone}</td></tr>
<tr><td>Motor</td><td>{selected['id']}</td></tr>
<tr><td>Kennzeichnung</td><td>{selected['marking']}</td></tr>
<tr><td>Max. Oberflächentemperatur</td><td>{selected['t_max_surface']:.1f} °C</td></tr>
<tr><td>Medientemperatur</td><td>{T_medium:.1f} °C</td></tr>
<tr><td>Temperaturabstand</td><td>{selected['t_max_surface'] - T_medium:.1f} K</td></tr>
</table>
<p>Hinweis: Bitte Konformität mit 2014/34/EU und EN 60079 prüfen.</p>
</body></html>"""
            st.download_button(
                "HTML-Dokumentation herunterladen",
                data=html,
                file_name=f"ATEX_Auslegung_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )

    except Exception as e:
        show_error(e, "ATEX")

def main():
    try:
        st.title("🔧 Pumpenauslegungstool")

        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Seite auswählen:",
                ["Viskositätsberechnung", "Mehrphasenpumpen", "ATEX-Auslegung"],
                index=0
            )

        if page == "Viskositätsberechnung":
            run_single_phase_pump()
        elif page == "Mehrphasenpumpen":
            run_multi_phase_pump()
        else:
            run_atex_selection()

        if DEBUG:
            st.caption("DEBUG aktiv: Fehlertraces werden in der App angezeigt.")

    except Exception as e:
        show_error(e, "main")
        st.stop()

if __name__ == "__main__":
    main()

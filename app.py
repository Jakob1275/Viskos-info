import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="Pumpenauslegungstool", layout="wide")

# =========================
# Hilfsfunktion: Umrechnung Konzentration <-> Gasvolumenstrom
# =========================
def gas_flow_from_concentration(C_cm3N_L, Q_liq_m3h):
    """
    Umrechnung: Konzentration (cmÂ³N/L) und FlÃ¼ssigkeitsstrom (mÂ³/h) zu Gasvolumenstrom (Norm) in L/min.
    Q_gas_norm_lmin = (C_cm3N_L / 1000.0) * Q_liq_lmin
    """
    Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
    return (C_cm3N_L / 1000.0) * Q_liq_lmin

# =========================
# Config / Globals
# =========================
DEBUG = False

G = 9.81
BAR_TO_PA = 1e5
P_N_BAR = 1.01325
T_N_K = 273.15
R_BAR_L = 0.08314

N0_RPM_DEFAULT = 2900
P_SUCTION_FIXED_BAR_ABS = 0.6
SAT_PENALTY_WEIGHT = 1.5
AIR_SOLUBILITY_REF_T_C = 20.0
AIR_SOLUBILITY_REF_P_BAR = 5.0
AIR_SOLUBILITY_REF_C_CM3N_L = 122.7
AIR_SOLUBILITY_REF_TABLE = [
    (2.0, 36.8),
    (2.5, 46.0),
    (3.0, 55.2),
    (3.5, 64.4),
    (4.0, 73.6),
    (4.5, 82.8),
    (5.0, 92.0),
    (5.5, 101.2),
    (6.0, 110.4),
    (6.5, 119.6),
    (7.0, 128.8),
    (7.5, 138.0),
    (8.0, 147.2),
    (8.5, 156.4),
    (9.0, 165.6),
    (9.5, 177.0),
    (10.0, 185.0),
]

MAPPE10_MPH_AIR_LMIN = {
    "MPH-603": {
        15.0: {
            "p_abs": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            "gas_lmin": [96.9, 93.75, 90.3, 87.45, 81.15, 75.0, 67.5, 59.25, 49.95, 40.2, 25.05],
            "solubility_lmin": [82.8, 92.0, 101.2, 110.4, 119.6, 128.8, 138.0, 147.2, 156.4, 165.6, 177.0],
        },
    },
}

MEDIA = {
    "Wasser": {"rho": 998.0, "nu": 1.0},
    "Ã–l (leicht)": {"rho": 850.0, "nu": 10.0},
    "Ã–l (schwer)": {"rho": 900.0, "nu": 100.0},
}

HENRY_CONSTANTS = {
    "Luft": {"A": 800.0, "B": 1500},   
    "N2": {"A": 900.0, "B": 1400},    
    "O2": {"A": 700.0, "B": 1600},    
    "CO2": {"A": 29.0, "B": 2400},    
}

AIR_COMPONENTS = [
    ("N2", 0.79),
    ("O2", 0.21),
]

REAL_GAS_FACTORS = {
    # FÃ¼r p < 10 bar ist Z â‰ˆ 1.0 fÃ¼r Luft/N2/O2 akzeptabel, darÃ¼ber sollte ein besseres Modell verwendet werden.
    "Luft": lambda p_bar, T_K: 1.0,
    "N2": lambda p_bar, T_K: 1.0,
    "O2": lambda p_bar, T_K: 1.0,
    "CO2": lambda p_bar, T_K: max(0.9, 1.0 - 0.001 * (p_bar - 1.0)),  # NÃ¤herung, fÃ¼r hohe DrÃ¼cke ungenau
}


def air_solubility_correction(p_bar_abs, T_celsius):
    try:
        base = 0.0
        for g, y in AIR_COMPONENTS:
            base += gas_solubility_cm3N_per_L(g, p_bar_abs, AIR_SOLUBILITY_REF_T_C, y_gas=y)
        if base <= 0:
            return 1.0
        p_vals = [p for p, _ in AIR_SOLUBILITY_REF_TABLE]
        c_vals = [c for _, c in AIR_SOLUBILITY_REF_TABLE]
        ref_raw = safe_interp(float(p_bar_abs), p_vals, c_vals)
        ref_at_5 = safe_interp(float(AIR_SOLUBILITY_REF_P_BAR), p_vals, c_vals)
        scale = (float(AIR_SOLUBILITY_REF_C_CM3N_L) / float(ref_at_5)) if ref_at_5 > 0 else 1.0
        ref = float(ref_raw) * float(scale)
        if ref <= 0:
            return 1.0
        return float(ref) / float(base)
    except Exception:
        return 1.0


def air_solubility_cm3N_L(p_bar_abs, T_celsius):
    total = 0.0
    for g, y in AIR_COMPONENTS:
        total += gas_solubility_cm3N_per_L(g, p_bar_abs, T_celsius, y_gas=y)
    return total * float(air_solubility_correction(p_bar_abs, T_celsius))


def mappe10_air_lmin_lookup(pump_id, gvf_pct, p_abs_bar, kind, gvf_tol=0.25, allow_nearest=False):
    pump_map = MAPPE10_MPH_AIR_LMIN.get(pump_id)
    if not pump_map:
        return None, None
    keys = sorted(pump_map.keys())
    if not keys:
        return None, None
    gvf_sel = min(keys, key=lambda k: abs(float(k) - float(gvf_pct)))
    if (not allow_nearest) and (abs(float(gvf_sel) - float(gvf_pct)) > float(gvf_tol)):
        return None, gvf_sel
    curve = pump_map.get(gvf_sel)
    if not curve or kind not in curve:
        return None, gvf_sel
    return safe_interp(float(p_abs_bar), curve["p_abs"], curve[kind]), gvf_sel


def show_error(e, context):
    if DEBUG:
        st.exception(e)
    else:
        st.error(f"Fehler in {context}: {e}")


# =========================
# Pump datasets
# =========================
PUMPS = [
    {
        "id": "VIS-50",
        "Qw": [0, 10, 20, 30, 40, 50, 60],
        "Hw": [40, 38, 36, 32, 28, 24, 18],
        "eta": [0.0, 0.35, 0.55, 0.70, 0.72, 0.68, 0.60],
        "Pw": [0.1, 1.0, 2.0, 3.0, 4.0, 4.8, 5.5],
        "max_viscosity": 200,
        "max_density": 1200,
    },
    {
        "id": "VIS-80",
        "Qw": [0, 15, 30, 45, 60, 75, 90],
        "Hw": [55, 52, 48, 42, 36, 28, 18],
        "eta": [0.0, 0.40, 0.60, 0.75, 0.78, 0.73, 0.60],
        "Pw": [0.2, 1.5, 3.5, 5.5, 7.5, 9.5, 12.0],
        "max_viscosity": 500,
        "max_density": 1200,
    },
]

MPH_PUMPS = [
    {
        "id": "MPH-602",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 60,
        "dp_max_bar": 8.4,
        "GVF_max": 0.1,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 2.5,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [8.4, 8.3, 8.0, 7.5, 6.8, 6.0, 5.0]},
            10: {"Q": [15, 20, 25, 30, 35, 40, 45], "dp": [5.6, 5.5, 5.3, 5.1, 4.8, 4.4, 3.9]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
            10: {"Q": [15, 20, 25, 30, 35, 40, 45], "P": [7.0, 7.5, 7.8, 8.1, 8.5, 9.0, 9.1]},
        },
    },
    {
        "id": "MPH-403",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 24,
        "dp_max_bar": 9.4,
        "GVF_max": 0.15,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 2.0,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 4, 8, 12, 16, 20, 24], "dp": [9.4, 9.3, 9.0, 8.5, 7.6, 6.6, 5.3]},
            15: {"Q": [9, 12, 16, 20], "dp": [6.5, 6.5, 5.5, 3.3]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 4, 8, 12, 16, 20, 24], "P": [3.0, 3.8, 4.2, 5.0, 5.8, 6.2, 7.0]},
            15: {"Q": [9, 12, 16, 20], "P": [3.8, 4.0, 4.2, 4.8]},
        },
    },
    {
        "id": "MPH-603",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 50,
        "dp_max_bar": 12.0,
        "GVF_max": 0.2,
        "n0_rpm": 2900,
        "max_viscosity": 500,
        "max_density": 1200,
        "NPSHr": 3.0,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 10, 15, 20, 25, 30, 35, 40, 45], "dp": [12, 12, 11.8, 11.5, 11, 10.5, 9.8, 9.0, 8.0]},
            5: {"Q": [10, 15, 20, 25, 30, 35, 40, 45], "dp": [11.5, 11.5, 11, 10.5, 9.8, 9.0, 8]},
            10: {"Q": [10, 15, 20, 25, 30, 35, 40], "dp": [10.5, 10.1, 9.8, 9.2, 8.4, 7.3, 6.3]},
            15: {"Q": [10, 15, 20, 25, 30, 35], "dp": [9.5, 9.1, 8.5, 7.8, 6.8, 5.5]},
            20: {"Q": [10, 15, 20, 25, 30], "dp": [6.8, 7, 6.4, 5.5, 4.2]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 10, 15, 20, 25, 30, 35, 40, 45], "P": [7.0, 9.0, 10.0, 11.0, 11.8, 12.2, 13.0, 13.8, 14.0]},
            5: {"Q": [10, 15, 20, 25, 30, 35, 40, 45], "P": [8.8, 9.0, 11.0, 13.0, 11.8, 12, 13.5, 14.0]},
            10: {"Q": [10, 15, 20, 25, 30, 35, 40], "P": [8.5, 8.5, 11.5, 11.0, 12.0]},
            15: {"Q": [10, 15, 20, 25, 30, 35], "P": [8.3, 8.7, 9.2, 9.9, 10.8, 11.5]},
            20: {"Q": [10, 15, 20, 25, 30], "P": [8, 9, 9.5, 10, 10.3]},
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
        if len(xp) != len(fp):
            n = min(len(xp), len(fp))
            xp = xp[:n]
            fp = fp[:n]
        if len(xp) < 2:
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


def align_xy(x_vals, y_vals):
    try:
        x_list = list(x_vals)
        y_list = list(y_vals)
        n = min(len(x_list), len(y_list))
        return x_list[:n], y_list[:n]
    except Exception:
        return list(x_vals), list(y_vals)


def m3h_to_lmin(m3h):
    return float(m3h) * 1000.0 / 60.0


def gas_flow_required_norm_lmin(Q_liq_m3h, C_target_cm3N_L):
    """
    Ziel-Gasmenge -> Gasvolumenstrom bei Normbedingungen [L/min].
    Q_liq_m3h ist der FlÃ¼ssigkeitsstrom (nicht Gesamtstrom).
    """
    return float(Q_liq_m3h) * float(C_target_cm3N_L) / 60.0


def oper_to_norm_ratio(p_bar_abs, T_celsius, gas):
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p_bar_abs, T_celsius), 0.5)
    return (float(p_bar_abs) / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)


def gas_flow_oper_lmin_from_gvf(Q_total_m3h, gvf_pct):
    """
    Gasvolumenstrom (operativ) aus GVF und Gesamtstrom.
    Q_total_m3h = Q_liq + Q_gas.
    """
    gvf_frac = safe_clamp(float(gvf_pct) / 100.0, 0.0, 0.99)
    Q_gas_m3h = float(Q_total_m3h) * gvf_frac
    return m3h_to_lmin(Q_gas_m3h)


def gvf_to_flow_split(Q_total_m3h, gvf_pct):
    """
    Teilt Gesamtstrom in Gas- und FlÃ¼ssigkeitsanteil basierend auf GVF.
    GVF = Q_gas / Q_total (operativ, bei Betriebsbedingungen).
    """
    gvf_frac = safe_clamp(float(gvf_pct) / 100.0, 0.0, 0.99)
    Q_gas_oper_m3h = float(Q_total_m3h) * gvf_frac
    Q_liq_m3h = float(Q_total_m3h) * (1.0 - gvf_frac)
    return Q_liq_m3h, Q_gas_oper_m3h


def dissolved_concentration_cm3N_L_from_pct(conc_pct):
    """
    Kennlinien-Prozentwert (0â€“100%) als gelÃ¶ste Konzentration in cmÂ³N/L.
    100% entspricht 1 L(N) Gas pro 1 L FlÃ¼ssigkeit => 1000 cmÂ³N/L.
    """
    conc_pct = safe_clamp(float(conc_pct), 0.0, 100.0)
    return (conc_pct / 100.0) * 1000.0


def gas_flow_norm_lmin_from_conc_pct(Q_liq_m3h, conc_pct):
    """
    Gasvolumenstrom (Norm) aus gelÃ¶ster Konzentration (% bezogen auf FlÃ¼ssigkeit).
    """
    Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
    return Q_liq_lmin * (safe_clamp(float(conc_pct), 0.0, 100.0) / 100.0)


def gas_oper_m3h_to_norm_lmin(Q_gas_oper_m3h, p_bar_abs, T_celsius, gas):
    """
    Operativen Gasvolumenstrom (mÂ³/h) in Norm-L/min umrechnen.
    """
    Q_gas_oper_lmin = m3h_to_lmin(Q_gas_oper_m3h)
    ratio = oper_to_norm_ratio(p_bar_abs, T_celsius, gas)
    return Q_gas_oper_lmin * max(ratio, 1e-12)


def gvf_from_norm_gas_flow(Q_gas_norm_lmin, Q_liq_m3h, p_bar, T_celsius, gas):
    """
    Berechnet GVF aus Norm-Gasvolumenstrom (freies Gas).
    """
    ratio = oper_to_norm_ratio(p_bar, T_celsius, gas)
    Q_gas_oper_lmin = float(Q_gas_norm_lmin) / max(ratio, 1e-12)
    Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
    Q_total_oper_lmin = Q_liq_lmin + Q_gas_oper_lmin
    gvf_pct = (Q_gas_oper_lmin / max(Q_total_oper_lmin, 1e-12)) * 100.0
    return safe_clamp(gvf_pct, 0.0, 99.0)


def validate_gvf_consistency(Q_total_m3h, gvf_pct, p_bar_abs, T_celsius, gas,
                             Q_gas_norm_lmin=None, tol=0.05):
    """
    PrÃ¼ft grundlegende Konsistenz der GVF-/Gas-Umrechnung.
    Gibt Liste von Warnungen zurÃ¼ck.
    """
    issues = []
    Q_total_m3h = float(Q_total_m3h)
    gvf_pct = safe_clamp(float(gvf_pct), 0.0, 99.0)

    Q_liq_m3h, Q_gas_oper_m3h = gvf_to_flow_split(Q_total_m3h, gvf_pct)
    Q_gas_norm_calc = gas_oper_m3h_to_norm_lmin(Q_gas_oper_m3h, p_bar_abs, T_celsius, gas)

    gvf_back = gvf_from_norm_gas_flow(Q_gas_norm_calc, Q_liq_m3h, p_bar_abs, T_celsius, gas)
    if abs(gvf_back - gvf_pct) > (tol * 100.0):
        issues.append("GVF-Umrechnung inkonsistent (Normâ†”Operativ).")

    if Q_gas_norm_lmin is not None and Q_gas_norm_lmin > 0:
        rel = abs(Q_gas_norm_calc - float(Q_gas_norm_lmin)) / max(float(Q_gas_norm_lmin), 1e-9)
        if rel > tol:
            issues.append("Q_gas_norm weicht von GVF-AbschÃ¤tzung ab.")

    if Q_liq_m3h <= 0 or Q_total_m3h <= 0:
        issues.append("Q_total/Q_liq nicht gÃ¼ltig (<=0).")

    return issues




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
WATER_EPS = 0.15  # ~15% Toleranz, damit "Wasser" nicht fÃ¤lschlich korrigiert wird


def is_effectively_water(nu_cSt: float) -> bool:
    return float(nu_cSt) <= (WATER_NU_CST + WATER_EPS)


# =========================
# HI ViskositÃ¤t (robust)
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
    RÃ¼ckrechnung von viskosem Betriebspunkt auf Ã¤quivalenten Wasser-Betriebspunkt.
    H_vis = H_water \cdot CH  =>  H_water = H_vis / CH
    Bei hÃ¶herer ViskositÃ¤t ist H_vis < H_water, daher ist CH < 1.
    Wichtig: Bei "Wasser" darf KEINE HI-Korrektur aktiv werden, sonst bekommst du
    unterschiedliche Q-P/Q-H-Kurven fÃ¼r Wasser vs. viskos.
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
    - Konsistente Berechnung: P wird direkt aus H_vis und Î·_vis berechnet
    - P_vis = (Ï g Q H_vis) / Î·_vis

    Option B (use_consistent_power=False, legacy):
    - Wasser-Leistungskurve basiert auf Pw-Datensatz (realistisch inkl. Verluste)
    - Viskose Leistung wird als Skalierung von Pw abgeleitet (damit Offsets/Verluste erhalten bleiben)
    - H_vis und eta_vis werden wie bisher Ã¼ber (CH, Ceta) korrigiert
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

        # KORREKTUR: Bei hÃ¶herer ViskositÃ¤t SINKT die FÃ¶rderhÃ¶he -> Multiplikation mit CH!
        hv = float(h) * max(float(CH), 1e-9)
        ev = safe_clamp(float(e) * float(Ceta), 0.05, 0.95)

        if use_consistent_power:
            # Konsistente Berechnung: P_vis direkt aus H_vis und Î·_vis
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            pv = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0
        else:
            # Legacy: Skalierung von Pw_ref (bewahrt Verluste/Offsets)
            # Theoretische Wasserleistung aus H&Î· nur fÃ¼r Skalierungsfaktor
            P_hyd_water_W = rho * G * (float(q) / 3600.0) * float(h)
            P_water_theory = (P_hyd_water_W / max(float(e), 1e-9)) / 1000.0

            # theoretische viskose Leistung (nur fÃ¼r Skalierungsfaktor)
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            P_vis_theory = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0

            # Skalierung der realistischen Wasser-Pw auf "viskos"
            if P_water_theory > 1e-6:
                scale = float(P_vis_theory) / float(P_water_theory)
            else:
                # Bei Qâ‰ˆ0 ist Theorie unbrauchbar -> konservativ nur Ã¼ber Effizienzfaktor skalieren
                scale = 1.0 / max(float(Ceta), 1e-9)

            pv = float(Pw_ref[i]) * float(scale)

        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)

    return Qw.tolist(), H_vis, eta_vis, P_vis


def water_power_curve_from_H_eta(pump, rho):
    """
    Konsistente Wasser-P-Kurve aus H(Q) und Î·(Q).
    Damit ist Q-P mathematisch konsistent und nicht abhÃ¤ngig von pump["Pw"].
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
# Gas / LÃ¶slichkeit / Konzentration
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
    Henry-basiert, Ausgabe: cmÂ³N/L (Normvolumen pro Liter FlÃ¼ssigkeit)
    Einheitenanalyse:
    - H: Henry-Konstante [barÂ·L/mol]
    - p: Partialdruck [bar]
    - C_mol_L: Konzentration [mol/L]
    - V_molar_oper: Molvolumen [L/mol] bei p,T
    - V_oper_L_per_L: Gasvolumen [L/L] bei p,T
    - ratio: Umrechnung operativ -> Normbedingungen
    """
    p = max(float(p_bar_abs), 1e-6)
    T_K = float(T_celsius) + 273.15
    H = max(henry_constant(gas, T_celsius), 1e-12)  # barÂ·L/mol
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)
    p_partial = safe_clamp(float(y_gas), 0.0, 1.0) * p

    # Henry: C = p/H (mol/L)
    C_mol_L = p_partial / H

    # Gasvolumen operativ (L/L) bei p,T
    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    # oper -> normal
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    # RÃ¼ckgabe: Normvolumen cmÂ³N/L
    return V_oper_L_per_L * ratio * 1000.0  # cmÂ³N/L


def free_gas_gvf_pct_at_suction_from_cm3N_L(free_cm3N_L, p_suction_bar_abs, T_celsius, gas):
    """
    Freies Gas (Norm cmÂ³N/L) -> GVF% (operativ) an Saugseite
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


def solubility_diagonal_curve_air_corrected(T_celsius, p_min=0.2, p_max=14.0, n=140):
    ps = np.linspace(p_min, p_max, n)
    sol = [air_solubility_cm3N_L(p, T_celsius) for p in ps]
    return ps, np.array(sol)


def pressure_required_for_C_target(gas, T_celsius, C_target_cm3N_L, y_gas=1.0, p_min=0.2, p_max=200.0):
    """
    Finde p_abs, so dass C_sat(p,T) â‰ˆ C_target (bisection).
    Gibt None zurÃ¼ck, wenn Ziel auÃŸerhalb des Suchbereichs liegt.
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
    Luft wird als N2/O2 betrachtet. Ziel ist vollstÃ¤ndiges LÃ¶sen ALLER Komponenten.
    Dazu wird je Komponente der erforderliche Druck berechnet und p_req = max(p_req_i) genommen.
    """
    targets = {gas_i: float(C_total_cm3N_L) * float(y) for gas_i, y in AIR_COMPONENTS}

    def f(p):
        return air_solubility_cm3N_L(p, T_celsius) - float(C_total_cm3N_L)

    lo, hi = float(p_min), float(p_max)
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        return lo, targets
    if fhi < 0:
        return None, targets
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-3:
            return mid, targets
        if fm >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), targets


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
# Mehrphase: Interpolation Ã¼ber Konzentrations-Kurven + Drehzahl
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


def gas_solubility_total_cm3N_L(gas_medium, p_bar_abs, T_celsius):
    if gas_medium == "Luft":
        return air_solubility_cm3N_L(p_bar_abs, T_celsius)
    return gas_solubility_cm3N_per_L(gas_medium, p_bar_abs, T_celsius, y_gas=1.0)


def choose_best_mph_pump(pumps, Q_total_m3h, dp_req_bar, gvf_free_pct, nu_cSt, rho_liq,
                        n_min_ratio=0.5, n_max_ratio=1.2,
                        w_power=0.5, w_eta=0.3, w_gas=0.2,
                        C_target_cm3N_L=0.0, p_suction_bar_abs=1.0, T_celsius=20.0, gas_medium="Luft",
                        allow_speed_adjustment=False, allow_partial_solution=False):
    """
    WÃ¤hlt beste Pumpe bei vorgegebenem Q_total und dp.
    Interpolation zwischen GVF-Kurven (auch 8/9/11% mÃ¶glich).
    """
    best = None

    Q_total_m3h = float(Q_total_m3h)
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

            dp_nom, _, _, _ = _dp_at_Q_gvf(pump, Q_total_m3h, gvf_free_pct)

            def dp_at_ratio(nr):
                if nr <= 0:
                    return 0.0
                Q_base = Q_total_m3h / nr
                dp_base, _, _, _ = _dp_at_Q_gvf(pump, Q_base, gvf_free_pct)
                return dp_base * (nr ** 2)

            def f(nr):
                return dp_at_ratio(nr) - dp_req

            n_ratio = bisect_root(f, n_min_ratio, n_max_ratio, it=80, tol=1e-4)
            if n_ratio is None:
                n_ratio = find_best_ratio_by_scan(dp_at_ratio, dp_req, n_min_ratio, n_max_ratio, steps=60, prefer_above=True)

            candidates = []

            P_nom, _, _, _ = _P_at_Q_gvf(pump, Q_total_m3h, gvf_free_pct)
            candidates.append({
                "pump": pump,
                "gvf_key": gvf_free_pct,
                "dp_avail": dp_nom,
                "P_req": P_nom,
                "n_ratio": 1.0,
                "n_rpm": pump["n0_rpm"],
                "mode": "Nenndrehzahl",
                "Q_m3h": Q_total_m3h,
            })

            if allow_speed_adjustment and n_ratio is not None:
                Q_base = Q_total_m3h / n_ratio
                dp_scaled = dp_at_ratio(n_ratio)
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
                    "Q_m3h": Q_total_m3h,
                })

            Q_liq_m3h, Q_gas_oper_m3h = gvf_to_flow_split(Q_total_m3h, gvf_free_pct)
            Q_gas_req_norm_lmin = gas_flow_required_norm_lmin(Q_liq_m3h, C_target_cm3N_L)

            Q_gas_pump_norm_lmin = gas_oper_m3h_to_norm_lmin(
                Q_gas_oper_m3h,
                p_suction_bar_abs,
                T_celsius,
                gas_medium
            )
            tol = 0.02

            for cand in candidates:
                dp_err = 0.0 if dp_req <= 0 else abs(cand["dp_avail"] - dp_req) / max(dp_req, 1e-6)

                P_spec = cand["P_req"] / max(Q_total_m3h, 1e-6)
                P_hyd_kW = (cand["dp_avail"] * BAR_TO_PA) * (Q_total_m3h / 3600.0) / 1000.0
                eta_est = safe_clamp(P_hyd_kW / max(cand["P_req"], 1e-9), 0.0, 1.0)
                eta_term = 1.0 - eta_est

                p_discharge = float(p_suction_bar_abs) + float(cand["dp_avail"])
                C_sat_total = gas_solubility_total_cm3N_L(gas_medium, p_discharge, T_celsius)
                Q_gas_solubility_norm_lmin = gas_flow_required_norm_lmin(Q_liq_m3h, C_sat_total)

                if gas_medium == "Luft":
                    mappe_gas_lmin, _ = mappe10_air_lmin_lookup(pump["id"], gvf_free_pct, p_discharge, "gas_lmin")
                    mappe_sat_lmin, _ = mappe10_air_lmin_lookup(pump["id"], gvf_free_pct, p_discharge, "solubility_lmin")
                    if mappe_gas_lmin is not None:
                        Q_gas_pump_norm_lmin = float(mappe_gas_lmin)
                    if mappe_sat_lmin is not None:
                        Q_gas_solubility_norm_lmin = float(mappe_sat_lmin)
                sat_excess = max(0.0, Q_gas_pump_norm_lmin - Q_gas_solubility_norm_lmin)
                sat_err = sat_excess / max(Q_gas_pump_norm_lmin, 1e-6)
                if (not allow_partial_solution) and (sat_err > tol):
                    continue

                gas_err = abs(Q_gas_pump_norm_lmin - Q_gas_req_norm_lmin) / max(Q_gas_req_norm_lmin, 1e-6)

                score = (
                    float(w_gas) * gas_err +
                    float(w_power) * P_spec +
                    float(w_eta) * eta_term +
                    float(SAT_PENALTY_WEIGHT) * sat_err +
                    3.0 * dp_err +
                    0.10 * abs(cand["n_ratio"] - 1.0)
                )
                cand["score"] = score
                cand["eta_est"] = eta_est
                cand["gas_err"] = gas_err
                cand["dp_err"] = dp_err
                if best is None or score < best["score"]:
                    best = cand

        except Exception:
            continue

    return best


def choose_best_mph_pump_autoQ(
    pumps, gas_target_norm_lmin, p_suction_bar_abs, T_celsius, gas_medium,
    safety_factor_pct, use_interpolated_gvf,
    nu_cSt, rho_liq,
    n_min_ratio=0.5, n_max_ratio=1.2,
    w_power=0.5, w_eta=0.3, w_gas=0.2,
    allow_speed_adjustment=False, allow_partial_solution=False
):
    """
    Pumpenauswahl fÃ¼r Mehrphasenpumpen mit Ziel-Gasmenge (Norm L/min).
    
    WICHTIG: Die Kennlinien-Prozentwerte sind NICHT GVF (freies Gas), sondern
    die Gesamtluftkonzentration (gelÃ¶st + frei) in % von 1000 cmÂ³N/L.
    
    - Q auf der Kennlinie = FlÃ¼ssigkeitsstrom (nicht Gesamtstrom)
    - Kennlinie-% â†’ C_kennlinie [cmÂ³N/L] = (%-Wert / 100) * 1000
    - Die Aufteilung gelÃ¶st/frei erfolgt nach dem Druck an der Druckseite (Henry)
    - Die Pumpe ist geeignet, wenn bei Druckseite die Zielmenge als SÃ¤ttigung gelÃ¶st werden kann
    """
    best = None
    gas_target_norm_lmin = float(gas_target_norm_lmin)

    def _conc_pct_to_cm3N_L(conc_pct):
        """Kennlinien-% â†’ Konzentration in cmÂ³N/L (100% = 1000 cmÂ³N/L)"""
        return (float(conc_pct) / 100.0) * 1000.0

    def _calc_Q_gas_from_conc(Q_liq_m3h, conc_cm3N_L):
        """Gasvolumenstrom (Norm L/min) aus Konzentration und FlÃ¼ssigkeitsstrom"""
        Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
        return (float(conc_cm3N_L) / 1000.0) * Q_liq_lmin

    def _target_Q_liq_from_conc_pct(conc_pct):
        """Berechne Q_liq, bei dem die Kennlinien-Konzentration die Zielmenge ergibt"""
        C_kennlinie = _conc_pct_to_cm3N_L(conc_pct)
        if C_kennlinie <= 0:
            return None
        # Q_gas_ziel = (C_kennlinie / 1000) * Q_liq_lmin
        # => Q_liq_lmin = Q_gas_ziel / (C_kennlinie / 1000)
        Q_liq_lmin = gas_target_norm_lmin / (C_kennlinie / 1000.0)
        return (Q_liq_lmin * 60.0) / 1000.0  # mÂ³/h

    for pump in pumps:
        try:
            if nu_cSt > pump.get("max_viscosity", 500):
                continue
            if rho_liq > pump.get("max_density", 1200):
                continue

            # Kennlinien-Keys sind Konzentrations-Prozente (NICHT GVF!)
            conc_keys = sorted(pump["curves_dp_vs_Q"].keys())
            any_curve = pump["curves_dp_vs_Q"][conc_keys[0]]
            Q_base = list(map(float, any_curve["Q"]))
            Q_base = [q for q in Q_base if q > 0]
            if not Q_base:
                continue

            qmin = max(min(Q_base), 1e-6)
            qmax = max(Q_base)

            def _scan_candidates(best_local=None):
                """
                Scannt alle Kombinationen aus Q_liq und Konzentrations-Kennlinie.
                
                Die Kennlinien-% sind die Gesamtluftkonzentration (gelÃ¶st + frei).
                Q auf der Kennlinie ist der FlÃ¼ssigkeitsstrom.
                """
                # Maximale Konzentration auf Kennlinien (in %)
                max_conc_pct = max(conc_keys)
                
                # Kandidaten: verschiedene Konzentrations-Kennlinien
                if use_interpolated_gvf:
                    conc_candidates = np.linspace(conc_keys[0], max_conc_pct, 15).tolist()
                else:
                    conc_candidates = list(conc_keys)
                
                for conc_pct in conc_candidates:
                    # Konzentration in cmÂ³N/L
                    C_kennlinie = _conc_pct_to_cm3N_L(conc_pct)
                    
                    # Finde Q_liq, bei dem diese Kennlinie die Zielmenge liefert
                    Q_liq_target = _target_Q_liq_from_conc_pct(conc_pct)
                    
                    # Bestimme Q-Bereich fÃ¼r diese Kennlinie
                    lo_key, hi_key, _ = _interp_between_gvf_keys(pump, conc_pct)
                    Q_lo = [q for q in pump["curves_dp_vs_Q"][lo_key]["Q"] if q > 0]
                    Q_hi = [q for q in pump["curves_dp_vs_Q"][hi_key]["Q"] if q > 0]
                    if not Q_lo or not Q_hi:
                        continue
                    qmin_conc = max(min(Q_lo), min(Q_hi))
                    qmax_conc = min(max(Q_lo), max(Q_hi))
                    if qmax_conc <= qmin_conc:
                        continue
                    
                    # Kandidaten-Q (FlÃ¼ssigkeitsstrom)
                    Q_candidates = np.linspace(qmin_conc, qmax_conc, 40).tolist()
                    if Q_liq_target is not None and qmin_conc <= Q_liq_target <= qmax_conc:
                        Q_candidates.append(float(Q_liq_target))
                    Q_candidates = sorted(set(Q_candidates))
                    
                    for Q_liq_m3h in Q_candidates:
                        Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
                        
                        # Gasmenge laut Kennlinie (Norm L/min)
                        Q_gas_kennlinie_lmin = _calc_Q_gas_from_conc(Q_liq_m3h, C_kennlinie)
                        
                        # PrÃ¼fung: Kennlinie muss mindestens Zielmenge liefern
                        if Q_gas_kennlinie_lmin < gas_target_norm_lmin * 0.98:
                            continue
                        
                        # Druckseite: dp aus Kennlinie ablesen
                        dp_avail, _, _, _ = _dp_at_Q_gvf(pump, Q_liq_m3h, conc_pct)
                        p_discharge = float(p_suction_bar_abs) + float(dp_avail)
                        
                        # LÃ¶slichkeit bei Druckseite (Henry)
                        C_sat_discharge = gas_solubility_total_cm3N_L(gas_medium, p_discharge, T_celsius)
                        
                        # Aufteilung gelÃ¶st/frei bei Druckseite
                        C_dissolved = min(C_kennlinie, C_sat_discharge)
                        C_free = max(0.0, C_kennlinie - C_sat_discharge)
                        
                        # Zielkonzentration (cmÂ³N/L)
                        C_ziel = (gas_target_norm_lmin / max(Q_liq_lmin, 1e-12)) * 1000.0
                        
                        # PrÃ¼fung: Zielmenge muss als SÃ¤ttigung lÃ¶sbar sein
                        if C_sat_discharge < C_ziel * 0.98:
                            continue
                        
                        # Gasmenge gelÃ¶st (Norm L/min)
                        Q_gas_dissolved_lmin = _calc_Q_gas_from_conc(Q_liq_m3h, C_dissolved)
                        Q_gas_free_lmin = _calc_Q_gas_from_conc(Q_liq_m3h, C_free)
                        
                        # Score: Je nÃ¤her Q_gas_dissolved an Q_gas_ziel, desto besser
                        gas_diff = abs(Q_gas_dissolved_lmin - gas_target_norm_lmin)
                        
                        # Hydraulische Leistung
                        P_req = (dp_avail * Q_liq_m3h) / 36.0
                        
                        # Randbedingungen (Pumpe nicht am Limit betreiben)
                        q_rel = Q_liq_m3h / max(pump["Q_max_m3h"], 1e-9)
                        edge_penalty = 0.0
                        if q_rel < 0.2 or q_rel > 0.9:
                            edge_penalty = 10.0
                        
                        score = gas_diff + 0.1 * P_req + edge_penalty
                        
                        cand = {
                            "pump": pump,
                            "conc_pct": conc_pct,
                            "C_kennlinie": C_kennlinie,
                            "Q_m3h": Q_liq_m3h,
                            "Q_liq_m3h": Q_liq_m3h,
                            "Q_liq_lmin": Q_liq_lmin,
                            "dp_avail": dp_avail,
                            "p_discharge": p_discharge,
                            "C_sat_discharge": C_sat_discharge,
                            "C_dissolved": C_dissolved,
                            "C_free": C_free,
                            "C_ziel": C_ziel,
                            "Q_gas_kennlinie_lmin": Q_gas_kennlinie_lmin,
                            "Q_gas_dissolved_lmin": Q_gas_dissolved_lmin,
                            "Q_gas_free_lmin": Q_gas_free_lmin,
                            "P_req": P_req,
                            "n_rpm": pump["n0_rpm"],
                            "n_ratio": 1.0,
                            "mode": "Nenndrehzahl",
                            "score": score,
                            "score2": score,
                            "solution_status": "strict" if C_free <= 0.01 else "partial",
                        }
                        
                        if best_local is None or score < best_local["score2"]:
                            best_local = cand
                
                return best_local

            best_local = _scan_candidates()

            if best_local is not None and (best is None or best_local["score2"] < best["score2"]):
                best = best_local

        except Exception:
            continue

    return best


# =========================
# Pages
# =========================
def run_single_phase_pump():
    try:
        st.header("Einphasenpumpen mit ViskositÃ¤tskorrektur")

        with st.expander("Eingaben â€“ aufklappen", expanded=True):
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                st.subheader("Betriebspunkt (viskos)")
                Q_vis_req = st.number_input("FÃ¶rderstrom Q_vis [mÂ³/h]", min_value=0.1, value=30.0, step=1.0)
                H_vis_req = st.number_input("FÃ¶rderhÃ¶he H_vis [m]", min_value=0.1, value=20.0, step=1.0)
            with colB:
                st.subheader("Medium")
                medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
                rho = st.number_input("Dichte Ï [kg/mÂ³]", min_value=1.0, value=float(MEDIA[medium]["rho"]), step=5.0)
                nu = st.number_input("Kinematische ViskositÃ¤t Î½ [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.5)
            with colC:
                st.subheader("Optionen")
                allow_out = st.checkbox("Auswahl auÃŸerhalb Kennlinie zulassen", value=True)
                reserve_pct = st.slider("Motorreserve [%]", 0, 30, 10)
                n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
                n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)
                use_consistent_power = st.checkbox("Konsistente Leistungsberechnung (P aus H & Î·)", value=True,
                                                   help="Empfohlen: Berechnet P direkt aus H und Î· statt Skalierung von Pw")

        # NPSH / KavitationsprÃ¼fung entfernt (auf Wunsch)

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

        # Baseline ohne VFD: Pumpe bei n0 liefert bei Q_vis_req eine bestimmte FÃ¶rderhÃ¶he H_pump_n0
        # (wenn H_pump_n0 > H_vis_req, wird die "zu viel" FÃ¶rderhÃ¶he als Verlust vernichtet,
        # die Pumpe muss sie aber trotzdem aufbauen -> Leistung basiert auf H_pump_n0)
        H_pump_n0 = safe_interp(Q_vis_req, pump["Qw"], pump["Hw"])
        eta_pump_n0 = safe_interp(Q_vis_req, pump["Qw"], pump["eta"])

        # ViskositÃ¤tswirkung auf Î· (bei Wasser ist Ceta=1 durch Wasser-Guard)
        eta_pump_n0_vis = safe_clamp(float(eta_pump_n0) * float(Ceta), 0.05, 0.95)

        P_hyd_throttle_w = rho * G * (Q_vis_req / 3600.0) * float(H_pump_n0)
        P_throttle_kW = (P_hyd_throttle_w / max(eta_pump_n0_vis, 1e-9)) / 1000.0

        # VFD-Fall: gleiche Anforderung Q_vis_req & H_vis_req, Î· am "Basis"-Punkt (Q_base) aus viskoser Kurve
        if n_ratio_opt is not None:
            n_opt_rpm = N0_RPM_DEFAULT * n_ratio_opt

            Q_base = Q_vis_req / n_ratio_opt
            eta_base = safe_interp(Q_base, Q_vis_curve, eta_vis_curve)
            eta_vfd = safe_clamp(float(eta_base), 0.05, 0.95)

            P_hyd_vfd_w = rho * G * (Q_vis_req / 3600.0) * float(H_vis_req)
            P_opt_kW = (P_hyd_vfd_w / max(eta_vfd, 1e-9)) / 1000.0

            saving_pct = ((P_throttle_kW - P_opt_kW) / P_throttle_kW * 100.0) if P_throttle_kW > 0 else 0.0

        st.subheader("Ergebnisse")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("GewÃ¤hlte Pumpe", best["id"])
        with c2:
            st.metric("Q_vis", f"{Q_vis_req:.2f} mÂ³/h")
            st.metric("H_vis", f"{H_vis_req:.2f} m")
        with c3:
            st.metric("Q_wasser (Ã„quivalent)", f"{Q_water:.2f} mÂ³/h")
            st.metric("H_wasser (Ã„quivalent)", f"{H_water:.2f} m")
        with c4:
            st.metric("Wellenleistung (viskos)", f"{P_vis_kW:.2f} kW")
            st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")
            st.metric("Leistung n0 (bei Q, H_n0)", f"{P_throttle_kW:.2f} kW")
        with c5:
            if n_opt_rpm is not None and saving_pct is not None:
                st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
                st.metric("Î·_vis", f"{eta_vis:.3f}")
                st.metric("Energieeinsparung", f"{saving_pct:.1f}%")
            else:
                st.info("Keine optimale Drehzahl im gewÃ¤hlten Bereich gefunden.")

        if not best["in_range"]:
            st.warning(f"Betriebspunkt auÃŸerhalb Wasserkennlinie! Bewertung bei Q={best['Q_eval']:.1f} mÂ³/h")

        # NPSH-PrÃ¼fung entfernt (auf Wunsch)

        st.subheader("Kennlinien")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # --- Q-H ---
        ax1.plot(pump["Qw"], pump["Hw"], "o-", label="Wasser (n0)")
        q_vis_plot = [q for q in Q_vis_curve if q > 0]
        h_vis_plot = [h for q, h in zip(Q_vis_curve, H_vis_curve) if q > 0]
        eta_vis_plot = [e for q, e in zip(Q_vis_curve, eta_vis_curve) if q > 0]
        p_vis_plot = [p for q, p in zip(Q_vis_curve, P_vis_curve) if q > 0]
        ax1.plot(q_vis_plot, h_vis_plot, "s--", label="Viskos (n0)")
        if n_ratio_opt is not None:
            Q_scaled = [q * n_ratio_opt for q in pump["Qw"]]
            H_scaled = [h * (n_ratio_opt ** 2) for h in pump["Hw"]]
            ax1.plot(Q_scaled, H_scaled, ":", label=f"Wasser (nâ‰ˆ{n_opt_rpm:.0f} rpm)")
            Q_vis_scaled = [q * n_ratio_opt for q in q_vis_plot]
            H_vis_scaled = [h * (n_ratio_opt ** 2) for h in h_vis_plot]
            ax1.plot(Q_vis_scaled, H_vis_scaled, "-.", label=f"Viskos (nâ‰ˆ{n_opt_rpm:.0f} rpm)")
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=90, label="BP (viskos)")
        ax1.scatter([Q_water], [best["H_at"]], marker="^", s=90, label="BP (Wasser-Ã„quiv.)")
        ax1.set_xlabel("Q [mÂ³/h]")
        ax1.set_ylabel("H [m]")
        ax1.set_title("Q-H")
        ax1.grid(True)
        ax1.legend()

        # --- Q-Î· ---
        ax2.plot(pump["Qw"], pump["eta"], "o-", label="Wasser (n0)")
        ax2.plot(q_vis_plot, eta_vis_plot, "s--", label="Viskos (n0)")
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=90, label="Î· (viskos)")
        ax2.set_xlabel("Q [mÂ³/h]")
        ax2.set_ylabel("Î· [-]")
        ax2.set_title("Q-Î·")
        ax2.grid(True)
        ax2.legend()

        # --- Q-P (konsistent!) ---
        method_label = "aus H & Î·" if use_consistent_power else "Pw Referenz"
        ax3.plot(Q_water_curve, P_water_curve, "o-", label=f"Wasser ({method_label})")
        ax3.plot(q_vis_plot, p_vis_plot, "s--", label=f"Viskos ({method_label})")

        if n_ratio_opt is not None:
            # AffinitÃ¤t: P ~ n^3, Q ~ n
            P_scaled = [p * (n_ratio_opt ** 3) for p in p_vis_plot]
            Q_scaled = [q * n_ratio_opt for q in q_vis_plot]
            ax3.plot(Q_scaled, P_scaled, ":", label=f"Viskos (nâ‰ˆ{n_opt_rpm:.0f} rpm)")

        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=90, label="BP (viskos)")

        ax3.set_xlabel("Q [mÂ³/h]")
        ax3.set_ylabel("P [kW]")
        title_method = "konsistent aus H & Î·" if use_consistent_power else "aus Pw-Datensatz"
        ax3.set_title(f"Q-P ({title_method})")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Detaillierter Rechenweg"):
            st.markdown("### 0) Ãœberblick")
            st.markdown("Ziel: Aus dem viskosen Betriebspunkt auf den Ã¤quivalenten Wasserpunkt zurÃ¼ckrechnen, Pumpe auswÃ¤hlen und Leistung bestimmen.")
            st.markdown("**Einheiten:** $Q$ in mÂ³/h, $H$ in m, $\nu$ in cSt, $\rho$ in kg/mÂ³.")

            st.markdown("---")
            st.markdown("### 1) Eingabedaten")
            st.latex(r"Q_{vis} = " + f"{Q_vis_req:.2f}" + r"\;\mathrm{m^3/h}")
            st.latex(r"H_{vis} = " + f"{H_vis_req:.2f}" + r"\;\mathrm{m}")
            st.latex(r"\nu = " + f"{nu:.2f}" + r"\;\mathrm{cSt}")
            st.latex(r"\rho = " + f"{rho:.0f}" + r"\;\mathrm{kg/m^3}")

            st.markdown("---")
            st.markdown("### 2) Einheitenumrechnung (fÃ¼r HIâ€‘Methode)")
            st.latex(r"Q_{gpm} = Q_{vis}\cdot 4.40287")
            st.latex(r"H_{ft} = H_{vis}\cdot 3.28084")
            Q_gpm_calc = Q_vis_req * 4.40287
            H_ft_calc = H_vis_req * 3.28084
            st.markdown(f"- Berechnet: **Q_gpm = {Q_gpm_calc:.2f}**, **H_ft = {H_ft_calc:.2f}**")
            st.caption("Umrechnung nÃ¶tig, weil die HIâ€‘Korrelation mit gpm/ft parametrisiert ist.")

            st.markdown("---")
            st.markdown("### 3) HIâ€‘ViskositÃ¤tsparameter $B$")
            st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q_{gpm}^{0.25}\cdot H_{ft}^{0.375}}")
            st.markdown(f"- Ergebnis: **B = {B:.3f}**")
            st.caption("$B$ steuert die Korrekturfaktoren fÃ¼r FÃ¶rderhÃ¶he und Wirkungsgrad.")

            st.info("Wasserâ€‘Guard aktiv: FÃ¼r Î½â‰ˆ1 cSt wird CH=CÎ·=1 gesetzt." if is_effectively_water(nu) else "HIâ€‘Korrektur aktiv (Î½ > ~1.15 cSt).")

            st.markdown("---")
            st.markdown("### 4) Korrekturfaktoren")
            st.latex(r"C_H=\exp\left(-0.165\cdot (\log_{10}(B))^{2.2}\right)")
            st.latex(r"C_\eta = 1 - 0.25\log_{10}(B) - 0.05(\log_{10}(B))^2")
            st.markdown(f"- **C_H = {CH:.3f}**, **C_Î· = {Ceta:.3f}**")
            st.markdown("- Hinweis: $C_H<1$ bedeutet geringere FÃ¶rderhÃ¶he bei hÃ¶herer ViskositÃ¤t.")
            st.caption("Wasserâ€‘Guard: Bei $\nu\approx 1$ cSt wird $C_H=C_\eta=1$ gesetzt.")

            st.markdown("---")
            st.markdown("### 5) RÃ¼ckrechnung auf Wasserâ€‘Ã„quivalent")
            st.latex(r"H_{vis} = H_{water}\cdot C_H \Rightarrow H_{water} = \frac{H_{vis}}{C_H}")
            st.latex(r"Q_w \approx Q_{vis} \; (C_Q \approx 1)")
            st.markdown(f"- Ergebnis: **Q_w = {Q_water:.2f} mÂ³/h**, **H_w = {H_water:.2f} m**")
            st.caption("Die Pumpenkennlinien sind fÃ¼r Wasser angegeben, daher die RÃ¼ckrechnung.")

            st.markdown("---")
            st.markdown("### 6) Wirkungsgrad und Wellenleistung (viskos)")
            st.latex(r"\eta_{vis} = \eta_{water}\cdot C_\eta")
            st.latex(r"P_{hyd}=\rho g Q H \quad,\quad P_{Welle}=\frac{P_{hyd}}{\eta}")
            st.markdown(f"- \(\eta_{{vis}} = {eta_vis:.3f}\)")
            st.markdown(f"- \(P_{{hyd}} = {P_hyd_W:.0f}\,W\) â†’ **P_welle = {P_vis_kW:.2f} kW**")
            st.markdown(f"- Motorreserve {reserve_pct}% â†’ **IEC = {P_motor_kW:.2f} kW**")
            st.caption("$P_{hyd}$ ist die hydraulische Leistung; die Wellenleistung berÃ¼cksichtigt den Wirkungsgrad.")

            st.markdown("---")
            st.markdown("### 7) Drehzahlvariation (AffinitÃ¤tsgesetze)")
            st.latex(r"H(Q,n)=H(Q/n,n_0)\cdot (n/n_0)^2")
            st.latex(r"P(n)=P(n_0)\cdot (n/n_0)^3")
            if n_ratio_opt is not None and P_opt_kW is not None:
                st.markdown(f"- Drehzahlfaktor: **n/n0 = {n_ratio_opt:.3f}** â†’ **n = {n_opt_rpm:.0f} rpm**")
                st.markdown(f"- **P_opt = {P_opt_kW:.2f} kW**, Einsparung **{saving_pct:.1f}%**")
            else:
                st.markdown("- Keine gÃ¼ltige optimale Drehzahl im Bereich gefunden.")
            st.caption("AffinitÃ¤tsgesetze gelten idealisiert fÃ¼r geometrisch Ã¤hnliche StrÃ¶mung.")

    except Exception as e:
        show_error(e, "Einphasenpumpen")


def run_multi_phase_pump():
    """
    Mehrphase:
    - Eingaben: Q_gas_ziel [L/min] (Norm-Volumenstrom), Gas, Medium, Temperatur
    - Saugseite FIX: p_s = P_SUCTION_FIXED_BAR_ABS (Unterdruck)
    - p_req (Austritt) so, dass ALLE Einzelgase vollstÃ¤ndig gelÃ¶st sind (bei Luft: N2+O2)
    - p_req wird als absoluter Druck gefÃ¼hrt => H_req (physikalisch mit rho)
    - Kennlinien-Prozentwerte = GVF (freies Gas, operativ)
    - Q ist NICHT vorgegeben: Betriebspunkt-Q wird aus Kennlinien als optimum bestimmt
    - GVF-Kurven werden interpoliert (8/9/11% mÃ¶glich)
    - Rechenweg ergÃ¤nzt
    """
    try:
        st.header("Mehrphasenpumpen-Auslegung (Q automatisch, Zielâ€‘Gasstrom Ã¼ber Druckseite)")

        with st.expander("Eingaben â€“ aufklappen", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                st.subheader("Gasstrom (Norm)")
                C_ziel_lmin = st.number_input(
                    "Ziel-Gasvolumenstrom Q_gas_ziel [L/min] (Norm)",
                    min_value=0.0, value=50.0, step=1.0
                )
                st.caption("Q_gas_ziel ist ein Norm-Volumenstrom. Umrechnung erfolgt Ã¼ber den FlÃ¼ssigkeitsstrom.")

            with c2:
                st.subheader("Gas / Medium / Temperatur")
                gas_medium = st.selectbox("Gas", list(HENRY_CONSTANTS.keys()), index=0)
                liquid_medium = st.selectbox("FlÃ¼ssigmedium", list(MEDIA.keys()), index=0)
                temperature = st.number_input("Temperatur T [Â°C]", min_value=-10.0, value=20.0, step=1.0)

            with c3:
                st.subheader("Optionen")
                safety_factor = 0.0
                use_interpolated_gvf = True
                st.markdown("**Optimierung (gewichtete Kombination)**")
                w_power = st.slider("Gewicht Energie (P)", 0.0, 1.0, 0.5, 0.05)
                w_eta = st.slider("Gewicht Wirkungsgrad (Î·)", 0.0, 1.0, 0.3, 0.05)
                w_gas = st.slider("Gewicht Luftmenge", 0.0, 1.0, 0.2, 0.05)
                allow_partial_solution = st.checkbox(
                    "Q_gas_zielâ€‘Optimierung (Druckseite, gelÃ¶st)",
                    value=True,
                    help="Bewertet gelÃ¶ste Konzentration an der Druckseite und wÃ¤hlt das Optimum zu Q_gas_ziel."
                )
                show_temp_band = st.checkbox("Temperaturband im LÃ¶slichkeitsdiagramm", value=True)
                show_ref_targets = st.checkbox("Referenzlinien (50/100/150 L/min Norm)", value=True)
                show_speed_alt = st.checkbox(
                    "Drehzahlanpassung als Alternative anzeigen",
                    value=True,
                    help="Vergleicht n0-Betrieb mit Drehzahlanpassung und zeigt mÃ¶gliche Energieeinsparung."
                )

        rho_liq = float(MEDIA[liquid_medium]["rho"])
        nu_liq = float(MEDIA[liquid_medium]["nu"])
        p_suction = float(P_SUCTION_FIXED_BAR_ABS)

        # Hinweis: Mehrphasen-Kennlinien werden direkt in bar ausgewertet (ohne Umrechnung in m).

        # 1) Pumpenauswahl: Q automatisch (Q_gas_ziel in L/min wird je Kandidat intern umgerechnet)
        best_pump = None
        w_sum = max(float(w_power + w_eta + w_gas), 1e-9)
        w_power_n = float(w_power) / w_sum
        w_eta_n = float(w_eta) / w_sum
        w_gas_n = float(w_gas) / w_sum

        best_pump = choose_best_mph_pump_autoQ(
            MPH_PUMPS,
            gas_target_norm_lmin=C_ziel_lmin,
            p_suction_bar_abs=p_suction,
            T_celsius=temperature,
            gas_medium=gas_medium,
            safety_factor_pct=safety_factor,
            use_interpolated_gvf=use_interpolated_gvf,
            nu_cSt=nu_liq,
            rho_liq=rho_liq,
            w_power=w_power_n,
            w_eta=w_eta_n,
            w_gas=w_gas_n,
            allow_speed_adjustment=False,
            allow_partial_solution=allow_partial_solution
        )

        best_pump_invalid = False
        solution_status = None
        if best_pump:
            # Neue Variablen aus korrigierter Logik
            dp_req = best_pump.get("dp_avail")
            dp_req_bar = dp_req if dp_req is not None else None
            p_discharge = best_pump.get("p_discharge", p_suction + (dp_req or 0.0))
            p_req = p_discharge  # p_req ist jetzt der tatsÃ¤chliche Austrittsdruck
            
            conc_pct = best_pump.get("conc_pct", 0.0)
            C_kennlinie = best_pump.get("C_kennlinie", 0.0)
            C_dissolved = best_pump.get("C_dissolved", 0.0)
            C_free = best_pump.get("C_free", 0.0)
            C_sat_discharge = best_pump.get("C_sat_discharge", 0.0)
            C_ziel = best_pump.get("C_ziel", 0.0)
            
            Q_liq_m3h = best_pump.get("Q_liq_m3h", best_pump.get("Q_m3h", 0.0))
            Q_gas_kennlinie_lmin = best_pump.get("Q_gas_kennlinie_lmin", 0.0)
            Q_gas_dissolved_lmin = best_pump.get("Q_gas_dissolved_lmin", 0.0)
            Q_gas_free_lmin = best_pump.get("Q_gas_free_lmin", 0.0)
            
            solution_status = best_pump.get("solution_status")
            
            # Validierung: Ist die Zielmenge erreichbar?
            tol = 0.02
            if Q_gas_kennlinie_lmin < C_ziel_lmin * (1.0 - tol):
                best_pump_invalid = True
            elif C_sat_discharge < C_ziel * (1.0 - tol):
                best_pump_invalid = True
        else:
            p_req = None
            dp_req = None
            dp_req_bar = None
            conc_pct = 0.0
            C_kennlinie = 0.0
            C_dissolved = 0.0
            C_free = 0.0
            C_sat_discharge = 0.0
            C_ziel = 0.0
            Q_liq_m3h = 0.0
            Q_gas_kennlinie_lmin = 0.0
            Q_gas_dissolved_lmin = 0.0
            Q_gas_free_lmin = 0.0

        def dissolved_free_at_pressure(p_abs_bar, C_target_cm3N_L, targets_local):
            dissolved = 0.0
            free = 0.0
            sat_total = 0.0
            if gas_medium == "Luft":
                sat_total = air_solubility_cm3N_L(p_abs_bar, temperature)
                dissolved = min(float(C_target_cm3N_L), sat_total)
                free = max(0.0, float(C_target_cm3N_L) - sat_total)
            else:
                C_sat = gas_solubility_cm3N_per_L(gas_medium, p_abs_bar, temperature, y_gas=1.0)
                sat_total = C_sat
                dissolved = min(float(C_target_cm3N_L), C_sat)
                free = max(0.0, float(C_target_cm3N_L) - C_sat)
            return dissolved, free, sat_total

        def cm3N_L_to_lmin(C_cm3N_L, Q_liq_m3h):
            Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
            return (float(C_cm3N_L) / 1000.0) * Q_liq_lmin

        # =========================
        # Ergebnisse
        # =========================
        if best_pump_invalid:
            st.warning("Keine gÃ¼ltige LÃ¶sung: Q_gas_ziel ist bei der gewÃ¤hlten Pumpe/Druckseite nicht vollstÃ¤ndig lÃ¶sbar.")
            best_pump = None
            p_req = None
            dp_req = None
            dp_req_bar = None
        st.subheader("Ergebnisse")
        if solution_status == "partial":
            st.warning("Hinweis: Q_gas_ziel ist mit den Kennlinien nicht vollstÃ¤ndig lÃ¶sbar. Es wird die bestmÃ¶gliche AnnÃ¤herung angezeigt.")

        with st.expander("Debug â€“ Mehrphasenâ€‘ZwischengrÃ¶ÃŸen"):
            if best_pump:
                # Neue Variablen aus der korrigierten Logik
                Q_liq_m3h_dbg = float(best_pump.get("Q_liq_m3h", best_pump.get("Q_m3h", 0.0)))
                Q_liq_lmin_dbg = float(best_pump.get("Q_liq_lmin", m3h_to_lmin(Q_liq_m3h_dbg)))
                dp_dbg = float(best_pump.get("dp_avail", 0.0))
                p_dbg = float(best_pump.get("p_discharge", p_suction + dp_dbg))
                conc_pct_dbg = float(best_pump.get("conc_pct", 0.0))
                C_kennlinie_dbg = float(best_pump.get("C_kennlinie", 0.0))

                # Gasmenge laut Kennlinie (Norm L/min)
                Q_gas_kennlinie_dbg = float(best_pump.get("Q_gas_kennlinie_lmin", 0.0))
                
                # LÃ¶slichkeit am Austritt
                C_sat_dbg = float(best_pump.get("C_sat_discharge", gas_solubility_total_cm3N_L(gas_medium, p_dbg, temperature)))
                
                # Aufteilung gelÃ¶st/frei
                C_dissolved_dbg = float(best_pump.get("C_dissolved", 0.0))
                C_free_dbg = float(best_pump.get("C_free", 0.0))
                Q_gas_dissolved_dbg = float(best_pump.get("Q_gas_dissolved_lmin", 0.0))
                Q_gas_free_dbg = float(best_pump.get("Q_gas_free_lmin", 0.0))
                
                # LÃ¶sbare Gasmenge bei p_aus
                Q_gas_losbar_dbg = gas_flow_from_concentration(C_sat_dbg, Q_liq_m3h_dbg)
                alles_geloest = Q_gas_kennlinie_dbg <= Q_gas_losbar_dbg

                st.markdown("**Betriebspunkt:**")
                st.write({
                    "Q_liq [mÂ³/h]": round(Q_liq_m3h_dbg, 3),
                    "Q_liq [L/min]": round(Q_liq_lmin_dbg, 2),
                    "Kennlinie [%]": round(conc_pct_dbg, 2),
                    "C_kennlinie [cmÂ³N/L]": round(C_kennlinie_dbg, 2),
                    "p_austritt [bar abs]": round(p_dbg, 3),
                })

                st.markdown("**Gasbilanz (Norm L/min):**")
                col_gas1, col_gas2 = st.columns(2)
                with col_gas1:
                    st.metric("Q_gas_ziel (Eingabe)", f"{float(C_ziel_lmin):.2f}")
                    st.metric("Q_gas_kennlinie (Pumpe)", f"{Q_gas_kennlinie_dbg:.2f}")
                with col_gas2:
                    st.metric("Q_gas_lÃ¶sbar (bei p_aus)", f"{Q_gas_losbar_dbg:.2f}")
                    if alles_geloest:
                        st.success(f"âœ… Freies Gas: {Q_gas_free_dbg:.2f} (alles lÃ¶sbar)")
                    else:
                        st.error(f"âŒ Freies Gas: {Q_gas_free_dbg:.2f}")

                st.markdown("**Konzentrationen (cmÂ³N/L):**")
                st.write({
                    "C_kennlinie (gesamt)": round(C_kennlinie_dbg, 2),
                    "C_dissolved (gelÃ¶st bei p_aus)": round(C_dissolved_dbg, 2),
                    "C_free (frei bei p_aus)": round(C_free_dbg, 2),
                    "C_sat (max lÃ¶sbar bei p_aus)": round(C_sat_dbg, 2),
                })

        Q_total_m3h_sel = float(best_pump["Q_m3h"]) if best_pump else None
        Q_liq_m3h_sel = float(best_pump.get("Q_liq_m3h", Q_total_m3h_sel)) if best_pump else None
        Q_liq_lmin_sel = m3h_to_lmin(Q_liq_m3h_sel) if Q_liq_m3h_sel else None
        conc_pct_sel = float(best_pump.get("conc_pct", 0.0)) if best_pump else 0.0
        C_target_cm3N_L = float(best_pump.get("C_ziel", 0.0)) if best_pump else 0.0

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Saugdruck (fix, Unterdruck)", f"{p_suction:.2f} bar(abs)")
            st.caption("Fix gesetzt, damit Luft eingesogen werden kann.")
        with r2:
            if Q_liq_m3h_sel is not None and best_pump:
                st.metric("Q_gas_kennlinie [L/min]", f"{Q_gas_kennlinie_lmin:.2f}")
                st.metric("Q_gas_gelÃ¶st [L/min]", f"{Q_gas_dissolved_lmin:.2f}")
                st.metric("Kennlinie [%]", f"{conc_pct_sel:.1f}")
            else:
                st.info("Kein Q verfÃ¼gbar (keine Pumpe gefunden).")
        with r3:
            if Q_liq_m3h_sel is not None:
                st.metric("Q_gas_ziel (Norm) [L/min]", f"{C_ziel_lmin:.1f}")
                st.metric("Q_flÃ¼ssig [L/min]", f"{Q_liq_lmin_sel:.1f}")
            else:
                st.info("Kein Q verfÃ¼gbar (keine Pumpe gefunden).")
        with r4:
            if p_req is None:
                st.warning("p_req nicht erreichbar â€“ Ziel zu hoch im Modell.")
            else:
                st.metric("p_austritt [bar abs]", f"{p_req:.2f}")
                st.metric("Î”p", f"{dp_req_bar:.2f} bar")

        if best_pump and dp_req is not None:
            p_discharge_disp = float(best_pump.get("p_discharge", p_suction + dp_req))
            C_dissolved_disp = float(best_pump.get("C_dissolved", 0.0))
            C_sat_disp = float(best_pump.get("C_sat_discharge", 0.0))
            C_free_disp = float(best_pump.get("C_free", 0.0))
            
            st.success(f"âœ… Empfohlene Pumpe: {best_pump['pump']['id']}")
            Q_liq_req_sel = float(best_pump.get("Q_liq_m3h", best_pump["Q_m3h"]))
            
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Betriebspunkt Q_liq", f"{Q_liq_req_sel:.2f} mÂ³/h")
            with p2:
                st.metric("p_austritt", f"{p_discharge_disp:.2f} bar")
            with p3:
                st.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
                st.metric("C_sat (max lÃ¶sbar)", f"{C_sat_disp:.1f} cmÂ³N/L")
            with p4:
                st.metric("Drehzahl / Modus", f"{best_pump['n_rpm']:.0f} rpm | {best_pump['mode']}")
                st.metric("C_dissolved (gelÃ¶st)", f"{C_dissolved_disp:.1f} cmÂ³N/L")

            if show_speed_alt:
                pump_sel = best_pump["pump"]
                Q_liq_m3h_req = float(best_pump.get("Q_liq_m3h", best_pump["Q_m3h"]))
                conc_pct_sel = float(best_pump.get("conc_pct", 0.0))

                def _build_speed_candidates(pump, Q_req, dp_req, conc_pct, n_min_ratio=0.5, n_max_ratio=1.2):
                    cand = {}
                    dp_nom, _, _, _ = _dp_at_Q_gvf(pump, Q_req, conc_pct)
                    if dp_nom >= dp_req:
                        P_nom, _, _, _ = _P_at_Q_gvf(pump, Q_req, conc_pct)
                        cand["nominal"] = {
                            "dp": dp_nom,
                            "P": P_nom,
                            "n_ratio": 1.0,
                            "n_rpm": pump["n0_rpm"],
                        }

                    def dp_at_ratio(nr):
                        if nr <= 0:
                            return 0.0
                        Q_base = Q_req / nr
                        dp_base, _, _, _ = _dp_at_Q_gvf(pump, Q_base, conc_pct)
                        return dp_base * (nr ** 2)

                    def f(nr):
                        return dp_at_ratio(nr) - dp_req

                    n_ratio = bisect_root(f, n_min_ratio, n_max_ratio, it=80, tol=1e-4)
                    if n_ratio is None:
                        n_ratio = find_best_ratio_by_scan(dp_at_ratio, dp_req, n_min_ratio, n_max_ratio, steps=60, prefer_above=True)

                    if n_ratio is not None:
                        Q_base = Q_req / n_ratio
                        dp_scaled = dp_at_ratio(n_ratio)
                        if dp_scaled >= dp_req:
                            P_base, _, _, _ = _P_at_Q_gvf(pump, Q_base, conc_pct)
                            P_scaled = P_base * (n_ratio ** 3)
                            cand["vfd"] = {
                                "dp": dp_scaled,
                                "P": P_scaled,
                                "n_ratio": n_ratio,
                                "n_rpm": pump["n0_rpm"] * n_ratio,
                            }
                    return cand

                cand_map = _build_speed_candidates(pump_sel, Q_liq_m3h_req, float(dp_req), conc_pct_sel)
                if "nominal" in cand_map and "vfd" in cand_map:
                    P_nom = cand_map["nominal"]["P"]
                    P_vfd = cand_map["vfd"]["P"]
                    saving_pct = ((P_nom - P_vfd) / P_nom * 100.0) if P_nom > 0 else 0.0

                    st.markdown("**Drehzahlanpassung (Alternative) â€“ Energievergleich**")
                    p_discharge_vfd = float(p_suction) + float(cand_map["vfd"]["dp"])
                    C_sat_vfd = gas_solubility_total_cm3N_L(gas_medium, p_discharge_vfd, temperature)
                    Q_gas_losbar_vfd = gas_flow_from_concentration(C_sat_vfd, Q_liq_m3h_req)

                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.metric("Q_gas_lÃ¶sbar (VFD) [L/min]", f"{Q_gas_losbar_vfd:.2f}")
                    with a2:
                        st.metric("p_austritt (VFD)", f"{p_discharge_vfd:.2f} bar")
                    with a3:
                        st.metric("n angepasst", f"{cand_map['vfd']['n_rpm']:.0f} rpm")
                    with a4:
                        st.metric("Energieeinsparung", f"{saving_pct:.1f}%")
                else:
                    st.info("Drehzahlanpassung als Alternative nicht darstellbar (p_abs oder nâ€‘Grenzen).")
        else:
            st.info("Keine geeignete Mehrphasenpumpe gefunden (oder p_req nicht bestimmbar).")

        # =========================
        # Diagramme
        # =========================
        st.subheader("Diagramme")
        if best_pump:
            Q_liq_plot_m3h = float(best_pump.get("Q_liq_m3h", best_pump["Q_m3h"]))
            q_liq_lmin_plot = m3h_to_lmin(Q_liq_plot_m3h)
        else:
            q_liq_lmin_plot = None
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

        # --- LÃ¶slichkeit ---
        if gas_medium == "Luft":
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve_air_corrected(T)
                        if q_liq_lmin_plot:
                            sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                            ax1.plot(p_arr, sol_lmin, "-", alpha=0.7, label=f"Luft (Gemisch) @ {T:.0f}Â°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve_air_corrected(temperature)
                if q_liq_lmin_plot:
                    sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                    ax1.plot(p_arr, sol_lmin, "-", label=f"Luft (Gemisch) @ {temperature:.0f}Â°C")

            for g, y in AIR_COMPONENTS:
                if show_temp_band:
                    for T in [temperature - 10, temperature, temperature + 10]:
                        if -10 <= T <= 150:
                            p_arr, sol_arr = solubility_diagonal_curve(g, T, y_gas=y)
                            if q_liq_lmin_plot:
                                corr_vals = np.array([air_solubility_correction(p, T) for p in p_arr])
                                sol_lmin = ((sol_arr * corr_vals) / 1000.0) * q_liq_lmin_plot
                                ax1.plot(p_arr, sol_lmin, "--", alpha=0.35, label="_nolegend_")
                else:
                    p_arr, sol_arr = solubility_diagonal_curve(g, temperature, y_gas=y)
                    if q_liq_lmin_plot:
                        corr_vals = np.array([air_solubility_correction(p, temperature) for p in p_arr])
                        sol_lmin = ((sol_arr * corr_vals) / 1000.0) * q_liq_lmin_plot
                        ax1.plot(p_arr, sol_lmin, "--", alpha=0.35, label="_nolegend_")

            if C_target_cm3N_L > 0 and q_liq_lmin_plot:
                C_target_lmin = (C_target_cm3N_L / 1000.0) * q_liq_lmin_plot
                ax1.axhline(C_target_lmin, linestyle=":", alpha=0.9, label="Q_gas_ziel (Luft)")
                ax1.text(13.8, C_target_lmin, "Q_gas_ziel (Luft)", va="center", ha="right", fontsize=8)

            if q_liq_lmin_plot:
                Csat_s_mix = air_solubility_cm3N_L(p_suction, temperature)
                ax1.scatter([p_suction], [(Csat_s_mix / 1000.0) * q_liq_lmin_plot], s=60, label="C_sat @ p_s (Luft)")

            if p_req is not None and C_target_cm3N_L > 0 and q_liq_lmin_plot:
                C_target_lmin = (C_target_cm3N_L / 1000.0) * q_liq_lmin_plot
                ax1.scatter([p_req], [C_target_lmin], s=110, marker="^", label="p_req (Luft)")
                ax1.axvline(p_req, linestyle=":", alpha=0.6, label="p_req")

        else:
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve(gas_medium, T, y_gas=1.0)
                        if q_liq_lmin_plot:
                            sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                            ax1.plot(p_arr, sol_lmin, "--", alpha=0.7,
                                     label=f"{gas_medium} @ {T:.0f}Â°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve(gas_medium, temperature, y_gas=1.0)
                if q_liq_lmin_plot:
                    sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                    ax1.plot(p_arr, sol_lmin, "--", label=f"{gas_medium} @ {temperature:.0f}Â°C")
                c_sat_curve_p = p_arr.tolist()
                c_sat_curve_c = sol_arr.tolist()

            if C_target_cm3N_L > 0 and q_liq_lmin_plot:
                C_target_lmin = (C_target_cm3N_L / 1000.0) * q_liq_lmin_plot
                ax1.axhline(C_target_lmin, linestyle=":", alpha=0.8)
                ax1.text(13.8, C_target_lmin, "Q_gas_ziel", va="center", ha="right", fontsize=9)

            if p_req is not None and C_target_cm3N_L > 0 and q_liq_lmin_plot:
                C_target_lmin = (C_target_cm3N_L / 1000.0) * q_liq_lmin_plot
                ax1.scatter([p_req], [C_target_lmin], s=110, marker="^", label="p_req")
                ax1.axvline(p_req, linestyle=":", alpha=0.6, label="p_req")

        if show_ref_targets and q_liq_lmin_plot:
            for Cref in [50.0, 100.0, 150.0]:
                if q_liq_lmin_plot:
                    Cref_lmin = (Cref / 1000.0) * q_liq_lmin_plot
                    ax1.axhline(Cref_lmin, linestyle=":", alpha=0.25)
                    ax1.text(13.8, Cref_lmin, f"{Cref_lmin:.1f} L/min", va="center", ha="right", fontsize=8)

        ax1.set_xlabel("Absolutdruck [bar]")
        ax1.set_ylabel("Gasvolumenstrom (Norm) [L/min]")
        ax1.set_title("LÃ¶slichkeit & Ziel: vollstÃ¤ndige LÃ¶sung")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlim(0, 14)
        if not q_liq_lmin_plot:
            ax1.text(0.5, 0.5, "Keine geeignete Pumpe / kein Q", ha="center", va="center", transform=ax1.transAxes)

        # --- Mehrphasen-Kennlinien ---
        if best_pump and dp_req is not None:
            pump = best_pump["pump"]
            Q_liq_m3h_sel_plot = float(best_pump.get("Q_liq_m3h", best_pump["Q_m3h"]))
            Q_lmin_sel = m3h_to_lmin(Q_liq_m3h_sel_plot)
            H_req_plot = dp_req
            H_avail_plot = best_pump["dp_avail"]
            conc_sel = float(best_pump.get("conc_pct", 0.0))
            n_ratio_sel = float(best_pump.get("n_ratio", 1.0))

            max_Q_lmin = 0.0
            max_H = 0.0

            for i, conc_key in enumerate(sorted(pump["curves_dp_vs_Q"].keys()), start=1):
                if conc_key > 30:
                    continue
                curve = pump["curves_dp_vs_Q"][conc_key]
                Q_curve, dp_curve = align_xy(curve["Q"], curve["dp"])
                Q_lmin = [m3h_to_lmin(q) for q in Q_curve]
                H_m = [float(dp) for dp in dp_curve]
                max_Q_lmin = max(max_Q_lmin, max(Q_lmin))
                max_H = max(max_H, max(H_m))
                ax2.plot(Q_lmin, H_m, "--", alpha=0.5, label=f"Kennlinie {i} ({conc_key:.0f}%)")

            if use_interpolated_gvf and conc_sel <= 30:
                # Interpolierte Kurve (BP liegt exakt darauf)
                base_keys = sorted(pump["curves_dp_vs_Q"].keys())
                base_curve = pump["curves_dp_vs_Q"][base_keys[0]]
                Q_interp = list(map(float, base_curve["Q"]))
                dp_interp = [_dp_at_Q_gvf(pump, q, conc_sel)[0] for q in Q_interp]
                Q_interp_scaled = [q * n_ratio_sel for q in Q_interp]
                H_interp = [dp * (n_ratio_sel ** 2) for dp in dp_interp]
                ax2.plot(
                    [m3h_to_lmin(q) for q in Q_interp_scaled],
                    H_interp,
                    "-",
                    linewidth=2.5,
                    label=f"Betriebskurve (interpoliert, {conc_sel:.1f}%, n={n_ratio_sel:.2f}Â·n0)"
                )
            elif not use_interpolated_gvf and conc_sel <= 30:
                # AusgewÃ¤hlte diskrete Kurve (BP liegt exakt darauf)
                sel_curve = pump["curves_dp_vs_Q"].get(conc_sel)
                if sel_curve:
                    Q_curve, dp_curve = align_xy(sel_curve["Q"], sel_curve["dp"])
                    Q_sel_curve = [q * n_ratio_sel for q in Q_curve]
                    H_sel_curve = [dp * (n_ratio_sel ** 2) for dp in dp_curve]
                    ax2.plot(
                        [m3h_to_lmin(q) for q in Q_sel_curve],
                        H_sel_curve,
                        "-",
                        linewidth=2.5,
                        label=f"Betriebskurve (ausgewÃ¤hlt, {conc_sel:.0f}%, n={n_ratio_sel:.2f}Â·n0)"
                    )

            ax2.scatter(Q_lmin_sel, H_avail_plot, s=110, marker="x", label="Betriebspunkt (auf Kennlinie)")
            ax2.scatter(Q_lmin_sel, H_req_plot, s=70, marker="o", facecolors="none", edgecolors="black", label="Anforderung (p_req)")
            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Î”p [bar]")
            ax2.set_title(f"Mehrphasen-Kennlinien: {pump['id']}")
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlim(0, max_Q_lmin * 1.1 if max_Q_lmin > 0 else 10)
            ax2.set_ylim(0, max_H * 1.1 if max_H > 0 else 10)
        else:
            ax2.text(0.5, 0.5, "Keine geeignete Pumpe / kein p_req", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Î”p [bar]")
            ax2.set_title("Mehrphasen-Kennlinien")
            ax2.grid(True)

        # --- LÃ¶slichkeit + Pumpenkennlinie (beide in L/min) ---
        c_sat_curve_p = None
        c_sat_curve_c = None
        p_abs_curve = None
        c_norm_curve = None
        q_gas_lmin_curve = None

        if gas_medium == "Luft":
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve_air_corrected(T)
                        if q_liq_lmin_plot:
                            sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                            ax3.plot(p_arr, sol_lmin, "-", alpha=0.7,
                                     label=f"LÃ¶slichkeit (Luft) @ {T:.0f}Â°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve_air_corrected(temperature)
                if q_liq_lmin_plot:
                    sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                    ax3.plot(p_arr, sol_lmin, "-", label=f"LÃ¶slichkeit (Luft) @ {temperature:.0f}Â°C")
                c_sat_curve_p = p_arr.tolist()
                c_sat_curve_c = sol_arr.tolist()

            for g, y in AIR_COMPONENTS:
                if show_temp_band:
                    for T in [temperature - 10, temperature, temperature + 10]:
                        if -10 <= T <= 150:
                            p_arr, sol_arr = solubility_diagonal_curve(g, T, y_gas=y)
                            if q_liq_lmin_plot:
                                corr_vals = np.array([air_solubility_correction(p, T) for p in p_arr])
                                sol_lmin = ((sol_arr * corr_vals) / 1000.0) * q_liq_lmin_plot
                                ax3.plot(p_arr, sol_lmin, "--", alpha=0.35, label="_nolegend_")
                else:
                    p_arr, sol_arr = solubility_diagonal_curve(g, temperature, y_gas=y)
                    if q_liq_lmin_plot:
                        corr_vals = np.array([air_solubility_correction(p, temperature) for p in p_arr])
                        sol_lmin = ((sol_arr * corr_vals) / 1000.0) * q_liq_lmin_plot
                        ax3.plot(p_arr, sol_lmin, "--", alpha=0.35, label="_nolegend_")
        else:
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_diagonal_curve(gas_medium, T, y_gas=1.0)
                        if q_liq_lmin_plot:
                            sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                            ax3.plot(p_arr, sol_lmin, "--", alpha=0.7,
                                     label=f"LÃ¶slichkeit ({gas_medium}) @ {T:.0f}Â°C")
            else:
                p_arr, sol_arr = solubility_diagonal_curve(gas_medium, temperature, y_gas=1.0)
                if q_liq_lmin_plot:
                    sol_lmin = (sol_arr / 1000.0) * q_liq_lmin_plot
                    ax3.plot(p_arr, sol_lmin, "--", label=f"LÃ¶slichkeit ({gas_medium}) @ {temperature:.0f}Â°C")
                c_sat_curve_p = p_arr.tolist()
                c_sat_curve_c = sol_arr.tolist()

        ax3.set_xlabel("Absolutdruck [bar]")
        ax3.set_ylabel("Gasvolumenstrom (Norm) [L/min]")
        ax3.set_title("LÃ¶slichkeit (umgerechnet) + Pumpenkennlinie (L/min)")
        ax3.grid(True)
        ax3.set_xlim(0, 14)

        if not q_liq_lmin_plot:
            ax3.text(0.5, 0.5, "Keine geeignete Pumpe / kein Q", ha="center", va="center", transform=ax3.transAxes)

        if q_liq_lmin_plot:
            ax3.axhline(C_ziel_lmin, linestyle=":", alpha=0.9, label="Q_gas_ziel")
            ax3.text(13.8, C_ziel_lmin, "Q_gas_ziel", va="center", ha="right", fontsize=8)
        if show_ref_targets:
            for Cref in [50.0, 100.0, 150.0]:
                if q_liq_lmin_plot:
                    Cref_lmin = (Cref / 1000.0) * q_liq_lmin_plot
                    ax3.axhline(Cref_lmin, linestyle=":", alpha=0.25)
                    ax3.text(13.8, Cref_lmin, f"{Cref_lmin:.1f} L/min", va="center", ha="right", fontsize=8)


        if best_pump and dp_req is not None:
            pump = best_pump["pump"]
            Q_liq_m3h_sel_plot = float(best_pump.get("Q_liq_m3h", best_pump["Q_m3h"]))
            conc_plot = float(best_pump.get("conc_pct", 0.0))
            n_ratio_sel = float(best_pump.get("n_ratio", 1.0))

            # Plot Pumpenkennlinien: Konzentration â†’ Gasmenge (Norm L/min)
            kennlinien_keys = sorted(pump["curves_dp_vs_Q"].keys())
            for conc_key in kennlinien_keys:
                if conc_key <= 30:
                    curve = pump["curves_dp_vs_Q"][conc_key]
                    Q_curve, dp_curve = align_xy(curve["Q"], curve["dp"])
                    Q_curve = [q * n_ratio_sel for q in Q_curve]
                    dp_curve = [dp * (n_ratio_sel ** 2) for dp in dp_curve]
            
                    # Konzentration in cmÂ³N/L
                    C_kennlinie = (float(conc_key) / 100.0) * 1000.0
                    
                    # FÃ¼r jeden Punkt auf der Kennlinie:
                    p_abs_curve = []
                    Q_gas_norm_curve = []
                    
                    for Q_liq, dp in zip(Q_curve, dp_curve):
                        # Druckseite:
                        p_discharge = float(p_suction) + dp
                        # Gasmenge (Norm L/min) aus Konzentration
                        Q_liq_lmin = m3h_to_lmin(Q_liq)
                        Q_gas_norm = (C_kennlinie / 1000.0) * Q_liq_lmin
                
                        p_abs_curve.append(p_discharge)
                        Q_gas_norm_curve.append(Q_gas_norm)
            
                    ax3.plot(p_abs_curve, Q_gas_norm_curve, "--", alpha=0.5, 
                            label=f"Kennlinie {conc_key:.0f}%")
            
            # Betriebspunkt markieren:
            C_kennlinie_bp = float(best_pump.get("C_kennlinie", 0.0))
            Q_gas_norm_bp = float(best_pump.get("Q_gas_kennlinie_lmin", 0.0))
            p_bp = float(best_pump.get("p_discharge", p_suction + best_pump["dp_avail"]))
            ax3.scatter([p_bp], [Q_gas_norm_bp], s=80, color="tab:red", marker="x", 
                       label="Betriebspunkt")

        ax3.legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)

        # =========================
        # Rechenweg (Mehrphase)
        # =========================
        with st.expander("Detaillierter Rechenweg (Mehrphase)"):
            st.markdown("### 0) Ãœberblick")
            st.markdown(
                "Ziel: Aus **Q_gas_ziel [L/min]**, Gas/Medium und Temperatur den erforderlichen Druck bestimmen, "
                "eine passende Kennlinie wÃ¤hlen und anschlieÃŸend die energetisch optimale Pumpe bestimmen."
            )
            st.markdown("**Hinweis:** LÃ¶slichkeit basiert auf vereinfachtem Henryâ€‘Modell mit $Z$â€‘NÃ¤herung.")
            st.markdown("**Neu:** Kennlinien-% = Gesamtluftkonzentration (nicht GVF), Q = FlÃ¼ssigkeitsstrom.")

            st.markdown("---")
            st.markdown("### 1) Eingaben & BasisgrÃ¶ÃŸen")
            st.markdown(f"- **p_s (fix):** {p_suction:.2f} bar(abs)")
            st.markdown(f"- **Q_gas_ziel:** {C_ziel_lmin:.1f} L/min (Norm)")
            if best_pump:
                Q_liq_lmin_rw = float(best_pump.get("Q_liq_lmin", 0.0))
                st.markdown(f"- FlÃ¼ssigkeitsstrom: **{Q_liq_lmin_rw:.1f} L/min**")
            st.markdown(f"- **Gas:** {gas_medium} | **Medium:** {liquid_medium} | **T:** {temperature:.1f} Â°C")
            st.caption("Alle DrÃ¼cke sind AbsolutdrÃ¼cke.")

            st.markdown("---")
            st.markdown("### 2) Kennlinien-Interpretation")
            st.latex(r"C_{kennlinie}[\%]=\frac{C_{gesamt}}{1000\,\text{cm}^3_N/L}\cdot 100")
            st.markdown("**Die Kennlinien-Prozente geben die Gesamtluftkonzentration an (gelÃ¶st + frei).**")
            st.markdown("- 100% entspricht 1000 cmÂ³N/L Luft in FlÃ¼ssigkeit")
            st.markdown("- Q auf der Kennlinie = FlÃ¼ssigkeitsstrom (ohne Gasphase)")
            if best_pump:
                st.markdown(f"- GewÃ¤hlte Kennlinie: **{best_pump.get('conc_pct', 0):.0f}%** = {best_pump.get('C_kennlinie', 0):.1f} cmÂ³N/L")
            st.caption("Diese Interpretation unterscheidet sich von GVF (freies Gas)!")

            st.markdown("---")
            st.markdown("### 3) LÃ¶slichkeit bei Druckseite (Henry)")
            st.latex(r"C_{sat}(p_{discharge},T)=\text{Henry-Modell}")
            if best_pump:
                p_d = best_pump.get("p_discharge", 0)
                C_sat_d = best_pump.get("C_sat_discharge", 0)
                st.markdown(f"- Druck Druckseite: **{p_d:.2f} bar(abs)**")
                st.markdown(f"- SÃ¤ttigungskonzentration bei diesem Druck: **{C_sat_d:.1f} cmÂ³N/L**")
            st.caption("Die Aufteilung in gelÃ¶st/frei erfolgt bei diesem Druck.")

            st.markdown("---")
            st.markdown("### 4) Aufteilung gelÃ¶st / frei (Druckseite)")
            st.latex(r"C_{dissolved}=\min(C_{kennlinie}, C_{sat}(p_{discharge}))")
            st.latex(r"C_{free}=\max(0, C_{kennlinie}-C_{sat}(p_{discharge}))")
            if best_pump:
                C_dis = best_pump.get("C_dissolved", 0)
                C_free = best_pump.get("C_free", 0)
                Q_dis = best_pump.get("Q_gas_dissolved_lmin", 0)
                Q_free = best_pump.get("Q_gas_free_lmin", 0)
                st.markdown(f"- GelÃ¶st: **{C_dis:.1f} cmÂ³N/L** â†’ **{Q_dis:.2f} L/min**")
                st.markdown(f"- Frei (Gasphase): **{C_free:.1f} cmÂ³N/L** â†’ **{Q_free:.2f} L/min**")
            st.caption("Nur das freie Gas bildet eine separate Gasphase im System.")

            st.markdown("---")
            st.markdown("### 5) Gasvolumenstrom-Bilanz")
            if best_pump:
                Q_gas_k = best_pump.get("Q_gas_kennlinie_lmin", 0)
                Q_gas_ziel = C_ziel_lmin
                st.markdown(f"- Gasvolumenstrom laut Kennlinie: **{Q_gas_k:.2f} L/min**")
                st.markdown(f"- Ziel-Gasvolumenstrom: **{Q_gas_ziel:.2f} L/min**")
                if Q_gas_k >= Q_gas_ziel:
                    st.success("Kennlinie foerdert mindestens die Zielmenge")
                else:
                    st.warning("Kennlinie unter Zielmenge")
            st.caption("Die Kennlinie muss mindestens die Ziel-Gasmenge foerdern koennen.")

            st.markdown("---")
            st.markdown("### 6) Pumpenauswahl & AffinitÃ¤tsgesetze")
            st.latex(r"\Delta p(Q,n)=\Delta p(Q/n,n_0)\cdot (n/n_0)^2")
            st.latex(r"P(n)=P(n_0)\cdot (n/n_0)^3")
            if best_pump:
                st.markdown(
                    f"- Auswahl: **{best_pump['pump']['id']}**, "
                    f"Q={best_pump['Q_m3h']:.2f} mÂ³/h, "
                    f"p_abs_avail={(p_suction + best_pump['dp_avail']):.2f} bar, "
                    f"P={best_pump['P_req']:.2f} kW, "
                    f"n={best_pump['n_rpm']:.0f} rpm ({best_pump['mode']})"
                )
                if dp_req is not None:
                    st.markdown(f"- Vergleich: p_req={p_req:.2f} bar â†” p_abs_avail={(p_suction + best_pump['dp_avail']):.2f} bar")
            else:
                st.markdown("- Keine geeignete Pumpe im Datensatz gefunden (oder p_req nicht bestimmbar).")
            st.caption("Die Auswahl minimiert eine gewichtete Zielfunktion; p_absâ€‘Abweichung wird mitbestraft.")

            st.markdown("---")
            st.markdown("### 7) Optimierungsziel (gewichtete Kombination)")
            st.markdown(
                "Die Auswahl minimiert eine gewichtete Zielfunktion aus **Energie**, **Wirkungsgrad** "
                "und **Gasmenge**, zusÃ¤tzlich mit kleinen Strafanteilen fÃ¼r DruckÃ¼berschuss und "
                "Drehzahlabweichung."
            )
            st.latex(r"\text{Score}=w_P\cdot P_{spec} + w_\eta\cdot(1-\eta_{est}) + w_g\cdot\varepsilon_{gas}")
            st.caption("$P_{spec}$ ist die spezifische Leistung, $\varepsilon_{gas}$ die Abweichung zur Zielâ€‘Gasmenge.")

    except Exception as e:
        show_error(e, "Mehrphasenpumpen")


def run_atex_selection():
    try:
        st.header("ATEX-Motorauslegung")

        with st.expander("ATEX-Eingaben â€“ aufklappen", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
                T_medium = st.number_input("Medientemperatur [Â°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)
            with c2:
                atmosphere = st.radio("AtmosphÃ¤re", ["Gas", "Staub"], index=0)
                if atmosphere == "Gas":
                    zone = st.selectbox("Zone", [0, 1, 2], index=2)
                else:
                    zone = st.selectbox("Zone", [20, 21, 22], index=2)
            with c3:
                t_margin = st.number_input("Temperaturabstand [K] (konservativ)", min_value=0.0, value=15.0, step=1.0)
                reserve_factor = st.number_input("Leistungsaufschlag [-] (z.B. 1.15)", min_value=1.0, value=1.15, step=0.01)

        st.subheader("Ergebnisse")

        if atmosphere == "Staub":
            st.error("âŒ Staub-Ex: Keine Motor-DatensÃ¤tze hinterlegt.")
            return
        if zone == 0:
            st.error("âŒ Zone 0: Keine Motor-DatensÃ¤tze hinterlegt.")
            return

        suitable = [m for m in ATEX_MOTORS if (zone in m["zone_suitable"])]
        if not suitable:
            st.error("âŒ Kein passender Motor-Datensatz fÃ¼r die gewÃ¤hlte Zone vorhanden.")
            return

        suitable = [m for m in suitable if (m["t_max_surface"] - t_margin) >= T_medium]
        if not suitable:
            st.error(f"âŒ Kein Motor verfÃ¼gbar fÃ¼r T_medium = {T_medium:.1f}Â°C (mit {t_margin:.0f} K Abstand).")
            return

        P_motor_min = P_req * float(reserve_factor)
        P_iec = motor_iec(P_motor_min)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wellenleistung", f"{P_req:.2f} kW")
        with c2:
            st.metric(f"Mindestleistung (+{(reserve_factor-1)*100:.0f}%)", f"{P_motor_min:.2f} kW")
        with c3:
            st.metric("IEC MotorgrÃ¶ÃŸe", f"{P_iec:.2f} kW")

        st.subheader("VerfÃ¼gbare ATEX-Motoren")
        selected = st.radio(
            "Motortyp wÃ¤hlen:",
            options=suitable,
            format_func=lambda x: f"{x['marking']} ({x['id']})"
        )
        st.success("âœ… GÃ¼ltige Konfiguration gefunden")

        with st.expander("Detaillierter Rechenweg (ATEX)"):
            st.markdown("### 1) Eingaben")
            st.markdown(f"- **Wellenleistung:** {P_req:.2f} kW")
            st.markdown(f"- **Medientemperatur:** {T_medium:.1f} Â°C")
            st.markdown(f"- **AtmosphÃ¤re / Zone:** {atmosphere} / Zone {zone}")
            st.markdown(f"- **Temperaturabstand Î”T:** {t_margin:.0f} K")
            st.markdown(f"- **Leistungsaufschlag:** {reserve_factor:.2f} (-)")
            st.caption("ATEXâ€‘Auswahl basiert auf Zone und Temperaturklasse.")

            st.markdown("### 2) Leistungsanforderung (MotorgrÃ¶ÃŸe)")
            st.latex(r"P_{motor,min}=f_{res}\cdot P_{welle}")
            st.markdown(f"- \(P_{{motor,min}} = {reserve_factor:.2f}\cdot {P_req:.2f} = {P_motor_min:.2f}\,kW\)")
            st.markdown(f"- **IEC-Stufe (nÃ¤chstgrÃ¶ÃŸer):** {P_iec:.2f} kW")
            st.caption("IECâ€‘Stufe ist die nÃ¤chsthÃ¶here Normleistung.")

            st.markdown("### 3) ATEX-ZulÃ¤ssigkeit (Zone)")
            st.markdown("- Filter: nur Motoren mit Freigabe fÃ¼r die gewÃ¤hlte Zone.")
            st.markdown(f"- Ergebnis: {len(suitable)} Motortyp(en) nach Zonenfilter.")

            st.markdown("### 4) Temperaturkriterium (OberflÃ¤chentemperatur)")
            st.latex(r"T_{surface,max}-T_{medium}\ge \Delta T")
            st.markdown(
                f"- Kriterium: \(T_{{surface,max}} - {T_medium:.1f} \ge {t_margin:.0f}\,K\)"
            )
            st.markdown(
                f"- Ergebnis: **{selected['t_max_surface']:.0f}Â°C** (Motor) â†’ Abstand = "
                f"**{selected['t_max_surface'] - T_medium:.1f} K**"
            )
            st.caption("So wird sichergestellt, dass die OberflÃ¤che die Temperaturklasse einhÃ¤lt.")

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
<tr><td>IEC MotorgrÃ¶ÃŸe</td><td>{P_iec:.2f} kW</td></tr>
<tr><td>AtmosphÃ¤re</td><td>{atmosphere}</td></tr>
<tr><td>Zone</td><td>{zone}</td></tr>
<tr><td>Motor</td><td>{selected['id']}</td></tr>
<tr><td>Kennzeichnung</td><td>{selected['marking']}</td></tr>
<tr><td>Max. OberflÃ¤chentemperatur</td><td>{selected['t_max_surface']:.1f} Â°C</td></tr>
<tr><td>Medientemperatur</td><td>{T_medium:.1f} Â°C</td></tr>
<tr><td>Temperaturabstand</td><td>{selected['t_max_surface'] - T_medium:.1f} K</td></tr>
</table>
<p>Hinweis: Bitte KonformitÃ¤t mit 2014/34/EU und EN 60079 prÃ¼fen.</p>
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
        st.title("ðŸ”§ Pumpenauslegungstool")

        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Seite auswÃ¤hlen:",
                ["ViskositÃ¤tsberechnung", "Mehrphasenpumpen", "ATEX-Auslegung"],
                index=0
            )

        if page == "ViskositÃ¤tsberechnung":
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

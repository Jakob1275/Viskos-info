# app.py — Streamlit Cloud lauffähig (Einphase + Mehrphase + ATEX)
# Fixes:
# - GVF an Druckseite = FREIES Gas an Druckseite (nicht "gesamt löslich")
# - Gesamtgas an Druckseite = gelöst (sol_d) + freies Gas (aus GVF)
# - Referenzlinien 10/15/20% im Löslichkeitsplot physikalisch korrekt:
#     -> als "Gesamtgas an Druckseite" (gelöst + freies Gas) in cm³N/L,
#        wobei das freie Gas aus GVF bei p_discharge/T berechnet wird.
# - Rechenweg/Markdown NULL-sicher
# - NPSH konsistent (bar -> Pa -> m) und Saugverluste abgezogen
# - Robust: Debug-Trace sichtbar (DEBUG=True)

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
BAR_TO_M_WATER = 10.21
N0_RPM_DEFAULT = 2900
BAR_TO_PA = 1e5

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
}

# sehr einfache Z-Approximation (nur Stabilität/Trend, kein EOS-Anspruch)
REAL_GAS_FACTORS = {
    "Luft": lambda p, T: max(0.85, 1.0 - 0.00008 * p),
    "Methan (CH4)": lambda p, T: max(0.80, 1.0 - 0.00015 * p),
    "CO2": lambda p, T: max(0.70, 0.90 + 0.00006 * (T - 273.15)),
}

# =========================
# Pumpendaten
# =========================
PUMPS = [
    {
        "id": "P1 (Edur LBU 20-100)",
        "Qw": [0, 10, 20, 30, 40, 50],
        "Hw": [30, 29, 27, 24, 20, 15],
        "eta": [0.35, 0.55, 0.65, 0.62, 0.55, 0.45],
        "Pw": [1.2, 2.8, 4.2, 5.5, 6.5, 7.2],
        "NPSHr": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
        "max_viscosity": 500,
        "max_density": 1200,
    },
]

MPH_PUMPS = [
    {
        "id": "MPH-40 (Edur MPH 40)",
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

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

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
    B = compute_B_HI(Q_vis_m3h, H_vis_m, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    Q_water = float(Q_vis_m3h)       # CQ ~ 1 (stabil)
    H_water = float(H_vis_m) / max(CH, 1e-9)
    return {"Q_water": Q_water, "H_water": H_water, "B": B, "CH": CH, "Ceta": Ceta}

def generate_viscous_curve(pump, nu_cSt, rho):
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    etaw = np.array(pump["eta"], dtype=float)

    H_vis, eta_vis, P_vis = [], [], []

    for q, h, e in zip(Qw, Hw, etaw):
        B = compute_B_HI(q if q > 0 else 1e-6, max(h, 1e-6), nu_cSt)
        CH, Ceta = viscosity_correction_factors(B)
        hv = h * CH
        ev = safe_clamp(e * Ceta, 0.05, 0.95)
        P_hyd_W = rho * G * (q / 3600.0) * hv
        pv = (P_hyd_W / max(ev, 1e-9)) / 1000.0
        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)

    return Qw.tolist(), np.array(H_vis).tolist(), np.array(eta_vis).tolist(), np.array(P_vis).tolist()

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
    p = max(float(p_bar_abs), 1e-6)
    T_K = float(T_celsius) + 273.15
    H = max(henry_constant(gas, T_celsius), 1e-12)
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)
    p_partial = safe_clamp(float(y_gas), 0.0, 1.0) * p

    # Henry
    C_mol_L = p_partial / H

    # Molvolumen oper (mit Z als einfache Korrektur)
    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    # oper -> normal
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L

def total_gas_from_gvf_free_out_cm3N_L(gvf_pct, p_bar_abs, T_celsius, gas):
    """
    GVF% an einer Stelle beschreibt freie Gasphase (operativ).
    Umrechnung in cm³N/L:
      Vgas_oper(L/Lliq) = gvf/(100-gvf)
      Vn(L/Lliq) = Voper * (p/Pn)*(Tn/T)*(1/Z)
    """
    gvf_pct = safe_clamp(float(gvf_pct), 0.0, 99.0)
    p = max(float(p_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vgas_oper_L_per_Lliq = gvf_pct / max(100.0 - gvf_pct, 1e-9)
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    Vn_L_per_L = Vgas_oper_L_per_Lliq * ratio
    return Vn_L_per_L * 1000.0  # cm³N/L

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

def solubility_diagonal_curve(gas, T_celsius, p_min=0.2, p_max=14.0, n=140):
    ps = np.linspace(p_min, p_max, n)
    sol = [gas_solubility_cm3N_per_L(gas, p, T_celsius) for p in ps]
    return ps, np.array(sol)

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
# Pump selection (Mehrphase) inkl. Drehzahl
# =========================
def choose_best_mph_pump(pumps, Q_req_m3h, dp_req_bar, gvf_free_pct, nu_cSt, rho_liq,
                        n_min_ratio=0.5, n_max_ratio=1.2):
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

            gvf_keys = sorted(pump["curves_dp_vs_Q"].keys())
            gvf_key = next((k for k in gvf_keys if k >= gvf_free_pct), gvf_keys[-1])

            curve = pump["curves_dp_vs_Q"][gvf_key]
            power_curve = pump["power_kW_vs_Q"][gvf_key]

            Qc = list(map(float, curve["Q"]))
            dpc = list(map(float, curve["dp"]))

            def dp_at_ratio(nr):
                if nr <= 0:
                    return 0.0
                Q_base = Q_req / nr
                dp_base = safe_interp(Q_base, Qc, dpc)
                return dp_base * (nr ** 2)

            def f(nr):
                return dp_at_ratio(nr) - dp_req

            dp_nom = None
            if min(Qc) <= Q_req <= max(Qc):
                dp_nom = safe_interp(Q_req, Qc, dpc)

            n_ratio = bisect_root(f, n_min_ratio, n_max_ratio, it=80, tol=1e-4)

            candidates = []

            # Nenndrehzahl
            if dp_nom is not None and dp_nom >= dp_req:
                P_nom = safe_interp(Q_req, power_curve["Q"], power_curve["P"])
                candidates.append({
                    "pump": pump,
                    "gvf_key": gvf_key,
                    "dp_avail": dp_nom,
                    "P_req": P_nom,
                    "n_ratio": 1.0,
                    "n_rpm": pump["n0_rpm"],
                    "mode": "Nenndrehzahl",
                })

            # Drehzahl angepasst
            if n_ratio is not None:
                Q_base = Q_req / n_ratio
                if min(Qc) <= Q_base <= max(Qc):
                    dp_scaled = dp_at_ratio(n_ratio)
                    if dp_scaled >= dp_req:
                        P_base = safe_interp(Q_base, power_curve["Q"], power_curve["P"])
                        P_scaled = P_base * (n_ratio ** 3)
                        candidates.append({
                            "pump": pump,
                            "gvf_key": gvf_key,
                            "dp_avail": dp_scaled,
                            "P_req": P_scaled,
                            "n_ratio": n_ratio,
                            "n_rpm": pump["n0_rpm"] * n_ratio,
                            "mode": "Drehzahl angepasst",
                        })

            for cand in candidates:
                score = abs(cand["dp_avail"] - dp_req) + 0.15 * abs(cand["n_ratio"] - 1.0)
                cand["score"] = score
                if best is None or score < best["score"]:
                    best = cand

        except Exception:
            continue

    return best

# =========================
# NPSH
# =========================
def calculate_npsh_available(p_suction_bar_abs, p_vapor_bar, H_suction_loss_m, rho, G):
    p_suction = float(p_suction_bar_abs) * BAR_TO_PA
    p_vapor = float(p_vapor_bar) * BAR_TO_PA
    loss = float(H_suction_loss_m)
    return max(0.0, (p_suction - p_vapor) / (float(rho) * float(G)) - loss)

# =========================
# Pages
# =========================
def run_single_phase_pump():
    try:
        st.header("Einphasenpumpen mit Viskositätskorrektur")

        with st.sidebar:
            st.header("Eingabeparameter (Einphase)")

            st.subheader("Betriebspunkt")
            Q_vis_req = st.number_input("Förderstrom [m³/h]", min_value=0.1, value=40.0, step=1.0)
            H_vis_req = st.number_input("Förderhöhe [m]", min_value=0.1, value=35.0, step=1.0)

            st.subheader("Medium")
            medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
            rho = st.number_input("Dichte [kg/m³]", min_value=1.0, value=float(MEDIA[medium]["rho"]), step=5.0)
            nu = st.number_input("Kinematische Viskosität [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.5)
            p_vapor = st.number_input("Dampfdruck [bar]", min_value=0.0, value=float(MEDIA[medium]["p_vapor"]), step=0.01)

            st.subheader("Optionen")
            allow_out = st.checkbox("Auswahl außerhalb Kennlinie zulassen", value=True)
            reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
            n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
            n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)
            H_suction_loss = st.number_input("Saugverluste [m]", min_value=0.0, value=1.0, step=0.1)
            p_suction_bar_abs = st.number_input("Saugdruck absolut [bar]", min_value=0.2, value=1.0, step=0.1)

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

        Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(pump, nu, rho)

        n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)

        n_opt_rpm = None
        P_opt_kW = None
        saving_pct = None
        if n_ratio_opt is not None:
            n_opt_rpm = N0_RPM_DEFAULT * n_ratio_opt
            Q_base = Q_vis_req / n_ratio_opt
            P_base = safe_interp(Q_base, Q_vis_curve, P_vis_curve)
            P_opt_kW = float(P_base) * (n_ratio_opt ** 3)
            P_nom_at_Q = safe_interp(Q_vis_req, Q_vis_curve, P_vis_curve)
            saving_pct = ((P_nom_at_Q - P_opt_kW) / P_nom_at_Q * 100.0) if P_nom_at_Q > 0 else 0.0

        npsh_available = calculate_npsh_available(
            p_suction_bar_abs=p_suction_bar_abs,
            p_vapor_bar=p_vapor,
            H_suction_loss_m=H_suction_loss,
            rho=rho,
            G=G
        )
        npsh_required = safe_interp(Q_vis_req, pump["Qw"], pump["NPSHr"])
        npsh_ok = npsh_available >= npsh_required

        st.subheader("Ergebnisse")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gewählte Pumpe", best["id"])
            st.metric("Q (viskos)", f"{Q_vis_req:.1f} m³/h")
        with col2:
            st.metric("η (viskos)", f"{eta_vis:.3f}")
            st.metric("H (viskos)", f"{H_vis_req:.1f} m")
        with col3:
            st.metric("Wellenleistung", f"{P_vis_kW:.2f} kW")
            st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")
        with col4:
            if n_ratio_opt is not None and saving_pct is not None:
                st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
                st.metric("Energieeinsparung", f"{saving_pct:.1f}%")
            else:
                st.warning("Keine optimale Drehzahl im gewählten Bereich gefunden.")

        if not best["in_range"]:
            st.warning(f"Betriebspunkt außerhalb Wasserkennlinie! Bewertung bei Q={best['Q_eval']:.1f} m³/h")

        if not npsh_ok:
            st.error(f"❌ NPSH-Problem: Verfügbar = {npsh_available:.2f} m, Erforderlich = {npsh_required:.2f} m")
        else:
            st.success(f"✅ NPSH OK: Verfügbar = {npsh_available:.2f} m, Erforderlich = {npsh_required:.2f} m")

        st.subheader("Kennlinien")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(pump["Qw"], pump["Hw"], "o-", label="Wasser (n0)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", label="Viskos (n0)")
        if n_ratio_opt is not None:
            Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            H_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
            ax1.plot(Q_scaled, H_scaled, ":", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")
        ax1.scatter([Q_water], [best["H_at"]], marker="^", s=90, label="BP (Wasser)")
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=90, label="BP (viskos)")
        ax1.set_xlabel("Q [m³/h]")
        ax1.set_ylabel("H [m]")
        ax1.set_title("Q-H")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(pump["Qw"], pump["eta"], "o-", label="Wasser (n0)")
        ax2.plot(Q_vis_curve, eta_vis_curve, "s--", label="Viskos (n0)")
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=90, label="η (viskos)")
        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η")
        ax2.grid(True)
        ax2.legend()

        ax3.plot(pump["Qw"], pump["Pw"], "o-", label="Wasser (P Daten)")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", label="Viskos (berechnet)")
        if n_ratio_opt is not None:
            P_scaled = [p * (n_ratio_opt ** 3) for p in P_vis_curve]
            Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            ax3.plot(Q_scaled, P_scaled, ":", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=90, label="BP (viskos)")
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Detaillierter Rechenweg"):
            st.markdown(f"""
### 1) Viskositätskorrektur (HI, stabil)
- Betriebspunkt: Q={Q_vis_req:.2f} m³/h, H={H_vis_req:.2f} m
- Kennzahl: B={B:.3f}
- Faktoren: C_H={CH:.3f}, C_η={Ceta:.3f}
- Wasserpunkt: Q_w={Q_water:.2f} m³/h, H_w={H_water:.2f} m

### 2) Leistung
- P_hyd={P_hyd_W:.0f} W
- P_welle={P_vis_kW:.2f} kW
- Motorreserve={reserve_pct}% -> IEC={P_motor_kW:.2f} kW

### 3) Drehzahloptimierung
""")
            if n_ratio_opt is not None and P_opt_kW is not None and saving_pct is not None:
                st.markdown(f"- n_opt={n_opt_rpm:.0f} rpm (n/n0={n_ratio_opt:.3f})")
                st.markdown(f"- P_opt={P_opt_kW:.2f} kW")
                st.markdown(f"- Einsparung={saving_pct:.1f}%")
            else:
                st.markdown("- Keine gültige optimale Drehzahl im Bereich gefunden.")
            st.markdown(f"""
### 4) NPSH
- NPSHa={npsh_available:.2f} m
- NPSHr={npsh_required:.2f} m
- Status: {"✅ OK" if npsh_ok else "❌ Problem"}
""")

    except Exception as e:
        show_error(e, "Einphasenpumpen")

def run_multi_phase_pump():
    try:
        st.header("Mehrphasenpumpen-Auslegung")

        with st.sidebar:
            st.header("Eingabeparameter (Mehrphase)")

            st.subheader("Prozessdaten")
            Q_req = st.number_input("Flüssigkeitsvolumenstrom [m³/h]", min_value=0.1, value=3.0, step=0.1)
            p_suction = st.number_input("Absolutdruck Saugseite [bar]", min_value=0.2, value=2.0, step=0.1)
            p_discharge = st.number_input("Absolutdruck Druckseite [bar]", min_value=0.2, value=7.0, step=0.1)
            dp_req = max(0.0, p_discharge - p_suction)
            H_suction_loss = st.number_input("Saugverluste [m]", min_value=0.0, value=0.0, step=0.1)

            st.subheader("Gasdefinition")
            gas_mode = st.radio("Gasvorgabe:", ["GVF an Druckseite [%]", "Recyclingstrom [m³/h]"], index=0)
            if gas_mode == "GVF an Druckseite [%]":
                gvf_out_pct = st.slider("GVF an Druckseite [%]", 0.0, 40.0, 5.4, 0.1)
                Q_rec = None
            else:
                Q_rec = st.number_input("Recyclingstrom [m³/h]", min_value=0.0, value=3.0, step=0.1)
                gvf_out_pct = None

            st.subheader("Medium")
            gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
            liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)
            temperature = st.number_input("Temperatur [°C]", min_value=-10.0, value=20.0, step=1.0)

            rho_liq = float(MEDIA[liquid_medium]["rho"])
            nu_liq = float(MEDIA[liquid_medium]["nu"])
            p_vapor = float(MEDIA[liquid_medium]["p_vapor"])

            st.subheader("Optionen")
            show_temp_band = st.checkbox("Temperaturband anzeigen", value=True)
            only_dissolved = st.checkbox("Nur gelöstes Gas zulassen", value=False)
            safety_factor = st.slider("Sicherheitsfaktor für GVF [%]", 0, 20, 10)

        # 1) Löslichkeit (cm³N/L)
        sol_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature)
        sol_d = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature)

        # 2) Gesamtgas an Druckseite (cm³N/L)
        frac = 0.0
        has_free_gas_discharge = False

        if gas_mode == "GVF an Druckseite [%]":
            # GVF_out ist FREIES Gas an Druckseite (operativ) -> Norm
            free_out_cm3N_L = total_gas_from_gvf_free_out_cm3N_L(gvf_out_pct, p_discharge, temperature, gas_medium)

            # Gesamtgas an Druckseite = gelöst (bis sol_d) + freies Gas
            total_cm3N_L = sol_d + free_out_cm3N_L
            has_free_gas_discharge = gvf_out_pct > 0.0

        else:
            frac = (Q_rec / Q_req) if Q_req > 0 else 0.0
            frac = safe_clamp(frac, 0.0, 5.0)
            # Recycling trägt proportional "druckseitig gelöstes" Gas in den Zulauf
            total_cm3N_L = sol_d * frac
            has_free_gas_discharge = False

        # 3) Freies Gas an Saugseite
        free_cm3N_L = max(0.0, total_cm3N_L - sol_s)
        gvf_free_pct = free_gas_gvf_pct_at_suction_from_cm3N_L(free_cm3N_L, p_suction, temperature, gas_medium)

        # 4) Sicherheitsaufschlag
        gvf_free_pct_safe = gvf_free_pct * (1.0 + safety_factor / 100.0)

        # 5) Pumpenauswahl
        best_pump = None
        if only_dissolved and free_cm3N_L > 0:
            best_pump = None
        else:
            best_pump = choose_best_mph_pump(
                MPH_PUMPS, Q_req, dp_req, gvf_free_pct_safe, nu_liq, rho_liq
            )

        # 6) NPSH
        npsh_available = calculate_npsh_available(
            p_suction_bar_abs=p_suction,
            p_vapor_bar=p_vapor,
            H_suction_loss_m=H_suction_loss,
            rho=rho_liq,
            G=G
        )
        npsh_required = best_pump["pump"].get("NPSHr", 2.5) if best_pump else 2.5
        npsh_ok = (npsh_available >= npsh_required) if best_pump else False

        # =========================
        # Ergebnisse
        # =========================
        st.subheader("Ergebnisse")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Flüssigkeitsstrom", f"{Q_req:.1f} m³/h")
            st.metric("Δp Anforderung", f"{dp_req:.2f} bar")
        with col2:
            st.metric("Löslichkeit Saugseite", f"{sol_s:.1f} cm³N/L")
            st.metric("Löslichkeit Druckseite", f"{sol_d:.1f} cm³N/L")
        with col3:
            st.metric("Gesamtgas (Druckseite)", f"{total_cm3N_L:.1f} cm³N/L")
            if gas_mode != "GVF an Druckseite [%]":
                st.metric("Recycling-Anteil", f"{frac:.2f}")
        with col4:
            st.metric("Freies Gas (Saugseite)", f"{free_cm3N_L:.1f} cm³N/L")
            st.metric("GVF frei (mit Sicherheit)", f"{gvf_free_pct_safe:.1f} %")

        # Meldungen
        if has_free_gas_discharge:
            st.info("ℹ️ An der Druckseite ist gemäß Eingabe eine freie Gasphase vorhanden (GVF > 0).")
        if free_cm3N_L > 0:
            st.warning("⚠️ An der Saugseite entsteht freie Gasphase (Mehrphasenbetrieb relevant).")
        if only_dissolved and free_cm3N_L > 0:
            st.error("❌ Auswahl blockiert: Nur gelöstes Gas ist zugelassen!")

        if best_pump:
            if not npsh_ok:
                st.error(f"❌ NPSH-Problem: Verfügbar = {npsh_available:.2f} m, Erforderlich = {npsh_required:.2f} m")
            else:
                st.success(f"✅ Empfohlene Pumpe: {best_pump['pump']['id']} (Kurve {best_pump['gvf_key']}% GVF)")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Δp verfügbar", f"{best_pump['dp_avail']:.2f} bar")
            with c2:
                st.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
            with c3:
                st.metric("Drehzahl", f"{best_pump['n_rpm']:.0f} rpm")
            with c4:
                st.metric("Modus", best_pump["mode"])
        else:
            st.info("Keine geeignete Mehrphasenpumpe gefunden (oder Auswahl blockiert).")

        # =========================
        # Diagramme
        # =========================
        st.subheader("Diagramme")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Löslichkeit-Kurven
        if show_temp_band:
            for T in [temperature - 10, temperature, temperature + 10]:
                if -10 <= T <= 150:
                    p_arr, sol_arr = solubility_diagonal_curve(gas_medium, T)
                    ax1.plot(p_arr, sol_arr, "--", label=f"{T:.0f}°C")
        else:
            p_arr, sol_arr = solubility_diagonal_curve(gas_medium, temperature)
            ax1.plot(p_arr, sol_arr, "--", label=f"{temperature:.0f}°C")

        # Referenzlinien 10/15/20% GVF (physikalisch konsistent!)
        # Wir interpretieren die Linien als "Gesamtgas an Druckseite" für diese GVF-Werte:
        # total_ref = sol_d(p) + free_out_from_gvf(p)  (bei gewählter Temperatur/Gas)
        gvf_refs = [10, 15, 20]
        for gvf in gvf_refs:
            free_ref = total_gas_from_gvf_free_out_cm3N_L(gvf, p_discharge, temperature, gas_medium)
            total_ref = sol_d + free_ref
            ax1.axhline(total_ref, linestyle=":", alpha=0.6)
            ax1.text(13.8, total_ref, f"{gvf}% GVF (Druck)", va="center", ha="right", fontsize=9)

        # Punkte
        ax1.scatter([p_suction], [sol_s], s=80, label="Saugseite (löslich)")
        ax1.scatter([p_discharge], [sol_d], s=80, label="Druckseite (löslich)")
        ax1.scatter([p_discharge], [total_cm3N_L], s=90, marker="x", label="Gesamtgas (Druckseite)")

        ax1.set_xlabel("Absolutdruck [bar]")
        ax1.set_ylabel("Gasgehalt [cm³N/L]")
        ax1.set_title(f"Gaslöslichkeit von {gas_medium} (Henry, cm³N/L)")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlim(0, 14)

        # Pumpenkennlinien (als Förderhöhe)
        if best_pump:
            pump = best_pump["pump"]
            Q_lmin_req = m3h_to_lmin(Q_req)
            H_req_m = dp_req * BAR_TO_M_WATER

            max_Q_lmin = 0.0
            max_H = 0.0

            for gvf_key in sorted(pump["curves_dp_vs_Q"].keys()):
                curve = pump["curves_dp_vs_Q"][gvf_key]
                Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
                H_m = [dp * BAR_TO_M_WATER for dp in curve["dp"]]
                max_Q_lmin = max(max_Q_lmin, max(Q_lmin))
                max_H = max(max_H, max(H_m))

                if gvf_key == best_pump["gvf_key"]:
                    ax2.plot(Q_lmin, H_m, "o-", linewidth=2, label=f"{gvf_key}% GVF (ausgewählt)")
                else:
                    ax2.plot(Q_lmin, H_m, "--", alpha=0.5, label=f"{gvf_key}% GVF")

            ax2.scatter(Q_lmin_req, H_req_m, s=90, marker="x", label="Betriebspunkt")

            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Förderhöhe [m]")
            ax2.set_title(f"Mehrphasen-Kennlinien: {pump['id']}")
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlim(0, max_Q_lmin * 1.1 if max_Q_lmin > 0 else 10)
            ax2.set_ylim(0, max_H * 1.1 if max_H > 0 else 10)
        else:
            ax2.text(0.5, 0.5, "Keine geeignete Pumpe", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_xlabel("Volumenstrom [L/min]")
            ax2.set_ylabel("Förderhöhe [m]")
            ax2.set_title("Mehrphasen-Kennlinien")
            ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Detaillierter Rechenweg"):
            st.markdown(f"""
### 1) Henry/Löslichkeit
- Gas: **{gas_medium}**, Flüssigkeit: **{liquid_medium}**
- T = **{temperature:.1f}°C**
- p_saug = **{p_suction:.2f} bar**, p_druck = **{p_discharge:.2f} bar**
- Löslichkeit Saugseite: **{sol_s:.1f} cm³N/L**
- Löslichkeit Druckseite: **{sol_d:.1f} cm³N/L**

### 2) Gesamtgas an Druckseite
""")
            if gas_mode == "GVF an Druckseite [%]":
                free_out_cm3N_L = total_gas_from_gvf_free_out_cm3N_L(gvf_out_pct, p_discharge, temperature, gas_medium)
                st.markdown(f"- Eingabe: GVF_druck={gvf_out_pct:.1f}% (freie Gasphase)")
                st.markdown(f"- Freies Gas an Druckseite (Norm): **{free_out_cm3N_L:.1f} cm³N/L**")
                st.markdown(f"- Gesamtgas (Druckseite) = gelöst ({sol_d:.1f}) + frei ({free_out_cm3N_L:.1f}) = **{total_cm3N_L:.1f} cm³N/L**")
            else:
                st.markdown(f"- Recyclingstrom={Q_rec:.2f} m³/h → Anteil={frac:.2f}")
                st.markdown(f"- Gesamtgas (Druckseite, angenähert) = sol_d * Anteil = **{total_cm3N_L:.1f} cm³N/L**")

            st.markdown(f"""
### 3) Freies Gas an Saugseite
- Freies Gas (Norm): **{free_cm3N_L:.1f} cm³N/L**
- GVF frei (Saugseite, oper): **{gvf_free_pct:.2f}%**
- Sicherheitsaufschlag: **{safety_factor}%** → **{gvf_free_pct_safe:.2f}%**

### 4) Pumpenauswahl
""")
            if best_pump:
                st.markdown(f"""
- Pumpe: **{best_pump['pump']['id']}**
- Kurve: **{best_pump['gvf_key']}% GVF**
- Δp verfügbar: **{best_pump['dp_avail']:.2f} bar**
- Leistung: **{best_pump['P_req']:.2f} kW**
- Drehzahl: **{best_pump['n_rpm']:.0f} rpm** ({best_pump['mode']})
""")
            else:
                st.markdown("- Keine geeignete Pumpe (oder Auswahl blockiert).")

            st.markdown(f"""
### 5) NPSH
- NPSHa={npsh_available:.2f} m
- NPSHr={npsh_required:.2f} m
- Status: {"✅ OK" if npsh_ok else "❌ Problem"}
""")

    except Exception as e:
        show_error(e, "Mehrphasenpumpen")

def run_atex_selection():
    try:
        st.header("ATEX-Motorauslegung")

        with st.sidebar:
            st.header("Eingabeparameter (ATEX)")

            st.subheader("Prozessdaten")
            P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
            T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

            st.subheader("Ex-Zone")
            atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"], index=0)
            if atmosphere == "Gas":
                zone = st.selectbox("Zone", [0, 1, 2], index=2)
            else:
                zone = st.selectbox("Zone", [20, 21, 22], index=2)

        st.subheader("Ergebnisse")

        if atmosphere == "Staub":
            st.error("❌ Staub-Ex: Keine Motor-Datensätze hinterlegt.")
            return

        if zone == 0:
            st.error("❌ Zone 0: Keine Motor-Datensätze hinterlegt.")
            return

        st.success(f"✅ Zone {zone} (Gas) ist grundsätzlich abbildbar.")

        t_margin = 15.0
        suitable = [
            m for m in ATEX_MOTORS
            if (zone in m["zone_suitable"]) and ((m["t_max_surface"] - t_margin) >= T_medium)
        ]

        if not suitable:
            st.error(f"❌ Kein Motor verfügbar für T_medium = {T_medium:.1f}°C (mit 15 K Abstand).")
            return

        P_motor_min = P_req * 1.15
        P_iec = motor_iec(P_motor_min)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wellenleistung", f"{P_req:.2f} kW")
        with c2:
            st.metric("Mindestleistung (+15%)", f"{P_motor_min:.2f} kW")
        with c3:
            st.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

        st.subheader("Verfügbare ATEX-Motoren")
        selected = st.radio(
            "Motortyp wählen:",
            options=suitable,
            format_func=lambda x: f"{x['marking']} ({x['id']})"
        )

        st.success("✅ Gültige Konfiguration gefunden")

        with st.expander("Technische Details"):
            st.markdown(f"""
- **Leistung:** {P_iec:.2f} kW
- **Kennzeichnung:** `{selected['marking']}`
- **Max. Oberflächentemperatur:** {selected['t_max_surface']:.1f}°C ({selected['temp_class']})
- **Medientemperatur:** {T_medium:.1f}°C
- **Temperaturabstand:** {selected['t_max_surface'] - T_medium:.1f} K (Anforderung: ≥ 15 K)
- **Kategorie:** {selected['category']}
- **Wirkungsgradklasse:** {selected.get('efficiency_class', 'N/A')}
- **Geeignet für Zone:** {', '.join(map(str, selected['zone_suitable']))}
""")

        if st.button("ATEX-Dokumentation exportieren"):
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>ATEX-Auslegung - {selected['id']}</title>
  <style>
    body {{ font-family: Arial; max-width: 900px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #2c3e50; }}
    .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>ATEX-Motorauslegung</h1>
  <p>Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>

  <h2>Eingaben</h2>
  <div class="metric">
    <strong>P:</strong> {P_req:.2f} kW<br/>
    <strong>T:</strong> {T_medium:.1f} °C<br/>
    <strong>Zone:</strong> {zone} (Gas)
  </div>

  <h2>Ergebnisse</h2>
  <table>
    <tr><th>Parameter</th><th>Wert</th></tr>
    <tr><td>Erforderliche Wellenleistung</td><td>{P_req:.2f} kW</td></tr>
    <tr><td>Mindestleistung (+15%)</td><td>{P_motor_min:.2f} kW</td></tr>
    <tr><td>IEC Motorgröße</td><td>{P_iec:.2f} kW</td></tr>
    <tr><td>Gewählter Motor</td><td>{selected['id']}</td></tr>
    <tr><td>Kennzeichnung</td><td>{selected['marking']}</td></tr>
    <tr><td>Max. Oberflächentemperatur</td><td>{selected['t_max_surface']:.1f}°C</td></tr>
    <tr><td>Temperaturabstand</td><td>{selected['t_max_surface'] - T_medium:.1f} K</td></tr>
    <tr><td>Kategorie</td><td>{selected['category']}</td></tr>
    <tr><td>Wirkungsgradklasse</td><td>{selected.get('efficiency_class','N/A')}</td></tr>
    <tr><td>Geeignet für Zone</td><td>{', '.join(map(str, selected['zone_suitable']))}</td></tr>
  </table>

  <h2>Hinweis</h2>
  <p>Bitte Konformität mit 2014/34/EU und EN 60079 prüfen.</p>
</body>
</html>
"""
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
                ["Einphasenpumpen (Viskosität)", "Mehrphasenpumpen", "ATEX-Auslegung"],
                index=1
            )

        if page == "Einphasenpumpen (Viskosität)":
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

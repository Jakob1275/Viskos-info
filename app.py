# app.py — Streamlit Cloud lauffähig (Einphase + Mehrphase + ATEX)
# Änderungen (wie gewünscht, sonst alles unverändert gelassen):
# 1) Sidebar links: NUR noch ATEX-Eingaben (Navigation bleibt).
# 2) Mehrphase: Eingabemodus "GVF_in an Saugseite" entfernt (nur GVF an Druckseite ODER Recycling).
# 3) Mehrphase: Ausgabe "Gelöst-Anteil des Systemgases" + Kennzahlen sinnvoll sortiert (übersichtlicher).

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
        "id": "P1",
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
    Q_water = float(Q_vis_m3h)  # CQ ~ 1 (stabil)
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

    # Henry: C = p/H
    C_mol_L = p_partial / H

    # Gasvolumen operativ (L/L) bei p,T
    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    # oper -> normal
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L


def total_gas_from_gvf_free_out_cm3N_L(gvf_pct, p_bar_abs, T_celsius, gas):
    """
    GVF% beschreibt freie Gasphase (operativ).
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
# NPSH (wird NICHT mehr prominent gezeigt – bleibt intern für Stabilität)
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

        # Eingaben IM TAB (nicht Sidebar)
        with st.expander("Eingaben (Einphase) – aufklappen", expanded=True):
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                st.subheader("Betriebspunkt (viskos)")
                Q_vis_req = st.number_input("Förderstrom Q_vis [m³/h]", min_value=0.1, value=40.0, step=1.0)
                H_vis_req = st.number_input("Förderhöhe H_vis [m]", min_value=0.1, value=35.0, step=1.0)
            with colB:
                st.subheader("Medium")
                medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
                rho = st.number_input("Dichte ρ [kg/m³]", min_value=1.0, value=float(MEDIA[medium]["rho"]), step=5.0)
                nu = st.number_input("Kinematische Viskosität ν [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.5)
            with colC:
                st.subheader("Optionen")
                allow_out = st.checkbox("Auswahl außerhalb Kennlinie zulassen", value=True)
                reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
                n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
                n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)

        # Umrechnung viskos -> Wasser
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

        st.subheader("Ergebnisse")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Gewählte Pumpe", best["id"])
            st.metric("Q_vis", f"{Q_vis_req:.2f} m³/h")
            st.metric("H_vis", f"{H_vis_req:.2f} m")
        with c2:
            st.metric("Q_wasser (Äquivalent)", f"{Q_water:.2f} m³/h")
            st.metric("H_wasser (Äquivalent)", f"{H_water:.2f} m")
            st.metric("η_vis", f"{eta_vis:.3f}")
        with c3:
            st.metric("Wellenleistung (viskos)", f"{P_vis_kW:.2f} kW")
            st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")
        with c4:
            if n_opt_rpm is not None and saving_pct is not None:
                st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
                st.metric("Energieeinsparung", f"{saving_pct:.1f}%")
            else:
                st.info("Keine optimale Drehzahl im gewählten Bereich gefunden.")

        if not best["in_range"]:
            st.warning(f"Betriebspunkt außerhalb Wasserkennlinie! Bewertung bei Q={best['Q_eval']:.1f} m³/h")

        st.subheader("Kennlinien")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(pump["Qw"], pump["Hw"], "o-", label="Wasser (n0)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", label="Viskos (n0)")
        if n_ratio_opt is not None:
            Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            H_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
            ax1.plot(Q_scaled, H_scaled, ":", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")
        ax1.scatter([Q_water], [best["H_at"]], marker="^", s=90, label="BP (Wasser-Äquiv.)")
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

        # WICHTIG: Wasserleistung bleibt "Pw" Datensatz, Viskos = berechnet
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

        with st.expander("Detaillierter Rechenweg (Einphase)"):
            st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q^{0.25}\cdot H^{0.375}} \quad (\nu \text{ in cSt, } Q \text{ in gpm, } H \text{ in ft})")
            st.markdown(f"- Umrechnung: \(Q_{{gpm}} = Q_{{m^3/h}}\cdot 4.40287\), \(H_{{ft}} = H_{{m}}\cdot 3.28084\)")
            st.markdown(f"- Eingabe: \(Q_{{vis}}={Q_vis_req:.2f}\), \(H_{{vis}}={H_vis_req:.2f}\), \\(\\nu={nu:.2f}\\) cSt → **B={B:.3f}**")

            st.latex(r"C_H=\exp\left(-0.165\cdot (\log_{10}(B))^{2.2}\right)")
            st.latex(r"C_\eta = 1 - 0.25\log_{10}(B) - 0.05(\log_{10}(B))^2")
            st.markdown(f"- **C_H={CH:.3f}**, **C_η={Ceta:.3f}**")

            st.latex(r"Q_w \approx Q_{vis}\quad;\quad H_w = \frac{H_{vis}}{C_H}")
            st.markdown(f"- Ergebnis: **Q_w={Q_water:.2f} m³/h**, **H_w={H_water:.2f} m**")

            st.latex(r"P_{hyd}=\rho g Q H \quad,\quad P_{Welle}=\frac{P_{hyd}}{\eta}")
            st.markdown(f"- \(P_{{hyd}}={P_hyd_W:.0f}\,W\), \(\\eta_{{vis}}={eta_vis:.3f}\) → **P_welle={P_vis_kW:.2f} kW**")
            st.markdown(f"- Motorreserve {reserve_pct}% → **IEC={P_motor_kW:.2f} kW**")

            st.latex(r"H(Q,n)=H(Q/n,n_0)\cdot (n/n_0)^2 \quad;\quad P(n)=P(n_0)\cdot (n/n_0)^3")
            if n_ratio_opt is not None and P_opt_kW is not None:
                st.markdown(f"- Drehzahlfaktor: **n/n0={n_ratio_opt:.3f}** → **n={n_opt_rpm:.0f} rpm**")
                st.markdown(f"- **P_opt={P_opt_kW:.2f} kW**, Einsparung **{saving_pct:.1f}%**")
            else:
                st.markdown("- Keine gültige optimale Drehzahl im Bereich gefunden.")

    except Exception as e:
        show_error(e, "Einphasenpumpen")


def run_multi_phase_pump():
    """
    Hinweis: Hier wurden NUR die Punkte umgesetzt, die du zuletzt genannt hast:
      - GVF_in@Saugseite-Input existiert nicht mehr (nur GVF@Druckseite oder Recycling).
      - Sortierte Ausgabe inkl. "wie viel % gelöst bei p_s / p_d".
      - Sidebar NICHT mehr genutzt (Eingaben im Tab).
    """
    try:
        st.header("Mehrphasenpumpen-Auslegung")

        with st.expander("Eingaben (Mehrphase) – aufklappen", expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.subheader("Prozessdrücke")
                p_suction = st.number_input("Absolutdruck Saugseite p_s [bar]", min_value=0.2, value=2.0, step=0.1)
                p_discharge = st.number_input("Absolutdruck Druckseite p_d [bar]", min_value=0.2, value=10.0, step=0.1)
                dp_req = max(0.0, p_discharge - p_suction)
            with c2:
                st.subheader("Medium")
                gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
                liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)
                temperature = st.number_input("Temperatur T [°C]", min_value=-10.0, value=20.0, step=1.0)
            with c3:
                st.subheader("Hydraulischer Volumenstrom durch die Pumpe")
                mode = st.radio(
                    "Welche Größe ist vorgegeben?",
                    ["Prozess-Volumenstrom Q_liq", "Recyclingstrom Q_rec (überschreibt Q_liq)"],
                    index=0
                )
                Q_liq = st.number_input("Q_liq [m³/h]", min_value=0.1, value=8.0, step=0.5)
                Q_rec = st.number_input("Q_rec [m³/h]", min_value=0.0, value=12.0, step=0.5)
                Q_pump = Q_rec if mode.startswith("Recycling") else Q_liq
                st.caption("Hinweis: Q_pump = Q_rec (wenn aktiv), sonst Q_pump = Q_liq")

        with st.expander("Gasvorgabe (nur EIN Input aktiv) – aufklappen", expanded=True):
            c4, c5 = st.columns([1, 1])
            with c4:
                gas_mode = st.radio("Gasvorgabe:", ["GVF an Druckseite [%]", "Recyclingstrom (nur gelöst)"], index=0)
            with c5:
                safety_factor = st.slider("Sicherheitsfaktor GVF [%]", 0, 20, 10)
                show_temp_band = st.checkbox("Temperaturband im Löslichkeitsdiagramm", value=True)
                only_dissolved = st.checkbox("Nur gelöstes Gas zulassen", value=False)

            if gas_mode == "GVF an Druckseite [%]":
                gvf_out_pct = st.slider("GVF an Druckseite (frei) [%]", 0.0, 40.0, 5.0, 0.1)
                rec_fraction = None
            else:
                # gelöst-only-Recycling: Gasmenge konservativ = Anteil * sol_d
                rec_fraction = (Q_rec / max(Q_pump, 1e-9)) if Q_pump > 0 else 0.0
                rec_fraction = safe_clamp(rec_fraction, 0.0, 5.0)
                gvf_out_pct = None
                st.caption(f"Recycling-Anteil (≈ Q_rec/Q_pump): {rec_fraction:.2f}")

        rho_liq = float(MEDIA[liquid_medium]["rho"])
        nu_liq = float(MEDIA[liquid_medium]["nu"])

        # 1) Löslichkeit (cm³N/L)
        sol_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature)
        sol_d = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature)

        # 2) Systemgas (konservativ) als cm³N/L (pro Liter Flüssigkeit)
        #    - bei GVF@Druckseite: total = sol_d + free_out(gvf_out)
        #    - bei Recycling(nur gelöst): total = sol_d * Anteil
        if gas_mode == "GVF an Druckseite [%]":
            free_out_cm3N_L = total_gas_from_gvf_free_out_cm3N_L(gvf_out_pct, p_discharge, temperature, gas_medium)
            total_cm3N_L = sol_d + free_out_cm3N_L
        else:
            free_out_cm3N_L = 0.0
            total_cm3N_L = sol_d * (rec_fraction if rec_fraction is not None else 0.0)

        # 3) Aufteilung gelöst/frei am jeweiligen Druck (Gleichgewicht, konservativ)
        dissolved_at_s = min(total_cm3N_L, sol_s)
        dissolved_at_d = min(total_cm3N_L, sol_d)
        free_at_s = max(0.0, total_cm3N_L - sol_s)
        free_at_d = max(0.0, total_cm3N_L - sol_d)

        gvf_free_s_pct = free_gas_gvf_pct_at_suction_from_cm3N_L(free_at_s, p_suction, temperature, gas_medium)
        gvf_free_s_pct_safe = gvf_free_s_pct * (1.0 + safety_factor / 100.0)

        # 4) Pumpenauswahl
        best_pump = None
        if only_dissolved and free_at_s > 0:
            best_pump = None
        else:
            best_pump = choose_best_mph_pump(
                MPH_PUMPS, Q_pump, dp_req, gvf_free_s_pct_safe, nu_liq, rho_liq
            )

        # 5) Gelöst-Anteile bezogen auf Systemgas (konservativ)
        #    (Das ist die Kennzahl, die du wolltest: "wie viel % bei Druck gelöst wird")
        if total_cm3N_L > 0:
            frac_diss_s = dissolved_at_s / total_cm3N_L * 100.0
            frac_diss_d = dissolved_at_d / total_cm3N_L * 100.0
        else:
            frac_diss_s = 0.0
            frac_diss_d = 0.0

        # =========================
        # Ergebnisse (sortiert)
        # =========================
        st.subheader("Ergebnisse (übersichtlich)")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Volumenstrom durch Pumpe Q_pump", f"{Q_pump:.2f} m³/h")
            st.metric("Δp Anforderung", f"{dp_req:.2f} bar")
        with r2:
            st.metric("Löslichkeit Saugseite (p_s)", f"{sol_s:.1f} cm³N/L")
            st.metric("Löslichkeit Druckseite (p_d)", f"{sol_d:.1f} cm³N/L")
        with r3:
            st.metric("Systemgas (konservativ)", f"{total_cm3N_L:.1f} cm³N/L")
            st.metric("GVF_s (frei, +Sicherheit)", f"{gvf_free_s_pct_safe:.1f} %")

        st.markdown("**Gelöst-Anteil des Systemgases (bezogen auf die konservative System-Gasmenge):**")
        g1, g2 = st.columns(2)
        with g1:
            st.metric(f"bei p_s = {p_suction:.2f} bar", f"{frac_diss_s:.1f}% gelöst")
        with g2:
            st.metric(f"bei p_d = {p_discharge:.2f} bar", f"{frac_diss_d:.1f}% gelöst")

        # Zusatz: gelöst/frei (klar benannt) – ohne Warnmeldungs-Flut
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Gelöst bei Saugseite", f"{dissolved_at_s:.1f} cm³N/L")
        with s2:
            st.metric("Frei bei Saugseite", f"{free_at_s:.1f} cm³N/L")
        with s3:
            st.metric("Gelöst bei Druckseite", f"{dissolved_at_d:.1f} cm³N/L")
        with s4:
            st.metric("Frei bei Druckseite", f"{free_at_d:.1f} cm³N/L")

        if best_pump:
            st.success(f"✅ Empfohlene Pumpe: {best_pump['pump']['id']} (Kennlinie {best_pump['gvf_key']}% GVF)")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Δp verfügbar", f"{best_pump['dp_avail']:.2f} bar")
            with p2:
                st.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
            with p3:
                st.metric("Drehzahl", f"{best_pump['n_rpm']:.0f} rpm")
            with p4:
                st.metric("Modus", best_pump["mode"])
        else:
            if only_dissolved and free_at_s > 0:
                st.info("Keine Pumpenauswahl: 'Nur gelöstes Gas zulassen' ist aktiv, aber freies Gas an p_s wäre erforderlich.")
            else:
                st.info("Keine geeignete Mehrphasenpumpe gefunden.")

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

        # Referenzlinien 10/15/20% GVF (Druckseite) als Gesamtgas bei p_d
        gvf_refs = [10, 15, 20]
        for gvf in gvf_refs:
            free_ref = total_gas_from_gvf_free_out_cm3N_L(gvf, p_discharge, temperature, gas_medium)
            total_ref = sol_d + free_ref
            ax1.axhline(total_ref, linestyle=":", alpha=0.6)
            ax1.text(13.8, total_ref, f"{gvf}% GVF@p_d → Gesamtgas", va="center", ha="right", fontsize=9)

        # Punkte
        ax1.scatter([p_suction], [sol_s], s=80, label="Löslichkeit @ p_s")
        ax1.scatter([p_discharge], [sol_d], s=80, label="Löslichkeit @ p_d")
        ax1.scatter([p_discharge], [total_cm3N_L], s=90, marker="x", label="Systemgas (konservativ)")

        ax1.set_xlabel("Absolutdruck [bar]")
        ax1.set_ylabel("Gasgehalt [cm³N/L]")
        ax1.set_title(f"Gaslöslichkeit {gas_medium} (Henry → cm³N/L)")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlim(0, 14)

        # Mehrphasen-Kennlinien (als Förderhöhe)
        if best_pump:
            pump = best_pump["pump"]
            Q_lmin_req = m3h_to_lmin(Q_pump)
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

            ax2.scatter(Q_lmin_req, H_req_m, s=90, marker="x", label="Betriebspunkt (Δp)")
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

        with st.expander("Detaillierter Rechenweg (Mehrphase)"):
            st.latex(r"C_{sol}(p,T)=\left(\frac{p}{H(T)}\right)\cdot \left(\frac{R\,T}{p}\,Z\right)\cdot \left(\frac{p}{p_N}\frac{T_N}{T}\frac{1}{Z}\right)\cdot 1000")
            st.markdown("- Hier: Henry-basiert, Ausgabe in **cm³N/L** (Normvolumen pro Liter Flüssigkeit).")
            st.markdown(f"- \(p_s={p_suction:.2f}\,bar\), \(p_d={p_discharge:.2f}\,bar\), \(T={temperature:.1f}°C\)")
            st.markdown(f"- Löslichkeit: \(C_{{sol,s}}={sol_s:.1f}\), \(C_{{sol,d}}={sol_d:.1f}\) cm³N/L")

            if gas_mode == "GVF an Druckseite [%]":
                st.latex(r"V_{gas,op}=\frac{GVF}{100-GVF}\quad;\quad V_N=V_{gas,op}\cdot\frac{p}{p_N}\frac{T_N}{T}\frac{1}{Z}")
                st.markdown(f"- Eingabe: \(GVF_{{out}}={gvf_out_pct:.1f}\%\) (frei @ p_d)")
                st.markdown(f"- Freies Gas @ p_d als Norm: **{free_out_cm3N_L:.1f} cm³N/L**")
                st.latex(r"C_{total}=C_{sol,d}+C_{free,out}")
                st.markdown(f"- **Systemgas (konservativ): {total_cm3N_L:.1f} cm³N/L**")
            else:
                st.latex(r"f=\frac{Q_{rec}}{Q_{pump}}\quad;\quad C_{total}=f\cdot C_{sol,d}")
                st.markdown(f"- Recycling-Anteil \(f\) ≈ **{rec_fraction:.2f}**")
                st.markdown(f"- **Systemgas (konservativ): {total_cm3N_L:.1f} cm³N/L**")

            st.latex(r"C_{diss}(p)=\min(C_{total},C_{sol}(p))\quad;\quad C_{free}(p)=\max(0, C_{total}-C_{sol}(p))")
            st.markdown(f"- @p_s: gelöst **{dissolved_at_s:.1f}**, frei **{free_at_s:.1f}** cm³N/L")
            st.markdown(f"- @p_d: gelöst **{dissolved_at_d:.1f}**, frei **{free_at_d:.1f}** cm³N/L")

            st.latex(r"\text{Gelöst-Anteil}=\frac{C_{diss}}{C_{total}}\cdot 100\%")
            st.markdown(f"- Gelöst-Anteil @p_s: **{frac_diss_s:.1f}%**, @p_d: **{frac_diss_d:.1f}%**")

            st.latex(r"GVF_s=\frac{V_{gas,op}}{1+V_{gas,op}}\cdot 100\%")
            st.markdown(f"- GVF_s (frei, operativ) **{gvf_free_s_pct:.2f}%** → mit Sicherheit **{gvf_free_s_pct_safe:.2f}%**")

    except Exception as e:
        show_error(e, "Mehrphasenpumpen")


def run_atex_selection():
    try:
        st.header("ATEX-Motorauslegung")

        # ATEX-EINGABEN SIND NUR HIER – und links in der Sidebar
        with st.sidebar:
            st.subheader("ATEX-Eingaben")
            P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
            T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)
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

        suitable = [
            m for m in ATEX_MOTORS
            if (zone in m["zone_suitable"])
        ]
        if not suitable:
            st.error("❌ Kein passender Motor-Datensatz für die gewählte Zone vorhanden.")
            return

        # Temperaturabstand (konservativ)
        t_margin = 15.0
        suitable = [m for m in suitable if (m["t_max_surface"] - t_margin) >= T_medium]
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

        with st.expander("Rechenweg / Kriterien"):
            st.latex(r"P_{motor,min}=1.15\cdot P_{welle}")
            st.markdown(f"- \(P_{{motor,min}} = 1.15\\cdot {P_req:.2f} = {P_motor_min:.2f}\,kW\)")
            st.markdown(f"- IEC-Stufe: **{P_iec:.2f} kW**")
            st.latex(r"T_{surface,max}-T_{medium}\ge 15K")
            st.markdown(f"- Temperaturabstand: \(200 - {T_medium:.1f} = {200 - T_medium:.1f}K\) (Anforderung ≥ 15K)")
            st.markdown(f"- Kennzeichnung: `{selected['marking']}` | Zone: {zone}")

        if st.button("ATEX-Dokumentation exportieren"):
            html = f"""
<!DOCTYPE html>
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
<tr><td>Mindestleistung (+15%)</td><td>{P_motor_min:.2f} kW</td></tr>
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
</body></html>
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

        # Sidebar: nur Navigation (und ATEX-Eingaben NUR im ATEX-Tab via run_atex_selection)
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Seite auswählen:",
                ["Einphasenpumpen (Viskosität)", "Mehrphasenpumpen", "ATEX-Auslegung"],
                index=0
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

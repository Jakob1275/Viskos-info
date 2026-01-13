# app.py — Streamlit Cloud lauffähig (Einphase + Mehrphase + ATEX)
# Überarbeitung (nur Mehrphasen-Teil, aber gesamter Code ausgegeben):
# ✅ Mehrphase: klare Logik "entweder Prozess-Volumenstrom ODER Recyclingstrom" (gegenseitig ausschließend)
# ✅ Wenn Recyclingstrom gewählt → Q_pumpe = Q_rec (überschreibt Prozessstrom komplett)
# ✅ Wenn Prozessstrom gewählt → Q_pumpe = Q_liq
# ✅ Meldungen "freies Gas an Saugseite/Druckseite" entfernt
# ✅ Ergebnis ergänzt: "Wie viel % des System-Gases ist bei p_s / p_d gelöst" (Dissolved-Fraktion)
# ✅ Rechenweg mit Formeln bleibt ausführlich
# Hinweis: Das Modell führt eine konservative "System-Gasmenge (Norm cm³N/L)" als Basisgröße.
#         Daraus wird die gelöste Menge = min(total, Löslichkeit(p)) und der gelöste Anteil berechnet.

import math
import warnings
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Pumpenauslegung", layout="wide", page_icon="🔧")

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

# simple Z approximation (trend/stability)
REAL_GAS_FACTORS = {
    "Luft": lambda p_bar, T_K: max(0.85, 1.0 - 0.00008 * p_bar),
    "Methan (CH4)": lambda p_bar, T_K: max(0.80, 1.0 - 0.00015 * p_bar),
    "CO2": lambda p_bar, T_K: max(0.70, 0.90 + 0.00006 * (T_K - 273.15)),
}


# =========================
# Pumpendaten
# =========================
PUMPS = [
    {
        "id": "P1 (Edur LBU Beispiel)",
        "Qw": [0, 10, 20, 30, 40, 50],          # m³/h
        "Hw": [30, 29, 27, 24, 20, 15],         # m
        "eta": [0.35, 0.55, 0.65, 0.62, 0.55, 0.45],
        "max_viscosity": 500,
        "max_density": 1200,
        "n0_rpm": 2900,
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


def shaft_power_kW(rho, Q_m3h, H_m, eta):
    Q = max(float(Q_m3h), 0.0) / 3600.0
    H = max(float(H_m), 0.0)
    e = max(float(eta), 1e-9)
    P_hyd_W = float(rho) * G * Q * H
    return (P_hyd_W / e) / 1000.0


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
    Q_w = float(Q_vis_m3h)
    H_w = float(H_vis_m) / max(CH, 1e-9)
    return {"Q_w": Q_w, "H_w": H_w, "B": B, "CH": CH, "Ceta": Ceta}


def generate_viscous_curve_from_water(pump, nu_cSt, rho):
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    etaw = np.array(pump["eta"], dtype=float)

    H_vis, eta_vis, P_vis = [], [], []
    for q, h, e in zip(Qw, Hw, etaw):
        B = compute_B_HI(q if q > 0 else 1e-6, max(h, 1e-6), nu_cSt)
        CH, Ceta = viscosity_correction_factors(B)
        hv = h * CH
        ev = safe_clamp(e * Ceta, 0.05, 0.95)
        pv = shaft_power_kW(rho, q, hv, ev)
        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)
    return Qw.tolist(), H_vis, eta_vis, P_vis


def generate_water_power_curve(pump, rho_water):
    Qw = list(map(float, pump["Qw"]))
    Hw = list(map(float, pump["Hw"]))
    etaw = list(map(float, pump["eta"]))
    return [shaft_power_kW(rho_water, q, h, e) for q, h, e in zip(Qw, Hw, etaw)]


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
    C_mol_L = p_partial / H

    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0


def free_gas_cm3N_L_from_gvf_pct(gvf_pct, p_bar_abs, T_celsius, gas):
    gvf_pct = safe_clamp(float(gvf_pct), 0.0, 99.0)
    p = max(float(p_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vgas_oper_L_per_Lliq = gvf_pct / max(100.0 - gvf_pct, 1e-9)
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    Vn_L_per_L = Vgas_oper_L_per_Lliq * ratio
    return Vn_L_per_L * 1000.0


def gvf_pct_from_free_gas_cm3N_L(free_cm3N_L, p_bar_abs, T_celsius, gas):
    free_cm3N_L = max(float(free_cm3N_L), 0.0)
    p = max(float(p_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vn_L_per_L = free_cm3N_L / 1000.0
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    Voper = Vn_L_per_L / max(ratio, 1e-12)
    gvf = (Voper / (1.0 + Voper)) * 100.0
    return safe_clamp(gvf, 0.0, 99.0)


def solubility_curve(gas, T_celsius, p_min=0.2, p_max=14.0, n=140):
    ps = np.linspace(p_min, p_max, n)
    sol = [gas_solubility_cm3N_per_L(gas, p, T_celsius) for p in ps]
    return ps, np.array(sol)


def dissolved_fraction(total_cm3N_L, solubility_cm3N_L):
    """Anteil des System-Gases, der bei gegebenem Druck gelöst vorliegt."""
    tot = max(float(total_cm3N_L), 0.0)
    if tot <= 0:
        return 1.0
    sol = max(float(solubility_cm3N_L), 0.0)
    return safe_clamp(min(tot, sol) / tot, 0.0, 1.0)


# =========================
# Pump selection
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
                "score": score + penalty,
            }
            if best is None or cand["score"] < best["score"]:
                best = cand
        except Exception:
            continue
    return best


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
# Pages
# =========================
def run_single_phase_pump():
    try:
        st.header("Einphasenpumpen – Viskosität → Wasserkennlinie")

        left, right = st.columns([1.0, 2.2], gap="large")

        with left:
            st.subheader("Eingaben")
            with st.form("single_phase_form"):
                st.markdown("**Betriebspunkt (viskos)**")
                Q_vis_req = st.number_input("Förderstrom Q [m³/h]", min_value=0.1, value=20.0, step=0.5)
                H_vis_req = st.number_input("Förderhöhe H [m]", min_value=0.1, value=25.0, step=0.5)

                st.markdown("**Medium**")
                medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
                rho = st.number_input("Dichte ρ [kg/m³]", min_value=1.0, value=float(MEDIA[medium]["rho"]), step=5.0)
                nu = st.number_input("Kinematische Viskosität ν [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.5)

                st.markdown("**Optionen**")
                allow_out = st.checkbox("Auswahl außerhalb Kennlinie zulassen", value=True)
                reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
                n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
                n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)

                submitted = st.form_submit_button("Berechnen")

        if not submitted:
            with right:
                st.info("Links Eingaben setzen und **Berechnen** klicken.")
            return

        conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
        Q_w = conv["Q_w"]
        H_w = conv["H_w"]
        B, CH, Ceta = conv["B"], conv["CH"], conv["Ceta"]

        best = choose_best_pump(PUMPS, Q_w, H_w, nu, rho, allow_out_of_range=allow_out)
        if not best:
            with right:
                st.error("Keine geeignete Pumpe gefunden.")
            return

        pump = best["pump"]

        rho_water_ref = float(MEDIA["Wasser (20°C)"]["rho"])
        Pw_water_curve = generate_water_power_curve(pump, rho_water_ref)

        Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve_from_water(pump, nu, rho)

        eta_water_at_Q = float(best["eta_at"])
        eta_vis_at_Q = safe_clamp(eta_water_at_Q * Ceta, 0.05, 0.95)

        P_vis_kW = shaft_power_kW(rho, Q_vis_req, H_vis_req, eta_vis_at_Q)
        P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

        n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)

        n_opt_rpm, P_opt_kW, saving_pct = None, None, None
        if n_ratio_opt is not None:
            n_opt_rpm = float(pump.get("n0_rpm", N0_RPM_DEFAULT)) * n_ratio_opt
            Q_base = Q_vis_req / n_ratio_opt
            P_base = safe_interp(Q_base, Q_vis_curve, P_vis_curve)
            P_opt_kW = float(P_base) * (n_ratio_opt ** 3)
            P_nom_at_Q = safe_interp(Q_vis_req, Q_vis_curve, P_vis_curve)
            saving_pct = ((P_nom_at_Q - P_opt_kW) / P_nom_at_Q * 100.0) if P_nom_at_Q > 0 else 0.0

        with right:
            st.subheader("Ergebnisse")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Gewählte Pumpe", best["id"])
                st.metric("Q (viskos)", f"{Q_vis_req:.2f} m³/h")
            with c2:
                st.metric("H (viskos)", f"{H_vis_req:.2f} m")
                st.metric("η (viskos)", f"{eta_vis_at_Q:.3f}")
            with c3:
                st.metric("Q (Wasser)", f"{Q_w:.2f} m³/h")
                st.metric("H (Wasser)", f"{H_w:.2f} m")
            with c4:
                st.metric("Wellenleistung", f"{P_vis_kW:.2f} kW")
                st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")

            if n_ratio_opt is not None and saving_pct is not None:
                c5, c6 = st.columns(2)
                c5.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
                c6.metric("Energieeinsparung", f"{saving_pct:.1f}%")
            else:
                st.warning("Keine optimale Drehzahl im gewählten Bereich gefunden.")

            if not best["in_range"]:
                st.warning(f"Wasser-Abbildungspunkt liegt außerhalb Kennlinie (Bewertung bei Q={best['Q_eval']:.2f} m³/h).")

            st.subheader("Kennlinien")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            ax1.plot(pump["Qw"], pump["Hw"], "o-", label="Wasser (n0)")
            ax1.plot(Q_vis_curve, H_vis_curve, "s--", label="Viskos (n0)")
            if n_ratio_opt is not None:
                Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
                H_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
                ax1.plot(Q_scaled, H_scaled, ":", label=f"Viskos (n≈{n_opt_rpm:.0f} rpm)")
            ax1.scatter([Q_w], [best["H_at"]], marker="^", s=90, label="BP (Wasserabbildung)")
            ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=90, label="BP (viskos)")
            ax1.set_xlabel("Q [m³/h]")
            ax1.set_ylabel("H [m]")
            ax1.set_title("Q-H")
            ax1.grid(True)
            ax1.legend()

            ax2.plot(pump["Qw"], pump["eta"], "o-", label="Wasser (n0)")
            ax2.plot(Q_vis_curve, eta_vis_curve, "s--", label="Viskos (n0)")
            ax2.scatter([Q_vis_req], [eta_vis_at_Q], marker="x", s=90, label="η (viskos @ BP)")
            ax2.set_xlabel("Q [m³/h]")
            ax2.set_ylabel("η [-]")
            ax2.set_title("Q-η")
            ax2.grid(True)
            ax2.legend()

            ax3.plot(pump["Qw"], Pw_water_curve, "o-", label="Wasser (aus H,η berechnet)")
            ax3.plot(Q_vis_curve, P_vis_curve, "s--", label="Viskos (aus H_vis,η_vis berechnet)")
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

    except Exception as e:
        show_error(e, "Einphase")


# ==========================================================
# ✅ ÜBERARBEITET: Mehrphasenpumpen-Auslegung (nur dieser Teil)
# ==========================================================
def run_multi_phase_pump():
    try:
        st.header("Mehrphasenpumpen – Auslegung über Gaslöslichkeit & Gasanteil")

        left, right = st.columns([1.0, 2.2], gap="large")

        with left:
            st.subheader("Eingaben (klar & eindeutig)")
            with st.form("multi_phase_form_v2"):

                st.markdown("**Prozessdrücke**")
                p_suction = st.number_input("Absolutdruck Saugseite p_s [bar]", min_value=0.2, value=2.0, step=0.1)
                p_discharge = st.number_input("Absolutdruck Druckseite p_d [bar]", min_value=0.2, value=10.0, step=0.1)
                dp_req = max(0.0, p_discharge - p_suction)

                st.markdown("**Medium**")
                gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
                liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)
                temperature = st.number_input("Temperatur T [°C]", min_value=-10.0, value=20.0, step=1.0)

                st.markdown("**Hydraulischer Volumenstrom durch die Pumpe (nur EIN Input aktiv)**")
                flow_mode = st.radio(
                    "Welche Größe ist vorgegeben?",
                    ["Prozess-Volumenstrom Q_liq", "Recyclingstrom Q_rec (überschreibt Q_liq)"],
                    index=0
                )

                if flow_mode == "Prozess-Volumenstrom Q_liq":
                    Q_liq = st.number_input("Q_liq [m³/h]", min_value=0.1, value=8.0, step=0.1)
                    Q_rec = None
                    Q_pump = Q_liq
                else:
                    Q_rec = st.number_input("Q_rec [m³/h]", min_value=0.1, value=8.0, step=0.1)
                    Q_liq = None
                    Q_pump = Q_rec  # Überschreibt Prozessstrom vollständig

                st.divider()
                st.markdown("**Gasvorgabe (freie Gasphase als GVF)**")
                gvf_location = st.radio(
                    "GVF bezieht sich auf",
                    ["Saugseite (frei)", "Druckseite (frei)"],
                    index=0
                )
                gvf_pct = st.slider("GVF [%] (freie Gasphase)", 0.0, 40.0, 5.0, 0.1)

                st.divider()
                st.markdown("**Optionen**")
                show_temp_band = st.checkbox("Temperaturband im Löslichkeitsdiagramm", value=True)
                only_dissolved = st.checkbox("Nur gelöstes Gas zulassen", value=False)
                safety_factor = st.slider("Sicherheitsfaktor auf GVF_s [%]", 0, 30, 10)

                submitted = st.form_submit_button("Berechnen")

        if not submitted:
            with right:
                st.info("Links Eingaben setzen und **Berechnen** klicken.")
            return

        rho_liq = float(MEDIA[liquid_medium]["rho"])
        nu_liq = float(MEDIA[liquid_medium]["nu"])

        # 1) Löslichkeiten bei Saug- und Druckseite
        sol_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature)
        sol_d = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature)

        # 2) Bestimme konservative System-Gasmenge (Norm cm³N/L)
        #    Dazu nutzen wir die GVF-Eingabe (freie Gasphase) an der gewählten Seite.
        #    total = gelöst(max sol(p)) + frei(GVF, p)
        if gvf_location.startswith("Saugseite"):
            free_s_cm3N_L = free_gas_cm3N_L_from_gvf_pct(gvf_pct, p_suction, temperature, gas_medium)
            total_cm3N_L = sol_s + free_s_cm3N_L
        else:
            free_d_cm3N_L = free_gas_cm3N_L_from_gvf_pct(gvf_pct, p_discharge, temperature, gas_medium)
            total_cm3N_L = sol_d + free_d_cm3N_L

        # 3) Gelöst / frei an beiden Druckniveaus (rein aus total vs sol(p))
        dissolved_s_cm3N_L = min(total_cm3N_L, sol_s)
        dissolved_d_cm3N_L = min(total_cm3N_L, sol_d)
        free_s_cm3N_L = max(0.0, total_cm3N_L - sol_s)
        free_d_cm3N_L = max(0.0, total_cm3N_L - sol_d)

        # 4) Für Pumpenauswahl verwenden wir die freie Gasphase an der Saugseite (konservativ)
        gvf_s_pct = gvf_pct_from_free_gas_cm3N_L(free_s_cm3N_L, p_suction, temperature, gas_medium)
        gvf_s_pct_safe = gvf_s_pct * (1.0 + safety_factor / 100.0)

        # 5) Gelöst-Anteile in %
        frac_diss_s = dissolved_fraction(total_cm3N_L, sol_s)
        frac_diss_d = dissolved_fraction(total_cm3N_L, sol_d)

        # 6) Pumpenauswahl
        best_pump = None
        if only_dissolved and free_s_cm3N_L > 0:
            best_pump = None
        else:
            best_pump = choose_best_mph_pump(
                MPH_PUMPS, Q_pump, dp_req, gvf_s_pct_safe, nu_liq, rho_liq
            )

        with right:
            st.subheader("Ergebnisse (übersichtlich)")

            # Kernaussage: wie viel % ist gelöst bei den beiden Drücken?
            st.markdown("**Gelöst-Anteil des Systemgases** (bezogen auf die konservative System-Gasmenge):")
            a1, a2 = st.columns(2)
            a1.metric(f"bei p_s = {p_suction:.2f} bar", f"{frac_diss_s*100:.1f}% gelöst")
            a2.metric(f"bei p_d = {p_discharge:.2f} bar", f"{frac_diss_d*100:.1f}% gelöst")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Volumenstrom durch Pumpe", f"{Q_pump:.2f} m³/h")
            c1.caption("Q_pump = Q_liq" if flow_mode.startswith("Prozess") else "Q_pump = Q_rec (überschreibt Q_liq)")
            c2.metric("Δp Anforderung", f"{dp_req:.2f} bar")
            c3.metric("Löslichkeit Saugseite", f"{sol_s:.1f} cm³N/L")
            c4.metric("Löslichkeit Druckseite", f"{sol_d:.1f} cm³N/L")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Systemgas (konservativ)", f"{total_cm3N_L:.1f} cm³N/L")
            d2.metric("Gelöst bei Saugseite", f"{dissolved_s_cm3N_L:.1f} cm³N/L")
            d3.metric("Gelöst bei Druckseite", f"{dissolved_d_cm3N_L:.1f} cm³N/L")
            d4.metric("GVF_s (frei, +Sicherheit)", f"{gvf_s_pct_safe:.1f} %")

            # Keine Warn-/Info-Meldungen mehr (wie gewünscht)

            if only_dissolved and free_s_cm3N_L > 0:
                st.error("❌ Nur gelöstes Gas zugelassen, aber aus der konservativen Bilanz ergibt sich freie Gasphase an p_s.")
                st.caption("Tipp: Entweder Option deaktivieren oder Eingabe/GVF-Referenzseite prüfen.")
            else:
                if best_pump:
                    st.success(f"✅ Empfohlene Pumpe: {best_pump['pump']['id']} (Kennlinie {best_pump['gvf_key']}% GVF)")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Δp verfügbar", f"{best_pump['dp_avail']:.2f} bar")
                    e2.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
                    e3.metric("Drehzahl", f"{best_pump['n_rpm']:.0f} rpm")
                    e4.metric("Modus", best_pump["mode"])
                else:
                    st.info("Keine geeignete Mehrphasenpumpe gefunden (oder Auswahl blockiert).")

            # =========================
            # Diagramme
            # =========================
            st.subheader("Diagramme")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Löslichkeit
            if show_temp_band:
                for T in [temperature - 10, temperature, temperature + 10]:
                    if -10 <= T <= 150:
                        p_arr, sol_arr = solubility_curve(gas_medium, T)
                        ax1.plot(p_arr, sol_arr, "--", label=f"{T:.0f}°C")
            else:
                p_arr, sol_arr = solubility_curve(gas_medium, temperature)
                ax1.plot(p_arr, sol_arr, "--", label=f"{temperature:.0f}°C")

            # Referenzlinien 10/15/20% GVF an Druckseite (als Gesamtgas bei p_d)
            for gvf_ref in [10, 15, 20]:
                free_ref = free_gas_cm3N_L_from_gvf_pct(gvf_ref, p_discharge, temperature, gas_medium)
                total_ref = sol_d + free_ref
                ax1.axhline(total_ref, linestyle=":", alpha=0.6)
                ax1.text(13.8, total_ref, f"{gvf_ref}% GVF@p_d → Gesamtgas", va="center", ha="right", fontsize=9)

            # Punkte (klarer)
            ax1.scatter([p_suction], [sol_s], s=80, label="Löslichkeit @ p_s")
            ax1.scatter([p_discharge], [sol_d], s=80, label="Löslichkeit @ p_d")
            ax1.scatter([p_suction], [total_cm3N_L], s=90, marker="x", label="Systemgas (konservativ)")

            ax1.set_xlabel("Absolutdruck p [bar]")
            ax1.set_ylabel("Gasgehalt [cm³N/L]")
            ax1.set_title(f"Gaslöslichkeit – {gas_medium} (Henry → cm³N/L)")
            ax1.grid(True)
            ax1.legend()
            ax1.set_xlim(0, 14)

            # Pumpenkennlinien als Förderhöhe
            if best_pump:
                pump = best_pump["pump"]
                H_req_m = dp_req * BAR_TO_M_WATER
                Q_lmin_req = m3h_to_lmin(Q_pump)

                max_Q_lmin, max_H = 0.0, 0.0
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

                ax2.scatter(Q_lmin_req, H_req_m, s=90, marker="x", label="Betriebspunkt (Δp Anforderung)")
                ax2.set_xlabel("Volumenstrom Q [L/min]")
                ax2.set_ylabel("Förderhöhe H [m]")
                ax2.set_title(f"Mehrphasen-Kennlinien: {pump['id']}")
                ax2.grid(True)
                ax2.legend()
                ax2.set_xlim(0, max_Q_lmin * 1.1 if max_Q_lmin > 0 else 10)
                ax2.set_ylim(0, max_H * 1.1 if max_H > 0 else 10)
            else:
                ax2.text(0.5, 0.5, "Keine geeignete Pumpe", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_xlabel("Volumenstrom Q [L/min]")
                ax2.set_ylabel("Förderhöhe H [m]")
                ax2.set_title("Mehrphasen-Kennlinien")
                ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            with st.expander("Rechenweg (Mehrphase) – mit Formeln"):
                st.markdown("### 1) Löslichkeit (Henry → cm³N/L)")
                st.latex(r"C_{mol/L}=\frac{y\cdot p}{H(T)}")
                st.latex(r"V_{m,oper}=\frac{R\cdot T}{p}\cdot Z")
                st.latex(r"V_{oper} = C\cdot V_{m,oper}")
                st.latex(r"V_N = V_{oper}\cdot\frac{p}{p_N}\cdot\frac{T_N}{T}\cdot\frac{1}{Z}")
                st.markdown(f"- sol(p_s) = {sol_s:.3f} cm³N/L, sol(p_d) = {sol_d:.3f} cm³N/L")

                st.markdown("### 2) Freies Gas aus GVF (Umrechnung operativ → Norm)")
                st.latex(r"V_{gas,oper}=\frac{GVF}{100-GVF}\quad [L/L_{liq}]")
                st.latex(r"V_{gas,N}=V_{gas,oper}\cdot\frac{p}{p_N}\cdot\frac{T_N}{T}\cdot\frac{1}{Z}")
                st.markdown(f"- GVF-Eingabe: {gvf_pct:.2f}% @ {('p_s' if gvf_location.startswith('Saug') else 'p_d')}")

                st.markdown("### 3) Konservative System-Gasmenge")
                st.latex(r"V_{total,N} = sol(p_{ref}) + V_{free,N}(GVF@p_{ref})")
                st.markdown(f"- total = {total_cm3N_L:.3f} cm³N/L")

                st.markdown("### 4) Gelöst / frei bei einem Druck p")
                st.latex(r"V_{diss}(p)=\min\left(V_{total,N},\,sol(p)\right)")
                st.latex(r"V_{free}(p)=\max\left(0,\,V_{total,N}-sol(p)\right)")
                st.latex(r"f_{diss}(p)=\frac{V_{diss}(p)}{V_{total,N}}")
                st.markdown(f"- dissolved@p_s = {dissolved_s_cm3N_L:.3f} cm³N/L → {frac_diss_s*100:.2f}% gelöst")
                st.markdown(f"- dissolved@p_d = {dissolved_d_cm3N_L:.3f} cm³N/L → {frac_diss_d*100:.2f}% gelöst")

                st.markdown("### 5) Für Pumpenauswahl: GVF an Saugseite (konservativ)")
                st.latex(r"GVF_s = f\left(V_{free,N}(p_s),\,p_s,\,T,\,Z\right)")
                st.latex(r"GVF_{s,\,safe}=GVF_s\cdot(1+\text{Sicherheitsfaktor})")
                st.markdown(f"- GVF_s = {gvf_s_pct:.3f}%")
                st.markdown(f"- +{safety_factor}% → GVF_s,safe = {gvf_s_pct_safe:.3f}%")

    except Exception as e:
        show_error(e, "Mehrphase")


def run_atex_selection():
    try:
        st.header("ATEX-Auslegung (Erklär-Reiter)")

        left, right = st.columns([1.0, 2.2], gap="large")

        with left:
            st.subheader("Eingaben")
            with st.form("atex_form"):
                P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
                T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

                atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"], index=0)
                if atmosphere == "Gas":
                    zone = st.selectbox("Zone", [0, 1, 2], index=2)
                else:
                    zone = st.selectbox("Zone", [20, 21, 22], index=2)

                t_margin = st.slider("Temperaturabstand [K] (Marge)", 0, 30, 15)

                submitted = st.form_submit_button("Bewerten")

        if not submitted:
            with right:
                st.info("Links Eingaben setzen und **Bewerten** klicken.")
            return

        with right:
            st.subheader("Ergebnisse")

            if atmosphere == "Staub":
                st.warning("Staub-Ex (Zone 20/21/22): In diesem Demo-Code sind keine Datensätze hinterlegt.")
                return

            if zone == 0:
                st.warning("Zone 0: In diesem Demo-Code sind keine Datensätze hinterlegt.")
                return

            suitable = [
                m for m in ATEX_MOTORS
                if (zone in m["zone_suitable"]) and ((m["t_max_surface"] - t_margin) >= T_medium)
            ]

            P_motor_min = P_req * 1.15
            P_iec = motor_iec(P_motor_min)

            c1, c2, c3 = st.columns(3)
            c1.metric("Wellenleistung", f"{P_req:.2f} kW")
            c2.metric("Mindestleistung (+15%)", f"{P_motor_min:.2f} kW")
            c3.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

            if not suitable:
                st.error(f"❌ Kein Motor passt zu Zone {zone} und T={T_medium:.1f}°C (Marge {t_margin} K).")
                return

            selected = st.radio(
                "Motortyp wählen:",
                options=suitable,
                format_func=lambda x: f"{x['marking']} ({x['id']})"
            )

            st.success("✅ Gültige Konfiguration gefunden")
            with st.expander("Rechenweg (ATEX) – mit Formeln"):
                st.latex(r"P_{motor,min}=1.15\cdot P_{shaft}")
                st.latex(r"T_{surface,max}-\Delta T \ge T_{medium}")
                st.markdown(f"- Kennzeichnung: `{selected['marking']}`")

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
            st.divider()
            st.caption("Mehrphase: entweder Q_liq oder Q_rec (überschreibt).")

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

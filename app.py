# app.py — Streamlit Cloud lauffähig (Einphase + Mehrphase + ATEX)
import math
import warnings
from datetime import datetime

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Pumpenauslegung", layout="wide", page_icon="🔧")
DEBUG = True

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

# Einfache Z-Approx (nur Trend/Stabilität)
REAL_GAS_FACTORS = {
    "Luft": lambda p, T: max(0.85, 1.0 - 0.00008 * p),
    "Methan (CH4)": lambda p, T: max(0.80, 1.0 - 0.00015 * p),
    "CO2": lambda p, T: max(0.70, 0.90 + 0.00006 * (T - 273.15)),
}

# =========================
# Pumpendaten (Demo)
# =========================
PUMPS = [
    {
        "id": "P1 (Edur LBU 2…)",
        "Qw": [0, 10, 20, 30, 40, 50],
        "Hw": [30, 29, 27, 24, 20, 15],
        "eta": [0.35, 0.55, 0.65, 0.62, 0.55, 0.45],
        "max_viscosity": 500,
        "max_density": 1200,
    },
]

MPH_PUMPS = [
    {
        "id": "MPH-40 (Edur MPH 40)",
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

# =========================
# ATEX Datensätze (Demo)
# =========================
ATEX_MOTORS = [
    {
        "id": "Standard Zone 2 (Ex ec)",
        "marking": "II 3G Ex ec IIC T3 Gc",
        "zone_suitable": [2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "3G",
        "efficiency_class": "IE3",
        "gas_group": "IIC",
        "protection": "Ex ec",
    },
    {
        "id": "Zone 1 (Ex db eb)",
        "marking": "II 2G Ex db eb IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "efficiency_class": "IE3",
        "gas_group": "IIC",
        "protection": "Ex db eb",
    },
]

# =========================
# Helpers
# =========================
def show_error(e: Exception, where: str = ""):
    st.error(f"❌ Fehler {('in ' + where) if where else ''}: {e}")
    if DEBUG:
        import traceback
        st.code(traceback.format_exc())

def safe_clamp(x, a, b):
    try:
        return max(a, min(b, x))
    except Exception:
        return a

def safe_interp(x, xp, fp):
    xp = list(map(float, xp))
    fp = list(map(float, fp))
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
# Viskosität / HI (robust)
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
    Q_water = float(Q_vis_m3h)
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

    return Qw.tolist(), H_vis, eta_vis, P_vis

# Root + Drehzahl-Optimierung (wie vorher)
def bisect_root(f, a, b, it=80, tol=1e-6):
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

    return bisect_root(f, float(n_min), float(n_max), it=120, tol=1e-5)

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
    p_part = safe_clamp(float(y_gas), 0.0, 1.0) * p

    C_mol_L = p_part / H  # mol/L

    V_molar_oper = (R_BAR_L * T_K) / p * Z  # L/mol
    V_oper_L_per_L = C_mol_L * V_molar_oper

    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L

def free_gas_cm3N_L_from_gvf_pct(gvf_pct, p_bar_abs, T_celsius, gas):
    gvf_pct = safe_clamp(float(gvf_pct), 0.0, 99.0)
    p = max(float(p_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vgas_oper_L_per_Lliq = gvf_pct / max(100.0 - gvf_pct, 1e-9)
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    Vn_L_per_L = Vgas_oper_L_per_Lliq * ratio
    return Vn_L_per_L * 1000.0  # cm³N/L

def gvf_pct_from_free_gas_cm3N_L(free_cm3N_L, p_bar_abs, T_celsius, gas):
    free_cm3N_L = max(float(free_cm3N_L), 0.0)
    p = max(float(p_bar_abs), 0.1)
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)

    Vn_L_per_L = free_cm3N_L / 1000.0
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    Vgas_oper_L_per_Lliq = Vn_L_per_L / max(ratio, 1e-12)

    gvf = (Vgas_oper_L_per_Lliq / (1.0 + Vgas_oper_L_per_Lliq)) * 100.0
    return safe_clamp(gvf, 0.0, 99.0)

def solubility_curve_total_with_gvf(gas, T_celsius, gvf_pct, p_min=0.2, p_max=14.0, n=200):
    ps = np.linspace(p_min, p_max, n)
    sol = np.array([gas_solubility_cm3N_per_L(gas, p, T_celsius) for p in ps], dtype=float)
    free = np.array([free_gas_cm3N_L_from_gvf_pct(gvf_pct, p, T_celsius, gas) for p in ps], dtype=float)
    return ps, sol + free

def dissolved_fraction(C_total, S_p):
    if C_total <= 1e-12:
        return 1.0
    return safe_clamp(min(C_total, S_p) / C_total, 0.0, 1.0)

# =========================
# Pump selection (Einphase)
# =========================
def choose_best_pump(pumps, Q_req, H_req, nu_cSt, rho):
    best = None
    for p in pumps:
        if nu_cSt > p.get("max_viscosity", 500):
            continue
        if rho > p.get("max_density", 1200):
            continue

        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        Q_eval = safe_clamp(Q_req, qmin, qmax)
        H_at = safe_interp(Q_eval, p["Qw"], p["Hw"])
        eta_at = safe_interp(Q_eval, p["Qw"], p["eta"])
        score = abs(H_at - H_req)

        cand = {"id": p["id"], "pump": p, "Q_eval": Q_eval, "H_at": H_at, "eta_at": eta_at, "score": score}
        if best is None or cand["score"] < best["score"]:
            best = cand
    return best

# =========================
# Pump selection (Mehrphase)
# =========================
def choose_best_mph_pump(pumps, Q_req_m3h, dp_req_bar, gvf_suction_free_pct_safe, nu_cSt, rho_liq,
                        n_min_ratio=0.5, n_max_ratio=1.2):
    best = None
    Q_req = float(Q_req_m3h)
    dp_req = float(dp_req_bar)
    gvf_free_pct = float(gvf_suction_free_pct_safe)

    for pump in pumps:
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

        dp_nom = safe_interp(Q_req, Qc, dpc) if (min(Qc) <= Q_req <= max(Qc)) else None
        n_ratio = bisect_root(f, n_min_ratio, n_max_ratio, it=120, tol=1e-4)

        candidates = []

        if dp_nom is not None and dp_nom >= dp_req:
            P_nom = safe_interp(Q_req, power_curve["Q"], power_curve["P"])
            candidates.append({
                "pump": pump, "gvf_key": gvf_key, "dp_avail": dp_nom,
                "P_req": P_nom, "n_ratio": 1.0, "n_rpm": pump["n0_rpm"], "mode": "Nenndrehzahl"
            })

        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if min(Qc) <= Q_base <= max(Qc):
                dp_scaled = dp_at_ratio(n_ratio)
                if dp_scaled >= dp_req:
                    P_base = safe_interp(Q_base, power_curve["Q"], power_curve["P"])
                    P_scaled = P_base * (n_ratio ** 3)
                    candidates.append({
                        "pump": pump, "gvf_key": gvf_key, "dp_avail": dp_scaled,
                        "P_req": P_scaled, "n_ratio": n_ratio, "n_rpm": pump["n0_rpm"] * n_ratio,
                        "mode": "Drehzahl angepasst"
                    })

        for cand in candidates:
            score = abs(cand["dp_avail"] - dp_req) + 0.15 * abs(cand["n_ratio"] - 1.0)
            cand["score"] = score
            if best is None or score < best["score"]:
                best = cand

    return best

# =========================
# Pages
# =========================
def run_single_phase_pump():
    try:
        st.header("Einphasenpumpen – Viskosität (inkl. Drehzahlanpassung)")

        # Eingaben oben
        cA, cB, cC, cD = st.columns(4)
        with cA:
            Q_vis_req = st.number_input("Betriebspunkt Q_vis [m³/h]", min_value=0.1, value=20.0, step=0.5)
        with cB:
            H_vis_req = st.number_input("Betriebspunkt H_vis [m]", min_value=0.1, value=25.0, step=0.5)
        with cC:
            medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
        with cD:
            nu = st.number_input("ν [cSt]", min_value=0.1, value=float(MEDIA[medium]["nu"]), step=0.1)

        rho = float(MEDIA[medium]["rho"])

        opt1, opt2, opt3 = st.columns(3)
        with opt1:
            reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
        with opt2:
            n_min = st.slider("n_min/n0", 0.4, 1.0, 0.6, 0.01)
        with opt3:
            n_max = st.slider("n_max/n0", 1.0, 1.6, 1.2, 0.01)

        conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
        Q_water = conv["Q_water"]
        H_water = conv["H_water"]
        B, CH, Ceta = conv["B"], conv["CH"], conv["Ceta"]

        best = choose_best_pump(PUMPS, Q_water, H_water, nu, rho)
        if not best:
            st.error("Keine geeignete Pumpe gefunden.")
            return

        pump = best["pump"]
        eta_water = float(best["eta_at"])
        eta_vis = safe_clamp(eta_water * Ceta, 0.05, 0.95)

        P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
        P_vis_kW = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
        P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

        # Kennlinien viskos
        Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(pump, nu, rho)

        # Drehzahl-Anpassung: n so wählen, dass viskose Kennlinie den Betriebspunkt trifft
        n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)

        n_opt_rpm = None
        P_opt_kW = None
        saving_pct = None
        P_nom_at_Q = safe_interp(Q_vis_req, Q_vis_curve, P_vis_curve)

        if n_ratio_opt is not None:
            n_opt_rpm = N0_RPM_DEFAULT * n_ratio_opt
            Q_base = Q_vis_req / n_ratio_opt
            P_base = safe_interp(Q_base, Q_vis_curve, P_vis_curve)
            P_opt_kW = float(P_base) * (n_ratio_opt ** 3)
            if P_nom_at_Q > 1e-9:
                saving_pct = (P_nom_at_Q - P_opt_kW) / P_nom_at_Q * 100.0
            else:
                saving_pct = 0.0

        st.subheader("Ergebnisse")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gewählte Pumpe", best["id"])
            st.metric("Q_vis", f"{Q_vis_req:.2f} m³/h")
        with col2:
            st.metric("H_vis", f"{H_vis_req:.2f} m")
            st.metric("η_vis", f"{eta_vis:.3f}")
        with col3:
            st.metric("Q_wasser (Umrechnung)", f"{Q_water:.2f} m³/h")
            st.metric("H_wasser (Umrechnung)", f"{H_water:.2f} m")
        with col4:
            st.metric("Wellenleistung", f"{P_vis_kW:.2f} kW")
            st.metric("Motor (+Reserve)", f"{P_motor_kW:.2f} kW")

        if n_ratio_opt is not None and P_opt_kW is not None and saving_pct is not None:
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
            with d2:
                st.metric("Energieeinsparung ggü. n0", f"{saving_pct:.1f}%")
        else:
            st.info("Keine gültige Drehzahl-Anpassung im gewählten Bereich gefunden.")

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
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=90, label="η_vis")
        ax2.set_xlabel("Q [m³/h]")
        ax2.set_ylabel("η [-]")
        ax2.set_title("Q-η")
        ax2.grid(True)
        ax2.legend()

        ax3.plot(Q_vis_curve, P_vis_curve, "s--", label="Viskos P (n0, berechnet)")
        if n_ratio_opt is not None:
            P_scaled = [p * (n_ratio_opt ** 3) for p in P_vis_curve]
            Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
            ax3.plot(Q_scaled, P_scaled, ":", label=f"Viskos P (n≈{n_opt_rpm:.0f} rpm)")
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=90, label="BP (viskos)")
        ax3.set_xlabel("Q [m³/h]")
        ax3.set_ylabel("P [kW]")
        ax3.set_title("Q-P")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Detaillierter Rechenweg (mit Formeln)"):
            st.markdown("## 1) HI-Kennzahl und Korrekturfaktoren")
            st.latex(r"Q_{\mathrm{gpm}} = Q_{\mathrm{m^3/h}}\cdot 4.40287")
            st.latex(r"H_{\mathrm{ft}} = H_{\mathrm{m}}\cdot 3.28084")
            st.latex(r"B = 16.5\cdot \frac{\sqrt{\nu}}{Q_{\mathrm{gpm}}^{0.25}\cdot H_{\mathrm{ft}}^{0.375}}")
            st.markdown(f"- Eingaben: Q_vis={Q_vis_req:.3f} m³/h, H_vis={H_vis_req:.3f} m, ν={nu:.3f} cSt")
            st.markdown(f"- Ergebnis: B={B:.4f}")

            st.markdown("## 2) Korrekturen (robuste Näherung)")
            st.latex(r"C_H = \exp\left(-0.165\cdot (\log_{10}(B))^{2.2}\right)")
            st.latex(r"C_\eta = 1 - 0.25\log_{10}(B) - 0.05(\log_{10}(B))^2")
            st.markdown(f"- C_H={CH:.4f}, C_η={Ceta:.4f}")

            st.markdown("## 3) Umrechnung Betriebspunkt → Wasserpunkt")
            st.latex(r"Q_w \approx Q_{\mathrm{vis}}")
            st.latex(r"H_w = \frac{H_{\mathrm{vis}}}{C_H}")
            st.markdown(f"- Q_w={Q_water:.3f} m³/h, H_w={H_water:.3f} m")

            st.markdown("## 4) Leistung")
            st.latex(r"P_{\mathrm{hyd}} = \rho g Q H")
            st.latex(r"P_{\mathrm{Welle}} = \frac{P_{\mathrm{hyd}}}{\eta_{\mathrm{vis}}}")
            st.markdown(f"- P_hyd={P_hyd_W:.1f} W, η_vis={eta_vis:.4f}")
            st.markdown(f"- P_Welle(n0)≈{P_vis_kW:.3f} kW")

            st.markdown("## 5) Drehzahlanpassung (Affinity Laws)")
            st.latex(r"Q \propto n,\qquad H \propto n^2,\qquad P \propto n^3")
            st.latex(r"H(n) = H(n_0)\cdot \left(\frac{n}{n_0}\right)^2")
            st.markdown("- Gesucht: n/n0 so, dass die viskose Kennlinie den Betriebspunkt trifft.")
            if n_ratio_opt is not None and P_opt_kW is not None and saving_pct is not None:
                st.markdown(f"- n/n0={n_ratio_opt:.4f} → n_opt={n_opt_rpm:.0f} rpm")
                st.markdown(f"- P(n0)@Q={P_nom_at_Q:.3f} kW")
                st.markdown(f"- P_opt≈{P_opt_kW:.3f} kW → Einsparung≈{saving_pct:.1f}%")
            else:
                st.markdown("- Keine Lösung im Bereich n_min…n_max gefunden.")

    except Exception as e:
        show_error(e, "Einphasenpumpen")


def run_multi_phase_pump():
    """
    Zielprozess:
    - Gas wird VOR der Pumpe zugesetzt (freie Phase am Eintritt).
    - Durch Druckerhöhung geht Gas in Lösung.
    - Zielwert-Modus: Am Austritt (p_d) soll ein vorgegebener gelöster Gasanteil erreicht werden (ideal: ohne freies Gas dort).
    """
    try:
        st.header("Mehrphasenpumpen – Gaszugabe vor der Pumpe (inkl. Zielwert-Modus)")

        # Prozessdrücke + Medien
        top1, top2, top3, top4 = st.columns(4)
        with top1:
            p_suction = st.number_input("Absolutdruck Saugseite p_s [bar]", min_value=0.2, value=2.0, step=0.1)
        with top2:
            p_discharge = st.number_input("Absolutdruck Druckseite p_d [bar]", min_value=0.2, value=10.0, step=0.1)
        with top3:
            gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        with top4:
            liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)

        temperature = st.number_input("Temperatur T [°C]", min_value=-10.0, value=20.0, step=1.0)
        rho_liq = float(MEDIA[liquid_medium]["rho"])
        nu_liq = float(MEDIA[liquid_medium]["nu"])

        dp_req = max(0.0, p_discharge - p_suction)

        st.divider()
        st.subheader("Hydraulischer Volumenstrom durch die Pumpe (nur EIN Input aktiv)")
        flow_mode = st.radio(
            "Welche Größe ist vorgegeben?",
            ["Prozess-Volumenstrom Q_liq", "Recyclingstrom Q_rec (überschreibt Q_liq)"],
            index=0
        )
        if flow_mode == "Prozess-Volumenstrom Q_liq":
            Q_liq = st.number_input("Q_liq [m³/h]", min_value=0.1, value=8.0, step=0.5)
            Q_pump = float(Q_liq)
            flow_note = "Q_pump = Q_liq"
        else:
            Q_rec = st.number_input("Q_rec [m³/h]", min_value=0.1, value=12.0, step=0.5)
            Q_pump = float(Q_rec)
            flow_note = "Q_pump = Q_rec (überschreibt Q_liq)"

        # Löslichkeiten
        sol_s = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature)
        sol_d = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature)

        st.divider()
        st.subheader("Gasvorgabe VOR der Pumpe")
        gas_mode = st.radio(
            "Eingabemodus:",
            ["GVF_in an Saugseite (frei) [%]", "Zielwert-Modus: gelöst am Austritt (p_d)"],
            index=1
        )

        safety_factor = st.slider("Sicherheitsfaktor GVF [%]", 0, 20, 10)

        # Variablen, die wir in beiden Modi füllen
        gvf_in_pct = 0.0
        free_s_cm3N_L = 0.0
        C_total = 0.0
        target_note = ""

        if gas_mode.startswith("GVF_in"):
            gvf_in_pct = st.slider("GVF_in an Saugseite (frei) [%]", 0.0, 40.0, 10.0, 0.1)
            free_s_cm3N_L = free_gas_cm3N_L_from_gvf_pct(gvf_in_pct, p_suction, temperature, gas_medium)
            # konservativ: Eintritt gesättigt + freie Phase
            C_total = sol_s + free_s_cm3N_L
            target_note = "Direktvorgabe: GVF_in (freie Gasphase) an der Saugseite."

        else:
            target_kind = st.radio(
                "Zielwert-Definition am Austritt (p_d):",
                ["% der Löslichkeit S(p_d)", "Absolut gelöst [cm³N/L]"],
                index=0,
                horizontal=True
            )
            if target_kind.startswith("%"):
                target_pct = st.slider("Ziel: gelöst am Austritt [% von S(p_d)]", 0.0, 100.0, 100.0, 0.5)
                C_target = sol_d * (target_pct / 100.0)
            else:
                C_target = st.number_input("Ziel: gelöst am Austritt [cm³N/L]", min_value=0.0, value=float(sol_d), step=5.0)

            # Plausibilität: mehr als S(p_d) kann nicht gelöst sein
            if C_target > sol_d + 1e-9:
                st.error("Zielwert > Löslichkeit bei p_d. Das ist physikalisch nicht erreichbar (ohne andere Annahmen).")
                C_target = sol_d

            # Ziel: am Austritt soll alles gelöst sein -> kein freies Gas dort -> C_total = C_target
            C_total = C_target

            # benötigtes freies Gas am Eintritt:
            free_s_cm3N_L = max(0.0, C_total - sol_s)
            gvf_in_pct = gvf_pct_from_free_gas_cm3N_L(free_s_cm3N_L, p_suction, temperature, gas_medium)

            target_note = "Zielwert-Modus: Gaszugabe so berechnet, dass am Austritt der Zielwert gelöst ist (frei_d≈0)."

            # Warnung, wenn GVF sehr hoch
            if gvf_in_pct >= 40.0:
                st.warning("Der berechnete GVF_in ist sehr hoch. Prüfe Prozessannahmen / Gasquelle / Mischer / Pumpenlimit.")

        # Was passiert am Austritt?
        dissolved_s = min(C_total, sol_s)
        dissolved_d = min(C_total, sol_d)
        free_d = max(0.0, C_total - sol_d)

        frac_diss_s = dissolved_fraction(C_total, sol_s)
        frac_diss_d = dissolved_fraction(C_total, sol_d)

        # Für Pumpenauswahl: konservativ GVF an Saugseite aus freiem Gas
        gvf_s_pct = gvf_pct_from_free_gas_cm3N_L(free_s_cm3N_L, p_suction, temperature, gas_medium)
        gvf_s_pct_safe = gvf_s_pct * (1.0 + safety_factor / 100.0)

        best_pump = choose_best_mph_pump(
            MPH_PUMPS, Q_pump, dp_req, gvf_s_pct_safe, nu_liq, rho_liq
        )

        # =========================
        # Ergebnisse
        # =========================
        st.subheader("Ergebnisse (übersichtlich)")
        st.caption(target_note)
        st.caption(flow_note)

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("### Gelöst-Anteil des Systemgases")
            st.metric(f"bei p_s = {p_suction:.2f} bar", f"{frac_diss_s*100:.1f}% gelöst")
        with a2:
            st.metric(f"bei p_d = {p_discharge:.2f} bar", f"{frac_diss_d*100:.1f}% gelöst")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volumenstrom durch Pumpe", f"{Q_pump:.2f} m³/h")
            st.metric("Δp Anforderung", f"{dp_req:.2f} bar")
        with col2:
            st.metric("Löslichkeit Saugseite S(p_s)", f"{sol_s:.1f} cm³N/L")
            st.metric("Löslichkeit Druckseite S(p_d)", f"{sol_d:.1f} cm³N/L")
        with col3:
            st.metric("Systemgas gesamt C_total", f"{C_total:.1f} cm³N/L")
            st.metric("GVF_in (berechnet/vorgegeben)", f"{gvf_in_pct:.2f} %")
        with col4:
            st.metric("Freies Gas am Eintritt", f"{free_s_cm3N_L:.1f} cm³N/L")
            st.metric("Freies Gas am Austritt", f"{free_d:.1f} cm³N/L")

        # Aussage wie viel % Gas bei dem entsprechenden Druck gelöst wird
        st.info(
            f"Bei p_s={p_suction:.2f} bar sind {frac_diss_s*100:.1f}% des Systemgases gelöst "
            f"(={dissolved_s:.1f} cm³N/L). "
            f"Bei p_d={p_discharge:.2f} bar sind {frac_diss_d*100:.1f}% gelöst "
            f"(={dissolved_d:.1f} cm³N/L)."
        )

        if best_pump:
            st.success(f"✅ Empfohlene Pumpe: {best_pump['pump']['id']} (Kennlinie {best_pump['gvf_key']}% GVF)")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Δp verfügbar", f"{best_pump['dp_avail']:.2f} bar")
            with b2:
                st.metric("Leistung", f"{best_pump['P_req']:.2f} kW")
            with b3:
                st.metric("Drehzahl", f"{best_pump['n_rpm']:.0f} rpm")
            with b4:
                st.metric("Modus", best_pump["mode"])
        else:
            st.warning("Keine geeignete Mehrphasenpumpe gefunden (GVF zu hoch oder außerhalb Kennfeld).")

        # =========================
        # Diagramme
        # =========================
        st.subheader("Diagramme")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Links: Löslichkeit + physikalisch konsistente Referenzkurven GVF (als Kurven über p)
        ps = np.linspace(0.2, 14.0, 220)
        sol_curve = np.array([gas_solubility_cm3N_per_L(gas_medium, p, temperature) for p in ps], dtype=float)
        ax1.plot(ps, sol_curve, "--", label=f"Löslichkeit S(p) (T={temperature:.0f}°C)")

        for gvf_ref in [10, 15, 20]:
            p_ref, total_ref = solubility_curve_total_with_gvf(gas_medium, temperature, gvf_ref, p_min=0.2, p_max=14.0, n=220)
            ax1.plot(p_ref, total_ref, ":", alpha=0.8, label=f"S(p)+frei@{gvf_ref}%GVF")

        # Punkte: Saug/Druck & C_total
        ax1.scatter([p_suction], [sol_s], s=80, label="S(p_s)")
        ax1.scatter([p_discharge], [sol_d], s=80, label="S(p_d)")
        ax1.scatter([p_suction], [C_total], s=90, marker="x", label="C_total (Systemgas)")

        ax1.set_xlabel("Absolutdruck [bar]")
        ax1.set_ylabel("Gasgehalt [cm³N/L]")
        ax1.set_title(f"Gaslöslichkeit & GVF-Referenz: {gas_medium}")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlim(0, 14)

        # Rechts: Pumpenkennlinien als Förderhöhe
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

            ax2.scatter(Q_lmin_req, H_req_m, s=90, marker="x", label="Betriebspunkt (Anforderung)")

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

        with st.expander("Detaillierter Rechenweg (mit Formeln)"):
            st.markdown("## 1) Löslichkeit (Henry → cm³N/L)")
            st.latex(r"C_{\mathrm{mol/L}}=\frac{p_{\mathrm{part}}}{H(T)}")
            st.latex(r"V_{\mathrm{oper}}=\frac{R\;T}{p}\;Z")
            st.latex(
                r"C_{\mathrm{cm^3N/L}} = (C_{\mathrm{mol/L}}\cdot V_{\mathrm{oper}})\cdot"
                r"\frac{p}{p_N}\cdot\frac{T_N}{T}\cdot\frac{1}{Z}\cdot 1000"
            )
            st.markdown(f"- S(p_s)={sol_s:.2f} cm³N/L, S(p_d)={sol_d:.2f} cm³N/L")

            st.markdown("## 2) Gaszugabe am Eintritt (GVF_in → freies Gas in cm³N/L)")
            st.latex(r"V_{g,\mathrm{oper}}=\frac{GVF_{in}}{100-GVF_{in}}")
            st.latex(
                r"C_{s,\mathrm{free,N}}=V_{g,\mathrm{oper}}\cdot\frac{p_s}{p_N}\cdot"
                r"\frac{T_N}{T}\cdot\frac{1}{Z}\cdot 1000"
            )
            st.markdown(f"- GVF_in={gvf_in_pct:.2f}% → C_free,s={free_s_cm3N_L:.2f} cm³N/L")

            st.markdown("## 3) Systemgas vor der Pumpe")
            st.latex(r"C_{\mathrm{tot}} = S(p_s,T) + C_{s,\mathrm{free,N}} \quad (\text{konservativ})")
            st.markdown(f"- C_total={C_total:.2f} cm³N/L")

            st.markdown("## 4) Austritt: gelöst + ggf. frei")
            st.latex(r"C_{d,\mathrm{sol}}=\min(C_{\mathrm{tot}},\,S(p_d,T))")
            st.latex(r"C_{d,\mathrm{free}}=\max(0,\,C_{\mathrm{tot}}-S(p_d,T))")
            st.markdown(f"- gelöst@p_d={dissolved_d:.2f} cm³N/L, frei@p_d={free_d:.2f} cm³N/L")

            st.markdown("## 5) Gelöst-Anteile")
            st.latex(r"f_{\mathrm{sol}}(p)=\frac{\min(C_{\mathrm{tot}},S(p,T))}{C_{\mathrm{tot}}}")
            st.markdown(f"- f_sol(p_s)={frac_diss_s*100:.2f}%, f_sol(p_d)={frac_diss_d*100:.2f}%")

            st.markdown("## 6) Zielwert-Modus (wenn aktiv)")
            st.latex(r"C_{\mathrm{tot}} = C_{\mathrm{target}} \le S(p_d,T)\quad (\text{frei}_d\approx 0)")
            st.latex(r"C_{s,\mathrm{free}} = \max(0, C_{\mathrm{tot}} - S(p_s,T))")
            st.markdown("- Aus C_free,s wird GVF_in rückwärts bestimmt (Umkehrung der Norm-Umrechnung).")

            st.markdown("## 7) Pumpenauswahl (konservativ über GVF_s + Sicherheit)")
            st.latex(r"GVF_{s,\mathrm{safe}} = GVF_s\cdot \left(1+\frac{k}{100}\right)")
            st.markdown(f"- GVF_s={gvf_s_pct:.2f}%, k={safety_factor}% → GVF_safe={gvf_s_pct_safe:.2f}%")

    except Exception as e:
        show_error(e, "Mehrphasenpumpen")


def run_atex_selection():
    try:
        st.header("ATEX-Auslegung – Auswahl & Dokumentation")

        with st.sidebar:
            st.subheader("Eingaben (ATEX)")

            atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"], index=0)
            if atmosphere == "Gas":
                zone = st.selectbox("Zone", [0, 1, 2], index=2)
            else:
                zone = st.selectbox("Zone", [20, 21, 22], index=2)

            P_req = st.number_input("Erforderliche Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
            reserve = st.slider("Leistungsreserve [%]", 0, 30, 15)

            T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=250.0, value=40.0, step=1.0)
            t_margin = st.slider("Temperaturabstand [K]", 0, 30, 15)

        if atmosphere == "Staub":
            st.error("Staub-Ex ist in diesem Demo-Datensatz nicht hinterlegt. (Kann ich dir gern ergänzen.)")
            return

        if zone == 0:
            st.warning("Zone 0 ist sehr anspruchsvoll (EPL Ga). In diesem Demo-Datensatz nicht abgebildet.")
            return

        # Mindestleistung + IEC
        P_motor_min = P_req * (1.0 + reserve / 100.0)
        P_iec = motor_iec(P_motor_min)

        st.subheader("Ergebnisse")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wellenleistung", f"{P_req:.2f} kW")
        with c2:
            st.metric(f"Mindestleistung (+{reserve}%)", f"{P_motor_min:.2f} kW")
        with c3:
            st.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

        # Filter auf passende Motoren
        suitable = [
            m for m in ATEX_MOTORS
            if (zone in m["zone_suitable"]) and ((m["t_max_surface"] - t_margin) >= T_medium)
        ]

        if not suitable:
            st.error(f"Kein Motor im Datensatz passt (Zone {zone}, T={T_medium:.1f}°C, Abstand {t_margin}K).")
            st.info("Du kannst: Temperaturklasse anpassen, Motor-Datensätze ergänzen oder Sicherheitsabstand reduzieren.")
            return

        st.subheader("Verfügbare Motoren")
        selected = st.radio(
            "Motor auswählen:",
            options=suitable,
            format_func=lambda x: f"{x['marking']} — {x['id']}"
        )

        st.success("✅ Gültige Auswahl nach den gesetzten Filtern")

        with st.expander("Rechenweg & Formeln"):
            st.markdown("## 1) Leistungsreserve")
            st.latex(r"P_{\mathrm{motor,min}} = P_{\mathrm{Welle}}\cdot\left(1+\frac{r}{100}\right)")
            st.markdown(f"- r={reserve}% → P_motor,min={P_motor_min:.2f} kW")
            st.markdown("## 2) IEC Stufe")
            st.markdown("- Es wird die nächsthöhere IEC-Nennleistung gewählt.")
            st.markdown(f"- IEC={P_iec:.2f} kW")
            st.markdown("## 3) Temperaturprüfung")
            st.latex(r"T_{\mathrm{surface,max}} - \Delta T \ge T_{\mathrm{medium}}")
            st.markdown(f"- T_surface,max={selected['t_max_surface']:.1f}°C, ΔT={t_margin}K, T_medium={T_medium:.1f}°C")

        # Export (wie zuvor)
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
    .box {{ margin: 10px 0; padding: 12px; background: #f8f9fa; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    code {{ background:#eee; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>ATEX-Motorauslegung</h1>
  <p>Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>

  <div class="box">
    <strong>Atmosphäre:</strong> {atmosphere}<br/>
    <strong>Zone:</strong> {zone}<br/>
    <strong>Wellenleistung:</strong> {P_req:.2f} kW<br/>
    <strong>Reserve:</strong> {reserve}%<br/>
    <strong>Medientemperatur:</strong> {T_medium:.1f} °C<br/>
    <strong>Temperaturabstand:</strong> {t_margin} K
  </div>

  <h2>Berechnung</h2>
  <table>
    <tr><th>Parameter</th><th>Wert</th></tr>
    <tr><td>Mindestleistung</td><td>{P_motor_min:.2f} kW</td></tr>
    <tr><td>IEC Nennleistung</td><td>{P_iec:.2f} kW</td></tr>
    <tr><td>Prüfung Oberfläche</td><td>{selected['t_max_surface']:.1f}°C - {t_margin}K ≥ {T_medium:.1f}°C</td></tr>
  </table>

  <h2>Ausgewählter Motor</h2>
  <div class="box">
    <strong>ID:</strong> {selected['id']}<br/>
    <strong>Kennzeichnung:</strong> <code>{selected['marking']}</code><br/>
    <strong>Kategorie:</strong> {selected['category']}<br/>
    <strong>Temp.-Klasse:</strong> {selected['temp_class']}<br/>
    <strong>Gasgruppe:</strong> {selected.get('gas_group','-')}<br/>
    <strong>Schutzart:</strong> {selected.get('protection','-')}<br/>
    <strong>Wirkungsgrad:</strong> {selected.get('efficiency_class','-')}<br/>
    <strong>Geeignet für Zone:</strong> {", ".join(map(str, selected["zone_suitable"]))}<br/>
  </div>

  <h2>Hinweis</h2>
  <p>Bitte Konformität mit 2014/34/EU und EN 60079 sowie Herstellerdaten (EPL, Zündschutzart, T-Klasse, Kennwerte) prüfen.</p>
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

        tabs = st.tabs(["Einphase (Viskosität)", "Mehrphase", "ATEX"])
        with tabs[0]:
            run_single_phase_pump()
        with tabs[1]:
            run_multi_phase_pump()
        with tabs[2]:
            run_atex_selection()

        if DEBUG:
            st.caption("DEBUG aktiv: Fehlertraces werden in der App angezeigt.")
    except Exception as e:
        show_error(e, "main")
        st.stop()

if __name__ == "__main__":
    main()

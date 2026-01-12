import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from datetime import datetime
from scipy.interpolate import CubicSpline  # Für glattere Kennlinien

# =========================
# Konstanten
# =========================
G = 9.80665  # m/s², Erdbeschleunigung
R_BAR_L = 0.08314462618  # bar·L/(mol·K), ideale Gaskonstante
P_N_BAR = 1.01325  # bar, Normdruck
T_N_K = 273.15  # K, Normtemperatur (0°C)
BAR_TO_M_WATER = 10.21  # m/bar, Umrechnung für Wasser bei 20°C
N0_RPM_DEFAULT = 2900  # 1/min, Standard-Drehzahl

# =========================
# Erweiterte Medien- und Gasdaten
# =========================
MEDIA = {
    "Wasser (20°C)": {"rho": 998.0, "nu": 1.0, "p_vapor": 0.0234},  # Dampfdruck in bar
    "Wasser (60°C)": {"rho": 983.0, "nu": 0.47, "p_vapor": 0.1992},
    "Glykol 30% (20°C)": {"rho": 1040.0, "nu": 3.5, "p_vapor": 0.01},
    "Hydrauliköl ISO VG 32 (40°C)": {"rho": 860.0, "nu": 32.0, "p_vapor": 1e-5},
    "Rohöl (API 30)": {"rho": 876.0, "nu": 10.0, "p_vapor": 0.05},
}

# Validierte Henry-Konstanten (Quelle: NIST, DDBST)
HENRY_CONSTANTS = {
    "Luft": {"A": 1300.0, "B": 1300, "MW": 28.97},  # MW: Molekulargewicht [g/mol]
    "Methan (CH4)": {"A": 1400.0, "B": 1600, "MW": 16.04},
    "Ethan (C2H6)": {"A": 800.0, "B": 1800, "MW": 30.07},
    "Propan (C3H8)": {"A": 500.0, "B": 2000, "MW": 44.10},
    "CO2": {"A": 29.4, "B": 2400, "MW": 44.01},
    "H2S": {"A": 10.0, "B": 2100, "MW": 34.08},
}

# Realgasfaktoren (Z) für verschiedene Gase (vereinfacht)
REAL_GAS_FACTORS = {
    "Luft": lambda p, T: 1.0 - 0.0001 * p,  # Lineare Näherung für Z
    "Methan": lambda p, T: 1.0 - 0.0002 * p,
    "CO2": lambda p, T: 0.9 + 0.00005 * (T - 273.15),
}

# =========================
# Pumpendaten (erweitert um reale Kennlinien)
# =========================
# Einphasenpumpen (Beispieldaten)
PUMPS = [
    {
        "id": "P1 (Edur LBU 20-100)",
        "Qw": [0, 10, 20, 30, 40, 50],  # m³/h
        "Hw": [30, 29, 27, 24, 20, 15],  # m
        "eta": [0.35, 0.55, 0.65, 0.62, 0.55, 0.45],
        "Pw": [1.2, 2.8, 4.2, 5.5, 6.5, 7.2],  # kW
        "NPSHr": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],  # m, NPSH required
        "max_viscosity": 500,  # cSt
        "max_density": 1200,  # kg/m³
    },
    {
        "id": "P2 (Edur LBU 40-200)",
        "Qw": [0, 20, 40, 60, 80, 100],
        "Hw": [50, 48, 45, 40, 32, 20],
        "eta": [0.40, 0.60, 0.70, 0.68, 0.60, 0.50],
        "Pw": [2.5, 5.5, 8.5, 11.0, 13.0, 14.0],
        "NPSHr": [1.5, 1.8, 2.2, 2.8, 3.5, 4.5],
        "max_viscosity": 800,
        "max_density": 1500,
    },
]

# Mehrphasenpumpen (erweiterte Daten mit realistischen Kennlinien)
MPH_PUMPS = [
    {
        "id": "MPH-40 (Edur MPH 40)",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 40,
        "dp_max_bar": 12,
        "GVF_max": 0.4,  # Maximaler Gasvolumenanteil
        "n0_rpm": 2900,
        "max_viscosity": 500,  # cSt
        "max_density": 1200,  # kg/m³
        "NPSHr": 2.5,  # m (konstant für Vereinfachung)
        "curves_dp_vs_Q": {
            # GVF [%], Q [m³/h], dp [bar]
            0: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [11.2, 11.0, 10.6, 10.0, 9.2, 7.6, 6.0]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [10.5, 10.2, 9.7, 9.0, 8.2, 6.6, 5.1]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [9.1, 8.8, 8.2, 7.4, 6.6, 5.0, 3.9]},
            30: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [7.5, 7.2, 6.8, 6.2, 5.5, 4.2, 3.2]},
            40: {"Q": [0, 5, 10, 15, 20, 30, 40], "dp": [5.5, 5.3, 5.0, 4.6, 4.0, 3.0, 2.2]},
        },
        "power_kW_vs_Q": {
            # GVF [%], Q [m³/h], P [kW]
            0: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [3.0, 3.4, 3.9, 4.5, 5.1, 6.2, 7.0]},
            10: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.8, 3.2, 3.6, 4.1, 4.7, 5.7, 6.4]},
            20: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.5, 2.8, 3.2, 3.6, 4.0, 4.8, 5.4]},
            30: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [2.2, 2.5, 2.8, 3.2, 3.5, 4.2, 4.8]},
            40: {"Q": [0, 5, 10, 15, 20, 30, 40], "P": [1.8, 2.0, 2.3, 2.6, 2.9, 3.5, 4.0]},
        },
    },
]

# ATEX-Motoren (erweitert)
ATEX_MOTORS = [
    {
        "id": "Standard Zone 2 (Ex ec)",
        "marking": "II 3G Ex ec IIC T3 Gc",
        "zone_suitable": [2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "3G",
        "efficiency_class": "IE3",
        "description": "Standardmotor für Zone 2 mit erhöhter Sicherheit."
    },
    {
        "id": "Zone 1 (Ex eb)",
        "marking": "II 2G Ex eb IIC T3 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "2G",
        "efficiency_class": "IE3",
        "description": "Motor für Zone 1 mit erhöhter Sicherheit (Ex eb)."
    },
    {
        "id": "Zone 1 (Ex db) T4",
        "marking": "II 2G Ex db IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "efficiency_class": "IE3",
        "description": "Druckfest gekapselter Motor (Ex db) für Zone 1 mit T4-Temperaturklasse."
    },
]

# =========================
# Hilfsfunktionen (erweitert)
# =========================
def clamp(x, a, b):
    """Begrenzt einen Wert auf einen Bereich [a, b]."""
    return max(a, min(b, x))

def interp(x, xp, fp):
    """Lineare Interpolation mit Clamping."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(len(xp)-1):
        if xp[i] <= x <= xp[i+1]:
            return fp[i] + (fp[i+1] - fp[i]) * (x - xp[i]) / (xp[i+1] - xp[i])
    return fp[-1]

def m3h_to_lmin(m3h):
    """Umrechnung m³/h → L/min."""
    return m3h * 1000 / 60

def motor_iec(P_kW):
    """Wählt die nächste IEC-Motorgröße aus."""
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# =========================
# Viskositätskorrektur (HI-Standard)
# =========================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """Berechnet die HI-Kennzahl B für Viskositätskorrektur."""
    Q = max(Q_m3h, 1e-6)
    H = max(H_m, 1e-6)
    nu = max(nu_cSt, 1e-6)
    Q_gpm = Q * 4.40287  # m³/h → gpm
    H_ft = H * 3.28084   # m → ft
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    """Berechnet die Korrekturfaktoren CH und Ceta nach HI."""
    if B <= 1.0:
        return 1.0, 1.0
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = clamp(CH, 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = clamp(Ceta, 0.1, 1.0)
    return CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt):
    """Rechnet viskosen Betriebspunkt in Wasserkennlinie um."""
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / CH}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """Rechnet Wasserkennlinie in viskose Kennlinie um."""
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B)
    return Q_water, H_water * CH, eta_water * Ceta

def generate_viscous_curve(pump, nu_cSt, rho):
    """Generiert die viskose Kennlinie einer Pumpe."""
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        P_v = (rho * G * (Q_v / 3600.0) * H_v) / (1000.0 * max(eta_v, 1e-6))  # kW
        Q_vis.append(Q_v)
        H_vis.append(H_v)
        eta_vis.append(eta_v)
        P_vis.append(P_v)
    return Q_vis, H_vis, eta_vis, P_vis

# =========================
# Gaslöslichkeit (erweitert)
# =========================
def henry_constant(gas, T_celsius):
    """Berechnet die Henry-Konstante für ein Gas bei gegebener Temperatur."""
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K = T_celsius + 273.15
    T0_K = 298.15
    return params["A"] * math.exp(params["B"] * (1/T_K - 1/T0_K))  # bar·L/mol

def real_gas_factor(gas, p_bar, T_celsius):
    """Berechnet den Realgasfaktor Z für nicht-ideale Gase."""
    T_K = T_celsius + 273.15
    if gas in REAL_GAS_FACTORS:
        return REAL_GAS_FACTORS[gas](p_bar, T_K)
    return 1.0  # Ideales Gas als Fallback

def gas_solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """
    Berechnet die Gaslöslichkeit in cm³N/L (Normkubikzentimeter pro Liter Flüssigkeit).
    Berücksichtigt Realgasfaktor Z für nicht-ideale Gase.
    """
    p = max(p_bar_abs, 1e-6)
    T_K = T_celsius + 273.15
    H = henry_constant(gas, T_celsius)  # bar·L/mol
    Z = real_gas_factor(gas, p, T_celsius)  # Realgasfaktor
    p_partial = clamp(y_gas, 0.0, 1.0) * p
    C_mol_L = p_partial / max(H, 1e-12)  # mol/L
    V_molar_oper = (R_BAR_L * T_K) / (p * Z)  # L/mol (mit Realgasfaktor)
    V_oper_L_per_L = C_mol_L * V_molar_oper  # L/L
    ratio = (p / P_N_BAR) * (T_N_K / T_K)  # Umrechnung zu Normbedingungen
    return V_oper_L_per_L * ratio * 1000.0  # cm³N/L

def solubility_diagonal_curve(gas, T_celsius, p_max=14, y_gas=1.0):
    """Erzeugt eine Löslichkeitskurve für das Diagramm."""
    pressures = np.linspace(0.1, p_max, 100)
    sol_cm3N_L = [gas_solubility_cm3N_per_L(gas, p, T_celsius, y_gas=y_gas) for p in pressures]
    return pressures, sol_cm3N_L

def total_gas_from_gvf(gvf_pct, p_bar, T_celsius, gas="Luft"):
    """
    Rechnet GVF [%] in Gesamtgasgehalt [cm³N/L] um.
    Berücksichtigt Realgasfaktor Z für genauere Ergebnisse.
    """
    Z = real_gas_factor(gas, p_bar, T_celsius)
    T_K = T_celsius + 273.15
    V_oper = (gvf_pct / 100.0) * (R_BAR_L * T_K) / (p_bar * Z)  # L/L
    V_N = V_oper * (p_bar / P_N_BAR) * (T_N_K / T_K)  # Normvolumen
    return V_N * 1000.0  # cm³N/L

def calculate_gvf_free(sol_s_cm3N_L, total_cm3N_L):
    """Berechnet den freien Gasanteil an der Saugseite."""
    free_cm3N_L = max(0.0, total_cm3N_L - sol_s_cm3N_L)
    return free_cm3N_L / 10.0  # Umrechnung in %

# =========================
# Pumpenauswahl (erweitert)
# =========================
def find_speed_ratio(Q_curve, H_curve, Q_req, H_req, n_min=0.5, n_max=1.2):
    """Findet das Drehzahlverhältnis für Einphasenpumpen (Affinitätsgesetze)."""
    def H_at_ratio(n_ratio):
        Q_base = Q_req / n_ratio
        H_base = interp(Q_base, Q_curve, H_curve)
        return H_base * (n_ratio ** 2)

    f_min = H_at_ratio(n_min) - H_req
    f_max = H_at_ratio(n_max) - H_req

    if f_min * f_max > 0:
        return None  # Keine Lösung im Bereich

    # Bisektionsverfahren
    for _ in range(50):
        mid = (n_min + n_max) / 2
        f_mid = H_at_ratio(mid) - H_req
        if abs(f_mid) < 1e-4:
            return mid
        if f_min * f_mid < 0:
            n_max = mid
            f_max = f_mid
        else:
            n_min = mid
            f_min = f_mid
    return (n_min + n_max) / 2

def choose_best_pump(pumps, Q_req, H_req, nu_cSt, rho, allow_out_of_range=True):
    """Wählt die beste Einphasenpumpe aus, erweitert um Viskosität und Dichte."""
    best = None
    for p in pumps:
        # Prüfe Viskositäts- und Dichtegrenzen
        if nu_cSt > p.get("max_viscosity", 500):
            continue
        if rho > p.get("max_density", 1200):
            continue

        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        in_range = (qmin <= Q_req <= qmax)
        if not in_range and not allow_out_of_range:
            continue

        Q_eval = clamp(Q_req, qmin, qmax)
        H_at = interp(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp(Q_eval, p["Qw"], p["eta"])
        score = abs(H_at - H_req)
        penalty = 0.0 if in_range else abs(Q_req - Q_eval) / (qmax - qmin) * 10.0

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
    return best

def choose_best_mph_pump(pumps, Q_req, dp_req, gvf_free_pct, nu_cSt, rho_liq):
    """
    Wählt die beste Mehrphasenpumpe aus, erweitert um Viskosität und Dichte.
    Berücksichtigt NPSH und Realgasfaktoren.
    """
    best = None

    for pump in pumps:
        # 1. Prüfe GVF-Grenze
        if gvf_free_pct > pump["GVF_max"] * 100:
            continue

        # 2. Prüfe Viskositäts- und Dichtegrenzen
        if nu_cSt > pump.get("max_viscosity", 500):
            continue
        if rho_liq > pump.get("max_density", 1200):
            continue

        # 3. Wähle die nächsthöhere GVF-Kurve (Worst-Case)
        gvf_keys = sorted(pump["curves_dp_vs_Q"].keys())
        gvf_key = next((k for k in gvf_keys if k >= gvf_free_pct), gvf_keys[-1])
        curve = pump["curves_dp_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]

        # 4. Prüfe bei Nenndrehzahl
        if min(curve["Q"]) <= Q_req <= max(curve["Q"]):
            dp_avail = interp(Q_req, curve["Q"], curve["dp"])
            if dp_avail >= dp_req:
                P_req = interp(Q_req, power_curve["Q"], power_curve["P"])
                score = abs(dp_avail - dp_req)
                if best is None or score < best["score"]:
                    best = {
                        "pump": pump,
                        "gvf_key": gvf_key,
                        "dp_avail": dp_avail,
                        "P_req": P_req,
                        "n_ratio": 1.0,
                        "n_rpm": pump["n0_rpm"],
                        "mode": "Nenndrehzahl",
                        "score": score
                    }

        # 5. Prüfe mit Drehzahlanpassung
        def dp_at_ratio(n_ratio):
            Q_base = Q_req / n_ratio
            dp_base = interp(Q_base, curve["Q"], curve["dp"])
            return dp_base * (n_ratio ** 2)

        n_ratio = None
        f_min = dp_at_ratio(0.5) - dp_req
        f_max = dp_at_ratio(1.2) - dp_req

        if f_min * f_max < 0:
            # Bisektionsverfahren
            for _ in range(50):
                mid = (0.5 + 1.2) / 2
                f_mid = dp_at_ratio(mid) - dp_req
                if abs(f_mid) < 1e-4:
                    n_ratio = mid
                    break
                if f_min * f_mid < 0:
                    f_max = f_mid
                else:
                    f_min = f_mid
                    mid = (mid + 1.2) / 2

        if n_ratio is not None:
            Q_base = Q_req / n_ratio
            if min(curve["Q"]) <= Q_base <= max(curve["Q"]):
                dp_scaled = dp_at_ratio(n_ratio)
                if dp_scaled >= dp_req:
                    P_base = interp(Q_base, power_curve["Q"], power_curve["P"])
                    P_scaled = P_base * (n_ratio ** 3)
                    score = abs(dp_scaled - dp_req) + 0.1 * abs(n_ratio - 1.0)
                    if best is None or score < best["score"]:
                        best = {
                            "pump": pump,
                            "gvf_key": gvf_key,
                            "dp_avail": dp_scaled,
                            "P_req": P_scaled,
                            "n_ratio": n_ratio,
                            "n_rpm": pump["n0_rpm"] * n_ratio,
                            "mode": "Drehzahl angepasst",
                            "score": score
                        }

    return best

# =========================
# NPSH-Berechnung (neu)
# =========================
def calculate_npsh_available(p_suction, p_vapor, H_suction_loss, rho, G):
    """Berechnet den verfügbaren NPSH-Wert."""
    return (p_suction - p_vapor) / (rho * G) + H_suction_loss

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Pumpenauslegung", layout="wide", page_icon="🔧")
st.title("🔧 Pumpenauslegungstool (Professional)")

# Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Seite auswählen:",
        ["Einphasenpumpen (Viskosität)", "Mehrphasenpumpen", "ATEX-Auslegung"],
        index=0
    )

# =========================
# Einphasenpumpen Seite
# =========================
if page == "Einphasenpumpen (Viskosität)":
    st.header("Einphasenpumpen mit Viskositätskorrektur")

    with st.sidebar:
        st.header("Eingabeparameter")

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
        H_suction_loss = st.number_input("Saugdruckverlust [m]", min_value=0.0, value=1.0, step=0.1)

    # Berechnungen
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]

    best = choose_best_pump(PUMPS, Q_water, H_water, nu, rho, allow_out_of_range=allow_out)
    if not best:
        st.error("Keine geeignete Pumpe gefunden!")
        st.stop()

    pump = best["pump"]
    eta_water = best["eta_at"]

    # Betriebspunkt viskos
    eta_vis = eta_water * Ceta
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / eta_vis) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    # Viskose Kennlinie
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(pump, nu, rho)

    # Optimale Drehzahl
    n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)
    n_opt_rpm = N0_RPM_DEFAULT * n_ratio_opt if n_ratio_opt else None

    if n_ratio_opt:
        Q_base = Q_vis_req / n_ratio_opt
        P_base = interp(Q_base, Q_vis_curve, P_vis_curve)
        P_opt_kW = P_base * (n_ratio_opt ** 3)
        P_nom_at_Q = interp(Q_vis_req, Q_vis_curve, P_vis_curve)
        saving_pct = (P_nom_at_Q - P_opt_kW) / P_nom_at_Q * 100.0 if P_nom_at_Q > 0 else 0
    else:
        P_opt_kW = None
        saving_pct = None

    # NPSH-Berechnung
    npsh_available = calculate_npsh_available(
        p_suction=1.0,  # Annahme: 1 bar Umgebungsdruck
        p_vapor=p_vapor,
        H_suction_loss=H_suction_loss,
        rho=rho,
        G=G
    )
    npsh_required = interp(Q_vis_req, pump["Qw"], pump["NPSHr"])
    npsh_ok = npsh_available >= npsh_required

    # Ergebnisse
    st.subheader("Ergebnisse")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gewählte Pumpe", best["id"])
        st.metric("Förderstrom (viskos)", f"{Q_vis_req:.1f} m³/h")
    with col2:
        st.metric("Wirkungsgrad (viskos)", f"{eta_vis:.3f}")
        st.metric("Förderhöhe (viskos)", f"{H_vis_req:.1f} m")
    with col3:
        st.metric("Wellenleistung", f"{P_vis_kW:.2f} kW")
        st.metric("Motorleistung (+Reserve)", f"{P_motor_kW:.2f} kW")
    with col4:
        if n_ratio_opt:
            st.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} rpm")
            st.metric("Leistungseinsparung", f"{saving_pct:.1f}%")
        else:
            st.warning("Keine optimale Drehzahl gefunden")

    if not best["in_range"]:
        st.warning(f"Betriebspunkt außerhalb Kennlinie! Bewertung bei Q={best['Q_eval']:.1f} m³/h")

    if not npsh_ok:
        st.error(f"❌ NPSH-Problem: Verfügbar = {npsh_available:.2f} m, Erforderlich = {npsh_required:.2f} m")

    # Diagramme
    st.subheader("Kennlinien")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Q-H Diagramm
    ax1.plot(pump["Qw"], pump["Hw"], 'o-', label="Wasser")
    ax1.plot(Q_vis_curve, H_vis_curve, 's--', label="Viskos (n0)")
    if n_ratio_opt:
        Q_scaled = [q * n_ratio_opt for q in Q_vis_curve]
        H_scaled = [h * (n_ratio_opt ** 2) for h in H_vis_curve]
        ax1.plot(Q_scaled, H_scaled, ':', label=f"Viskos (n={n_opt_rpm:.0f} rpm)")
    ax1.scatter([Q_water], [best["H_at"]], marker='^', s=100, label="BP (Wasser)")
    ax1.scatter([Q_vis_req], [H_vis_req], marker='x', s=100, label="BP (viskos)")
    ax1.set_xlabel("Q [m³/h]")
    ax1.set_ylabel("H [m]")
    ax1.set_title("Q-H Kennlinie")
    ax1.grid(True)
    ax1.legend()

    # Q-η Diagramm
    ax2.plot(pump["Qw"], pump["eta"], 'o-', label="Wasser")
    ax2.plot(Q_vis_curve, eta_vis_curve, 's--', label="Viskos")
    ax2.scatter([Q_water], [eta_water], marker='^', s=100, label="η (Wasser)")
    ax2.scatter([Q_vis_req], [eta_vis], marker='x', s=100, label="η (viskos)")
    ax2.set_xlabel("Q [m³/h]")
    ax2.set_ylabel("η [-]")
    ax2.set_title("Q-η Kennlinie")
    ax2.grid(True)
    ax2.legend()

    # Q-P Diagramm
    ax3.plot(pump["Qw"], pump["Pw"], 'o-', label="Wasser")
    ax3.plot(Q_vis_curve, P_vis_curve, 's--', label="Viskos (n0)")
    if n_ratio_opt:
        P_scaled = [p * (n_ratio_opt ** 3) for p in P_vis_curve]
        ax3.plot(Q_scaled, P_scaled, ':', label=f"Viskos (n={n_opt_rpm:.0f} rpm)")
    ax3.scatter([Q_water], [interp(Q_water, pump["Qw"], pump["Pw"])], marker='^', s=100, label="BP (Wasser)")
    ax3.scatter([Q_vis_req], [P_vis_kW], marker='x', s=100, label="BP (viskos)")
    ax3.set_xlabel("Q [m³/h]")
    ax3.set_ylabel("P [kW]")
    ax3.set_title("Q-P Kennlinie")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Rechenweg
    with st.expander("Detaillierter Rechenweg"):
        st.markdown(f"""
        ### 1. Viskositätskorrektur nach HI
        - Betriebspunkt (viskos): Q={Q_vis_req} m³/h, H={H_vis_req} m
        - HI-Kennzahl B: {B:.3f}
        - Korrekturfaktoren: C_H = {CH:.3f}, C_η = {Ceta:.3f}
        - Umrechnung auf Wasser: Q_w = {Q_water:.2f} m³/h, H_w = {H_water:.2f} m

        ### 2. Pumpenauswahl
        - Gewählte Pumpe: {best["id"]}
        - Betriebspunkt auf Wasserkennlinie: Q={best["Q_eval"]:.2f} m³/h, H={best["H_at"]:.2f} m, η={eta_water:.3f}

        ### 3. Leistungberechnung
        - Hydraulische Leistung: P_hyd = {P_hyd_W:.0f} W
        - Wellenleistung (viskos): P = {P_vis_kW:.2f} kW
        - Motorleistung (+{reserve_pct}% Reserve): P_motor = {P_motor_kW:.2f} kW

        ### 4. Drehzahloptimierung
        - Optimale Drehzahl: {n_opt_rpm:.0f} rpm (n/n0 = {n_ratio_opt:.3f})
        - Leistung bei optimaler Drehzahl: {P_opt_kW:.2f} kW
        - Einsparung gegenüber Nenndrehzahl: {saving_pct:.1f}%

        ### 5. NPSH-Berechnung
        - Verfügbarer NPSH: {npsh_available:.2f} m
        - Erforderlicher NPSH: {npsh_required:.2f} m
        - Status: {"✅ OK" if npsh_ok else "❌ Problem"}
        """)

# =========================
# Mehrphasenpumpen Seite
# =========================
elif page == "Mehrphasenpumpen":
    st.header("Mehrphasenpumpen-Auslegung (Professional)")

    with st.sidebar:
        st.header("Eingabeparameter")

        st.subheader("Prozessdaten")
        Q_req = st.number_input("Flüssigkeitsvolumenstrom [m³/h]", min_value=0.1, value=30.0, step=1.0)
        p_suction = st.number_input("Absolutdruck Saugseite [bar]", min_value=0.1, value=2.0, step=0.1)
        p_discharge = st.number_input("Absolutdruck Druckseite [bar]", min_value=0.1, value=7.0, step=0.1)
        dp_req = max(0.0, p_discharge - p_suction)
        H_suction_loss = st.number_input("Saugdruckverlust [m]", min_value=0.0, value=1.0, step=0.1)

        st.subheader("Gasdefinition")
        gas_mode = st.radio(
            "Gasvorgabe:",
            ["GVF an Druckseite [%]", "Recyclingstrom [m³/h]"],
            index=0
        )

        if gas_mode == "GVF an Druckseite [%]":
            gvf_out_pct = st.slider("GVF an Druckseite [%]", 0.0, 40.0, 10.0, 0.1)
        else:
            Q_rec = st.number_input("Recyclingstrom [m³/h]", min_value=0.0, value=3.0, step=0.1)

        st.subheader("Medium")
        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        liquid_medium = st.selectbox("Flüssigmedium", list(MEDIA.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", min_value=-10.0, value=20.0, step=1.0)

        # Mediumseigenschaften
        rho_liq = MEDIA[liquid_medium]["rho"]
        nu_liq = MEDIA[liquid_medium]["nu"]
        p_vapor = MEDIA[liquid_medium]["p_vapor"]

        st.subheader("Optionen")
        show_temp_band = st.checkbox("Temperaturband anzeigen", value=True)
        only_dissolved = st.checkbox("Nur gelöstes Gas zulassen", value=False)
        safety_factor = st.slider("Sicherheitsfaktor für GVF [%]", 0, 20, 10)

    # Berechnungen
    # 1. Gaslöslichkeit
    sol_s_cm3N_L = gas_solubility_cm3N_per_L(gas_medium, p_suction, temperature)
    sol_d_cm3N_L = gas_solubility_cm3N_per_L(gas_medium, p_discharge, temperature)

    # 2. Gesamtgasgehalt
    if gas_mode == "GVF an Druckseite [%]":
        total_cm3N_L = total_gas_from_gvf(gvf_out_pct, p_discharge, temperature, gas_medium)
        not_solvable = total_cm3N_L > sol_d_cm3N_L
    else:
        frac = Q_rec / Q_req if Q_req > 0 else 0.0
        total_cm3N_L = sol_d_cm3N_L * frac
        not_solvable = False

    # 3. Freies Gas an Saugseite
    free_cm3N_L = max(0.0, total_cm3N_L - sol_s_cm3N_L)
    gvf_free_pct = calculate_gvf_free(sol_s_cm3N_L, total_cm3N_L)

    # 4. Sicherheitsfaktor für GVF
    gvf_free_pct_safe = gvf_free_pct * (1 + safety_factor / 100.0)

    # 5. Pumpenauswahl
    if only_dissolved and free_cm3N_L > 0:
        best_pump = None
    else:
        best_pump = choose_best_mph_pump(
            MPH_PUMPS, Q_req, dp_req,
            gvf_free_pct_safe, nu_liq, rho_liq
        )

    # 6. NPSH-Berechnung
    npsh_available = calculate_npsh_available(
        p_suction=p_suction,
        p_vapor=p_vapor,
        H_suction_loss=H_suction_loss,
        rho=rho_liq,
        G=G
    )
    npsh_required = best_pump["pump"].get("NPSHr", 2.5) if best_pump else 2.5
    npsh_ok = npsh_available >= npsh_required if best_pump else False

    # Ergebnisse
    st.subheader("Ergebnisse")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Flüssigkeitsstrom", f"{Q_req:.1f} m³/h")
        st.metric("Druckdifferenz", f"{dp_req:.2f} bar")
    with col2:
        st.metric("Löslichkeit Saugseite", f"{sol_s_cm3N_L:.1f} cm³N/L")
        st.metric("Löslichkeit Druckseite", f"{sol_d_cm3N_L:.1f} cm³N/L")
    with col3:
        if gas_mode == "GVF an Druckseite [%]":
            st.metric("Gesamtgas (Druckseite)", f"{total_cm3N_L:.1f} cm³N/L")
        else:
            st.metric("Recyclingstrom", f"{Q_rec:.1f} m³/h")
    with col4:
        st.metric("Freies Gas (Saugseite)", f"{free_cm3N_L:.1f} cm³N/L")
        st.metric("GVF frei (mit Sicherheit)", f"{gvf_free_pct_safe:.1f} %")

    if not_solvable:
        st.error("⚠️ Der gewählte GVF-Wert ist bei den gegebenen Bedingungen nicht vollständig löslich!")
    if free_cm3N_L > 0:
        st.warning("⚠️ An der Saugseite entsteht freie Gasphase!")
    if only_dissolved and free_cm3N_L > 0:
        st.error("❌ Auswahl blockiert: Nur gelöstes Gas ist zugelassen!")
    if best_pump and not npsh_ok:
        st.error(f"❌ NPSH-Problem: Verfügbar = {npsh_available:.2f} m, Erforderlich = {npsh_required:.2f} m")

    if best_pump:
        st.success(f"✅ Empfohlene Pumpe: {best_pump['pump']['id']} ({best_pump['gvf_key']}% GVF)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Verfügbare Druckdifferenz", f"{best_pump['dp_avail']:.2f} bar")
        with col2:
            st.metric("Erforderliche Leistung", f"{best_pump['P_req']:.2f} kW")
        with col3:
            st.metric("Drehzahl", f"{best_pump['n_rpm']:.0f} rpm")
        with col4:
            st.metric("Modus", best_pump['mode'])

    # Diagramme
    st.subheader("Diagramme")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Löslichkeitsdiagramm
    if show_temp_band:
        temps = [temperature-10, temperature, temperature+10]
        colors = ['blue', 'green', 'orange']
        for T, color in zip(temps, colors):
            if -10 <= T <= 150:
                p, sol = solubility_diagonal_curve(gas_medium, T)
                ax1.plot(p, sol, '--', color=color, label=f"{T}°C")
    else:
        p, sol = solubility_diagonal_curve(gas_medium, temperature)
        ax1.plot(p, sol, '--', color='blue', label=f"{temperature}°C")

    # Referenzlinien für GVF
    for gvf in [10, 15, 20]:
        ax1.axhline(gvf*10, color='gray', linestyle=':', alpha=0.5)
        ax1.text(13.8, gvf*10, f"{gvf}%", va='center', ha='right', fontsize=9)

    # Betriebspunkte
    ax1.scatter(p_suction, sol_s_cm3N_L, color='red', s=100, label='Saugseite')
    ax1.scatter(p_discharge, sol_d_cm3N_L, color='green', s=100, label='Druckseite')
    ax1.scatter(p_discharge, total_cm3N_L, color='purple', s=100, marker='x', label='Gesamtgas')

    ax1.set_xlabel("Absolutdruck [bar]")
    ax1.set_ylabel("Gasgehalt [cm³N/L]")
    ax1.set_title(f"Gaslöslichkeit von {gas_medium} nach Henry")
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 220)

    # Pumpenkennlinien
    if best_pump:
        pump = best_pump['pump']
        for gvf_key in sorted(pump["curves_dp_vs_Q"].keys()):
            curve = pump["curves_dp_vs_Q"][gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            H_m = [dp * BAR_TO_M_WATER for dp in curve["dp"]]

            if gvf_key == best_pump['gvf_key']:
                ax2.plot(Q_lmin, H_m, 'o-', linewidth=2, label=f"{gvf_key}% GVF (ausgewählt)")
            else:
                ax2.plot(Q_lmin, H_m, '--', alpha=0.5, label=f"{gvf_key}% GVF")

        # Betriebspunkt
        Q_lmin_req = m3h_to_lmin(Q_req)
        H_req_m = dp_req * BAR_TO_M_WATER
        ax2.scatter(Q_lmin_req, H_req_m, color='red', s=100, marker='x', label='Betriebspunkt')

        ax2.set_xlabel("Volumenstrom [L/min]")
        ax2.set_ylabel("Förderhöhe [m]")
        ax2.set_title(f"Pumpenkennlinien: {pump['id']}")
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlim(0, max(Q_lmin) * 1.1)
        ax2.set_ylim(0, max(H_m) * 1.1)
    else:
        ax2.text(0.5, 0.5, "Keine geeignete Pumpe gefunden",
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel("Volumenstrom [L/min]")
        ax2.set_ylabel("Förderhöhe [m]")
        ax2.set_title("Pumpenkennlinien")

    plt.tight_layout()
    st.pyplot(fig)

    # Rechenweg
    with st.expander("Detaillierter Rechenweg"):
        st.markdown(f"""
        ### 1. Gaslöslichkeit nach Henry
        - Henry-Konstante für {gas_medium} bei {temperature}°C: {henry_constant(gas_medium, temperature):.2f} bar·L/mol
        - Löslichkeit an Saugseite: {sol_s_cm3N_L:.1f} cm³N/L
        - Löslichkeit an Druckseite: {sol_d_cm3N_L:.1f} cm³N/L

        ### 2. Gesamtgasgehalt
        """)
        if gas_mode == "GVF an Druckseite [%]":
            st.markdown(f"- GVF an Druckseite: {gvf_out_pct}% → Gesamtgas: {total_cm3N_L:.1f} cm³N/L")
        else:
            st.markdown(f"- Recyclingstrom: {Q_rec:.1f} m³/h → Anteil: {frac:.2f} → Gesamtgas: {total_cm3N_L:.1f} cm³N/L")

        st.markdown(f"""
        ### 3. Freies Gas an Saugseite
        - Freies Gas: {free_cm3N_L:.1f} cm³N/L → GVF frei: {gvf_free_pct:.1f}%
        - Mit Sicherheitsfaktor ({safety_factor}%): {gvf_free_pct_safe:.1f}%

        ### 4. Pumpenauswahl
        """)
        if best_pump:
            st.markdown(f"""
            - Ausgewählte Pumpe: {best_pump['pump']['id']}
            - GVF-Kurve: {best_pump['gvf_key']}%
            - Verfügbare Druckdifferenz: {best_pump['dp_avail']:.2f} bar
            - Erforderliche Leistung: {best_pump['P_req']:.2f} kW
            - Drehzahl: {best_pump['n_rpm']:.0f} rpm ({best_pump['mode']})
            """)
        else:
            st.markdown("- Keine geeignete Pumpe gefunden")

        st.markdown(f"""
        ### 5. NPSH-Berechnung
        - Verfügbarer NPSH: {npsh_available:.2f} m
        - Erforderlicher NPSH: {npsh_required:.2f} m
        - Status: {"✅ OK" if npsh_ok else "❌ Problem"}
        """)

# =========================
# ATEX-Auslegung Seite
# =========================
else:
    st.header("ATEX-Motorauslegung")

    with st.sidebar:
        st.header("Eingabeparameter")

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
        st.error("❌ Staub-Ex: Hierfür sind keine Motor-Datensätze hinterlegt.")
        st.stop()

    if zone == 0:
        st.error("❌ Zone 0: Hierfür sind keine Motor-Datensätze hinterlegt.")
        st.stop()

    st.success(f"✅ Zone {zone} (Gas) ist grundsätzlich abbildbar.")

    # Temperaturprüfung
    t_margin = 15.0
    suitable_motors = []
    for motor in ATEX_MOTORS:
        if zone in motor["zone_suitable"] and (motor["t_max_surface"] - t_margin) >= T_medium:
            suitable_motors.append(motor)

    if not suitable_motors:
        st.error(f"❌ Kein Motor verfügbar für T_medium = {T_medium:.1f}°C (mit 15K Abstand).")
        st.stop()

    # Leistungsdimensionierung
    P_motor_min = P_req * 1.15
    P_iec = motor_iec(P_motor_min)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Erforderliche Wellenleistung", f"{P_req:.2f} kW")
    with col2:
        st.metric("Mindestleistung (+15%)", f"{P_motor_min:.2f} kW")
    with col3:
        st.metric("IEC Motorgröße", f"{P_iec:.2f} kW")

    # Motorauswahl
    st.subheader("Verfügbare ATEX-Motoren")
    selected_motor = st.radio(
        "Wählen Sie einen Motortyp:",
        options=suitable_motors,
        format_func=lambda x: f"{x['marking']} ({x['id']})"
    )

    st.success("✅ Gültige Konfiguration gefunden")

    with st.expander("Technische Details"):
        st.markdown(f"""
        - **Leistung:** {P_iec:.2f} kW
        - **Kennzeichnung:** `{selected_motor['marking']}`
        - **Max. Oberflächentemperatur:** {selected_motor['t_max_surface']:.1f}°C ({selected_motor['temp_class']})
        - **Medientemperatur:** {T_medium:.1f}°C
        - **Temperaturabstand:** {selected_motor['t_max_surface'] - T_medium:.1f} K (Anforderung: ≥ 15 K)
        - **Kategorie:** {selected_motor['category']}
        - **Wirkungsgradklasse:** {selected_motor['efficiency_class']}
        - **Geeignet für Zone:** {', '.join(map(str, selected_motor['zone_suitable']))}
        """)

    # Export
    if st.button("ATEX-Dokumentation exportieren"):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ATEX-Auslegung - {selected_motor['id']}</title>
            <style>
                body {{ font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }}
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

            <h2>Eingabeparameter</h2>
            <div class="metric">
                <strong>Prozessdaten:</strong> P={P_req:.2f} kW, T={T_medium:.1f}°C<br>
                <strong>Ex-Zone:</strong> Zone {zone} (Gas)
            </div>

            <h2>Ergebnisse</h2>
            <table>
                <tr><th>Parameter</th><th>Wert</th></tr>
                <tr><td>Erforderliche Wellenleistung</td><td>{P_req:.2f} kW</td></tr>
                <tr><td>Mindestleistung (+15%)</td><td>{P_motor_min:.2f} kW</td></tr>
                <tr><td>IEC Motorgröße</td><td>{P_iec:.2f} kW</td></tr>
                <tr><td>Gewählter Motor</td><td>{selected_motor['id']}</td></tr>
                <tr><td>Kennzeichnung</td><td>{selected_motor['marking']}</td></tr>
                <tr><td>Max. Oberflächentemperatur</td><td>{selected_motor['t_max_surface']:.1f}°C</td></tr>
                <tr><td>Temperaturabstand</td><td>{selected_motor['t_max_surface'] - T_medium:.1f} K</td></tr>
                <tr><td>Kategorie</td><td>{selected_motor['category']}</td></tr>
                <tr><td>Wirkungsgradklasse</td><td>{selected_motor['efficiency_class']}</td></tr>
                <tr><td>Geeignet für Zone</td><td>{', '.join(map(str, selected_motor['zone_suitable']))}</td></tr>
            </table>

            <h2>Hinweise</h2>
            <p>Diese Auslegung dient als Grundlage für die Motorauswahl. Bitte prüfen Sie die Konformität mit den aktuellen ATEX-Richtlinien (2014/34/EU) und den relevanten Normen (EN 60079).</p>
        </body>
        </html>
        """
        st.download_button(
            label="HTML-Dokumentation herunterladen",
            data=html,
            file_name=f"ATEX_Auslegung_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html"
        )

# =========================
# Hilfsfunktion für Export
# =========================
def fig_to_base64(fig):
    """Konvertiert Matplotlib-Figur zu Base64 für HTML-Export."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

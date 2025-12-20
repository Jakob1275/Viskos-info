import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from iapws import IAPWS97
import re
import math

G = 9.80665  # m/s²

# ---------------------------
# Pumpenkennlinien mit Leistung
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

MPH_PUMPS = [
    {
        "id": "MPH-50",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 25,
        "p_max_bar": 9,
        "GVF_max": 0.4,
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
        "id": "MPH-200",
        "type": "Mehrphasenpumpe",
        "Q_max_m3h": 100,
        "p_max_bar": 30,
        "GVF_max": 0.4,
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
    }
]
# ---------------------------
# ATEX-Datenbank
# ---------------------------
ATEX_MOTORS = [
    {
        "id": "Standard Zone 2 (ec)",
        "marking": "II 3G Ex ec IIC T3 Gc",
        "zone_suitable": [2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "3G",
        "description": "Standardlösung für Zone 2. Wirtschaftlichste Option. Wählbar, wenn T3 (200°C) für das Gas ausreicht und die Medientemperatur < 185°C ist."
    },
    {
        "id": "Zone 1 (eb)", 
        "marking": "II 2G Ex eb IIC T3 Gb", 
        "zone_suitable": [1, 2],
        "temp_class": "T3",
        "t_max_surface": 200.0,
        "category": "2G",
        "description": "Standardlösung für Zone 1 ('Erhöhte Sicherheit'). Wählbar, wenn T3 ausreichend ist. Leichter und günstiger als druckfeste Kapselung."
    },
    {
        "id": "Zone 1 (db eb) T4",
        "marking": "II 2G Ex db eb IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "description": "Für Zone 1 mit strengeren Temperaturanforderungen (T4). Kombination: Motor druckfest (db), Anschlusskasten erhöhte Sicherheit (eb) für einfachere Installation."
    },
    {
        "id": "Zone 1 (db) T4",
        "marking": "II 2G Ex db IIC T4 Gb",
        "zone_suitable": [1, 2],
        "temp_class": "T4",
        "t_max_surface": 135.0,
        "category": "2G",
        "description": "Für Zone 1 (T4). Vollständig druckfeste Kapselung inkl. Anschlusskasten. Robusteste Ausführung für raue Umgebungen."
    }
]

# Temperaturklassen Definition
TEMP_CLASSES_LIMITS = {
    "T1": 450.0,
    "T2": 300.0,
    "T3": 200.0,
    "T4": 135.0,
    "T5": 100.0,
    "T6": 85.0
}

HENRY_CONSTANTS = {
    "Luft": {"A": 1400.0, "B": 1500},
    "CO2": {"A": 29.4, "B": 2400},
    "O2": {"A": 1500.0, "B": 1500},
    "N2": {"A": 1650.0, "B": 1300},
    "CH4": {"A": 1400.0, "B": 1600},
}

def clamp(x, a, b): return max(a, min(b, x))

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s: return s
    return steps[-1]

# Viskositätskorrektur
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    Q, H, nu = max(Q_m3h, 1e-6), max(H_m, 1e-6), max(nu_cSt, 1e-6)
    Q_gpm, H_ft = Q * 4.40287, H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B, nu_cSt):
    if B <= 1.0: return 1.0, 1.0
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = clamp(CH, 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = clamp(Ceta, 0.1, 1.0)
    return CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt):
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B, nu_cSt)
    return {"B": B, "CH": CH, "Ceta": Ceta, "Q_water": Q_vis, "H_water": H_vis / CH}

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    B = compute_B_HI(Q_water, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B, nu_cSt)
    return Q_water, H_water * CH, max(1e-6, eta_water * Ceta)

def generate_viscous_curve(pump, nu_cSt, rho):
    Q_vis, H_vis, eta_vis, P_vis = [], [], [], []
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        P_v = (rho * G * Q_v * H_v) / (3600 * 1000 * max(eta_v, 1e-6))
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
        if not in_range and not allow_out_of_range: continue
        Q_eval = clamp(Q_water, qmin, qmax)
        penalty = 0.0 if in_range else abs(Q_water - Q_eval) / max(qmax - qmin, 1e-9) * 10.0
        H_at = interp_clamped(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp_clamped(Q_eval, p["Qw"], p["eta"])
        score = abs(H_at - H_water) + penalty
        cand = {"id": p["id"], "pump": p, "in_range": in_range, "Q_eval": Q_eval,
                "H_at": H_at, "eta_at": eta_at, "errH": abs(H_at - H_water), "score": score}
        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    return best

def linspace(start, stop, num):
    if num == 1: return [start]
    if num < 1: return []
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
            return ys[i-1] + (ys[i] - ys[i-1]) * (x - xs[i-1]) / (xs[i] - xs[i-1])
    return ys[-1]
    
def henry_constant(gas, T_celsius):
    """Temperaturabhängige Henry-Konstante"""
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1/T_K - 1/T0_K))

def gas_solubility_volumetric(gas, p_bar, T_celsius):
    """Löslichkeit in cm³/L"""
    H = henry_constant(gas, T_celsius)
    C_mol_L = p_bar / H
    return C_mol_L * 22400

def solubility_curve_vs_pressure(gas, T_celsius, p_max=14):
    """Löslichkeitskurve"""
    pressures = linspace(0, p_max, 100)
    solubilities = [gas_solubility_volumetric(gas, p, T_celsius) for p in pressures]
    return pressures, solubilities

def affinity_laws_scale(Q_nom, H_nom, P_nom, n_ratio):
    """
    Affinitätsgesetze für Kreiselpumpen:
    Q ~ n
    H ~ n²
    P ~ n³
    
    n_ratio = n_new / n_nominal
    """
    return Q_nom * n_ratio, H_nom * (n_ratio**2), P_nom * (n_ratio**3)

def find_speed_for_duty_point(pump_curve_Q, pump_curve_H, Q_req, H_req):
    """
    Findet die optimale Drehzahl, um den geforderten Betriebspunkt zu treffen.
    
    Vorgehensweise:
    1. Finde Punkt auf Nennkennlinie bei Q_req
    2. Berechne benötigtes n_ratio: sqrt(H_req / H_nominal)
    
    Returns: n_ratio (1.0 = Nenndrehzahl), oder None wenn nicht möglich
    """
    # Interpoliere H bei Q_req auf Nennkennlinie
    H_nom_at_Q = interp_clamped(Q_req, pump_curve_Q, pump_curve_H)
    
    # Benötigte Drehzahl: H ~ n²  =>  n_ratio = sqrt(H_req / H_nom)
    if H_nom_at_Q <= 0:
        return None
    
    n_ratio = math.sqrt(H_req / H_nom_at_Q)
    
    # Prüfe ob Drehzahl im sinnvollen Bereich (50% - 110%)
    if n_ratio < 0.5 or n_ratio > 1.1:
        return None
    
    return n_ratio

def choose_best_mph_pump_with_speed(MPH_PUMPS, Q_req, p_req, gvf_req):
    """
    Erweiterte Pumpenauswahl mit Drehzahlanpassung
    
    Strategie:
    1. Prüfe ob Pumpe grundsätzlich geeignet (Q_max, GVF_max)
    2. Nutze WORST-CASE GVF-Kurve (höchster erwarteter GVF)
    3. Berechne optimale Drehzahl für Betriebspunkt
    4. Prüfe ob bei dieser Drehzahl p >= p_req
    """
    best = None
    
    for pump in MPH_Pumps:
        # Check 1: Volumenstrom (mit Sicherheit, da Drehzahl < 110%)
        if Q_req > pump["Q_max_m3h"] * 1.1:  
            continue
        
        # Check 2: GVF - nehme HÖCHSTE verfügbare Kurve <= gvf_req (Worst Case)
        available_gvf_keys = [k for k in pump["curves_p_vs_Q"].keys() if k <= gvf_req]
        if not available_gvf_keys:
            # Wenn gvf_req < kleinste Kurve, nutze kleinste Kurve
            gvf_key = min(pump["curves_p_vs_Q"].keys())
        else:
            # WICHTIG: Nutze HÖCHSTE Kurve als Worst-Case-Szenario
            gvf_key = max(available_gvf_keys)
        
        # Wenn gvf_req die Pumpengrenze überschreitet
        if gvf_req > pump["GVF_max"] * 100:
            continue
        
        curve = pump["curves_p_vs_Q"][gvf_key]
        power_curve = pump["power_kW_vs_Q"][gvf_key]
        
        # Prüfe ob Q im Kennlinienbereich
        Q_min_curve = min(curve["Q"])
        Q_max_curve = max(curve["Q"])
        
        # Variante A: Betrieb bei Nenndrehzahl
        if Q_min_curve <= Q_req <= Q_max_curve:
            p_available_nom = interp_clamped(Q_req, curve["Q"], curve["p"])
            P_nom = interp_clamped(Q_req, power_curve["Q"], power_curve["P"])
            
            if p_available_nom >= p_req * 0.95:  # 5% Toleranz
                score = abs(p_available_nom - p_req) + abs(gvf_key - gvf_req) * 0.2
                
                cand = {
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "p_available": p_available_nom,
                    "P_required": P_nom,
                    "n_ratio": 1.0,
                    "n_rpm": None,  # Nenndrehzahl (unbekannt ohne weitere Daten)
                    "score": score,
                    "p_reserve": p_available_nom - p_req,
                    "mode": "Nenndrehzahl"
                }
                
                if best is None or score < best["score"]:
                    best = cand
        
        # Variante B: Drehzahlanpassung
        # Finde optimale Drehzahl für Q_req und p_req
        n_ratio = find_speed_for_duty_point(curve["Q"], curve["p"], Q_req, p_req)
        
        if n_ratio is not None:
            # Skaliere Kennlinie mit Affinitätsgesetzen
            Q_scaled_max = Q_max_curve * n_ratio
            
            # Prüfe ob Q_req noch im Bereich
            if Q_req <= Q_scaled_max:
                # Berechne verfügbaren Druck bei skalierter Kennlinie
                p_available_scaled = p_req  # Per Definition, da wir n_ratio so berechnet haben
                
                # Leistung skalieren (P ~ n³)
                P_scaled = interp_clamped(Q_req / n_ratio, power_curve["Q"], power_curve["P"]) * (n_ratio**3)
                
                # Score: Bevorzuge Nenndrehzahl, aber mit Drehzahlanpassung ist auch OK
                score = abs(1.0 - n_ratio) * 5.0 + abs(gvf_key - gvf_req) * 0.2
                
                cand = {
                    "pump": pump,
                    "gvf_curve": gvf_key,
                    "p_available": p_available_scaled,
                    "P_required": P_scaled,
                    "n_ratio": n_ratio,
                    "n_rpm": None,  # Könnte berechnet werden wenn Nenndrehzahl bekannt
                    "score": score,
                    "p_reserve": 0.0,  # Exakt getroffen
                    "mode": f"Drehzahl {n_ratio*100:.1f}%"
                }
                
                if best is None or score < best["score"]:
                    best = cand
    
    return best

# Streamlit App
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
    st.info(f"**Aktiv:** {'Pumpen' if st.session_state.page=='pump' else 'Mehrphasen'}")

# PAGE: PUMPEN
if st.session_state.page == "pump":
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur")
    
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben")
        Q_vis_req = st.number_input("Qᵥ, Förderstrom [m³/h]", 0.1, 300.0, 40.0, 1.0)
        H_vis_req = st.number_input("Hᵥ, Förderhöhe [m]", 0.1, 300.0, 35.0, 1.0)
        mk = st.selectbox("Medium", list(MEDIA.keys()), 0)
        rho_def, nu_def = MEDIA[mk]
        rho = st.number_input("ρ [kg/m³]", 1.0, 2000.0, float(rho_def), 5.0)
        nu = st.number_input("ν [cSt]", 0.1, 1000.0, float(nu_def), 0.5)
        allow_out = st.checkbox("Auswahl außerhalb Kennlinie", True)
        reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
    
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water, H_water, B, CH, Ceta = conv["Q_water"], conv["H_water"], conv["B"], conv["CH"], conv["Ceta"]
    
    st.info(f"{'✅' if B < 1.0 else '⚠️'} B = {B:.2f} {'< 1.0 → Geringe Viskositätseffekte' if B < 1.0 else '≥ 1.0 → Viskositätskorrektur erforderlich'}")
    
    st.markdown("### 📊 Umrechnung viskos → Wasser")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q_Wasser", f"{Q_water:.2f} m³/h")
    col2.metric("H_Wasser", f"{H_water:.2f} m", f"+{H_water-H_vis_req:.1f}m")
    col3.metric("B-Zahl", f"{B:.2f}")
    col4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")
    
    best = choose_best_pump(pumps, Q_water, H_water, allow_out_of_range=True)
    if not best:
        st.error("❌ Keine Pumpe gefunden!")
        st.stop()
    
    p = best["pump"]
    eta_water, eta_vis = best["eta_at"], best["eta_at"] * Ceta
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / max(eta_vis, 1e-6)) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))
    
    # HERVORGEHOBENE ERGEBNISSE
    st.divider()
    st.markdown("### ✅ **AUSLEGUNGSERGEBNIS**")
    st.success(f"**Gewählte Pumpe: {best['id']}**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Q", f"{Q_req:.1f} m³/h", f"{m3h_to_lmin(Q_req):.1f} l/min")
    col2.metric("p verfügbar", f"{best['p_available']:.2f} bar")
    col3.metric("Drehzahl", f"{best['n_ratio']*100:.1f}%", 
               "Nenn-n" if abs(best['n_ratio'] - 1.0) < 0.05 else "variable n")
    col4.metric("GVF (Eingabe)", f"{gvf_req:.0f} %")
    col5.metric("Leistung", f"{best['P_required']:.2f} kW")
    col6.metric("Max. GVF Pumpe", f"{best['pump']['GVF_max']*100:.0f} %")
    
    if not best["in_range"]:
        st.warning(f"⚠️ Q außerhalb Kennlinie ({min(p['Qw'])}...{max(p['Qw'])} m³/h)")
    
    Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(p, nu, rho)
    P_water_kW_op = interp_clamped(Q_water, p["Qw"], p["Pw"])
    
    st.divider()
    st.markdown("### 📈 Kennlinien")
    tab1, tab2, tab3 = st.tabs(["Q-H", "Q-η", "Q-P"])
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(p["Qw"], p["Hw"], "o-", linewidth=2, color="blue", label=f"{p['id']} (Wasser)")
        ax1.plot(Q_vis_curve, H_vis_curve, "s--", linewidth=2.5, color="red", label=f"{p['id']} (viskos)")
        ax1.scatter([Q_water], [H_water], marker="^", s=150, color="blue", edgecolors="black", linewidths=2, label="BP (Wasser)", zorder=5)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=200, color="red", linewidths=3, label="BP (viskos)", zorder=5)
        ax1.set_xlabel("Q [m³/h]", fontsize=12)
        ax1.set_ylabel("H [m]", fontsize=12)
        ax1.set_title("Q-H Kennlinien", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        st.pyplot(fig1, clear_figure=True)
    
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(p["Qw"], p["eta"], "o-", linewidth=2, color="blue", label=f"{p['id']} (Wasser)")
        ax2.plot(Q_vis_curve, eta_vis_curve, "s--", linewidth=2.5, color="red", label=f"{p['id']} (viskos)")
        ax2.scatter([Q_water], [eta_water], marker="^", s=150, color="blue", edgecolors="black", linewidths=2, label="η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=200, color="red", linewidths=3, label="η (viskos)", zorder=5)
        ax2.set_xlabel("Q [m³/h]", fontsize=12)
        ax2.set_ylabel("η [-]", fontsize=12)
        ax2.set_title("Q-η Kennlinien", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        st.pyplot(fig2, clear_figure=True)
    
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(p["Qw"], p["Pw"], "o-", linewidth=2, color="blue", label=f"{p['id']} (Wasser)")
        ax3.plot(Q_vis_curve, P_vis_curve, "s--", linewidth=2.5, color="red", label=f"{p['id']} (viskos, ρ={rho:.0f})")
        ax3.scatter([Q_water], [P_water_kW_op], marker="^", s=150, color="blue", edgecolors="black", linewidths=2, label="BP (Wasser)", zorder=5)
        ax3.scatter([Q_vis_req], [P_vis_kW], marker="x", s=200, color="red", linewidths=3, label="BP (viskos)", zorder=5)
        ax3.set_xlabel("Q [m³/h]", fontsize=12)
        ax3.set_ylabel("P [kW]", fontsize=12)
        ax3.set_title("Q-P Kennlinien", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        st.pyplot(fig3, clear_figure=True)
    
    # Rechenweg
    with st.expander("📘 Rechenweg & Theorie", expanded=False):
        st.markdown(f"""
        ## Viskositätskorrektur nach Hydraulic Institute
        
        ### 1️⃣ Gegeben (viskoses Medium)
        - Q_vis = {Q_vis_req:.2f} m³/h
        - H_vis = {H_vis_req:.2f} m
        - ν = {nu:.2f} cSt
        - ρ = {rho:.1f} kg/m³
        
        ### 2️⃣ B-Zahl berechnen
    
        **Klassische dimensionslose spezifische Drehzahl (B):**
        """)

        # 2. LaTeX-Aufruf für die klassische Formel (MUSS SEPARAT SEIN)
        st.latex(r"""
            B = \frac{n \cdot \sqrt{Q}}{(g \cdot H)^{0.75}}
        """)

        # 3. Kleiner Markdown für die Überschrift der zweiten Formel
        st.markdown(r"**Verwendete Kennzahl (B-Zahl) für diese Anwendung:**")
    
        # 4. LaTeX-Aufruf für die anwendungsspezifische Formel (MUSS SEPARAT SEIN)
        st.latex(r"""
            B_{\text{Code}} = \frac{16.5 \cdot \sqrt{\nu}}{Q^{0.25} \cdot H^{0.375}} 
        """)
            
        # 5. Code-Block für die Implementierung (MUSS SEPARAT SEIN)
        st.code(
            f"""
    B = 16.5 * nu**0.5 / (Q**0.25 * H**0.375)
    B = {B:.2f}
            """
        )
        st.markdown(f"""
        ### 3️⃣ Korrekturfaktoren bestimmen
        - **CH** (Förderhöhe): {CH:.3f}
        - **Cη** (Wirkungsgrad): {Ceta:.3f}
        
        ### 4️⃣ Umrechnung auf Wasserkennlinie
        ```
        Q_Wasser = Q_vis = {Q_water:.2f} m³/h  (bleibt konstant!)
        H_Wasser = H_vis / CH = {H_vis_req:.2f} / {CH:.3f} = {H_water:.2f} m
        ```
        
        ### 5️⃣ Pumpenauswahl
        Beste Pumpe: **{best["id"]}**
        - H_Pumpe(Q_Wasser) = {best["H_at"]:.2f} m
        - η_Wasser(Q) = {eta_water:.3f}
        
        ### 6️⃣ Rückrechnung auf viskos
        ```
        H_viskos = H_Wasser × CH = {H_water:.2f} × {CH:.3f} = {H_water * CH:.2f} m
        η_viskos = η_Wasser × Cη = {eta_water:.3f} × {Ceta:.3f} = {eta_vis:.3f}
        ```
        
        ### 7️⃣ Leistungsberechnung
        ```
        P_hyd = ρ × g × Q × H = {P_hyd_W:.0f} W
        P_Welle = P_hyd / η_viskos = {P_vis_kW:.2f} kW
        P_Motor = P_Welle × (1 + Reserve) = {P_motor_kW:.2f} kW
        ```
        ---
        ## 📚 Normenbezug
        - Hydraulic Institute Standards (ANSI/HI)
        - ISO/TR 17766 (Pumps - Viscosity correction)
        - DIN EN ISO 9906 (Abnahmeprüfung auf Wasserbasis)
        """)

# =========================================================
# PAGE 2: Mehrphase
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("⚗️ Mehrphasen: Löslichkeit + Pumpenkennlinien + automatische Auswahl")

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben")

        gas_medium = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)
        temperature = st.number_input("Temperatur [°C]", -10.0, 100.0, 20.0, 1.0)

        st.divider()
        st.subheader("Betriebspunkt")
        
        Q_req = st.number_input("Volumenstrom Q [m³/h]", 0.1, 100.0, 15.0, 1.0)
        gvf_req = st.slider("Gasanteil GVF [%]", 0, 40, 10, 1)
        p_req = st.number_input("Systemdruck p [bar]", 0.1, 30.0, 5.0, 0.5)

        st.divider()
        st.subheader("Plot-Optionen")
        show_temp_band = st.checkbox("Löslichkeit bei T-10/T/T+10 zeigen", value=True)

    # --- Automatische Pumpenauswahl ---
    best = choose_best_mph_pump_with_speed(MPH_PUMPS, Q_req, p_req, gvf_req)

    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: Löslichkeit vs. Druck ---
    temp_variants = [temperature - 10, temperature, temperature + 10]
    temp_variants = [t for t in temp_variants if -10 <= t <= 100]

    colors_temp = ['lightblue', 'darkblue', 'blue']
    for i, T in enumerate(temp_variants):
        pressures, solubilities = solubility_curve_vs_pressure(gas_medium, T, p_max=15)
        ax1.plot(pressures, solubilities, '--', linewidth=2, color=colors_temp[i],
                 label=f"Löslichkeit {gas_medium} {T:.0f}°C")

    # Betriebspunkt Löslichkeit
    sol_at_op = gas_solubility_volumetric(gas_medium, p_req, temperature)
    ax1.scatter([p_req], [sol_at_op], s=200, marker='o', color='red',
                edgecolors='black', linewidths=2, zorder=5, label='Betriebspunkt')

    ax1.set_xlabel("Druck [bar]", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Löslichkeit [cm³/l]", fontsize=12, fontweight='bold')
    ax1.set_title("Gaslöslichkeit nach Henry's Law", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 180)

    Q_req_lmin = m3h_to_lmin(Q_req)
    
    # --- PLOT 2: p-Q Pumpenkennlinien ---
    if best:
        pump = best["pump"]
        curves = pump["curves_p_vs_Q"]
    
        # Zeichne alle GVF-Kurven dieser Pumpe
        gvf_colors = {0: 'black', 5: 'green', 10: 'blue', 15: 'red', 20: 'purple'}
    
        for gvf_key in sorted(curves.keys()):
            curve = curves[gvf_key]
            Q_lmin = [m3h_to_lmin(q) for q in curve["Q"]]
            lw = 3.0 if gvf_key == best["gvf_curve"] else 1.8
            alpha = 1.0 if gvf_key == best["gvf_curve"] else 0.5
            color = gvf_colors.get(gvf_key, 'gray')
            ax2.plot(Q_lmin, curve["p"], 'o-', linewidth=lw, alpha=alpha, color=color,
                    label=f"{pump['id']} ({gvf_key}% GVF)")
    
        # Betriebspunkt
        ax2.scatter([Q_req_lmin], [p_req], s=200, marker='o', color='red',
                    edgecolors='black', linewidths=2, zorder=5, label='Betriebspunkt')
    
        ax2.set_xlabel("Volumenstrom Q [l/min]", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Druck p [bar]", fontsize=12, fontweight='bold')
        ax2.set_title(f"Mehrphasen-Kennlinien: {pump['id']}", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xlim(0, max(m3h_to_lmin(pump["Q_max_m3h"]), Q_req_lmin * 1.2))
        ax2.set_ylim(0, pump["p_max_bar"] * 1.1)
    else:
        ax2.text(0.5, 0.5, '❌ Keine geeignete Pumpe gefunden',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='red')
        ax2.set_xlabel("Volumenstrom Q [l/min]", fontsize=12)
        ax2.set_ylabel("Druck p [bar]", fontsize=12)
        ax2.set_title("Mehrphasen-Kennlinien", fontsize=14)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Ergebnisse
    st.divider()

    if best:
        # Löslichkeit berechnen
        sol_L_L = sol_at_op / 1000.0
        gvf_max = (sol_L_L / (1.0 + sol_L_L)) * 100
    
        st.markdown("### ✅ **Betriebspunkt Maximaler Löslichkeit**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("**Druck**", f"**{p_req:.2f} bar**")
        col2.metric("**Temperatur**", f"**{temperature:.1f} °C**")
        col3.metric("**Max. Löslichkeit**", f"**{sol_L_L:.3f} L/L**")
        col4.metric("**Max. GVF**", f"**{gvf_max:.1f} %**")
    
        st.markdown("### 🔧 **Empfohlene Pumpe**")
        st.success(f"**{best['pump']['id']}** bei **{best['gvf_curve']}% GVF**")
    
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Q", f"{Q_req:.1f} m³/h", f"{m3h_to_lmin(Q_req):.1f} l/min")
        col2.metric("p verfügbar", f"{best['p_available']:.2f} bar")
        col3.metric("GVF (Eingabe)", f"{gvf_req:.0f} %")
        col4.metric("Leistung", f"{best['P_required']:.2f} kW")
        col5.metric("Max. GVF Pumpe", f"{best['pump']['GVF_max']*100:.0f} %")
    
        if gvf_req > gvf_max:
            st.warning(f"⚠️ Eingegebener GVF ({gvf_req}%) liegt über max. Löslichkeit ({gvf_max:.1f}%) → Freies Gas vorhanden!")
        else:
            st.info(f"ℹ️ GVF liegt im Löslichkeitsbereich → Gas kann vollständig gelöst werden")
    
    else:
        st.error("❌ **Keine geeignete Pumpe gefunden!**")
        st.markdown("""
        **Mögliche Gründe:**
        - Volumenstrom zu hoch für verfügbare Pumpen
        - Gasanteil übersch reitet Pumpengrenze (max. 40%)
        - Systemdruck kann von keiner Pumpe erreicht werden
    
        **Empfehlung:**
        - Reduzieren Sie Q, GVF oder p
        - Oder wählen Sie eine größere Pumpe
        """)

    # Pumpenspezifikationen
    st.divider()
    st.markdown("### 📋 Verfügbare Pumpen")

    cols = st.columns(len(MPH_PUMPS))
    for i, pump in enumerate(MPH_PUMPS):
        with cols[i]:
            selected = best and best["pump"]["id"] == pump["id"]
            if selected:
                st.success(f"✅ **{pump['id']}**")
            else:
                st.info(f"**{pump['id']}**")
            st.caption(f"Typ: {pump['type']}")
            st.caption(f"Q_max: {pump['Q_max_m3h']} m³/h")
            st.caption(f"p_max: {pump['p_max_bar']} bar")
            st.caption(f"GVF_max: {pump['GVF_max']*100:.0f}%")

    # Footer
    st.divider()
    
    # Erklärung
    # Rechenweg und Theorie
    with st.expander("📘 Rechenweg & Theorie: Mehrphasen-Auslegung", expanded=False):
        if best:
            # Berechnete Werte
            H_henry = henry_constant(gas_medium, temperature)
            C_mol_L = p_req / H_henry
        
            st.markdown(f"""
            ## Mehrphasen-Pumpenauslegung: Gaslöslichkeit und GVF
        
            ### 1️⃣ Gegeben (Betriebsbedingungen)
            - Gas: **{gas_medium}**
            - Druck: **p = {p_req:.2f} bar**
            - Temperatur: **T = {temperature:.1f} °C**
            - Volumenstrom: **Q = {Q_req:.1f} m³/h** ({Q_req_lmin:.1f} l/min)
            - GVF (Eingabe): **{gvf_req:.0f}%**
        
            ---
        
            ### 2️⃣ Henry's Law: Gaslöslichkeit berechnen
        
            **Grundprinzip:** Die Löslichkeit eines Gases in einer Flüssigkeit ist proportional zum Partialdruck.
        
            ```
            C = p / H(T)
            ```
        
            #### Schritt 2a: Henry-Konstante bei Betriebstemperatur
        
            Die Henry-Konstante ändert sich mit der Temperatur nach van't Hoff:
        
            ```
            H(T) = H₀ × exp[B × (1/T - 1/T₀)]
            ```
        
            Für **{gas_medium}** bei **T = {temperature:.1f}°C**:
            ```
            H({temperature:.1f}°C) = {H_henry:.1f} bar·L/mol
            ```
        
            #### Schritt 2b: Konzentration berechnen
        
            ```
            C = p / H(T) = {p_req:.2f} / {H_henry:.1f} = {C_mol_L:.6f} mol/L
            ```
        
            #### Schritt 2c: Umrechnung auf Volumenverhältnis
        
            Mit dem idealen Gasgesetz (1 mol Gas = 22.400 cm³ bei STP):
        
            ```
            V_Gas/V_Flüssigkeit = C × 22.400 cm³/mol
            V_Gas/V_Flüssigkeit = {C_mol_L:.6f} × 22.400 = {sol_at_op:.2f} cm³/L
            ```
        
            Umrechnung auf L/L:
            ```
            V_Gas/V_Flüssigkeit = {sol_at_op:.2f} cm³/L ÷ 1000 = {sol_L_L:.3f} L/L
            ```
        
            **Interpretation:** Pro Liter Flüssigkeit können maximal **{sol_L_L:.3f} Liter** Gas gelöst werden.
        
            ---
        
            ### 3️⃣ Gas Volume Fraction (GVF) berechnen
        
            Der **GVF** beschreibt den Volumenanteil des Gases im Zweiphasengemisch:
        
            ```
            GVF = V_Gas / (V_Gas + V_Flüssigkeit)
            ```
        
            Bei **1 Liter Flüssigkeit** mit **{sol_L_L:.3f} L** gelöstem Gas:
        
            ```
            GVF = {sol_L_L:.3f} / (1.0 + {sol_L_L:.3f})
            GVF = {sol_L_L:.3f} / {(1.0 + sol_L_L):.3f}
            GVF = {(sol_L_L / (1.0 + sol_L_L)):.4f}
            GVF = {gvf_max:.2f}%
            ```
        
            **Ergebnis:** Bei diesem Betriebspunkt beträgt der maximale Gasvolumenanteil **{gvf_max:.2f}%**.
        
            ---
        
            ### 4️⃣ Pumpenauswahl und Bewertung
        
            #### Gewählte Pumpe: **{best['pump']['id']}**
        
            - Kennlinie: **{best['gvf_curve']}% GVF**
            - Max. GVF der Pumpe: **{best['pump']['GVF_max']*100:.0f}%**
            - Berechneter max. GVF (aus Löslichkeit): **{gvf_max:.2f}%**
            - Eingabe GVF: **{gvf_req:.0f}%**
        
            **Bewertung:**
            {"✅ **Eingegebener GVF liegt unter max. Löslichkeit** → Gas kann vollständig gelöst werden" if gvf_req <= gvf_max else "⚠️ **Eingegebener GVF liegt über max. Löslichkeit** → Freies Gas vorhanden!"}
        
            #### Kennlinien-Check
        
            - p verfügbar bei Q: **{best['p_available']:.2f} bar**
            - p erforderlich: **{p_req:.2f} bar** {"✅" if best['p_available'] >= p_req else "❌"}
            - Leistungsaufnahme: **{best['P_required']:.2f} kW**
            - **Drehzahl-Modus**: {best['mode']}
            - **Drehzahl-Verhältnis**: n/n_nom = {best['n_ratio']:.3f}
            {f"- **Energieeinsparung**: ~{(1 - best['n_ratio']**3)*100:.1f}% bei Teillast" if best['n_ratio'] < 0.95 else ""}
            
            ---
            
            ### 4a️⃣ Affinitätsgesetze für Drehzahlanpassung
            
            Bei Kreiselpumpen gelten folgende Zusammenhänge:
            
            ```
            Q ~ n        (Volumenstrom proportional zur Drehzahl)
            H ~ n²       (Förderhöhe proportional zu Drehzahl²)
            P ~ n³       (Leistung proportional zu Drehzahl³)
            ```
            
            **Für diesen Betriebspunkt:**
            - Nenndrehzahl würde bei Q={Q_req:.1f} m³/h einen Druck von {interp_clamped(Q_req, pump['curves_p_vs_Q'][best['gvf_curve']]['Q'], pump['curves_p_vs_Q'][best['gvf_curve']]['p']):.2f} bar liefern
            - Erforderlich sind: {p_req:.2f} bar
            - Optimale Drehzahl: {best['n_ratio']*100:.1f}% der Nenndrehzahl
            
            **Energetischer Vorteil:**
            {"Durch die Drehzahlreduktion wird die Leistungsaufnahme um etwa " + f"{(1 - best['n_ratio']**3)*100:.1f}% reduziert (P ~ n³)" if best['n_ratio'] < 0.95 else "Betrieb bei Nenndrehzahl optimal"}

            ---
        
            ### 5️⃣ Physikalische Zusammenhänge
        
            #### Warum steigen die Löslichkeitskurven?
        
            Nach Henry's Law ist die Löslichkeit **direkt proportional zum Druck**:
            - Höherer Druck → Mehr Gasmoleküle werden in die Flüssigkeit "gedrückt"
            - Niedrigerer Druck → Gas entweicht aus der Lösung (Ausgasung)
        
            #### p-Q Kennlinien der Mehrphasenpumpen
        
            Die p-Q Kennlinien zeigen den **verfügbaren Druck bei gegebenem Volumenstrom**:
            - Bei niedrigem Q: Pumpe kann höheren Druck aufbauen
            - Bei hohem Q: Druck sinkt (typisches Pumpenverhalten)
            - Mit steigendem GVF: Kennlinien verschieben sich nach unten (Leistungsreduktion)
        
            #### Temperatureinfluss
        
            Höhere Temperatur → **Niedrigere Löslichkeit**
            - Die Henry-Konstante H(T) steigt mit der Temperatur
            - Weniger Gas kann gelöst werden
            - Höheres Ausgasungsrisiko
        
            ---
        
            ### 6️⃣ Praktische Auslegungsempfehlungen
        
            ✅ **Sicherheitsabstände einhalten:**
            - GVF: Mind. **10-15%** unter Pumpengrenze betreiben
            - Druck: Mind. **10%** Reserve zum erforderlichen Druck
            - Berücksichtigung von Druckschwankungen im System
        
            ✅ **Kritische Betriebspunkte vermeiden:**
            - Saugseite: Niedrige Drücke → Ausgasungsrisiko
            - Hohe Temperaturen: Reduzierte Löslichkeit
            - GVF > Löslichkeitsgrenze → Freies Gas vorhanden
        
            ✅ **Für Engineering-Auslegung:**
            - Immer Herstellerdaten für spezifischen Volumenstrom verwenden
            - MPH-Kennlinien vom Hersteller anfordern
            - NPSH-Anforderungen bei Mehrphasenbetrieb prüfen
            - Druckverluste in Rohrleitung berücksichtigen
        
            ---
        
            ### 📚 Normen und Standards
        
            - **API 610**: Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries
            - **ISO 13709**: Centrifugal pumps for petroleum, petrochemical and natural gas industries
            - **VDI 2173**: Pumps for liquids - Selection of pump types
        
            **Hinweis zur Gaslöslichkeit:**
            - Henry-Konstanten sind experimentell bestimmte Werte
            - Gelten für ideale Bedingungen (reine Komponenten, niedrige Konzentration)
            - Bei Gemischen: Partialdruck jeder Komponente einzeln betrachten
            """)
        else:
            st.info("💡 Keine Pumpe gefunden - passen Sie die Parameter an, um die Berechnung zu sehen.")
            st.markdown("""
            ## Grundlagen der Mehrphasen-Auslegung
        
            ### Henry's Law
        
            Die Löslichkeit von Gasen in Flüssigkeiten folgt dem **Henry'schen Gesetz**:
        
            ```
            C = p / H(T)
            ```
        
            - Bei **höherem Druck** löst sich mehr Gas
            - Bei **höherer Temperatur** sinkt die Löslichkeit
            - Die Henry-Konstante H(T) ist gasspezifisch
        
            ### Gas Volume Fraction (GVF)
        
            Der GVF beschreibt den Volumenanteil des freien Gases:
        
            ```
            GVF = V_Gas / (V_Gas + V_Flüssigkeit)
            ```
        
            - **GVF = 0%**: Reiner Flüssigkeitsstrom (einphasig)
            - **GVF > 0%**: Zweiphasenströmung (Gas + Flüssigkeit)
            - **GVF → 100%**: Reiner Gasstrom
            """)

    st.caption("⚗️ Mehrphasen-Pumpenauswahl v3.0 | Für Engineering immer Herstellerdaten verwenden!")
# =========================================================
# PAGE 3: ATEX-MOTORAUSWAHL
# =========================================================
elif st.session_state.page == "atex":
    st.subheader("⚡ ATEX-Motorauslegung")
    st.caption("Auslegung nach RL 2014/34/EU")

    # Layout: Links Eingabe, Rechts Logik/Ergebnis
    col_in, col_res = st.columns([1, 2])

    with col_in:
        st.header("1. Prozessdaten")
        
        # Leistungsvorgabe (kann aus vorherigen Reitern kommen oder manuell)
        P_req_input = st.number_input("Erf. Wellenleistung Pumpe [kW]", 
                                      min_value=0.1, value=5.5, step=0.5,
                                      help="Leistung am Betriebspunkt der Pumpe")
        
        T_medium = st.number_input("Medientemperatur [°C]", 
                                   min_value=-20.0, max_value=200.0, value=40.0, step=1.0)

        st.divider()
        st.header("2. Zonen-Definition")
        
        # Atmosphäre: Gas oder Staub
        atmosphere = st.radio("Atmosphäre", ["G (Gas)", "D (Staub)"], index=0)
        
        # Zone Auswahl
        if atmosphere == "G (Gas)":
            zone_select = st.selectbox("Ex-Zone (Gas)", [0, 1, 2], index=2,
                                       help="Zone 0: ständig, Zone 1: gelegentlich, Zone 2: selten [cite: 4]")
        else:
            zone_select = st.selectbox("Ex-Zone (Staub)", [20, 21, 22], index=2,
                                       help="Zone 20: ständig, Zone 21: gelegentlich, Zone 22: selten [cite: 4]")

    with col_res:
        st.markdown("### 📋 ATEX-Konformitätsprüfung")
        
        valid_config = True

        # --- PRÜFUNG 1: Zonen-Machbarkeit  ---
        if atmosphere == "D (Staub)":
            st.error("❌ **Staub-Ex (Atmosphäre D):** Für diese Anforderung haben wir kein passendes Aggregat.")
            st.warning("Laut Vorschrift: 'Für Aufstellung in ... Ex-Atmosphäre D (Staub) haben wir kein passendes Aggregat' ")
            valid_config = False
            
        elif zone_select == 0:
            st.error("❌ **Zone 0:** Für Zone 0 haben wir kein passendes Aggregat.")
            st.warning("Laut Vorschrift: 'Für Aufstellung in Zone 0 ... haben wir kein passendes Aggregat' ")
            valid_config = False
            
        else:
            st.success(f"✅ Zone {zone_select} (Gas) ist machbar.")
            
        # --- PRÜFUNG 2: Temperaturklasse [cite: 15, 16, 17] ---
        if valid_config:
            st.markdown("#### Temperatur-Check")
            
            # Sicherheitsabstand 15K [cite: 16, 17]
            t_margin = 15.0
            
            # Geeignete Motoren filtern
            suitable_motors = []
            
            for m in ATEX_MOTORS:
                # 1. Zonen-Check
                if zone_select in m["zone_suitable"]:
                    # 2. Temperatur-Check: T_surface - 15K >= T_medium
                    if (m["t_max_surface"] - t_margin) >= T_medium:
                        suitable_motors.append(m)
            
            if not suitable_motors:
                st.error(f"❌ Keine Motoren verfügbar für T_medium = {T_medium}°C!")
                st.markdown(f"""
                **Grund:** Der Abstand zwischen Medientemperatur und max. Oberflächentemperatur 
                muss mind. **{t_margin} K** betragen[cite: 17].
                
                Benötigte T-Klasse bei {T_medium}°C: **min. {T_medium + t_margin}°C** zulässig.
                """)
            else:
                # --- PRÜFUNG 3: Leistungsdimensionierung [cite: 13, 14] ---
                st.markdown("#### Leistungsdimensionierung")
                
                # Regel: 15% Reserve [cite: 13, 14]
                P_motor_min = P_req_input * 1.15
                P_iec = motor_iec(P_motor_min) # Helper Funktion nutzen
                
                col1, col2, col3 = st.columns(3)
                col1.metric("P_Pumpe", f"{P_req_input:.2f} kW")
                col2.metric("P_min (+15%)", f"{P_motor_min:.2f} kW", help="Mind. 15% Reserve gefordert ")
                col3.metric("IEC Motorgröße", f"**{P_iec:.2f} kW**")
                
                st.divider()
                st.markdown("### 🔧 Verfügbare ATEX-Motoren")
                
                selection = st.radio("Wählen Sie einen Motortyp:", 
                                     options=suitable_motors,
                                     format_func=lambda x: f"{x['marking']} ({x['id']})")
                
                if selection:
                    # --- HIER IST DER NEUE TEIL FÜR DIE ERKLÄRUNG ---
                    st.info(f"ℹ️ **Warum dieser Motor?**\n\n{selection['description']}")
                    
                    st.success("✅ **Gültige Konfiguration gefunden**")
                    
                    # Detail-Ausgabe
                    with st.expander("Technische Details anzeigen", expanded=True):
                        st.markdown(f"""
                        **Spezifikation:**
                        * **Leistung:** {P_iec:.2f} kW (inkl. Reserve)
                        * **Kennzeichnung:** `{selection['marking']}`
                        * **Zündschutzart:** {selection['id']}
                        * **Max. Oberfläche:** {selection['t_max_surface']}°C ({selection['temp_class']})
                        """)
                        
                        # Prüfung der 15K Regel visuell darstellen
                        delta_t = selection['t_max_surface'] - T_medium
                        if delta_t >= 15.0:
                             st.caption(f"✅ Temperaturabstand: {delta_t:.1f} K (Vorschrift: min. 15 K)")
                        else:
                             # Dies sollte durch den Filter oben eigentlich nicht passieren, 
                             # aber als Fallback zur Sicherheit:
                             st.error(f"❌ Temperaturabstand zu gering: {delta_t:.1f} K")

                    st.caption(f"Bestellbezeichnung: Pumpe + {selection['marking']}")

    # Info-Expander mit Zonen-Definitionen
    with st.expander("ℹ️ Definition der Ex-Zonen [cite: 4]"):
        st.markdown("""
        | Zone (Gas) | Beschreibung | Häufigkeit |
        | :--- | :--- | :--- |
        | **Zone 0** | Ständige, lang andauernde Gefahr | Häufig |
        | **Zone 1** | Gelegentliche Gefahr im Normalbetrieb | Gelegentlich |
        | **Zone 2** | Normalerweise keine Gefahr / nur kurzzeitig | Selten |
        """)

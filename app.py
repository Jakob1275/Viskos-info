import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Konstanten & Physikalische Basics
# =========================
G = 9.80665  # m/s²
P_N_BAR = 1.01325
T_N_K = 273.15 
N0_RPM_DEFAULT = 2900

# =========================
# Pumpendaten & ATEX Datenbank
# =========================
PUMPS = [
    {"id": "P1", "Qw": [0, 15, 30, 45, 60], "Hw": [55, 53, 48, 40, 28],
     "eta": [0.28, 0.52, 0.68, 0.66, 0.52], "Pw": [1.1, 3.9, 5.8, 6.2, 7.3], "n0": 2900},
    {"id": "P2", "Qw": [0, 20, 40, 60, 80], "Hw": [48, 46, 40, 30, 18],
     "eta": [0.30, 0.60, 0.72, 0.68, 0.55], "Pw": [1.5, 4.2, 7.4, 8.5, 9.2], "n0": 2900},
]

MEDIA = {
    "Wasser (20°C)": (998.0, 1.0),
    "Hydrauliköl ISO VG 46 (40°C)": (870.0, 46.0),
    "Glykol 30% (20°C)": (1040.0, 3.5),
}

ATEX_MOTORS = [
    {"id": "Standard Zone 2 (ec)", "marking": "II 3G Ex ec IIC T3 Gc", "zone_suitable": [2],
     "temp_class": "T3", "t_max_surface": 200.0, "category": "3G",
     "description": "Standardlösung für Zone 2."},
    {"id": "Zone 1 (eb)", "marking": "II 2G Ex eb IIC T3 Gb", "zone_suitable": [1, 2],
     "temp_class": "T3", "t_max_surface": 200.0, "category": "2G",
     "description": "Standardlösung für Zone 1 ('Erhöhte Sicherheit')."},
    {"id": "Zone 1 (db eb) T4", "marking": "II 2G Ex db eb IIC T4 Gb", "zone_suitable": [1, 2],
     "temp_class": "T4", "t_max_surface": 135.0, "category": "2G",
     "description": "Zone 1 mit strengeren Temperaturanforderungen (T4)."},
]

# =========================
# Helper Funktionen
# =========================
def interp_clamped(x, xs, ys):
    return np.interp(x, xs, ys)

def clamp(x, a, b):
    return max(a, min(b, x))

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s: return s
    return steps[-1]

# --- Viskositäts-Logik ---
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    Q_gpm = max(Q_m3h, 0.1) * 4.40287
    H_ft = max(H_m, 0.1) * 3.28084
    return 16.5 * (max(nu_cSt, 0.1) ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    if B <= 1.0: return 1.0, 1.0
    CH = clamp(math.exp(-0.165 * (math.log10(B) ** 2.2)), 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = clamp(1.0 - 0.25 * log_B - 0.05 * (log_B ** 2), 0.1, 1.0)
    return CH, Ceta

def get_viscous_head_at_speed(pump, Q_req, n_ratio, nu_cSt):
    Q_equiv = Q_req / n_ratio
    H_water = interp_clamped(Q_equiv, pump["Qw"], pump["Hw"])
    B = compute_B_HI(Q_req, H_water * (n_ratio**2), nu_cSt)
    CH, _ = viscosity_correction_factors(B)
    return H_water * (n_ratio ** 2) * CH

def find_optimal_speed(pump, Q_req, H_req, nu_cSt):
    lo, hi = 0.4, 1.3
    for _ in range(25):
        mid = (lo + hi) / 2
        if get_viscous_head_at_speed(pump, Q_req, mid, nu_cSt) > H_req:
            hi = mid
        else: lo = mid
    return mid

# --- Mehrphasen-Logik ---
def get_solubility_limit(p_abs, T_c):
    # Modell: 6.5 bar @ 20°C ~ 100 cm3N/L
    ref_p, ref_sol = 6.5, 100.0
    return ref_sol * (p_abs / ref_p) * (293.15 / (T_c + 273.15))

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="Pumpen-Expertentool", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "viscosity"

with st.sidebar:
    st.header("📍 Navigation")
    if st.button("Einphasen (Viskosität)", use_container_width=True): st.session_state.page = "viscosity"
    if st.button("Mehrphasen (Löslichkeit)", use_container_width=True): st.session_state.page = "mph"
    if st.button("ATEX Motoren", use_container_width=True): st.session_state.page = "atex"

# ---------------------------------------------------------
# PAGE: VISKOSITÄT
# ---------------------------------------------------------
if st.session_state.page == "viscosity":
    st.title("🔄 Viskosität & Drehzahloptimierung")
    col_in, col_res = st.columns([1, 2])
    
    with col_in:
        st.subheader("⚙️ Eingaben")
        Q_req = st.number_input("Q [m³/h]", 1.0, 100.0, 40.0)
        H_req = st.number_input("H [m]", 1.0, 100.0, 35.0)
        med = st.selectbox("Medium", list(MEDIA.keys()))
        rho, nu = MEDIA[med]
        reserve = st.slider("Motorreserve [%]", 0, 30, 15)
        pump = PUMPS[0]

    n_opt_ratio = find_optimal_speed(pump, Q_req, H_req, nu)
    n_opt_rpm = n_opt_ratio * pump["n0"]
    
    # Energie-Vergleich
    B_nom = compute_B_HI(Q_req, interp_clamped(Q_req, pump["Qw"], pump["Hw"]), nu)
    _, Ceta_nom = viscosity_correction_factors(B_nom)
    eta_nom = interp_clamped(Q_req, pump["Qw"], pump["eta"]) * Ceta_nom
    P_nom = (rho * G * (Q_req/3600) * H_req) / (max(eta_nom, 0.01) * 1000)

    B_opt = compute_B_HI(Q_req, H_req, nu)
    _, Ceta_opt = viscosity_correction_factors(B_opt)
    eta_opt = interp_clamped(Q_req/n_opt_ratio, pump["Qw"], pump["eta"]) * Ceta_opt
    P_opt = (rho * G * (Q_req/3600) * H_req) / (max(eta_opt, 0.01) * 1000)
    
    P_motor = motor_iec(P_opt * (1 + reserve/100))

    with col_res:
        st.subheader("✅ Ergebnis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimale Drehzahl", f"{n_opt_rpm:.0f} RPM")
        c2.metric("IEC-Motor", f"{P_motor:.2f} kW")
        c3.metric("Einsparung", f"{P_nom - P_opt:.2f} kW", f"-{((P_nom-P_opt)/P_nom)*100:.1f}%")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pump["Qw"], pump["Hw"], color="gray", alpha=0.3, label="Wasser (2900 RPM)")
        qs_plot = np.linspace(0, max(pump["Qw"]) * n_opt_ratio, 50)
        hs_plot = [get_viscous_head_at_speed(pump, q, n_opt_ratio, nu) for q in qs_plot]
        ax.plot(qs_plot, hs_plot, color="#1f77b4", linewidth=2, label=f"Viskos bei {n_opt_rpm:.0f} RPM")
        ax.scatter([Q_req], [H_req], color='red', marker='x', s=100, label="Betriebspunkt", zorder=5)
        ax.set_xlabel("Q [m³/h]"); ax.set_ylabel("H [m]"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with st.expander("📘 Rechenweg"):
        st.latex(r"B = 16.5 \cdot \frac{\sqrt{\nu}}{Q_{gpm}^{0.25} \cdot H_{ft}^{0.375}}")
        st.write(f"Aktueller Faktor B: {B_opt:.3f}")
        st.write(f"Wirkungsgrad-Korrektur Ceta: {Ceta_opt:.3f}")

# ---------------------------------------------------------
# PAGE: MEHRPHASEN
# ---------------------------------------------------------
elif st.session_state.page == "mph":
    st.title("⚗️ Mehrphasen-Check")
    col_in, col_res = st.columns([1, 2])
    
    with col_in:
        temp = st.number_input("Temperatur [°C]", 0.0, 100.0, 20.0)
        p_abs = st.number_input("Saugdruck p_s [bar abs]", 1.0, 15.0, 6.7)
        gvf_pct = st.slider("Gasanteil (GVF) [%]", 0, 25, 15)
        
    gas_load = gvf_pct * 10.0
    max_sol = get_solubility_limit(p_abs, temp)
    is_possible = gas_load <= max_sol

    with col_res:
        if is_possible: st.success("✅ Gas vollständig gelöst.")
        else: st.error("❌ NICHT MÖGLICH: Gasanteil überschreitet Löslichkeit!")

        fig, ax = plt.subplots(figsize=(10, 5))
        pressures = np.linspace(0, 14, 100)
        for t_val in [10, 20, 30]:
            sols = [get_solubility_limit(p, t_val) for p in pressures]
            ax.plot(pressures, sols, "--", alpha=0.5, label=f"Löslichkeit {t_val}°C")
        ax.scatter([p_abs], [gas_load], color=("green" if is_possible else "red"), marker='x', s=150, zorder=5)
        ax.axhline(gas_load, color="gray", alpha=0.2, linestyle=":")
        ax.set_xlabel("Druck [bar abs]"); ax.set_ylabel("cm³N/l (10% = 100)"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    with st.expander("📘 Rechenweg Mehrphasen"):
        st.markdown(f"**1. Gaslast:** {gvf_pct}% $\\rightarrow$ **{gas_load} cm³N/L**")
        st.markdown(f"**2. Löslichkeit (Henry):** Max. **{max_sol:.1f} cm³N/L** bei {p_abs} bar")
        st.markdown(f"**3. Ergebnis:** { 'Löslich' if is_possible else 'Freies Gas (nicht möglich)' }")

# ---------------------------------------------------------
# PAGE: ATEX
# ---------------------------------------------------------
elif st.session_state.page == "atex":
    st.title("🛡️ ATEX Motorenauswahl")
    st.info("Wählen Sie die Zone und Anforderungen für den elektrischen Antrieb.")
    
    zone = st.radio("Explosionsschutzzone", ["Zone 2 (Gelegentlich)", "Zone 1 (Häufig)"])
    z_val = 2 if "2" in zone else 1
    
    suitable = [m for m in ATEX_MOTORS if z_val in m["zone_suitable"]]
    
    for m in suitable:
        with st.container():
            col1, col2 = st.columns([1, 3])
            col1.subheader(m["id"])
            col2.markdown(f"**Kennzeichnung:** `{m['marking']}`")
            col2.write(f"Temperaturklasse: {m['temp_class']} | Max. Oberfläche: {m['t_max_surface']}°C")
            col2.write(m["description"])
            st.divider()

st.divider()
st.caption("Pumpen-Tool 2026 | Inklusive ATEX, Viskositäts-Optimierung und Mehrphasen-Löslichkeitscheck.")

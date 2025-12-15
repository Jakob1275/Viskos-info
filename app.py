import math
import streamlit as st
import matplotlib.pyplot as plt
from iapws import IAPWS97

G = 9.80665  # m/s²

# ---------------------------
# 5 typische Pumpenkennlinien (Wasser) – Demo-Daten
# ---------------------------
PUMPS = [
    {"id": "P1 (klein, steil)",
     "Qw": [0, 15, 30, 45, 60],
     "Hw": [55, 53, 48, 40, 28],
     "eta": [0.28, 0.52, 0.68, 0.66, 0.52]},
    {"id": "P2 (mittel, ausgewogen)",
     "Qw": [0, 20, 40, 60, 80],
     "Hw": [48, 46, 40, 30, 18],
     "eta": [0.30, 0.60, 0.72, 0.68, 0.55]},
    {"id": "P3 (höherer Durchfluss)",
     "Qw": [0, 30, 60, 90, 120],
     "Hw": [42, 41, 36, 26, 14],
     "eta": [0.25, 0.55, 0.73, 0.70, 0.58]},
    {"id": "P4 (höhere Förderhöhe)",
     "Qw": [0, 15, 30, 45, 60],
     "Hw": [70, 68, 62, 52, 40],
     "eta": [0.22, 0.48, 0.66, 0.65, 0.50]},
    {"id": "P5 (flacher, effizient im mittleren Bereich)",
     "Qw": [0, 25, 50, 75, 100],
     "Hw": [46, 44, 38, 28, 16],
     "eta": [0.30, 0.62, 0.75, 0.72, 0.60]},
]

# ---------------------------
# Medium-Datenbank (typische Richtwerte)
# ---------------------------
MEDIA = {
    "Wasser (20°C)": (998.0, 1.0),
    "Wasser (60°C)": (983.0, 0.47),
    "Glykol 30% (20°C) (typ.)": (1040.0, 3.5),
    "Hydrauliköl ISO VG 32 (40°C) (typ.)": (860.0, 32.0),
    "Hydrauliköl ISO VG 46 (40°C) (typ.)": (870.0, 46.0),
    "Hydrauliköl ISO VG 68 (40°C) (typ.)": (880.0, 68.0),
}

# ---------------------------
# Helpers
# ---------------------------
def interp_clamped(x, xs, ys):
    """Lineare Interpolation mit Clamping an den Rändern"""
    if len(xs) < 2:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]

def clamp(x, a, b):
    """Wert auf Bereich [a, b] begrenzen"""
    return max(a, min(b, x))

def motor_iec(P_kW):
    """Nächstgrößere IEC-Motornennleistung"""
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# ---------------------------
# KORREKTE Viskositätskorrektur nach Hydraulic Institute
# ---------------------------
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """
    B-Zahl nach Hydraulic Institute Standard
    B = 16.5 * ν^0.5 / (Q^0.25 * H^0.375)
    
    Einheiten: Q in m³/h, H in m, ν in cSt
    """
    Q = max(Q_m3h, 1e-6)
    H = max(H_m, 1e-6)
    nu = max(nu_cSt, 1e-6)
    
    # Umrechnung auf US-Einheiten für HI-Formel
    Q_gpm = Q * 4.40287  # m³/h -> gal/min
    H_ft = H * 3.28084   # m -> ft
    
    B = 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))
    return B

def viscosity_correction_factors(B, nu_cSt):
    """
    Korrekturfaktoren nach Hydraulic Institute (empirische Formeln)
    
    WICHTIG: Q bleibt konstant (CQ = 1.0)!
    Nur H und η werden korrigiert.
    
    Returns: (CH, Ceta)
    """
    # ALLE Medien werden umgerechnet (auch Wasser)
    if B <= 1.0:
        return 1.0, 1.0
    
    # Empirische Formeln (basierend auf HI-Kurven)
    # Diese sind Näherungen - für echte Auslegung HI-Diagramme verwenden!
    
    # Förderhöhen-Korrekturfaktor
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = clamp(CH, 0.3, 1.0)
    
    # Wirkungsgrad-Korrekturfaktor
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = clamp(Ceta, 0.1, 1.0)
    
    return CH, Ceta

def viscous_to_water_point(Q_vis, H_vis, nu_cSt):
    """
    KORREKTE Umrechnung: viskoses Medium -> Wasserkennlinie
    
    Q bleibt konstant! Nur H wird korrigiert.
    """
    B = compute_B_HI(Q_vis, H_vis, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B, nu_cSt)
    
    Q_water = Q_vis  # Q bleibt gleich!
    H_water = H_vis / CH  # H muss höher sein auf Wasserkennlinie
    
    return {
        "B": B,
        "CH": CH,
        "Ceta": Ceta,
        "Q_water": Q_water,
        "H_water": H_water
    }

def water_to_viscous_point(Q_water, H_water, eta_water, nu_cSt):
    """
    KORREKTE Umrechnung: Wasserkennlinie -> viskoses Medium
    
    Q bleibt konstant! H und η werden korrigiert.
    """
    Q_vis = Q_water  # Q bleibt gleich!
    
    # B-Zahl für vorläufigen viskosen Punkt berechnen
    # (Näherung: H_vis ≈ H_water zunächst)
    B = compute_B_HI(Q_vis, H_water, nu_cSt)
    CH, Ceta = viscosity_correction_factors(B, nu_cSt)
    
    H_vis = H_water * CH  # Förderhöhe sinkt bei Viskosität
    eta_vis = eta_water * Ceta  # Wirkungsgrad sinkt bei Viskosität
    
    return Q_vis, H_vis, max(1e-6, eta_vis)

def generate_viscous_curve(pump, nu_cSt):
    """Erzeugt viskose Kennlinie aus Wasserkennlinie"""
    Q_vis = []
    H_vis = []
    eta_vis = []
    
    for Q_w, H_w, eta_w in zip(pump["Qw"], pump["Hw"], pump["eta"]):
        Q_v, H_v, eta_v = water_to_viscous_point(Q_w, H_w, eta_w, nu_cSt)
        Q_vis.append(Q_v)
        H_vis.append(H_v)
        eta_vis.append(eta_v)
    
    return Q_vis, H_vis, eta_vis

# ---------------------------
# Pumpenauswahl
# ---------------------------
def choose_best_pump(pumps, Q_water, H_water, allow_out_of_range=True):
    """
    Wählt beste Pumpe basierend auf Wasserkennlinie
    """
    best = None
    
    for p in pumps:
        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        in_range = (qmin <= Q_water <= qmax)
        
        if not in_range and not allow_out_of_range:
            continue
        
        Q_eval = clamp(Q_water, qmin, qmax)
        penalty = 0.0
        
        if not in_range:
            span = max(qmax - qmin, 1e-9)
            penalty = abs(Q_water - Q_eval) / span * 10.0
        
        H_at = interp_clamped(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp_clamped(Q_eval, p["Qw"], p["eta"])
        errH = abs(H_at - H_water)
        score = errH + penalty
        
        cand = {
            "id": p["id"],
            "pump": p,
            "in_range": in_range,
            "Q_eval": Q_eval,
            "H_at": H_at,
            "eta_at": eta_at,
            "errH": errH,
            "score": score,
        }
        
        if best is None or score < best["score"] - 1e-9:
            best = cand
        elif abs(score - best["score"]) <= 1e-9 and eta_at > best["eta_at"]:
            best = cand
    
    return best

# ---------------------------
# Sättigung (IAPWS-IF97)
# ---------------------------
def sat_temperature_from_pressure(p_bar_abs):
    """Sättigungstemperatur aus Druck (IAPWS-IF97)"""
    p_mpa = p_bar_abs * 0.1
    w = IAPWS97(P=p_mpa, x=0)
    return w.T - 273.15

def sat_pressure_from_temperature(t_c):
    """Sättigungsdruck aus Temperatur (IAPWS-IF97)"""
    w = IAPWS97(T=t_c + 273.15, x=0)
    return w.P * 10.0

def saturation_curve_pT(T_min=0.0, T_max=350.0, n=200):
    """Sättigungslinie Wasser: p_sätt(T)"""
    Ts = [T_min + (T_max - T_min) * i / (n - 1) for i in range(n)]
    ps = []
    for T in Ts:
        try:
            ps.append(sat_pressure_from_temperature(T))
        except:
            ps.append(float("nan"))
    return Ts, ps

# ---------------------------
# Gaslöslichkeit (Henry's Law)
# ---------------------------
# Henry-Konstanten bei 25°C [bar·L/mol]
HENRY_CONSTANTS = {
    "CO2": 29.4,
    "O2": 769.2,
    "N2": 1639.3,
    "CH4": 714.3,
    "H2": 1282.1,
    "H2S": 10.3,
}

# Temperaturabhängigkeit (van't Hoff)
HENRY_TEMP_PARAMS = {
    "CO2": {"A": 29.4, "B": 2400},
    "O2": {"A": 769.2, "B": 1500},
    "N2": {"A": 1639.3, "B": 1300},
    "CH4": {"A": 714.3, "B": 1600},
    "H2": {"A": 1282.1, "B": 500},
    "H2S": {"A": 10.3, "B": 2100},
}

def henry_constant(gas, T_celsius):
    """
    Temperaturabhängige Henry-Konstante nach van't Hoff:
    H(T) = H₀ × exp[B × (1/T - 1/T₀)]
    """
    params = HENRY_TEMP_PARAMS.get(gas)
    if not params:
        return HENRY_CONSTANTS.get(gas, 1000.0)
    
    T_K = T_celsius + 273.15
    T0_K = 298.15  # 25°C
    H0 = params["A"]
    B = params["B"]
    
    H_T = H0 * math.exp(B * (1/T_K - 1/T0_K))
    return H_T

def gas_solubility_henry(gas, p_partial_bar, T_celsius):
    """
    Gaslöslichkeit nach Henry's Law:
    C = p / H(T)
    
    Returns: Konzentration [mol/L]
    """
    H = henry_constant(gas, T_celsius)
    C = p_partial_bar / H
    return C

def max_dissolved_gas_content(gas, p_total_bar, T_celsius, gas_fraction=1.0):
    """
    Maximaler gelöster Gasgehalt bei Sättigung
    
    Args:
        gas: Gastyp
        p_total_bar: Gesamtdruck [bar abs]
        T_celsius: Temperatur [°C]
        gas_fraction: Molenbruch des Gases im Gasgemisch [-]
    
    Returns: dict mit Löslichkeit und Volumenanteil
    """
    p_partial = p_total_bar * gas_fraction
    C_mol_L = gas_solubility_henry(gas, p_partial, T_celsius)
    
    # Umrechnung auf Volumenanteil (bei Standardbedingungen)
    # 1 mol Gas = 22.4 L bei STP
    V_gas_L_per_L = C_mol_L * 22.4
    
    # GVF bei Ausgasung
    GVF = V_gas_L_per_L / (1.0 + V_gas_L_per_L)
    
    return {
        "concentration_mol_L": C_mol_L,
        "volume_ratio": V_gas_L_per_L,
        "GVF_at_degassing": GVF,
        "partial_pressure_bar": p_partial
    }

def solubility_curve(gas, T_celsius, p_range=(0.1, 20.0), n=100):
    """
    Löslichkeitskurve C(p) bei konstanter Temperatur
    """
    ps = [p_range[0] + (p_range[1] - p_range[0]) * i / (n - 1) for i in range(n)]
    Cs = [gas_solubility_henry(gas, p, T_celsius) for p in ps]
    return ps, Cs

# ---------------------------
# Mehrphasenpumpen-Datenbank
# ---------------------------
MPH_PUMPS = [
    {
        "id": "MPH-100 (Twin-Screw)",
        "type": "Schraubenspindelpumpe",
        "Q_max_m3h": 100,
        "p_max_bar": 40,
        "GVF_max": 0.95,
        "Q_base": [0, 20, 40, 60, 80, 100],
        "H_base": [35, 34, 32, 28, 22, 15],
        "eta_base": [0.45, 0.62, 0.70, 0.68, 0.60, 0.50],
    },
    {
        "id": "MPH-200 (Helico-Axial)",
        "type": "Heliko-axial",
        "Q_max_m3h": 200,
        "p_max_bar": 30,
        "GVF_max": 0.85,
        "Q_base": [0, 40, 80, 120, 160, 200],
        "H_base": [28, 27, 25, 22, 17, 10],
        "eta_base": [0.40, 0.58, 0.68, 0.66, 0.58, 0.48],
    },
    {
        "id": "MPH-50 (Compact Twin-Screw)",
        "type": "Schraubenspindelpumpe",
        "Q_max_m3h": 50,
        "p_max_bar": 50,
        "GVF_max": 0.90,
        "Q_base": [0, 10, 20, 30, 40, 50],
        "H_base": [45, 44, 41, 36, 28, 18],
        "eta_base": [0.42, 0.60, 0.68, 0.66, 0.58, 0.48],
    },
]

def choose_mph_pump(Q_req, p_req, GVF_req):
    """
    Wählt geeignete Mehrphasenpumpe basierend auf Anforderungen
    """
    H_req = p_req * 10.197  # bar -> m Wassersäule (näherungsweise)
    
    candidates = []
    for pump in MPH_PUMPS:
        if Q_req > pump["Q_max_m3h"]:
            continue
        if GVF_req > pump["GVF_max"]:
            continue
        
        # Förderhöhe bei Q prüfen
        H_at_Q = interp_clamped(Q_req, pump["Q_base"], pump["H_base"])
        
        if H_at_Q >= H_req * 0.8:  # 80% Sicherheitsfaktor
            score = abs(H_at_Q - H_req) + abs(Q_req - pump["Q_max_m3h"]/2) * 0.1
            candidates.append({
                "pump": pump,
                "H_at_Q": H_at_Q,
                "score": score
            })
    
    if not candidates:
        return None
    
    return min(candidates, key=lambda x: x["score"])

# ---------------------------
# Gas-Derating (vereinfacht)
# ---------------------------
def gas_derating_factor_H(gvf, k=1.4, exp=0.85):
    """Förderhöhen-Derating bei Gasanteil"""
    gvf = clamp(gvf, 0.0, 0.6)
    f = 1.0 - k * (gvf ** exp)
    return clamp(f, 0.2, 1.0)

def gas_derating_factor_eta(gvf, k=2.0, exp=0.9):
    """Wirkungsgrad-Derating bei Gasanteil"""
    gvf = clamp(gvf, 0.0, 0.6)
    f = 1.0 - k * (gvf ** exp)
    return clamp(f, 0.1, 1.0)

def apply_gas_derating_curve(Q, H, eta, gvf):
    """Wendet Gas-Derating auf Kennlinie an"""
    fH = gas_derating_factor_H(gvf)
    feta = gas_derating_factor_eta(gvf)
    Hg = [h * fH for h in H]
    etag = [max(1e-6, e * feta) for e in eta]
    return Q, Hg, etag

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(
    page_title="Pumpenauslegung: Viskos + Mehrphasen",
    layout="wide"
)

st.title("🔧 Pumpenauslegung")
st.caption("Viskositätskorrektur nach Hydraulic Institute + Mehrphasen-Analyse")

if "page" not in st.session_state:
    st.session_state.page = "pump"

# Navigation
with st.sidebar:
    st.header("📍 Navigation")
    col1, col2 = st.columns(2)
    if col1.button("🔄 Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if col2.button("⚗️ Mehrphasen", use_container_width=True):
        st.session_state.page = "mph"
    
    page_names = {"pump": "Pumpen", "mph": "Mehrphasen"}
    st.info(f"**Aktiv:** {page_names.get(st.session_state.page, 'Pumpen')}")

# =========================================================
# PAGE 1: PUMPEN
# =========================================================
if st.session_state.page == "pump":
    st.subheader("🔄 Pumpenauswahl mit Viskositätskorrektur")
    
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben")
        
        Q_vis_req = st.number_input(
            "Qᵥ (Fördervolumenstrom) [m³/h]",
            min_value=0.1, max_value=300.0, value=40.0, step=1.0
        )
        H_vis_req = st.number_input(
            "Hᵥ (Förderhöhe) [m]",
            min_value=0.1, max_value=300.0, value=35.0, step=1.0
        )
        
        mk = st.selectbox("Medium", list(MEDIA.keys()), index=0)
        rho_def, nu_def = MEDIA[mk]
        
        rho = st.number_input("ρ (Dichte) [kg/m³]", min_value=1.0, value=float(rho_def), step=5.0)
        nu = st.number_input("ν (kinematische Viskosität) [cSt]", min_value=0.1, value=float(nu_def), step=0.5)
        
        allow_out = st.checkbox("Auswahl auch außerhalb Kennlinienbereich", value=True)
        reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)
    
    # KORREKTE Berechnung
    conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    B = conv["B"]
    CH = conv["CH"]
    Ceta = conv["Ceta"]
    
    # Info-Box
    if B < 1.0:
        st.info(f"✅ B = {B:.2f} < 1.0 → Geringe Viskositätseffekte")
    else:
        st.warning(f"⚠️ B = {B:.2f} ≥ 1.0 → Viskositätskorrektur erforderlich")
    
    # Ergebnisse Umrechnung
    st.markdown("### 📊 Umrechnung viskos → Wasser")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q_Wasser", f"{Q_water:.2f} m³/h", help="Q bleibt konstant!")
    col2.metric("H_Wasser", f"{H_water:.2f} m", delta=f"+{H_water-H_vis_req:.1f}m")
    col3.metric("B-Zahl", f"{B:.2f}")
    col4.metric("CH / Cη", f"{CH:.3f} / {Ceta:.3f}")
    
    # Pumpenauswahl
    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out)
    
    if best is None:
        st.error("❌ Keine passende Pumpe gefunden!")
        st.stop()
    
    p = best["pump"]
    
    st.divider()
    st.markdown("### ✅ Gewählte Pumpe")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pumpe", best["id"])
    col2.metric("H_Pumpe (Wasser)", f"{best['H_at']:.2f} m")
    col3.metric("Abweichung ΔH", f"{best['errH']:.2f} m")
    
    if not best["in_range"]:
        st.warning(f"⚠️ Q_Wasser = {Q_water:.1f} m³/h liegt außerhalb der Kennlinie ({min(p['Qw'])}...{max(p['Qw'])} m³/h)")
    
    # Leistungsberechnung
    eta_water = best["eta_at"]
    eta_vis = eta_water * Ceta
    
    P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
    P_vis_kW = (P_hyd_W / max(eta_vis, 1e-6)) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))
    
    st.markdown("### ⚡ Leistung & Motor")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("η_Wasser", f"{eta_water:.3f}")
    col2.metric("η_viskos", f"{eta_vis:.3f}", delta=f"{(eta_vis-eta_water):.3f}")
    col3.metric("P_viskos", f"{P_vis_kW:.2f} kW")
    col4.metric(f"Motor (+{reserve_pct}%)", f"{P_motor_kW:.2f} kW")
    
    # Kennlinien generieren
    Q_vis_curve, H_vis_curve, eta_vis_curve = generate_viscous_curve(p, nu)
    
    # Plots
    st.divider()
    st.markdown("### 📈 Kennlinien")
    
    tab1, tab2 = st.tabs(["Q-H Kennlinie", "Q-η Kennlinie"])
    
    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Alle Wasserkennlinien
        for pp in PUMPS:
            alpha = 1.0 if pp["id"] == p["id"] else 0.3
            ax1.plot(pp["Qw"], pp["Hw"], marker="o", linestyle="-", 
                    label=pp["id"], alpha=alpha, linewidth=2 if pp["id"]==p["id"] else 1)
        
        # Viskose Kennlinie der gewählten Pumpe
        ax1.plot(Q_vis_curve, H_vis_curve, marker="s", linestyle="--", 
                linewidth=2.5, color="red", label=f"{p['id']} (viskos)")
        
        # Betriebspunkte
        ax1.scatter([Q_water], [H_water], marker="^", s=150, color="blue", 
                   edgecolors="black", linewidths=2, label="Betriebspunkt (Wasser)", zorder=5)
        ax1.scatter([Q_vis_req], [H_vis_req], marker="x", s=200, color="red", 
                   linewidths=3, label="Betriebspunkt (viskos)", zorder=5)
        
        ax1.set_xlabel("Volumenstrom Q [m³/h]", fontsize=12)
        ax1.set_ylabel("Förderhöhe H [m]", fontsize=12)
        ax1.set_title("Q-H Kennlinien: Wasser vs. viskoses Medium", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9)
        st.pyplot(fig1, clear_figure=True)
    
    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Alle Wasserwirkungsgrade
        for pp in PUMPS:
            alpha = 1.0 if pp["id"] == p["id"] else 0.3
            ax2.plot(pp["Qw"], pp["eta"], marker="o", linestyle="-", 
                    label=pp["id"], alpha=alpha, linewidth=2 if pp["id"]==p["id"] else 1)
        
        # Viskoser Wirkungsgrad
        ax2.plot(Q_vis_curve, eta_vis_curve, marker="s", linestyle="--", 
                linewidth=2.5, color="red", label=f"{p['id']} (viskos)")
        
        # Betriebspunkte
        ax2.scatter([Q_water], [eta_water], marker="^", s=150, color="blue", 
                   edgecolors="black", linewidths=2, label="η (Wasser)", zorder=5)
        ax2.scatter([Q_vis_req], [eta_vis], marker="x", s=200, color="red", 
                   linewidths=3, label="η (viskos)", zorder=5)
        
        ax2.set_xlabel("Volumenstrom Q [m³/h]", fontsize=12)
        ax2.set_ylabel("Wirkungsgrad η [-]", fontsize=12)
        ax2.set_title("Q-η Kennlinien: Wasser vs. viskoses Medium", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9)
        st.pyplot(fig2, clear_figure=True)
    
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
        ```
        B = 16.5 × ν^0.5 / (Q^0.25 × H^0.375)
        B = {B:.2f}
        ```
        
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
# PAGE 2: SÄTTIGUNG
# =========================================================
elif st.session_state.page == "mph":
    st.subheader("💧 Sättigungsanalyse (Mehrphasen)")
    
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Eingaben")

        gas = st.selectbox("Gasmedium", list(HENRY_CONSTANTS.keys()), index=0)

        pump_id = st.selectbox(
            "Mehrphasenpumpe",
            [p["id"] for p in MPH_PUMPS],
            index=0
        )
        pump = next(p for p in MPH_PUMPS if p["id"] == pump_id)

        T_c = st.number_input("Temperatur T [°C]", min_value=-10.0, max_value=200.0, value=20.0, step=1.0)
        p_abs = st.number_input("Druck p_abs [bar]", min_value=0.1, max_value=200.0, value=5.0, step=0.1)

        gvf = st.slider("Gasanteil GVF [-]", 0.0, 0.95, 0.10, 0.01)
        Q_req = st.number_input("Betriebspunkt Q [m³/h]", min_value=0.1, max_value=float(pump["Q_max_m3h"]), value=min(40.0, float(pump["Q_max_m3h"])), step=1.0)

        gas_fraction = st.slider("Molenbruch Gas im Gasgemisch [-]", 0.0, 1.0, 1.0, 0.05)

    # 1) Henry: maximal gelöstes Gas
    sat = max_dissolved_gas_content(gas, p_abs, T_c, gas_fraction=gas_fraction)

    col1, col2, col3 = st.columns(3)
    col1.metric("Henry-Löslichkeit C", f"{sat['concentration_mol_L']:.4f} mol/L")
    col2.metric("Gasvolumen @STP", f"{sat['volume_ratio']:.3f} L/L")
    col3.metric("GVF bei Ausgasung", f"{sat['GVF_at_degassing']:.3f}")

    st.caption(f"Partialdruck: {sat['partial_pressure_bar']:.2f} bar | Henry H(T): {henry_constant(gas, T_c):.1f} bar·L/mol")

    # 2) Löslichkeitskurve C(p)
    ps, Cs = solubility_curve(gas, T_c, p_range=(0.1, max(20.0, p_abs*1.2)), n=120)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ps, Cs, marker=None)
    ax.scatter([p_abs], [sat["concentration_mol_L"]], s=80)
    ax.set_xlabel("Druck p_abs [bar]")
    ax.set_ylabel("Löslichkeit C [mol/L]")
    ax.set_title(f"Löslichkeitskurve nach Henry (Gas: {gas}, T={T_c:.1f}°C)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # 3) MPH-Kennlinie und Gas-Derating
    H_req = p_abs * 10.197  # grob bar->mWS

    H0 = interp_clamped(Q_req, pump["Q_base"], pump["H_base"])
    eta0 = interp_clamped(Q_req, pump["Q_base"], pump["eta_base"])

    Qg, Hg, etag = apply_gas_derating_curve(pump["Q_base"], pump["H_base"], pump["eta_base"], gvf)

    H_g = interp_clamped(Q_req, Qg, Hg)
    eta_g = interp_clamped(Q_req, Qg, etag)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("H_req (≈)", f"{H_req:.1f} m")
    col2.metric("H_base @Q", f"{H0:.1f} m")
    col3.metric("H_gas @Q", f"{H_g:.1f} m")
    col4.metric("η_gas @Q", f"{eta_g:.3f}")

    # Plot Q-H base vs gas
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(pump["Q_base"], pump["H_base"], linewidth=2, label="MPH Basis")
    ax2.plot(Qg, Hg, linewidth=2, linestyle="--", label=f"MPH mit Gas (GVF={gvf:.2f})")
    ax2.scatter([Q_req], [H_req], s=90, marker="^", label="Betriebspunkt (Anforderung)")
    ax2.scatter([Q_req], [H_g], s=90, marker="x", label="Betriebspunkt (Gas-derated)")
    ax2.set_xlabel("Volumenstrom Q [m³/h]")
    ax2.set_ylabel("Förderhöhe H [m]")
    ax2.set_title("Gaskennlinie aus MPH-Kennlinie (Derating)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

        # 4) „Sättigungszustand (Q,p) ablesbar“ – einfache Ampel-Logik
    # (Interpretation: wenn GVF im Betrieb > GVF bei Ausgasung -> Ausgasungsrisiko hoch)
    st.markdown("### ✅ Sättigungs-/Ausgasungs-Check (vereinfachte Logik)")
    if gvf > sat["GVF_at_degassing"]:
        st.warning("⚠️ Eingestellter GVF liegt über dem GVF, der sich bei Ausgasung aus gelöstem Gas ergeben würde → Ausgasung/Blasenbildung plausibel.")
    else:
        st.success("✅ Eingestellter GVF liegt im Bereich des aus Henry-Löslichkeit ableitbaren Zustands (vereinfachte Plausibilität).")
        
        mode = st.radio("Berechnungsmodus", 
                       ["T_sätt aus p_abs", "p_sätt aus T"], 
                       index=0)
    
    t_sat = None
    p_sat = None
    
    if mode == "T_sätt aus p_abs":
        col1, col2 = st.columns(2)
        with col1:
            p_bar_abs = st.number_input("Druck p_abs [bar]", 
                                       min_value=0.01, max_value=1000.0, 
                                       value=1.013, step=0.1)
            t_op = st.number_input("Betriebstemperatur T_op [°C]", 
                                  min_value=-20.0, max_value=800.0, 
                                  value=20.0, step=1.0)
            margin = st.number_input("Sicherheitsabstand ΔT [K]", 
                                    min_value=0.0, max_value=50.0, 
                                    value=5.0, step=0.5)
        
        try:
            t_sat = sat_temperature_from_pressure(p_bar_abs)
            with col2:
                st.metric("T_sätt [°C]", f"{t_sat:.2f}")
                dt = t_sat - t_op
                st.metric("ΔT = T_sätt - T_op [K]", f"{dt:.2f}")
                
                if dt < 0:
                    st.error("❌ T_op > T_sätt → Flash/Sieden!")
                elif dt < margin:
                    st.warning("⚠️ Geringe thermische Reserve")
                else:
                    st.success("✅ Ausreichende Reserve")
        except Exception as e:
            st.error(f"Fehler: {e}")
    
    else:
        col1, col2 = st.columns(2)
        with col1:
            t_c = st.number_input("Temperatur T [°C]", 
                                 min_value=-20.0, max_value=800.0, 
                                 value=100.0, step=1.0)
            p_op = st.number_input("Betriebsdruck p_abs [bar]", 
                                  min_value=0.01, max_value=1000.0, 
                                  value=1.013, step=0.1)

    
# ---------------------------
    # Rechenweg Mehrphasen (FINAL BEREINIGT)
    # ---------------------------
    with st.expander("📘 Rechenweg & Gas-Derating Theorie", expanded=False):
        fH = gas_derating_factor_H(gvf)
        feta = gas_derating_factor_eta(gvf)

        st.markdown(f"""
        ## Gas-Derating (Vereinfachtes Modell) 

        Dieses Modell simuliert die Reduktion von Förderhöhe (H) und Wirkungsgrad (η) einer Kreiselpumpe bei der Förderung eines Mediums mit einem **Gasvolumenanteil (GVF)**.

        ### 1️⃣ Gegeben
        - Erforderlicher Volumenstrom Q: **{Q_req:.2f} m³/h**
        - Gasvolumenanteil (Input) GVF: **{gvf:.3f}**
        - Gewählte Pumpe (Basis, Flüssig): **{pump["id"]}**

        ### 2️⃣ Basiswerte (Flüssig-Kennlinie)
        - H₀ (Basis-Förderhöhe bei Q): **{H0:.2f} m**
        - η₀ (Basis-Wirkungsgrad bei Q): **{eta0:.3f}**

        ### 3️⃣ Derating-Faktoren berechnen
        Die Derating-Faktoren $F_H$ und $F_{{\\eta}}$ werden basierend auf dem GVF berechnet (hier nach empirischen Formeln, z.B. Samoilov-Ansatz):
        
        * **Förderhöhen-Faktor ($F_H$):**
            $$F_H = 1.0 - 1.4 \\cdot (\\text{{GVF}}^{{0.85}}) = {fH:.3f}$$
            Derating: **{(1-fH)*100:.1f} %**

        * **Wirkungsgrad-Faktor ($F_{{\\eta}}$):**
            $$F_{{\\eta}} = 1.0 - 2.0 \\cdot (\\text{{GVF}}^{{0.9}}) = {feta:.3f}$$
            Derating: **{(1-feta)*100:.1f} %**
        
        ### 4️⃣ Berechneter Betriebspunkt (mit Gas)
        Der Volumenstrom $$Q$$ wird nicht korrigiert, da die Pumpe das **Gesamtvolumen** fördert. Nur $$H$$ (Förderhöhe) und $$\\eta$$ (Wirkungsgrad) werden korrigiert:
        
        ```
        Q_Gas = Q_req = {Q_req:.2f} m³/h
        H_Gas = H₀ × F_H = {H0:.2f} × {fH:.3f} = {H_g:.2f} m
        η_Gas = η₀ × F_η = {eta0:.3f} × {feta:.3f} = {eta_g:.3f}
        ```
        ---
        ## 📚 Henry's Law (Gelöstes Gas)
        
        Der im oberen Bereich berechnete gelöste Gasgehalt zeigt, wie viel Gas bei **{p_abs:.1f} bar** und **{T_c:.1f}°C** maximal im Medium gelöst sein kann (relevant für die Risikoabschätzung):
        
        - Löslichkeit C: **{sat['concentration_mol_L']:.4f} mol/L**
        - Maximaler GVF bei vollständiger Ausgasung: **{sat['GVF_at_degassing']:.3f}**
        """)
    
    # 4) „Sättigungszustand (Q,p) ablesbar“ – einfache Ampel-Logik
    # (Interpretation: wenn GVF im Betrieb > GVF bei Ausgasung -> Ausgasungsrisiko hoch)
    st.markdown("### ✅ Sättigungs-/Ausgasungs-Check (vereinfachte Logik)")
    if gvf > sat["GVF_at_degassing"]:
        st.warning("⚠️ Eingestellter GVF liegt über dem GVF, der sich bei Ausgasung aus gelöstem Gas ergeben würde → Ausgasung/Blasenbildung plausibel.")
    else:
        st.success("✅ Eingestellter GVF liegt im Bereich des aus Henry-Löslichkeit ableitbaren Zustands (vereinfachte Plausibilität).")
        
        mode = st.radio("Berechnungsmodus", 
                       ["T_sätt aus p_abs", "p_sätt aus T"], 
                       index=0)
    
    t_sat = None
    p_sat = None
    
    if mode == "T_sätt aus p_abs":
        col1, col2 = st.columns(2)
        with col1:
            p_bar_abs = st.number_input("Druck p_abs [bar]", 
                                       min_value=0.01, max_value=1000.0, 
                                       value=1.013, step=0.1)
            t_op = st.number_input("Betriebstemperatur T_op [°C]", 
                                  min_value=-20.0, max_value=800.0, 
                                  value=20.0, step=1.0)
            margin = st.number_input("Sicherheitsabstand ΔT [K]", 
                                    min_value=0.0, max_value=50.0, 
                                    value=5.0, step=0.5)
        
        try:
            t_sat = sat_temperature_from_pressure(p_bar_abs)
            with col2:
                st.metric("T_sätt [°C]", f"{t_sat:.2f}")
                dt = t_sat - t_op
                st.metric("ΔT = T_sätt - T_op [K]", f"{dt:.2f}")
                
                if dt < 0:
                    st.error("❌ T_op > T_sätt → Flash/Sieden!")
                elif dt < margin:
                    st.warning("⚠️ Geringe thermische Reserve")
                else:
                    st.success("✅ Ausreichende Reserve")
        except Exception as e:
            st.error(f"Fehler: {e}")
    
    else:
        col1, col2 = st.columns(2)
        with col1:
            t_c = st.number_input("Temperatur T [°C]", 
                                 min_value=-20.0, max_value=800.0, 
                                 value=100.0, step=1.0)
            p_op = st.number_input("Betriebsdruck p_abs [bar]", 
                                  min_value=0.01, max_value=1000.0, 
                                  value=1.013, step=0.1)

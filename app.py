import math
import streamlit as st
import matplotlib.pyplot as plt
from iapws import IAPWS97

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
    {"id": "MPH-100 (Twin-Screw)", "type": "Schraubenspindel", "Q_max_m3h": 100,
     "p_max_bar": 40, "GVF_max": 0.95, "Q_base": [0, 20, 40, 60, 80, 100],
     "H_base": [35, 34, 32, 28, 22, 15], "eta_base": [0.45, 0.62, 0.70, 0.68, 0.60, 0.50]},
    {"id": "MPH-200 (Helico-Axial)", "type": "Heliko-axial", "Q_max_m3h": 200,
     "p_max_bar": 30, "GVF_max": 0.85, "Q_base": [0, 40, 80, 120, 160, 200],
     "H_base": [28, 27, 25, 22, 17, 10], "eta_base": [0.40, 0.58, 0.68, 0.66, 0.58, 0.48]},
    {"id": "MPH-50 (Compact)", "type": "Schraubenspindel", "Q_max_m3h": 50,
     "p_max_bar": 50, "GVF_max": 0.90, "Q_base": [0, 10, 20, 30, 40, 50],
     "H_base": [45, 44, 41, 36, 28, 18], "eta_base": [0.42, 0.60, 0.68, 0.66, 0.58, 0.48]},
]

HENRY_CONSTANTS = {"CO2": 29.4, "O2": 769.2, "N2": 1639.3, "CH4": 714.3, "H2": 1282.1, "H2S": 10.3}
HENRY_TEMP_PARAMS = {
    "CO2": {"A": 29.4, "B": 2400}, "O2": {"A": 769.2, "B": 1500},
    "N2": {"A": 1639.3, "B": 1300}, "CH4": {"A": 714.3, "B": 1600},
    "H2": {"A": 1282.1, "B": 500}, "H2S": {"A": 10.3, "B": 2100},
}

# Helper Functions
def interp_clamped(x, xs, ys):
    if len(xs) < 2: return ys[0]
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            return ys[i-1] + (ys[i] - ys[i-1]) * (x - xs[i-1]) / (xs[i] - xs[i-1])
    return ys[-1]

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

# Sättigung
def sat_temperature_from_pressure(p_bar_abs):
    w = IAPWS97(P=p_bar_abs * 0.1, x=0)
    return w.T - 273.15

def sat_pressure_from_temperature(t_c):
    w = IAPWS97(T=t_c + 273.15, x=0)
    return w.P * 10.0

def saturation_curve_pT(T_min=0.0, T_max=350.0, n=200):
    Ts = [T_min + (T_max - T_min) * i / (n - 1) for i in range(n)]
    ps = []
    for T in Ts:
        try: ps.append(sat_pressure_from_temperature(T))
        except: ps.append(float("nan"))
    return Ts, ps

# Henry's Law
def henry_constant(gas, T_celsius):
    params = HENRY_TEMP_PARAMS.get(gas)
    if not params: return HENRY_CONSTANTS.get(gas, 1000.0)
    T_K, T0_K = T_celsius + 273.15, 298.15
    return params["A"] * math.exp(params["B"] * (1/T_K - 1/T0_K))

def gas_solubility_henry(gas, p_partial_bar, T_celsius):
    return p_partial_bar / henry_constant(gas, T_celsius)

def max_dissolved_gas_content(gas, p_total_bar, T_celsius, gas_fraction=1.0):
    p_partial = p_total_bar * gas_fraction
    C_mol_L = gas_solubility_henry(gas, p_partial, T_celsius)
    V_gas_L_per_L = C_mol_L * 22.4
    GVF = V_gas_L_per_L / (1.0 + V_gas_L_per_L)
    return {"concentration_mol_L": C_mol_L, "volume_ratio": V_gas_L_per_L,
            "GVF_at_degassing": GVF, "partial_pressure_bar": p_partial}

def solubility_curve(gas, T_celsius, p_range=(0.1, 20.0), n=100):
    ps = [p_range[0] + (p_range[1] - p_range[0]) * i / (n - 1) for i in range(n)]
    Cs = [gas_solubility_henry(gas, p, T_celsius) for p in ps]
    return ps, Cs

# Gas-Derating
def gas_derating_factor_H(gvf): return clamp(1.0 - 1.4 * (clamp(gvf, 0, 0.6) ** 0.85), 0.2, 1.0)
def gas_derating_factor_eta(gvf): return clamp(1.0 - 2.0 * (clamp(gvf, 0, 0.6) ** 0.9), 0.1, 1.0)

def apply_gas_derating_curve(Q, H, eta, gvf):
    fH, feta = gas_derating_factor_H(gvf), gas_derating_factor_eta(gvf)
    return Q, [h * fH for h in H], [max(1e-6, e * feta) for e in eta]

def choose_mph_pump(Q_req, p_req, GVF_req):
    H_req = p_req * 10.197
    candidates = []
    for pump in MPH_PUMPS:
        if Q_req > pump["Q_max_m3h"] or GVF_req > pump["GVF_max"]: continue
        H_at_Q = interp_clamped(Q_req, pump["Q_base"], pump["H_base"])
        if H_at_Q >= H_req * 0.8:
            score = abs(H_at_Q - H_req) + abs(Q_req - pump["Q_max_m3h"]/2) * 0.1
            candidates.append({"pump": pump, "H_at_Q": H_at_Q, "score": score})
    return min(candidates, key=lambda x: x["score"]) if candidates else None

# Streamlit App
st.set_page_config(page_title="Pumpenauslegung", layout="wide")
st.title("🔧 Pumpenauslegungstool")

if "page" not in st.session_state:
    st.session_state.page = "pump"

with st.sidebar:
    st.header("📍 Navigation")
    col1, col2 = st.columns(2)
    if col1.button("🔄 Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if col2.button("⚗️ Mehrphasen", use_container_width=True):
        st.session_state.page = "mph"
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
    
    best = choose_best_pump(PUMPS, Q_water, H_water, allow_out)
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("**H bei Q**", f"**{best['H_at']:.2f} m**")
    col2.metric("**η_viskos**", f"**{eta_vis:.3f}**")
    col3.metric("**P_Welle**", f"**{P_vis_kW:.2f} kW**")
    col4.metric("**Motor**", f"**{P_motor_kW:.2f} kW**")
    col5.metric("**Reserve**", f"**{reserve_pct}%**")
    
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
    

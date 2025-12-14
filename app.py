import math
import streamlit as st
import matplotlib.pyplot as plt
from iapws import IAPWS97  # IAPWS-IF97 steam tables

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
    return max(a, min(b, x))

def motor_iec(P_kW):
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# ---------------------------
# Viskositätslogik (mit Wasser-Bypass)
# ---------------------------
def compute_B(Q_vis_m3h, H_vis_m, nu_cSt):
    Q = max(Q_vis_m3h, 1e-12)
    H = max(H_vis_m, 1e-12)
    nu = max(nu_cSt, 1e-12)
    return 280.0 * (nu ** 0.5) / ((Q ** 0.25) * (H ** 0.125))

def viscosity_factors_from_B(B, nu_cSt, nu_water_threshold=1.5):
    # Wasser-/dünnflüssig-Bypass: Referenzkennlinie ist Wasser -> keine Korrektur
    if nu_cSt <= nu_water_threshold:
        return 1.0, 1.0, 1.0

    if B <= 1.0:
        return 1.0, 1.0, 1.0

    Cq = math.exp(-0.165 * (math.log10(B) ** 3.15))
    Cq = clamp(Cq, 0.0, 1.0)
    Ch = Cq

    Beff = clamp(B, 1.0000001, 40.0)
    Ceta = Beff ** (-(0.0547 * (Beff ** 0.69)))
    Ceta = clamp(Ceta, 0.0, 1.0)

    return Cq, Ch, Ceta

def viscous_to_water_equivalent(Qv, Hv, nu_cSt):
    B = compute_B(Qv, Hv, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)
    if Cq <= 1e-12 or Ch <= 1e-12:
        Cq, Ch, Ceta = 1.0, 1.0, 1.0
    return {"B": B, "Cq": Cq, "Ch": Ch, "Ceta": Ceta, "Qw": Qv / Cq, "Hw": Hv / Ch}

def water_point_to_viscous(Qw, Hw, eta_w, nu_cSt, max_iter=60, tol=1e-10):
    if nu_cSt <= 1.5:
        return Qw, Hw, max(1e-6, eta_w)

    Qv = max(Qw, 1e-12)
    Hv = max(Hw, 1e-12)

    for _ in range(max_iter):
        B = compute_B(Qv, Hv, nu_cSt)
        Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)

        Qv_new = max(1e-12, Cq * Qw)
        Hv_new = max(1e-12, Ch * Hw)

        dq = abs(Qv_new - Qv) / max(Qv, 1e-12)
        dh = abs(Hv_new - Hv) / max(Hv, 1e-12)
        Qv, Hv = Qv_new, Hv_new

        if max(dq, dh) < tol:
            break

    B = compute_B(Qv, Hv, nu_cSt)
    _, _, Ceta = viscosity_factors_from_B(B, nu_cSt)
    eta_v = max(1e-6, Ceta * eta_w)
    return Qv, Hv, eta_v

# ---------------------------
# Pumpenauswahl (robust)
# ---------------------------
def choose_best_pump_robust(pumps, Qw_target, Hw_target, allow_out_of_range=True):
    best = None
    for p in pumps:
        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        in_range = (qmin <= Qw_target <= qmax)

        if not in_range and not allow_out_of_range:
            continue

        Q_eval = Qw_target
        penalty = 0.0
        if not in_range:
            Q_eval = clamp(Qw_target, qmin, qmax)
            span = max(qmax - qmin, 1e-9)
            penalty = abs(Qw_target - Q_eval) / span * 10.0  # "virtuelle" Strafe

        H_at = interp_clamped(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp_clamped(Q_eval, p["Qw"], p["eta"])
        errH = abs(H_at - Hw_target)
        score = errH + penalty

        cand = {
            "id": p["id"], "pump": p, "in_range": in_range,
            "Q_eval": Q_eval, "penalty": penalty,
            "H_at": H_at, "eta_at": eta_at, "errH": errH, "score": score,
        }

        if best is None:
            best = cand
        else:
            if cand["score"] < best["score"] - 1e-9:
                best = cand
            elif abs(cand["score"] - best["score"]) <= 1e-9 and cand["eta_at"] > best["eta_at"]:
                best = cand

    return best

# ---------------------------
# Sättigung (IAPWS-IF97)
# ---------------------------
def sat_temperature_from_pressure(p_bar_abs: float) -> float:
    p_mpa = p_bar_abs * 0.1
    w = IAPWS97(P=p_mpa, x=0)  # saturated liquid
    return w.T - 273.15

def sat_pressure_from_temperature(t_c: float) -> float:
    w = IAPWS97(T=t_c + 273.15, x=0)
    return w.P * 10.0  # MPa -> bar

# ---------------------------
# App Layout + Navigation per Buttons
# ---------------------------
st.set_page_config(page_title="Mehrphasen-Auslegung: Pumpe + Sättigung", layout="centered")
st.title("Auslegung: viskos / Kennlinien + Sättigung (Mehrphasen)")

if "page" not in st.session_state:
    st.session_state.page = "pump"

with st.sidebar:
    st.header("Navigation")
    colA, colB = st.columns(2)
    if colA.button("Pumpen", use_container_width=True):
        st.session_state.page = "pump"
    if colB.button("Sättigung", use_container_width=True):
        st.session_state.page = "sat"

    st.caption(f"Aktiver Bereich: **{ 'Pumpen' if st.session_state.page=='pump' else 'Sättigung' }**")

# =========================================================
# PAGE 1: PUMPEN
# =========================================================
if st.session_state.page == "pump":
    st.subheader("Pumpenauswahl & Kennlinien (Wasser ↔ viskos)")

    with st.sidebar:
        st.divider()
        st.subheader("Eingaben (Pumpen)")
        Qv_req = st.number_input("Qν (gefordert) [m³/h]", min_value=0.1, max_value=300.0, value=40.0, step=1.0)
        Hv_req = st.number_input("Hν (gefordert) [m]", min_value=0.1, max_value=300.0, value=35.0, step=1.0)

        mk = st.selectbox("Medium", list(MEDIA.keys()), index=0)
        rho_def, nu_def = MEDIA[mk]
        rho = st.number_input("ρ [kg/m³]", min_value=1.0, value=float(rho_def), step=5.0)
        nu = st.number_input("ν [cSt]", min_value=0.1, value=float(nu_def), step=0.5)

        allow_out = st.checkbox("Best fit auch wenn Qw außerhalb liegt", value=True)
        reserve_pct = st.slider("Motorreserve [%]", 0, 30, 15)

    conv = viscous_to_water_equivalent(Qv_req, Hv_req, nu_cSt=nu)
    Qw_eq, Hw_eq = conv["Qw"], conv["Hw"]
    B_req, Cq_req, Ch_req, Ceta_req = conv["B"], conv["Cq"], conv["Ch"], conv["Ceta"]

    if nu <= 1.5:
        st.info("Medium ~ Wasser: Korrekturfaktoren = 1 → Qw=Qν und Hw=Hν.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qw [m³/h]", f"{Qw_eq:.2f}")
    c2.metric("Hw [m]", f"{Hw_eq:.2f}")
    c3.metric("Cq / Ch", f"{Cq_req:.3f} / {Ch_req:.3f}")
    c4.metric("B [-]", f"{B_req:.2f}")

    best = choose_best_pump_robust(PUMPS, Qw_target=Qw_eq, Hw_target=Hw_eq, allow_out_of_range=allow_out)
    if best is None:
        st.error("Keine Pumpe konnte bewertet werden.")
        st.stop()

    p = best["pump"]

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Ausgewählte Pumpe", best["id"])
    c2.metric("H_pump(Q) [m]", f"{best['H_at']:.2f}")
    c3.metric("|ΔH| [m]", f"{best['errH']:.2f}")
    st.write(f"ηw(Q) ≈ **{best['eta_at']:.3f}**")

    if not best["in_range"]:
        st.warning(
            f"Qw={Qw_eq:.2f} liegt außerhalb der Kennlinie (Range: {min(p['Qw'])}…{max(p['Qw'])}). "
            "Auswertung am Rand + Strafbewertung."
        )

    # Leistung / Motor am viskosen Betriebspunkt
    eta_v_req = max(1e-6, Ceta_req * best["eta_at"])
    P_hyd_W = float(rho) * G * (float(Qv_req) / 3600.0) * float(Hv_req)
    P_vis_kW = (P_hyd_W / eta_v_req) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("ην [-]", f"{eta_v_req:.3f}")
    c2.metric("Pν [kW]", f"{P_vis_kW:.2f}")
    c3.metric(f"Motor +{reserve_pct}% [kW]", f"{P_motor_kW:.2f}")

    # viskose Kennlinie der ausgewählten Pumpe erzeugen
    Qv_curve, Hv_curve, eta_v_curve = [], [], []
    for Qw_i, Hw_i, eta_w_i in zip(p["Qw"], p["Hw"], p["eta"]):
        Qv_i, Hv_i, eta_v_i = water_point_to_viscous(Qw_i, Hw_i, eta_w_i, nu_cSt=nu)
        Qv_curve.append(Qv_i); Hv_curve.append(Hv_i); eta_v_curve.append(eta_v_i)

    # Marker (clamp)
    Qw_plot = clamp(Qw_eq, min(p["Qw"]), max(p["Qw"]))
    H_w_curve_at_Qw = interp_clamped(Qw_plot, p["Qw"], p["Hw"])
    eta_w_curve_at_Qw = interp_clamped(Qw_plot, p["Qw"], p["eta"])
    Qv_plot = clamp(Qv_req, min(Qv_curve), max(Qv_curve))
    H_v_curve_at_Qv = interp_clamped(Qv_plot, Qv_curve, Hv_curve)
    eta_v_curve_at_Qv = interp_clamped(Qv_plot, Qv_curve, eta_v_curve)

    st.divider()
    st.subheader("Kennlinien (Q-H und Q-η)")

    fig1, ax1 = plt.subplots()
    for pp in PUMPS:
        ax1.plot(pp["Qw"], pp["Hw"], marker="o", linestyle="-", label=pp["id"])
    ax1.plot(Qv_curve, Hv_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos)")

    ax1.scatter([Qw_eq], [Hw_eq], marker="^", s=70, label="Wasseräquivalent (Qw,Hw)")
    ax1.scatter([Qv_req], [Hv_req], marker="x", s=80, label="Arbeitspunkt viskos (Qν,Hν)")
    ax1.scatter([Qw_plot], [H_w_curve_at_Qw], marker="s", s=55, label="H_w auf Auswahlkennlinie")
    ax1.scatter([Qv_plot], [H_v_curve_at_Qv], marker="s", s=55, label="H_ν auf viskoser Kennlinie")

    ax1.set_xlabel("Q [m³/h]"); ax1.set_ylabel("H [m]")
    ax1.set_title("Q-H: Wasserkennlinien + viskose Kennlinie der Auswahl")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots()
    for pp in PUMPS:
        ax2.plot(pp["Qw"], pp["eta"], marker="o", linestyle="-", label=pp["id"])
    ax2.plot(Qv_curve, eta_v_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos η)")

    ax2.scatter([Qw_plot], [eta_w_curve_at_Qw], marker="x", s=80, label="ηw auf Auswahlkennlinie")
    ax2.scatter([Qv_req], [eta_v_req], marker="^", s=70, label="ην am Arbeitspunkt")
    ax2.scatter([Qv_plot], [eta_v_curve_at_Qv], marker="s", s=55, label="ην auf viskoser Kennlinie")

    ax2.set_xlabel("Q [m³/h]"); ax2.set_ylabel("η [-]")
    ax2.set_title("Q-η: Wasserkennlinien + viskose η-Kennlinie der Auswahl")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    # ---------------------------
    # Rechenweg (unten, ausführlich)
    # ---------------------------
    st.divider()
    st.subheader("Rechenweg & Normbezug (Pumpen/viskos)")

st.markdown(r"""
**Ziel:** Aus einem geforderten Betriebspunkt im viskosen Medium $(Q_\nu, H_\nu)$ wird auf Basis einer **Wasser-Referenzkennlinie** eine geeignete Pumpe ausgewählt und der Betriebspunkt auf das viskose Medium zurückgeführt (inkl. Wirkungsgrad- und Leistungsabschätzung).

---

### A) Referenzbasis und Gültigkeitsrahmen
- **Herstellerkennlinien** werden in der Regel für **Wasser** ermittelt und angegeben. Normative Prüf- und Abnahmebedingungen für Kreiselpumpen beziehen sich typischerweise auf Wasser bzw. wasserähnliche Prüfmedien (z. B. DIN EN ISO 9906 als Prüf-/Abnahmereferenz).
- **Folgerung:** Viskositätskorrekturen sind grundsätzlich eine **Umrechnung von der Wasserkennlinie** auf das reale (viskose) Fördermedium – nicht umgekehrt „neue“ Kennlinienmessung.

---

### B) Schritt 1 — Umrechnung viskos $\rightarrow$ Wasseräquivalent
1. **Eingaben** (Betriebspunkt im Fördermedium):  

   $$Q_\nu\;[\mathrm{m^3/h}],\quad H_\nu\;[\mathrm{m}],\quad \nu\;[\mathrm{cSt}],\quad \rho\;[\mathrm{kg/m^3}]$$

2. **Kennzahl $B$** (empirische Kennzahl der Viskositätskorrektur; in dieser App wie implementiert):  

   $$B = 280 \cdot \frac{\nu^{0.5}}{Q_\nu^{0.25}\cdot H_\nu^{0.125}}$$

3. **Korrekturfaktoren** aus $B$:  

   $$C_Q,\; C_H,\; C_\eta$$

   - Für „wasserähnliche“ Medien wird in der Praxis häufig **keine Korrektur** angesetzt. In dieser App gilt daher als Schutzregel:

   $$\nu \le 1.5\;\mathrm{cSt}\;\Rightarrow\; C_Q=C_H=C_\eta = 1$$

4. **Wasseräquivalentpunkt** (Rückrechnung auf Wasser):  

   $$Q_w = \frac{Q_\nu}{C_Q},\qquad H_w = \frac{H_\nu}{C_H}$$

**Interpretation:** $(Q_w, H_w)$ ist der **vergleichbare Punkt** auf der Wasserkennlinie, der zur Auswahl (und Plausibilisierung) herangezogen wird.

---

### C) Schritt 2 — Pumpenauswahl über Wasserkennlinien
- Für jede Pumpenkennlinie wird die Förderhöhe am Wasseräquivalentpunkt bestimmt (Interpolation):

  $$H_{\mathrm{pump}}(Q_w)$$

- Abweichung:

  $$\Delta H = \left|H_{\mathrm{pump}}(Q_w) - H_w\right|$$

- **Auswahlregel (App):** „beste“ Pumpe = kleinste $\Delta H$; bei Gleichstand wird die Pumpe mit höherem $\eta_w(Q_w)$ bevorzugt.
- Liegt $Q_w$ außerhalb des Kennlinienbereichs, kann optional ein **Best-Fit am Rand** erfolgen (Clamp + Strafterm), um eine robuste Auswahl für Demonstrationszwecke zu ermöglichen. Für reale Auslegung sollte in diesem Fall eine **passende Kennlinie/Trim/Drehzahl** oder eine andere Baugröße herangezogen werden.

---

### D) Schritt 3 — Wirkungsgrad- und Leistungsabschätzung im viskosen Medium
1. Wasserwirkungsgrad am Auswahlpunkt: $\eta_w(Q_w)$

2. Viskoser Wirkungsgrad (Korrektur):  

   $$\eta_\nu = C_\eta \cdot \eta_w(Q_w)$$

3. Hydraulische Leistung (mit $Q_\nu$ in $\mathrm{m^3/h}$):  

   $$P_{\mathrm{hyd}} = \rho\, g \cdot \left(\frac{Q_\nu}{3600}\right)\cdot H_\nu$$

4. Wellenleistung (vereinfachte Abschätzung):  

   $$P_\nu = \frac{P_{\mathrm{hyd}}}{\eta_\nu}$$

5. Motorauswahl (Reserve + nächstgrößere IEC-Stufe):  

   $$P_{\mathrm{Motor}} = P_\nu \cdot \left(1+\mathrm{Reserve}\right)$$

---

### E) Norm-/Standardbezug und Grenzen
- **Wasser-Referenz und Prüfbezug:** Pumpenkennlinien/Abnahmemessungen werden normativ üblicherweise auf Wasser bzw. wasserähnliche Prüfbedingungen referenziert (z. B. DIN EN ISO 9906).  
- **Viskositätskorrekturen:** Vorgehensweisen zur Korrektur von Q-H- und $\eta$-Kennlinien für viskose (typisch Newtonsche) Flüssigkeiten sind in einschlägigen Leitfäden/Standards beschrieben (z. B. ISO/TR 17766 und/oder Hydraulic Institute Methoden).  
- **Gültigkeit:** Die Korrektur ist eine **Näherung**; Abweichungen sind zu erwarten bei stark nicht-newtonschen Medien, Gasanteilen/Mehrphasenströmung, sehr kleinen/ sehr großen spezifischen Drehzahlen, oder wenn der Hersteller bereits **viskose Kennlinien** bereitstellt (dann nicht doppelt korrigieren).
""")

# =========================================================
# PAGE 2: SÄTTIGUNG
# =========================================================
else:
    st.subheader("Sättigungskalkulator Wasser unter Druck (Mehrphasen)")

    with st.sidebar:
        st.divider()
        st.subheader("Eingaben (Sättigung)")
        mode = st.radio("Berechnung", ["T_sätt aus p_abs", "p_sätt aus T"], index=0)

    if mode == "T_sätt aus p_abs":
        col1, col2 = st.columns(2)
        with col1:
            p_bar_abs = st.number_input("Absolutdruck p_abs [bar]", min_value=0.01, max_value=1000.0, value=1.013, step=0.1)
            t_op = st.number_input("Betriebstemperatur T_op [°C]", min_value=-20.0, max_value=800.0, value=20.0, step=1.0)
            margin = st.number_input("Sicherheitsabstand ΔT [K]", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

        try:
            t_sat = sat_temperature_from_pressure(float(p_bar_abs))
            with col2:
                st.metric("T_sätt [°C]", f"{t_sat:.2f}")
                dt = t_sat - float(t_op)
                st.metric("ΔT = T_sätt - T_op [K]", f"{dt:.2f}")

                if dt < 0:
                    st.error("T_op > T_sätt → Flash/Boiling sehr wahrscheinlich (Mehrphasenbildung).")
                elif dt < margin:
                    st.warning("T_op nahe T_sätt → geringe Reserve, Mehrphasen-/Kavitationsrisiko steigt.")
                else:
                    st.success("Ausreichende thermische Reserve zur Sättigung (bezogen auf p_abs).")
        except Exception as e:
            st.error(f"IF97-Berechnung fehlgeschlagen: {e}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            t_c = st.number_input("Temperatur T [°C]", min_value=-20.0, max_value=800.0, value=100.0, step=1.0)
            p_op = st.number_input("Betriebsdruck p_abs [bar]", min_value=0.01, max_value=1000.0, value=1.013, step=0.1)
            margin_p = st.number_input("Sicherheitsabstand Δp [bar]", min_value=0.0, max_value=50.0, value=0.2, step=0.05)

        try:
            p_sat = sat_pressure_from_temperature(float(t_c))
            with col2:
                st.metric("p_sätt [bar abs]", f"{p_sat:.3f}")
                dp = float(p_op) - p_sat
                st.metric("Δp = p_op - p_sätt [bar]", f"{dp:.3f}")

                if dp < 0:
                    st.error("p_op < p_sätt → Flash/Boiling sehr wahrscheinlich (Mehrphasenbildung).")
                elif dp < margin_p:
                    st.warning("p_op nahe p_sätt → geringe Druckreserve, Mehrphasen-/Kavitationsrisiko steigt.")
                else:
                    st.success("Ausreichende Druckreserve zur Sättigung.")
        except Exception as e:
            st.error(f"IF97-Berechnung fehlgeschlagen: {e}")

    # ---------------------------
    # Rechenweg (unten, ausführlich)
    # ---------------------------
    st.divider()
    st.subheader("Rechenweg & Normbezug (Sättigung / Mehrphasen)")

    st.markdown(
        """
**Ziel:** Abschätzen, ob bei gegebenem Druck/Temperatur Wasser (oder Wasseranteil) in den **Sättigungsbereich** kommt → Risiko von Flash/Boiling/Mehrphasenbildung.

### A) Physikalische Grundlage
- In der Dampf-Flüssig-Gleichgewichts-Linie gilt:
  - **T_sätt = f(p_abs)** oder äquivalent **p_sätt = f(T)**
- Mehrphasenbildung (Flash) wird wahrscheinlich, wenn:
  - **T_op > T_sätt(p_abs)** oder
  - **p_op < p_sätt(T)**

### B) Berechnung in der App (IAPWS-IF97)
Die App nutzt die industrielle Wasserdampf-Formulierung **IAPWS-IF97** (Region 4 für Sättigungslinie):
- **T_sätt aus p_abs:** Aus absolutem Druck wird die Sättigungstemperatur berechnet.
- **p_sätt aus T:** Aus Temperatur wird der Sättigungsdruck berechnet.

### C) Sicherheitsabstände
- Thermische Reserve:
  - **ΔT = T_sätt − T_op** (K)
- Druckreserve:
  - **Δp = p_op − p_sätt** (bar)

### D) Warum das für Mehrphasenpumpen wichtig ist
- In Mehrphasen-/Kavitations-Szenarien ist besonders kritisch:
  - **Saugseite / Laufradeintritt** (lokales Druckminimum)
  - lokale Druckabfälle (Eintritt, Vorrotation, Drosseln, Ventile)
- Der Kalkulator liefert eine schnelle „Ampel“ zur Frage:
  - **Liege ich thermodynamisch schon im Sättigungs-/Flashbereich?**

**Hinweis:** Der Rechner gilt für **reines Wasser**. Für Gemische (KSS, Glykol, Salz, gelöste Gase) verschiebt sich die Sättigung.
"""
    )

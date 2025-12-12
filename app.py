import math
import streamlit as st
import matplotlib.pyplot as plt

from iapws import IAPWS97  # IF97 steam tables

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
    steps = [
        0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
        7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75
    ]
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
            penalty = abs(Qw_target - Q_eval) / span * 10.0  # 10 m virtuelle Strafe

        H_at = interp_clamped(Q_eval, p["Qw"], p["Hw"])
        eta_at = interp_clamped(Q_eval, p["Qw"], p["eta"])
        errH = abs(H_at - Hw_target)
        score = errH + penalty

        cand = {
            "id": p["id"], "pump": p,
            "in_range": in_range, "Q_eval": Q_eval, "penalty": penalty,
            "H_at": H_at, "eta_at": eta_at, "errH": errH, "score": score
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
# Sättigungs-Rechner (IF97)
# ---------------------------
def sat_temperature_from_pressure(p_bar_abs: float) -> float:
    """Sättigungstemperatur [°C] aus absolutem Druck [bar]."""
    p_mpa = p_bar_abs * 0.1
    w = IAPWS97(P=p_mpa, x=0)  # saturated liquid
    return w.T - 273.15

def sat_pressure_from_temperature(t_c: float) -> float:
    """Sättigungsdruck [bar abs] aus Temperatur [°C]."""
    w = IAPWS97(T=t_c + 273.15, x=0)
    return w.P * 10.0  # MPa -> bar

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Mehrphasen-Auslegung: Pumpe + Sättigung", layout="centered")
st.title("Auslegung: viskoser Arbeitspunkt + Sättigung (Mehrphasen)")

tab1, tab2 = st.tabs(["1) Pumpenauswahl & Kennlinien", "2) Sättigungskalkulator Wasser"])

# =========================================================
# TAB 1
# =========================================================
with tab1:
    with st.sidebar:
        st.header("Pumpen-Tab: Eingaben")

        st.subheader("Anforderung im Medium (viskos)")
        Qv_req = st.number_input("Qν (gefordert) [m³/h]", min_value=0.1, max_value=300.0, value=40.0, step=1.0, key="Qv_req")
        Hv_req = st.number_input("Hν (gefordert) [m]", min_value=0.1, max_value=300.0, value=35.0, step=1.0, key="Hv_req")

        st.subheader("Medium")
        mk = st.selectbox("Medium auswählen", list(MEDIA.keys()), index=0, key="mk")
        rho_def, nu_def = MEDIA[mk]
        rho = st.number_input("ρ [kg/m³] (anpassbar)", min_value=1.0, value=float(rho_def), step=5.0, key="rho")
        nu = st.number_input("ν [cSt] (anpassbar)", min_value=0.1, value=float(nu_def), step=0.5, key="nu")

        st.subheader("Auswahlverhalten")
        allow_out = st.checkbox("Wenn Qw außerhalb liegt: trotzdem 'Best fit' wählen", value=True, key="allow_out")

        st.subheader("Motorreserve")
        reserve_pct = st.slider("Reserve [%]", 0, 30, 15, key="reserve_pct")

    conv = viscous_to_water_equivalent(Qv_req, Hv_req, nu_cSt=nu)
    Qw_eq, Hw_eq = conv["Qw"], conv["Hw"]
    B_req, Cq_req, Ch_req, Ceta_req = conv["B"], conv["Cq"], conv["Ch"], conv["Ceta"]

    if nu <= 1.5:
        st.info("Medium ~ Wasser: Korrekturfaktoren werden auf 1 gesetzt → Qw=Qν und Hw=Hν.")

    st.subheader("1) Wasseräquivalenzpunkt (für Pumpenauswahl)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qw [m³/h]", f"{Qw_eq:.2f}")
    c2.metric("Hw [m]", f"{Hw_eq:.2f}")
    c3.metric("Cq / Ch", f"{Cq_req:.3f} / {Ch_req:.3f}")
    c4.metric("B [-]", f"{B_req:.2f}")

    best = choose_best_pump_robust(PUMPS, Qw_target=Qw_eq, Hw_target=Hw_eq, allow_out_of_range=allow_out)
    if best is None:
        st.error("Keine Pumpe konnte bewertet werden (unerwartet).")
        st.stop()

    st.divider()
    st.subheader("2) Beste Pumpe (Wasserkennlinie)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ausgewählte Pumpe", best["id"])
    c2.metric("H_pump(Q) [m]", f"{best['H_at']:.2f}")
    c3.metric("|ΔH| [m]", f"{best['errH']:.2f}")
    st.write(f"ηw(Q) ≈ **{best['eta_at']:.3f}**")

    if not best["in_range"]:
        p = best["pump"]
        st.warning(
            f"Qw={Qw_eq:.2f} liegt außerhalb der Kennlinie (Range: {min(p['Qw'])}…{max(p['Qw'])}). "
            "Es wurde am Rand ausgewertet und eine Strafbewertung genutzt."
        )

    eta_v_req = max(1e-6, Ceta_req * best["eta_at"])
    P_hyd_W = float(rho) * G * (float(Qv_req) / 3600.0) * float(Hv_req)
    P_vis_kW = (P_hyd_W / eta_v_req) / 1000.0
    P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

    st.divider()
    st.subheader("3) Motor (viskoser Betrieb)")
    c1, c2, c3 = st.columns(3)
    c1.metric("ην [-]", f"{eta_v_req:.3f}")
    c2.metric("Pν [kW]", f"{P_vis_kW:.2f}")
    c3.metric(f"Motor +{reserve_pct}% [kW]", f"{P_motor_kW:.2f}")

    # Viskose Kennlinie der ausgewählten Pumpe erzeugen (nur diese Pumpe)
    p = best["pump"]
    Qv_curve, Hv_curve, eta_v_curve = [], [], []
    for Qw_i, Hw_i, eta_w_i in zip(p["Qw"], p["Hw"], p["eta"]):
        Qv_i, Hv_i, eta_v_i = water_point_to_viscous(Qw_i, Hw_i, eta_w_i, nu_cSt=nu)
        Qv_curve.append(Qv_i)
        Hv_curve.append(Hv_i)
        eta_v_curve.append(eta_v_i)

    # Markerpunkte (clamp für plot)
    Qw_plot = clamp(Qw_eq, min(p["Qw"]), max(p["Qw"]))
    H_w_curve_at_Qw = interp_clamped(Qw_plot, p["Qw"], p["Hw"])
    eta_w_curve_at_Qw = interp_clamped(Qw_plot, p["Qw"], p["eta"])

    Qv_plot = clamp(Qv_req, min(Qv_curve), max(Qv_curve))
    H_v_curve_at_Qv = interp_clamped(Qv_plot, Qv_curve, Hv_curve)
    eta_v_curve_at_Qv = interp_clamped(Qv_plot, Qv_curve, eta_v_curve)

    st.divider()
    st.subheader("Kennlinien-Visualisierung")

    fig1, ax1 = plt.subplots()
    for pp in PUMPS:
        ax1.plot(pp["Qw"], pp["Hw"], marker="o", linestyle="-", label=pp["id"])
    ax1.plot(Qv_curve, Hv_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos)")

    ax1.scatter([Qw_eq], [Hw_eq], marker="^", s=70, label="Wasseräquivalent (Qw,Hw)")
    ax1.scatter([Qv_req], [Hv_req], marker="x", s=80, label="Arbeitspunkt viskos (Qν,Hν)")
    ax1.scatter([Qw_plot], [H_w_curve_at_Qw], marker="s", s=55, label="H_w auf Auswahlkennlinie")
    ax1.scatter([Qv_plot], [H_v_curve_at_Qv], marker="s", s=55, label="H_ν auf viskoser Auswahlkennlinie")

    ax1.set_xlabel("Q [m³/h]")
    ax1.set_ylabel("H [m]")
    ax1.set_title("Q-H: 5 Wasserkennlinien + viskose Kennlinie der Auswahl")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots()
    for pp in PUMPS:
        ax2.plot(pp["Qw"], pp["eta"], marker="o", linestyle="-", label=pp["id"])
    ax2.plot(Qv_curve, eta_v_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos η)")

    ax2.scatter([Qw_plot], [eta_w_curve_at_Qw], marker="x", s=80, label="ηw auf Auswahlkennlinie")
    ax2.scatter([Qv_req], [eta_v_req], marker="^", s=70, label="ην am Arbeitspunkt")
    ax2.scatter([Qv_plot], [eta_v_curve_at_Qv], marker="s", s=55, label="ην auf viskoser Auswahlkennlinie")

    ax2.set_xlabel("Q [m³/h]")
    ax2.set_ylabel("η [-]")
    ax2.set_title("Q-η: 5 Wasserkennlinien + viskose η-Kennlinie der Auswahl")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader("Sättigungskalkulator Wasser unter Druck (IF97)")
    st.caption("Für Mehrphasenpumpen ist entscheidend, ob bei gegebenem Druck die Betriebstemperatur über der Sättigung liegt (Flash/Boiling-Risiko).")

    mode = st.radio("Berechnung", ["Sättigungstemperatur aus Druck", "Sättigungsdruck aus Temperatur"], horizontal=True)

    colA, colB = st.columns(2)

    if mode == "Sättigungstemperatur aus Druck":
        with colA:
            p_bar_abs = st.number_input("Absolutdruck p_abs [bar]", min_value=0.01, max_value=300.0, value=1.013, step=0.1)
            t_op = st.number_input("Betriebstemperatur T_op [°C] (optional)", min_value=-20.0, max_value=400.0, value=20.0, step=1.0)
            margin = st.number_input("Sicherheitsabstand ΔT [K]", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

        try:
            t_sat = sat_temperature_from_pressure(float(p_bar_abs))
            with colB:
                st.metric("T_sätt [°C]", f"{t_sat:.2f}")
                st.write(f"Sättigung bei p_abs={p_bar_abs:.3f} bar: **T_sätt ≈ {t_sat:.2f} °C**")

                dt = t_sat - float(t_op)
                st.metric("ΔT = T_sätt - T_op [K]", f"{dt:.2f}")
                if dt < 0:
                    st.error("T_op liegt über T_sätt → Flash/Boiling sehr wahrscheinlich (Mehrphasenbildung).")
                elif dt < margin:
                    st.warning("T_op liegt nahe an T_sätt → geringe Reserve, Mehrphasen-/Kavitationsrisiko steigt.")
                else:
                    st.success("Ausreichende thermische Reserve zur Sättigung (bezogen auf p_abs).")

        except Exception as e:
            st.error(f"Berechnung fehlgeschlagen (IF97): {e}")

    else:
        with colA:
            t_c = st.number_input("Temperatur T [°C]", min_value=-20.0, max_value=400.0, value=100.0, step=1.0)
            p_op = st.number_input("Betriebsdruck p_abs [bar] (optional)", min_value=0.01, max_value=300.0, value=1.013, step=0.1)
            margin_p = st.number_input("Sicherheitsabstand Δp [bar]", min_value=0.0, max_value=50.0, value=0.2, step=0.05)

        try:
            p_sat = sat_pressure_from_temperature(float(t_c))
            with colB:
                st.metric("p_sätt [bar abs]", f"{p_sat:.3f}")
                st.write(f"Sättigung bei T={t_c:.2f} °C: **p_sätt ≈ {p_sat:.3f} bar(abs)**")

                dp = float(p_op) - p_sat
                st.metric("Δp = p_op - p_sätt [bar]", f"{dp:.3f}")
                if dp < 0:
                    st.error("p_op liegt unter p_sätt → Flash/Boiling sehr wahrscheinlich (Mehrphasenbildung).")
                elif dp < margin_p:
                    st.warning("p_op liegt nahe an p_sätt → geringe Druckreserve, Mehrphasen-/Kavitationsrisiko steigt.")
                else:
                    st.success("Ausreichende Druckreserve zur Sättigung.")

        except Exception as e:
            st.error(f"Berechnung fehlgeschlagen (IF97): {e}")

    st.divider()
    st.markdown("### Hinweis zur Praxis")
    st.write(
        "- Für Mehrphasenpumpen ist p_abs entlang der Strömung relevant (Saugseite / Laufradeintritt besonders kritisch).\n"
        "- Der Rechner nutzt IF97-Sattdampfgleichungen (reines Wasser). Für Mediengemische (KSS, Glykol etc.) verschiebt sich die Sättigung."
    )

import math
import streamlit as st
import matplotlib.pyplot as plt

G = 9.80665  # m/s²

# ---------------------------
# Typische Pumpenkennlinie (Wasser) – fest hinterlegt (Demo)
# ---------------------------
PUMP_QW = [0, 20, 40, 60, 80]                 # m³/h
PUMP_HW = [48, 46, 40, 30, 18]                # m
PUMP_ETAW = [0.30, 0.60, 0.72, 0.68, 0.55]    # -

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
# Hilfsfunktionen
# ---------------------------
def interp(x, xs, ys):
    """Lineare Interpolation mit Clamping am Rand."""
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


def motor_iec(P_kW):
    """Nächstgrößere typische IEC-Motorstufe (vereinfachte Liste)."""
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
    """Kennzahl B (Korrelationsform)."""
    Q = max(Q_vis_m3h, 1e-12)
    H = max(H_vis_m, 1e-12)
    nu = max(nu_cSt, 1e-12)
    return 280.0 * (nu ** 0.5) / ((Q ** 0.25) * (H ** 0.125))


def viscosity_factors_from_B(B, nu_cSt, nu_water_threshold=1.5):
    """
    Liefert (Cq, Ch, Ceta).
    WICHTIG: Für Wasser/dünnflüssig (ν <= threshold) keine Korrektur => C=1.
    """
    if nu_cSt <= nu_water_threshold:
        return 1.0, 1.0, 1.0

    if B <= 1.0:
        return 1.0, 1.0, 1.0

    # Cq ≈ Ch
    Cq = math.exp(-0.165 * (math.log10(B) ** 3.15))
    Cq = min(max(Cq, 0.0), 1.0)
    Ch = Cq

    # Ceta (clamped)
    Beff = min(max(B, 1.0000001), 40.0)
    Ceta = Beff ** (-(0.0547 * (Beff ** 0.69)))
    Ceta = min(max(Ceta, 0.0), 1.0)

    return Cq, Ch, Ceta


def viscous_to_water_equivalent(Qv, Hv, nu_cSt):
    """
    Rückrechnung viskos -> Wasseräquivalent:
      Qv = Cq*Qw => Qw = Qv/Cq
      Hv = Ch*Hw => Hw = Hv/Ch
    Mit Wasser-Bypass.
    """
    B = compute_B(Qv, Hv, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)

    Qw = Qv / Cq
    Hw = Hv / Ch

    return {"Qw": Qw, "Hw": Hw, "B": B, "Cq": Cq, "Ch": Ch, "Ceta": Ceta}


def water_point_to_viscous(Qw, Hw, eta_w, nu_cSt, max_iter=50, tol=1e-10):
    """
    Vorwärtsumrechnung eines Wasser-Kennlinienpunktes (Qw, Hw, eta_w) -> (Qv, Hv, eta_v)
    Iteration, weil B über viskose Größen definiert wird und Cq/Ch davon abhängen.
    Mit Wasser-Bypass.
    """
    # Wasser-Bypass: bleibt identisch
    if nu_cSt <= 1.5:
        return Qw, Hw, eta_w, 1.0, 1.0, 1.0, 1.0

    Qv = max(Qw, 1e-12)
    Hv = max(Hw, 1e-12)

    for _ in range(max_iter):
        B = compute_B(Qv, Hv, nu_cSt)
        Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)

        Qv_new = Cq * Qw
        Hv_new = Ch * Hw

        dq = abs(Qv_new - Qv) / max(Qv, 1e-12)
        dh = abs(Hv_new - Hv) / max(Hv, 1e-12)

        Qv, Hv = max(Qv_new, 1e-12), max(Hv_new, 1e-12)

        if max(dq, dh) < tol:
            break

    # final factors and eta
    B = compute_B(Qv, Hv, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)
    eta_v = max(1e-6, Ceta * eta_w)

    return Qv, Hv, eta_v, B, Cq, Ch, Ceta


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Wasser + viskose Kennlinie", layout="centered")
st.title("Pumpenkennlinie: Wasser vs. viskos (mit korrektem Wasser-Fall)")

with st.sidebar:
    st.header("Anforderung im Medium (viskos)")
    Qv_req = st.number_input("Qν (gefordert) [m³/h]", min_value=0.1, max_value=200.0, value=40.0, step=1.0)
    Hv_req = st.number_input("Hν (gefordert) [m]", min_value=0.1, max_value=200.0, value=35.0, step=1.0)

    st.header("Medium")
    mk = st.selectbox("Medium auswählen", list(MEDIA.keys()), index=0)
    rho_default, nu_default = MEDIA[mk]
    rho = st.number_input("ρ [kg/m³] (anpassbar)", min_value=1.0, value=float(rho_default), step=5.0)
    nu = st.number_input("ν [cSt] (anpassbar)", min_value=0.1, value=float(nu_default), step=0.5)

    st.header("Motorreserve")
    reserve_pct = st.slider("Reserve [%]", 0, 30, 15)

# ---------------------------
# 1) Rückrechnung Arbeitspunkt viskos -> Wasseräquivalent
# ---------------------------
conv = viscous_to_water_equivalent(Qv_req, Hv_req, nu_cSt=nu)
Qw_eq = conv["Qw"]
Hw_eq = conv["Hw"]
B_req = conv["B"]
Cq_req, Ch_req, Ceta_req = conv["Cq"], conv["Ch"], conv["Ceta"]

# Wasserkennlinie am äquivalenten Durchfluss auswerten (zur Plausibilisierung)
H_w_curve_at_Qw = interp(Qw_eq, PUMP_QW, PUMP_HW)
eta_w_at_Qw = interp(Qw_eq, PUMP_QW, PUMP_ETAW)

# Viskoser Wirkungsgrad am Arbeitspunkt (aus Ceta und Wasser-eta am äquivalenten Punkt)
eta_v_req = max(1e-6, Ceta_req * eta_w_at_Qw)

# Leistung / Motor für viskosen Betriebspunkt (Qv_req, Hv_req)
P_hyd_W = float(rho) * G * (float(Qv_req) / 3600.0) * float(Hv_req)
P_vis_kW = (P_hyd_W / eta_v_req) / 1000.0
P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))

st.subheader("Ergebnis Arbeitspunkt & Motor")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Qw (äquiv.) [m³/h]", f"{Qw_eq:.2f}")
c2.metric("Hw (äquiv.) [m]", f"{Hw_eq:.2f}")
c3.metric("Cq / Ch", f"{Cq_req:.3f} / {Ch_req:.3f}")
c4.metric("B (am Arbeitspunkt)", f"{B_req:.2f}")

st.write(f"ηw(Qw) ≈ **{eta_w_at_Qw:.3f}**,  ην ≈ **{eta_v_req:.3f}**")
st.write(f"Leistung viskos: **{P_vis_kW:.2f} kW**")
st.write(f"Motor (+{reserve_pct}%): **{P_motor_kW:.2f} kW (IEC)**")

if nu <= 1.5:
    st.info("Medium ist (nahe) Wasser → Korrekturfaktoren werden auf 1 gesetzt → Qν = Qw und Hν = Hw.")

st.divider()

# ---------------------------
# 2) Viskose Kennlinie aus Wasserkennlinie erzeugen (punktweise Umrechnung)
# ---------------------------
Qv_curve = []
Hv_curve = []
eta_v_curve = []
meta_curve = []  # optional: B, Cq, Ch, Ceta pro Punkt

for Qw_i, Hw_i, eta_w_i in zip(PUMP_QW, PUMP_HW, PUMP_ETAW):
    Qv_i, Hv_i, eta_v_i, B_i, Cq_i, Ch_i, Ceta_i = water_point_to_viscous(
        Qw_i, Hw_i, eta_w_i, nu_cSt=nu
    )
    Qv_curve.append(Qv_i)
    Hv_curve.append(Hv_i)
    eta_v_curve.append(eta_v_i)
    meta_curve.append((B_i, Cq_i, Ch_i, Ceta_i))

# Für Markierung: Kennlinienwert im viskosen Plot bei Qv_req
# (Interpolation auf viskoser Kennlinie)
# Hinweis: Qv_curve muss monoton sein; bei dieser typischen Kurve ist das i. d. R. gegeben.
Hv_vis_curve_at_Qv = interp(Qv_req, Qv_curve, Hv_curve)
eta_vis_curve_at_Qv = interp(Qv_req, Qv_curve, eta_v_curve)

# ---------------------------
# 3) Visualisierung: Q-H (Wasser + viskos)
# ---------------------------
st.subheader("Visualisierung: Q-H und Q-η (Wasser vs. viskos)")

fig1, ax1 = plt.subplots()
ax1.plot(PUMP_QW, PUMP_HW, marker="o", linestyle="-", label="H(Q) Wasser (typ.)")
ax1.plot(Qv_curve, Hv_curve, marker="o", linestyle="-", label="Hν(Qν) viskos (umgerechnet)")

# Markiere Arbeitspunkt viskos und Wasseräquivalent
ax1.scatter([Qv_req], [Hv_req], marker="x", s=80, label="Arbeitspunkt viskos (Qν,Hν)")
ax1.scatter([Qw_eq], [Hw_eq], marker="^", s=70, label="Wasseräquivalent (Qw,Hw)")

# Markiere Kennlinienwerte an denselben Durchflüssen (zur Plausibilisierung)
ax1.scatter([Qw_eq], [H_w_curve_at_Qw], marker="s", s=55, label="Wasserkennlinie bei Qw")
ax1.scatter([Qv_req], [Hv_vis_curve_at_Qv], marker="s", s=55, label="Viskose Kennlinie bei Qν")

ax1.set_xlabel("Q [m³/h]")
ax1.set_ylabel("H [m]")
ax1.set_title("Q-H: Wasserkennlinie und viskose Kennlinie")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1, clear_figure=True)

# ---------------------------
# 4) Visualisierung: Q-η (Wasser + viskos)
# ---------------------------
fig2, ax2 = plt.subplots()
ax2.plot(PUMP_QW, PUMP_ETAW, marker="o", linestyle="-", label="η(Q) Wasser (typ.)")
ax2.plot(Qv_curve, eta_v_curve, marker="o", linestyle="-", label="ην(Qν) viskos (umgerechnet)")

ax2.scatter([Qw_eq], [eta_w_at_Qw], marker="x", s=80, label="ηw bei Qw")
ax2.scatter([Qv_req], [eta_v_req], marker="^", s=70, label="ην am Arbeitspunkt")

ax2.scatter([Qv_req], [eta_vis_curve_at_Qv], marker="s", s=55, label="Viskose Kennlinie bei Qν")

ax2.set_xlabel("Q [m³/h]")
ax2.set_ylabel("η [-]")
ax2.set_title("Q-η: Wasserkennlinie und viskose (korrigierte) Kennlinie")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2, clear_figure=True)

# ---------------------------
# 5) Optional: Detailtabelle für viskose Kennlinie
# ---------------------------
with st.expander("Details: umgerechnete viskose Kennlinienpunkte"):
    rows = []
    for (Qw_i, Hw_i, eta_w_i, Qv_i, Hv_i, eta_v_i, meta) in zip(
        PUMP_QW, PUMP_HW, PUMP_ETAW, Qv_curve, Hv_curve, eta_v_curve, meta_curve
    ):
        B_i, Cq_i, Ch_i, Ceta_i = meta
        rows.append({
            "Qw": Qw_i, "Hw": Hw_i, "ηw": round(eta_w_i, 4),
            "Qν": round(Qv_i, 4), "Hν": round(Hv_i, 4), "ην": round(eta_v_i, 4),
            "B": round(B_i, 4), "Cq": round(Cq_i, 4), "Ch": round(Ch_i, 4), "Cη": round(Ceta_i, 4)
        })
    st.dataframe(rows, use_container_width=True)

st.caption(
    "Hinweis: Die Wasserkennlinie ist eine typische Demonstrationskennlinie. "
    "Für reale Auslegung müssen herstellerspezifische Kennlinien hinterlegt werden."
)

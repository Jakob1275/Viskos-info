# app.py
import math
import streamlit as st
import matplotlib.pyplot as plt

G = 9.80665  # m/s²

# ---------------------------
# Typische Pumpenkennlinie (Wasser) – fest hinterlegt
# ---------------------------
PUMP_Q = [0, 20, 40, 60, 80]                 # m³/h
PUMP_H = [48, 46, 40, 30, 18]                # m
PUMP_ETA = [0.30, 0.60, 0.72, 0.68, 0.55]    # -

# ---------------------------
# Medium-Datenbank (typische Richtwerte)
# ---------------------------
MEDIA = {
    "Wasser (20°C)": (998.0, 1.0),
    "Hydrauliköl ISO VG 32": (860.0, 32.0),
    "Hydrauliköl ISO VG 46": (870.0, 46.0),
    "Hydrauliköl ISO VG 68": (880.0, 68.0),
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
            x0, y0 = xs[i-1], ys[i-1]
            x1, y1 = xs[i], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]

def compute_B(Qv, Hv, nu):
    Qv = max(Qv, 1e-12)
    Hv = max(Hv, 1e-12)
    nu = max(nu, 1e-12)
    return 280.0 * (nu ** 0.5) / ((Qv ** 0.25) * (Hv ** 0.125))

def viscosity_factors(B):
    """Gibt (Cq, Ch, Ceta) zurück."""
    if B <= 1.0:
        return 1.0, 1.0, 1.0

    # Cq ≈ Ch
    Cq = math.exp(-0.165 * (math.log10(B) ** 3.15))
    Cq = min(max(Cq, 0.0), 1.0)
    Ch = Cq

    # Ceta (clamped auf B<=40)
    Beff = min(max(B, 1.0000001), 40.0)
    Ceta = Beff ** (-(0.0547 * (Beff ** 0.69)))
    Ceta = min(max(Ceta, 0.0), 1.0)

    return Cq, Ch, Ceta

def motor_iec(P_kW):
    """Nächstgrößere typische IEC-Motorstufe (vereinfachte Liste)."""
    steps = [0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Viskoser Arbeitspunkt → Q-H & Q-η", layout="centered")
st.title("Viskoser Arbeitspunkt → Wasseräquivalent → typische Kennlinie (mit Plots)")

st.sidebar.header("Anforderung (Medium / viskos)")
Qv = st.sidebar.number_input("Qν [m³/h]", min_value=1.0, max_value=80.0, value=40.0, step=1.0)
Hv = st.sidebar.number_input("Hν [m]", min_value=1.0, max_value=60.0, value=35.0, step=1.0)

medium = st.sidebar.selectbox("Medium", list(MEDIA.keys()))
rho, nu = MEDIA[medium]

reserve = st.sidebar.slider("Motorreserve [%]", 0, 30, 15)

# ---------------------------
# Rückrechnung viskos → Wasseräquivalent
# ---------------------------
B = compute_B(Qv, Hv, nu)
Cq, Ch, Ceta = viscosity_factors(B)

Qw = Qv / Cq
Hw = Hv / Ch

# Kennlinienwerte am Wasserpunkt
H_w_curve = interp(Qw, PUMP_Q, PUMP_H)
eta_w = interp(Qw, PUMP_Q, PUMP_ETA)

# Wirkungsgrad viskos (Korrektur)
eta_v = Ceta * eta_w
eta_v = max(eta_v, 1e-6)

# Leistung viskos am geforderten viskosen Betriebspunkt (Qv, Hv)
P_hyd_W = rho * G * (Qv / 3600.0) * Hv
P_vis_kW = (P_hyd_W / eta_v) / 1000.0
P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve / 100.0))

# ---------------------------
# Ergebnisse oben
# ---------------------------
st.subheader("Kennzahlen & Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Qw [m³/h]", f"{Qw:.1f}")
c2.metric("Hw [m]", f"{Hw:.1f}")
c3.metric("B [-]", f"{B:.2f}")
c4.metric("ην [-]", f"{eta_v:.3f}")

st.write(f"Leistung viskos: **{P_vis_kW:.2f} kW**")
st.write(f"Motor (+{reserve}%): **{P_motor_kW:.2f} kW (IEC)**")

st.divider()
st.subheader("Visualisierung der typischen Kennlinien")

# ---------------------------
# Plot 1: Q-H
# ---------------------------
fig1, ax1 = plt.subplots()
ax1.plot(PUMP_Q, PUMP_H, marker="o", linestyle="-", label="H(Q) Wasser (typ.)")

# Markiere Wasseräquivalentpunkt (Qw, Hw) und den Kennlinienwert H(Qw)
ax1.scatter([Qw], [H_w_curve], marker="x", s=80, label="Punkt auf Kennlinie bei Qw")
ax1.scatter([Qw], [Hw], marker="^", s=60, label="Wasseräquivalent (Qw, Hw)")

# Optionale Hilfslinien
ax1.axvline(Qw, linewidth=1, linestyle="--")
ax1.axhline(Hw, linewidth=1, linestyle="--")

ax1.set_xlabel("Q [m³/h]")
ax1.set_ylabel("H [m]")
ax1.set_title("Q-H-Kennlinie (Wasser) mit Wasseräquivalentpunkt")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1, clear_figure=True)

# ---------------------------
# Plot 2: Q-eta
# ---------------------------
fig2, ax2 = plt.subplots()
ax2.plot(PUMP_Q, PUMP_ETA, marker="o", linestyle="-", label="η(Q) Wasser (typ.)")

# Markiere ηw bei Qw und ηv (korrigiert)
ax2.scatter([Qw], [eta_w], marker="x", s=80, label="ηw bei Qw")
ax2.scatter([Qw], [eta_v], marker="^", s=60, label="ην = Cη·ηw (bei Qw)")

ax2.axvline(Qw, linewidth=1, linestyle="--")
ax2.set_xlabel("Q [m³/h]")
ax2.set_ylabel("η [-]")
ax2.set_title("Q-η-Kennlinie (Wasser) + viskose η-Korrektur")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2, clear_figure=True)

st.caption(
    "Hinweis: Die Kennlinie ist eine typische Demonstrationskennlinie. "
    "Für reale Auslegung müssen herstellerspezifische Kennlinien hinterlegt werden."
)

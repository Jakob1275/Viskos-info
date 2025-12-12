# app.py
import math
import streamlit as st

G = 9.80665  # m/s²

# ---------------------------
# Typische Pumpenkennlinie (Wasser)
# ---------------------------
PUMP_Q = [0, 20, 40, 60, 80]          # m³/h
PUMP_H = [48, 46, 40, 30, 18]         # m
PUMP_ETA = [0.30, 0.60, 0.72, 0.68, 0.55]  # -

# ---------------------------
# Medium-Datenbank
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
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, y0 = xs[i-1], ys[i-1]
            x1, y1 = xs[i], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def compute_B(Qv, Hv, nu):
    return 280 * (nu ** 0.5) / ((Qv ** 0.25) * (Hv ** 0.125))

def viscosity_factors(B):
    if B <= 1:
        return 1.0, 1.0, 1.0
    Cq = math.exp(-0.165 * (math.log10(B) ** 3.15))
    Ch = Cq
    Beff = min(B, 40)
    Ceta = Beff ** (-(0.0547 * Beff ** 0.69))
    return Cq, Ch, Ceta

def motor_iec(P):
    steps = [0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45]
    for s in steps:
        if P <= s:
            return s
    return steps[-1]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Viskoser Arbeitspunkt → Pumpe & Motor")
st.title("Viskoser Arbeitspunkt → typische Kreiselpumpe")

st.sidebar.header("Anforderung (Medium)")
Qv = st.sidebar.number_input("Qν [m³/h]", 1.0, 80.0, 40.0)
Hv = st.sidebar.number_input("Hν [m]", 1.0, 60.0, 35.0)

medium = st.sidebar.selectbox("Medium", list(MEDIA.keys()))
rho, nu = MEDIA[medium]

reserve = st.sidebar.slider("Motorreserve [%]", 0, 30, 15)

# ---------------------------
# Rückrechnung auf Wasser
# ---------------------------
B = compute_B(Qv, Hv, nu)
Cq, Ch, Ceta = viscosity_factors(B)

Qw = Qv / Cq
Hw = Hv / Ch

# ---------------------------
# Arbeitspunkt auf typischer Kennlinie
# ---------------------------
Hw_pump = interp(Qw, PUMP_Q, PUMP_H)
eta_w = interp(Qw, PUMP_Q, PUMP_ETA)
eta_v = Ceta * eta_w

P_hyd = rho * G * (Qv / 3600) * Hv
P_vis = P_hyd / eta_v / 1000
P_motor = motor_iec(P_vis * (1 + reserve / 100))

# ---------------------------
# Ergebnisse
# ---------------------------
st.subheader("Ergebnisse")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qw [m³/h]", f"{Qw:.1f}")
c2.metric("Hw [m]", f"{Hw:.1f}")
c3.metric("ην [-]", f"{eta_v:.3f}")
c4.metric("B [-]", f"{B:.2f}")

st.markdown("### Motorleistung")
st.write(f"- Leistung viskos: **{P_vis:.2f} kW**")
st.write(f"- Motor (+{reserve}%): **{P_motor:.1f} kW (IEC)**")

st.caption(
    "Hinweis: Es wird eine typische Wasserkennlinie einer radialen Kreiselpumpe verwendet. "
    "Für reale Auslegung sind herstellerspezifische Kennlinien zu hinterlegen."
)

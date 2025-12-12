import math
import streamlit as st
import matplotlib.pyplot as plt

G = 9.80665  # m/s²

# ---------------------------
# 5 typische Pumpenkennlinien (Wasser) – Demo-Daten
# Jede Pumpe: Q [m³/h], H [m], eta [-]
# ---------------------------
PUMPS = [
    {
        "id": "P1 (klein, steil)",
        "Qw": [0, 15, 30, 45, 60],
        "Hw": [55, 53, 48, 40, 28],
        "eta": [0.28, 0.52, 0.68, 0.66, 0.52],
    },
    {
        "id": "P2 (mittel, ausgewogen)",
        "Qw": [0, 20, 40, 60, 80],
        "Hw": [48, 46, 40, 30, 18],
        "eta": [0.30, 0.60, 0.72, 0.68, 0.55],
    },
    {
        "id": "P3 (höherer Durchfluss)",
        "Qw": [0, 30, 60, 90, 120],
        "Hw": [42, 41, 36, 26, 14],
        "eta": [0.25, 0.55, 0.73, 0.70, 0.58],
    },
    {
        "id": "P4 (höhere Förderhöhe)",
        "Qw": [0, 15, 30, 45, 60],
        "Hw": [70, 68, 62, 52, 40],
        "eta": [0.22, 0.48, 0.66, 0.65, 0.50],
    },
    {
        "id": "P5 (flacher, effizient im mittleren Bereich)",
        "Qw": [0, 25, 50, 75, 100],
        "Hw": [46, 44, 38, 28, 16],
        "eta": [0.30, 0.62, 0.75, 0.72, 0.60],
    },
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
    # Wasser-/dünnflüssig-Bypass
    if nu_cSt <= nu_water_threshold:
        return 1.0, 1.0, 1.0

    if B <= 1.0:
        return 1.0, 1.0, 1.0

    Cq = math.exp(-0.165 * (math.log10(B) ** 3.15))
    Cq = min(max(Cq, 0.0), 1.0)
    Ch = Cq

    Beff = min(max(B, 1.0000001), 40.0)
    Ceta = Beff ** (-(0.0547 * (Beff ** 0.69)))
    Ceta = min(max(Ceta, 0.0), 1.0)

    return Cq, Ch, Ceta

def viscous_to_water_equivalent(Qv, Hv, nu_cSt):
    B = compute_B(Qv, Hv, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)
    if Cq <= 0 or Ch <= 0:
        raise ValueError("Cq/Ch <= 0. Prüfe Eingaben.")
    return {
        "B": B, "Cq": Cq, "Ch": Ch, "Ceta": Ceta,
        "Qw": Qv / Cq,
        "Hw": Hv / Ch
    }

def water_point_to_viscous(Qw, Hw, eta_w, nu_cSt, max_iter=60, tol=1e-10):
    # Wasser-Bypass
    if nu_cSt <= 1.5:
        return Qw, Hw, max(1e-6, eta_w), 1.0, 1.0, 1.0, 1.0

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

    B = compute_B(Qv, Hv, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, nu_cSt)
    eta_v = max(1e-6, Ceta * eta_w)
    return Qv, Hv, eta_v, B, Cq, Ch, Ceta

# ---------------------------
# Best-Pump-Auswahl
# ---------------------------
def choose_best_pump(pumps, Qw_target, Hw_target):
    """
    Auswahlkriterium:
      - Qw_target muss im Bereich der Pumpe liegen
      - minimiere |H_pump(Qw_target) - Hw_target|
      - bei Gleichstand: höhere eta_w bevorzugen
    """
    best = None
    for p in pumps:
        qmin, qmax = min(p["Qw"]), max(p["Qw"])
        if not (qmin <= Qw_target <= qmax):
            continue

        H_at = interp(Qw_target, p["Qw"], p["Hw"])
        eta_at = interp(Qw_target, p["Qw"], p["eta"])
        errH = abs(H_at - Hw_target)

        cand = {
            "id": p["id"],
            "errH": errH,
            "H_at": H_at,
            "eta_at": eta_at,
            "pump": p
        }

        if best is None:
            best = cand
        else:
            # primär nach errH, sekundär nach höherer eta
            if cand["errH"] < best["errH"] - 1e-9:
                best = cand
            elif abs(cand["errH"] - best["errH"]) <= 1e-9 and cand["eta_at"] > best["eta_at"]:
                best = cand

    if best is None:
        raise ValueError("Keine Pumpe hat den Qw-Äquivalenzpunkt im Kennlinienbereich.")
    return best

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Best-Pump Auswahl + viskose Kennlinie", layout="centered")
st.title("Viskoser Arbeitspunkt → Wasseräquivalent → beste Pumpe (aus 5 Kennlinien)")

with st.sidebar:
    st.header("Anforderung im Medium (viskos)")
    Qv_req = st.number_input("Qν (gefordert) [m³/h]", min_value=0.1, max_value=250.0, value=40.0, step=1.0)
    Hv_req = st.number_input("Hν (gefordert) [m]", min_value=0.1, max_value=250.0, value=35.0, step=1.0)

    st.header("Medium")
    mk = st.selectbox("Medium auswählen", list(MEDIA.keys()), index=0)
    rho_def, nu_def = MEDIA[mk]
    rho = st.number_input("ρ [kg/m³] (anpassbar)", min_value=1.0, value=float(rho_def), step=5.0)
    nu = st.number_input("ν [cSt] (anpassbar)", min_value=0.1, value=float(nu_def), step=0.5)

    st.header("Motorreserve")
    reserve_pct = st.slider("Reserve [%]", 0, 30, 15)

# 1) Rückrechnung viskos -> Wasseräquivalent
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

# 2) Beste Pumpe auswählen
best = choose_best_pump(PUMPS, Qw_target=Qw_eq, Hw_target=Hw_eq)
p = best["pump"]

st.divider()
st.subheader("2) Beste Pumpe (Wasserkennlinie)")
c1, c2, c3 = st.columns(3)
c1.metric("Ausgewählte Pumpe", best["id"])
c2.metric("H_pump(Qw) [m]", f"{best['H_at']:.2f}")
c3.metric("|ΔH| [m]", f"{best['errH']:.2f}")
st.write(f"ηw(Qw) ≈ **{best['eta_at']:.3f}**")

# 3) Motorleistung im viskosen Arbeitspunkt (η viskos = Ceta * ηw am Qw)
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

# 4) Viskose Kennlinie der ausgewählten Pumpe erzeugen
Qv_curve, Hv_curve, eta_v_curve = [], [], []
for Qw_i, Hw_i, eta_w_i in zip(p["Qw"], p["Hw"], p["eta"]):
    Qv_i, Hv_i, eta_v_i, *_ = water_point_to_viscous(Qw_i, Hw_i, eta_w_i, nu_cSt=nu)
    Qv_curve.append(Qv_i)
    Hv_curve.append(Hv_i)
    eta_v_curve.append(eta_v_i)

# Interpolierte Punkte auf ausgewählter Kennlinie (Wasser/viskos) für Markierungen
H_w_curve_at_Qw = interp(Qw_eq, p["Qw"], p["Hw"])
eta_w_curve_at_Qw = interp(Qw_eq, p["Qw"], p["eta"])
H_v_curve_at_Qv = interp(Qv_req, Qv_curve, Hv_curve)
eta_v_curve_at_Qv = interp(Qv_req, Qv_curve, eta_v_curve)

st.divider()
st.subheader("Kennlinien-Visualisierung (alle Pumpen + ausgewählte Pumpe hervorgehoben)")

# Plot Q-H
fig1, ax1 = plt.subplots()
for pp in PUMPS:
    ax1.plot(pp["Qw"], pp["Hw"], marker="o", linestyle="-", label=pp["id"])
ax1.scatter([Qw_eq], [Hw_eq], marker="^", s=70, label="Wasseräquivalent (Qw,Hw)")
ax1.scatter([Qv_req], [Hv_req], marker="x", s=80, label="Arbeitspunkt viskos (Qν,Hν)")
ax1.scatter([Qw_eq], [H_w_curve_at_Qw], marker="s", s=55, label="H_w auf ausgewählter Kennlinie")
ax1.scatter([Qv_req], [H_v_curve_at_Qv], marker="s", s=55, label="H_ν auf viskoser Kennlinie")

# Zusätzlich: viskose Kennlinie der ausgewählten Pumpe
ax1.plot(Qv_curve, Hv_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos)")

ax1.set_xlabel("Q [m³/h]")
ax1.set_ylabel("H [m]")
ax1.set_title("Q-H: 5 Wasserkennlinien + viskose Kennlinie der Auswahl")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1, clear_figure=True)

# Plot Q-eta
fig2, ax2 = plt.subplots()
for pp in PUMPS:
    ax2.plot(pp["Qw"], pp["eta"], marker="o", linestyle="-", label=pp["id"])
ax2.scatter([Qw_eq], [eta_w_curve_at_Qw], marker="x", s=80, label="ηw bei Qw (Auswahl)")
ax2.scatter([Qv_req], [eta_v_req], marker="^", s=70, label="ην am Arbeitspunkt")
ax2.plot(Qv_curve, eta_v_curve, marker="o", linestyle="--", label=f"{best['id']} (viskos η)")

ax2.set_xlabel("Q [m³/h]")
ax2.set_ylabel("η [-]")
ax2.set_title("Q-η: 5 Wasserkennlinien + viskose η-Kennlinie der Auswahl")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2, clear_figure=True)

with st.expander("Kurz-Rechengang"):
    st.markdown(
        f"""
- Eingabe (viskos): **Qν={Qv_req:.2f} m³/h**, **Hν={Hv_req:.2f} m**, **ν={nu:.2f} cSt**, **ρ={rho:.1f} kg/m³**
- Umrechnung: B={B_req:.2f} → Cq={Cq_req:.3f}, Ch={Ch_req:.3f}, Cη={Ceta_req:.3f}
- Wasseräquivalent: **Qw=Qν/Cq={Qw_eq:.2f}**, **Hw=Hν/Ch={Hw_eq:.2f}**
- Pumpenauswahl: minimiere |H_pump(Qw) − Hw| → **{best['id']}**, |ΔH|={best['errH']:.2f} m
- Motor: ην=Cη·ηw(Qw)={eta_v_req:.3f} → Pν=ρgQνHν/ην={P_vis_kW:.2f} kW → Motor (Reserve)={P_motor_kW:.2f} kW
"""
    )

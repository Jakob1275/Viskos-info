# app.py
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import streamlit as st

G = 9.80665  # m/s²


# ---------------------------
# Medium library (typische Richtwerte; in UI überschreibbar)
# ---------------------------
@dataclass(frozen=True)
class MediumProps:
    name: str
    rho_kgm3: float
    nu_cSt: float  # mm²/s


MEDIA_DB = {
    "Wasser (20°C)": MediumProps("Wasser (20°C)", 998.0, 1.0),
    "Wasser (60°C)": MediumProps("Wasser (60°C)", 983.0, 0.47),
    "Glykol 30% (20°C) (typ.)": MediumProps("Glykol 30% (20°C) (typ.)", 1040.0, 3.5),
    "Hydrauliköl ISO VG 32 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 32 (40°C) (typ.)", 860.0, 32.0),
    "Hydrauliköl ISO VG 46 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 46 (40°C) (typ.)", 870.0, 46.0),
    "Hydrauliköl ISO VG 68 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 68 (40°C) (typ.)", 880.0, 68.0),
    "Heizöl EL (20°C) (typ.)": MediumProps("Heizöl EL (20°C) (typ.)", 840.0, 6.0),
}


# ---------------------------
# Helpers
# ---------------------------
def interp1_clamped(x: float, xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Interpolation: xs/ys müssen gleiche Länge haben und >= 2 Punkte.")
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs, ys = zip(*pairs)
    xs, ys = list(xs), list(ys)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(ys[-1])


def motor_step_iec(p_kW: float) -> float:
    """Nächstgrößere typische IEC-Motorstufe (vereinfachte Liste)."""
    steps = [
        0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
        7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250
    ]
    for s in steps:
        if p_kW <= s:
            return s
    return steps[-1]


# ---------------------------
# Viskositäts-Umrechnung: viskos -> Wasseräquivalent + η-Korrektur
# (Korrelation wie in deinem bisherigen App-Ansatz; log-Basis wählbar)
# ---------------------------
def viscosity_factors_from_B(B: float, log_base: str = "log10") -> Tuple[float, float, float]:
    """
    Liefert (Cq, Ch, Ceta) aus B.
    - Für B<=1: 1.0
    - Cq~Ch über empirische Korrelation
    - Ceta über empirische Korrelation (clamped)
    """
    def _log(x: float) -> float:
        return math.log(x) if log_base == "ln" else math.log10(x)

    if B <= 1.0:
        return 1.0, 1.0, 1.0

    expo = -0.165 * (_log(B) ** 3.15)
    Cq = math.exp(expo)  # 2.71^x ~ e^x
    Ch = Cq
    Cq = min(max(Cq, 0.0), 1.0)
    Ch = min(max(Ch, 0.0), 1.0)

    Beff = min(max(B, 1.0000001), 40.0)
    Ceta = Beff ** (-(0.0547 * (Beff ** 0.69)))  # alpha=1.0
    Ceta = min(max(Ceta, 0.0), 1.0)

    return Cq, Ch, Ceta


def compute_B(Q_vis_m3h: float, H_vis_m: float, nu_cSt: float) -> float:
    Q = max(Q_vis_m3h, 1e-12)
    H = max(H_vis_m, 1e-12)
    nu = max(nu_cSt, 1e-12)
    return 280.0 * (nu ** 0.50) / ((Q ** 0.25) * (H ** 0.125))


def viscous_to_water_equivalent(
    Q_vis_m3h: float,
    H_vis_m: float,
    nu_cSt: float,
    log_base: str = "log10",
) -> dict:
    """
    Rückrechnung:
      Q_vis = Cq * Q_w  => Q_w = Q_vis / Cq
      H_vis = Ch * H_w  => H_w = H_vis / Ch
    B wird aus viskosem Arbeitspunkt gebildet.
    """
    if Q_vis_m3h < 0 or H_vis_m < 0:
        raise ValueError("Qν und Hν müssen >= 0 sein.")
    B = compute_B(Q_vis_m3h, H_vis_m, nu_cSt)
    Cq, Ch, Ceta = viscosity_factors_from_B(B, log_base=log_base)

    if Cq <= 0 or Ch <= 0:
        raise ValueError("Cq/Ch wurden <=0. Prüfe Eingaben (v.a. ν, Q, H).")

    Qw = Q_vis_m3h / Cq
    Hw = H_vis_m / Ch

    return {
        "B": B, "Cq": Cq, "Ch": Ch, "Ceta": Ceta,
        "Qw_m3h": Qw, "Hw_m": Hw
    }


# ---------------------------
# Pumpenkennlinien laden & Best-Fit Auswahl
# ---------------------------
def read_pump_curve_csv(file_bytes: bytes):
    """
    Erwartetes long-format CSV:
      pump_id,Q_m3h,H_m,eta  (eta optional)
    oder:
      pump_id,Q_m3h,H_m,P_kW (P_kW optional)

    Mindestens: pump_id,Q_m3h,H_m
    Zusätzlich: eta ODER P_kW (mindestens eins ist für Motorwahl empfohlen)
    """
    import pandas as pd
    df = pd.read_csv(io.BytesIO(file_bytes))
    required = {"pump_id", "Q_m3h", "H_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV muss Spalten {sorted(required)} enthalten. Gefunden: {list(df.columns)}")

    # normalize
    df = df.copy()
    df["pump_id"] = df["pump_id"].astype(str)
    df["Q_m3h"] = df["Q_m3h"].astype(float)
    df["H_m"] = df["H_m"].astype(float)
    if "eta" in df.columns:
        df["eta"] = df["eta"].astype(float)
    if "P_kW" in df.columns:
        df["P_kW"] = df["P_kW"].astype(float)

    # sanity: eta can be in % -> normalize if >1.5
    if "eta" in df.columns:
        if df["eta"].max() > 1.5:
            df["eta"] = df["eta"] / 100.0

    return df


def pump_point_at_Q(df_pump, Q_target: float) -> dict:
    """Interpoliert H(Q) und optional eta(Q) / P(Q) für einen Pumpensatz (eine pump_id)."""
    qs = df_pump["Q_m3h"].tolist()
    hs = df_pump["H_m"].tolist()
    H = interp1_clamped(Q_target, qs, hs)

    out = {"H_m": H}

    if "eta" in df_pump.columns:
        etas = df_pump["eta"].tolist()
        out["eta"] = interp1_clamped(Q_target, qs, etas)

    if "P_kW" in df_pump.columns:
        ps = df_pump["P_kW"].tolist()
        out["P_kW"] = interp1_clamped(Q_target, qs, ps)

    return out


def choose_best_pump(df_all, Qw: float, Hw: float, consider_eta: bool = True) -> dict:
    """
    Wählt die Pumpe mit minimalem Fehler |H(Qw) - Hw|.
    Optional: bei ähnlichem Fehler bevorzugt höhere η.
    """
    best = None
    for pid, grp in df_all.groupby("pump_id"):
        # check Q range
        qmin, qmax = grp["Q_m3h"].min(), grp["Q_m3h"].max()
        if not (qmin <= Qw <= qmax):
            continue
        pt = pump_point_at_Q(grp, Qw)
        err = abs(pt["H_m"] - Hw)

        score = err
        eta = pt.get("eta", None)
        # tie-breaker: higher eta
        if consider_eta and eta is not None:
            score = err - 0.01 * eta  # kleine Bevorzugung

        cand = {"pump_id": pid, "err": err, "score": score, **pt}
        if best is None or cand["score"] < best["score"]:
            best = cand

    if best is None:
        raise ValueError("Keine Pumpe gefunden: Qw liegt außerhalb aller Kennlinienbereiche (oder CSV leer).")
    return best


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Viskos → Wasser → Pumpenauswahl", layout="centered")
st.title("Viskoser Arbeitspunkt → Wasseräquivalent → Pumpe & Motor")

st.caption(
    "Workflow: (Qν,Hν) im Medium → Rückrechnung (Qw,Hw) → Auswahl aus Wasser-Kennlinien → Motorleistung für viskosen Betrieb."
)

# Inputs
with st.sidebar:
    st.header("Anforderung im Medium (viskos)")
    Qv = st.number_input("Qν [m³/h]", min_value=0.0, value=40.0, step=1.0)
    Hv = st.number_input("Hν [m]", min_value=0.0, value=35.0, step=1.0)

    st.header("Medium")
    mk = st.selectbox("Medium auswählen", list(MEDIA_DB.keys()), index=3)
    props = MEDIA_DB[mk]
    rho = st.number_input("ρ [kg/m³]", min_value=1.0, value=float(props.rho_kgm3), step=5.0)
    nu = st.number_input("ν [cSt]", min_value=0.1, value=float(props.nu_cSt), step=0.5)

    st.header("Umrechnung")
    log_base = st.selectbox("log(B) Basis", ["log10", "ln"], index=0)

    st.header("Motorreserve")
    reserve_pct = st.number_input("Reserve [%]", min_value=0.0, value=15.0, step=1.0)

st.subheader("1) Rückrechnung auf Wasseräquivalent")
try:
    conv = viscous_to_water_equivalent(Q_vis_m3h=float(Qv), H_vis_m=float(Hv), nu_cSt=float(nu), log_base=str(log_base))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qw [m³/h]", f"{conv['Qw_m3h']:.2f}")
    c2.metric("Hw [m]", f"{conv['Hw_m']:.2f}")
    c3.metric("B [-]", f"{conv['B']:.3f}")
    c4.metric("Cq / Ch", f"{conv['Cq']:.3f} / {conv['Ch']:.3f}")

    with st.expander("Details Umrechnung"):
        st.write({
            "B": conv["B"],
            "Cq": conv["Cq"],
            "Ch": conv["Ch"],
            "Ceta (für η-Korrektur)": conv["Ceta"],
        })
except Exception as e:
    st.error(f"Umrechnung fehlgeschlagen: {e}")
    st.stop()

st.divider()
st.subheader("2) Pumpenkennlinien (Wasser) hochladen & auswählen")

st.info(
    "CSV (long-format) mit Spalten: pump_id,Q_m3h,H_m und zusätzlich **eta** (0..1 oder %) "
    "oder **P_kW**. Mindestens eins (eta oder P_kW) wird empfohlen für die Motorwahl."
)

template = (
    "pump_id,Q_m3h,H_m,eta\n"
    "P1,10,48,0.55\nP1,20,46,0.62\nP1,30,42,0.68\nP1,40,36,0.70\nP1,50,28,0.66\n"
    "P2,10,60,0.48\nP2,20,56,0.58\nP2,30,50,0.65\nP2,40,42,0.67\nP2,50,32,0.63\n"
)
st.download_button("CSV-Template herunterladen", data=template.encode("utf-8"),
                   file_name="pump_curves_template.csv", mime="text/csv")

uploaded = st.file_uploader("Pumpenkennlinien CSV", type=["csv"])

if uploaded is None:
    st.stop()

try:
    df = read_pump_curve_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"CSV-Fehler: {e}")
    st.stop()

st.success(f"Kennlinien geladen: {df['pump_id'].nunique()} Pumpen, {len(df)} Punkte")

Qw = conv["Qw_m3h"]
Hw = conv["Hw_m"]

# choose best
try:
    best = choose_best_pump(df, Qw=Qw, Hw=Hw, consider_eta=True)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Beste passende Pumpe (Wasseräquivalent)")
c1, c2, c3 = st.columns(3)
c1.metric("Pumpe", str(best["pump_id"]))
c2.metric("H(Qw) [m]", f"{best['H_m']:.2f}")
c3.metric("|ΔH| [m]", f"{best['err']:.2f}")

if "eta" in best:
    st.write(f"Interpolierter Wasser-Wirkungsgrad ηw(Qw): **{best['eta']:.3f}**")

if "P_kW" in best:
    st.write(f"Interpolierte Wasser-Leistung Pw(Qw): **{best['P_kW']:.2f} kW**")

st.divider()
st.subheader("3) Motorleistung im viskosen Betriebspunkt")

# motor calc:
# We prefer: if eta exists -> correct efficiency and compute Pvis from Qv,Hv,rho
# else if P_kW exists -> show Pw; still need eta to correct for viscosity (otherwise rough)
# We'll do:
# - If eta exists: eta_vis = Ceta * eta_w ; Pvis from hydraulics / eta_vis
# - Else: estimate eta_vis = 0.5*(typ) and warn

warnings = []

eta_w = None
eta_vis = None

if "eta" in best:
    eta_w = float(best["eta"])
    eta_vis = max(1e-6, conv["Ceta"] * eta_w)
else:
    warnings.append("Keine η-Kennlinie vorhanden. Für saubere Motorwahl bitte η(Q) liefern. "
                    "Ich rechne ersatzweise mit η=0.5 (grobe Näherung).")
    eta_w = 0.5
    eta_vis = 0.5

Qv_m3s = float(Qv) / 3600.0
Pvis_W = (float(rho) * G * Qv_m3s * float(Hv)) / float(eta_vis)
Pvis_kW = Pvis_W / 1000.0

Pvis_kW_with_reserve = Pvis_kW * (1.0 + float(reserve_pct) / 100.0)
motor_kW = motor_step_iec(Pvis_kW_with_reserve)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ην [-]", f"{eta_vis:.3f}")
c2.metric("Pν [kW]", f"{Pvis_kW:.2f}")
c3.metric(f"Pν + {reserve_pct:.0f}% [kW]", f"{Pvis_kW_with_reserve:.2f}")
c4.metric("Motor (IEC) [kW]", f"{motor_kW:.2f}")

if warnings:
    for w in warnings:
        st.warning(w)

with st.expander("Rechengang (kurz)"):
    st.markdown(
        f"""
- Rückrechnung Wasseräquivalent:  
  \(Q_w = Q_\\nu / C_Q = {Qw:.2f}\), \(H_w = H_\\nu / C_H = {Hw:.2f}\)
- Pumpenauswahl auf Basis Wasserkennlinie: finde Pumpe mit \(H(Q_w) \\approx H_w\)
- Wirkungsgrad viskos: \(\\eta_\\nu = C_\\eta \\cdot \\eta_w = {eta_vis:.3f}\)
- Leistung viskos: \(P_\\nu = \\rho g Q_\\nu H_\\nu / \\eta_\\nu = {Pvis_kW:.2f}\\,kW\)
- Motor: \(P_\\nu\\) + Reserve ⇒ nächstgrößere IEC-Stufe
"""
    )

# Optional: show table
with st.expander("Kennlinienpunkte der ausgewählten Pumpe"):
    import pandas as pd
    df_best = df[df["pump_id"] == best["pump_id"]].sort_values("Q_m3h")
    st.dataframe(df_best, use_container_width=True)

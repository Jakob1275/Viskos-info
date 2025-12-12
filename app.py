from __future__ import annotations

import math
from dataclasses import dataclass
import streamlit as st

G = 9.80665  # m/s²


@dataclass(frozen=True)
class MediumProps:
    name: str
    rho_kgm3: float
    nu_cSt: float  # mm²/s


# Simple built-in media library (typical values; user can override in UI)
MEDIA_DB: dict[str, MediumProps] = {
    "Wasser (20°C)": MediumProps("Wasser (20°C)", rho_kgm3=998.0, nu_cSt=1.0),
    "Wasser (60°C)": MediumProps("Wasser (60°C)", rho_kgm3=983.0, nu_cSt=0.47),
    "Glykol 30% (20°C) (typ.)": MediumProps("Glykol 30% (20°C) (typ.)", rho_kgm3=1040.0, nu_cSt=3.5),
    "Hydrauliköl ISO VG 32 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 32 (40°C) (typ.)", rho_kgm3=860.0, nu_cSt=32.0),
    "Hydrauliköl ISO VG 46 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 46 (40°C) (typ.)", rho_kgm3=870.0, nu_cSt=46.0),
    "Hydrauliköl ISO VG 68 (40°C) (typ.)": MediumProps("Hydrauliköl ISO VG 68 (40°C) (typ.)", rho_kgm3=880.0, nu_cSt=68.0),
    "Heizöl EL (20°C) (typ.)": MediumProps("Heizöl EL (20°C) (typ.)", rho_kgm3=840.0, nu_cSt=6.0),
}


def viscosity_correction_iterative(
    Qw_m3h: float,
    Hw_m: float,
    eta_w: float,
    rho: float,
    nu_cSt: float,
    alpha: float = 1.0,
    log_base: str = "log10",  # "log10" or "ln"
    max_iter: int = 60,
    tol: float = 1e-8,
) -> dict:
    """
    Iterative water -> viscous conversion using equations shown in Wilo planning guide page (B, Cq~Ch, Cη).  [oai_citation:1‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)

    Notes:
      - B uses Qvis [m³/h], Hvis [m], Vvis [cSt]
      - Cq ≈ Ch ≈ (2.71)^(-0.165 * (log B)^3.15)
      - Cη = B^(-(0.0547 * alpha * B^0.69)) for 1 < B < 40 (clamped outside)
    """
    if Qw_m3h < 0 or Hw_m < 0:
        raise ValueError("Qw und Hw müssen >= 0 sein.")
    if not (0 < eta_w <= 1):
        raise ValueError("ηw muss zwischen 0 und 1 liegen.")
    if rho <= 0 or nu_cSt <= 0:
        raise ValueError("ρ und ν müssen > 0 sein.")
    if alpha <= 0:
        raise ValueError("alpha muss > 0 sein.")

    # start guess: viscous ~ water
    Qvis = max(Qw_m3h, 1e-12)
    Hvis = max(Hw_m, 1e-12)

    def _log(x: float) -> float:
        if log_base == "ln":
            return math.log(x)
        return math.log10(x)

    for _ in range(max_iter):
        # Step 1: parameter B   [oai_citation:2‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)
        B = 280.0 * (nu_cSt ** 0.50) / ((Qvis ** 0.25) * (Hvis ** 0.125))

        # Step 2: Cq ≈ Ch   [oai_citation:3‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)
        if B <= 1.0:
            Cq = 1.0
            Ch = 1.0
        else:
            expo = -0.165 * (_log(B) ** 3.15)
            Cq = math.exp(expo)  # 2.71^x ~ e^x   [oai_citation:4‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)
            Ch = Cq

            # clamp to sane range
            Cq = min(max(Cq, 0.0), 1.0)
            Ch = min(max(Ch, 0.0), 1.0)

        # water->visc update (inverse of Step 3 in the pdf)   [oai_citation:5‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)
        Qnew = Cq * Qw_m3h
        Hnew = Ch * Hw_m

        # convergence
        dq = abs(Qnew - Qvis) / max(Qvis, 1e-12)
        dh = abs(Hnew - Hvis) / max(Hvis, 1e-12)

        Qvis, Hvis = max(Qnew, 1e-12), max(Hnew, 1e-12)

        if max(dq, dh) < tol:
            break

    # Step 5: efficiency correction   [oai_citation:6‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)
    # Wilo text explicitly states it for 1 < B < 40; we clamp outside.
    B = 280.0 * (nu_cSt ** 0.50) / ((Qvis ** 0.25) * (Hvis ** 0.125))
    if B <= 1.0:
        Ceta = 1.0
    else:
        Beff = min(max(B, 1.0000001), 40.0)
        Ceta = Beff ** (-(0.0547 * alpha * (Beff ** 0.69)))
        Ceta = min(max(Ceta, 0.0), 1.0)

    eta_vis = Ceta * eta_w
    if eta_vis <= 0:
        raise ValueError("Korrigierter Wirkungsgrad <= 0. Prüfe Eingaben/alpha.")

    # Power
    Qvis_m3s = Qvis / 3600.0
    P_W = (rho * G * Qvis_m3s * Hvis) / eta_vis
    P_kW = P_W / 1000.0

    return {
        "Qw_m3h": Qw_m3h,
        "Hw_m": Hw_m,
        "eta_w": eta_w,
        "rho_kgm3": rho,
        "nu_cSt": nu_cSt,
        "B": B,
        "Cq": Cq,
        "Ch": Ch,
        "Ceta": Ceta,
        "Qvis_m3h": Qvis,
        "Hvis_m": Hvis,
        "eta_vis": eta_vis,
        "Pvis_kW": P_kW,
        "log_base": log_base,
        "alpha": alpha,
    }


# ---------------- UI ----------------
st.set_page_config(page_title="Wasser → viskos (Kreiselpumpe)", layout="centered")
st.title("Wasser → viskos umrechnen (nur Medium auswählen)")

st.caption(
    "App rechnet von Wasser-Kennlinienwerten (Qw, Hw, ηw) auf viskos (Qvis, Hvis, ηvis, Pvis) "
    "mit B-/Korrekturfaktor-Gleichungen aus dem Wilo-Guide.  [oai_citation:7‡wilo.cdn.mediamid.com](https://wilo.cdn.mediamid.com/cdndoc/wilo110215/799929/wilo110215.pdf)"
)

with st.sidebar:
    st.header("Wasser-Arbeitspunkt")
    Qw = st.number_input("Qw [m³/h]", min_value=0.0, value=50.0, step=1.0)
    Hw = st.number_input("Hw [m]", min_value=0.0, value=40.0, step=1.0)
    etaw_pct = st.number_input("ηw [%]", min_value=0.1, max_value=100.0, value=75.0, step=0.5)
    eta_w = float(etaw_pct) / 100.0

    st.header("Medium-Auswahl")
    medium_key = st.selectbox("Medium", list(MEDIA_DB.keys()), index=5)
    props = MEDIA_DB[medium_key]

    # show + allow override (still "selection only" by default)
    rho = st.number_input("ρ [kg/m³] (aus Auswahl, anpassbar)", min_value=1.0, value=float(props.rho_kgm3), step=5.0)
    nu = st.number_input("ν [cSt] (aus Auswahl, anpassbar)", min_value=0.1, value=float(props.nu_cSt), step=0.5)

    st.header("Modellparameter")
    alpha = st.number_input("α [-] (Default 1.0)", min_value=0.1, value=1.0, step=0.1,
                            help="Empirischer Faktor in Cη-Gleichung; Default 1.0 (falls nicht anders kalibriert).")
    log_base = st.selectbox("log(B) Basis", ["log10", "ln"], index=0,
                            help="Im Wilo-Guide steht 'log B'; häufig ist log10 gemeint. Du kannst hier umschalten.")

st.subheader("Ergebnis")
if st.button("Umrechnen", type="primary"):
    try:
        res = viscosity_correction_iterative(
            Qw_m3h=float(Qw),
            Hw_m=float(Hw),
            eta_w=float(eta_w),
            rho=float(rho),
            nu_cSt=float(nu),
            alpha=float(alpha),
            log_base=str(log_base),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qvis [m³/h]", f"{res['Qvis_m3h']:.2f}")
        c2.metric("Hvis [m]", f"{res['Hvis_m']:.2f}")
        c3.metric("ηvis [-]", f"{res['eta_vis']:.3f}")
        c4.metric("Pvis [kW]", f"{res['Pvis_kW']:.2f}")

        with st.expander("Korrekturfaktoren & B"):
            st.write(
                {
                    "B": round(res["B"], 6),
                    "Cq": round(res["Cq"], 6),
                    "Ch": round(res["Ch"], 6),
                    "Cη": round(res["Ceta"], 6),
                    "α": res["alpha"],
                    "log(B) Basis": res["log_base"],
                }
            )

        with st.expander("Alle Werte (JSON)"):
            st.json({k: (round(v, 10) if isinstance(v, float) else v) for k, v in res.items()})

    except Exception as e:
        st.error(str(e))

st.caption(
    "Hinweis: ν und ρ kommen aus der Medium-Auswahl (typische Richtwerte) und können bei Bedarf überschrieben werden. "
    "Für genaue Auslegung immer ν(T) des konkreten Mediums verwenden."
)

"""
PumpDesign Pro – Industrielles Pumpenauslegungstool
====================================================
Einzel-Datei-Anwendung mit externen JSON-Datendateien.

Verzeichnisstruktur:
    app.py
    data/
        pumps.json
        media.json
        mph_pumps.json
        atex_motors.json
"""

# ══════════════════════════════════════════════════════════════════
# Section 1 · Imports & Configuration
# ══════════════════════════════════════════════════════════════════
import html as html_mod
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="PumpDesign Pro", layout="wide", page_icon="🔧")

# --- Identity ---
VERSION = "3.0.0"
TOOL_NAME = "PumpDesign Pro"

# --- Physical constants ---
G = 9.81            # gravitational acceleration [m/s²]
BAR_TO_PA = 1e5     # conversion factor
P_ATM_BAR = 1.01325 # atmospheric pressure [bar]
P_NORMAL_BAR = 1.01325
T_NORMAL_K = 273.15
R_GAS_CONST = 0.08314   # ideal gas constant [bar·L / (mol·K)]

# --- Tool defaults ---
RATED_SPEED_RPM = 2900
WATER_VISCOSITY_CST = 1.0
WATER_VISCOSITY_TOL = 0.15  # treat as water if ν ≤ 1.15 cSt

# --- Data directory ---
DATA_DIR = Path(__file__).parent

# --- Plotly color palette (Viridis 10-stop hex) ---
_VIRIDIS_STOPS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
]

# --- Chart color roles ---
_CLR_PRIMARY  = "#1565C0"   # water curves / primary
_CLR_VISCOUS  = "#E53935"   # viscous curves
_CLR_SYSTEM   = "#2E7D32"   # system curve / NPSHa / motor line
_CLR_OP       = "gold"      # operating point marker
_CLR_WARN     = "#F57F17"   # warning line
_CHART_COLORS = ["#1565C0", "#E53935", "#2E7D32", "#F57F17"]

# ══════════════════════════════════════════════════════════════════
# Section 2 · Data Models
# ══════════════════════════════════════════════════════════════════
class PumpStandard(Enum):
    API_610   = "API 610 (Prozesschemie)"
    ISO_5199  = "ISO 5199 (Technische Spezifikation)"
    ISO_2858  = "ISO 2858 (Kreiselpumpen)"
    DIN_EN_733 = "DIN EN 733 (Chemienorm)"
    VDMA_24297 = "VDMA 24297"


class MaterialClass(Enum):
    # (display_name, price_factor, max_temp_c)
    CAST_IRON     = ("Grauguss GG25",            1.0,  50)
    DUCTILE_IRON  = ("Sphäroguss GGG40",         1.05, 80)
    STAINLESS_304 = ("Edelstahl 1.4301 (304)",   1.8, 300)
    STAINLESS_316 = ("Edelstahl 1.4401 (316)",   2.2, 400)
    DUPLEX        = ("Duplex 1.4462",             3.5, 500)
    HASTELLOY_C   = ("Hastelloy C-276",           8.0, 800)
    TITANIUM      = ("Titan Gr.2",               12.0, 600)


class SealType(Enum):
    # (display_name, price_factor, max_pressure_bar)
    PACKING            = ("Stopfbuchse",                        0.8, 100)
    SINGLE_MECHANICAL  = ("Einfache Gleitringdichtung",         1.0, 200)
    DOUBLE_MECHANICAL  = ("Doppelte Gleitringdichtung",         1.5, 350)
    MAGNETIC_COUPLING  = ("Magnetkupplung (dichtungslos)",      2.0, 400)


@dataclass
class ProjectInfo:
    project_id:   str = ""
    project_name: str = ""
    customer:     str = ""
    location:     str = ""
    engineer:     str = ""
    revision:     str = "A"
    date:         str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    notes:        str = ""
    standard_name: str = "ISO 5199"


@dataclass
class ProcessConditions:
    flow_m3h:            float = 0.0
    head_m:              float = 0.0
    suction_pressure_bar: float = 1.0
    discharge_pressure_bar: float = 0.0
    temperature_c:       float = 20.0
    vapor_pressure_bar:  float = 0.023
    density_kgm3:        float = 998.0
    viscosity_cst:       float = 1.0
    solids_pct:          float = 0.0
    ph_value:            float = 7.0
    chloride_ppm:        float = 0.0


@dataclass
class PipelineData:
    suction_length_m:      float = 10.0
    suction_diameter_mm:   float = 150.0
    suction_roughness_mm:  float = 0.05
    suction_fittings_k:    float = 5.0
    discharge_length_m:    float = 100.0
    discharge_diameter_mm: float = 125.0
    discharge_roughness_mm: float = 0.05
    discharge_fittings_k:  float = 15.0
    static_head_m:         float = 20.0
    geodetic_suction_m:    float = 2.0


@dataclass
class EconomicData:
    electricity_price_eur_kwh: float = 0.15
    operating_hours_yr:        int   = 8000
    lifetime_years:            int   = 15
    maintenance_pct:           float = 3.0
    discount_rate_pct:         float = 5.0
    installation_factor:       float = 1.5


# ══════════════════════════════════════════════════════════════════
# Section 3 · Data Loading
# ══════════════════════════════════════════════════════════════════
def _load_json(filename: str) -> any:
    """Load a JSON file from the data directory; raise on failure."""
    path = DATA_DIR / filename
    if not path.exists():
        st.error(f"Datendatei nicht gefunden: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_pumps() -> List[dict]:
    data = _load_json("pumps.json")
    # Normalise key names for backward-compatibility with calculation code
    pumps = []
    for p in (data or []):
        pumps.append({
            **p,
            "Qw":    p["flow_m3h"],
            "Hw":    p["head_m"],
            "eta":   p["efficiency"],
            "Pw":    p["power_kw"],
            "NPSHr": p["npsh_required_m"],
        })
    return pumps


@st.cache_data
def load_media() -> Dict[str, dict]:
    return _load_json("media.json") or {}


@st.cache_data
def load_mph_pumps() -> List[dict]:
    """Load multiphase pumps and convert JSON string-keys to int-keyed curve dicts."""
    raw = _load_json("mph_pumps.json") or []
    pumps = []
    for p in raw:
        converted = {**p}
        converted["curves_dp_vs_Q"] = {
            int(k): {"Q": v["flow_m3h"], "dp": v["dp_bar"]}
            for k, v in p["pressure_curves"].items()
        }
        converted["power_kW_vs_Q"] = {
            int(k): {"Q": v["flow_m3h"], "P": v["power_kw"]}
            for k, v in p["power_curves"].items()
        }
        pumps.append(converted)
    return pumps


@st.cache_data
def load_atex_motors() -> List[dict]:
    return _load_json("atex_motors.json") or []


# ══════════════════════════════════════════════════════════════════
# Section 4 · Utility Functions
# ══════════════════════════════════════════════════════════════════
def clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, x))
    except Exception:
        return lo


def lerp(x: float, x_arr, y_arr) -> float:
    """Piecewise-linear interpolation; clamps at boundary."""
    try:
        xs, ys = list(x_arr), list(y_arr)
        n = min(len(xs), len(ys))
        if n < 2:
            return ys[0] if ys else 0.0
        xs, ys = xs[:n], ys[:n]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(n - 1):
            if xs[i] <= x <= xs[i + 1]:
                dx = xs[i + 1] - xs[i]
                return ys[i] if dx == 0 else ys[i] + (ys[i+1] - ys[i]) * (x - xs[i]) / dx
        return ys[-1]
    except Exception:
        return (list(y_arr) or [0.0])[-1]


def trim_arrays(x, y) -> Tuple[list, list]:
    """Trim two sequences to the same (shorter) length."""
    x, y = list(x), list(y)
    n = min(len(x), len(y))
    return x[:n], y[:n]


def m3h_to_lpm(m3h: float) -> float:
    return float(m3h) * 1000.0 / 60.0


def lpm_to_m3h(lpm: float) -> float:
    return float(lpm) * 60.0 / 1000.0


def next_iec_motor(power_kw: float) -> float:
    """Return the next-larger standard IEC motor rating."""
    steps = [
        0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
        7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250, 315, 400,
    ]
    for s in steps:
        if power_kw <= s:
            return s
    return steps[-1]


def is_water(nu_cst: float) -> bool:
    return float(nu_cst) <= (WATER_VISCOSITY_CST + WATER_VISCOSITY_TOL)


# ══════════════════════════════════════════════════════════════════
# Section 5 · Gas Physics (Henry's Law)
# ══════════════════════════════════════════════════════════════════
HENRY_CONSTANTS = {
    "Luft": {"A": 800.0,  "B": 1500},
    "N2":   {"A": 900.0,  "B": 1400},
    "O2":   {"A": 700.0,  "B": 1600},
    "CO2":  {"A":  29.0,  "B": 2400},
}

AIR_COMPONENTS = [("N2", 0.79), ("O2", 0.21)]

_REAL_GAS_FACTORS = {
    "Luft": lambda p, T: 1.0,
    "N2":   lambda p, T: 1.0,
    "O2":   lambda p, T: 1.0,
    "CO2":  lambda p, T: max(0.9, 1.0 - 0.001 * (p - 1.0)),
}

_AIR_SOLUBILITY_TABLE = [
    (2.0, 36.8), (2.5, 46.0), (3.0, 55.2), (3.5, 64.4), (4.0, 73.6),
    (4.5, 82.8), (5.0, 92.0), (5.5, 101.2), (6.0, 110.4), (6.5, 119.6),
    (7.0, 128.8), (7.5, 138.0), (8.0, 147.2), (8.5, 156.4), (9.0, 165.6),
    (9.5, 177.0), (10.0, 185.0),
]
_AIR_SOL_REF_P = 5.0
_AIR_SOL_REF_C = 92.0
_AIR_SOL_REF_T = 20.0


def henry_constant(gas: str, T_c: float) -> float:
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K = float(T_c) + 273.15
    return params["A"] * math.exp(params["B"] * (1.0 / T_K - 1.0 / 298.15))


def real_gas_factor(gas: str, p_bar: float, T_c: float) -> float:
    T_K = float(T_c) + 273.15
    fn = _REAL_GAS_FACTORS.get(gas, lambda p, T: 1.0)
    return float(fn(float(p_bar), T_K))


def gas_solubility_cm3n_per_liter(gas: str, p_bar: float, T_c: float, y_gas: float = 1.0) -> float:
    """Gas solubility [cm³(N)/L] via Henry's Law with fugacity correction.

    Henry's Law: C [mol/L] = fugacity / H = p_partial * Z / H
    Z applied to p_partial acts as fugacity coefficient (enhances solubility at
    high pressure vs. ideal-gas assumption). Conversion to normal volume uses
    molar volume at normal conditions (≈22.4 L/mol at 0 °C, 1.01325 bar).
    """
    p = max(float(p_bar), 1e-6)
    T_K = float(T_c) + 273.15
    H = max(henry_constant(gas, T_c), 1e-12)
    Z = max(real_gas_factor(gas, p, T_c), 0.5)
    p_partial = clamp(float(y_gas), 0.0, 1.0) * p
    # Fugacity-corrected dissolution: f = p_partial * Z
    C_mol_l = (p_partial * Z) / H
    # Molar volume at normal conditions (Z_normal ≈ 1)
    V_molar_normal = R_GAS_CONST * T_NORMAL_K / P_NORMAL_BAR   # ≈ 22.4 L/mol
    return C_mol_l * V_molar_normal * 1000.0   # → cm³(N)/L


def _air_solubility_correction(p_bar: float, T_c: float) -> float:
    try:
        base = sum(
            gas_solubility_cm3n_per_liter(g, p_bar, _AIR_SOL_REF_T, y) for g, y in AIR_COMPONENTS
        )
        if base <= 0:
            return 1.0
        p_vals = [r[0] for r in _AIR_SOLUBILITY_TABLE]
        c_vals = [r[1] for r in _AIR_SOLUBILITY_TABLE]
        ref_raw = lerp(float(p_bar), p_vals, c_vals)
        ref_at_5 = lerp(_AIR_SOL_REF_P, p_vals, c_vals)
        scale = (_AIR_SOL_REF_C / ref_at_5) if ref_at_5 > 0 else 1.0
        return (ref_raw * scale / base) if base > 0 else 1.0
    except Exception:
        return 1.0


def air_solubility_cm3n_per_liter(p_bar: float, T_c: float) -> float:
    total = sum(
        gas_solubility_cm3n_per_liter(g, p_bar, T_c, y) for g, y in AIR_COMPONENTS
    )
    return total * _air_solubility_correction(p_bar, T_c)


def total_gas_solubility(gas: str, p_bar: float, T_c: float) -> float:
    if gas == "Luft":
        return air_solubility_cm3n_per_liter(p_bar, T_c)
    return gas_solubility_cm3n_per_liter(gas, p_bar, T_c, y_gas=1.0)


# ══════════════════════════════════════════════════════════════════
# Section 6 · Hydraulic Calculations
# ══════════════════════════════════════════════════════════════════
def reynolds_number(v_ms: float, d_m: float, nu_m2s: float) -> float:
    return (v_ms * d_m) / max(nu_m2s, 1e-9)


def friction_factor(Re: float, d_m: float, roughness_mm: float) -> float:
    """Colebrook-White friction factor; uses Blasius in laminar regime."""
    if Re < 2300:
        return 64.0 / max(Re, 1e-6)
    k = roughness_mm / 1000.0
    _arg = k / (3.7 * d_m) + 5.74 / (Re ** 0.9)
    f = 0.25 / (math.log10(max(_arg, 1e-10)) ** 2) if _arg > 0 else 0.02
    f = max(f, 1e-6)   # guard against non-positive initial value
    for _ in range(20):
        rhs = -2.0 * math.log10(k / (3.7 * d_m) + 2.51 / (Re * math.sqrt(max(f, 1e-9))))
        f_new = (1.0 / rhs ** 2) if rhs != 0 else f
        if abs(f_new - f) < 1e-8:
            break
        f = f_new
    return f


def pipe_head_loss(
    Q_m3h: float, length_m: float, diameter_mm: float,
    roughness_mm: float, k_fittings: float,
    density_kgm3: float, viscosity_cst: float,
) -> float:
    """Darcy-Weisbach head loss [m] including fitting losses."""
    if Q_m3h <= 0 or length_m <= 0 or diameter_mm <= 0:
        return 0.0
    Q_m3s = Q_m3h / 3600.0
    d = diameter_mm / 1000.0
    A = math.pi * (d / 2) ** 2
    v = Q_m3s / max(A, 1e-9)
    Re = reynolds_number(v, d, viscosity_cst * 1e-6)
    f = friction_factor(Re, d, roughness_mm)
    h_pipe     = f * (length_m / d) * v ** 2 / (2 * G)
    h_fittings = k_fittings * v ** 2 / (2 * G)
    return h_pipe + h_fittings


def system_curve(pipeline: PipelineData, process: ProcessConditions, Q_range) -> List[float]:
    """System head [m] for each flow in Q_range."""
    return [
        pipeline.static_head_m
        + pipe_head_loss(Q, pipeline.suction_length_m, pipeline.suction_diameter_mm,
                         pipeline.suction_roughness_mm, pipeline.suction_fittings_k,
                         process.density_kgm3, process.viscosity_cst)
        + pipe_head_loss(Q, pipeline.discharge_length_m, pipeline.discharge_diameter_mm,
                         pipeline.discharge_roughness_mm, pipeline.discharge_fittings_k,
                         process.density_kgm3, process.viscosity_cst)
        for Q in Q_range
    ]


def npsh_available(pipeline: PipelineData, process: ProcessConditions, Q_m3h: float) -> float:
    """NPSHa [m] at the specified flow rate."""
    h_loss = pipe_head_loss(
        Q_m3h, pipeline.suction_length_m, pipeline.suction_diameter_mm,
        pipeline.suction_roughness_mm, pipeline.suction_fittings_k,
        process.density_kgm3, process.viscosity_cst,
    )
    p_term = (P_ATM_BAR - process.vapor_pressure_bar) * BAR_TO_PA / (process.density_kgm3 * G)
    return max(p_term + pipeline.geodetic_suction_m - h_loss, 0.0)


# ══════════════════════════════════════════════════════════════════
# Section 7 · Viscosity Correction (HI Method)
# ══════════════════════════════════════════════════════════════════
def compute_B_hi(Q_m3h: float, H_m: float, nu_cst: float) -> float:
    """HI viscosity parameter B."""
    Q_gpm = max(float(Q_m3h), 1e-6) * 4.40287
    H_ft  = max(float(H_m),   1e-6) * 3.28084
    nu    = max(float(nu_cst),  1e-6)
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))


def viscosity_factors(B: float) -> Tuple[float, float]:
    """Head correction CH and efficiency correction Cη from parameter B."""
    if B <= 1.0:
        return 1.0, 1.0
    log_B = math.log10(B)
    CH   = clamp(math.exp(-0.165 * (log_B ** 2.2)), 0.3, 1.0)
    Ceta = clamp(1.0 - 0.25 * log_B - 0.05 * log_B ** 2, 0.1, 1.0)
    return CH, Ceta


def viscous_to_water_equivalent(Q_vis: float, H_vis: float, nu_cst: float) -> dict:
    """Convert viscous operating point to water-equivalent for pump selection."""
    B = compute_B_hi(Q_vis, H_vis, nu_cst)
    CH, Ceta = (1.0, 1.0) if is_water(nu_cst) else viscosity_factors(B)
    return {
        "Q_water": float(Q_vis),
        "H_water": float(H_vis) / max(CH, 1e-9),
        "B": B, "CH": CH, "Ceta": Ceta,
    }


def build_viscous_curves(pump: dict, nu_cst: float, density_kgm3: float) -> Tuple[list, list, list, list]:
    """Generate viscosity-corrected Q-H, Q-η, Q-P curves."""
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    ew = np.array(pump["eta"], dtype=float)
    H_vis, eta_vis, P_vis = [], [], []

    for q, h, e in zip(Qw, Hw, ew):
        B = compute_B_hi(max(q, 1e-6), max(h, 1e-6), nu_cst)
        CH, Ceta = (1.0, 1.0) if is_water(nu_cst) else viscosity_factors(B)
        hv = float(h) * max(CH, 1e-9)
        ev = clamp(float(e) * Ceta, 0.05, 0.95)
        P_hyd_W = density_kgm3 * G * (float(q) / 3600.0) * hv
        pv = (P_hyd_W / max(ev, 1e-9)) / 1000.0
        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)

    return Qw.tolist(), H_vis, eta_vis, P_vis


# ══════════════════════════════════════════════════════════════════
# Section 8 · Speed & Affinity Laws
# ══════════════════════════════════════════════════════════════════
def _bisect(f, a: float, b: float, iterations: int = 70, tol: float = 1e-6) -> Optional[float]:
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        return None
    lo, hi, flo = a, b, fa
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm) or abs(fm) < tol:
            return mid
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def find_speed_ratio(
    Q_curve: list, H_curve: list,
    Q_req: float, H_req: float,
    n_min: float = 0.5, n_max: float = 1.2,
) -> Optional[float]:
    """Find speed ratio n/n₀ such that the pump delivers (Q_req, H_req)."""
    Q_curve = list(map(float, Q_curve))
    H_curve = list(map(float, H_curve))

    def residual(nr: float) -> float:
        if nr <= 0:
            return 1e9
        H_base = lerp(Q_req / nr, Q_curve, H_curve)
        return H_base * nr ** 2 - H_req

    return _bisect(residual, float(n_min), float(n_max))


# ══════════════════════════════════════════════════════════════════
# Section 9 · Economic Analysis (LCC)
# ══════════════════════════════════════════════════════════════════
def compute_lcc(
    pump_price: float, P_shaft_kw: float, motor_eta: float,
    econ: EconomicData,
    material_factor: float = 1.0, seal_factor: float = 1.0,
) -> dict:
    """Life-Cycle Cost calculation following Europump methodology.

    motor_eta: motor efficiency (not pump efficiency). Converts shaft power
    to electrical input power: P_el = P_shaft / motor_eta.
    Typical range 0.88–0.96 depending on IEC motor size.
    """
    pump_cost     = pump_price * material_factor * seal_factor
    install_cost  = pump_cost * econ.installation_factor
    initial_cost  = pump_cost + install_cost

    P_actual_kw       = P_shaft_kw / max(motor_eta, 0.1)
    annual_energy_kwh = P_actual_kw * econ.operating_hours_yr
    annual_energy_eur = annual_energy_kwh * econ.electricity_price_eur_kwh
    annual_maint_eur  = pump_cost * (econ.maintenance_pct / 100.0)

    r = econ.discount_rate_pct / 100.0
    n = econ.lifetime_years
    annuity = ((1 - (1 + r) ** (-n)) / r) if r > 0 else float(n)

    npv_energy = annual_energy_eur * annuity
    npv_maint  = annual_maint_eur  * annuity
    lcc_total  = initial_cost + npv_energy + npv_maint

    return {
        "pump_cost":        pump_cost,
        "install_cost":     install_cost,
        "initial_cost":     initial_cost,
        "annual_energy_eur": annual_energy_eur,
        "annual_maint_eur":  annual_maint_eur,
        "npv_energy":       npv_energy,
        "npv_maint":        npv_maint,
        "lcc_total":        lcc_total,
        "energy_share_pct": (npv_energy / lcc_total * 100) if lcc_total > 0 else 0,
        "co2_annual_kg":    annual_energy_kwh * 0.4,
    }


# ══════════════════════════════════════════════════════════════════
# Section 10 · Standards Compliance & Material Recommendation
# ══════════════════════════════════════════════════════════════════
def check_compliance(pump: dict, process: ProcessConditions, standard: PumpStandard) -> List[dict]:
    issues = []

    if standard == PumpStandard.API_610:
        if not pump.get("api_610_compliant", False):
            issues.append({"severity": "error", "msg": "Pumpe nicht API 610 konform"})
        eta_list = pump.get("eta", [])
        Q_list   = pump.get("Qw", [])
        if eta_list and Q_list:
            bep_idx = eta_list.index(max(eta_list))
            Q_bep   = Q_list[bep_idx]
            Q_lo, Q_hi = Q_bep * 0.7, Q_bep * 1.2
            if not (Q_lo <= process.flow_m3h <= Q_hi):
                issues.append({
                    "severity": "warning",
                    "msg": f"Betriebspunkt außerhalb API 610-Fenster (70–120% BEP: {Q_lo:.1f}–{Q_hi:.1f} m³/h)",
                })
    elif standard == PumpStandard.ISO_5199:
        if not pump.get("iso_5199_compliant", False):
            issues.append({"severity": "error", "msg": "Pumpe nicht ISO 5199 konform"})

    if process.temperature_c > pump.get("max_temp_c", 120):
        issues.append({"severity": "error", "msg": f"Temperatur {process.temperature_c}°C > max. {pump.get('max_temp_c', 120)}°C"})
    if process.temperature_c < pump.get("min_temp_c", -20):
        issues.append({"severity": "error", "msg": f"Temperatur {process.temperature_c}°C < min. {pump.get('min_temp_c', -20)}°C"})
    if process.viscosity_cst > pump.get("max_viscosity_cst", 200):
        issues.append({"severity": "warning", "msg": f"Viskosität {process.viscosity_cst} cSt > empfohlen {pump.get('max_viscosity_cst', 200)} cSt"})

    return issues


def recommend_materials(process: ProcessConditions) -> List[str]:
    recs: List[str] = []
    if process.ph_value < 4 or process.ph_value > 10:
        recs += ["STAINLESS_316", "HASTELLOY_C"]
    if process.chloride_ppm > 200:
        recs += ["DUPLEX", "TITANIUM"]
    elif process.chloride_ppm > 50:
        recs.append("STAINLESS_316")
    if process.temperature_c > 150:
        recs += ["STAINLESS_316", "HASTELLOY_C"]
    if process.solids_pct > 5:
        recs.append("DUCTILE_IRON")
    return list(set(recs)) if recs else ["CAST_IRON", "STAINLESS_304"]


# ══════════════════════════════════════════════════════════════════
# Section 11 · Pump Selection
# ══════════════════════════════════════════════════════════════════
def select_best_pump(
    pumps: List[dict], Q_req: float, H_req: float,
    nu_cst: float, density_kgm3: float,
    allow_out_of_range: bool = True,
) -> Optional[dict]:
    best = None
    for p in pumps:
        try:
            if nu_cst > p.get("max_viscosity_cst", 500):
                continue
            if density_kgm3 > p.get("max_density_kgm3", 1200):
                continue
            q_min, q_max = min(p["Qw"]), max(p["Qw"])
            in_range = q_min <= Q_req <= q_max
            if not in_range and not allow_out_of_range:
                continue
            Q_eval = clamp(Q_req, q_min, q_max)
            H_at   = lerp(Q_eval, p["Qw"], p["Hw"])
            eta_at = lerp(Q_eval, p["Qw"], p["eta"])
            penalty = 0.0 if in_range else abs(Q_req - Q_eval) / max(q_max - q_min, 1e-9) * 10.0
            score = abs(H_at - H_req) + penalty
            candidate = {
                "id": p["id"], "pump": p, "in_range": in_range,
                "Q_eval": Q_eval, "H_at": H_at, "eta_at": eta_at, "score": score,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate
        except Exception:
            continue
    return best


# --- Multiphase helpers ---
def _gvf_bracket(pump: dict, gvf_pct: float) -> Tuple[int, int, float]:
    keys = sorted(pump["curves_dp_vs_Q"].keys())
    if gvf_pct <= keys[0]:
        return keys[0], keys[0], 0.0
    if gvf_pct >= keys[-1]:
        return keys[-1], keys[-1], 0.0
    lo = max(k for k in keys if k <= gvf_pct)
    hi = min(k for k in keys if k >= gvf_pct)
    w = (gvf_pct - lo) / (hi - lo) if hi != lo else 0.0
    return lo, hi, w


def dp_at_operating_point(pump: dict, Q_m3h: float, gvf_pct: float) -> Tuple[float, int, int, float]:
    lo, hi, w = _gvf_bracket(pump, gvf_pct)
    c_lo, c_hi = pump["curves_dp_vs_Q"][lo], pump["curves_dp_vs_Q"][hi]
    dp = (1 - w) * lerp(Q_m3h, c_lo["Q"], c_lo["dp"]) + w * lerp(Q_m3h, c_hi["Q"], c_hi["dp"])
    return dp, lo, hi, w


def power_at_operating_point(pump: dict, Q_m3h: float, gvf_pct: float) -> Tuple[float, int, int, float]:
    lo, hi, w = _gvf_bracket(pump, gvf_pct)
    p_lo, p_hi = pump["power_kW_vs_Q"][lo], pump["power_kW_vs_Q"][hi]
    P = (1 - w) * lerp(Q_m3h, p_lo["Q"], p_lo["P"]) + w * lerp(Q_m3h, p_hi["Q"], p_hi["P"])
    return P, lo, hi, w


# ══════════════════════════════════════════════════════════════════
# Section 12 · Export
# ══════════════════════════════════════════════════════════════════
def render_datasheet_html(
    project: ProjectInfo, process: ProcessConditions,
    pump: dict, results: dict,
) -> str:
    # Escape all user-controlled strings to prevent XSS (CWE-79)
    esc = html_mod.escape
    pump_id  = esc(str(pump.get('id', 'N/A')))
    pump_mfr = esc(str(pump.get('manufacturer', '')))
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <title>Datenblatt – {pump_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 210mm; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #2c5aa0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; width: 40%; }}
    .footer {{ color: #888; font-size: 0.85em; margin-top: 20px; }}
  </style>
</head>
<body>
<h1>Technisches Datenblatt</h1>
<p>
  <strong>Projekt:</strong> {esc(str(project.project_id))} &nbsp;|&nbsp;
  <strong>Rev.:</strong> {esc(str(project.revision))} &nbsp;|&nbsp;
  <strong>Datum:</strong> {esc(str(project.date))} &nbsp;|&nbsp;
  <strong>Bearbeiter:</strong> {esc(str(project.engineer))}
</p>
<h2>Prozessdaten</h2>
<table>
  <tr><th>Förderstrom</th><td>{process.flow_m3h:.2f} m³/h</td></tr>
  <tr><th>Förderhöhe</th><td>{process.head_m:.2f} m</td></tr>
  <tr><th>Temperatur</th><td>{process.temperature_c:.1f} °C</td></tr>
  <tr><th>Dichte</th><td>{process.density_kgm3:.1f} kg/m³</td></tr>
  <tr><th>Viskosität</th><td>{process.viscosity_cst:.2f} cSt</td></tr>
  <tr><th>Dampfdruck</th><td>{process.vapor_pressure_bar:.4f} bar(a)</td></tr>
</table>
<h2>Pumpe: {pump_id} – {pump_mfr}</h2>
<table>
  <tr><th>Typ</th><td>{pump.get('pump_type','–')}</td></tr>
  <tr><th>Wirkungsgrad</th><td>{results.get('eta', 0)*100:.1f} %</td></tr>
  <tr><th>Wellenleistung</th><td>{results.get('P_shaft_kw', 0):.2f} kW</td></tr>
  <tr><th>Motorleistung (IEC)</th><td>{results.get('P_motor_kw', 0):.1f} kW</td></tr>
  <tr><th>NPSHr</th><td>{results.get('NPSHr', 0):.2f} m</td></tr>
  <tr><th>NPSHa</th><td>{results.get('NPSHa', 0):.2f} m</td></tr>
  <tr><th>NPSH-Reserve</th><td>{results.get('NPSHa', 0) - results.get('NPSHr', 0):.2f} m</td></tr>
</table>
<p class="footer">Erstellt mit {TOOL_NAME} v{VERSION} am {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body>
</html>"""


def export_project_json(
    project: ProjectInfo, process: ProcessConditions,
    pipeline: PipelineData, econ: EconomicData,
    results: dict,
) -> str:
    payload = {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "project":  asdict(project),
        "process":  asdict(process),
        "pipeline": asdict(pipeline),
        "economic": asdict(econ),
        "results":  results,
    }
    return json.dumps(payload, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════
# Section 12b · UI Helpers
# ══════════════════════════════════════════════════════════════════

def _inject_css():
    """Inject custom CSS for the professional engineering theme."""
    st.markdown("""
<style>
/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border-left: 4px solid #1565C0;
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
}
[data-testid="metric-container"] label {
    text-transform: uppercase;
    font-size: 0.70rem;
    letter-spacing: 0.06em;
    color: #555;
}

/* ── Page header ── */
.pumpdesign-header {
    padding: 18px 0 10px 0;
    border-bottom: 3px solid #1565C0;
    margin-bottom: 18px;
}
.pumpdesign-header h1 {
    margin: 0;
    font-size: 1.8rem;
    color: #1A1A2E;
}
.pumpdesign-header p {
    margin: 4px 0 0 0;
    color: #555;
    font-size: 0.92rem;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-ok      { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
.badge-warning { background: #FFF8E1; color: #F57F17; border: 1px solid #FFE082; }
.badge-error   { background: #FFEBEE; color: #C62828; border: 1px solid #EF9A9A; }
.badge-info    { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1A2744 !important;
}
[data-testid="stSidebar"] * {
    color: #E8EDF5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2E4070 !important;
}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    overflow: hidden;
}

/* ── Section subheaders ── */
h3 {
    color: #1565C0;
}
</style>
""", unsafe_allow_html=True)


def _page_header(icon: str, title: str, subtitle: str = ""):
    """Render a styled page header with accent underline."""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="pumpdesign-header">'
        f'<h1>{icon} {title}</h1>{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, unit: str = "", status: str = "neutral", delta: str = ""):
    """Render a metric card with colored left border via native st.metric."""
    display = f"{value} {unit}".strip()
    st.metric(label=label, value=display, delta=delta if delta else None)


# ══════════════════════════════════════════════════════════════════
# Section 13 · Session State
# ══════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "project":    ProjectInfo(),
        "process":    ProcessConditions(),
        "pipeline":   PipelineData(),
        "economic":   EconomicData(),
        "selected_pump": None,
        "calc_results":  {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="padding:12px 0 8px 0;">'
            f'<span style="font-size:1.4rem;font-weight:700;color:#E8EDF5;">🔧 {TOOL_NAME}</span>'
            f'<span style="margin-left:8px;background:#1565C0;color:#fff;font-size:0.68rem;'
            f'padding:2px 7px;border-radius:10px;vertical-align:middle;">v{VERSION}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        with st.expander("📋 Projektdaten", expanded=False):
            p = st.session_state.project
            p.project_id   = st.text_input("Projekt-ID",   value=p.project_id,   placeholder="PRJ-2024-001")
            p.project_name = st.text_input("Projektname",  value=p.project_name)
            p.customer     = st.text_input("Kunde",         value=p.customer)
            p.engineer     = st.text_input("Bearbeiter",    value=p.engineer)
            p.revision     = st.text_input("Revision",      value=p.revision)
            p.standard_name = st.selectbox("Auslegungsnorm", [s.value for s in PumpStandard], index=1)

        st.divider()
        _page_icons = {
            "Einphasenpumpen":  "⚙️",
            "Mehrphasenpumpen": "🌊",
            "ATEX-Auslegung":   "⚡",
            "Pumpenvergleich":  "⚖️",
            "Dokumentation":    "📚",
        }
        page = st.radio(
            "Navigation",
            list(_page_icons.keys()),
            format_func=lambda p: f"{_page_icons[p]} {p}",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            f'<span style="font-size:0.72rem;color:#8899BB;">© {datetime.now().year} {TOOL_NAME}</span>',
            unsafe_allow_html=True,
        )
    return page


# ══════════════════════════════════════════════════════════════════
# Section 14 · Page: Single-Phase Pump
# ══════════════════════════════════════════════════════════════════
def _render_process_tab(media: dict):
    """Render process inputs; return (Q, H, nu, rho, T, p_vap, reserve_pct, n_min, n_max)."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Betriebspunkt")
        Q = st.number_input("Förderstrom Q [m³/h]", min_value=0.1, value=30.0, step=1.0)
        H = st.number_input("Förderhöhe H [m]",     min_value=0.1, value=20.0, step=1.0)

    with col2:
        st.subheader("Medium")
        medium = st.selectbox("Medium", list(media.keys()))
        rho   = st.number_input("Dichte ρ [kg/m³]",   min_value=1.0,
                                 value=float(media[medium]["density_kgm3"]), step=5.0)
        nu    = st.number_input("Viskosität ν [cSt]", min_value=0.1,
                                 value=float(media[medium]["viscosity_cst"]), step=0.5)
        T     = st.number_input("Temperatur [°C]",     min_value=-60.0, max_value=400.0, value=20.0)
        p_vap = st.number_input("Dampfdruck [bar(a)]", min_value=0.0,
                                 value=float(media[medium].get("vapor_pressure_bar", 0.023)),
                                 step=0.01, format="%.4f")

    with col3:
        st.subheader("Optionen")
        allow_out    = st.checkbox("Auswahl außerhalb Kennlinie", value=True)
        reserve_pct  = st.slider("Motorreserve [%]", 0, 30, 10)
        n_min        = st.slider("n_min / n₀", 0.4, 1.0, 0.6, 0.01)
        n_max        = st.slider("n_max / n₀", 1.0, 1.6, 1.2, 0.01)

    return Q, H, nu, rho, T, p_vap, allow_out, reserve_pct, n_min, n_max


def _render_pipeline_tab():
    st.subheader("Rohrleitungsdaten")
    pl = st.session_state.pipeline
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Saugseite**")
        pl.suction_length_m    = st.number_input("Länge [m]",          min_value=0.0, value=pl.suction_length_m,    key="sl")
        pl.suction_diameter_mm = st.number_input("Durchmesser [mm]",   min_value=10.0, value=pl.suction_diameter_mm, key="sd")
        pl.suction_fittings_k  = st.number_input("Σk Armaturen",       min_value=0.0, value=pl.suction_fittings_k,  key="sk")
        pl.geodetic_suction_m  = st.number_input("Geodät. Saughöhe [m]", value=pl.geodetic_suction_m,               key="sg")
    with col2:
        st.markdown("**Druckseite**")
        pl.discharge_length_m    = st.number_input("Länge [m]",        min_value=0.0, value=pl.discharge_length_m,    key="dl")
        pl.discharge_diameter_mm = st.number_input("Durchmesser [mm]", min_value=10.0, value=pl.discharge_diameter_mm, key="dd")
        pl.discharge_fittings_k  = st.number_input("Σk Armaturen",     min_value=0.0, value=pl.discharge_fittings_k,  key="dk")
        pl.static_head_m         = st.number_input("Statische Höhe [m]", min_value=0.0, value=pl.static_head_m,        key="ds")


def _render_economic_tab():
    st.subheader("Wirtschaftlichkeit (LCC)")
    e = st.session_state.economic
    col1, col2, col3 = st.columns(3)
    with col1:
        e.electricity_price_eur_kwh = st.number_input("Strompreis [€/kWh]",    min_value=0.01, value=e.electricity_price_eur_kwh, step=0.01)
        e.operating_hours_yr        = st.number_input("Betriebsstunden [h/a]", min_value=100,  value=e.operating_hours_yr,         step=100)
    with col2:
        e.lifetime_years    = st.number_input("Lebensdauer [Jahre]",    min_value=1,   value=e.lifetime_years)
        e.maintenance_pct   = st.number_input("Wartung [%/a]",          min_value=0.0, value=e.maintenance_pct, step=0.5)
    with col3:
        e.discount_rate_pct   = st.number_input("Diskontierungszins [%]", min_value=0.0, value=e.discount_rate_pct, step=0.5)
        e.installation_factor = st.number_input("Installationsfaktor",   min_value=1.0, value=e.installation_factor, step=0.1)


def _render_single_phase_results(Q_vis, H_vis, nu, rho, reserve_pct, n_min, n_max, pumps):
    """Calculate and render results for single-phase pump selection."""
    conv    = viscous_to_water_equivalent(Q_vis, H_vis, nu)
    Q_water = conv["Q_water"]
    H_water = conv["H_water"]
    CH, Ceta = conv["CH"], conv["Ceta"]

    best = select_best_pump(pumps, Q_water, H_water, nu, rho,
                            allow_out_of_range=st.session_state.get("_allow_out", True))
    if not best:
        st.error("❌ Keine geeignete Pumpe für die angegebenen Bedingungen gefunden.")
        return

    pump     = best["pump"]
    eta_water = float(best["eta_at"])
    eta_vis   = clamp(eta_water * Ceta, 0.05, 0.95)
    P_hyd_W   = rho * G * (Q_vis / 3600.0) * H_vis
    P_shaft   = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
    P_motor   = next_iec_motor(P_shaft * (1.0 + reserve_pct / 100.0))

    NPSHa = npsh_available(st.session_state.pipeline, st.session_state.process, Q_vis)
    NPSHr = lerp(Q_vis, pump["Qw"], pump.get("NPSHr", [2.0] * len(pump["Qw"])))
    NPSH_margin = NPSHa - NPSHr

    Qv_curve, Hv_curve, etav_curve, Pv_curve = build_viscous_curves(pump, nu, rho)
    n_ratio = find_speed_ratio(Qv_curve, Hv_curve, Q_vis, H_vis, n_min, n_max)

    results = {
        "Q_op": Q_vis, "H_op": H_vis,
        "eta": eta_vis, "P_shaft_kw": P_shaft, "P_motor_kw": P_motor,
        "NPSHa": NPSHa, "NPSHr": NPSHr,
    }
    st.session_state.calc_results = results
    st.session_state.selected_pump = pump

    # ---- Norm compliance ----
    standard = PumpStandard.ISO_5199
    for s in PumpStandard:
        if s.value == st.session_state.project.standard_name:
            standard = s
            break
    for issue in check_compliance(pump, st.session_state.process, standard):
        if issue["severity"] == "error":
            st.error(issue["msg"])
        else:
            st.warning(issue["msg"])

    # ---- Key metrics ----
    st.subheader("📊 Ergebnisse")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pumpe", pump["id"])
    c2.metric("Wirkungsgrad",  f"{eta_vis*100:.1f} %")
    c2.metric("Wellenleistung", f"{P_shaft:.2f} kW")
    c3.metric("Motorleistung",  f"{P_motor:.1f} kW")
    if n_ratio:
        c3.metric("Opt. Drehzahl", f"{RATED_SPEED_RPM * n_ratio:.0f} rpm")
    c4.metric("NPSHa", f"{NPSHa:.2f} m")
    c4.metric("NPSHr", f"{NPSHr:.2f} m")
    if NPSH_margin >= 1.0:
        c5.success(f"NPSH-Reserve: {NPSH_margin:.2f} m ✓")
    elif NPSH_margin >= 0.5:
        c5.warning(f"NPSH-Reserve: {NPSH_margin:.2f} m")
    else:
        c5.error(f"NPSH-Reserve: {NPSH_margin:.2f} m ⚠ KAVITATION")

    if not is_water(nu):
        st.info(f"Viskositätskorrektur (HI): B = {conv['B']:.2f} → CH = {CH:.3f}, Cη = {Ceta:.3f}")

    # ---- Material & Seal selection ----
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔩 Werkstoff")
        rec_mats  = recommend_materials(st.session_state.process)
        avail_mats = pump.get("materials_available", ["CAST_IRON"])
        suitable  = [m for m in rec_mats if m in avail_mats] or avail_mats[:1]
        sel_mat   = st.selectbox("Werkstoff", suitable,
                                  format_func=lambda x: MaterialClass[x].value[0] if x in MaterialClass.__members__ else x)
    with col2:
        st.subheader("🔧 Dichtung")
        avail_seals = pump.get("seals_available", ["SINGLE_MECHANICAL"])
        sel_seal    = st.selectbox("Dichtung", avail_seals,
                                    format_func=lambda x: SealType[x].value[0] if x in SealType.__members__ else x)

    # ---- LCC ----
    st.divider()
    st.subheader("💰 Lebenszykluskosten (LCC)")
    mat_f  = MaterialClass[sel_mat].value[1]  if sel_mat  in MaterialClass.__members__  else 1.0
    seal_f = SealType[sel_seal].value[1]       if sel_seal in SealType.__members__       else 1.0
    lcc = compute_lcc(pump.get("price_eur", 5000), P_shaft, 0.93,
                      st.session_state.economic, mat_f, seal_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investition",    f"{lcc['initial_cost']:,.0f} €")
    c2.metric("Energie (NPV)",  f"{lcc['npv_energy']:,.0f} €")
    c3.metric("Wartung (NPV)",  f"{lcc['npv_maint']:,.0f} €")
    c4.metric("LCC Gesamt",     f"{lcc['lcc_total']:,.0f} €")
    st.caption(f"Energieanteil: {lcc['energy_share_pct']:.0f}% | CO₂ ca. {lcc['co2_annual_kg']:,.0f} kg/a")

    # ---- Curves ----
    st.divider()
    st.subheader("📈 Kennlinien")

    q_vis_pos = [q for q in Qv_curve if q > 0]
    h_vis_pos = [h for q, h in zip(Qv_curve, Hv_curve) if q > 0]
    e_vis_pos = [e * 100 for q, e in zip(Qv_curve, etav_curve) if q > 0]
    p_vis_pos = [p for q, p in zip(Qv_curve, Pv_curve) if q > 0]
    Q_rng     = np.linspace(0.1, max(pump["Qw"]) * 1.2, 60)
    H_sys     = system_curve(st.session_state.pipeline, st.session_state.process, Q_rng)
    NPSHr_curve = pump.get("NPSHr", [2.0] * len(pump["Qw"]))
    NPSHa_curve = [npsh_available(st.session_state.pipeline, st.session_state.process, q) for q in pump["Qw"]]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Q-H Kennlinie", "Q-η Kennlinie", "Q-P Kennlinie", "NPSH"),
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    # (1,1) Q-H
    fig.add_trace(go.Scatter(x=pump["Qw"], y=pump["Hw"], mode="lines+markers",
        name="Wasser", line=dict(color=_CLR_PRIMARY, width=2),
        marker=dict(size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=q_vis_pos, y=h_vis_pos, mode="lines+markers",
        name=f"Viskos ν={nu:.1f} cSt", line=dict(color=_CLR_VISCOUS, width=2, dash="dash"),
        marker=dict(symbol="square", size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(Q_rng), y=H_sys, mode="lines",
        name="Anlage", line=dict(color=_CLR_SYSTEM, width=2, dash="dot"),
        showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[Q_vis], y=[H_vis], mode="markers",
        name="Betriebspunkt", marker=dict(symbol="star", size=16, color=_CLR_OP,
        line=dict(color="#333", width=1))), row=1, col=1)

    # (1,2) Q-η
    fig.add_trace(go.Scatter(x=pump["Qw"], y=[e * 100 for e in pump["eta"]], mode="lines+markers",
        name="Wasser (η)", line=dict(color=_CLR_PRIMARY, width=2),
        marker=dict(size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=q_vis_pos, y=e_vis_pos, mode="lines+markers",
        name="Viskos (η)", line=dict(color=_CLR_VISCOUS, width=2, dash="dash"),
        marker=dict(symbol="square", size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[Q_vis], y=[eta_vis * 100], mode="markers",
        name="OP (η)", marker=dict(symbol="star", size=16, color=_CLR_OP,
        line=dict(color="#333", width=1)), showlegend=False), row=1, col=2)

    # (2,1) Q-P
    fig.add_trace(go.Scatter(x=pump["Qw"], y=pump["Pw"], mode="lines+markers",
        name="Wasser (P)", line=dict(color=_CLR_PRIMARY, width=2),
        marker=dict(size=6), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=q_vis_pos, y=p_vis_pos, mode="lines+markers",
        name="Viskos (P)", line=dict(color=_CLR_VISCOUS, width=2, dash="dash"),
        marker=dict(symbol="square", size=6), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[min(pump["Qw"]), max(pump["Qw"])],
        y=[P_motor, P_motor], mode="lines",
        name=f"Motor {P_motor:.1f} kW",
        line=dict(color=_CLR_SYSTEM, width=2, dash="longdash"),
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[Q_vis], y=[P_shaft], mode="markers",
        name="OP (P)", marker=dict(symbol="star", size=16, color=_CLR_OP,
        line=dict(color="#333", width=1)), showlegend=False), row=2, col=1)

    # (2,2) NPSH
    fig.add_trace(go.Scatter(x=pump["Qw"], y=NPSHr_curve, mode="lines+markers",
        name="NPSHr", line=dict(color=_CLR_VISCOUS, width=2),
        marker=dict(size=6), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=pump["Qw"], y=NPSHa_curve, mode="lines",
        name="NPSHa", line=dict(color=_CLR_SYSTEM, width=2),
        fill="tonexty", fillcolor="rgba(46,125,50,0.10)", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[Q_vis], y=[NPSHr], mode="markers",
        name="OP NPSHr", marker=dict(symbol="star", size=14, color=_CLR_VISCOUS,
        line=dict(color="#333", width=1)), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[Q_vis], y=[NPSHa], mode="markers",
        name="OP NPSHa", marker=dict(symbol="star", size=14, color=_CLR_SYSTEM,
        line=dict(color="#333", width=1)), showlegend=False), row=2, col=2)

    fig.update_layout(
        template="plotly_white", height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40),
        paper_bgcolor="#FAFBFC", plot_bgcolor="#FAFBFC",
    )
    fig.update_xaxes(title_text="Q [m³/h]")
    fig.update_yaxes(title_text="H [m]", row=1, col=1)
    fig.update_yaxes(title_text="η [%]", row=1, col=2)
    fig.update_yaxes(title_text="P [kW]", row=2, col=1)
    fig.update_yaxes(title_text="NPSH [m]", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Export ----
    st.divider()
    st.subheader("📤 Export")
    c1, c2 = st.columns(2)
    with c1:
        html = render_datasheet_html(st.session_state.project, st.session_state.process, pump, results)
        st.download_button("📄 Datenblatt (HTML)", data=html,
                           file_name=f"Datenblatt_{pump['id']}_{datetime.now():%Y%m%d}.html",
                           mime="text/html")
    with c2:
        js = export_project_json(st.session_state.project, st.session_state.process,
                                  st.session_state.pipeline, st.session_state.economic, results)
        st.download_button("💾 Projekt (JSON)", data=js,
                           file_name=f"Projekt_{datetime.now():%Y%m%d}.json",
                           mime="application/json")


def render_single_phase_page(pumps: List[dict], media: dict):
    try:
        _page_header("⚙️", "Einphasenpumpen", "Pumpenauswahl mit Viskositätskorrektur nach HI-Methode")
        tab_proc, tab_pipe, tab_econ = st.tabs(
            ["Prozessdaten", "Rohrleitung", "Wirtschaftlichkeit"]
        )

        with tab_proc:
            Q, H, nu, rho, T, p_vap, allow_out, reserve_pct, n_min, n_max = _render_process_tab(media)
            st.session_state.process.flow_m3h         = Q
            st.session_state.process.head_m           = H
            st.session_state.process.density_kgm3     = rho
            st.session_state.process.viscosity_cst    = nu
            st.session_state.process.temperature_c    = T
            st.session_state.process.vapor_pressure_bar = p_vap
            st.session_state["_allow_out"] = allow_out

        with tab_pipe:
            _render_pipeline_tab()

        with tab_econ:
            _render_economic_tab()

        st.divider()
        if Q > 0 and H > 0:
            _render_single_phase_results(Q, H, nu, rho, reserve_pct, n_min, n_max, pumps)
        else:
            st.info("ℹ️ Bitte Förderstrom Q und Förderhöhe H in den Prozessdaten eingeben.")

    except Exception as e:
        st.error(f"Fehler in Einphasenpumpen: {e}")


# ══════════════════════════════════════════════════════════════════
# Section 15 · Page: Multi-Phase Pump
# ══════════════════════════════════════════════════════════════════

def _bisect_pressure(gas: str, T_c: float, C_target: float,
                     p_lo: float = 0.5, p_hi: float = 50.0) -> Optional[float]:
    """Return minimum pressure [bar] at which gas solubility ≥ C_target [cm³N/L]."""
    if total_gas_solubility(gas, p_hi, T_c) < C_target:
        return None
    for _ in range(60):
        mid = (p_lo + p_hi) / 2.0
        if total_gas_solubility(gas, mid, T_c) >= C_target:
            p_hi = mid
        else:
            p_lo = mid
        if p_hi - p_lo < 0.01:
            break
    return p_hi


def _diagnose_mph_failure(pump: dict, Q_gas_req: float, Q_gas_oper_lpm: float,
                          p_suction: float, gas: str, T_c: float) -> str:
    """Return a human-readable explanation of why a pump cannot satisfy the requirement."""
    Q_max_lpm = m3h_to_lpm(pump["max_flow_m3h"])
    gvf_at_max = Q_gas_oper_lpm / (Q_gas_oper_lpm + Q_max_lpm) * 100.0
    if gvf_at_max > pump.get("max_gvf_pct", 20):
        return (f"GVF = {gvf_at_max:.0f}% > max. {pump.get('max_gvf_pct', 20)}% "
                f"– Gas-Anteil am Eingang zu hoch")
    p_dis_max  = p_suction + pump.get("max_dp_bar", 10)
    C_sat_max  = total_gas_solubility(gas, p_dis_max, T_c)
    Q_sol_max  = (C_sat_max / 1000.0) * Q_max_lpm
    if Q_sol_max < Q_gas_req:
        return (f"Max. löslich {Q_sol_max:.0f} L/min < gefordert {Q_gas_req:.0f} L/min "
                f"– auch bei max. Druck ({p_dis_max:.1f} bar) nicht ausreichend")
    return "Kein konsistenter Betriebspunkt gefunden"


def _render_no_solution_hints(gas: str, T_c: float, Q_gas_req: float, p_suction: float):
    """Show the minimum discharge pressure needed at various liquid flows."""
    st.divider()
    st.subheader("💡 Was würde helfen?")
    st.markdown("Mindest-Austrittsdruck für vollständige Gaslösung bei verschiedenen Flüssigkeitsströmen:")
    rows = []
    for Q_ref in [5, 10, 20, 30, 50]:
        C_needed = (Q_gas_req / m3h_to_lpm(Q_ref)) * 1000.0
        p_needed = _bisect_pressure(gas, T_c, C_needed)
        if p_needed:
            rows.append({
                "Q_liq [m³/h]": Q_ref,
                "Benötigte Konzentration [cm³N/L]": f"{C_needed:.0f}",
                "Mindest-Austrittsdruck [bar(a)]": f"{p_needed:.1f}",
                "Erforderliche Druckerhöhung Δp [bar]": f"{p_needed - p_suction:.1f}",
            })
    if rows:
        st.dataframe(rows, use_container_width=True)
    st.info("Tipp: Höherer Flüssigkeitsstrom oder höherer Saugdruck reduziert den erforderlichen Δp.")


def render_multi_phase_page(mph_pumps: List[dict], media: dict):
    try:
        _page_header("🌊", "Mehrphasenpumpen-Auslegung", "Gaslösung nach Henry-Gesetz · GVF-Berechnung")
        st.info(
            "Die Pumpenauswahl erfolgt automatisch aus **Gasvolumenstrom** und **Medium**. "
            "Der GVF am Pumpeneingang wird physikalisch berechnet. "
            "Primäre Auswahlbedingung: vollständige Gaslösung am Druckaustritt (Henry-Gesetz)."
        )

        tab_in, tab_curves = st.tabs(["📝 Eingaben & Ergebnisse", "📈 Kennlinien"])

        # ── Inputs (simplified) ────────────────────────────────────
        with tab_in:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Gasanforderung")
                Q_gas_target_lpm = st.number_input(
                    "Ziel-Gasvolumenstrom [L/min, Normbedingungen]",
                    min_value=1.0, max_value=500.0, value=80.0, step=5.0,
                    help="Normvolumenstrom des zu lösenden Gases (0 °C / 1,013 bar)",
                )
                p_suction = st.number_input(
                    "Saugdruck [bar(a)]",
                    min_value=0.3, max_value=10.0, value=0.6, step=0.1,
                    help="Absoluter Druck an der Pumpen-Saugseite",
                )
                safety_pct = st.slider(
                    "Sicherheitszuschlag [%]", 0, 30, 10,
                    help="Auslegungsreserve auf den Mindest-Gasvolumenstrom",
                )

            with col2:
                st.subheader("Medium")
                gas_medium  = st.selectbox("Gas",         list(HENRY_CONSTANTS.keys()))
                liq_medium  = st.selectbox("Flüssigkeit", list(media.keys()))
                temperature = st.number_input("Temperatur [°C]", -10.0, 80.0, 20.0, 1.0)
                rho_liq = media[liq_medium]["density_kgm3"]
                nu_liq  = media[liq_medium]["viscosity_cst"]
                st.caption(f"ρ = {rho_liq} kg/m³  |  ν = {nu_liq} cSt")

        # ── Physics-based selection ────────────────────────────────
        # Gas flow including safety margin
        Q_gas_req = Q_gas_target_lpm * (1.0 + safety_pct / 100.0)

        # Operational gas volume at suction pressure (ideal gas law)
        T_K = temperature + 273.15
        Q_gas_oper_lpm = Q_gas_req * (P_NORMAL_BAR / p_suction) * (T_K / T_NORMAL_K)

        ok_results:   List[dict] = []
        fail_results: List[dict] = []

        for pump in mph_pumps:
            if nu_liq  > pump.get("max_viscosity_cst", 500):  continue
            if rho_liq > pump.get("max_density_kgm3", 1200):  continue

            ref_key = sorted(pump["curves_dp_vs_Q"].keys())[0]
            Q_all   = [q for q in pump["curves_dp_vs_Q"][ref_key]["Q"] if q > 0]
            Q_lo, Q_hi = max(min(Q_all), 1.0), min(max(Q_all), pump["max_flow_m3h"])

            best_for_pump: Optional[dict] = None

            for Q_liq in np.linspace(Q_lo, Q_hi, 80):
                Q_liq_lpm = m3h_to_lpm(Q_liq)

                # ① Luftanteil = GESAMTER Gasanteil am Volumenstrom bei Saugbedingungen
                #    (gelöstes + freies Gas, als ob alles frei wäre)
                #    → das ist der Kennlinien-Parameter der Pumpe ("prozentualer Luftanteil")
                gas_loading_pct = Q_gas_oper_lpm / (Q_gas_oper_lpm + Q_liq_lpm) * 100.0
                if gas_loading_pct > pump.get("max_gvf_pct", 20):
                    continue  # max. Luftanteil der Pumpe überschritten

                # ① Freier GVF am Eingang (gelöstes Gas belegt kein Volumen)
                #    GVF_frei = Anteil des noch NICHT gelösten Gases bei p_suction
                C_total_in   = (Q_gas_req / Q_liq_lpm) * 1000.0          # [cm³N/L]
                C_sat_in     = total_gas_solubility(gas_medium, p_suction, temperature)
                C_free_in    = max(0.0, C_total_in - C_sat_in)
                Q_free_oper  = (C_free_in / 1000.0) * Q_liq_lpm * (P_NORMAL_BAR / p_suction) * (T_K / T_NORMAL_K)
                gvf_free_pct = Q_free_oper / (Q_free_oper + Q_liq_lpm) * 100.0
                # Hinweis: bei ausreichend hohem Saugdruck kann gvf_free_pct = 0 sein
                #          (alles bereits am Eingang gelöst), GVF Austritt ist IMMER 0

                # ② Kennlinie bei (Q_liq, Luftanteil) – NICHT beim freien GVF
                dp_avail, *_ = dp_at_operating_point(pump, Q_liq, gas_loading_pct)
                P_shaft, *_  = power_at_operating_point(pump, Q_liq, gas_loading_pct)
                if dp_avail <= 0:
                    continue

                # ③ Solubility at discharge – PRIMARY CONSTRAINT
                p_discharge    = p_suction + dp_avail
                C_sat_dis      = total_gas_solubility(gas_medium, p_discharge, temperature)
                Q_solvable_lpm = (C_sat_dis / 1000.0) * Q_liq_lpm

                if Q_solvable_lpm < Q_gas_req:
                    continue  # discharge pressure too low to dissolve all gas

                # ④ Score: prefer high efficiency / low specific power
                P_hyd   = (dp_avail * BAR_TO_PA) * (Q_liq / 3600.0) / 1000.0
                eta_est = clamp(P_hyd / max(P_shaft, 0.1), 0.05, 0.92)
                score   = P_shaft / max(Q_liq, 1.0) + (1.0 - eta_est)

                cand = {
                    "pump": pump, "pump_id": pump["id"],
                    "Q_liq_m3h": Q_liq, "Q_liq_lpm": Q_liq_lpm,
                    "gas_loading_pct": gas_loading_pct,   # Kennlinien-Parameter (gelöst + frei)
                    "gvf_free_pct": gvf_free_pct,         # echter freier GVF am Eingang
                    "dp_bar": dp_avail, "p_discharge": p_discharge,
                    "P_shaft_kw": P_shaft, "eta_est": eta_est,
                    "C_sat_dis": C_sat_dis,
                    "Q_solvable_lpm": Q_solvable_lpm,
                    "solubility_margin_pct": (Q_solvable_lpm / Q_gas_req - 1.0) * 100.0,
                    "n_rpm": pump["rated_speed_rpm"],
                    "score": score,
                }
                if best_for_pump is None or score < best_for_pump["score"]:
                    best_for_pump = cand

            if best_for_pump:
                ok_results.append(best_for_pump)
            else:
                fail_results.append({
                    "pump_id": pump["id"], "pump": pump,
                    "fail_reason": _diagnose_mph_failure(
                        pump, Q_gas_req, Q_gas_oper_lpm, p_suction, gas_medium, temperature
                    ),
                })

        ok_results.sort(key=lambda x: x["score"])

        # ── Results ────────────────────────────────────────────────
        with tab_in:
            st.divider()
            if not ok_results:
                st.error("❌ Keine der verfügbaren Pumpen kann den Gasvolumenstrom vollständig lösen.")
                for r in fail_results:
                    st.warning(f"**{r['pump_id']}**: {r['fail_reason']}")
                _render_no_solution_hints(gas_medium, temperature, Q_gas_req, p_suction)
                return

            best = ok_results[0]
            st.success(f"✅ Empfehlung: **{best['pump_id']}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Flüssigkeitsstrom",         f"{best['Q_liq_m3h']:.1f} m³/h")
            c1.metric("Luftanteil (Kennlinie)",    f"{best['gas_loading_pct']:.1f} %",
                      help="Gesamter Gasanteil inkl. gelöstem Anteil – Kennlinienparameter")
            c2.metric("Freier GVF Eingang",        f"{best['gvf_free_pct']:.1f} %",
                      help="Tatsächlich freies (ungelöstes) Gas am Pumpeneingang")
            c2.metric("GVF Austritt",              "0 %  ✅",
                      help="Gesamtes Gas ist am Druckaustritt gelöst – Pflichtbedingung")
            c3.metric("Druckerhöhung Δp",          f"{best['dp_bar']:.2f} bar")
            c3.metric("Austrittsdruck",            f"{best['p_discharge']:.2f} bar(a)")
            c4.metric("Wellenleistung",            f"{best['P_shaft_kw']:.2f} kW")
            c4.metric("Löslichkeitsreserve",       f"+{best['solubility_margin_pct']:.0f} %")

            # Solubility balance
            st.divider()
            st.subheader("🧪 Löslichkeitsbilanz")
            col1, col2 = st.columns([3, 1])
            with col1:
                C_req     = (Q_gas_req / best["Q_liq_lpm"]) * 1000.0
                C_sat_in  = total_gas_solubility(gas_medium, p_suction, temperature)
                C_free_in = max(0.0, C_req - C_sat_in)
                st.markdown(f"""
| Parameter | Wert |
|---|---|
| Geforderter Gasvolumenstrom | **{Q_gas_target_lpm:.1f} L/min** (Norm) |
| Mit Sicherheit (+{safety_pct} %) | **{Q_gas_req:.1f} L/min** |
| Gewählter Flüssigkeitsstrom | {best['Q_liq_m3h']:.1f} m³/h |
| Benötigte Konzentration | {C_req:.1f} cm³N/L |
| Löslichkeit bei Saugdruck ({p_suction} bar) | {C_sat_in:.1f} cm³N/L |
| Bereits gelöst bei Saugdruck | {min(C_sat_in, C_req):.1f} cm³N/L ({min(C_sat_in/C_req*100, 100):.0f}% des Gesamtgases) |
| Freies Gas am Eingang (GVF_frei) | {C_free_in:.1f} cm³N/L → **{best['gvf_free_pct']:.1f} %** (freier GVF) |
| Luftanteil Kennlinie | **{best['gas_loading_pct']:.1f} %** (gelöst + frei) |
| **GVF Druckaustritt** | **0 %** – vollständig gelöst ✅ |
| Löslichkeit bei Austrittsdruck ({best['p_discharge']:.1f} bar) | **{best['C_sat_dis']:.1f} cm³N/L** ✅ |
| Max. löslicher Gasstrom | **{best['Q_solvable_lpm']:.1f} L/min** (+{best['solubility_margin_pct']:.0f} % Reserve) |
""")
            with col2:
                auslastung = min(Q_gas_req / max(best["Q_solvable_lpm"], 0.01), 1.0)
                st.markdown("**Löslichkeits-auslastung**")
                st.progress(auslastung, text=f"{auslastung*100:.0f} %")

            # Motor sizing
            st.divider()
            st.subheader("⚡ Motorauslegung")
            P_mot_min = best["P_shaft_kw"] * 1.15
            P_mot_iec = next_iec_motor(P_mot_min)
            c1, c2, c3 = st.columns(3)
            c1.metric("Wellenleistung",              f"{best['P_shaft_kw']:.2f} kW")
            c2.metric("Mindest-Motorleistung (+15%)", f"{P_mot_min:.2f} kW")
            c3.metric("IEC-Motorgröße",              f"{P_mot_iec} kW")

            # All pumps overview
            st.divider()
            st.subheader("🔄 Alle Pumpen – Übersicht")
            rows = []
            for r in ok_results:
                rows.append({
                    "Pumpe": r["pump_id"], "Status": "✅",
                    "Q_liq [m³/h]": f"{r['Q_liq_m3h']:.1f}",
                    "Luftanteil [%]": f"{r['gas_loading_pct']:.1f}",
                    "GVF frei Eingang [%]": f"{r['gvf_free_pct']:.1f}",
                    "Δp [bar]": f"{r['dp_bar']:.2f}",
                    "P [kW]": f"{r['P_shaft_kw']:.1f}",
                    "η [%]": f"{r['eta_est']*100:.0f}",
                    "Löslichkeitsreserve": f"+{r['solubility_margin_pct']:.0f} %",
                })
            for r in fail_results:
                rows.append({
                    "Pumpe": r["pump_id"], "Status": "❌",
                    "Q_liq [m³/h]": "–", "Luftanteil [%]": "–", "GVF frei Eingang [%]": "–", "Δp [bar]": "–",
                    "P [kW]": "–", "η [%]": "–",
                    "Löslichkeitsreserve": r["fail_reason"],
                })
            st.dataframe(rows, use_container_width=True)

        # ── Curves tab ─────────────────────────────────────────────
        with tab_curves:
            if not ok_results:
                return
            best      = ok_results[0]
            pump      = best["pump"]
            P_mot_iec = next_iec_motor(best["P_shaft_kw"] * 1.15)
            gvf_keys  = sorted(pump["curves_dp_vs_Q"].keys())
            n_gvf     = len(gvf_keys)
            _gvf_colors = [
                _VIRIDIS_STOPS[int(i * (len(_VIRIDIS_STOPS) - 1) / max(n_gvf - 1, 1))]
                for i in range(n_gvf)
            ]
            C_req     = (Q_gas_req / best["Q_liq_lpm"]) * 1000.0

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Q-Δp Kennlinien",
                    "Q-P Kennlinien",
                    f"Gas-Löslichkeit – Henry-Gesetz ({gas_medium})",
                    f"{pump['id']} – Löslichkeitsreserve vs. Q_liq",
                ),
                specs=[[{}, {}], [{}, {"secondary_y": True}]],
                vertical_spacing=0.13, horizontal_spacing=0.10,
            )

            # 1 · Q-Δp family curves
            for idx, (gvf, col) in enumerate(zip(gvf_keys, _gvf_colors)):
                qv, dpv = trim_arrays(pump["curves_dp_vs_Q"][gvf]["Q"],
                                      pump["curves_dp_vs_Q"][gvf]["dp"])
                if len(qv) >= 2:
                    fig.add_trace(go.Scatter(
                        x=qv, y=dpv, mode="lines+markers",
                        name=f"GVF {gvf} %", line=dict(color=col, width=2),
                        marker=dict(size=4), legendgroup="gvf_dp",
                        showlegend=(idx == 0),
                    ), row=1, col=1)
            # OP cross-lines
            fig.add_hline(y=best["dp_bar"], line_dash="dot", line_color=_CLR_VISCOUS,
                          line_width=1, opacity=0.5, row=1, col=1)
            fig.add_vline(x=best["Q_liq_m3h"], line_dash="dot", line_color=_CLR_VISCOUS,
                          line_width=1, opacity=0.5, row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[best["Q_liq_m3h"]], y=[best["dp_bar"]], mode="markers",
                name=f"Betriebspunkt (Luftanteil {best['gas_loading_pct']:.1f} %)",
                marker=dict(symbol="star", size=18, color=_CLR_OP, line=dict(color="#333", width=1)),
            ), row=1, col=1)

            # 2 · Q-P family curves
            for idx, (gvf, col) in enumerate(zip(gvf_keys, _gvf_colors)):
                if gvf in pump["power_kW_vs_Q"]:
                    qv, pv = trim_arrays(pump["power_kW_vs_Q"][gvf]["Q"],
                                        pump["power_kW_vs_Q"][gvf]["P"])
                    if len(qv) >= 2:
                        fig.add_trace(go.Scatter(
                            x=qv, y=pv, mode="lines+markers",
                            name=f"GVF {gvf} % (P)", line=dict(color=col, width=2),
                            marker=dict(symbol="square", size=4),
                            showlegend=False,
                        ), row=1, col=2)
            # Motor line
            q_all = pump["curves_dp_vs_Q"][gvf_keys[0]]["Q"]
            fig.add_hline(y=P_mot_iec, line_dash="longdash", line_color=_CLR_SYSTEM,
                          line_width=2, annotation_text=f"Motor {P_mot_iec} kW",
                          annotation_position="top left", row=1, col=2)
            fig.add_trace(go.Scatter(
                x=[best["Q_liq_m3h"]], y=[best["P_shaft_kw"]], mode="markers",
                name="OP (P)", showlegend=False,
                marker=dict(symbol="star", size=18, color=_CLR_OP, line=dict(color="#333", width=1)),
            ), row=1, col=2)

            # 3 · Henry's Law solubility
            p_max_chart = max(best["p_discharge"] * 1.6, 12.0)
            p_rng = np.linspace(0.3, p_max_chart, 100)
            C_sol = [total_gas_solubility(gas_medium, p, temperature) for p in p_rng]
            fig.add_trace(go.Scatter(
                x=list(p_rng), y=C_sol, mode="lines",
                name=f"Löslichkeit {gas_medium}",
                line=dict(color=_CLR_PRIMARY, width=2.5),
                fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
            ), row=2, col=1)
            # Required concentration line
            fig.add_hline(y=C_req, line_dash="dashdot", line_color=_CLR_WARN,
                          line_width=1.5,
                          annotation_text=f"Benötigt {C_req:.0f} cm³N/L",
                          annotation_position="top left", row=2, col=1)
            # Suction and discharge pressure markers
            fig.add_vline(x=p_suction, line_dash="dot", line_color="gray",
                          line_width=1.5, row=2, col=1)
            fig.add_vline(x=best["p_discharge"], line_dash="dash", line_color=_CLR_VISCOUS,
                          line_width=1.5,
                          annotation_text=f"p_dis {best['p_discharge']:.1f} bar",
                          annotation_position="top right", row=2, col=1)
            fig.add_trace(go.Scatter(
                x=[best["p_discharge"]], y=[best["C_sat_dis"]], mode="markers",
                name=f"+{best['solubility_margin_pct']:.0f}% Reserve",
                marker=dict(symbol="star", size=16, color=_CLR_VISCOUS, line=dict(color="#333", width=1)),
            ), row=2, col=1)

            # 4 · Solubility margin vs Q_liq (primary) + gas loading % (secondary)
            Q_scan = np.linspace(max(pump["max_flow_m3h"] * 0.1, 1.0),
                                 pump["max_flow_m3h"], 80)
            margins, gvf_scan = [], []
            for Ql in Q_scan:
                Ql_lpm = m3h_to_lpm(Ql)
                gvf_p  = Q_gas_oper_lpm / (Q_gas_oper_lpm + Ql_lpm) * 100.0
                gvf_scan.append(gvf_p)
                if gvf_p > pump.get("max_gvf_pct", 20):
                    margins.append(np.nan)
                else:
                    dp_v, *_ = dp_at_operating_point(pump, Ql, gvf_p)
                    p_d = p_suction + max(dp_v, 0)
                    C_d = total_gas_solubility(gas_medium, p_d, temperature)
                    margins.append((C_d / 1000.0 * Ql_lpm / Q_gas_req - 1.0) * 100.0)

            m_arr = np.array(margins, dtype=float)
            m_pos = np.where(m_arr > 0, m_arr, 0.0)
            m_neg = np.where(m_arr < 0, m_arr, 0.0)

            # Positive fill (ok region)
            fig.add_trace(go.Scatter(
                x=list(Q_scan), y=list(m_pos), mode="lines",
                name="Reserve > 0", fill="tozeroy",
                fillcolor="rgba(46,125,50,0.15)",
                line=dict(color=_CLR_SYSTEM, width=2),
            ), row=2, col=2, secondary_y=False)
            # Negative fill (insufficient region)
            fig.add_trace(go.Scatter(
                x=list(Q_scan), y=list(m_neg), mode="lines",
                name="Reserve < 0", fill="tozeroy",
                fillcolor="rgba(198,40,40,0.15)",
                line=dict(color=_CLR_VISCOUS, width=0),
                showlegend=False,
            ), row=2, col=2, secondary_y=False)
            fig.add_hline(y=0, line_dash="dash", line_color=_CLR_VISCOUS,
                          line_width=1.5, row=2, col=2)
            fig.add_trace(go.Scatter(
                x=[best["Q_liq_m3h"]], y=[best["solubility_margin_pct"]], mode="markers",
                name="OP", showlegend=False,
                marker=dict(symbol="star", size=18, color=_CLR_OP, line=dict(color="#333", width=1)),
            ), row=2, col=2, secondary_y=False)
            # Secondary: gas loading %
            fig.add_trace(go.Scatter(
                x=list(Q_scan), y=gvf_scan, mode="lines",
                name="Luftanteil [%]", line=dict(color=_CLR_SYSTEM, width=1.5, dash="dash"),
                opacity=0.6,
            ), row=2, col=2, secondary_y=True)
            fig.add_hline(
                y=pump.get("max_gvf_pct", 20), line_dash="dot", line_color=_CLR_SYSTEM,
                line_width=1, opacity=0.6, row=2, col=2,
            )

            fig.update_layout(
                template="plotly_white", height=750,
                title_text=(
                    f"{pump['id']}  |  Q_gas = {Q_gas_target_lpm:.0f} L/min "
                    f"({gas_medium}, {temperature} °C)"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=80, b=40),
                paper_bgcolor="#FAFBFC",
            )
            fig.update_xaxes(title_text="Q_liq [m³/h]", row=1, col=1)
            fig.update_xaxes(title_text="Q_liq [m³/h]", row=1, col=2)
            fig.update_xaxes(title_text="Druck [bar(a)]", row=2, col=1)
            fig.update_xaxes(title_text="Q_liq [m³/h]", row=2, col=2)
            fig.update_yaxes(title_text="Δp [bar]", row=1, col=1)
            fig.update_yaxes(title_text="P [kW]", row=1, col=2)
            fig.update_yaxes(title_text="C_sat [cm³N/L]", row=2, col=1)
            fig.update_yaxes(title_text="Löslichkeitsreserve [%]", row=2, col=2, secondary_y=False)
            fig.update_yaxes(title_text="Luftanteil [%]", row=2, col=2, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Export
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                summary = (
                    f"MEHRPHASEN-PUMPENAUSLEGUNG\n"
                    f"==========================\n"
                    f"Datum: {datetime.now():%Y-%m-%d %H:%M}\n\n"
                    f"ANFORDERUNG\n-----------\n"
                    f"Gas:                     {gas_medium}\n"
                    f"Flüssigkeit:             {liq_medium}\n"
                    f"Temperatur:              {temperature} °C\n"
                    f"Saugdruck:               {p_suction} bar(a)\n"
                    f"Gasvolumenstrom:         {Q_gas_target_lpm:.1f} L/min (Norm)\n"
                    f"Mit Sicherheit (+{safety_pct}%): {Q_gas_req:.1f} L/min\n\n"
                    f"EMPFOHLENE PUMPE: {best['pump_id']}\n"
                    f"--------------------\n"
                    f"Flüssigkeitsstrom:       {best['Q_liq_m3h']:.1f} m³/h\n"
                    f"Luftanteil (Kennlinie):  {best['gas_loading_pct']:.1f} %\n"
                    f"Freier GVF (Eingang):    {best['gvf_free_pct']:.1f} %\n"
                    f"Druckerhöhung:           {best['dp_bar']:.2f} bar\n"
                    f"Austrittsdruck:          {best['p_discharge']:.2f} bar(a)\n"
                    f"Wellenleistung:          {best['P_shaft_kw']:.2f} kW\n"
                    f"IEC Motor:               {P_mot_iec} kW\n"
                    f"Löslichkeit Austritt:    {best['C_sat_dis']:.1f} cm³N/L\n"
                    f"Löslichkeitsreserve:     +{best['solubility_margin_pct']:.0f} %\n"
                )
                st.download_button("📄 Zusammenfassung (TXT)", data=summary,
                                   file_name=f"MPH_{datetime.now():%Y%m%d}.txt",
                                   mime="text/plain")
            with col2:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input": {
                        "gas": gas_medium, "liquid": liq_medium,
                        "temperature_c": temperature,
                        "p_suction_bar": p_suction,
                        "Q_gas_target_lpm": Q_gas_target_lpm,
                        "safety_pct": safety_pct,
                    },
                    "result": {
                        "pump_id": best["pump_id"],
                        "Q_liq_m3h": best["Q_liq_m3h"],
                        "gvf_inlet_pct": best["gvf_pct"],
                        "dp_bar": best["dp_bar"],
                        "p_discharge_bar": best["p_discharge"],
                        "P_shaft_kw": best["P_shaft_kw"],
                        "P_motor_iec_kw": P_mot_iec,
                        "C_sat_discharge_cm3n_l": best["C_sat_dis"],
                        "solubility_margin_pct": best["solubility_margin_pct"],
                    },
                }
                st.download_button("💾 Export (JSON)",
                                   data=json.dumps(export_data, indent=2),
                                   file_name=f"MPH_{datetime.now():%Y%m%d}.json",
                                   mime="application/json")

    except Exception as e:
        st.error(f"Fehler in Mehrphasenpumpen: {e}")


# ══════════════════════════════════════════════════════════════════
# Section 16 · Page: ATEX Motor Selection
# ══════════════════════════════════════════════════════════════════
def render_atex_page(atex_motors: List[dict]):
    try:
        _page_header("⚡", "ATEX-Motorauslegung", "Auslegung nach ATEX 2014/34/EU und EN 60079")
        st.info("Auslegung nach ATEX 2014/34/EU und EN 60079")

        TEMP_CLASS_LIMITS = {"T1": 450, "T2": 300, "T3": 200, "T4": 135, "T5": 100, "T6": 85}

        tab_in, tab_doc = st.tabs(["📝 Eingaben & Auswahl", "📋 Normreferenz"])

        with tab_in:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Leistung")
                P_req          = st.number_input("Wellenleistung [kW]",  min_value=0.1, value=5.5, step=0.5)
                reserve_factor = st.number_input("Leistungsreserve [-]", min_value=1.0, max_value=1.5, value=1.15, step=0.05)
                efficiency_cls = st.selectbox("Effizienzklasse", ["IE2", "IE3", "IE4"], index=1)

            with col2:
                st.subheader("Explosionsschutz")
                atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"])
                if atmosphere == "Gas":
                    zone = st.selectbox("Zone", [0, 1, 2],
                                        format_func=lambda x: f"Zone {x} – {'Ständig' if x==0 else 'Gelegentlich' if x==1 else 'Selten'}")
                    gas_group = st.selectbox("Gasgruppe", ["IIA", "IIB", "IIC"], index=2,
                                             help="IIA: Propan | IIB: Ethylen | IIC: Wasserstoff/Acetylen")
                else:
                    zone = st.selectbox("Zone", [20, 21, 22],
                                        format_func=lambda x: f"Zone {x} – {'Ständig' if x==20 else 'Gelegentlich' if x==21 else 'Selten'}")
                    gas_group = "IIIC"

                temp_class = st.selectbox("Temperaturklasse", list(TEMP_CLASS_LIMITS.keys()), index=3,
                                          help="T1:450 °C · T2:300 °C · T3:200 °C · T4:135 °C · T5:100 °C · T6:85 °C")

            with col3:
                st.subheader("Umgebung")
                T_ambient = st.number_input("Umgebungstemperatur [°C]", -40.0, 60.0, 40.0)
                T_medium  = st.number_input("Medientemperatur [°C]",    -20.0, 200.0, 40.0)
                altitude  = st.number_input("Aufstellhöhe [m ü.NN]",    0, 4000, 0, 100)
                t_margin  = st.number_input("Temperatursicherheit [K]", 0.0, 50.0, 15.0)

            # Protection type by zone
            if atmosphere == "Gas":
                prot_opts = (
                    ["Ex ia (Eigensicherheit)"] if zone == 0 else
                    ["Ex d (Druckfeste Kapselung)", "Ex e (Erhöhte Sicherheit)", "Ex p (Überdruckkapselung)"] if zone == 1 else
                    ["Ex ec (Erhöhte Sicherheit)", "Ex nA (Nicht-funkend)"]
                )
            else:
                prot_opts = ["Ex tD (Staubdicht)", "Ex pD (Überdruckkapselung)"]
            sel_prot = st.selectbox("Zündschutzart", prot_opts)

            st.divider()
            # ---- Calculation ----
            t_max_allowed = TEMP_CLASS_LIMITS[temp_class] - t_margin
            P_motor_min   = P_req * reserve_factor

            if altitude > 1000:
                alt_factor   = 1.0 - (altitude - 1000) * 0.0001
                P_motor_min /= max(alt_factor, 0.5)
                st.warning(f"Höhenkorrektur bei {altitude} m: Motorleistung +{(1/alt_factor-1)*100:.1f}%")

            P_motor_iec    = next_iec_motor(P_motor_min)
            suitable_motors = [
                m for m in atex_motors
                if zone in m["zones_suitable"] and m["max_surface_temp_c"] <= t_max_allowed
            ]

            st.subheader("📊 Ergebnisse")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Wellenleistung",      f"{P_req:.2f} kW")
            c2.metric("Min. Motorleistung",  f"{P_motor_min:.2f} kW")
            c3.metric("IEC Motorgröße",      f"{P_motor_iec} kW")
            c4.metric("Max. Oberfl.-Temp.",  f"{t_max_allowed:.0f} °C")

            if not suitable_motors:
                st.error("❌ Kein passender ATEX-Motor gefunden.")
                st.info("Tipp: niedrigere Temperaturklasse wählen oder Zündschutzart prüfen.")
            else:
                st.success(f"✅ {len(suitable_motors)} passende(r) Motor(en) gefunden")
                rows = [{
                    "Typ": m["id"], "Kennzeichnung": m["marking"],
                    "T-Klasse": m["temp_class"], "Max. T [°C]": m["max_surface_temp_c"],
                    "Effizienz": m["efficiency_class"],
                    "Preis ca.": f"{P_motor_iec * 180 * m['price_factor']:,.0f} €",
                } for m in suitable_motors]
                st.dataframe(rows, use_container_width=True)

                best_motor = suitable_motors[0]
                category   = "1G" if zone == 0 else "2G" if zone == 1 else "3G"
                epl        = "Ga" if zone == 0 else "Gb" if zone == 1 else "Gc"
                prot_code  = sel_prot.split()[1].lower()
                marking    = f"II {category} Ex {prot_code} {gas_group} {temp_class} {epl}"

                st.divider()
                st.subheader("🏆 Empfehlung")
                st.code(marking)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
| Parameter | Wert |
|---|---|
| Motor-Typ | {best_motor['id']} |
| IEC-Leistung | {P_motor_iec} kW |
| Effizienzklasse | {efficiency_cls} |
| Zündschutzart | {sel_prot} |
| Gasgruppe | {gas_group} |
| Temperaturklasse | {temp_class} |
| Zone | {zone} |
""")
                with col2:
                    st.markdown(f"""
| Betriebsbedingung | Wert |
|---|---|
| Umgebungstemperatur | {T_ambient} °C |
| Medientemperatur | {T_medium} °C |
| Aufstellhöhe | {altitude} m |
| Zul. Oberflächentemp. | {t_max_allowed:.0f} °C |
""")

                # Export
                st.divider()
                export_txt = f"""ATEX-MOTORAUSLEGUNG
===================
Datum: {datetime.now():%Y-%m-%d %H:%M}

ATEX-Kennzeichnung: {marking}

ANFORDERUNG
-----------
Wellenleistung:          {P_req} kW
Leistungsreserve:        {reserve_factor}
Erforderl. Motorleistg.: {P_motor_min:.2f} kW
IEC Motorgröße:          {P_motor_iec} kW

EXPLOSIONSSCHUTZ
----------------
Atmosphäre:              {atmosphere}
Zone:                    {zone}
Gasgruppe:               {gas_group}
Temperaturklasse:        {temp_class}
Zündschutzart:           {sel_prot}

EMPFOHLENER MOTOR
-----------------
Typ:                     {best_motor['id']}
Kennzeichnung:           {best_motor['marking']}
"""
                st.download_button("📄 ATEX-Datenblatt (TXT)", data=export_txt,
                                   file_name=f"ATEX_{datetime.now():%Y%m%d}.txt", mime="text/plain")

        with tab_doc:
            st.markdown("""
## ATEX-Richtlinie 2014/34/EU

### Zoneneinteilung (Gas / Dampf)
| Zone | Beschreibung | Beispiele |
|------|-------------|-----------|
| 0 | Ständig oder langzeitig explosionsfähige Atmosphäre | Innenraum von Tanks |
| 1 | Gelegentlich bei Normalbetrieb | Umgebung von Entlüftungen |
| 2 | Selten und kurzzeitig | Allg. Anlagenbereiche |

### Temperaturklassen
| Klasse | Max. Oberflächentemperatur |
|--------|---------------------------|
| T1 | 450 °C |
| T2 | 300 °C |
| T3 | 200 °C |
| T4 | 135 °C |
| T5 | 100 °C |
| T6 | 85 °C |

### Gasgruppen
| Gruppe | Typische Gase |
|--------|--------------|
| IIA | Propan, Butan, Benzin |
| IIB | Ethylen, Stadtgas |
| IIC | Wasserstoff, Acetylen |

### Zündschutzarten (Auswahl)
| Kürzel | Bezeichnung | Zone |
|--------|-------------|------|
| Ex d | Druckfeste Kapselung | 1, 2 |
| Ex e / ec | Erhöhte Sicherheit | 1, 2 |
| Ex p | Überdruckkapselung | 1, 2 |
| Ex i / ia | Eigensicherheit | 0, 1, 2 |
| Ex nA | Nicht-funkend | 2 |
""")

    except Exception as e:
        st.error(f"Fehler in ATEX-Auslegung: {e}")


# ══════════════════════════════════════════════════════════════════
# Section 17 · Page: Pump Comparison (improved)
# ══════════════════════════════════════════════════════════════════
def render_comparison_page(pumps: List[dict], media: dict):
    _page_header("⚖️", "Pumpenvergleich", "Überlagerte Kennlinien und Betriebspunkt-Vergleich")

    if len(pumps) < 2:
        st.warning("Mindestens 2 Pumpen in der Datenbank erforderlich.")
        return

    selected_ids = st.multiselect(
        "Pumpen auswählen (2–3)", [p["id"] for p in pumps],
        default=[pumps[0]["id"], pumps[1]["id"]], max_selections=3,
    )

    if len(selected_ids) < 2:
        st.info("Bitte mindestens 2 Pumpen auswählen.")
        return

    sel_pumps = [p for p in pumps if p["id"] in selected_ids]

    # Optional operating point
    with st.expander("🎯 Betriebspunkt für Vergleich (optional)", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Q_op = st.number_input("Förderstrom Q [m³/h]", min_value=0.1, value=30.0, step=1.0)
        with col2:
            H_op = st.number_input("Förderhöhe H [m]",     min_value=0.1, value=25.0, step=1.0)
        with col3:
            med  = st.selectbox("Medium", list(media.keys()), key="cmp_med")
            nu   = float(media[med]["viscosity_cst"])
            rho  = float(media[med]["density_kgm3"])
        with col4:
            nu_override = st.number_input("Viskosität ν [cSt] (Override)", min_value=0.1, value=nu, step=0.5)
            nu = nu_override

    # ---- Metric table ----
    st.subheader("📋 Kennwerte-Vergleich")
    rows = []
    op_results = {}
    for p in sel_pumps:
        Q_eval  = clamp(Q_op, min(p["Qw"]), max(p["Qw"]))
        H_at    = lerp(Q_eval, p["Qw"], p["Hw"])
        eta_wat = lerp(Q_eval, p["Qw"], p["eta"])
        P_wat   = lerp(Q_eval, p["Qw"], p["Pw"])
        # Viscosity correction
        conv    = viscous_to_water_equivalent(Q_op, H_op, nu)
        CH, Ceta = conv["CH"], conv["Ceta"]
        eta_vis = clamp(eta_wat * Ceta, 0.05, 0.95)
        P_hyd_W = rho * G * (Q_op / 3600.0) * H_op
        P_shaft = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
        P_motor = next_iec_motor(P_shaft * 1.1)
        NPSHr   = lerp(Q_eval, p["Qw"], p.get("NPSHr", [2.0]*len(p["Qw"])))

        op_results[p["id"]] = {
            "eta_vis": eta_vis, "P_shaft": P_shaft, "P_motor": P_motor, "NPSHr": NPSHr,
        }
        rows.append({
            "Pumpe":            p["id"],
            "Typ":              p.get("pump_type", "–"),
            "Q_max [m³/h]":     max(p["Qw"]),
            "H_max [m]":        max(p["Hw"]),
            "η_max [%]":        f"{max(p['eta'])*100:.1f}",
            "η @ Betriebspunkt [%]": f"{eta_vis*100:.1f}",
            "P_Welle [kW]":     f"{P_shaft:.2f}",
            "P_Motor (IEC) [kW]": f"{P_motor:.1f}",
            "NPSHr @ Q [m]":   f"{NPSHr:.2f}",
            "Preis [€]":        f"{p.get('price_eur', 0):,}",
        })

    st.dataframe(rows, use_container_width=True)

    # Highlight best at operating point
    best_eta_id  = max(op_results, key=lambda k: op_results[k]["eta_vis"])
    lowest_P_id  = min(op_results, key=lambda k: op_results[k]["P_shaft"])
    lowest_NPSH  = min(op_results, key=lambda k: op_results[k]["NPSHr"])
    col1, col2, col3 = st.columns(3)
    col1.success(f"🏆 Bester Wirkungsgrad: **{best_eta_id}** ({op_results[best_eta_id]['eta_vis']*100:.1f}%)")
    col2.info(f"⚡ Geringste Leistung: **{lowest_P_id}** ({op_results[lowest_P_id]['P_shaft']:.2f} kW)")
    col3.info(f"💧 Geringster NPSHr: **{lowest_NPSH}** ({op_results[lowest_NPSH]['NPSHr']:.2f} m)")

    # ---- Overlaid curves ----
    st.divider()
    st.subheader("📈 Überlagerte Kennlinien")

    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Q-H
    ax = axes[0]
    for i, p in enumerate(sel_pumps):
        ax.plot(p["Qw"], p["Hw"], color=COLORS[i], marker="o", lw=2, label=p["id"])
        if not is_water(nu):
            _, Hv, _, _ = build_viscous_curves(p, nu, rho)
            qv_pos = [q for q in p["Qw"] if q > 0]
            hv_pos = [h for q, h in zip(p["Qw"], Hv) if q > 0]
            ax.plot(qv_pos, hv_pos, color=COLORS[i], ls="--", lw=1.5, alpha=0.6)
    ax.axhline(H_op, color="gray", ls=":", alpha=0.6, label=f"H_req = {H_op} m")
    ax.axvline(Q_op, color="gray", ls=":", alpha=0.6, label=f"Q_req = {Q_op} m³/h")
    ax.scatter([Q_op], [H_op], s=120, c="black", marker="*", zorder=10, label="Betriebspunkt")
    ax.set(xlabel="Q [m³/h]", ylabel="H [m]", title="Q-H Kennlinien")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Q-η
    ax = axes[1]
    for i, p in enumerate(sel_pumps):
        ax.plot(p["Qw"], [e * 100 for e in p["eta"]], color=COLORS[i], marker="o", lw=2, label=p["id"])
        if not is_water(nu):
            _, _, etav, _ = build_viscous_curves(p, nu, rho)
            qv_pos  = [q for q in p["Qw"] if q > 0]
            etv_pos = [e * 100 for q, e in zip(p["Qw"], etav) if q > 0]
            ax.plot(qv_pos, etv_pos, color=COLORS[i], ls="--", lw=1.5, alpha=0.6)
    ax.axvline(Q_op, color="gray", ls=":", alpha=0.6)
    ax.set(xlabel="Q [m³/h]", ylabel="η [%]", title="Q-η Kennlinien")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Q-P
    ax = axes[2]
    for i, p in enumerate(sel_pumps):
        ax.plot(p["Qw"], p["Pw"], color=COLORS[i], marker="o", lw=2, label=p["id"])
        if not is_water(nu):
            _, _, _, Pv = build_viscous_curves(p, nu, rho)
            qv_pos = [q for q in p["Qw"] if q > 0]
            pv_pos = [pp for q, pp in zip(p["Qw"], Pv) if q > 0]
            ax.plot(qv_pos, pv_pos, color=COLORS[i], ls="--", lw=1.5, alpha=0.6)
    ax.axvline(Q_op, color="gray", ls=":", alpha=0.6)
    ax.set(xlabel="Q [m³/h]", ylabel="P [kW]", title="Q-P Kennlinien")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    if not is_water(nu):
        for ax in axes:
            ax.text(0.98, 0.02, f"── Wasser  ╌╌ ν={nu:.1f} cSt",
                    transform=ax.transAxes, fontsize=7, ha="right", color="gray")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---- Efficiency bar chart at operating point ----
    st.divider()
    st.subheader(f"🎯 Wirkungsgrade am Betriebspunkt (Q={Q_op} m³/h, H={H_op} m)")
    fig2, (ax_eta, ax_P) = plt.subplots(1, 2, figsize=(10, 4))

    pump_names = [p["id"] for p in sel_pumps]
    eta_vals   = [op_results[pid]["eta_vis"] * 100 for pid in pump_names]
    P_vals     = [op_results[pid]["P_shaft"]        for pid in pump_names]
    bar_cols   = [COLORS[i] for i in range(len(pump_names))]

    ax_eta.bar(pump_names, eta_vals, color=bar_cols, alpha=0.8)
    ax_eta.set(ylabel="η [%]", title="Wirkungsgrad im Betriebspunkt")
    for bar_idx, v in enumerate(eta_vals):
        ax_eta.text(bar_idx, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax_eta.grid(alpha=0.3, axis="y")

    ax_P.bar(pump_names, P_vals, color=bar_cols, alpha=0.8)
    ax_P.set(ylabel="P [kW]", title="Wellenleistung im Betriebspunkt")
    for bar_idx, v in enumerate(P_vals):
        ax_P.text(bar_idx, v * 1.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax_P.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════
# Section 18 · Page: Documentation (improved)
# ══════════════════════════════════════════════════════════════════
def render_documentation_page():
    _page_header("📚", "Dokumentation & Berechnungsgrundlagen", "Normen, Formeln und Werkstoffinformation")

    tab_fund, tab_calc, tab_std, tab_mat, tab_gloss = st.tabs(
        ["🔬 Grundlagen", "🧮 Berechnungen", "📐 Normen", "⚗️ Werkstoffe", "📖 Glossar"]
    )

    with tab_fund:
        st.markdown("""
## Kreiselpumpen – Grundprinzip

Eine Kreiselpumpe wandelt mechanische Energie (Drehmoment × Drehzahl) über ein Laufrad in hydraulische Energie um.
Die Strömung wird durch Fliehkraft nach außen beschleunigt; die kinetische Energie wird im Spiralgehäuse in Druckenergie umgewandelt.

### Kennlinie

Die **Q-H-Kennlinie** zeigt die Förderhöhe H [m] in Abhängigkeit vom Volumenstrom Q [m³/h].
Sie fällt mit zunehmendem Durchfluss ab (instabile Kurven sind zu vermeiden).

### BEP – Best Efficiency Point

Der **Beste Wirkungsgrad-Punkt (BEP)** ist der Auslegungspunkt der Pumpe:
- Bei BEP: minimale Radialkräfte, geringste Kavitationsneigung
- Betrieb 70–120% des BEP-Durchflusses empfohlen (API 610)

### Ähnlichkeitsgesetze (Affinity Laws)

Bei Drehzahländerung von n₀ auf n gilt:

$$Q_2 = Q_1 \\cdot \\frac{n_2}{n_1} \\qquad H_2 = H_1 \\cdot \\left(\\frac{n_2}{n_1}\\right)^2 \\qquad P_2 = P_1 \\cdot \\left(\\frac{n_2}{n_1}\\right)^3$$

Diese Gesetze gelten exakt für geometrisch ähnliche Betriebszustände.

### Spezifische Drehzahl

Die dimensionslose spezifische Drehzahl charakterisiert den Pumpentyp:

$$n_s = n \\cdot \\frac{\\sqrt{Q}}{H^{0.75}}$$

| n_s (min⁻¹) | Pumpentyp |
|---|---|
| 10–30 | Radialrad (Hochdruck) |
| 30–80 | Radialrad (Normaldruck) |
| 80–160 | Halbaxialrad |
| 160–300 | Axialrad |
""")

    with tab_calc:
        st.markdown("""
## Berechnungsgrundlagen

### 1 · Viskositätskorrektur nach HI-Methode

Für viskose Medien (ν > 1 cSt) werden Q-H- und η-Kurven korrigiert.

**Parameter B:**
$$B = 16{,}5 \\cdot \\frac{\\sqrt{\\nu_{cSt}}}{Q_{gpm}^{0{,}25} \\cdot H_{ft}^{0{,}375}}$$

**Korrekturfaktoren:**
$$C_H = \\exp\\!\\left(-0{,}165 \\cdot (\\log_{10} B)^{2{,}2}\\right), \\quad B > 1$$
$$C_\\eta = 1 - 0{,}25 \\cdot \\log_{10} B - 0{,}05 \\cdot (\\log_{10} B)^2$$

Für B ≤ 1 gilt C_H = C_η = 1 (keine Korrektur erforderlich).

**Viskose Kennlinie:**
$$H_{vis} = H_w \\cdot C_H \\qquad \\eta_{vis} = \\eta_w \\cdot C_\\eta$$

---

### 2 · Rohrreibungsverluste (Darcy-Weisbach)

$$h_v = \\lambda \\cdot \\frac{L}{d} \\cdot \\frac{v^2}{2g} + \\sum\\!K \\cdot \\frac{v^2}{2g}$$

Reibungszahl λ nach **Colebrook-White** (iterativ):
$$\\frac{1}{\\sqrt{\\lambda}} = -2 \\log_{10}\\!\\left(\\frac{k/d}{3{,}7} + \\frac{2{,}51}{Re \\cdot \\sqrt{\\lambda}}\\right)$$

Laminare Strömung (Re < 2300): λ = 64/Re

---

### 3 · NPSH – Netto-Positiver Saughub

**NPSHa** (verfügbar, aus Anlage):
$$NPSH_a = \\frac{p_{atm} - p_v}{\\rho \\cdot g} + h_{geo} - h_{v,s}$$

**NPSHr** (erforderlich, Pumpenangabe) muss immer kleiner als NPSHa sein.

Empfohlene Sicherheitsreserve: NPSHa − NPSHr ≥ 0,5 m (ISO 9906), ≥ 1,0 m (API 610)

---

### 4 · Lebenszykluskosten (LCC nach Europump)

$$LCC = C_{invest} + C_{install} + NPV(C_{Energie}) + NPV(C_{Wartung})$$

**Barwert (NPV):**
$$NPV = C_{jährlich} \\cdot \\frac{1 - (1+r)^{-n}}{r}$$

mit Diskontierungszins r und Nutzungsdauer n.

---

### 5 · Gaslöslichkeit (Henry-Gesetz)

$$C = \\frac{p_{partial}}{H(T)} \\quad [\\text{mol/L}]$$

Die Henry-Konstante temperaturabhängig:
$$H(T) = H_0 \\cdot \\exp\\!\\left[B_H \\cdot \\left(\\frac{1}{T} - \\frac{1}{T_0}\\right)\\right]$$
""")

    with tab_std:
        st.markdown("""
## Normenübersicht

### API 610 (12. Ausgabe) – Kreiselpumpen für die Öl- und Gasindustrie
- Schwerste Bauweise, lange Lebensdauer (> 20 Jahre)
- Betriebsfenster: 70–120% des BEP-Durchflusses
- NPSH-Reserve ≥ 1,0 m (oft mehr)
- Alle Druckteile aus Stahl oder legiertem Stahl

### ISO 5199 – Technische Anforderungen (Chemienorm)
- Standard für allgemeine Prozessanwendungen
- Weniger streng als API 610
- Mehrere Werkstoffklassen verfügbar (GG bis Sonderlegierungen)

### ISO 2858 – Maßnorm für Kreiselpumpen
- Einheitliche Anschlussmaße und Einbaumaße
- Erleichtert Pumpentausch zwischen Herstellern

### DIN EN 733 (ehemals DIN 24255)
- Normpumpen (Blockbauweise) für allgemeine Anwendungen
- Typische Nenndrehzahl 1450 / 2900 min⁻¹

### ISO 9906 – Abnahmeprüfungen
- Messtoleranzklassen 1, 2, 3 (Klasse 1 am genauesten)
- Grundlage für Werksabnahmetest

### EN ISO 21457 – Werkstoffauswahl für Erdölausrüstung
- Leitfaden für korrosionsbeständige Werkstoffe
""")

    with tab_mat:
        st.markdown("""
## Werkstoff-Auswahlführer

| Werkstoff | Preis-Faktor | pH-Bereich | Chlorid | Temperatur | Anmerkung |
|---|---|---|---|---|---|
| Grauguss GG25 | 1,0× | 6–8 | < 50 ppm | bis 120 °C | Standard, günstig |
| Sphäroguss GGG40 | 1,05× | 5–9 | < 100 ppm | bis 180 °C | Bessere Zähigkeit |
| Edelstahl 1.4301 (304) | 1,8× | 4–10 | < 50 ppm | bis 300 °C | Allgemeine Chemie |
| Edelstahl 1.4401 (316) | 2,2× | 3–11 | < 200 ppm | bis 400 °C | Erhöhte Chloridbeständigkeit |
| Duplex 1.4462 | 3,5× | 2–12 | < 1000 ppm | bis 250 °C | Stress-Korrosionsriss |
| Hastelloy C-276 | 8,0× | 0–14 | unbegrenzt | bis 800 °C | Aggressive Säuren/Laugen |
| Titan Gr. 2 | 12,0× | 0–14 | unbegrenzt | bis 600 °C | Meerwasser, HNO₃ |

### Schnell-Entscheidungsmatrix

| Bedingung | Empfehlung |
|---|---|
| pH < 4 oder pH > 10 | Edelstahl 316 oder Hastelloy C |
| Chlorid > 200 ppm | Duplex oder Titan |
| Chlorid 50–200 ppm | Edelstahl 316 |
| T > 150 °C | Edelstahl 316 oder Hastelloy C |
| Feststoffanteil > 5% | Sphäroguss GGG40 |
| Neutrale Wasseranwendung | Grauguss GG25 (kostengünstig) |

### Dichtungsauswahl

| Dichtungstyp | Anwendung | Druckgrenze |
|---|---|---|
| Stopfbuchse | Einfache Medien, niedrige Anforderungen | bis 10 bar |
| Einfache Gleitringdichtung | Standard-Chemie | bis 20 bar |
| Doppelte Gleitringdichtung | Gefährliche/toxische Medien | bis 35 bar |
| Magnetkupplung (dichtungslos) | Maximale Dichtheit, Ex-Bereich | bis 40 bar |
""")

    with tab_gloss:
        st.markdown("""
## Glossar

| Begriff | Erklärung |
|---|---|
| **BEP** | Best Efficiency Point – Punkt höchsten Wirkungsgrads |
| **GVF** | Gas Volume Fraction – Gasvolumenanteil am Gesamtstrom [%] |
| **HI** | Hydraulic Institute – amerikanischer Normierungsverband |
| **LCC** | Life Cycle Costs – Lebenszykluskosten |
| **NPSHa** | Net Positive Suction Head available – verfügbarer Saughub der Anlage |
| **NPSHr** | Net Positive Suction Head required – Mindest-Saughub der Pumpe |
| **Kavitation** | Dampfblasenbildung bei Unterschreitung des Dampfdrucks → Schäden |
| **Q** | Förderstrom / Volumenstrom [m³/h] |
| **H** | Förderhöhe [m] |
| **η (eta)** | Wirkungsgrad der Pumpe [–] oder [%] |
| **ν (nu)** | Kinematische Viskosität [cSt = mm²/s] |
| **ρ (rho)** | Dichte des Fördermediums [kg/m³] |
| **Δp** | Druckdifferenz (Druckerhöhung) [bar] |
| **B** | HI-Viskositätsparameter (dimensionslos) |
| **CH, Cη** | Korrekturbeiwerte für Kopf und Wirkungsgrad bei viskoser Strömung |
| **Anlagenkennlinie** | Systemkurve H_sys(Q) = H_stat + H_v,Rohr(Q) |
| **Betriebspunkt** | Schnittpunkt von Pumpen- und Anlagenkennlinie |
| **n_s** | Spezifische Drehzahl (charakterisiert Pumpentyp) |
| **IEC** | International Electrotechnical Commission – Motornorm |
| **ATEX** | AT-mosphères EX-plosibles – EU-Explosionsschutzrichtlinie |
""")


# ══════════════════════════════════════════════════════════════════
# Section 19 · Main
# ══════════════════════════════════════════════════════════════════
def main():
    init_session_state()
    _inject_css()

    # Load data (cached)
    pumps      = load_pumps()
    media      = load_media()
    mph_pumps  = load_mph_pumps()
    atex_mots  = load_atex_motors()

    page = _render_sidebar()

    if page == "Einphasenpumpen":
        render_single_phase_page(pumps, media)
    elif page == "Mehrphasenpumpen":
        render_multi_phase_page(mph_pumps, media)
    elif page == "ATEX-Auslegung":
        render_atex_page(atex_mots)
    elif page == "Pumpenvergleich":
        render_comparison_page(pumps, media)
    else:
        render_documentation_page()


if __name__ == "__main__":
    main()

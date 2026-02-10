import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import json
import io
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

st.set_page_config(page_title="Industrielles Pumpenauslegungstool", layout="wide")

# =========================
# Industrielle Konfiguration
# =========================
VERSION = "2.0.0"
TOOL_NAME = "PumpDesign Pro"

class PumpStandard(Enum):
    API_610 = "API 610 (Prozesschemie)"
    ISO_5199 = "ISO 5199 (Technische Spezifikation)"
    ISO_2858 = "ISO 2858 (Kreiselpumpen)"
    DIN_EN_733 = "DIN EN 733 (Chemienorm)"
    VDMA_24297 = "VDMA 24297"

class MaterialClass(Enum):
    CAST_IRON = ("Grauguss GG25", 1.0, 50)
    DUCTILE_IRON = ("Sphäroguss GGG40", 1.05, 80)
    STAINLESS_304 = ("Edelstahl 1.4301 (304)", 1.8, 300)
    STAINLESS_316 = ("Edelstahl 1.4401 (316)", 2.2, 400)
    DUPLEX = ("Duplex 1.4462", 3.5, 500)
    HASTELLOY_C = ("Hastelloy C-276", 8.0, 800)
    TITANIUM = ("Titan Gr.2", 12.0, 600)

class SealType(Enum):
    PACKING = ("Stopfbuchse", 0.8, 100)
    SINGLE_MECHANICAL = ("Einfache Gleitringdichtung", 1.0, 200)
    DOUBLE_MECHANICAL = ("Doppelte Gleitringdichtung", 1.5, 350)
    MAGNETIC_COUPLING = ("Magnetkupplung (dichtungslos)", 2.0, 400)

@dataclass
class ProjectInfo:
    project_id: str = ""
    project_name: str = ""
    customer: str = ""
    location: str = ""
    engineer: str = ""
    revision: str = "A"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    notes: str = ""
    standard: str = "ISO 5199"

@dataclass
class ProcessConditions:
    flow_rate_m3h: float = 0.0
    head_m: float = 0.0
    suction_pressure_bar: float = 1.0
    discharge_pressure_bar: float = 0.0
    temperature_c: float = 20.0
    vapor_pressure_bar: float = 0.023
    density_kg_m3: float = 998.0
    viscosity_cst: float = 1.0
    solids_pct: float = 0.0
    ph_value: float = 7.0
    chloride_ppm: float = 0.0

@dataclass
class PipelineData:
    suction_length_m: float = 10.0
    suction_diameter_mm: float = 150.0
    suction_roughness_mm: float = 0.05
    suction_fittings_k: float = 5.0
    discharge_length_m: float = 100.0
    discharge_diameter_mm: float = 125.0
    discharge_roughness_mm: float = 0.05
    discharge_fittings_k: float = 15.0
    static_head_m: float = 20.0
    geodetic_suction_m: float = 2.0

@dataclass
class EconomicData:
    energy_cost_eur_kwh: float = 0.15
    operating_hours_year: int = 8000
    expected_lifetime_years: int = 15
    maintenance_cost_pct: float = 3.0
    discount_rate_pct: float = 5.0
    installation_factor: float = 1.5

# =========================
# Config / Globals
# =========================
DEBUG = False

G = 9.81
BAR_TO_PA = 1e5
P_N_BAR = 1.01325
T_N_K = 273.15
R_BAR_L = 0.08314

N0_RPM_DEFAULT = 2900
P_SUCTION_FIXED_BAR_ABS = 0.6
SAT_PENALTY_WEIGHT = 1.5
AIR_SOLUBILITY_REF_T_C = 20.0
AIR_SOLUBILITY_REF_P_BAR = 5.0
AIR_SOLUBILITY_REF_C_CM3N_L = 92.0
AIR_SOLUBILITY_REF_TABLE = [
    (2.0, 36.8),
    (2.5, 46.0),
    (3.0, 55.2),
    (3.5, 64.4),
    (4.0, 73.6),
    (4.5, 82.8),
    (5.0, 92.0),
    (5.5, 101.2),
    (6.0, 110.4),
    (6.5, 119.6),
    (7.0, 128.8),
    (7.5, 138.0),
    (8.0, 147.2),
    (8.5, 156.4),
    (9.0, 165.6),
    (9.5, 177.0),
    (10.0, 185.0),
]

WATER_NU_CST = 1.0
WATER_EPS = 0.15

# =========================
# Basis-Hilfsfunktionen (MÜSSEN ZUERST DEFINIERT WERDEN)
# =========================
def safe_clamp(x, a, b):
    """Begrenzt x auf Intervall [a, b]"""
    try:
        return max(a, min(b, x))
    except Exception:
        return a

def safe_interp(x, xp, fp):
    """Lineare Interpolation mit Fehlerbehandlung"""
    try:
        xp = list(xp)
        fp = list(fp)
        if len(xp) != len(fp):
            n = min(len(xp), len(fp))
            xp, fp = xp[:n], fp[:n]
        if len(xp) < 2:
            return fp[0] if fp else 0.0
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                if xp[i + 1] == xp[i]:
                    return fp[i]
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])
        return fp[-1]
    except Exception:
        return fp[-1] if fp else 0.0

def align_xy(x_vals, y_vals):
    """Gleicht Längen von x und y an"""
    x_list, y_list = list(x_vals), list(y_vals)
    n = min(len(x_list), len(y_list))
    return x_list[:n], y_list[:n]

def m3h_to_lmin(m3h):
    """Konvertiert m³/h zu L/min"""
    return float(m3h) * 1000.0 / 60.0

def lmin_to_m3h(lmin):
    """Konvertiert L/min zu m³/h"""
    return float(lmin) * 60.0 / 1000.0

def motor_iec(P_kW):
    """Nächste IEC-Motorgröße"""
    steps = [0.12, 0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5,
             7.5, 11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250, 315, 400]
    for s in steps:
        if P_kW <= s:
            return s
    return steps[-1]

def show_error(e, context):
    """Zeigt Fehler an"""
    if DEBUG:
        st.exception(e)
    else:
        st.error(f"Fehler in {context}: {e}")

def is_effectively_water(nu_cSt: float) -> bool:
    """Prüft ob Medium effektiv Wasser ist"""
    return float(nu_cSt) <= (WATER_NU_CST + WATER_EPS)

# =========================
# Gas-Funktionen
# =========================
HENRY_CONSTANTS = {
    "Luft": {"A": 800.0, "B": 1500},
    "N2": {"A": 900.0, "B": 1400},
    "O2": {"A": 700.0, "B": 1600},
    "CO2": {"A": 29.0, "B": 2400},
}

AIR_COMPONENTS = [("N2", 0.79), ("O2", 0.21)]

REAL_GAS_FACTORS = {
    "Luft": lambda p_bar, T_K: 1.0,
    "N2": lambda p_bar, T_K: 1.0,
    "O2": lambda p_bar, T_K: 1.0,
    "CO2": lambda p_bar, T_K: max(0.9, 1.0 - 0.001 * (p_bar - 1.0)),
}

def henry_constant(gas, T_celsius):
    """Henry-Konstante für Gas bei Temperatur"""
    params = HENRY_CONSTANTS.get(gas, {"A": 1400.0, "B": 1500})
    T_K = float(T_celsius) + 273.15
    T0_K = 298.15
    return params["A"] * math.exp(params["B"] * (1 / T_K - 1 / T0_K))

def real_gas_factor(gas, p_bar, T_celsius):
    """Realgasfaktor Z"""
    T_K = float(T_celsius) + 273.15
    if gas in REAL_GAS_FACTORS:
        return float(REAL_GAS_FACTORS[gas](float(p_bar), T_K))
    return 1.0

def gas_solubility_cm3N_per_L(gas, p_bar_abs, T_celsius, y_gas=1.0):
    """Gas-Löslichkeit in cm³N/L"""
    p = max(float(p_bar_abs), 1e-6)
    T_K = float(T_celsius) + 273.15
    H = max(henry_constant(gas, T_celsius), 1e-12)
    Z = max(real_gas_factor(gas, p, T_celsius), 0.5)
    p_partial = safe_clamp(float(y_gas), 0.0, 1.0) * p
    C_mol_L = p_partial / H
    V_molar_oper = (R_BAR_L * T_K) / p * Z
    V_oper_L_per_L = C_mol_L * V_molar_oper
    ratio = (p / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)
    return V_oper_L_per_L * ratio * 1000.0

def air_solubility_correction(p_bar_abs, T_celsius):
    """Korrektur für Luft-Löslichkeit"""
    try:
        base = sum(gas_solubility_cm3N_per_L(g, p_bar_abs, AIR_SOLUBILITY_REF_T_C, y_gas=y) 
                   for g, y in AIR_COMPONENTS)
        if base <= 0:
            return 1.0
        p_vals = [p for p, _ in AIR_SOLUBILITY_REF_TABLE]
        c_vals = [c for _, c in AIR_SOLUBILITY_REF_TABLE]
        ref_raw = safe_interp(float(p_bar_abs), p_vals, c_vals)
        ref_at_5 = safe_interp(float(AIR_SOLUBILITY_REF_P_BAR), p_vals, c_vals)
        scale = (float(AIR_SOLUBILITY_REF_C_CM3N_L) / float(ref_at_5)) if ref_at_5 > 0 else 1.0
        ref = float(ref_raw) * float(scale)
        return float(ref) / float(base) if ref > 0 else 1.0
    except Exception:
        return 1.0

def air_solubility_cm3N_L(p_bar_abs, T_celsius):
    """Luft-Löslichkeit in cm³N/L"""
    total = sum(gas_solubility_cm3N_per_L(g, p_bar_abs, T_celsius, y_gas=y) for g, y in AIR_COMPONENTS)
    return total * float(air_solubility_correction(p_bar_abs, T_celsius))

def gas_solubility_total_cm3N_L(gas_medium, p_bar_abs, T_celsius):
    """Gesamte Gas-Löslichkeit"""
    if gas_medium == "Luft":
        return air_solubility_cm3N_L(p_bar_abs, T_celsius)
    return gas_solubility_cm3N_per_L(gas_medium, p_bar_abs, T_celsius, y_gas=1.0)

def gas_flow_from_concentration(C_cm3N_L, Q_liq_m3h):
    """Gasvolumenstrom aus Konzentration"""
    Q_liq_lmin = m3h_to_lmin(Q_liq_m3h)
    return (C_cm3N_L / 1000.0) * Q_liq_lmin

def gas_flow_required_norm_lmin(Q_liq_m3h, C_target_cm3N_L):
    """Benötigter Gasvolumenstrom in Norm-L/min"""
    return float(Q_liq_m3h) * float(C_target_cm3N_L) / 60.0

def oper_to_norm_ratio(p_bar_abs, T_celsius, gas):
    """Verhältnis operativ zu Normbedingungen"""
    T_K = float(T_celsius) + 273.15
    Z = max(real_gas_factor(gas, p_bar_abs, T_celsius), 0.5)
    return (float(p_bar_abs) / P_N_BAR) * (T_N_K / T_K) * (1.0 / Z)

def gas_oper_m3h_to_norm_lmin(Q_gas_oper_m3h, p_bar_abs, T_celsius, gas):
    """Konvertiert operativen Gasvolumenstrom zu Norm-L/min"""
    Q_gas_oper_lmin = m3h_to_lmin(Q_gas_oper_m3h)
    ratio = oper_to_norm_ratio(p_bar_abs, T_celsius, gas)
    return Q_gas_oper_lmin * max(ratio, 1e-12)

def gvf_to_flow_split(Q_total_m3h, gvf_pct):
    """Teilt Gesamtstrom in Flüssig- und Gasanteil"""
    gvf_frac = safe_clamp(float(gvf_pct) / 100.0, 0.0, 0.99)
    Q_gas_oper_m3h = float(Q_total_m3h) * gvf_frac
    Q_liq_m3h = float(Q_total_m3h) * (1.0 - gvf_frac)
    return Q_liq_m3h, Q_gas_oper_m3h

# =========================
# Mehrphasen Lookup
# =========================
MAPPE10_MPH_AIR_LMIN = {
    "MPH-603": {
        15.0: {
            "p_abs": [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            "gas_lmin": [96.9, 93.75, 90.3, 87.45, 81.15, 75.0, 67.5, 59.25, 49.95, 40.2, 25.05],
            "solubility_lmin": [82.8, 92.0, 101.2, 110.4, 119.6, 128.8, 138.0, 147.2, 156.4, 165.6, 177.0],
        },
    },
}

def mappe10_air_lmin_lookup(pump_id, gvf_pct, p_abs_bar, kind, gvf_tol=0.25, allow_nearest=False):
    """Lookup in MAPPE10 Tabelle"""
    pump_map = MAPPE10_MPH_AIR_LMIN.get(pump_id)
    if not pump_map:
        return None, None
    keys = sorted(pump_map.keys())
    if not keys:
        return None, None
    gvf_sel = min(keys, key=lambda k: abs(float(k) - float(gvf_pct)))
    if (not allow_nearest) and (abs(float(gvf_sel) - float(gvf_pct)) > float(gvf_tol)):
        return None, gvf_sel
    curve = pump_map.get(gvf_sel)
    if not curve or kind not in curve:
        return None, gvf_sel
    return safe_interp(float(p_abs_bar), curve["p_abs"], curve[kind]), gvf_sel

# =========================
# Viskositätskorrektur (HI-Methode)
# =========================
def compute_B_HI(Q_m3h, H_m, nu_cSt):
    """HI-Parameter B berechnen"""
    Q = max(float(Q_m3h), 1e-6)
    H = max(float(H_m), 1e-6)
    nu = max(float(nu_cSt), 1e-6)
    Q_gpm = Q * 4.40287
    H_ft = H * 3.28084
    return 16.5 * (nu ** 0.5) / ((Q_gpm ** 0.25) * (H_ft ** 0.375))

def viscosity_correction_factors(B):
    """Korrekturfaktoren CH und Ceta aus B"""
    if B <= 1.0:
        return 1.0, 1.0
    CH = math.exp(-0.165 * (math.log10(B) ** 2.2))
    CH = safe_clamp(CH, 0.3, 1.0)
    log_B = math.log10(B)
    Ceta = 1.0 - 0.25 * log_B - 0.05 * (log_B ** 2)
    Ceta = safe_clamp(Ceta, 0.1, 1.0)
    return CH, Ceta

def viscous_to_water_point(Q_vis_m3h, H_vis_m, nu_cSt):
    """Rückrechnung von viskosem Betriebspunkt auf Wasser-Äquivalent"""
    B = compute_B_HI(Q_vis_m3h, H_vis_m, nu_cSt)
    if is_effectively_water(nu_cSt):
        CH, Ceta = 1.0, 1.0
    else:
        CH, Ceta = viscosity_correction_factors(B)
    Q_water = float(Q_vis_m3h)
    H_water = float(H_vis_m) / max(CH, 1e-9)
    return {"Q_water": Q_water, "H_water": H_water, "B": B, "CH": CH, "Ceta": Ceta}

def generate_viscous_curve(pump, nu_cSt, rho, use_consistent_power=True):
    """Generiert viskose Kennlinien aus Wasserkennlinien"""
    Qw = np.array(pump["Qw"], dtype=float)
    Hw = np.array(pump["Hw"], dtype=float)
    etaw = np.array(pump["eta"], dtype=float)
    Pw_ref = np.array(pump["Pw"], dtype=float)
    H_vis, eta_vis, P_vis = [], [], []
    
    for i, (q, h, e) in enumerate(zip(Qw, Hw, etaw)):
        B = compute_B_HI(q if q > 0 else 1e-6, max(h, 1e-6), nu_cSt)
        if is_effectively_water(nu_cSt):
            CH, Ceta = 1.0, 1.0
        else:
            CH, Ceta = viscosity_correction_factors(B)
        hv = float(h) * max(float(CH), 1e-9)
        ev = safe_clamp(float(e) * float(Ceta), 0.05, 0.95)
        if use_consistent_power:
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            pv = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0
        else:
            P_hyd_water_W = rho * G * (float(q) / 3600.0) * float(h)
            P_water_theory = (P_hyd_water_W / max(float(e), 1e-9)) / 1000.0
            P_hyd_vis_W = rho * G * (float(q) / 3600.0) * hv
            P_vis_theory = (P_hyd_vis_W / max(ev, 1e-9)) / 1000.0
            scale = float(P_vis_theory) / float(P_water_theory) if P_water_theory > 1e-6 else 1.0 / max(float(Ceta), 1e-9)
            pv = float(Pw_ref[i]) * float(scale)
        H_vis.append(hv)
        eta_vis.append(ev)
        P_vis.append(pv)
    return Qw.tolist(), H_vis, eta_vis, P_vis

# =========================
# Wurzelfindung und Drehzahl
# =========================
def bisect_root(f, a, b, it=70, tol=1e-6):
    """Bisektionsverfahren zur Wurzelfindung"""
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        return None
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        return None
    lo, hi, flo, fhi = a, b, fa, fb
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if not np.isfinite(fm):
            return None
        if abs(fm) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)

def find_best_ratio_by_scan(dp_at_ratio_fn, dp_req, n_min, n_max, steps=60, prefer_above=True):
    """Findet bestes Drehzahlverhältnis durch Scan"""
    dp_req = float(dp_req)
    n_min = float(n_min)
    n_max = float(n_max)
    if n_max <= n_min:
        return None
    values = []
    for i in range(steps + 1):
        nr = n_min + (n_max - n_min) * (i / steps)
        dp = dp_at_ratio_fn(nr)
        if np.isfinite(dp):
            values.append((nr, dp))
    if not values:
        return None
    if prefer_above:
        above = [(nr, dp) for nr, dp in values if dp >= dp_req]
        if above:
            return min(above, key=lambda x: x[1] - dp_req)[0]
    return min(values, key=lambda x: abs(x[1] - dp_req))[0]

def find_speed_ratio(Q_curve, H_curve, Q_req, H_req, n_min=0.5, n_max=1.2):
    """Findet optimales Drehzahlverhältnis"""
    Q_curve = list(map(float, Q_curve))
    H_curve = list(map(float, H_curve))
    Q_req = float(Q_req)
    H_req = float(H_req)
    def f(nr):
        if nr <= 0:
            return 1e9
        Q_base = Q_req / nr
        H_base = safe_interp(Q_base, Q_curve, H_curve)
        return (H_base * (nr ** 2)) - H_req
    return bisect_root(f, float(n_min), float(n_max), it=80, tol=1e-5)

# =========================
# Rohrleitungsberechnung
# =========================
def reynolds_number(v_ms, d_m, nu_m2s):
    """Reynoldszahl berechnen"""
    return (v_ms * d_m) / max(nu_m2s, 1e-9)

def friction_factor_colebrook(Re, d_m, k_mm):
    """Rohrreibungszahl nach Colebrook-White"""
    if Re < 2300:
        return 64.0 / max(Re, 1e-6)
    k_m = k_mm / 1000.0
    f = 0.25 / (math.log10(k_m / (3.7 * d_m) + 5.74 / (Re ** 0.9))) ** 2
    for _ in range(20):
        rhs = -2.0 * math.log10(k_m / (3.7 * d_m) + 2.51 / (Re * math.sqrt(max(f, 1e-9))))
        f_new = 1.0 / (rhs ** 2) if rhs != 0 else f
        if abs(f_new - f) < 1e-8:
            break
        f = f_new
    return f

def pipe_head_loss(Q_m3h, L_m, D_mm, k_mm, K_fittings, rho, nu_cSt):
    """Druckverlust in Rohrleitung [m]"""
    if Q_m3h <= 0 or L_m <= 0 or D_mm <= 0:
        return 0.0
    Q_m3s = Q_m3h / 3600.0
    D_m = D_mm / 1000.0
    A = math.pi * (D_m / 2) ** 2
    v = Q_m3s / max(A, 1e-9)
    nu_m2s = nu_cSt * 1e-6
    Re = reynolds_number(v, D_m, nu_m2s)
    f = friction_factor_colebrook(Re, D_m, k_mm)
    h_pipe = f * (L_m / D_m) * (v ** 2) / (2 * G)
    h_fittings = K_fittings * (v ** 2) / (2 * G)
    return h_pipe + h_fittings

def calculate_system_curve(pipeline: PipelineData, process: ProcessConditions, Q_range):
    """Systemkurve berechnen"""
    H_system = []
    for Q in Q_range:
        h_static = pipeline.static_head_m
        h_suction = pipe_head_loss(
            Q, pipeline.suction_length_m, pipeline.suction_diameter_mm,
            pipeline.suction_roughness_mm, pipeline.suction_fittings_k,
            process.density_kg_m3, process.viscosity_cst
        )
        h_discharge = pipe_head_loss(
            Q, pipeline.discharge_length_m, pipeline.discharge_diameter_mm,
            pipeline.discharge_roughness_mm, pipeline.discharge_fittings_k,
            process.density_kg_m3, process.viscosity_cst
        )
        H_system.append(h_static + h_suction + h_discharge)
    return H_system

def calculate_npsh_available(pipeline: PipelineData, process: ProcessConditions, Q_m3h):
    """NPSHa berechnen"""
    p_atm = 1.01325
    h_suction_loss = pipe_head_loss(
        Q_m3h, pipeline.suction_length_m, pipeline.suction_diameter_mm,
        pipeline.suction_roughness_mm, pipeline.suction_fittings_k,
        process.density_kg_m3, process.viscosity_cst
    )
    p_term = (p_atm - process.vapor_pressure_bar) * BAR_TO_PA / (process.density_kg_m3 * G)
    NPSHa = p_term + pipeline.geodetic_suction_m - h_suction_loss
    return max(NPSHa, 0.0)

# =========================
# LCC-Berechnung
# =========================
def calculate_lcc(pump_price: float, P_kW: float, eta: float, econ: EconomicData,
                  material_factor: float = 1.0, seal_factor: float = 1.0) -> Dict:
    """Lebenszykluskosten berechnen"""
    pump_cost = pump_price * material_factor * seal_factor
    installation_cost = pump_cost * econ.installation_factor
    initial_cost = pump_cost + installation_cost
    
    P_actual = P_kW / max(eta, 0.1)
    annual_energy_kwh = P_actual * econ.operating_hours_year
    annual_energy_cost = annual_energy_kwh * econ.energy_cost_eur_kwh
    annual_maintenance = pump_cost * (econ.maintenance_cost_pct / 100.0)
    
    r = econ.discount_rate_pct / 100.0
    n = econ.expected_lifetime_years
    annuity_factor = (1 - (1 + r) ** (-n)) / r if r > 0 else n
    
    npv_energy = annual_energy_cost * annuity_factor
    npv_maintenance = annual_maintenance * annuity_factor
    lcc_total = initial_cost + npv_energy + npv_maintenance
    
    return {
        "pump_cost": pump_cost,
        "installation_cost": installation_cost,
        "initial_cost": initial_cost,
        "annual_energy_cost": annual_energy_cost,
        "annual_maintenance": annual_maintenance,
        "npv_energy": npv_energy,
        "npv_maintenance": npv_maintenance,
        "lcc_total": lcc_total,
        "energy_share_pct": (npv_energy / lcc_total) * 100 if lcc_total > 0 else 0,
        "co2_annual_kg": annual_energy_kwh * 0.4,
    }

# =========================
# Normprüfung und Werkstoffempfehlung
# =========================
def check_standard_compliance(pump: dict, process: ProcessConditions, standard: PumpStandard) -> List[Dict]:
    """Prüft Einhaltung von Industrienormen"""
    issues = []
    
    if standard == PumpStandard.API_610:
        if not pump.get("api_610_compliant", False):
            issues.append({"severity": "error", "msg": "Pumpe nicht API 610 konform"})
        if pump.get("stages", 1) > 1:
            issues.append({"severity": "warning", "msg": "Mehrstufige Pumpe: Axialschubausgleich prüfen"})
        Q_bep = pump["Qw"][pump["eta"].index(max(pump["eta"]))] if pump["eta"] else pump["Qw"][len(pump["Qw"])//2]
        Q_min_api = Q_bep * 0.7
        Q_max_api = Q_bep * 1.2
        if process.flow_rate_m3h < Q_min_api or process.flow_rate_m3h > Q_max_api:
            issues.append({
                "severity": "warning",
                "msg": f"Betriebspunkt außerhalb API 610 Fenster (70-120% BEP: {Q_min_api:.1f}-{Q_max_api:.1f} m³/h)"
            })
    elif standard == PumpStandard.ISO_5199:
        if not pump.get("iso_5199_compliant", False):
            issues.append({"severity": "error", "msg": "Pumpe nicht ISO 5199 konform"})
    
    if process.temperature_c > pump.get("max_temp_c", 120):
        issues.append({"severity": "error", "msg": f"Temperatur {process.temperature_c}°C > max {pump.get('max_temp_c', 120)}°C"})
    if process.temperature_c < pump.get("min_temp_c", -20):
        issues.append({"severity": "error", "msg": f"Temperatur {process.temperature_c}°C < min {pump.get('min_temp_c', -20)}°C"})
    if process.viscosity_cst > pump.get("max_viscosity", 200):
        issues.append({"severity": "warning", "msg": f"Viskosität {process.viscosity_cst} cSt > empfohlen {pump.get('max_viscosity', 200)} cSt"})
    
    return issues

def recommend_materials(process: ProcessConditions) -> List[str]:
    """Empfiehlt Werkstoffe basierend auf Prozessbedingungen"""
    recommendations = []
    if process.ph_value < 4 or process.ph_value > 10:
        recommendations.extend(["STAINLESS_316", "HASTELLOY_C"])
    if process.chloride_ppm > 200:
        recommendations.extend(["DUPLEX", "TITANIUM"])
    elif process.chloride_ppm > 50:
        recommendations.append("STAINLESS_316")
    if process.temperature_c > 150:
        recommendations.extend(["STAINLESS_316", "HASTELLOY_C"])
    if process.solids_pct > 5:
        recommendations.append("DUCTILE_IRON")
    if not recommendations:
        recommendations = ["CAST_IRON", "STAINLESS_304"]
    return list(set(recommendations))

# =========================
# Session State
# =========================
def init_session_state():
    """Initialisiert Session State"""
    if "project" not in st.session_state:
        st.session_state.project = ProjectInfo()
    if "process" not in st.session_state:
        st.session_state.process = ProcessConditions()
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = PipelineData()
    if "economic" not in st.session_state:
        st.session_state.economic = EconomicData()
    if "selected_pump" not in st.session_state:
        st.session_state.selected_pump = None
    if "calculation_results" not in st.session_state:
        st.session_state.calculation_results = {}

# =========================
# Export-Funktionen
# =========================
def generate_datasheet_html(project: ProjectInfo, process: ProcessConditions, 
                            pump: dict, results: dict) -> str:
    """Generiert technisches Datenblatt als HTML"""
    return f"""<!DOCTYPE html>
<html lang="de">
<head><meta charset="UTF-8"><title>Datenblatt - {pump.get('id', 'N/A')}</title>
<style>body{{font-family:Arial;max-width:210mm;margin:0 auto;padding:20px;}} 
h1{{color:#2c5aa0;}}table{{border-collapse:collapse;width:100%;margin:10px 0;}} 
th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}th{{background:#f5f5f5;width:40%;}}</style>
</head><body>
<h1>Technisches Datenblatt</h1>
<p><strong>Projekt:</strong> {project.project_id} | <strong>Rev:</strong> {project.revision} | <strong>Datum:</strong> {project.date}</p>
<h2>Prozessdaten</h2>
<table><tr><th>Förderstrom</th><td>{process.flow_rate_m3h:.2f} m³/h</td></tr>
<tr><th>Förderhöhe</th><td>{process.head_m:.2f} m</td></tr>
<tr><th>Temperatur</th><td>{process.temperature_c:.1f} °C</td></tr>
<tr><th>Dichte</th><td>{process.density_kg_m3:.1f} kg/m³</td></tr>
<tr><th>Viskosität</th><td>{process.viscosity_cst:.2f} cSt</td></tr></table>
<h2>Pumpe: {pump.get('id', 'N/A')}</h2>
<table><tr><th>Wirkungsgrad</th><td>{results.get('eta', 0)*100:.1f} %</td></tr>
<tr><th>Wellenleistung</th><td>{results.get('P_shaft', 0):.2f} kW</td></tr>
<tr><th>Motorleistung</th><td>{results.get('P_motor', 0):.2f} kW</td></tr>
<tr><th>NPSHr</th><td>{results.get('NPSHr', 0):.2f} m</td></tr>
<tr><th>NPSHa</th><td>{results.get('NPSHa', 0):.2f} m</td></tr></table>
<p><em>Erstellt mit {TOOL_NAME} v{VERSION}</em></p>
</body></html>"""

def generate_project_json(project: ProjectInfo, process: ProcessConditions,
                          pipeline: PipelineData, economic: EconomicData,
                          results: dict) -> str:
    """Exportiert Projektdaten als JSON"""
    data = {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "project": asdict(project),
        "process": asdict(process),
        "pipeline": asdict(pipeline),
        "economic": asdict(economic),
        "results": results,
    }
    return json.dumps(data, indent=2, default=str)

# =========================
# Pumpendatenbanken
# =========================
MEDIA = {
    "Wasser": {"rho": 998.0, "nu": 1.0, "vapor_pressure_20c": 0.023},
    "Wasser (60°C)": {"rho": 983.0, "nu": 0.47, "vapor_pressure_20c": 0.199},
    "Öl (leicht)": {"rho": 850.0, "nu": 10.0, "vapor_pressure_20c": 0.001},
    "Öl (schwer)": {"rho": 900.0, "nu": 100.0, "vapor_pressure_20c": 0.0001},
    "Glykol 30%": {"rho": 1040.0, "nu": 2.5, "vapor_pressure_20c": 0.015},
}

PUMPS = [
    {
        "id": "VIS-50",
        "manufacturer": "Visko Systems",
        "type": "Kreiselpumpe",
        "Qw": [0, 10, 20, 30, 40, 50, 60],
        "Hw": [40, 38, 36, 32, 28, 24, 18],
        "eta": [0.0, 0.35, 0.55, 0.70, 0.72, 0.68, 0.60],
        "Pw": [0.1, 1.0, 2.0, 3.0, 4.0, 4.8, 5.5],
        "NPSHr": [1.0, 1.2, 1.5, 2.0, 2.8, 3.8, 5.0],
        "max_viscosity": 200, "max_density": 1200,
        "max_temp_c": 120, "min_temp_c": -20,
        "price_eur": 4500,
        "materials_available": ["CAST_IRON", "STAINLESS_304", "STAINLESS_316"],
        "seals_available": ["PACKING", "SINGLE_MECHANICAL", "DOUBLE_MECHANICAL"],
        "api_610_compliant": False, "iso_5199_compliant": True,
    },
    {
        "id": "VIS-80",
        "manufacturer": "Visko Systems",
        "type": "Kreiselpumpe",
        "Qw": [0, 15, 30, 45, 60, 75, 90],
        "Hw": [55, 52, 48, 42, 36, 28, 18],
        "eta": [0.0, 0.40, 0.60, 0.75, 0.78, 0.73, 0.60],
        "Pw": [0.2, 1.5, 3.5, 5.5, 7.5, 9.5, 12.0],
        "NPSHr": [1.2, 1.5, 2.0, 2.5, 3.5, 4.5, 6.0],
        "max_viscosity": 500, "max_density": 1200,
        "max_temp_c": 180, "min_temp_c": -40,
        "price_eur": 7800,
        "materials_available": ["CAST_IRON", "DUCTILE_IRON", "STAINLESS_304", "STAINLESS_316", "DUPLEX"],
        "seals_available": ["SINGLE_MECHANICAL", "DOUBLE_MECHANICAL", "MAGNETIC_COUPLING"],
        "api_610_compliant": True, "iso_5199_compliant": True,
    },
    {
        "id": "VIS-100HD",
        "manufacturer": "Visko Systems",
        "type": "Hochdruck-Kreiselpumpe",
        "Qw": [0, 20, 40, 60, 80, 100, 120],
        "Hw": [120, 115, 108, 98, 85, 70, 50],
        "eta": [0.0, 0.42, 0.62, 0.76, 0.80, 0.75, 0.65],
        "Pw": [0.5, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0],
        "NPSHr": [1.5, 2.0, 2.5, 3.2, 4.0, 5.0, 6.5],
        "max_viscosity": 300, "max_density": 1400,
        "max_temp_c": 200, "min_temp_c": -60,
        "price_eur": 15500,
        "materials_available": ["STAINLESS_304", "STAINLESS_316", "DUPLEX", "HASTELLOY_C"],
        "seals_available": ["DOUBLE_MECHANICAL", "MAGNETIC_COUPLING"],
        "api_610_compliant": True, "iso_5199_compliant": True,
    },
]

MPH_PUMPS = [
    {
        "id": "MPH-602", "type": "Mehrphasenpumpe", "manufacturer": "Visko Systems",
        "Q_max_m3h": 60, "dp_max_bar": 8.4, "GVF_max": 0.1, "n0_rpm": 2900,
        "max_viscosity": 500, "max_density": 1200, "NPSHr": 2.5, "price_eur": 28000,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 10, 20, 30, 40, 50, 60], "dp": [8.4, 8.3, 8.0, 7.5, 6.8, 6.0, 5.0]},
            10: {"Q": [15, 20, 25, 30, 35, 40, 45], "dp": [5.6, 5.5, 5.3, 5.1, 4.8, 4.4, 3.9]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 10, 20, 30, 40, 50, 60], "P": [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]},
            10: {"Q": [15, 20, 25, 30, 35, 40, 45], "P": [7.0, 7.5, 7.8, 8.1, 8.5, 9.0, 9.1]},
        },
    },
    {
        "id": "MPH-403", "type": "Mehrphasenpumpe", "manufacturer": "Visko Systems",
        "Q_max_m3h": 24, "dp_max_bar": 9.4, "GVF_max": 0.15, "n0_rpm": 2900,
        "max_viscosity": 500, "max_density": 1200, "NPSHr": 2.0, "price_eur": 22000,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 4, 8, 12, 16, 20, 24], "dp": [9.4, 9.3, 9.0, 8.5, 7.6, 6.6, 5.3]},
            15: {"Q": [9, 12, 16, 20], "dp": [6.5, 6.5, 5.5, 3.3]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 4, 8, 12, 16, 20, 24], "P": [3.0, 3.8, 4.2, 5.0, 5.8, 6.2, 7.0]},
            15: {"Q": [9, 12, 16, 20], "P": [3.8, 4.0, 4.2, 4.8]},
        },
    },
    {
        "id": "MPH-603", "type": "Mehrphasenpumpe", "manufacturer": "Visko Systems",
        "Q_max_m3h": 50, "dp_max_bar": 12.0, "GVF_max": 0.2, "n0_rpm": 2900,
        "max_viscosity": 500, "max_density": 1200, "NPSHr": 3.0, "price_eur": 35000,
        "curves_dp_vs_Q": {
            0: {"Q": [0, 10, 15, 20, 25, 30, 35, 40, 45], "dp": [12, 12, 11.8, 11.5, 11, 10.5, 9.8, 9.0, 8.0]},
            5: {"Q": [10, 15, 20, 25, 30, 35, 40, 45], "dp": [11.5, 11.5, 11, 10.5, 9.8, 9.0, 8]},
            10: {"Q": [10, 15, 20, 25, 30, 35, 40], "dp": [10.5, 10.1, 9.8, 9.2, 8.4, 7.3, 6.3]},
            15: {"Q": [10, 15, 20, 25, 30, 35], "dp": [9.5, 9.1, 8.5, 7.8, 6.8, 5.5]},
            20: {"Q": [10, 15, 20, 25, 30], "dp": [6.8, 7, 6.4, 5.5, 4.2]},
        },
        "power_kW_vs_Q": {
            0: {"Q": [0, 10, 15, 20, 25, 30, 35, 40, 45], "P": [7.0, 9.0, 10.0, 11.0, 11.8, 12.2, 13.0, 13.8, 14.0]},
            5: {"Q": [10, 15, 20, 25, 30, 35, 40, 45], "P": [8.8, 9.0, 11.0, 13.0, 11.8, 12, 13.5, 14.0]},
            10: {"Q": [10, 15, 20, 25, 30, 35, 40], "P": [8.5, 8.5, 10.0, 10.5, 11.0, 11.5, 12.0]},
            15: {"Q": [10, 15, 20, 25, 30, 35], "P": [8.3, 8.7, 9.2, 9.9, 10.8, 11.5]},
            20: {"Q": [10, 15, 20, 25, 30], "P": [8, 9, 9.5, 10, 10.3]},
        },
    },
]

ATEX_MOTORS = [
    {"id": "Standard Zone 2 (Ex ec)", "marking": "II 3G Ex ec IIC T3 Gc",
     "zone_suitable": [2], "temp_class": "T3", "t_max_surface": 200.0,
     "category": "3G", "efficiency_class": "IE3", "price_factor": 1.0},
    {"id": "Zone 1 (Ex eb)", "marking": "II 2G Ex eb IIC T4 Gb",
     "zone_suitable": [1, 2], "temp_class": "T4", "t_max_surface": 135.0,
     "category": "2G", "efficiency_class": "IE3", "price_factor": 1.3},
    {"id": "Zone 0/1 (Ex d)", "marking": "II 1G Ex d IIC T4 Ga",
     "zone_suitable": [0, 1, 2], "temp_class": "T4", "t_max_surface": 135.0,
     "category": "1G", "efficiency_class": "IE3", "price_factor": 1.8},
]

# =========================
# Pumpenauswahl
# =========================
def choose_best_pump(pumps, Q_req, H_req, nu_cSt, rho, allow_out_of_range=True):
    """Wählt beste Einphasen-Pumpe"""
    best = None
    for p in pumps:
        try:
            if nu_cSt > p.get("max_viscosity", 500):
                continue
            if rho > p.get("max_density", 1200):
                continue
            qmin, qmax = min(p["Qw"]), max(p["Qw"])
            in_range = (qmin <= Q_req <= qmax)
            if (not in_range) and (not allow_out_of_range):
                continue
            Q_eval = safe_clamp(Q_req, qmin, qmax)
            H_at = safe_interp(Q_eval, p["Qw"], p["Hw"])
            eta_at = safe_interp(Q_eval, p["Qw"], p["eta"])
            score = abs(H_at - H_req)
            penalty = 0.0 if in_range else abs(Q_req - Q_eval) / max((qmax - qmin), 1e-9) * 10.0
            cand = {"id": p["id"], "pump": p, "in_range": in_range,
                    "Q_eval": Q_eval, "H_at": H_at, "eta_at": eta_at, "score": score + penalty}
            if best is None or cand["score"] < best["score"]:
                best = cand
        except Exception:
            continue
    return best

# =========================
# Mehrphasen-Interpolation
# =========================
def _interp_between_gvf_keys(pump, gvf_pct):
    keys = sorted(pump["curves_dp_vs_Q"].keys())
    if gvf_pct <= keys[0]:
        return keys[0], keys[0], 0.0
    if gvf_pct >= keys[-1]:
        return keys[-1], keys[-1], 0.0
    lo = max([k for k in keys if k <= gvf_pct])
    hi = min([k for k in keys if k >= gvf_pct])
    if hi == lo:
        return lo, hi, 0.0
    w = (gvf_pct - lo) / (hi - lo)
    return lo, hi, w

def _dp_at_Q_gvf(pump, Q_m3h, gvf_pct):
    lo, hi, w = _interp_between_gvf_keys(pump, gvf_pct)
    c_lo = pump["curves_dp_vs_Q"][lo]
    c_hi = pump["curves_dp_vs_Q"][hi]
    dp_lo = safe_interp(Q_m3h, c_lo["Q"], c_lo["dp"])
    dp_hi = safe_interp(Q_m3h, c_hi["Q"], c_hi["dp"])
    return (1 - w) * dp_lo + w * dp_hi, lo, hi, w

def _P_at_Q_gvf(pump, Q_m3h, gvf_pct):
    lo, hi, w = _interp_between_gvf_keys(pump, gvf_pct)
    p_lo = pump["power_kW_vs_Q"][lo]
    p_hi = pump["power_kW_vs_Q"][hi]
    P_lo = safe_interp(Q_m3h, p_lo["Q"], p_lo["P"])
    P_hi = safe_interp(Q_m3h, p_hi["Q"], p_hi["P"])
    return (1 - w) * P_lo + w * P_hi, lo, hi, w

# =========================
# UI Funktionen
# =========================
def render_project_header():
    """Rendert Projekt-Header in Sidebar"""
    with st.sidebar:
        st.markdown(f"### {TOOL_NAME}")
        st.markdown(f"**Version {VERSION}**")
        st.divider()
        
        with st.expander("📋 Projektdaten", expanded=False):
            st.session_state.project.project_id = st.text_input(
                "Projekt-ID", value=st.session_state.project.project_id, placeholder="PRJ-2024-001")
            st.session_state.project.project_name = st.text_input(
                "Projektname", value=st.session_state.project.project_name)
            st.session_state.project.customer = st.text_input(
                "Kunde", value=st.session_state.project.customer)
            st.session_state.project.engineer = st.text_input(
                "Bearbeiter", value=st.session_state.project.engineer)
            st.session_state.project.revision = st.text_input(
                "Revision", value=st.session_state.project.revision)
            st.session_state.project.standard = st.selectbox(
                "Auslegungsnorm", [s.value for s in PumpStandard], index=1)

def run_single_phase_pump():
    """Einphasen-Pumpenauslegung"""
    try:
        st.header("🔧 Einphasenpumpen mit Viskositätskorrektur")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Prozessdaten", "Rohrleitung", "Wirtschaftlichkeit", "Ergebnisse"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Betriebspunkt")
                Q_vis_req = st.number_input("Förderstrom Q [m³/h]", min_value=0.1, value=30.0, step=1.0)
                H_vis_req = st.number_input("Förderhöhe H [m]", min_value=0.1, value=20.0, step=1.0)
                st.session_state.process.flow_rate_m3h = Q_vis_req
                st.session_state.process.head_m = H_vis_req
                
            with col2:
                st.subheader("Medium")
                medium = st.selectbox("Medium", list(MEDIA.keys()), index=0)
                rho = st.number_input("Dichte [kg/m³]", min_value=1.0, 
                                      value=float(MEDIA[medium]["rho"]), step=5.0)
                nu = st.number_input("Viskosität ν [cSt]", min_value=0.1, 
                                     value=float(MEDIA[medium]["nu"]), step=0.5)
                T_proc = st.number_input("Temperatur [°C]", min_value=-60.0, max_value=400.0, value=20.0)
                p_vapor = st.number_input("Dampfdruck [bar(a)]", min_value=0.0, 
                                          value=float(MEDIA[medium].get("vapor_pressure_20c", 0.023)), 
                                          step=0.01, format="%.4f")
                st.session_state.process.density_kg_m3 = rho
                st.session_state.process.viscosity_cst = nu
                st.session_state.process.temperature_c = T_proc
                st.session_state.process.vapor_pressure_bar = p_vapor
                
            with col3:
                st.subheader("Optionen")
                allow_out = st.checkbox("Auswahl außerhalb Kennlinie", value=True)
                reserve_pct = st.slider("Motorreserve [%]", 0, 30, 10)
                n_min = st.slider("n_min/n₀", 0.4, 1.0, 0.6, 0.01)
                n_max = st.slider("n_max/n₀", 1.0, 1.6, 1.2, 0.01)
                use_consistent_power = st.checkbox("Konsistente P-Berechnung", value=True)
        
        with tab2:
            st.subheader("Rohrleitungsdaten")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Saugseite**")
                st.session_state.pipeline.suction_length_m = st.number_input("Länge [m]", min_value=0.0, value=10.0, key="s_len")
                st.session_state.pipeline.suction_diameter_mm = st.number_input("Durchmesser [mm]", min_value=10.0, value=150.0, key="s_dia")
                st.session_state.pipeline.suction_fittings_k = st.number_input("Σk Armaturen", min_value=0.0, value=5.0, key="s_k")
                st.session_state.pipeline.geodetic_suction_m = st.number_input("Geodät. Saughöhe [m]", value=2.0, key="s_geo")
            with col2:
                st.markdown("**Druckseite**")
                st.session_state.pipeline.discharge_length_m = st.number_input("Länge [m]", min_value=0.0, value=100.0, key="d_len")
                st.session_state.pipeline.discharge_diameter_mm = st.number_input("Durchmesser [mm]", min_value=10.0, value=125.0, key="d_dia")
                st.session_state.pipeline.discharge_fittings_k = st.number_input("Σk Armaturen", min_value=0.0, value=15.0, key="d_k")
                st.session_state.pipeline.static_head_m = st.number_input("Statische Höhe [m]", min_value=0.0, value=20.0, key="d_static")
        
        with tab3:
            st.subheader("Wirtschaftlichkeit (LCC)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.economic.energy_cost_eur_kwh = st.number_input("Strompreis [€/kWh]", min_value=0.01, value=0.15, step=0.01)
                st.session_state.economic.operating_hours_year = st.number_input("Betriebsstunden [h/a]", min_value=100, value=8000, step=100)
            with col2:
                st.session_state.economic.expected_lifetime_years = st.number_input("Lebensdauer [Jahre]", min_value=1, value=15)
                st.session_state.economic.maintenance_cost_pct = st.number_input("Wartung [%/a]", min_value=0.0, value=3.0, step=0.5)
            with col3:
                st.session_state.economic.discount_rate_pct = st.number_input("Diskontierungszins [%]", min_value=0.0, value=5.0, step=0.5)
                st.session_state.economic.installation_factor = st.number_input("Installationsfaktor", min_value=1.0, value=1.5, step=0.1)
        
        with tab4:
            # Berechnung
            conv = viscous_to_water_point(Q_vis_req, H_vis_req, nu)
            Q_water, H_water = conv["Q_water"], conv["H_water"]
            B, CH, Ceta = conv["B"], conv["CH"], conv["Ceta"]
            
            best = choose_best_pump(PUMPS, Q_water, H_water, nu, rho, allow_out_of_range=allow_out)
            if not best:
                st.error("❌ Keine geeignete Pumpe gefunden.")
                return
            
            pump = best["pump"]
            eta_water = float(best["eta_at"])
            eta_vis = safe_clamp(eta_water * Ceta, 0.05, 0.95)
            P_hyd_W = rho * G * (Q_vis_req / 3600.0) * H_vis_req
            P_vis_kW = (P_hyd_W / max(eta_vis, 1e-9)) / 1000.0
            P_motor_kW = motor_iec(P_vis_kW * (1.0 + reserve_pct / 100.0))
            
            # NPSH
            NPSHa = calculate_npsh_available(st.session_state.pipeline, st.session_state.process, Q_vis_req)
            NPSHr = safe_interp(Q_vis_req, pump["Qw"], pump.get("NPSHr", [2.0]*len(pump["Qw"])))
            NPSH_margin = NPSHa - NPSHr
            
            # Drehzahl
            Q_vis_curve, H_vis_curve, eta_vis_curve, P_vis_curve = generate_viscous_curve(pump, nu, rho, use_consistent_power)
            n_ratio_opt = find_speed_ratio(Q_vis_curve, H_vis_curve, Q_vis_req, H_vis_req, n_min, n_max)
            
            results = {"Q_op": Q_vis_req, "H_op": H_vis_req, "eta": eta_vis,
                       "P_shaft": P_vis_kW, "P_motor": P_motor_kW, "NPSHa": NPSHa, "NPSHr": NPSHr}
            st.session_state.calculation_results = results
            st.session_state.selected_pump = pump
            
            st.subheader("📊 Ergebnisse")
            
            # Normprüfung
            standard = PumpStandard.ISO_5199
            for s in PumpStandard:
                if s.value == st.session_state.project.standard:
                    standard = s
                    break
            compliance = check_standard_compliance(pump, st.session_state.process, standard)
            for c in compliance:
                if c["severity"] == "error":
                    st.error(c["msg"])
                elif c["severity"] == "warning":
                    st.warning(c["msg"])
            
            # Metriken
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Pumpe", pump["id"])
            with col2:
                st.metric("Wirkungsgrad", f"{eta_vis*100:.1f}%")
                st.metric("Wellenleistung", f"{P_vis_kW:.2f} kW")
            with col3:
                st.metric("Motorleistung", f"{P_motor_kW:.1f} kW")
                if n_ratio_opt:
                    st.metric("Opt. Drehzahl", f"{N0_RPM_DEFAULT * n_ratio_opt:.0f} rpm")
            with col4:
                st.metric("NPSHa", f"{NPSHa:.2f} m")
                st.metric("NPSHr", f"{NPSHr:.2f} m")
            with col5:
                if NPSH_margin >= 1.0:
                    st.success(f"NPSH-Reserve: {NPSH_margin:.2f} m ✓")
                elif NPSH_margin >= 0.5:
                    st.warning(f"NPSH-Reserve: {NPSH_margin:.2f} m")
                else:
                    st.error(f"NPSH-Reserve: {NPSH_margin:.2f} m (KAVITATION!)")
            
            # Werkstoff & Dichtung
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔩 Werkstoff")
                recommended_mats = recommend_materials(st.session_state.process)
                available_mats = pump.get("materials_available", ["CAST_IRON"])
                suitable_mats = [m for m in recommended_mats if m in available_mats]
                if not suitable_mats:
                    suitable_mats = available_mats[:1]
                selected_material = st.selectbox("Werkstoff", suitable_mats,
                    format_func=lambda x: MaterialClass[x].value[0] if x in MaterialClass.__members__ else x)
            with col2:
                st.subheader("🔧 Dichtung")
                available_seals = pump.get("seals_available", ["SINGLE_MECHANICAL"])
                selected_seal = st.selectbox("Dichtung", available_seals,
                    format_func=lambda x: SealType[x].value[0] if x in SealType.__members__ else x)
            
            # LCC
            st.divider()
            st.subheader("💰 Lebenszykluskosten")
            mat_factor = MaterialClass[selected_material].value[1] if selected_material in MaterialClass.__members__ else 1.0
            seal_factor = SealType[selected_seal].value[1] if selected_seal in SealType.__members__ else 1.0
            
            lcc = calculate_lcc(pump.get("price_eur", 5000), P_vis_kW, eta_vis,
                               st.session_state.economic, mat_factor, seal_factor)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Investition", f"{lcc['initial_cost']:,.0f} €")
            with col2:
                st.metric("Energie (NPV)", f"{lcc['npv_energy']:,.0f} €")
            with col3:
                st.metric("Wartung (NPV)", f"{lcc['npv_maintenance']:,.0f} €")
            with col4:
                st.metric("LCC Gesamt", f"{lcc['lcc_total']:,.0f} €")
            
            # Kennlinien
            st.divider()
            st.subheader("📈 Kennlinien")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Q-H
            ax1 = axes[0, 0]
            ax1.plot(pump["Qw"], pump["Hw"], "b-o", label="Wasser", linewidth=2)
            q_vis_plot = [q for q in Q_vis_curve if q > 0]
            h_vis_plot = [h for q, h in zip(Q_vis_curve, H_vis_curve) if q > 0]
            ax1.plot(q_vis_plot, h_vis_plot, "r--s", label=f"Viskos ν={nu:.1f} cSt", linewidth=2)
            Q_range = np.linspace(0.1, max(pump["Qw"]) * 1.2, 50)
            H_system = calculate_system_curve(st.session_state.pipeline, st.session_state.process, Q_range)
            ax1.plot(Q_range, H_system, "g-.", label="Anlagenkennlinie", linewidth=2)
            ax1.scatter([Q_vis_req], [H_vis_req], s=150, c="red", marker="*", zorder=5, label="Betriebspunkt")
            ax1.set_xlabel("Q [m³/h]"); ax1.set_ylabel("H [m]"); ax1.set_title("Q-H Kennlinie")
            ax1.grid(True, alpha=0.3); ax1.legend()
            
            # Q-η
            ax2 = axes[0, 1]
            ax2.plot(pump["Qw"], [e*100 for e in pump["eta"]], "b-o", label="Wasser", linewidth=2)
            eta_vis_plot = [e*100 for q, e in zip(Q_vis_curve, eta_vis_curve) if q > 0]
            ax2.plot(q_vis_plot, eta_vis_plot, "r--s", label="Viskos", linewidth=2)
            ax2.scatter([Q_vis_req], [eta_vis*100], s=100, c="red", marker="*", zorder=5)
            ax2.set_xlabel("Q [m³/h]"); ax2.set_ylabel("η [%]"); ax2.set_title("Q-η Kennlinie")
            ax2.grid(True, alpha=0.3); ax2.legend()
            
            # Q-P
            ax3 = axes[1, 0]
            ax3.plot(pump["Qw"], pump["Pw"], "b-o", label="Wasser", linewidth=2)
            p_vis_plot = [p for q, p in zip(Q_vis_curve, P_vis_curve) if q > 0]
            ax3.plot(q_vis_plot, p_vis_plot, "r--s", label="Viskos", linewidth=2)
            ax3.scatter([Q_vis_req], [P_vis_kW], s=100, c="red", marker="*", zorder=5)
            ax3.axhline(P_motor_kW, color="green", linestyle="--", label=f"Motor {P_motor_kW:.1f} kW")
            ax3.set_xlabel("Q [m³/h]"); ax3.set_ylabel("P [kW]"); ax3.set_title("Q-P Kennlinie")
            ax3.grid(True, alpha=0.3); ax3.legend()
            
            # NPSH
            ax4 = axes[1, 1]
            NPSHr_curve = pump.get("NPSHr", [2.0] * len(pump["Qw"]))
            ax4.plot(pump["Qw"], NPSHr_curve, "b-o", label="NPSHr", linewidth=2)
            NPSHa_curve = [calculate_npsh_available(st.session_state.pipeline, st.session_state.process, q) 
                          for q in pump["Qw"]]
            ax4.plot(pump["Qw"], NPSHa_curve, "g--", label="NPSHa", linewidth=2)
            ax4.scatter([Q_vis_req], [NPSHr], s=100, c="blue", marker="*", zorder=5)
            ax4.scatter([Q_vis_req], [NPSHa], s=100, c="green", marker="*", zorder=5)
            ax4.set_xlabel("Q [m³/h]"); ax4.set_ylabel("NPSH [m]"); ax4.set_title("NPSH-Kurven")
            ax4.grid(True, alpha=0.3); ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Export
            st.divider()
            st.subheader("📤 Export")
            col1, col2 = st.columns(2)
            with col1:
                html = generate_datasheet_html(st.session_state.project, st.session_state.process, pump, results)
                st.download_button("📄 Datenblatt (HTML)", data=html,
                    file_name=f"Datenblatt_{pump['id']}_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")
            with col2:
                json_data = generate_project_json(st.session_state.project, st.session_state.process,
                    st.session_state.pipeline, st.session_state.economic, results)
                st.download_button("💾 Projekt (JSON)", data=json_data,
                    file_name=f"Projekt_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")
    
    except Exception as e:
        show_error(e, "Einphasenpumpen")

def run_multi_phase_pump():
    """Vollständige Mehrphasen-Pumpenauslegung mit Gas-Löslichkeitsberechnung"""
    try:
        st.header("🌊 Mehrphasenpumpen-Auslegung")
        st.info("Auslegung für Gas-Flüssigkeits-Gemische mit Berücksichtigung der Gaslöslichkeit nach Henry")
        
        # === EINGABEN ===
        tab_input, tab_result, tab_curves = st.tabs(["📝 Eingaben", "📊 Ergebnisse", "📈 Kennlinien"])
        
        with tab_input:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Gasanforderung")
                input_mode = st.radio("Eingabemodus", ["Gasvolumenstrom (Norm)", "Konzentration"], index=0)
                
                if input_mode == "Gasvolumenstrom (Norm)":
                    Q_gas_target_lmin = st.number_input(
                        "Ziel-Gasvolumenstrom [L/min] (Norm)", 
                        min_value=1.0, max_value=500.0, value=80.0, step=5.0
                    )
                    C_target_info = None
                else:
                    C_target_cm3N_L = st.number_input(
                        "Ziel-Konzentration [cm³N/L]", 
                        min_value=10.0, max_value=500.0, value=100.0, step=10.0
                    )
                    Q_liq_for_calc = st.number_input(
                        "Flüssigkeitsstrom für Berechnung [m³/h]",
                        min_value=1.0, max_value=100.0, value=15.0, step=1.0
                    )
                    Q_gas_target_lmin = gas_flow_from_concentration(C_target_cm3N_L, Q_liq_for_calc)
                    C_target_info = C_target_cm3N_L
                    st.info(f"→ Entspricht {Q_gas_target_lmin:.1f} L/min (Norm) bei {Q_liq_for_calc} m³/h")
                
                p_suction = st.number_input(
                    "Saugdruck [bar(a)]", 
                    min_value=0.3, max_value=5.0, value=0.6, step=0.1,
                    help="Absolutdruck an der Pumpen-Saugseite"
                )
                
                dp_required = st.number_input(
                    "Erforderliche Druckerhöhung [bar]",
                    min_value=0.0, max_value=15.0, value=6.0, step=0.5,
                    help="0 = automatisch optimieren"
                )
            
            with col2:
                st.subheader("Medium")
                gas_medium = st.selectbox("Gas", list(HENRY_CONSTANTS.keys()), index=0)
                liquid_medium = st.selectbox("Flüssigkeit", list(MEDIA.keys()), index=0)
                temperature = st.number_input("Temperatur [°C]", min_value=-10.0, max_value=80.0, value=20.0, step=1.0)
                
                rho_liq = MEDIA[liquid_medium]["rho"]
                nu_liq = MEDIA[liquid_medium]["nu"]
                st.caption(f"ρ = {rho_liq} kg/m³, ν = {nu_liq} cSt")
                
                st.subheader("Sicherheit")
                safety_margin_pct = st.slider("Sicherheitszuschlag [%]", 0, 30, 10)
            
            with col3:
                st.subheader("Optimierung")
                allow_speed_adj = st.checkbox("Drehzahlanpassung erlauben", value=True)
                allow_partial = st.checkbox("Teilsättigung erlauben", value=False,
                    help="Erlaubt Betriebspunkte mit nicht vollständig gelöstem Gas")
                use_interpolation = st.checkbox("GVF-Interpolation", value=True,
                    help="Interpolation zwischen Kennlinien-GVF-Werten")
                
                st.subheader("Gewichtung")
                w_power = st.slider("Energie", 0.0, 1.0, 0.4, 0.05)
                w_eta = st.slider("Wirkungsgrad", 0.0, 1.0, 0.3, 0.05)
                w_gas = st.slider("Gasmenge-Genauigkeit", 0.0, 1.0, 0.3, 0.05)
        
        # === BERECHNUNG ===
        Q_gas_with_safety = Q_gas_target_lmin * (1 + safety_margin_pct / 100.0)
        results = []
        
        for pump in MPH_PUMPS:
            if nu_liq > pump.get("max_viscosity", 500):
                continue
            if rho_liq > pump.get("max_density", 1200):
                continue
            
            gvf_keys = sorted(pump["curves_dp_vs_Q"].keys())
            best_for_pump = None
            
            for gvf_pct in (gvf_keys if not use_interpolation else 
                           np.linspace(min(gvf_keys), max(gvf_keys), 20)):
                
                # Q-Bereich für diese Kennlinie
                lo_key, hi_key, _ = _interp_between_gvf_keys(pump, gvf_pct)
                Q_list = pump["curves_dp_vs_Q"][lo_key]["Q"]
                Q_min_curve = max(min([q for q in Q_list if q > 0]), 1.0)
                Q_max_curve = max(Q_list)
                
                for Q_liq in np.linspace(Q_min_curve, Q_max_curve, 30):
                    # Druckerhöhung bei diesem Q und GVF
                    dp_avail, _, _, _ = _dp_at_Q_gvf(pump, Q_liq, gvf_pct)
                    P_shaft, _, _, _ = _P_at_Q_gvf(pump, Q_liq, gvf_pct)
                    
                    if dp_required > 0 and abs(dp_avail - dp_required) > 2.0:
                        continue  # dp passt nicht
                    
                    # Druckseitiger Druck
                    p_discharge = p_suction + dp_avail
                    
                    # Gas-Löslichkeit bei Druckseite (Henry)
                    C_sat_discharge = gas_solubility_total_cm3N_L(gas_medium, p_discharge, temperature)
                    
                    # Konzentration entsprechend Kennlinie
                    # GVF% der Kennlinie = freies Gas / (freies Gas + Flüssigkeit)
                    # Bei MPH-Pumpen: Kennlinien-% oft = Gesamtgas-Konzentration
                    C_kennlinie = (gvf_pct / 100.0) * 1000.0  # Vereinfachung: % → cm³N/L
                    
                    # Gasvolumenstrom bei diesem Betriebspunkt
                    Q_liq_lmin = m3h_to_lmin(Q_liq)
                    Q_gas_pump_lmin = (C_kennlinie / 1000.0) * Q_liq_lmin
                    
                    # Prüfe ob Zielmenge erreicht wird
                    if Q_gas_pump_lmin < Q_gas_with_safety * 0.9:
                        continue  # Zu wenig Gas
                    
                    # Prüfe Löslichkeit
                    Q_gas_solvable_lmin = (C_sat_discharge / 1000.0) * Q_liq_lmin
                    
                    if not allow_partial and Q_gas_pump_lmin > Q_gas_solvable_lmin * 1.05:
                        continue  # Nicht genug löslich
                    
                    # Hydraulischer Wirkungsgrad (Schätzung)
                    P_hyd = (dp_avail * BAR_TO_PA) * (Q_liq / 3600.0) / 1000.0
                    eta_est = safe_clamp(P_hyd / max(P_shaft, 0.1), 0.1, 0.9)
                    
                    # Score berechnen
                    gas_error = abs(Q_gas_pump_lmin - Q_gas_with_safety) / Q_gas_with_safety
                    power_term = P_shaft / max(Q_liq, 1.0)  # spezifische Leistung
                    eta_term = 1.0 - eta_est
                    
                    # Bonus für exakte dp-Erfüllung
                    dp_error = abs(dp_avail - dp_required) / max(dp_required, 1.0) if dp_required > 0 else 0
                    
                    score = (w_gas * gas_error + 
                            w_power * power_term * 0.1 + 
                            w_eta * eta_term +
                            dp_error * 2.0)
                    
                    candidate = {
                        "pump": pump,
                        "pump_id": pump["id"],
                        "Q_liq_m3h": Q_liq,
                        "gvf_pct": gvf_pct,
                        "dp_bar": dp_avail,
                        "p_discharge": p_discharge,
                        "P_shaft_kW": P_shaft,
                        "eta_est": eta_est,
                        "Q_gas_pump_lmin": Q_gas_pump_lmin,
                        "Q_gas_solvable_lmin": Q_gas_solvable_lmin,
                        "C_sat_discharge": C_sat_discharge,
                        "C_kennlinie": C_kennlinie,
                        "gas_error_pct": gas_error * 100,
                        "score": score,
                        "n_rpm": pump["n0_rpm"],
                        "solvable": Q_gas_pump_lmin <= Q_gas_solvable_lmin * 1.05,
                    }
                    
                    if best_for_pump is None or score < best_for_pump["score"]:
                        best_for_pump = candidate
            
            if best_for_pump:
                results.append(best_for_pump)
        
        # Sortiere nach Score
        results.sort(key=lambda x: x["score"])
        
        # === ERGEBNISSE ===
        with tab_result:
            if not results:
                st.error("❌ Keine geeignete Pumpe gefunden für die angegebenen Bedingungen.")
                st.info("Versuchen Sie:\n- Geringeren Gasvolumenstrom\n- Höheren Saugdruck\n- 'Teilsättigung erlauben' aktivieren")
                return
            
            best = results[0]
            
            st.success(f"✅ Empfehlung: **{best['pump_id']}**")
            
            # Hauptmetriken
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Flüssigkeitsstrom", f"{best['Q_liq_m3h']:.1f} m³/h")
                st.metric("Gasvolumenstrom", f"{best['Q_gas_pump_lmin']:.1f} L/min (Norm)")
            with col2:
                st.metric("Druckerhöhung", f"{best['dp_bar']:.2f} bar")
                st.metric("Austrittsdruck", f"{best['p_discharge']:.2f} bar(a)")
            with col3:
                st.metric("Wellenleistung", f"{best['P_shaft_kW']:.2f} kW")
                st.metric("Wirkungsgrad", f"{best['eta_est']*100:.1f} %")
            with col4:
                st.metric("GVF (Kennlinie)", f"{best['gvf_pct']:.1f} %")
                st.metric("Drehzahl", f"{best['n_rpm']} rpm")
            
            # Löslichkeits-Analyse
            st.divider()
            st.subheader("🧪 Gas-Löslichkeitsanalyse")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                | Parameter | Wert |
                |-----------|------|
                | Ziel-Gasmenge | {Q_gas_target_lmin:.1f} L/min (Norm) |
                | Mit Sicherheit | {Q_gas_with_safety:.1f} L/min (Norm) |
                | Pumpe liefert | {best['Q_gas_pump_lmin']:.1f} L/min (Norm) |
                | Max. löslich (Druckseite) | {best['Q_gas_solvable_lmin']:.1f} L/min (Norm) |
                """)
            
            with col2:
                # Löslichkeits-Status
                ratio = best['Q_gas_pump_lmin'] / max(best['Q_gas_solvable_lmin'], 0.01)
                if ratio <= 1.0:
                    st.success(f"✅ Gas vollständig löslich ({ratio*100:.0f}% der Sättigung)")
                elif ratio <= 1.1:
                    st.warning(f"⚠️ Nahe Sättigung ({ratio*100:.0f}%)")
                else:
                    st.error(f"❌ Übersättigung! ({ratio*100:.0f}%) - Freies Gas am Austritt")
                
                st.markdown(f"""
                **Sättigungskonzentration bei {best['p_discharge']:.1f} bar(a):**  
                C_sat = {best['C_sat_discharge']:.1f} cm³N/L
                """)
            
            # Alternative Pumpen
            if len(results) > 1:
                st.divider()
                st.subheader("🔄 Alternative Pumpen")
                
                alt_data = []
                for r in results[1:4]:  # Top 3 Alternativen
                    alt_data.append({
                        "Pumpe": r["pump_id"],
                        "Q_liq [m³/h]": f"{r['Q_liq_m3h']:.1f}",
                        "Δp [bar]": f"{r['dp_bar']:.2f}",
                        "P [kW]": f"{r['P_shaft_kW']:.1f}",
                        "η [%]": f"{r['eta_est']*100:.0f}",
                        "Gas [L/min]": f"{r['Q_gas_pump_lmin']:.1f}",
                        "Löslich": "✅" if r["solvable"] else "⚠️",
                    })
                
                if alt_data:
                    st.dataframe(alt_data, use_container_width=True)
            
            # Motor-Auswahl
            st.divider()
            st.subheader("⚡ Motorauslegung")
            
            P_motor_min = best['P_shaft_kW'] * 1.15  # 15% Reserve
            P_motor_iec = motor_iec(P_motor_min)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wellenleistung", f"{best['P_shaft_kW']:.2f} kW")
            with col2:
                st.metric("Min. Motorleistung (+15%)", f"{P_motor_min:.2f} kW")
            with col3:
                st.metric("IEC Motorgröße", f"{P_motor_iec} kW")
        
        # === KENNLINIEN ===
        with tab_curves:
            if not results:
                st.warning("Keine Ergebnisse für Kennlinien-Darstellung")
                return
            
            best = results[0]
            pump = best["pump"]
            P_motor_iec = motor_iec(best['P_shaft_kW'] * 1.15)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Q-Δp Kennlinien für verschiedene GVF
            ax1 = axes[0, 0]
            gvf_keys = sorted(pump["curves_dp_vs_Q"].keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(gvf_keys)))
            
            for gvf, color in zip(gvf_keys, colors):
                curve = pump["curves_dp_vs_Q"][gvf]
                # Arrays alignen für sicheres Plotten
                q_vals, dp_vals = align_xy(curve["Q"], curve["dp"])
                if len(q_vals) >= 2:
                    ax1.plot(q_vals, dp_vals, "-o", color=color, 
                            label=f"GVF {gvf}%", linewidth=2, markersize=4)
            
            ax1.scatter([best["Q_liq_m3h"]], [best["dp_bar"]], 
                       s=200, c="red", marker="*", zorder=10, label="Betriebspunkt")
            ax1.axhline(best["dp_bar"], color="red", linestyle="--", alpha=0.5)
            ax1.axvline(best["Q_liq_m3h"], color="red", linestyle="--", alpha=0.5)
            ax1.set_xlabel("Q [m³/h]"); ax1.set_ylabel("Δp [bar]")
            ax1.set_title(f"{pump['id']} - Druckerhöhung vs. Volumenstrom")
            ax1.legend(loc="best", fontsize=8); ax1.grid(True, alpha=0.3)
            
            # 2. Q-P Kennlinien
            ax2 = axes[0, 1]
            for gvf, color in zip(gvf_keys, colors):
                if gvf in pump["power_kW_vs_Q"]:
                    curve = pump["power_kW_vs_Q"][gvf]
                    # Arrays alignen für sicheres Plotten
                    q_vals, p_vals = align_xy(curve["Q"], curve["P"])
                    if len(q_vals) >= 2:
                        ax2.plot(q_vals, p_vals, "-s", color=color,
                                label=f"GVF {gvf}%", linewidth=2, markersize=4)
            
            ax2.scatter([best["Q_liq_m3h"]], [best["P_shaft_kW"]], 
                       s=200, c="red", marker="*", zorder=10)
            ax2.axhline(P_motor_iec, color="green", linestyle="--", 
                       label=f"Motor {P_motor_iec} kW")
            ax2.set_xlabel("Q [m³/h]"); ax2.set_ylabel("P [kW]")
            ax2.set_title("Leistungsaufnahme vs. Volumenstrom")
            ax2.legend(loc="best", fontsize=8); ax2.grid(True, alpha=0.3)
            
            # 3. Gas-Löslichkeit vs. Druck
            ax3 = axes[1, 0]
            p_range = np.linspace(1, 12, 50)
            C_sat_curve = [gas_solubility_total_cm3N_L(gas_medium, p, temperature) for p in p_range]
            ax3.plot(p_range, C_sat_curve, "b-", linewidth=2, label=f"{gas_medium} Löslichkeit")
            ax3.axvline(best["p_discharge"], color="red", linestyle="--", 
                       label=f"Austrittsdruck {best['p_discharge']:.1f} bar")
            ax3.axhline(best["C_sat_discharge"], color="red", linestyle=":", alpha=0.7)
            ax3.scatter([best["p_discharge"]], [best["C_sat_discharge"]], 
                       s=150, c="red", marker="*", zorder=10)
            C_target = (Q_gas_with_safety / m3h_to_lmin(best["Q_liq_m3h"])) * 1000 if best["Q_liq_m3h"] > 0 else 0
            ax3.axhline(C_target, color="orange", linestyle="-.", 
                       label=f"Ziel {C_target:.0f} cm³N/L")
            ax3.set_xlabel("Druck [bar(a)]"); ax3.set_ylabel("C_sat [cm³N/L]")
            ax3.set_title(f"Gas-Löslichkeit ({gas_medium}, {temperature}°C)")
            ax3.legend(loc="best", fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_xlim(0, 14)
            
            # 4. Pumpenvergleich
            ax4 = axes[1, 1]
            if results:
                pump_names = [r["pump_id"] for r in results]
                scores = [r["score"] for r in results]
                colors_bar = ["green" if r["solvable"] else "orange" for r in results]
                ax4.barh(pump_names, scores, color=colors_bar, alpha=0.7)
                ax4.set_xlabel("Score (niedriger = besser)")
                ax4.set_title("Pumpenvergleich")
                ax4.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Export
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                # Ergebnis-Zusammenfassung als Text
                summary = f"""
# Mehrphasen-Pumpenauslegung
Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Anforderung
- Ziel-Gasmenge: {Q_gas_target_lmin:.1f} L/min (Norm)
- Mit Sicherheit: {Q_gas_with_safety:.1f} L/min (Norm)
- Gas: {gas_medium}
- Flüssigkeit: {liquid_medium}
- Temperatur: {temperature}°C
- Saugdruck: {p_suction} bar(a)

## Empfohlene Pumpe: {best['pump_id']}
- Flüssigkeitsstrom: {best['Q_liq_m3h']:.1f} m³/h
- Druckerhöhung: {best['dp_bar']:.2f} bar
- Austrittsdruck: {best['p_discharge']:.2f} bar(a)
- Wellenleistung: {best['P_shaft_kW']:.2f} kW
- Motorleistung (IEC): {P_motor_iec} kW
- Gasvolumenstrom: {best['Q_gas_pump_lmin']:.1f} L/min (Norm)
- Löslichkeit: {'OK' if best['solvable'] else 'WARNUNG - Übersättigung'}

## Gas-Löslichkeitsanalyse
- Sättigungskonzentration: {best['C_sat_discharge']:.1f} cm³N/L bei {best['p_discharge']:.1f} bar(a)
                """
                st.download_button("📄 Zusammenfassung (TXT)", data=summary,
                    file_name=f"MPH_Auslegung_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain")
            
            with col2:
                # JSON Export
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input": {
                        "Q_gas_target_lmin": Q_gas_target_lmin,
                        "gas_medium": gas_medium,
                        "liquid_medium": liquid_medium,
                        "temperature_c": temperature,
                        "p_suction_bar": p_suction,
                    },
                    "result": {
                        "pump_id": best["pump_id"],
                        "Q_liq_m3h": best["Q_liq_m3h"],
                        "dp_bar": best["dp_bar"],
                        "P_shaft_kW": best["P_shaft_kW"],
                        "P_motor_iec_kW": P_motor_iec,
                        "Q_gas_pump_lmin": best["Q_gas_pump_lmin"],
                        "solvable": best["solvable"],
                    }
                }
                st.download_button("💾 Export (JSON)", 
                    data=json.dumps(export_data, indent=2),
                    file_name=f"MPH_Auslegung_{datetime.now().strftime('%Y%m%d')}.json", 
                    mime="application/json")
    
    except Exception as e:
        show_error(e, "Mehrphasenpumpen")

def run_atex_selection():
    """Erweiterte ATEX-Motorauslegung nach EN 60079"""
    try:
        st.header("⚡ ATEX-Motorauslegung")
        st.info("Auslegung nach ATEX 2014/34/EU und EN 60079")
        
        tab1, tab2 = st.tabs(["📝 Eingaben & Auswahl", "📋 Dokumentation"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Leistungsanforderung")
                P_req = st.number_input("Wellenleistung [kW]", min_value=0.1, value=5.5, step=0.5)
                reserve_factor = st.number_input("Leistungsreserve [-]", min_value=1.0, max_value=1.5, value=1.15, step=0.05)
                efficiency_class = st.selectbox("Effizienzklasse", ["IE2", "IE3", "IE4"], index=1)
            
            with col2:
                st.subheader("Explosionsschutz")
                atmosphere = st.radio("Atmosphäre", ["Gas", "Staub"], index=0)
                
                if atmosphere == "Gas":
                    zone = st.selectbox("Zone", 
                        options=[0, 1, 2],
                        format_func=lambda x: f"Zone {x} - {'Ständig' if x==0 else 'Gelegentlich' if x==1 else 'Selten'}"
                    )
                    gas_group = st.selectbox("Gasgruppe", 
                        options=["IIA", "IIB", "IIC"],
                        index=2,
                        help="IIA: Propan, IIB: Ethylen, IIC: Wasserstoff/Acetylen")
                else:
                    zone = st.selectbox("Zone", 
                        options=[20, 21, 22],
                        format_func=lambda x: f"Zone {x} - {'Ständig' if x==20 else 'Gelegentlich' if x==21 else 'Selten'}")
                    gas_group = "IIIC"  # Staub
                
                temp_class = st.selectbox("Temperaturklasse",
                    options=["T1", "T2", "T3", "T4", "T5", "T6"],
                    index=3,
                    help="T1: 450°C, T2: 300°C, T3: 200°C, T4: 135°C, T5: 100°C, T6: 85°C")
            
            with col3:
                st.subheader("Umgebungsbedingungen")
                T_ambient = st.number_input("Umgebungstemperatur [°C]", min_value=-40.0, max_value=60.0, value=40.0)
                T_medium = st.number_input("Medientemperatur [°C]", min_value=-20.0, max_value=200.0, value=40.0)
                altitude = st.number_input("Aufstellhöhe [m ü. NN]", min_value=0, max_value=4000, value=0, step=100)
                t_margin = st.number_input("Temperatur-Sicherheit [K]", min_value=0.0, value=15.0)
            
            # Temperaturklassen-Grenzwerte
            TEMP_CLASS_LIMITS = {"T1": 450, "T2": 300, "T3": 200, "T4": 135, "T5": 100, "T6": 85}
            t_max_allowed = TEMP_CLASS_LIMITS[temp_class] - t_margin
            
            # Zündschutzarten basierend auf Zone
            if atmosphere == "Gas":
                if zone == 0:
                    protection_types = ["Ex ia (Eigensicherheit)", "Ex d (Druckfest) + Ex ia"]
                elif zone == 1:
                    protection_types = ["Ex d (Druckfeste Kapselung)", "Ex e (Erhöhte Sicherheit)", 
                                        "Ex p (Überdruckkapselung)", "Ex nA (Funkenfreie Betriebsmittel)"]
                else:  # Zone 2
                    protection_types = ["Ex ec (Erhöhte Sicherheit)", "Ex nA (Nicht-funkend)", 
                                        "Ex tc (Staubschutz durch Gehäuse)"]
            else:  # Staub
                protection_types = ["Ex tD (Staubdicht)", "Ex pD (Überdruckkapselung)", "Ex iaD (Eigensicher)"]
            
            selected_protection = st.selectbox("Zündschutzart", protection_types)
            
            st.divider()
            
            # Berechnung
            P_motor_min = P_req * reserve_factor
            
            # Höhenkorrektur (Leistungsreduzierung über 1000m)
            if altitude > 1000:
                altitude_factor = 1.0 - (altitude - 1000) * 0.01 / 100  # 1% pro 100m über 1000m
                P_motor_min = P_motor_min / altitude_factor
                st.warning(f"⚠️ Höhenkorrektur: Motorleistung um {(1/altitude_factor - 1)*100:.1f}% erhöht")
            
            P_motor_iec = motor_iec(P_motor_min)
            
            # Passende Motoren filtern
            suitable_motors = []
            for motor in ATEX_MOTORS:
                # Zonenprüfung
                if zone not in motor["zone_suitable"]:
                    continue
                # Temperaturprüfung
                if motor["t_max_surface"] > t_max_allowed:
                    continue
                suitable_motors.append(motor)
            
            # Ergebnisse
            st.subheader("📊 Ergebnisse")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wellenleistung", f"{P_req:.2f} kW")
            with col2:
                st.metric("Min. Motorleistung", f"{P_motor_min:.2f} kW")
            with col3:
                st.metric("IEC Motorgröße", f"{P_motor_iec} kW")
            with col4:
                st.metric("Max. Oberfl.-Temp.", f"{t_max_allowed:.0f} °C")
            
            if not suitable_motors:
                st.error("❌ Kein passender ATEX-Motor für diese Anforderungen verfügbar!")
                st.info("Mögliche Lösungen:\n- Niedrigere Temperaturklasse wählen\n- Andere Zündschutzart prüfen")
            else:
                st.success(f"✅ {len(suitable_motors)} passende Motor(en) gefunden")
                
                # Motor-Tabelle
                motor_data = []
                for m in suitable_motors:
                    base_price = P_motor_iec * 180  # Basis-Schätzpreis
                    motor_data.append({
                        "Typ": m["id"],
                        "Kennzeichnung": m["marking"],
                        "T-Klasse": m["temp_class"],
                        "Max. T [°C]": m["t_max_surface"],
                        "Effizienz": m["efficiency_class"],
                        "Preis ca.": f"{base_price * m['price_factor']:,.0f} €",
                    })
                
                st.dataframe(motor_data, use_container_width=True)
                
                # Empfehlung
                best_motor = suitable_motors[0]
                st.divider()
                st.subheader("🏆 Empfehlung")
                
                # ATEX-Kennzeichnung generieren
                category = "1G" if zone == 0 else "2G" if zone == 1 else "3G"
                epl = "Ga" if zone == 0 else "Gb" if zone == 1 else "Gc"
                
                marking = f"⚡ II {category} Ex {selected_protection.split()[1].lower()} {gas_group} {temp_class} {epl}"
                
                st.code(marking, language=None)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Ausgewählter Motor:** {best_motor['id']}
                    
                    | Parameter | Wert |
                    |-----------|------|
                    | Leistung | {P_motor_iec} kW |
                    | Effizienzklasse | {efficiency_class} |
                    | Zündschutzart | {selected_protection} |
                    | Gasgruppe | {gas_group} |
                    | Temperaturklasse | {temp_class} |
                    | Zone | {zone} |
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Betriebsbedingungen:**
                    
                    | Parameter | Wert |
                    |-----------|------|
                    | Umgebungstemperatur | {T_ambient} °C |
                    | Medientemperatur | {T_medium} °C |
                    | Aufstellhöhe | {altitude} m |
                    | Max. Oberflächentemp. | {t_max_allowed} °C |
                    """)
                
                # Export
                st.divider()
                atex_summary = f"""
ATEX-MOTORAUSLEGUNG
===================
Datum: {datetime.now().strftime('%Y-%m-%d %H:%M')}

ANFORDERUNG
-----------
Wellenleistung: {P_req} kW
Leistungsreserve: {reserve_factor}
Erforderliche Motorleistung: {P_motor_min:.2f} kW
IEC Motorgröße: {P_motor_iec} kW

EXPLOSIONSSCHUTZ
----------------
Atmosphäre: {atmosphere}
Zone: {zone}
Gasgruppe: {gas_group}
Temperaturklasse: {temp_class}
Zündschutzart: {selected_protection}

ATEX-KENNZEICHNUNG
------------------
{marking}

UMGEBUNG
--------
Umgebungstemperatur: {T_ambient} °C
Medientemperatur: {T_medium} °C
Aufstellhöhe: {altitude} m ü. NN

EMPFOHLENER MOTOR
-----------------
Typ: {best_motor['id']}
Kennzeichnung: {best_motor['marking']}
                """
                
                st.download_button("📄 ATEX-Datenblatt exportieren", 
                    data=atex_summary,
                    file_name=f"ATEX_Auslegung_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain")
        
        with tab2:
            st.markdown("""
            ## ATEX-Richtlinie 2014/34/EU
            
            ### Zoneneinteilung (Gas)
            | Zone | Häufigkeit | Beispiel |
            |------|------------|----------|
            | 0 | Ständig oder langzeitig | Inneres von Tanks |
            | 1 | Gelegentlich bei Normalbetrieb | Umgebung von Entlüftungen |
            | 2 | Selten und kurzzeitig | Allgemeine Anlagenbereiche |
            
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
            - **Ex d**: Druckfeste Kapselung
            - **Ex e**: Erhöhte Sicherheit  
            - **Ex p**: Überdruckkapselung
            - **Ex i**: Eigensicherheit
            - **Ex n**: Nicht-funkend (Zone 2)
            """)
    
    except Exception as e:
        show_error(e, "ATEX")

def run_comparison_tool():
    """Pumpenvergleich"""
    st.header("⚖️ Pumpenvergleich")
    selected_pumps = st.multiselect("Pumpen auswählen (max. 3)", [p["id"] for p in PUMPS], max_selections=3)
    
    if len(selected_pumps) >= 2:
        data = []
        for pid in selected_pumps:
            pump = next((p for p in PUMPS if p["id"] == pid), None)
            if pump:
                data.append({
                    "Pumpe": pump["id"],
                    "Q max [m³/h]": max(pump["Qw"]),
                    "H max [m]": max(pump["Hw"]),
                    "η max [%]": max(pump["eta"]) * 100,
                    "Preis [€]": pump.get("price_eur", "N/A"),
                })
        st.dataframe(data, use_container_width=True)
    else:
        st.warning("Bitte mindestens 2 Pumpen auswählen.")

def run_help_documentation():
    """Hilfe"""
    st.header("📚 Dokumentation")
    st.markdown("""
    ## Berechnungsgrundlagen
    
    ### Viskositätskorrektur (HI-Methode)
    $$B = 16.5 \\cdot \\frac{\\sqrt{\\nu}}{Q^{0.25} \\cdot H^{0.375}}$$
    
    ### NPSH
    $$NPSH_a = \\frac{p_{atm} - p_v}{\\rho g} + h_{geo} - h_{v,s}$$
    
    ### LCC
    Lebenszykluskosten nach Europump-Methodik.
    """)

def main():
    """Hauptfunktion"""
    try:
        init_session_state()
        render_project_header()
        
        st.title(f"🔧 {TOOL_NAME}")
        st.caption(f"Version {VERSION}")
        
        with st.sidebar:
            st.divider()
            page = st.radio("🧭 Navigation",
                ["Einphasenpumpen", "Mehrphasenpumpen", "ATEX-Auslegung", "Pumpenvergleich", "Dokumentation"])
            st.divider()
            st.caption(f"© {datetime.now().year} PumpDesign Pro")
        
        if page == "Einphasenpumpen":
            run_single_phase_pump()
        elif page == "Mehrphasenpumpen":
            run_multi_phase_pump()
        elif page == "ATEX-Auslegung":
            run_atex_selection()
        elif page == "Pumpenvergleich":
            run_comparison_tool()
        else:
            run_help_documentation()
        
    except Exception as e:
        show_error(e, "main")
        st.stop()

if __name__ == "__main__":
    main()

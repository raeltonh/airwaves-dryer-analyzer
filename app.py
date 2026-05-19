# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Belts & Heat Zones — Temperature vs Time", layout="wide")

# -------------------------
# THEME
# -------------------------
st.sidebar.header("App Theme")
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, horizontal=True, key="ui_theme")

st.title("Temperature vs Time — Belts = Heat Zones")
st.caption(
    "Belts only define Heat Zones (duration + setpoint). Upload TXT/XML/XLSX/CSV or enter manually, "
    "map series to Sides (Left/Center/Right), align by threshold, and export PNG/CSV/PDF. "
    "Machine presets are auto-applied; Atlas models use 1 belt, Apollo uses 4 belts (editable)."
)

# =========================
# CONSTANTS & HELPERS
# =========================
MACHINES = ["Atlas Max", "Atlas Max Plus", "Atlas Max Poly", "Avalanche HD6", "Apollo"]  # Atlas* => 1 belt
UNIT_OPTIONS = ["°C", "°F"]
ALLOWED_CHANNELS = {"CH1", "CH2", "CH3"}  # only three channels

DEFAULT_SETPOINT_F = 305.0
DEFAULT_BELT_DURATION_MIN = 2.0

DEFAULT_BELT_COLORS = {"BELT 1": "#1f77b4","BELT 2": "#ff7f0e","BELT 3": "#2ca02c","BELT 4": "#d62728"}
DEFAULT_CHANNEL_COLORS = {"CH1": "#1f77b4", "CH2": "#ff7f0e", "CH3": "#2ca02c"}  # Left/Center/Right
DEFAULT_SERIES_COLORS = {"S1": "#9467bd", "S2": "#8c564b", "S3": "#e377c2"}

MACHINE_PRESETS = {
    "Atlas Max":       {"belts": 1, "durations": [4.5],                   "setpoints_F": [320]},
    "Atlas Max Plus":  {"belts": 1, "durations": [4.5],                   "setpoints_F": [320]},
    "Atlas Max Poly":  {"belts": 1, "durations": [4.5],                   "setpoints_F": [320]},
    "Avalanche HD6":   {"belts": 2, "durations": [4.5, 4.5],              "setpoints_F": [320, 320], "manual_capture_min": [4.5, 4.5], "min_belts": 1, "max_belts": 2},
    "Apollo":          {"belts": 4, "durations": [4.5, 3.0, 3.0, 3.0],    "setpoints_F": [342, 305, 305, 305], "manual_capture_min": [4.5, 3.0, 3.0, 3.0], "min_belts": 1, "max_belts": 4},
}

FLEXIBLE_MACHINE_BELT_RANGE = {
    name: (
        int(cfg.get("min_belts", cfg.get("belts", 1))),
        int(cfg.get("max_belts", cfg.get("belts", 1)))
    )
    for name, cfg in MACHINE_PRESETS.items()
    if int(cfg.get("max_belts", cfg.get("belts", 1))) > int(cfg.get("min_belts", cfg.get("belts", 1)))
}

def f_to_c(v: float) -> float: return (v - 32.0) * (5.0/9.0)
def c_to_f(v: float) -> float: return v * 9.0/5.0 + 32.0
def series_c_to_display(s: pd.Series, unit_display: str) -> pd.Series:
    return s if unit_display == "°C" else s.apply(lambda x: c_to_f(float(x)) if pd.notna(x) else x)

def lighten(hex_color: str, factor: float = 0.65) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor); g = int(g + (255 - g) * factor); b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def fmt_mmss(x, pos=None):
    if x is None or np.isnan(x): return ""
    total_s = int(round(float(x) * 60)); m, s = divmod(total_s, 60)
    return f"{m:d}:{s:02d}"

def get_ch_side() -> dict:
    if "ch_side" not in st.session_state:
        st.session_state["ch_side"] = {"CH1": "Left", "CH2": "Center", "CH3": "Right"}
    return st.session_state["ch_side"]

def filter_allowed_channels(df: pd.DataFrame) -> pd.DataFrame:
    if "channel" in df.columns:
        return df[df["channel"].isin(ALLOWED_CHANNELS)].copy()
    return df

def to_display_unit(val_c: float, unit_display: str) -> float:
    return val_c if unit_display == "°C" else c_to_f(val_c)

@st.cache_resource(show_spinner=False)
def load_plotting_libs(selected_theme: str):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

    plt.style.use("dark_background" if selected_theme == "Dark" else "default")
    return plt, PdfPages, Line2D, Patch, FuncFormatter, MaxNLocator, MultipleLocator

# =========================
# PARSERS (TXT / XML / XLSX / CSV)
# =========================
def parse_sd_txt(text: str, filename: str) -> pd.DataFrame:
    import re

    rows = []
    raw_lines = [line.rstrip("\r") for line in text.splitlines() if line.strip()]

    header_line = next(
        (
            line for line in raw_lines
            if "date" in line.lower() and "time" in line.lower() and "unit" in line.lower()
        ),
        "",
    )

    column_slices = {}
    anchor_header_start = None
    if header_line:
        header_tokens = list(re.finditer(r"\S+", header_line))
        for idx, match in enumerate(header_tokens):
            label = match.group(0).strip().lower()
            start = match.start()
            end = header_tokens[idx + 1].start() if idx + 1 < len(header_tokens) else None
            column_slices[label] = (start, end)
        if "date" in column_slices:
            anchor_header_start = column_slices["date"][0]

    def read_field(line: str, offset: int, *names: str) -> str:
        for name in names:
            bounds = column_slices.get(name.lower())
            if bounds is None:
                continue
            start, end = bounds
            start = max(0, start + offset)
            end = None if end is None else max(0, end + offset)
            return line[start:end].strip()
        return ""

    for raw in raw_lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if not (stripped.startswith("AT") or stripped.startswith("MN")):
            continue
        if "date" in stripped.lower() and "time" in stripped.lower():
            continue

        offset = 0
        if column_slices and anchor_header_start is not None:
            m_date = re.search(r"\d{4}-\d{2}-\d{2}", line)
            if m_date:
                offset = m_date.start() - anchor_header_start

        date = read_field(line, offset, "date")
        time = read_field(line, offset, "time")
        interval = read_field(line, offset, "int")
        unit = read_field(line, offset, "unit")

        channel_values = []
        if column_slices:
            for i in range(1, 5):
                channel_values.append((i, read_field(line, offset, f"{i}ch", f"ch{i}")))
        else:
            toks = stripped.split()
            if len(toks) < 6:
                continue
            date, time, interval = toks[1], toks[2], toks[3]
            unit = toks[-1]
            channel_values = list(enumerate(toks[4:-1], start=1))

        ts = pd.to_datetime(f"{date} {time}", errors="coerce")
        if pd.isna(ts):
            continue

        unit = (unit or "C").strip().upper()
        for i, val in channel_values:
            if i > 3:  # ignore CH4+
                continue
            if val in ("", None):
                continue
            try:
                v = float(str(val).replace(",", "."))
            except Exception:
                continue
            rows.append({
                "source": filename, "timestamp": ts, "interval": interval,
                "channel": f"CH{i}", "temp_value": v, "unit": unit
            })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["timestamp"])
    df["temp_C"] = df.apply(lambda r: f_to_c(r["temp_value"]) if str(r["unit"]).upper()=="F" else r["temp_value"], axis=1)
    df = df.sort_values(["source","timestamp","channel"])
    df["t0"] = df.groupby("source")["timestamp"].transform("min")
    df["elapsed_min"] = (df["timestamp"] - df["t0"]).dt.total_seconds()/60.0
    return df[["source","timestamp","elapsed_min","channel","temp_C","unit"]]

def parse_xml_placeholder(xml_text: str, filename: str) -> pd.DataFrame:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception:
        return pd.DataFrame()
    rows=[]
    for elem in root.iter():
        t = elem.attrib.get("time") or elem.attrib.get("timestamp")
        unit = elem.attrib.get("unit","C")
        if not t: continue
        try: ts = pd.to_datetime(t)
        except Exception: continue
        for i in range(1,5):
            if i>3: continue
            v = elem.attrib.get(f"ch{i}") or elem.attrib.get(f"CH{i}")
            if v is None: continue
            try: fv = float(v)
            except Exception: continue
            rows.append({
                "source": filename, "timestamp": ts, "elapsed_min": None,
                "channel": f"CH{i}",
                "temp_C": fv if str(unit).upper()=="C" else f_to_c(fv),
                "unit": unit
            })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["source","timestamp","channel"])
    df["t0"] = df.groupby("source")["timestamp"].transform("min")
    df["elapsed_min"] = (df["timestamp"] - df["t0"]).dt.total_seconds()/60.0
    return df[["source","timestamp","elapsed_min","channel","temp_C","unit"]]

def parse_profile_xlsx(path_or_bytes, filename: str = "profile.xlsx") -> pd.DataFrame:
    import datetime as dt, re
    try:
        df = pd.read_excel(path_or_bytes, sheet_name=0, header=None)
    except Exception:
        return pd.DataFrame()

    date, date_regex = None, re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
    for row in df.itertuples(index=False):
        for val in row:
            if isinstance(val, str):
                m = date_regex.search(val)
                if m:
                    d = pd.to_datetime(m.group(1), errors="coerce")
                    if pd.notna(d): date = pd.Timestamp(d.date()); break
        if date is not None: break
    if date is None: date = pd.Timestamp("1970-01-01")

    time_col, best_ct = None, -1
    for col in df.columns:
        ct = sum(isinstance(v, dt.time) for v in df[col].dropna())
        if ct > best_ct and ct >= 3:
            best_ct, time_col = ct, col
    if time_col is None: return pd.DataFrame()

    header_row = None
    for i, v in enumerate(df[time_col].tolist()):
        if isinstance(v, str) and ("min" in v.lower() or "time" in v.lower()):
            header_row = i; break
    if header_row is None:
        for i, v in enumerate(df[time_col].tolist()):
            if isinstance(v, dt.time):
                header_row = max(i-1, 0); break

    side_cols, side_names = [], []
    for offset in range(1,5):
        if len(side_cols) >= 3: break
        col = time_col + offset
        if col not in df.columns: continue
        header_val = df.iat[header_row, col] if header_row is not None else None
        name = header_val.strip().lower() if isinstance(header_val, str) and header_val.strip() else f"ch{offset}"
        below = pd.to_numeric(df.iloc[(header_row or 0)+1:, col], errors="coerce")
        if below.notna().sum() >= 3:
            side_cols.append(col); side_names.append(name)

    if not side_cols: return pd.DataFrame()

    rows=[]
    for i in range((header_row or 0)+1, len(df)):
        import datetime as dt
        tcell = df.iat[i, time_col]
        if not isinstance(tcell, dt.time): continue
        secs = tcell.hour*3600 + tcell.minute*60 + tcell.second
        elapsed_min = secs/60.0
        ts = date + pd.Timedelta(seconds=secs)
        for c,name in zip(side_cols, side_names):
            val = pd.to_numeric(df.iat[i, c], errors="coerce")
            if pd.isna(val): continue
            unit="F"; temp_C = (val - 32.0) * (5.0/9.0)
            nm = name.strip().lower()
            if nm.startswith("left"):   ch, side = "CH1","Left"
            elif nm.startswith("center"): ch, side = "CH2","Center"
            elif nm.startswith("right"):  ch, side = "CH3","Right"
            else:
                idx = side_cols.index(c) + 1
                if idx>3: continue
                ch, side = {1:("CH1","Left"),2:("CH2","Center"),3:("CH3","Right")}[idx]
            rows.append({"source": filename,"timestamp": ts,"elapsed_min": elapsed_min,
                         "channel": ch,"side": side,"temp_C": float(temp_C),"unit": unit})
    out = pd.DataFrame(rows)
    if out.empty: return out
    out["t0"] = out.groupby("source")["elapsed_min"].transform("min")
    out["elapsed_min"] = out["elapsed_min"] - out["t0"]
    return out[["source","timestamp","elapsed_min","channel","side","temp_C","unit"]]

def parse_csv_generic(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Flexible CSV:
      - supports: time/timestamp/elapsed_min
      - CH1..CH3 or left/center/right
      - unit (C/F) optional
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()

    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    elapsed_min = None
    if "elapsed_min" in df.columns:
        elapsed_min = pd.to_numeric(df["elapsed_min"], errors="coerce")
    elif "time" in df.columns:
        def to_min(v: str):
            v = str(v)
            try:
                if ":" in v:
                    parts = v.split(":")
                    if len(parts) == 2:
                        m, s = int(parts[0]), int(parts[1]); return m + s/60
                    if len(parts) == 3:
                        h, m, s = int(parts[0]), int(parts[1]), int(parts[2]); return h*60 + m + s/60
                return float(v)
            except:
                return np.nan
        elapsed_min = df["time"].astype(str).map(to_min)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            t0 = ts.min()
            elapsed_min = (ts - t0).dt.total_seconds()/60.0

    if elapsed_min is None: return pd.DataFrame()

    channel_cols = []
    for key in ["ch1","ch2","ch3","left","center","right"]:
        if key in df.columns:
            channel_cols.append(key)
    if not channel_cols:
        for c in df.columns:
            if any(k in c for k in ["ch1","ch2","ch3","left","center","right"]):
                channel_cols.append(c)
    channel_cols = channel_cols[:3]
    if not channel_cols: return pd.DataFrame()

    unit = "C"
    if "unit" in df.columns:
        u = str(df["unit"].dropna().iloc[0]).upper()
        unit = "F" if "F" in u else "C"

    rows=[]
    for idx_row in range(len(df)):
        em = df.get("elapsed_min", pd.Series([np.nan]*len(df))).iloc[idx_row]
        if pd.isna(em): em = elapsed_min.iloc[idx_row] if idx_row < len(elapsed_min) else np.nan
        if pd.isna(em): continue
        for idx, col in enumerate(channel_cols, start=1):
            try:
                val = pd.to_numeric(df.iloc[idx_row][col], errors="coerce")
            except Exception:
                val = np.nan
            if pd.isna(val): continue
            name = str(col).lower()
            if "left" in name: ch, side = "CH1", "Left"
            elif "center" in name: ch, side = "CH2", "Center"
            elif "right" in name: ch, side = "CH3", "Right"
            else:
                ch = f"CH{idx}"
                side = {"CH1": "Left", "CH2": "Center", "CH3": "Right"}[ch]
            temp_C = val if unit == "C" else f_to_c(val)
            rows.append({"source": filename, "timestamp": pd.NaT, "elapsed_min": float(em),
                         "channel": ch, "side": side, "temp_C": float(temp_C), "unit": unit})
    return pd.DataFrame(rows)

# -------- Alignment
def align_by_threshold(df_c: pd.DataFrame, threshold_c: float) -> pd.DataFrame:
    def shift_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("elapsed_min").copy()
        crossed = g[g["temp_C"] >= threshold_c]
        g["elapsed_min_aligned"] = g["elapsed_min"] - (crossed.iloc[0]["elapsed_min"] if not crossed.empty else 0.0)
        return g
    return df_c.groupby(["source","channel"], group_keys=False).apply(shift_group)

# -------- Metrics per window + extra KPIs
def metrics_in_windows(df_c: pd.DataFrame,
                       windows: Dict[str, Tuple[float,float]],
                       setpoint_c: Dict[str,float]) -> pd.DataFrame:
    rows=[]
    for (src,ch), g_all in df_c.groupby(["source","channel"]):
        for belt,(start,end) in windows.items():
            g = g_all[(g_all["elapsed_min"]>=start)&(g_all["elapsed_min"]<=end)].sort_values("elapsed_min")
            if g.empty:
                rows.append({"Source":src,"Channel":ch,"Belt":belt,"Start (min)":start,"End (min)":end,
                             "Dur (min)":end-start,"Time to SP (min)":np.nan,
                             "Mean (°C)":np.nan,"Std (°C)":np.nan,"Min (°C)":np.nan,
                             "Max (°C)":np.nan,"Range (°C)":np.nan,"% ≥ SP":np.nan,
                             "Peak-to-Peak (°C)":np.nan,"RMS (°C)":np.nan,"Ramp Rate to SP (°C/min)":np.nan})
                continue
            sp=setpoint_c[belt]
            meet=g[g["temp_C"]>=sp]
            t_to_sp=(meet.iloc[0]["elapsed_min"]-start) if not meet.empty else np.nan
            p2p = g["temp_C"].max() - g["temp_C"].min()
            rms = float(np.sqrt(np.mean(np.square(g["temp_C"] - g["temp_C"].mean())))) if len(g) >= 2 else 0.0
            ramp_rate = np.nan
            if not np.isnan(t_to_sp) and t_to_sp > 0:
                t0_temp = g.iloc[0]["temp_C"]
                ramp_rate = (sp - t0_temp) / t_to_sp if t_to_sp > 0 else np.nan
            rows.append({"Source":src,"Channel":ch,"Belt":belt,"Start (min)":start,"End (min)":end,
                         "Dur (min)":end-start,"Time to SP (min)":t_to_sp,
                         "Mean (°C)":g["temp_C"].mean(),"Std (°C)":g["temp_C"].std(),
                         "Min (°C)":g["temp_C"].min(),"Max (°C)":g["temp_C"].max(),
                         "Range (°C)":g["temp_C"].max()-g["temp_C"].min(),"% ≥ SP":(g["temp_C"]>=sp).mean()*100.0,
                         "Peak-to-Peak (°C)":p2p, "RMS (°C)":rms, "Ramp Rate to SP (°C/min)":ramp_rate})
    return pd.DataFrame(rows)

# =========================
# SIDEBAR — CONTROLS (Machine + Data Source + Presets + Session)
# =========================
st.sidebar.header("Controls")

def _reset_all():
    for k in list(st.session_state.keys()): del st.session_state[k]
    st.rerun()

if st.sidebar.button("🔄 Reset & Clear", use_container_width=True, key="btn_reset_all"):
    _reset_all()

# Machine & Data Source, lado a lado
row_md = st.sidebar.columns(2)
with row_md[0]:
    st.markdown("**Machine**")
    machine = st.radio("Machine", MACHINES, index=MACHINES.index("Apollo"),
                       label_visibility="collapsed", key="machine_select")
with row_md[1]:
    st.markdown("**Data Source**")
    data_mode = st.radio("Data Source", ["TXT (SD card)", "XML", "CSV", "Manual"],
                         index=0, label_visibility="collapsed", key="datasource_select")

# Display unit
st.sidebar.markdown("**Display Unit**")
unit_display = st.sidebar.radio("", UNIT_OPTIONS,
                                index=1 if machine == "Apollo" else 0,
                                horizontal=True, label_visibility="collapsed", key="display_unit")

# --- Auto-apply machine preset on change ---
def apply_machine_preset_for(machine_name: str):
    preset = MACHINE_PRESETS.get(machine_name)
    if not preset: return
    n = int(preset["belts"])
    st.session_state["n_belts"] = n
    # belts cfg
    st.session_state["belts_cfg"] = {}
    manual_caps = preset.get("manual_capture_min") or []
    for i in range(n):
        b = f"BELT {i+1}"
        dur = float(preset["durations"][i]) if i < len(preset["durations"]) else DEFAULT_BELT_DURATION_MIN
        spF = float(preset["setpoints_F"][i]) if i < len(preset["setpoints_F"]) else DEFAULT_SETPOINT_F
        sp_disp = spF if unit_display=="°F" else f_to_c(spF)
        manual_capture = float(manual_caps[i]) if i < len(manual_caps) else dur
        st.session_state["belts_cfg"][b] = {
            "duration_min": dur,
            "setpoint_display": sp_disp,
            "color": DEFAULT_BELT_COLORS.get(b, "#888"),
            "manual_capture_min": manual_capture,
        }
    # channels default sides
    st.session_state.setdefault("ch_side", {"CH1":"Left","CH2":"Center","CH3":"Right"})

if "last_machine" not in st.session_state:
    st.session_state["last_machine"] = machine
    apply_machine_preset_for(machine)

if machine != st.session_state.get("last_machine"):
    st.session_state["last_machine"] = machine
    apply_machine_preset_for(machine)
    st.rerun()

st.sidebar.markdown("---")
# Reset to machine defaults (reaplica o preset da máquina atual)
if st.sidebar.button("Reset to machine defaults", use_container_width=True, key="btn_reset_to_defaults"):
    apply_machine_preset_for(machine)
    st.rerun()

# ---- Session save/load
st.sidebar.markdown("**Session Save / Load**")
def serialize_session() -> Dict[str, Any]:
    return {
        "machine": machine,
        "unit_display": unit_display,
        "ch_side": get_ch_side(),
        "channel_colors": {
            "CH1": st.session_state.get("chcol_CH1", DEFAULT_CHANNEL_COLORS["CH1"]),
            "CH2": st.session_state.get("chcol_CH2", DEFAULT_CHANNEL_COLORS["CH2"]),
            "CH3": st.session_state.get("chcol_CH3", DEFAULT_CHANNEL_COLORS["CH3"]),
        },
        "belts_cfg": st.session_state.get("belts_cfg", {}),
        "n_belts": st.session_state.get("n_belts", 1),
        "align_on": st.session_state.get("align_on", False),
        "threshold_display": st.session_state.get("threshold_display", 250.0 if unit_display=="°F" else 120.0),
        "show_minmax": st.session_state.get("show_minmax", True),
        "min_line": st.session_state.get("min_line", 300.0 if unit_display=="°F" else 150.0),
        "max_line": st.session_state.get("max_line", 320.0 if unit_display=="°F" else 160.0),
    }

def apply_loaded_session(cfg: Dict[str, Any]):
    for k, v in cfg.items():
        st.session_state[k] = v
    st.rerun()

c_s1, c_s2 = st.sidebar.columns(2)
with c_s1:
    sess_json = json.dumps(serialize_session(), indent=2).encode("utf-8")
    st.download_button("Save Session (JSON)", data=sess_json, file_name="session_config.json",
                       mime="application/json", use_container_width=True, key="btn_save_session")
with c_s2:
    up = st.file_uploader("Load Session", type=["json"], accept_multiple_files=False,
                          label_visibility="collapsed", key="upload_session")
    if up is not None:
        try:
            cfg = json.loads(up.read().decode("utf-8"))
            apply_loaded_session(cfg)
        except Exception as e:
            st.sidebar.error(f"Invalid session file: {e}")

# ---- Channel colors (lado a lado)
st.sidebar.markdown("---")
st.sidebar.markdown("**Channel Colors**")
cc1, cc2, cc3 = st.sidebar.columns(3)
with cc1:
    ch1_col = st.color_picker("CH1", DEFAULT_CHANNEL_COLORS["CH1"], key="chcol_CH1", label_visibility="collapsed"); st.caption("**CH1**")
with cc2:
    ch2_col = st.color_picker("CH2", DEFAULT_CHANNEL_COLORS["CH2"], key="chcol_CH2", label_visibility="collapsed"); st.caption("**CH2**")
with cc3:
    ch3_col = st.color_picker("CH3", DEFAULT_CHANNEL_COLORS["CH3"], key="chcol_CH3", label_visibility="collapsed"); st.caption("**CH3**")
channel_colors = {"CH1": ch1_col, "CH2": ch2_col, "CH3": ch3_col}

# Manual series colors (opcional)
series_color_mode = False
prefer_series_legend = False
series_colors = DEFAULT_SERIES_COLORS.copy()
st.sidebar.markdown("---")
if data_mode == "Manual":
    st.sidebar.markdown("**Series Colors (optional)**")
    series_color_mode = st.sidebar.toggle("Use series colors", value=False, key="toggle_series_colors")
    if series_color_mode:
        s1, s2, s3 = st.sidebar.columns(3)
        with s1:
            series_colors["S1"] = st.color_picker("S1", DEFAULT_SERIES_COLORS["S1"], key="scol_S1",
                                                  label_visibility="collapsed"); st.caption("**S1**")
        with s2:
            series_colors["S2"] = st.color_picker("S2", DEFAULT_SERIES_COLORS["S2"], key="scol_S2",
                                                  label_visibility="collapsed"); st.caption("**S2**")
        with s3:
            series_colors["S3"] = st.color_picker("S3", DEFAULT_SERIES_COLORS["S3"], key="scol_S3",
                                                  label_visibility="collapsed"); st.caption("**S3**")
prefer_series_legend = st.sidebar.toggle("Prefer series legend (if available)", value=False, key="toggle_series_legend")

st.sidebar.markdown("---")
st.sidebar.markdown("**Show Min / Max lines**")
show_minmax = st.sidebar.toggle("", value=st.session_state.get("show_minmax", True),
                                label_visibility="collapsed", key="toggle_minmax")
st.session_state["show_minmax"] = show_minmax

default_min = 300.0 if unit_display=="°F" else 150.0
default_max = 320.0 if unit_display=="°F" else 160.0
cmm1, cmm2 = st.sidebar.columns(2)
with cmm1:
    st.sidebar.markdown("**Min value**")
    min_line = st.sidebar.number_input("", value=float(st.session_state.get("min_line", default_min)),
                                       label_visibility="collapsed", key="num_min_line")
with cmm2:
    st.sidebar.markdown("**Max value**")
    max_line = st.sidebar.number_input("", value=float(st.session_state.get("max_line", default_max)),
                                       label_visibility="collapsed", key="num_max_line")
st.session_state["min_line"] = min_line; st.session_state["max_line"] = max_line

st.sidebar.markdown("---")
st.sidebar.markdown("**Align curves by threshold**")
align_on = st.sidebar.toggle("", value=st.session_state.get("align_on", False),
                             label_visibility="collapsed", key="toggle_align")
st.session_state["align_on"] = align_on
st.sidebar.markdown("**Alignment threshold**")
threshold_display = st.sidebar.number_input("", value=float(st.session_state.get("threshold_display", 250.0 if unit_display=="°F" else 120.0)),
                                            label_visibility="collapsed", key="num_align_threshold")
st.session_state["threshold_display"] = threshold_display

st.sidebar.markdown("---")
# --- Time axis (with tooltips)
st.sidebar.markdown("**Time axis**")
tick_mode = st.sidebar.selectbox(
    "X ticks",
    ["Auto", "Every 5 s", "Every 10 s", "Every 15 s", "Every 30 s", "Every 1 min", "Every 2 min"],
    index=0,
    key="sel_tick_mode",
    help=(
        "• Auto: posiciona até N rótulos de tempo de forma inteligente (use o controle abaixo para N).\n"
        "• Passo fixo: força espaçamento uniforme (ex.: a cada 15s, 30s, 1min), ótimo para gravações longas."
    ),
)
max_labels = st.sidebar.slider(
    "Max labels (Auto)",
    6, 24, 12,
    key="sld_max_labels",
    help=(
        "Válido apenas quando 'X ticks' está em Auto.\n"
        "Define o N máximo de rótulos no eixo X. Reduza para deixar o eixo menos poluído; aumente para mais granularidade."
    ),
)

def apply_time_axis(ax, x_max: float, mode: str, max_n: int):
    if mode == "Auto":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_n, prune='both'))
    else:
        step_map = {"Every 5 s": 5, "Every 10 s": 10, "Every 15 s": 15, "Every 30 s": 30, "Every 1 min": 60, "Every 2 min": 120}
        step_s = step_map.get(mode, 30)
        ax.xaxis.set_major_locator(MultipleLocator(step_s/60.0))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_mmss))
    ax.set_xlim(0, x_max if x_max>0 else None)

# =========================
# =========================
# BELTS = HEAT ZONES
# =========================
st.subheader("Belts (each Belt defines one Heat Zone)")

# Belts count: hidden & fixed to 1 for Atlas models; editable only for Apollo
preset_cfg = MACHINE_PRESETS.get(machine, {"belts": 1})
preset_belts = int(preset_cfg.get("belts", 1))
if machine in FLEXIBLE_MACHINE_BELT_RANGE:
    min_belts, max_belts = FLEXIBLE_MACHINE_BELT_RANGE[machine]
    label = f"Number of BELTs ({machine})"
    n_belts = st.number_input(label, min_value=min_belts, max_value=max_belts,
                              value=int(st.session_state.get("n_belts", preset_belts)),
                              step=1, key=f"num_belts_{machine.replace(' ', '_')}")
    st.session_state["n_belts"] = int(n_belts)
    belt_labels = [f"BELT {i}" for i in range(1, int(n_belts)+1)]

    st.markdown("**Colors, Duration & Setpoint per Belt**")
    if "belts_cfg" not in st.session_state or not st.session_state["belts_cfg"]:
        st.session_state["belts_cfg"] = {}
        for i, b in enumerate(belt_labels):
            sp_default = DEFAULT_SETPOINT_F if unit_display=="°F" else f_to_c(DEFAULT_SETPOINT_F)
            st.session_state["belts_cfg"][b] = {"duration_min": DEFAULT_BELT_DURATION_MIN,
                                                "setpoint_display": float(sp_default),
                                                "color": DEFAULT_BELT_COLORS.get(b, "#888"),
                                                "manual_capture_min": DEFAULT_BELT_DURATION_MIN}
    belts_cfg: Dict[str, Dict[str, float]] = st.session_state["belts_cfg"]
    belt_bg: Dict[str,str] = {}

    # sincroniza cfg com quantidade atual
    current_set = set(belt_labels)
    for b in list(belts_cfg.keys()):
        if b not in current_set:
            del belts_cfg[b]
    for b in belt_labels:
        if b not in belts_cfg:
            sp_default = DEFAULT_SETPOINT_F if unit_display=="°F" else f_to_c(DEFAULT_SETPOINT_F)
            belts_cfg[b] = {"duration_min": DEFAULT_BELT_DURATION_MIN,
                            "setpoint_display": float(sp_default),
                            "color": DEFAULT_BELT_COLORS.get(b, "#888"),
                            "manual_capture_min": DEFAULT_BELT_DURATION_MIN}
        else:
            belts_cfg[b].setdefault("manual_capture_min", float(belts_cfg[b].get("duration_min", DEFAULT_BELT_DURATION_MIN)))

    # grid 2 colunas para Apollo
    for row_start in range(0, len(belt_labels), 2):
        row_belts = belt_labels[row_start:row_start+2]
        cols = st.columns(2)
        for col, belt in zip(cols, row_belts):
            with col:
                c1, c2 = st.columns([1,1])
                with c1:
                    belt_col = st.color_picker(f"{belt} • Color", belts_cfg[belt]["color"], key=f"col_{belt}")
                with c2:
                    sp_disp = st.number_input(f"{belt} • Setpoint ({unit_display})",
                                              value=float(belts_cfg[belt]["setpoint_display"]),
                                              key=f"sp_{belt}")
                dur = st.number_input(f"{belt} • Duration (min)", min_value=0.25, max_value=240.0,
                                      value=float(belts_cfg[belt]["duration_min"]), step=0.25, key=f"dur_{belt}")
                manual_cap = st.number_input(f"{belt} • Manual capture (min)", min_value=0.25, max_value=480.0,
                                             value=float(belts_cfg[belt]["manual_capture_min"]), step=0.25, key=f"man_{belt}")
                belts_cfg[belt] = {
                    "duration_min": float(dur),
                    "setpoint_display": float(sp_disp),
                    "color": belt_col,
                    "manual_capture_min": float(manual_cap),
                }
                belt_bg[belt] = lighten(belt_col, 0.75)

else:
    # Atlas models: exatamente 1 belt, UI compacta
    st.session_state["n_belts"] = 1
    belt_labels = ["BELT 1"]

    # inicia cfg se necessário
    if "belts_cfg" not in st.session_state or "BELT 1" not in st.session_state["belts_cfg"]:
        sp_default = DEFAULT_SETPOINT_F if unit_display=="°F" else f_to_c(DEFAULT_SETPOINT_F)
        st.session_state["belts_cfg"] = {
            "BELT 1": {
                "duration_min": 4.5,
                "setpoint_display": float(sp_default),
                "color": DEFAULT_BELT_COLORS["BELT 1"],
                "manual_capture_min": 4.5,
            }
        }

    bcfg = st.session_state["belts_cfg"]["BELT 1"]
    bcfg.setdefault("manual_capture_min", float(bcfg.get("duration_min", 4.5)))
    with st.container(border=True):
        st.markdown("**Single Belt (Atlas)**")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            belt_col = st.color_picker("BELT 1 • Color", bcfg["color"], key="col_BELT1_atlas")
        with c2:
            sp_disp = st.number_input(f"BELT 1 • Setpoint ({unit_display})",
                                      value=float(bcfg["setpoint_display"]), key="sp_BELT1_atlas")
        with c3:
            dur = st.number_input("BELT 1 • Duration (min)", min_value=0.25, max_value=240.0,
                                  value=float(bcfg["duration_min"]), step=0.25, key="dur_BELT1_atlas")
        manual_cap = st.number_input("BELT 1 • Manual capture (min)", min_value=0.25, max_value=480.0,
                                     value=float(bcfg["manual_capture_min"]), step=0.25, key="man_BELT1_atlas")
        # salva de volta
        st.session_state["belts_cfg"]["BELT 1"] = {
            "duration_min": float(dur),
            "setpoint_display": float(sp_disp),
            "color": belt_col,
            "manual_capture_min": float(manual_cap),
        }
    belt_bg = {"BELT 1": lighten(st.session_state["belts_cfg"]["BELT 1"]["color"], 0.75)}

# chips de legenda visual das zonas
chips = " ".join(
    f"<span style='display:inline-block;padding:4px 8px;border-radius:12px;background:{(belt_bg.get(b) if 'belt_bg' in locals() else '#eee')};"
    f"border:1px solid #999;margin-right:6px;color:#000;font-weight:600'>{b}</span>" for b in belt_labels
)
st.markdown(chips, unsafe_allow_html=True)
st.divider()

# =========================
# CHANNEL SIDES (TXT/XML/XLSX/CSV)
# =========================
st.subheader("Channel Sides (only)")
ch_side = get_ch_side()
cols = st.columns(3); side_opts = ["Left","Center","Right"]
for idx,ch in enumerate(["CH1","CH2","CH3"]):
    with cols[idx]:
        ch_side[ch] = st.selectbox(f"{ch} • Side", side_opts, index=idx, key=f"side_{ch}")
st.session_state["ch_side"] = ch_side

# =========================
# DATA LOADING (TXT / XML / XLSX / CSV / Manual)
# =========================
frames: List[pd.DataFrame] = []

def _finish_loaded_df(df_all: pd.DataFrame):
    df_all = filter_allowed_channels(df_all)
    if "side" not in df_all.columns:
        df_all["side"] = df_all["channel"].map(get_ch_side()).fillna("Center")
    else:
        df_all["side"] = df_all["side"].fillna("Center")
    frames.append(df_all)

def decode_uploaded_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")

if data_mode == "TXT (SD card)":
    st.subheader("Upload TXT files (SD card)")
    st.caption("Arraste arquivos `.txt` / `.TXT` da Atlas/Apollo ou clique para selecionar. No macOS, o botão Upload só habilita depois de clicar no arquivo.")
    uploaded_txt = st.file_uploader(
        "Select one or more TXT files",
        type=["txt"],
        accept_multiple_files=True,
        key="upload_txt",
        help="Aceita arquivos TXT de SD card, inclusive Atlas Max Plus.",
    )
    if uploaded_txt:
        parsed=[]
        for uf in uploaded_txt:
            if not uf.name.lower().endswith(".txt"):
                st.warning(f"Ignoring unsupported file in TXT mode: {uf.name}")
                continue
            text=decode_uploaded_text(uf.getvalue())
            df=parse_sd_txt(text, uf.name)
            if not df.empty:
                parsed.append(df)
            else:
                st.warning(f"Could not parse TXT file: {uf.name}")
        if parsed:
            df_all=pd.concat(parsed, ignore_index=True)
            _finish_loaded_df(df_all)
            st.success(f"Imported {len(df_all)} rows from {len(parsed)} file(s).")
            with st.expander("Preview (first rows)"): st.dataframe(df_all.head(200), use_container_width=True)
        else: st.info("Could not parse valid data from the uploaded TXT files.")

elif data_mode == "XML":
    st.subheader("Upload XML or Excel profile")
    st.caption("Accepts .xml (placeholder) and .xlsx (curing profile style).")
    uploaded = st.file_uploader("Select one or more .XML / .XLSX",
                                type=["xml","XML","xlsx","XLSX"], accept_multiple_files=True, key="upload_xml_xlsx")
    if uploaded:
        parsed=[]
        for uf in uploaded:
            name=uf.name; data=uf.read()
            if name.lower().endswith((".xlsx",)):
                df=parse_profile_xlsx(io.BytesIO(data), name)
            else:
                try: text=data.decode(errors="ignore")
                except Exception: text=""
                df=parse_xml_placeholder(text, name)
            if not df.empty: parsed.append(df)
        if parsed:
            df_all=pd.concat(parsed, ignore_index=True)
            _finish_loaded_df(df_all)
            st.success(f"Imported {len(df_all)} rows from {len(parsed)} file(s).")
            with st.expander("Preview (first rows)", expanded=False): st.dataframe(df_all.head(200), use_container_width=True)
        else: st.info("Could not parse valid data from the uploaded files.")

elif data_mode == "CSV":
    st.subheader("Upload CSV")
    uploaded_csv = st.file_uploader("Select one or more .CSV", type=["csv", "CSV"], accept_multiple_files=True, key="upload_csv")
    if uploaded_csv:
        parsed=[]
        for uf in uploaded_csv:
            data=uf.read()
            df=parse_csv_generic(data, uf.name)
            if not df.empty: parsed.append(df)
        if parsed:
            df_all=pd.concat(parsed, ignore_index=True)
            _finish_loaded_df(df_all)
            st.success(f"Imported {len(df_all)} rows from {len(parsed)} file(s).")
            with st.expander("Preview (first rows)", expanded=False): st.dataframe(df_all.head(200), use_container_width=True)
        else: st.info("Could not parse valid data from CSV files.")

else:
    # ------------- MANUAL INPUT: S1/S2/S3 -------------
    st.subheader("Manual input")

    belt_options = belt_labels if belt_labels else ["BELT 1"]
    manual_belt = st.selectbox("Select belt for manual entry", belt_options, index=0, key="manual_selected_belt")
    belts_cfg = st.session_state.get("belts_cfg", {})
    manual_default = float(belts_cfg.get(manual_belt, {}).get("manual_capture_min", DEFAULT_BELT_DURATION_MIN))

    manual_defaults = st.session_state.setdefault("manual_duration_defaults", {})
    belt_changed = "manual_selected_belt_last" not in st.session_state or manual_belt != st.session_state["manual_selected_belt_last"]
    prev_default = manual_defaults.get(manual_belt)
    current_value = st.session_state.get("manual_duration_min")
    if belt_changed:
        st.session_state["manual_duration_min"] = manual_default
    else:
        if prev_default is None:
            st.session_state.setdefault("manual_duration_min", manual_default)
        elif manual_default != prev_default:
            if current_value is None or abs(float(current_value) - float(prev_default)) < 1e-6:
                st.session_state["manual_duration_min"] = manual_default
    manual_defaults[manual_belt] = manual_default
    st.session_state["manual_duration_defaults"] = manual_defaults
    st.session_state["manual_selected_belt_last"] = manual_belt
    st.session_state.setdefault("manual_duration_min", manual_default)

    c_top1, c_top2, c_top3 = st.columns(3)
    with c_top1:
        interval_s = st.number_input("Sample interval (seconds)", min_value=1, max_value=3600, value=30, step=1, key="manual_interval_s")
    with c_top2:
        duration_min_manual = st.number_input("Total duration (min)", min_value=0.1, max_value=480.0,
                                              value=float(st.session_state.get("manual_duration_min", manual_default)),
                                              step=0.5, key="manual_duration_min")
    with c_top3:
        manual_unit = st.selectbox("Manual data unit", UNIT_OPTIONS, index=0, key="manual_unit")

    st.markdown("**How many series will you enter? (map each to a Side below)**")
    n_series = st.number_input("Number of series (S1..S3)", min_value=1, max_value=3, value=1, step=1, key="manual_n_series")

    side_to_ch = {"Left": "CH1", "Center": "CH2", "Right": "CH3"}
    series_sides: List[str] = []
    cols_series = st.columns(n_series)
    for i in range(n_series):
        with cols_series[i]:
            series_sides.append(st.selectbox(f"S{i+1} side", ["Left","Center","Right"],
                                             index=min(i,2), key=f"manual_side_{i}"))

    headers = [f"S{i+1}" for i in range(n_series)]
    series_to_side = dict(zip(headers, series_sides))
    series_to_channel = {h: side_to_ch[series_to_side[h]] for h in headers}

    total_seconds = int(round(float(duration_min_manual)*60))
    interval_s = int(interval_s)
    times = list(range(0, total_seconds + 1, interval_s))
    if times[-1] != total_seconds: times.append(total_seconds)

    manual_meta = ("manual_S_headers", manual_belt, tuple(times), tuple(headers))
    if ("manual_table" not in st.session_state) or (st.session_state.get("manual_meta") != manual_meta):
        table = pd.DataFrame({"time_s": times, "time_min": [t/60 for t in times]})
        for h in headers: table[h] = pd.Series([None]*len(times), dtype="float")
        st.session_state["manual_table"] = table
        st.session_state["manual_meta"] = manual_meta

    edited = st.data_editor(st.session_state["manual_table"], num_rows="dynamic",
                            use_container_width=True, key="manual_editor")

    long_df = edited.melt(id_vars=["time_s","time_min"], value_vars=headers,
                          var_name="series", value_name="temp_val").dropna(subset=["temp_val"])
    long_df["channel"] = long_df["series"].map(series_to_channel)
    long_df["side"] = long_df["series"].map(series_to_side)
    long_df["temp_C"] = np.where(manual_unit=="°C", long_df["temp_val"].astype(float),
                                 long_df["temp_val"].astype(float).apply(f_to_c))
    long_df = long_df.rename(columns={"time_min":"elapsed_min"})
    long_df["source"] = f"{machine} (manual)"; long_df["unit"] = "C"; long_df["belt"] = manual_belt
    frames.append(long_df[["source","elapsed_min","series","channel","temp_C","unit","side","belt"]])

# =========================
# CONSOLIDATION & PREP
# =========================
df_final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df_final.empty:
    st.info("Load/enter data to render the chart."); st.stop()

df_final = filter_allowed_channels(df_final)
df_final["source"] = df_final["source"].astype(str) + f" — {machine}"
if "side" not in df_final.columns: df_final["side"] = "Center"
df_final["side"] = df_final["side"].astype(str)
if "series" not in df_final.columns: df_final["series"] = ""  # non-manual sources
if "belt" not in df_final.columns:
    df_final["belt"] = ""
df_final["belt"] = df_final["belt"].fillna("").astype(str)

# Build sequential belt windows + setpoints
windows: Dict[str, Tuple[float,float]] = {}; cum=0.0
setpoint_c: Dict[str,float] = {}; setpoint_display: Dict[str,float] = {}
belt_labels = [f"BELT {i}" for i in range(1, int(st.session_state["n_belts"])+1)]
for belt in belt_labels:
    cfg = st.session_state["belts_cfg"][belt]
    dur = float(cfg["duration_min"]); start, end = cum, cum+dur; windows[belt]=(start,end); cum=end
    sp_disp = float(cfg["setpoint_display"]); setpoint_display[belt]=sp_disp
    setpoint_c[belt] = sp_disp if unit_display=="°C" else f_to_c(sp_disp)

# Alignment
thresh_c = threshold_display if unit_display=="°C" else f_to_c(threshold_display)
df_c = align_by_threshold(df_final.copy(), thresh_c) if align_on else df_final.copy()
x_col = "elapsed_min_aligned" if align_on and "elapsed_min_aligned" in df_c.columns else "elapsed_min"

# Prepare for plotting
df_plot = df_c.copy(); df_plot["y"] = series_c_to_display(df_plot["temp_C"], unit_display)

# =========================
# AUTO-DETECT INTERVAL + VALIDATION
# =========================
def detect_interval_minutes(df: pd.DataFrame, xcolumn: str) -> float:
    x = pd.to_numeric(df[xcolumn], errors="coerce").dropna().sort_values().values
    if len(x) < 3: return np.nan
    diffs = np.diff(x); diffs = diffs[diffs > 0]
    if len(diffs) == 0: return np.nan
    return float(np.median(diffs))

def validate_dataset(df: pd.DataFrame, unit_display: str, xcolumn: str, setpoints: Dict[str,float]):
    msgs = []
    min_plaus_f, max_plaus_f = 50, 600
    y_vals_disp = df["y"].dropna().values
    if len(y_vals_disp):
        if unit_display=="°F" and np.nanmin(y_vals_disp) < min_plaus_f:
            msgs.append("Temperatures below ~50°F detected. Check units or sensor.")
        if unit_display=="°F" and np.nanmax(y_vals_disp) > max_plaus_f:
            msgs.append("Temperatures above ~600°F detected. Check units or sensor.")
    for (src,ch), g in df.groupby(["source","channel"]):
        gx = pd.to_numeric(g[xcolumn], errors="coerce").dropna().values
        if len(gx) >= 2 and np.any(np.diff(gx) < 0):
            msgs.append(f"Non-monotonic time sequence in {src} / {ch}.")
    approx_dt_min = detect_interval_minutes(df, xcolumn)
    if not np.isnan(approx_dt_min):
        s = int(round(approx_dt_min * 60))
        msgs.append(f"Detected sample interval ≈ {s} s.")
    for belt,(s,e) in windows.items():
        for (src,ch), g in df[(df[xcolumn]>=s)&(df[xcolumn]<=e)].groupby(["source","channel"]):
            if len(g) < 3:
                msgs.append(f"Few samples in window {belt} for {src}/{ch} — metrics may be unstable.")
    for b, sp in setpoint_display.items():
        if unit_display == "°F" and (sp < 150 or sp > 450):
            msgs.append(f"Unusual setpoint for {b}: {sp:.1f}°F (check recipe).")
        if unit_display == "°C" and (sp < 65 or sp > 230):
            msgs.append(f"Unusual setpoint for {b}: {sp:.1f}°C (check recipe).")
    if msgs:
        with st.expander("Validation & Data Quality", expanded=True):
            for m in msgs:
                st.warning(m)

validate_dataset(df_plot, unit_display, x_col, setpoint_c)

# =========================
# METRICS / INSIGHTS
# =========================
tmp = df_c.copy(); tmp["elapsed_min"] = tmp[x_col]
m = metrics_in_windows(tmp, windows, setpoint_c)

display_cols = {}
for base in ["Mean (°C)","Std (°C)","Min (°C)","Max (°C)","Range (°C)","Peak-to-Peak (°C)","RMS (°C)","Ramp Rate to SP (°C/min)"]:
    show = base.replace("(°C)", f"({unit_display})").replace("°C/min", f"{unit_display}/min")
    if unit_display == "°C":
        m[show] = m[base]
    else:
        if "Rate" in base:
            m[show] = m[base].apply(lambda v: v if pd.isna(v) else (v * 9.0/5.0))
        else:
            m[show] = m[base].apply(lambda v: v if pd.isna(v) else c_to_f(v))
    display_cols[base] = show

m = m.round(3)
insights = m.groupby("Belt").agg({
    "Dur (min)": "mean",
    "Time to SP (min)": "mean",
    display_cols["Mean (°C)"]: "mean",
    display_cols["Std (°C)"]: "mean",
    display_cols["Min (°C)"]: "min",
    display_cols["Max (°C)"]: "max",
    display_cols["Range (°C)"]: "mean",
    display_cols["Peak-to-Peak (°C)"]: "mean",
    display_cols["RMS (°C)"]: "mean",
    display_cols["Ramp Rate to SP (°C/min)"]: "mean",
    "% ≥ SP": "mean"
}).reset_index().round(3)

# =========================
# CHARTS
# =========================
plt, PdfPages, Line2D, Patch, FuncFormatter, MaxNLocator, MultipleLocator = load_plotting_libs(theme)
figures: List[Tuple[str, plt.Figure]] = []

total_window = max((end for (start, end) in windows.values()), default=0.0)
x_series = pd.to_numeric(df_plot[x_col], errors="coerce")
x_max_data = float(np.nanmax(x_series.values)) if not x_series.empty else 0.0
if not np.isfinite(x_max_data): x_max_data = 0.0
x_max = max(total_window, x_max_data)

def line_color(row) -> str:
    if series_color_mode and row.get("series"):
        return series_colors.get(str(row["series"]), channel_colors.get(str(row["channel"]), "#444"))
    return channel_colors.get(str(row["channel"]), "#444")

# OVERVIEW
fig, ax = plt.subplots(figsize=(22, 9))
fig.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.12)

for belt,(start,end) in windows.items():
    color_zone = lighten(st.session_state["belts_cfg"][belt]["color"], 0.75)
    ax.axvspan(start, end, alpha=0.35, color=color_zone)

grp_cols = ["source","channel","side"] if not series_color_mode else ["source","series","channel","side"]
for keys, g in df_plot.groupby(grp_cols):
    sample_row = g.iloc[0]
    color = line_color(sample_row)
    label = " — ".join([str(k) for k in keys if k])
    ax.plot(g[x_col], g["y"], label=label, color=color, linewidth=2)

for b in belt_labels:
    ax.axhline(y=setpoint_display[b], linewidth=1.8, linestyle="--",
               color=st.session_state["belts_cfg"][b]["color"], label=f"{b} SP {setpoint_display[b]:.0f}{unit_display.replace('°','')}")

if show_minmax:
    ax.axhline(y=min_line, color="g", linewidth=2, label="Min")
    ax.axhline(y=max_line, color="r", linewidth=2, label="Max")

def apply_time_axis(ax, x_max: float, mode: str, max_n: int):
    if mode == "Auto":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_n, prune='both'))
    else:
        step_map = {"Every 5 s": 5, "Every 10 s": 10, "Every 15 s": 15, "Every 30 s": 30, "Every 1 min": 60, "Every 2 min": 120}
        step_s = step_map.get(mode, 30)
        ax.xaxis.set_major_locator(MultipleLocator(step_s/60.0))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_mmss))
    ax.set_xlim(0, x_max if x_max>0 else None)

apply_time_axis(ax, x_max, st.session_state.get("sel_tick_mode","Auto"), st.session_state.get("sld_max_labels", 12))
ax.set_xlabel("Elapsed time (mm:ss)"+(" (aligned)" if align_on else ""))
ax.set_ylabel(f"Temperature ({unit_display})")
ax.set_title(f"Temperature Curves — {machine} (Belts define Heat Zones)")

zone_patches = [Patch(facecolor=lighten(st.session_state["belts_cfg"][b]["color"], 0.75), edgecolor="none", alpha=0.8, label=b) for b in belt_labels]
chan_handles = [Line2D([0],[0], color=channel_colors[ch], lw=2, label=ch) for ch in ["CH1","CH2","CH3"]]
extra = [Line2D([0],[0], color=st.session_state["belts_cfg"][b]["color"], lw=2, linestyle="--", label=f"{b} SP") for b in belt_labels]
if show_minmax: extra += [Line2D([0],[0], color="g", lw=2, label="Min"), Line2D([0],[0], color="r", lw=2, label="Max")]

leg1 = ax.legend(handles=zone_patches, title="Heat Zones", loc="upper left", bbox_to_anchor=(1.01,1.0), borderaxespad=0.)
ax.add_artist(leg1)

if (series_color_mode or st.session_state.get("toggle_series_legend", False)) and (df_plot["series"] != "").any():
    series_used = sorted([s for s in df_plot["series"].unique() if s])
    series_handles = [Line2D([0],[0], color=series_colors.get(s, "#666"), lw=2, label=s) for s in series_used]
    ax.legend(handles=series_handles+extra, title="Series / Setpoints", loc="lower left",
              bbox_to_anchor=(1.01,0.0), borderaxespad=0.)
else:
    ax.legend(handles=chan_handles+extra, title="Channels / Setpoints", loc="lower left",
              bbox_to_anchor=(1.01,0.0), borderaxespad=0.)

ax.grid(True, alpha=0.25)
st.pyplot(fig, clear_figure=False, use_container_width=True)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
st.download_button("Download Overview (PNG)", data=buf.getvalue(), file_name="overview.png",
                   mime="image/png", key="dl_overview_png")
figures.append(("Overview", fig))

# INSIGHTS
st.markdown("### Insights (by Belt)")
st.dataframe(insights, use_container_width=True)

# PER-BELT
for belt in belt_labels:
    start, end = windows[belt]
    sp_disp = setpoint_display[belt]
    subset = df_plot[(df_plot[x_col]>=start) & (df_plot[x_col]<=end)]
    if subset.empty: continue

    fig_b, ax_b = plt.subplots(figsize=(20, 7))
    fig_b.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.14)
    ax_b.axvspan(start, end, alpha=0.35, color=lighten(st.session_state["belts_cfg"][belt]["color"], 0.75))

    used_channels=set(); used_series=set()
    grp_cols_b = ["source","channel","side"] if not series_color_mode else ["source","series","channel","side"]
    for keys, g in subset.groupby(grp_cols_b):
        sample_row = g.iloc[0]
        color = line_color(sample_row)
        ax_b.plot(g[x_col], g["y"], label=" — ".join([str(k) for k in keys if k]), color=color, linewidth=2)
        used_channels.add(str(sample_row["channel"]))
        if sample_row.get("series"): used_series.add(str(sample_row["series"]))

    ax_b.axhline(y=sp_disp, linewidth=1.8, linestyle="--",
                 label=f"SP {sp_disp:.0f}{unit_display.replace('°','')}",
                 color=st.session_state["belts_cfg"][belt]["color"])

    apply_time_axis(ax_b, end, st.session_state.get("sel_tick_mode","Auto"), st.session_state.get("sld_max_labels",12))
    ax_b.set_xlim(start, end)
    ax_b.set_xlabel("Elapsed time (mm:ss)"+(" (aligned)" if align_on else ""))
    ax_b.set_ylabel(f"Temperature ({unit_display})")
    ax_b.set_title(f"{belt} — Heat Zone Window")

    extra_belt = [Line2D([0],[0], color=st.session_state["belts_cfg"][belt]["color"], lw=2, linestyle="--",
                         label=f"SP {sp_disp:.0f}{unit_display.replace('°','')}")]

    if (series_color_mode or st.session_state.get("toggle_series_legend", False)) and used_series:
        series_handles_belt = [Line2D([0],[0], color=series_colors.get(s, "#666"), lw=2, label=s) for s in sorted(used_series)]
        ax_b.legend(handles=series_handles_belt+extra_belt, title="Series / SP", loc="center left",
                    bbox_to_anchor=(1.01,0.5), borderaxespad=0.)
    else:
        chan_handles_belt = [Line2D([0],[0], color=channel_colors[ch], lw=2, label=ch) for ch in sorted(used_channels)]
        ax_b.legend(handles=chan_handles_belt+extra_belt, title="Channels / SP", loc="center left",
                    bbox_to_anchor=(1.01,0.5), borderaxespad=0.)

    ax_b.grid(True, alpha=0.25)
    st.pyplot(fig_b, clear_figure=False, use_container_width=True)

    buf_b = io.BytesIO()
    fig_b.savefig(buf_b, format="png", dpi=150, bbox_inches="tight")
    st.download_button(f"Download {belt} (PNG)", data=buf_b.getvalue(),
                       file_name=f"{belt.lower().replace(' ','_')}.png", mime="image/png", key=f"dl_{belt}_png")
    figures.append((belt, fig_b))

# =========================
# EXPORT CSV
# =========================
def assign_zone(t: float) -> str:
    for b,(s,e) in windows.items():
        if s <= t <= e: return b
    return ""

csv_df = df_plot.copy()
csv_df["unit_display"] = unit_display
csv_df["channel_color"] = csv_df["channel"].map(channel_colors)
csv_df["series_color"] = np.where(csv_df.get("series","")!="", csv_df["series"].map(DEFAULT_SERIES_COLORS), None)
csv_df["side"] = csv_df["side"].astype(str)
csv_df["zone"] = csv_df[x_col].apply(assign_zone)
st.download_button("Download CSV (consolidated data)",
                   data=csv_df.to_csv(index=False).encode("utf-8"),
                   file_name="temperature_vs_time.csv", mime="text/csv", key="dl_csv_all")

# =========================
# PDF REPORT (Header + pages)
# =========================
def dataframe_to_fig(df: pd.DataFrame, title: str, dark: bool = (theme=="Dark")) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis('off')
    if dark: ax.set_facecolor("#0e1117"); fig.patch.set_facecolor("#0e1117"); txt="#EAEAEA"
    else: txt="#222222"
    ax.set_title(title, fontsize=14, color=txt, pad=10)
    tb = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', colLoc='center', loc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.2)
    return fig

def header_page(machine: str,
                unit_display: str,
                belts_cfg: Dict[str, Dict[str, float]],
                ch_side: Dict[str,str],
                channel_colors: Dict[str,str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis('off')
    if theme=="Dark":
        ax.set_facecolor("#0e1117"); fig.patch.set_facecolor("#0e1117"); txt="#EAEAEA"
    else:
        txt="#222222"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"Temperature vs Time — Report\nMachine: {machine} | Unit: {unit_display}\nGenerated: {now}"
    ax.text(0.02, 0.92, title, fontsize=14, color=txt, va="top", ha="left", fontweight="bold")

    lines=[]
    lines.append("Belts & Setpoints:")
    for b,cfg in belts_cfg.items():
        lines.append(
            f"  • {b}: Duration {cfg['duration_min']:.2f} min | Manual {cfg.get('manual_capture_min', cfg['duration_min']):.2f} min "
            f"| SP {cfg['setpoint_display']:.1f}{unit_display.replace('°','')} | Color {cfg['color']}"
        )
    lines.append("")
    lines.append("Channels:")
    for ch, side in ch_side.items():
        col = channel_colors.get(ch, "#000000")
        lines.append(f"  • {ch}: Side {side} | Color {col}")
    ax.text(0.02, 0.55, "\n".join(lines), fontsize=11, color=txt, va="top", ha="left", family="monospace")

    ax.text(0.02, 0.08, "Notes:", fontsize=12, color=txt, va="bottom", ha="left", fontweight="bold")
    ax.text(0.02, 0.05, "— This report includes Overview, per-Belt charts, and summary metrics (Insights).",
            fontsize=10, color=txt, va="bottom", ha="left")
    return fig

st.markdown("### Report")
if st.button("📄 Download PDF Report", use_container_width=True, key="btn_pdf"):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        fig_hdr = header_page(machine, unit_display, st.session_state["belts_cfg"], get_ch_side(), channel_colors)
        pdf.savefig(fig_hdr, bbox_inches='tight')
        fig_ins = dataframe_to_fig(insights, "Insights (by Belt)", dark=(theme=="Dark"))
        pdf.savefig(fig_ins, bbox_inches='tight')
        for title, fig_save in figures:
            pdf.savefig(fig_save, bbox_inches='tight')
        fig_m = dataframe_to_fig(m.round(3), "Metrics per Belt Window (per Source/Channel)", dark=(theme=="Dark"))
        pdf.savefig(fig_m, bbox_inches='tight')
    st.download_button("Save PDF", data=pdf_buf.getvalue(),
                       file_name="belts_heatzones_report.pdf", mime="application/pdf", key="dl_pdf_report")

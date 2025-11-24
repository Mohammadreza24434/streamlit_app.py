import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from io import BytesIO

st.set_page_config(page_title="MZT Pro", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main {background: #0a0e17; color: #e0e0e0;}
    .sidebar .sidebar-content {background: linear-gradient(135deg, #1e3c72, #2a5298);}
    h1, h2, h3 {color: #00ffff; text-align: center; text-shadow: 0 0 10px #00ffff;}
    .stButton>button {background: linear-gradient(45deg, #00d4ff, #0099cc); color: white; border: none;
                      border-radius: 12px; font-weight: bold; box-shadow: 0 0 15px #00ffff;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>MZT Pro</h1>", unsafe_allow_html=True)
st.markdown("<h2>Advanced Gaussian Dispersion Modeling</h2>", unsafe_allow_html=True)
st.markdown("---")

CHEMICALS = {
    "Ammonia (NH3)":           {"molwt":17.03, "IDLH":300,  "ERPG1":25,   "ERPG2":200,  "ERPG3":1000},
    "Chlorine (Cl2)":          {"molwt":70.90, "IDLH":10,   "ERPG1":1,    "ERPG2":3,    "ERPG3":20},
    "Hydrogen Sulfide (H2S)":  {"molwt":34.08, "IDLH":100,  "ERPG1":0.5,  "ERPG2":30,   "ERPG3":100},
    "Sulfur Dioxide (SO2)":    {"molwt":64.06, "IDLH":100,  "ERPG1":0.3,  "ERPG2":15,   "ERPG3":75},
    "Hydrogen Fluoride (HF)":  {"molwt":20.01, "IDLH":30,   "ERPG1":5,    "ERPG2":20,   "ERPG3":50},
    "Hydrogen Chloride (HCl)": {"molwt":36.46, "IDLH":50,   "ERPG1":3,    "ERPG2":20,   "ERPG3":150},
    "Phosgene":                {"molwt":98.92, "IDLH":2,    "ERPG1":0.2,  "ERPG2":0.5,  "ERPG3":1},
    "Methyl Isocyanate":       {"molwt":57.05, "IDLH":3,    "ERPG1":0.2,  "ERPG2":1,    "ERPG3":5},
    "Hydrogen Cyanide":        {"molwt":27.03, "IDLH":50,   "ERPG1":10,   "ERPG2":20,   "ERPG3":50},
    "Bromine":                 {"molwt":159.8, "IDLH":3,    "ERPG1":0.1,  "ERPG2":0.5,  "ERPG3":5},
    "Acrylonitrile":           {"molwt":53.06, "IDLH":85,   "ERPG1":10,   "ERPG2":35,   "ERPG3":75},
    "Ethylene Oxide":          {"molwt":44.05, "IDLH":800,  "ERPG1":5,    "ERPG2":50,   "ERPG3":500},
    "Formaldehyde":            {"molwt":30.03, "IDLH":20,   "ERPG1":1,    "ERPG2":10,   "ERPG3":25},
    "Carbon Monoxide":         {"molwt":28.01, "IDLH":1200, "ERPG1":200,  "ERPG2":350,  "ERPG3":500},
    "Methanol":                {"molwt":32.04, "IDLH":6000, "ERPG1":200,  "ERPG2":1000, "ERPG3":5000},
    "Benzene":                 {"molwt":78.11, "IDLH":500,  "ERPG1":50,   "ERPG2":150,  "ERPG3":1000},
    "Toluene":                 {"molwt":92.14, "IDLH":500,  "ERPG1":50,   "ERPG2":300,  "ERPG3":1000},
    "Phosphine":               {"molwt":34.00, "IDLH":50,   "ERPG1":0.3,  "ERPG2":2,    "ERPG3":10},
    "Methane":                 {"molwt":16.04, "LEL":5.0},
    "Propane":                 {"molwt":44.10, "LEL":2.1},
    "Butane":                  {"molwt":58.12, "LEL":1.8},
    "Hydrogen":                {"molwt":2.02,  "LEL":4.0},
    "LPG":                     {"molwt":48.0,  "LEL":2.0},
    "LNG":                     {"molwt":17.0,  "LEL":5.0},
    "Acetone":                 {"molwt":58.08, "LEL":2.5},
    "Ethanol":                 {"molwt":46.07, "IDLH":3300, "ERPG1":100,  "ERPG2":500,  "ERPG3":2500},
    "Nitric Acid":             {"molwt":63.01, "IDLH":25},
    "Sulfuric Acid":           {"molwt":98.08, "IDLH":15},
    "Oleum":                   {"molwt":80.06, "IDLH":10},
    "Aniline":                 {"molwt":93.13, "IDLH":100},
    "Carbon Disulfide":        {"molwt":76.14, "IDLH":500,  "ERPG1":10,   "ERPG2":50,   "ERPG3":200},
    "Acrolein":                {"molwt":56.06, "IDLH":2,    "ERPG1":0.1,  "ERPG2":0.5,  "ERPG3":3},
    "Vinyl Chloride":          {"molwt":62.50, "IDLH":1000, "ERPG1":10,   "ERPG2":50,   "ERPG3":500},
    "Styrene":                 {"molwt":104.15,"IDLH":700,  "ERPG1":20,   "ERPG2":100,  "ERPG3":500},
    "Hydrazine":               {"molwt":32.05, "IDLH":50,   "ERPG1":0.5,  "ERPG2":3,    "ERPG3":30},
    "Chloroform":              {"molwt":119.38,"IDLH":500},
    "Dichloromethane":         {"molwt":84.93, "IDLH":2300},
    "Arsine":                  {"molwt":77.95, "IDLH":3},
    "Fluorine":                {"molwt":38.00, "IDLH":25},
    "Nitrogen Dioxide":        {"molwt":46.01, "IDLH":20,   "ERPG1":1,    "ERPG2":15,   "ERPG3":30}
}

def advanced_gaussian(Q, u, H, stability, x_max=60):
    x = np.linspace(10, x_max*1000, 1200)
    y = np.linspace(-x_max*1000, x_max*1000, 1200)
    X, Y = np.meshgrid(x, y)
    sy = {'A':0.22,'B':0.16,'C':0.11,'D':0.08,'E':0.06,'F':0.04}.get(stability,0.08) * X**0.894
    sz = {'A':0.20,'B':0.12,'C':0.08,'D':0.06,'E':0.03,'F':0.02}.get(stability,0.06) * X**0.894
    C = (Q/(2*np.pi*u*sy*sz)) * np.exp(-0.5*(Y/sy)**2) * (np.exp(-0.5*((0-H)/sz)**2) + np.exp(-0.5*((0+H)/sz)**2))
    return X/1000, Y/1000, np.nan_to_num(C, nan=0.0)

with st.sidebar:
    st.header("Scenario Parameters")
    chem_name = st.selectbox("Select Chemical", sorted(CHEMICALS.keys()))
    chem = CHEMICALS[chem_name]
    molwt = chem["molwt"]
    Q = st.slider("Release Rate (g/s)", 10, 50000, 5000, 500)
    u = st.slider("Wind Speed (m/s)", 0.5, 25.0, 4.0, 0.1)
    temp_c = st.slider("Temperature (°C)", -10, 50, 25)
    H = st.slider("Release Height (m)", 0.0, 150.0, 5.0, 0.5)
    stability = st.selectbox("Stability Class", ["A","B","C","D","E","F"], index=3)
    view_mode = st.radio("Display Mode", ["Filled Contours", "Line Contours"])

X_km, Y_km, C_gm3 = advanced_gaussian(Q, u, H, stability)
C_ppm = C_gm3 * 24.45 / molwt * (298.15 / (temp_c + 273.15))

fig, ax = plt.subplots(figsize=(18, 12))

if view_mode == "Filled Contours":
    cont = ax.contourf(X_km, Y_km, C_ppm, levels=80, cmap="inferno", norm=LogNorm(vmin=1e-3, vmax=C_ppm.max()), alpha=0.92)
    plt.colorbar(cont, ax=ax, shrink=0.75, pad=0.02, label="Concentration (ppm)")

else:
    levels = np.logspace(np.log10(max(0.01, C_ppm.max()*1e-6)), np.log10(C_ppm.max()), 25)
    ax.contour(X_km, Y_km, C_ppm, levels=levels, colors='#00ffff', linewidths=1.6, alpha=0.95)

zones = [("ERPG3","red",6), ("ERPG2","orange",5), ("ERPG1","yellow",4), ("IDLH","crimson",5), ("LEL","magenta",6)]
threat_info = []

for key, color, width in zones:
    if key in chem:
        level = chem[key] * (10000 if key == "LEL" else 1)
        if C_ppm.max() >= level * 0.4:
            ax.contour(X_km, Y_km, C_ppm, levels=[level], colors=color, linewidths=width)
            unit = "%" if key == "LEL" else "ppm"
            threat_info.append(f"<span style='color:{color};font-weight:bold'>{key}</span>: {chem[key]:.2f} {unit}")

ax.set_title(f"MZT Pro — {chem_name}\nQ: {Q:,} g/s | u: {u} m/s | H: {H} m | Class: {stability}", 
             fontsize=20, color="#00ffff", pad=30, weight="bold")
ax.set_xlabel("Downwind Distance (km)", fontsize=14, color="white")
ax.set_ylabel("Crosswind Distance (km)", fontsize=14, color="white")
ax.grid(True, alpha=0.2, color="#333333")
ax.set_facecolor("#0d1117")
fig.patch.set_facecolor("#0a0e17")
ax.tick_params(colors='white')
ax.set_xlim(0, 50)
ax.set_ylim(-25, 25)
ax.axis("equal")

st.pyplot(fig)

if threat_info:
    st.markdown("### Threat Zones")
    st.markdown("<div style='background:#111;padding:15px;border-radius:10px;text-align:center;font-size:16px'>" +
                "  |  ".join(threat_info) + "</div>", unsafe_allow_html=True)

st.markdown("### Maximum Threat Distances (km)")
cols = st.columns(5)
for col, name, color in zip(cols, ["ERPG-3", "ERPG-2", "ERPG-1", "IDLH", "LEL"], ["red", "orange", "yellow", "crimson", "magenta"]):
    key = name.replace("-", "") if "ERPG" in name else name
    if key in chem:
        level = chem[key] * (10000 if key == "LEL" else 1)
        dists = X_km[0][np.where(np.max(C_ppm, axis=0) >= level)]
        dist = dists[-1] if len(dists)>0 else 0.0
        col.markdown(f"<h4 style='color:{color}'>{name}</h4><h1 style='color:{color}'>{dist:.2f}</h1>", unsafe_allow_html=True)
    else:
        col.markdown(f"<h4 style='color:gray'>{name}</h4><h2>N/A</h2>", unsafe_allow_html=True)

buf = BytesIO()
fig.savefig(buf, format="png", dpi=500, bbox_inches="tight", facecolor="#0a0e17")
buf.seek(0)
st.download_button("Download Map (500 DPI PNG)", buf, f"MZT_Pro_{chem_name.replace(' ', '_')}.png", "image/png")

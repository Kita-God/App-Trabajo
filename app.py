# -*- coding: utf-8 -*-
"""
Yessi Collection — Sistema de gestión de inventario
Multimaterial (kardex con base de datos SQLite) + política SS-ROP-EOQ para lana
+ pronóstico Holt-Winters + panel gerencial.
"""

import os
import math
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statistics import NormalDist

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    TIENE_HW = True
except Exception:
    TIENE_HW = False

import plotly.graph_objects as go


@st.cache_data
def pronosticar(serie_tuple):
    serie = list(serie_tuple)
    m = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12).fit()
    return [max(0.0, float(x)) for x in m.forecast(12)]


# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
st.set_page_config(page_title="Yessi Collection · Inventario",
                   page_icon="🧶", layout="wide")

MARCA, MARCA_LT, MARCA_DK = "#8C3A3A", "#EFE0E0", "#6E2D2D"
VERDE, AMARILLO, ROJO = "#C6EFCE", "#FFEB9C", "#FFC7CE"
VERDE_TX, ROJO_TX = "#1B5E20", "#9C0006"
MESES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
         "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
MESES_CORTO = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
MESES_ESP = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
             "septiembre","octubre","noviembre","diciembre"]
TEMP_ALTA = [5, 6, 7, 8]   # mayo a agosto (temporada alta de quiebres)

# Demanda histórica (tabla maestra oficial)
DEM_2024 = [950, 980, 1000, 1050, 1150, 1200, 1200, 1150, 1000, 950, 920, 980]
DEM_2025 = [950, 980, 1000, 1100, 1250, 1300, 1350, 1300, 1100, 1000, 920, 1050]

# Catálogo de materiales (lana = Clase A con política; resto mínimo-máximo)
MATERIALES = {
    "Lana dralón":       {"unidad": "kg",    "clase": "A", "min": None},
    "Hilo de remalle":   {"unidad": "conos", "clase": "B", "min": 20},
    "Botones":           {"unidad": "unid",  "clase": "B", "min": 400},
    "Cierres":           {"unidad": "unid",  "clase": "B", "min": 100},
    "Etiquetas":         {"unidad": "unid",  "clase": "C", "min": 500},
    "Bolsas de empaque": {"unidad": "unid",  "clase": "C", "min": 500},
    "Agujas":            {"unidad": "unid",  "clase": "C", "min": 20},
}
SEED = [("Lana dralón", 180.0), ("Hilo de remalle", 60.0), ("Botones", 980.0),
        ("Cierres", 300.0), ("Etiquetas", 1500.0), ("Bolsas de empaque", 1400.0),
        ("Agujas", 50.0)]
DB_PATH = "inventario_yessi.db"

# =====================================================================
# BASE DE DATOS (SQLite) — el historial NO se borra
# =====================================================================
def _con():
    return sqlite3.connect(DB_PATH)

def init_db():
    c = _con()
    c.execute("""CREATE TABLE IF NOT EXISTS mov(
        id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, material TEXT,
        tipo TEXT, cantidad REAL, comentario TEXT)""")
    c.commit()
    if c.execute("SELECT COUNT(*) FROM mov").fetchone()[0] == 0:
        hoy = datetime.now().strftime("%Y-%m-%d")
        c.executemany("INSERT INTO mov(fecha,material,tipo,cantidad,comentario) VALUES(?,?,?,?,?)",
                      [(hoy, m, "Entrada", q, "Saldo inicial") for m, q in SEED])
        c.commit()
    c.close()

def add_mov(fecha, material, tipo, cantidad, comentario):
    c = _con()
    c.execute("INSERT INTO mov(fecha,material,tipo,cantidad,comentario) VALUES(?,?,?,?,?)",
              (fecha, material, tipo, float(cantidad), comentario))
    c.commit(); c.close()

def get_movs(material=None):
    c = _con()
    if material:
        df = pd.read_sql_query("SELECT fecha AS Fecha, tipo AS Tipo, cantidad AS Cantidad, "
                               "comentario AS Comentario FROM mov WHERE material=? ORDER BY id DESC",
                               c, params=(material,))
    else:
        df = pd.read_sql_query("SELECT fecha AS Fecha, material AS Material, tipo AS Tipo, "
                               "cantidad AS Cantidad, comentario AS Comentario FROM mov ORDER BY id DESC", c)
    c.close(); return df

def saldo(material):
    c = _con()
    e = c.execute("SELECT COALESCE(SUM(cantidad),0) FROM mov WHERE material=? AND tipo='Entrada'",
                  (material,)).fetchone()[0]
    s = c.execute("SELECT COALESCE(SUM(cantidad),0) FROM mov WHERE material=? AND tipo='Salida'",
                  (material,)).fetchone()[0]
    c.close(); return e - s

def borrar_ultimo(material):
    c = _con()
    row = c.execute("SELECT id FROM mov WHERE material=? ORDER BY id DESC LIMIT 1",
                    (material,)).fetchone()
    if row:
        c.execute("DELETE FROM mov WHERE id=?", (row[0],)); c.commit()
    c.close()

init_db()

# =====================================================================
# ESTILO DE MARCA (CSS)
# =====================================================================
st.markdown(f"""
<style>
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
.stTabs [aria-selected="true"] {{ color: {MARCA} !important; }}
div[data-testid="stMetricValue"] {{ color: {MARCA}; font-weight: 700; }}
h1, h2, h3 {{ color: {MARCA}; }}
section[data-testid="stSidebar"] {{ background-color: {MARCA_LT}; }}
.stButton button {{ border-radius: 8px; }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

def kpi_card(icono, titulo, valor, sub, fondo, texto):
    return (f"<div style='background:{fondo};padding:18px 12px;border-radius:16px;"
            f"box-shadow:0 4px 14px rgba(140,58,58,0.10);text-align:center;height:158px;"
            f"display:flex;flex-direction:column;justify-content:center;'>"
            f"<div style='font-size:28px;line-height:1;'>{icono}</div>"
            f"<div style='font-size:11px;color:{texto};opacity:0.75;text-transform:uppercase;"
            f"letter-spacing:1px;margin-top:5px;'>{titulo}</div>"
            f"<div style='font-size:30px;font-weight:800;color:{texto};line-height:1.15;'>{valor}</div>"
            f"<div style='font-size:11.5px;color:{texto};opacity:0.7;'>{sub}</div></div>")

# =====================================================================
# ENCABEZADO
# =====================================================================
st.markdown(
    f"<div style='display:flex;align-items:center;gap:14px;'>"
    f"<div style='font-size:40px;'>🧶</div><div>"
    f"<h1 style='color:{MARCA};margin:0;font-size:30px;'>Yessi Collection</h1>"
    f"<p style='color:gray;margin:0;font-size:15px;'>Sistema de gestión de inventario</p>"
    f"</div></div>", unsafe_allow_html=True)
st.divider()

# =====================================================================
# BARRA LATERAL — parámetros de la política de lana
# =====================================================================
st.sidebar.header("⚙️ Parámetros de la política")
st.sidebar.caption("Aplican a la lana dralón (material crítico). Cambiar solo si cambian "
                   "las condiciones del negocio.")
ratio   = st.sidebar.number_input("Lana por chompa (kg)", 0.1, value=0.4, step=0.05, format="%.2f")
L       = st.sidebar.number_input("Lead time normal (días)", 1, value=3, step=1)
L_alta  = st.sidebar.number_input("Lead time temporada alta (días)", 1, value=5, step=1)
sigma_L = st.sidebar.number_input("Variabilidad del lead time (días)", 0.0, value=1.0, step=0.5, format="%.1f")
NS      = st.sidebar.slider("Nivel de servicio (%)", 80, 99, 95)
Cu      = st.sidebar.number_input("Costo lana (S/ /kg)", 1.0, value=30.0, step=1.0, format="%.2f")
i_pct   = st.sidebar.number_input("Tasa mantener inventario (%/año)", 1, value=25, step=1)
S_ped   = st.sidebar.number_input("Costo emitir pedido (S/)", 1.0, value=33.0, step=1.0, format="%.2f")

with st.sidebar.expander("📁 Datos de demanda (uso anual)"):
    df_in = pd.DataFrame({"Mes": MESES, "Demanda 2024": DEM_2024, "Demanda 2025": DEM_2025})
    df_e = st.data_editor(df_in, hide_index=True, use_container_width=True, disabled=["Mes"])
dem24 = [float(x) for x in df_e["Demanda 2024"].tolist()]
dem25 = [float(x) for x in df_e["Demanda 2025"].tolist()]

# =====================================================================
# CÁLCULOS DE LA POLÍTICA (lana, base demanda)
# =====================================================================
kg25 = [d * ratio for d in dem25]
D = sum(kg25); DIAS_ANIO = 264
d_barra = D / DIAS_ANIO
tasas = [k / 22 for k in kg25]
sigma_d = float(np.std(tasas, ddof=1))
d_pico = max(kg25) / 22
Z = NormalDist().inv_cdf(NS / 100)
H = (i_pct / 100) * Cu
SS = Z * np.sqrt(L * sigma_d**2 + d_barra**2 * sigma_L**2)
ROP = d_barra * L + SS
ROP_alta = d_pico * L_alta + SS
EOQ = np.sqrt(2 * D * S_ped / H)
N = D / EOQ
costo_pol = N * S_ped + (EOQ / 2 + SS) * H

hoy = datetime.now(); mes_idx = hoy.month - 1
es_alta = hoy.month in TEMP_ALTA
ROP_vig = ROP_alta if es_alta else ROP
tasa_hoy = tasas[mes_idx] if tasas[mes_idx] > 0 else d_barra

# Saldos de todos los materiales
saldos = {m: saldo(m) for m in MATERIALES}
saldo_lana = saldos["Lana dralón"]
dias_cob = saldo_lana / tasa_hoy if tasa_hoy > 0 else 0

def estado_material(mat):
    s = saldos[mat]
    if mat == "Lana dralón":
        return ("PEDIR", ROJO, ROJO_TX) if s <= ROP_vig else ("OK", VERDE, VERDE_TX)
    mn = MATERIALES[mat]["min"]
    return ("PEDIR", ROJO, ROJO_TX) if s <= mn else ("OK", VERDE, VERDE_TX)

estado_lana = estado_material("Lana dralón")[0]
n_pedir = sum(1 for m in MATERIALES if estado_material(m)[0] == "PEDIR")

# =====================================================================
# PESTAÑAS
# =====================================================================
t_panel, t_alerta, t_inv, t_pol, t_plan = st.tabs([
    "📋 Panel gerencial", "🔔 Alerta diaria", "📦 Inventario y movimientos",
    "📊 Política de lana", "📅 Plan de pedidos"])

# ─────────────── PANEL GERENCIAL ───────────────
with t_panel:
    st.markdown(f"<h3 style='color:{MARCA};margin-bottom:2px;'>📊 Panel gerencial</h3>"
                f"<p style='color:gray;margin-top:0;'>Vista general del inventario en tiempo real</p>",
                unsafe_allow_html=True)

    color_est = ROJO if estado_lana == "PEDIR" else VERDE
    txt_est = "🔴 REQUIERE PEDIDO" if estado_lana == "PEDIR" else "🟢 ABASTECIDO"
    temp_ban = "temporada alta 🔥" if es_alta else "temporada normal ❄️"
    extra = f" · {n_pedir} material(es) por pedir" if n_pedir else " · todos los materiales OK"
    st.markdown(
        f"<div style='background:{color_est};padding:18px;border-radius:14px;text-align:center;"
        f"font-size:22px;font-weight:bold;'>Lana dralón: {txt_est} — {saldo_lana:.0f} kg "
        f"· {dias_cob:.0f} días de cobertura · {temp_ban}{extra}</div>", unsafe_allow_html=True)
    st.write("")

    col_g, col_d = st.columns([1.1, 2])
    with col_g:
        max_g = math.ceil(max(saldo_lana * 1.15, ROP_alta + EOQ) / 50) * 50
        fig_d = go.Figure(go.Indicator(
            mode="gauge+number", value=saldo_lana,
            number={"suffix": " kg", "font": {"size": 30}},
            title={"text": "Nivel de lana", "font": {"size": 15}},
            gauge={"axis": {"range": [0, max_g]},
                   "bar": {"color": MARCA, "thickness": 0.28},
                   "steps": [{"range": [0, SS], "color": ROJO},
                             {"range": [SS, ROP_vig], "color": AMARILLO},
                             {"range": [ROP_vig, max_g], "color": VERDE}],
                   "threshold": {"line": {"color": MARCA, "width": 4}, "thickness": 0.85,
                                 "value": ROP_vig}}))
        fig_d.update_layout(height=260, margin=dict(t=42, b=5, l=15, r=15))
        st.plotly_chart(fig_d, use_container_width=True)
    with col_d:
        st.markdown("<p style='color:gray;font-weight:700;font-size:13px;letter-spacing:1px;"
                    "margin-bottom:8px;'>📐 POLÍTICA DE LANA</p>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.markdown(kpi_card("🛡️", "Stock seguridad", f"{SS:.0f} kg", "reserva mínima",
                             "#F7F2F2", "#333"), unsafe_allow_html=True)
        p2.markdown(kpi_card("🎯", "Punto de reorden", f"{ROP_vig:.0f} kg", "pedir aquí",
                             "#F7F2F2", "#333"), unsafe_allow_html=True)
        p3.markdown(kpi_card("📦", "Cantidad a pedir", f"{EOQ:.0f} kg", "cada pedido",
                             "#F7F2F2", "#333"), unsafe_allow_html=True)
        st.write("")
        st.markdown("<p style='color:gray;font-weight:700;font-size:13px;letter-spacing:1px;"
                    "margin-bottom:8px;'>🎯 MEJORA ESPERADA</p>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        q1.markdown(kpi_card("✅", "Disponibilidad", "≥ 98%", "antes 90%",
                             "#E8F5E9", VERDE_TX), unsafe_allow_html=True)
        q2.markdown(kpi_card("🚫", "Quiebres / año", "0", "antes varios",
                             "#E8F5E9", VERDE_TX), unsafe_allow_html=True)
        q3.markdown(kpi_card("🧶", "Lana faltante", "0 kg", "antes 512 kg",
                             "#E8F5E9", VERDE_TX), unsafe_allow_html=True)

    st.write("")
    st.markdown("<p style='color:gray;font-weight:700;font-size:13px;letter-spacing:1px;"
                "margin-bottom:6px;'>📦 ESTADO DE TODOS LOS MATERIALES</p>", unsafe_allow_html=True)
    filas = []
    for m, info in MATERIALES.items():
        est, _, _ = estado_material(m)
        ref = f"{ROP_vig:.0f} (ROP)" if m == "Lana dralón" else f"{info['min']:.0f} (nivel mínimo)"
        filas.append({"Material": m, "Clase": info["clase"],
                      "Saldo": f"{saldos[m]:.0f} {info['unidad']}",
                      "Punto de pedido": ref, "Estado": "🔴 PEDIR" if est == "PEDIR" else "🟢 OK"})
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.caption("📌 Los saldos provienen del kardex (pestaña Inventario y movimientos), guardado "
               "en base de datos. La lana usa la política SS-ROP-EOQ; los demás, control mínimo-máximo.")

# ─────────────── ALERTA DIARIA ───────────────
with t_alerta:
    st.subheader("¿Tengo que pedir lana hoy?")
    temp_txt = "ALTA 🔥" if es_alta else "NORMAL ❄️"
    st.info(f"📅 Hoy es **{hoy.strftime('%d/%m/%Y')}** — {MESES[mes_idx]}, temporada **{temp_txt}**. "
            f"Punto de reorden vigente: **{ROP_vig:.0f} kg**")

    pedido_transito = st.number_input("📦 ¿Pedido de lana ya realizado y en camino? (kg) — 0 si no",
                                      min_value=0.0, value=0.0, step=10.0, format="%.1f")
    posicion = saldo_lana + pedido_transito

    col_g, col_i = st.columns([1, 1])
    with col_g:
        max_g = math.ceil(max(saldo_lana * 1.15, ROP_alta + EOQ) / 50) * 50
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=saldo_lana, number={"suffix": " kg", "font": {"size": 32}},
            title={"text": "Nivel de lana en almacén", "font": {"size": 16}},
            gauge={"axis": {"range": [0, max_g]}, "bar": {"color": MARCA, "thickness": 0.28},
                   "steps": [{"range": [0, SS], "color": ROJO}, {"range": [SS, ROP_vig], "color": AMARILLO},
                             {"range": [ROP_vig, max_g], "color": VERDE}],
                   "threshold": {"line": {"color": MARCA, "width": 4}, "thickness": 0.85, "value": ROP_vig}}))
        fig_g.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption(f"🔴 Crítico < {SS:.0f} · 🟡 Pedir {SS:.0f}–{ROP_vig:.0f} · 🟢 OK > {ROP_vig:.0f} kg")
    with col_i:
        st.write("")
        if posicion <= ROP_vig:
            st.markdown(f"<div style='background:{ROJO};color:{ROJO_TX};padding:22px;border-radius:12px;"
                        f"font-size:26px;font-weight:bold;text-align:center;'>🔴 PEDIR AHORA<br>{EOQ:.0f} kg</div>",
                        unsafe_allow_html=True)
            st.write("")
            st.metric("Días de cobertura restantes", f"{dias_cob:.0f} días")
            if dias_cob < L_alta:
                st.error(f"⚠️ Urgente: cobertura de {dias_cob:.0f} días y el proveedor puede tardar "
                         f"hasta {L_alta} días. Pide hoy mismo.")
        else:
            st.markdown(f"<div style='background:{VERDE};color:{VERDE_TX};padding:22px;border-radius:12px;"
                        f"font-size:26px;font-weight:bold;text-align:center;'>🟢 OK<br>No es necesario pedir</div>",
                        unsafe_allow_html=True)
            st.write("")
            dias_rop = max(0, (posicion - ROP_vig) / tasa_hoy) if tasa_hoy > 0 else 0
            f_ped = hoy + timedelta(days=int(dias_rop))
            f_str = f"{f_ped.day} de {MESES_ESP[f_ped.month-1]} de {f_ped.year}"
            st.metric("Días de cobertura", f"{dias_cob:.0f} días")
            st.markdown(f"<div style='background:{MARCA_LT};border-left:5px solid {MARCA};padding:14px;"
                        f"border-radius:8px;'>📆 <b>Próxima fecha sugerida de pedido:</b><br>"
                        f"<span style='font-size:20px;color:{MARCA};font-weight:bold;'>{f_str}</span>"
                        f"<br><span style='color:gray;font-size:12px;'>En ~{int(dias_rop)} días</span></div>",
                        unsafe_allow_html=True)

    # Otros materiales por pedir
    otros = [m for m in MATERIALES if m != "Lana dralón" and estado_material(m)[0] == "PEDIR"]
    st.write("")
    if otros:
        st.warning("🔔 **Otros materiales que también requieren pedido:** " +
                   ", ".join(f"{m} (saldo {saldos[m]:.0f} {MATERIALES[m]['unidad']}, "
                             f"mínimo {MATERIALES[m]['min']:.0f})" for m in otros))
    else:
        st.success("✅ Los demás materiales (hilos, botones, etc.) están por encima de su nivel mínimo.")
    st.caption("El saldo se calcula automáticamente desde el kardex. La app detecta la fecha y aplica "
               "el punto de reorden de temporada alta o normal según el mes.")

# ─────────────── INVENTARIO Y MOVIMIENTOS (multimaterial) ───────────────
with t_inv:
    st.subheader("Inventario y registro de movimientos")
    st.caption("Registra cada entrada (llegó material) y salida (se usó en producción) de cualquier "
               "material. El saldo se calcula solo y se guarda en base de datos: no se borra al cerrar la app.")

    # Selección de material
    mat = st.selectbox("Material", list(MATERIALES.keys()))
    info = MATERIALES[mat]; s = saldos[mat]; est, col_e, tx_e = estado_material(mat)
    ref_txt = (f"Punto de reorden: {ROP_vig:.0f} kg" if mat == "Lana dralón"
               else f"Nivel mínimo: {info['min']:.0f} {info['unidad']}")

    cA, cB = st.columns([1.4, 1])
    with cA:
        st.markdown(f"<div style='background:{col_e};padding:18px;border-radius:12px;text-align:center;'>"
                    f"<span style='color:{tx_e};font-size:13px;'>Saldo actual de {mat}</span><br>"
                    f"<span style='font-size:32px;color:{tx_e};font-weight:bold;'>{s:.0f} {info['unidad']}</span>"
                    f"<br><span style='color:{tx_e};font-size:13px;'>{'🔴 Hay que pedir' if est=='PEDIR' else '🟢 Nivel suficiente'} · {ref_txt}</span></div>",
                    unsafe_allow_html=True)
    with cB:
        st.metric("Clase ABC", info["clase"])
        st.caption("A = política SS-ROP-EOQ\nB y C = mínimo-máximo")

    st.markdown("##### ➕ Registrar movimiento de " + mat)
    with st.form("nuevo_mov", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        f_fecha = c1.date_input("Fecha", value=hoy)
        f_tipo = c2.selectbox("Tipo", ["Entrada", "Salida"])
        f_cant = c3.number_input(f"Cantidad ({info['unidad']})", min_value=0.0, step=5.0, format="%.1f")
        f_com = st.text_input("Comentario (opcional)", placeholder="Ej: recepción de pedido / consumo")
        ok = st.form_submit_button("➕ Registrar", use_container_width=True)
        if ok and f_cant > 0:
            add_mov(str(f_fecha), mat, f_tipo, f_cant, f_com)
            st.rerun()

    st.markdown(f"##### 📒 Historial de {mat}")
    dfh = get_movs(mat)
    if not dfh.empty:
        dfh_disp = dfh.copy(); dfh_disp["Cantidad"] = dfh_disp["Cantidad"].map(lambda x: f"{x:.1f}")
        st.dataframe(dfh_disp, hide_index=True, use_container_width=True)
        b1, b2 = st.columns(2)
        if b1.button(f"↩️ Deshacer último movimiento de {mat}", use_container_width=True):
            borrar_ultimo(mat); st.rerun()
        b2.caption("El historial se guarda en la base de datos `inventario_yessi.db`.")
    else:
        st.caption("Sin movimientos para este material todavía.")

# ─────────────── POLÍTICA DE LANA ───────────────
with t_pol:
    st.subheader("Política de inventario de la lana dralón")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stock de seguridad (SS)", f"{SS:.1f} kg")
    c2.metric("ROP — temporada normal", f"{ROP:.0f} kg")
    c3.metric("ROP — temporada alta", f"{ROP_alta:.0f} kg")
    c4, c5, c6 = st.columns(3)
    c4.metric("Cantidad económica (EOQ)", f"{EOQ:.0f} kg")
    c5.metric("Pedidos al año", f"{N:.0f}", help=f"Uno cada {DIAS_ANIO/N:.0f} días")
    c6.metric("Costo anual de la política", f"S/ {costo_pol:,.0f}")

    st.divider()
    st.markdown("#### ❓ ¿Qué significa cada término?")
    with st.expander("📦 Stock de seguridad (SS)"):
        st.write(f"Reserva de lana que se mantiene siempre como colchón ante atrasos del proveedor o "
                 f"consumo mayor al normal. Para Yessi: **{SS:.0f} kg**.")
        st.latex(r"SS = Z \cdot \sqrt{L \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2}")
    with st.expander("🎯 Punto de reorden (ROP)"):
        st.write(f"Nivel en que se debe pedir. En temporada alta sube a **{ROP_alta:.0f} kg** por el mayor consumo.")
        st.latex(r"ROP = \bar{d} \cdot L + SS")
    with st.expander("📊 Cantidad económica de pedido (EOQ)"):
        st.write(f"Cuánto pedir cada vez para minimizar costos. Para Yessi: **{EOQ:.0f} kg**.")
        st.latex(r"EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}")
    with st.expander("🛡️ Nivel de servicio"):
        st.write(f"Probabilidad de no quedarse sin stock durante la espera del proveedor: **{NS}%**.")
    with st.expander("🚚 Lead time"):
        st.write(f"Tiempo de entrega del proveedor: **{L} días** normal y **{L_alta} días** en temporada alta.")
    with st.expander("🔮 Pronóstico Holt-Winters"):
        st.write("Predice la demanda futura aprendiendo la tendencia y la estacionalidad del histórico.")
    st.info(f"📌 Demanda anual de lana: {D:,.0f} kg = {sum(dem25):,.0f} chompas × {ratio:.2f} kg/chompa.")

# ─────────────── PLAN DE PEDIDOS ───────────────
with t_plan:
    st.subheader("Plan de pedidos de lana para el próximo año")
    st.caption("Pronóstico Holt-Winters sobre la demanda 2024–2025.")
    if not TIENE_HW:
        st.error("Instala 'statsmodels' para ver el pronóstico.")
    else:
        serie = dem24 + dem25
        fc = pronosticar(tuple(serie))
        anio = 2026
        filas = []
        for k, mes in enumerate(MESES):
            lana = fc[k] * ratio
            alta = (k + 1) in TEMP_ALTA
            filas.append({"Mes": mes, "Demanda proyectada (und)": round(fc[k]),
                          "Lana requerida (kg)": round(lana),
                          "Temporada": "🔥 Alta" if alta else "Normal",
                          "Pedidos estimados": max(1, round(lana / EOQ))})
        dfp = pd.DataFrame(filas)
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(f"Demanda proyectada {anio}", f"{sum(fc):,.0f} und")
        cc2.metric(f"Lana requerida {anio}", f"{sum(fc)*ratio:,.0f} kg")
        cc3.metric("Crecimiento vs 2025", f"{(sum(fc)/sum(dem25)-1)*100:+.1f}%")
        st.dataframe(dfp, hide_index=True, use_container_width=True)

        x_h = [f"{m}-24" for m in MESES_CORTO] + [f"{m}-25" for m in MESES_CORTO]
        x_f = [f"{m}-26" for m in MESES_CORTO]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_h, y=serie, mode="lines+markers", name="Histórico (real)",
                                 line=dict(color=MARCA, width=2.5), marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=[x_h[-1]] + x_f, y=[serie[-1]] + fc, mode="lines+markers",
                                 name=f"Pronóstico {anio}", line=dict(color="#C9A96B", width=2.5, dash="dash"),
                                 marker=dict(size=5)))
        fig.update_layout(title="Demanda de chompas: histórico y pronóstico", xaxis_title="Mes",
                          yaxis_title="Unidades", height=420, legend=dict(orientation="h", y=1.12),
                          plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"))
        fig.update_xaxes(tickangle=-55)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("🧶 La lana dralón (Clase A) se gestiona con política SS-ROP-EOQ y pronóstico. "
           "Los demás materiales (Clase B y C) con control mínimo-máximo. Historial en base de datos SQLite.")

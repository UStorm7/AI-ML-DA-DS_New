"""Interactive dashboard. Run with:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.build_warehouse import build  # noqa: E402
from src.config import DB_PATH  # noqa: E402

st.set_page_config(page_title="Olist E-commerce Analytics", layout="wide")
BLUE, ORANGE = "#2a78d6", "#eb6834"


@st.cache_resource
def connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        build().close()
    return duckdb.connect(str(DB_PATH), read_only=True)


con = connection()
st.title("Olist E-commerce: delivery, satisfaction & retention")

states = con.execute("SELECT DISTINCT customer_state FROM fct_delivered ORDER BY 1").df()["customer_state"].tolist()
cats = con.execute(
    "SELECT main_category FROM fct_delivered GROUP BY 1 HAVING COUNT(*) >= 300 ORDER BY COUNT(*) DESC"
).df()["main_category"].tolist()

c1, c2 = st.columns(2)
sel_states = c1.multiselect("Customer state", states, placeholder="All states")
sel_cats = c2.multiselect("Product category", cats, placeholder="All categories")

where, params = ["1=1"], []
if sel_states:
    where.append(f"customer_state IN ({','.join('?' * len(sel_states))})")
    params += sel_states
if sel_cats:
    where.append(f"main_category IN ({','.join('?' * len(sel_cats))})")
    params += sel_cats
flt = " AND ".join(where)


def q(sql: str):
    return con.execute(sql.format(flt=flt), params).df()


kpi = q("""SELECT COUNT(*) orders, SUM(payment_value) gmv, AVG(payment_value) aov,
                  AVG(is_late::INT)*100 late_pct, AVG(review_score) review
           FROM fct_delivered WHERE {flt}""").iloc[0]
k = st.columns(5)
k[0].metric("Orders", f"{kpi.orders:,.0f}")
k[1].metric("GMV", f"R$ {kpi.gmv / 1e6:,.2f}M")
k[2].metric("Avg order value", f"R$ {kpi.aov:,.0f}")
k[3].metric("Late deliveries", f"{kpi.late_pct:.1f}%")
k[4].metric("Avg review", f"{kpi.review:.2f} / 5")

monthly = q("""SELECT order_month, COUNT(*) orders, AVG(is_late::INT)*100 late_pct
               FROM fct_delivered WHERE {flt} GROUP BY 1 ORDER BY 1""")
left, right = st.columns(2)
left.plotly_chart(px.line(monthly, x="order_month", y="orders", title="Orders per month",
                          color_discrete_sequence=[BLUE]), use_container_width=True)
right.plotly_chart(px.line(monthly, x="order_month", y="late_pct", title="Late-delivery rate (%)",
                           color_discrete_sequence=[ORANGE]), use_container_width=True)

delay = q("""SELECT ROUND(GREATEST(LEAST(days_late, 20), -30)) AS days_vs_promise,
                    AVG(review_score) avg_review, COUNT(*) orders
             FROM fct_delivered WHERE {flt} AND review_score IS NOT NULL GROUP BY 1 ORDER BY 1""")
fig = px.scatter(delay, x="days_vs_promise", y="avg_review", size="orders",
                 title="Average review vs. days delivered after promised date (negative = early)",
                 color_discrete_sequence=[BLUE])
fig.add_vline(x=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

st.subheader("States")
st.dataframe(q("""SELECT customer_state, COUNT(*) orders, ROUND(AVG(is_late::INT)*100,1) late_pct,
                         ROUND(AVG(delivery_days),1) avg_delivery_days, ROUND(AVG(review_score),2) avg_review
                  FROM fct_delivered WHERE {flt} GROUP BY 1 ORDER BY orders DESC"""),
             use_container_width=True, hide_index=True)

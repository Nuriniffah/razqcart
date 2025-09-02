import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests

# ---------- CONFIGURATION ----------
st.set_page_config(page_title="Business Insights Dashboard", layout="wide")

# ---------- USER AUTH CHECK ----------
if "user" not in st.session_state or not st.session_state.user:
    st.error("🔒 Please log in first via the Home page.")
    st.stop()

# ---------- LOAD ORDER DATA ----------
@st.cache_data
def load_order_data():
    df = pd.read_csv('order.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=False, errors='coerce')
    df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
    if 'Revenue' not in df.columns:
        df['Revenue'] = pd.to_numeric(df['Quantity'], errors='coerce') * df['Amount']
    return df

df_orders = load_order_data()

# ---------- RFM CUSTOMER SEGMENTATION ----------
def rfm_segmentation(df):
    NOW = df['Order Date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer').agg({
        'Order Date': lambda x: (NOW - x.max()).days,
        'ID': 'nunique',
        'Revenue': 'sum'
    }).reset_index()

    rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']

    rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)

    def segment_customer(row):
        if row['R_score'] >= 4 and row['F_score'] >= 3 and row['M_score'] >= 3:
            return 'High Value'
        elif row['R_score'] >= 3 and row['F_score'] == 1:
            return 'New'
        elif row['R_score'] == 1 and row['F_score'] <= 2:
            return 'At-Risk'
        else:
            return 'Loyal'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm

rfm = rfm_segmentation(df_orders)

# ---------- MAIN DASHBOARD ----------
st.title("Business Insights Dashboard")

# ---------- KPI SECTION ----------
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Customers", len(rfm))
col2.metric("💰 Total Revenue", f"${rfm['Monetary'].sum():,.2f}")
col3.metric("📦 Avg Order Value", f"${rfm['Monetary'].mean():,.2f}")

st.markdown("---")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["Customer Segments", "Sales Anomalies"])

# ------------------ TAB 1: CUSTOMER SEGMENTS ------------------
with tab1:
    st.subheader("RFM-Based Customer Segmentation")

    segment_summary = rfm['Segment'].value_counts().reset_index()
    segment_summary.columns = ['Segment', 'Count']

    col1, col2 = st.columns([1, 2])
    with col1:
        st.table(segment_summary)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=segment_summary, x='Segment', y='Count', palette='Set2', ax=ax)
        ax.set_title("Customer Segments")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Number of Customers")
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)

    with st.expander("RFM Data"):
        st.dataframe(rfm[['Customer', 'Recency', 'Frequency', 'Monetary', 'Segment']], use_container_width=True)

    st.subheader("Recency Insights")
    st.metric("Average Recency (days)", f"{rfm['Recency'].mean():.1f}")

    st.markdown("Most Recent Customers")
    most_recent = rfm.sort_values('Recency').head(5)
    st.dataframe(most_recent[['Customer', 'Recency', 'Frequency', 'Monetary']])

# ------------------ TAB 2: SALES ANOMALY DETECTION ------------------
with tab2:
    st.subheader("Sales Anomaly Detection")

    @st.cache_data
    def load_anomaly_data():
        df = pd.read_csv("synthetic_anomaly_data.csv")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df.set_index("Date", inplace=True)
        return df

    df_anomaly = load_anomaly_data()

    # Anomaly detection
    window = 10
    df_anomaly["RollingMean"] = df_anomaly["Value"].rolling(window).mean()
    df_anomaly["RollingStd"] = df_anomaly["Value"].rolling(window).std()
    df_anomaly["Zscore"] = (df_anomaly["Value"] - df_anomaly["RollingMean"]) / df_anomaly["RollingStd"]
    df_anomaly["Anomaly"] = df_anomaly["Zscore"].abs() > 3
    anomalies = df_anomaly[df_anomaly["Anomaly"]]

    st.line_chart(df_anomaly["Value"])

    if not anomalies.empty:
        st.warning("⚠️ Anomalies Detected!")
        st.dataframe(anomalies[["Value", "Zscore"]])
    else:
        st.success("✅ No anomalies detected.")

# ------------------ TAB 3: INVENTORY (OPTIONAL) ------------------
# with tab3:
#     st.subheader("📦 Real-Time Inventory Monitor")

#     API_URL = "http://127.0.0.1:5000/api/stock"

#     @st.cache_data(ttl=10)
#     def fetch_live_stock():
#         try:
#             response = requests.get(API_URL)
#             data = response.json()
#             return pd.DataFrame(data)
#         except:
#             return pd.DataFrame(columns=["item_id", "item_name", "quantity"])

#     df_stock = fetch_live_stock()

#     if df_stock.empty:
#         st.info("No inventory data available. Is your API running?")
#     else:
#         st.dataframe(df_stock)

#         for idx, row in df_stock.iterrows():
#             col1, col2, col3 = st.columns([4, 1, 1])
#             col1.write(f"**{row['item_name']}** — {row['quantity']} in stock")
#             if col2.button("➕ Restock", key=f"restock_{idx}"):
#                 r = requests.post(f"{API_URL}/update", json={"item_id": int(row['item_id']), "amount": 10})
#                 st.success(f"Restocked {row['item_name']}")
#                 st.rerun()
#             if col3.button("➖ Decrease", key=f"decrease_{idx}"):
#                 r = requests.post(f"{API_URL}/update", json={"item_id": int(row['item_id']), "amount": -1})
#                 st.warning(f"Decreased {row['item_name']}")
#                 st.rerun()

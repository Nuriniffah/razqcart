import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
# from mail import send_email_alert

st.set_page_config(page_title="Business Insights", layout="wide")
st.title("Sales Demand")

# --- Tabs for Forecast and Segmentation ---
tab1, tab2 = st.tabs(["Customer Segments", "Sales Anomaly"])

# ----------------------
# TAB 1: Forecast & Anomaly
# ----------------------
with tab1:
    st.subheader("")
    # --- Load Data ---
df = pd.read_csv('order.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Clean monetary columns
df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

if 'Revenue' not in df.columns:
    df['Revenue'] = pd.to_numeric(df['Quantity'], errors='coerce') * pd.to_numeric(df['Amount'], errors='coerce')

# Reference date for Recency calculation
NOW = df['Order Date'].max() + pd.Timedelta(days=1)

# --- RFM Calculation ---
rfm = df.groupby('Customer').agg({
    'Order Date': lambda x: (NOW - x.max()).days,      # Recency
    'ID': 'nunique',                                   # Frequency
    'Revenue': 'sum'                                   # Monetary
}).reset_index()

rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']

# --- Scoring ---
rfm['R_score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)

# --- Segment Customers ---
def segment_customer(row):
    if row['R_score'] >= 4 and row['F_score'] >= 3 and row['M_score'] >= 3:
        return 'High Value'
    elif row['R_score'] >= 3 and row['F_score'] == 1:
        return 'New'
    elif row['R_score'] == 1 and row['F_score'] <= 2:
        return 'At-Risk'
    else:
        return 'Loyal'  # Fallback segment

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Customer Segments", layout="wide")
st.subheader("Customer Segmentation")
st.write("AI-labeled segments based on RFM (Recency, Frequency, Monetary) analysis.")

# --- Segment Summary Table ---
segment_summary_df = rfm['Segment'].value_counts().reset_index()
segment_summary_df.columns = ['Segment', 'Count']

st.subheader("Segment Summary Table")
st.table(segment_summary_df)

# --- Matplotlib Bar Chart with Straight X-axis Labels ---
st.subheader("Segment Distribution")

fig, ax = plt.subplots(figsize=(6, 3))  # Width x Height in inches
sns.barplot(data=segment_summary_df, x='Segment', y='Count', palette='Set2', ax=ax)

# Make x-axis labels straight
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title("Customer Segments (RFM)", fontsize=10, fontweight='normal')
ax.set_ylabel("Customers", fontsize=5, fontweight='light')
ax.set_xlabel("Segment", fontsize=5, fontweight='light')

# Add count labels on top of bars
for i, row in segment_summary_df.iterrows():
   ax.text(i, row['Count'] + 0.5, str(row['Count']), ha='center', fontsize=9, fontweight='light')


st.pyplot(fig)

# --- Optional: Show RFM table
st.subheader("🔍 RFM Sample Data")
st.dataframe(rfm[['Customer', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(10))

# --- 📅 Recency Insights ---
st.subheader("Recency Insights")

# Average recency
average_recency = rfm['Recency'].mean()
st.metric(label="Average Recency (days)", value=f"{average_recency:.1f}")

# Top 5 most recent customers
st.markdown("Most Recent Customers")
most_recent = rfm.sort_values('Recency').head(5)[['Customer', 'Recency', 'Frequency', 'Monetary']]
st.dataframe(most_recent)



#     API_URL = "http://127.0.0.1:5000/api/stock"

#     @st.cache_data(ttl=10)
#     def fetch_live_stock():
#         response = requests.get(API_URL)
#         data = response.json()
#         return pd.DataFrame(data)

#     df_stock = fetch_live_stock()

#     st.subheader("📦 Real-Time Stock Dashboard")
#     st.dataframe(df_stock)

# # --- Load Data ---
# df = pd.read_csv('orders.csv')
# df['Order Date'] = pd.to_datetime(df['Order Date'])

# # --- Streamlit UI ---
# st.set_page_config(page_title="📦 Item Forecast", layout="wide")
# st.title("📦 Stock Prediction (SARIMA)")
# st.write("Forecast monthly item demand based on historical order data.")

# # --- Select Item ---
# available_items = df['Item'].dropna().unique()
# selected_item = st.selectbox("Select an Item to Forecast", sorted(available_items))

# # --- Select Forecast Period ---
# forecast_periods = st.slider("Forecast Period (Months)", min_value=3, max_value=24, value=6)

# if selected_item:
#     st.success(f"Forecasting for: **{selected_item}**")

#     # --- Prepare Data ---
#     item_ts = df[df['Item'] == selected_item][['Order Date', 'Quantity']]
#     item_ts.set_index('Order Date', inplace=True)
#     monthly_ts = item_ts['Quantity'].resample('MS').sum()

#     if len(monthly_ts) < 6:
#         st.warning("Not enough data to build a reliable forecast.")
#     else:
#         # --- Fit SARIMA Model ---
#         model = SARIMAX(monthly_ts,
#                         order=(1, 1, 1),
#                         seasonal_order=(1, 1, 0, 12),
#                         enforce_stationarity=False,
#                         enforce_invertibility=False)

#         results = model.fit(disp=False)

#         # --- Forecast ---
#         forecast = results.get_forecast(steps=forecast_periods)
#         predicted_mean = forecast.predicted_mean
#         confidence_intervals = forecast.conf_int()

#         # --- Forecast Table ---
#         forecast_df = predicted_mean.reset_index()
#         forecast_df.columns = ['Date', 'Predicted Quantity']
#         forecast_df['Predicted Quantity'] = forecast_df['Predicted Quantity'].round().astype(int)

#         st.subheader("📊 Forecasted Demand")
#         st.table(forecast_df)

#         # --- Forecast Plot ---
#         st.subheader("📈 Forecast Plot")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.plot(monthly_ts, label='Historical Monthly Sales', color='royalblue')
#         ax.plot(predicted_mean, label='Forecasted Sales', color='darkorange', linestyle='--')
#         ax.fill_between(confidence_intervals.index,
#                         confidence_intervals.iloc[:, 0],
#                         confidence_intervals.iloc[:, 1],
#                         color='orange', alpha=0.2, label='Confidence Interval')

#         ax.set_title(f'Stock Prediction for: {selected_item}', fontsize=14)
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Quantity Sold')
#         ax.legend()
#         ax.grid(True, linestyle='--', alpha=0.6)

#         st.pyplot(fig)

# ----------------------
# TAB 2: Customer Segments
# ----------------------
# st.title("📊 Forecast") 

if "user" not in st.session_state or not st.session_state.user:
    st.error("🔒 Please log in first via the Home page.")
    st.stop()
    
# Load your data
df = pd.read_csv("synthetic_anomaly_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Anomaly detection
window = 10
df["RollingMean"] = df["Value"].rolling(window).mean()
df["RollingStd"] = df["Value"].rolling(window).std()
df["Zscore"] = (df["Value"] - df["RollingMean"]) / df["RollingStd"]
df["Anomaly"] = df["Zscore"].abs() > 3

anomalies = df[df["Anomaly"]]

# Streamlit UI
st.title("Sales Anomaly Detection")
st.line_chart(df["Value"])

if not anomalies.empty:
    st.warning("Anomalies Detected!")
    st.dataframe(anomalies[["Value", "Zscore"]])

    # Optional: trigger email alert
    msg = ""
    for _, row in anomalies.iterrows():
        msg += f"{row.name.date()}: Value = {row['Value']:.2f}, Z-score = {row['Zscore']:.2f}\n"

    send_email_alert("Anomaly Detected in Dashboard", msg, "nuriniffah18@gmail.com")
else:
    st.success("No anomalies detected.")

# Wait 30 seconds before rerun (careful: this blocks UI)
time.sleep(30)
st.rerun()

for idx, row in df.iterrows():
    col1, col2, col3 = st.columns([4, 1, 1])
    col1.write(f"**{row['item_name']}** — {row['quantity']} in stock")
    if col2.button("Restock", key=f"restock_{idx}"):
        r = requests.post(f"{API_URL}/update", json={"item_id": int(row['item_id']), "amount": 10})
        st.success(f"Restocked {row['item_name']}")
        st.rerun()
    if col3.button("Decrease", key=f"decrease_{idx}"):
        r = requests.post(f"{API_URL}/update", json={"item_id": int(row['item_id']), "amount": -1})
        st.warning(f"Decreased {row['item_name']}")
        st.rerun()


API_URL = "http://127.0.0.1:5000/api/stock"

@st.cache_data(ttl=10)  # refresh every 10 seconds
def fetch_live_stock():
    response = requests.get(API_URL)
    data = response.json()
    df = pd.DataFrame(data)
    return df

df = fetch_live_stock()

st.set_page_config(page_title="Real-Time Stock Dashboard", layout="wide")
st.title("Real-Time Stock Dashboard")
st.dataframe(df)


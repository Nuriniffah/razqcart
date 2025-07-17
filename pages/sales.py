import streamlit as st
import pandas as pd

st.title("Sales Overview")


if "user" not in st.session_state or not st.session_state.user:
    st.error("🔒 Please log in first via the Home page.")
    st.stop()
    
# --- Load order data ---
df = pd.read_csv("order.csv")  # Make sure this file has 'Order Date', 'Amount', and 'ID' or similar

# --- Clean data ---
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

# --- Filter today's sales ---
today = pd.Timestamp.today().normalize()
today_sales = df[df['Order Date'].dt.normalize() == today]

# --- Metrics ---
total_sales = len(today_sales)
total_revenue = today_sales['Amount'].sum()

# --- Page UI ---
st.set_page_config(page_title="🏠 Home", layout="wide")
st.title("Daily Sales")

col1, col2 = st.columns(2)
col1.metric("🛒 Sales Today", f"{total_sales}")
col2.metric("💰 Revenue Today", f"${total_revenue:,.2f}")

st.markdown("---")
st.subheader("📅 Recent Sales (Today)")
st.dataframe(today_sales[['Order Date', 'Customer', 'Amount']].sort_values('Order Date', ascending=False))


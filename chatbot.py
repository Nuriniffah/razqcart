import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("👥 Customer Segments") 

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
# st.title("👥 Customer Segmentation Dashboard")
st.write("AI-labeled segments based on RFM (Recency, Frequency, Monetary) analysis.")

# --- Segment Summary Table ---
segment_summary_df = rfm['Segment'].value_counts().reset_index()
segment_summary_df.columns = ['Segment', 'Count']

st.subheader("📊 Segment Summary Table")
st.table(segment_summary_df)

# --- Matplotlib Bar Chart with Straight X-axis Labels ---
st.subheader("📈 Segment Distribution")

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
st.subheader("📅 Recency Insights")

# Average recency
average_recency = rfm['Recency'].mean()
st.metric(label="Average Recency (days)", value=f"{average_recency:.1f}")

# Top 5 most recent customers
st.markdown("#### 🏅 Most Recent Customers")
most_recent = rfm.sort_values('Recency').head(5)[['Customer', 'Recency', 'Frequency', 'Monetary']]
st.dataframe(most_recent)

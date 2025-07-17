import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Inventory Management") 

if "user" not in st.session_state or not st.session_state.user:
    st.error("Please log in first via the Home page.")
    st.stop()

# Load CSV
df = pd.read_csv('inventories.csv')

st.set_page_config(page_title="Inventory", layout="wide")

st.dataframe({
    "Product": ["Milk", "Flour", "Sugar", "Cheese", "Bread", "Egg", "Butter", "Chocolate chip", "Vanilla", "Peanut butter"],
    "Current Stock": ["2 cartons", "5.65 kg", "2 kg", "8 kg", "2.1 kg", "20", "500g", "2 kg", "50 ml", "454g"],
    #  "Demand": ["High", "High", "Medium", "Low", "Medium", "Medium", "High", "Medium", "High", "Low"]
})

# st.set_page_config(page_title="📦 Demand Predictor", layout="centered")
# st.title("📈 Demand Prediction from Inventory Data")

# Load CSV
df = pd.read_csv('inventories.csv')

# Verify columns
required_columns = ['Item', 'Demand']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# Encode item names
label_encoder = LabelEncoder()
df['Item_encoded'] = label_encoder.fit_transform(df['Item'])

# Prepare features
X = df[['Item_encoded']]
y = df['Demand']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# User input: Dropdown for item
item_selected = st.selectbox("Select an item to predict its demand:", sorted(df['Item'].unique()))

# Predict button
if st.button("🔍 Predict Demand"):
    item_encoded = label_encoder.transform([item_selected])[0]
    input_df = pd.DataFrame({'Item_encoded': [item_encoded]})
    predicted_demand = clf.predict(input_df)[0]
    st.success(f"📦 Predicted demand for **{item_selected}** is: **{predicted_demand}**")

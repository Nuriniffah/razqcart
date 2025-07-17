import streamlit as st
import streamlit as st
from supabase import create_client, Client

# Supabase setup
SUPABASE_URL = "https://wrdnoxojriebevriwivj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndyZG5veG9qcmllYmV2cml3aXZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0MzczMzEsImV4cCI6MjA2NzAxMzMzMX0.tPOulv3Hxg57h-xmmSANqtdNs7SauoGyCEKZJ45qGSo"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def require_login():
    if "user" not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        st.set_page_config(page_title="Login", layout="centered")
        st.title("Secure Login")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                st.session_state.user = response.user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid login credentials.")
        st.stop()  # Stops the rest of the app from running

st.set_page_config(page_title="Secure Dashboard", layout="wide")

# Session state for user login
if "user" not in st.session_state:
    st.session_state.user = None

# Force login page first
if not st.session_state.user:
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state.user = response.user
            st.success("Login successful! Reloading...")
            st.rerun()
        else:
            st.error("Invalid credentials or unverified account.")
    st.stop()

# If logged in: show your main dashboard tabs
st.sidebar.success(f"Logged in as: {st.session_state.user.email}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---

pg = st.navigation(pages=["pages/sales.py","pages/analytics.py", "pages/contact.py", "pages/inventory.py"])

# # --- RUN NAVIGATION ---
pg.run()
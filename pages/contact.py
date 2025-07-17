import streamlit as st
import datetime
import pandas as pd
from supabase import create_client, Client
import smtplib
from email.mime.text import MIMEText

# Supabase setup
SUPABASE_URL = "https://wrdnoxojriebevriwivj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndyZG5veG9qcmllYmV2cml3aXZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0MzczMzEsImV4cCI6MjA2NzAxMzMzMX0.tPOulv3Hxg57h-xmmSANqtdNs7SauoGyCEKZJ45qGSo"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Email (SMTP) setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "nuriniffah18@gmail.com"
EMAIL_PASSWORD = "Hello1234"  # Use an app password, not your real one

def send_email(name, email, message):
    subject = "New Contact Form Submission"
    body = f"From: {name}\nEmail: {email}\n\nMessage:\n{message}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_SENDER

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_SENDER, msg.as_string())

# CONTACT US PAGE #

st.set_page_config(page_title="Contact Us", layout="centered")
st.title("Contact Us")
st.write("We're here to help! Feel free to leave your message below.")

# --- Contact Form ---
with st.form("contact_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")
    submitted = st.form_submit_button("Send Message")
    
    if submitted:
        # Insert into Supabase
        supabase.table("contact_messages").insert({
            "name": name,
            "email": email,
            "message": message
        }).execute()

        # Send Email
        send_email(name, email, message)

        st.success("Your message has been sent! We will get back to you soon.")
    else:
        st.warning("Please fill in all required fields.")

import streamlit as st
import subprocess
import sys

st.title("Face Detection System")

if st.button("Open Face Detection"):
    subprocess.Popen([sys.executable, "face_detection.py"])
    st.success("Face Detection System Started")
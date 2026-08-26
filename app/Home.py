import streamlit as st

from ui import render_mockup


st.set_page_config(page_title="구단 선택", page_icon="⚾", layout="wide")
render_mockup("home")

import streamlit as st

from ui import render_mockup


st.set_page_config(page_title="구단 상황실", page_icon="⚾", layout="wide")
render_mockup("war")

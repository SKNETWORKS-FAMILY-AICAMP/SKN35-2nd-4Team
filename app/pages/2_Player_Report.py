import streamlit as st

from ui import render_mockup


st.set_page_config(page_title="선수 리포트", page_icon="⚾", layout="wide")
render_mockup("report")

import streamlit as st

st.set_page_config(page_title="LawLens", page_icon="⚖️", layout="wide")
st.title("⚖️ LawLens")
st.caption("Paragraph-cited summaries of Indian court judgments. A decision-support prototype: "
           "outputs are leads for a human to check against the source, never legal advice.")

from lawlens.ui import summariser_tab

summariser_tab.render()

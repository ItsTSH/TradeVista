import streamlit as st

# Page Setup
screener = st.Page(
    page = "views/screener.py",
    title = "TradeVista | Stock Screener",
    icon = "📊",
    default = True,
)

prediction = st.Page(
    page = "views/predictions.py",
    title = "TradeVista | Stock Prediction",
    icon = "📈",
)

# Navigation Setup
pg = st.navigation(pages = [screener, prediction])

# sidebar content
st.sidebar.text("TradeVista ©️ 2025")

# Run Navigation
pg.run()
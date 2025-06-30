import streamlit as st

# Page Setup
screener = st.Page(
    page = "views/screener.py",
    title = "Stock Screener",
    icon = "📊",
    default = True,
)

analysis = st.Page(
    page = "views/analysis.py",
    title = "Stock Analysis",
    icon = "🧠"
)
prediction = st.Page(
    page = "views/predictions.py",
    title = "Stock Prediction",
    icon = "📈",
)

# Navigation Setup
pg = st.navigation(pages = [screener, analysis, prediction])

# sidebar content
st.sidebar.text("TradeVista ©️ 2025")

# Run Navigation
pg.run()
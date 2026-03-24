import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: white;
        color: grey;
    }
    
    /* Main page background */
    [data-testid="stAppViewContainer"] {
        background-color: #ff4b4b;  /* blue */
        color: white;
    }

    /* Header text color */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Optional: buttons */
    div.stButton > button:first-child {
        background-color: white;
        color: black;
        border-radius: 10px;
    }
    </style>
    """,  
    unsafe_allow_html=True
)
    
# ---------------------------
# App title
# ---------------------------
st.title("Portfolio Optimizer")

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Asset Data")

r_1 = st.sidebar.number_input("Asset 1 Expected Return (%)", value=5.0) / 100
sd_1 = st.sidebar.number_input("Asset 1 Standard Deviation (%)", value=9.0) / 100

r_2 = st.sidebar.number_input("Asset 2 Expected Return (%)", value=12.0) / 100
sd_2 = st.sidebar.number_input("Asset 2 Standard Deviation (%)", value=20.0) / 100

rho = st.sidebar.number_input("Correlation", min_value=-1.0, max_value=1.0, value=0.2)

r_f = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100

st.sidebar.header("Your Preferences")
gamma = st.sidebar.slider("Risk Aversion (gamma)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)

# ---------------------------
# Functions
# ---------------------------
def portfolio_ret(w1, r1, r2):
    """Portfolio expected return for weight w1 in asset 1"""
    return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
    """Portfolio standard deviation for weight w1 in asset 1"""
    return np.sqrt(w1**2 * sd1**2 + (1 - w1)**2 * sd2**2 + 2 * rho * w1 * (1 - w1) * sd1 * sd2)

# ---------------------------
# Tangency portfolio (Sharpe-maximizing)
# ---------------------------
weights = np.linspace(0, 1, 1000)
sharpe_ratios = []

for w in weights:
    ret = portfolio_ret(w, r_1, r_2)
    sd = portfolio_sd(w, sd_1, sd_2, rho)
    sharpe = (ret - r_f) / sd if sd > 0 else -np.inf
    sharpe_ratios.append(sharpe)

max_idx = np.argmax(sharpe_ratios)
w1_tangency = weights[max_idx]
w2_tangency = 1 - w1_tangency

ret_tangency = portfolio_ret(w1_tangency, r_1, r_2)
sd_tangency = portfolio_sd(w1_tangency, sd_1, sd_2, rho)

# ---------------------------
# Optimal portfolio weights
# ---------------------------
w_tangency_optimal = (ret_tangency - r_f) / (gamma * sd_tangency**2) if sd_tangency > 0 else 0

# Compute raw weights
w1_optimal = w_tangency_optimal * w1_tangency
w2_optimal = w_tangency_optimal * w2_tangency
w_rf_optimal = 1 - (w1_optimal + w2_optimal)

# ---------------------------
# Apply weight constraints: 0–100%
# ---------------------------
w1_optimal = np.clip(w1_optimal, 0, 1)
w2_optimal = np.clip(w2_optimal, 0, 1)
w_rf_optimal = np.clip(w_rf_optimal, 0, 1)

# Normalize so total = 100%
total = w1_optimal + w2_optimal + w_rf_optimal
w1_optimal /= total
w2_optimal /= total
w_rf_optimal /= total

# Recompute portfolio return & risk
ret_optimal = w1_optimal * r_1 + w2_optimal * r_2 + w_rf_optimal * r_f
sd_optimal = np.sqrt(
    (w1_optimal * sd_1)**2 + (w2_optimal * sd_2)**2 + 2 * w1_optimal * w2_optimal * rho * sd_1 * sd_2
)

# ---------------------------
# Display results
# ---------------------------
tab1, tab2 = st.tabs(["📊 Results", "📈 Graph"])

with tab1:
    st.header("Your Optimal Portfolio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk-Free Asset", f"{w_rf_optimal*100:.2f}%")
    col2.metric("Asset 1", f"{w1_optimal*100:.2f}%")
    col3.metric("Asset 2", f"{w2_optimal*100:.2f}%")

    st.write("")
    col1, col2 = st.columns(2)
    col1.metric("Expected Return", f"{ret_optimal*100:.2f}%")
    col2.metric("Risk (Std Dev)", f"{sd_optimal*100:.2f}%")

# ---------------------------
# Efficient frontier & graph
# ---------------------------
with tab2:
    st.header("Portfolio Visualization")

    weights_plot = np.linspace(0, 1, 200)
    returns_frontier = [portfolio_ret(w, r_1, r_2) for w in weights_plot]
    sds_frontier = [portfolio_sd(w, sd_1, sd_2, rho) for w in weights_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sds_frontier, returns_frontier, 'b-', linewidth=2, label='Efficient Frontier')

    # Capital Market Line
    sd_max = max(sds_frontier) * 1.2
    sd_cml = np.linspace(0, sd_max, 100)
    ret_cml = r_f + (ret_tangency - r_f)/sd_tangency * sd_cml if sd_tangency > 0 else r_f*np.ones_like(sd_cml)
    ax.plot(sd_cml, ret_cml, 'g--', linewidth=2, label='Capital Market Line')

    # Tangency portfolio
    ax.scatter(sd_tangency, ret_tangency, color='red', s=200, marker='*', label='Tangency Portfolio')

    # Optimal portfolio
    ax.scatter(sd_optimal, ret_optimal, color='orange', s=200, marker='D', label='Your Optimal Portfolio')

    # Risk-free asset
    ax.scatter(0, r_f, color='green', s=150, marker='s', label='Risk-Free Asset')

    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Portfolio Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

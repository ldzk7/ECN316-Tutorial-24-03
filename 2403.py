import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Page styling
# ---------------------------
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: grey;
        color: black;
    }

    /* Main page background */
    .stApp {
        background-color: tan !important;
        color: black;
    }

    /* Header text color */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Optional buttons */
    div.stButton > button:first-child {
        background-color: blue;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# App title
# ---------------------------
st.title("Sustainable Finance Portfolio Optimisation App")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Portfolio Inputs")

r_h = st.sidebar.number_input("Asset 1 Expected Return (%)", value=5.0) / 100
sd_h = st.sidebar.number_input("Asset 1 Std Dev (%)", value=9.0) / 100

r_f = st.sidebar.number_input("Asset 2 Expected Return (%)", value=12.0) / 100
sd_f = st.sidebar.number_input("Asset 2 Std Dev (%)", value=20.0) / 100

rho_hf = st.sidebar.slider("Correlation between Asset 1 & 2", -1.0, 1.0, -0.2, 0.01)
r_free = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0) / 100
gamma = st.sidebar.slider("Risk Aversion (γ)", 0.1, 10.0, 5.0, 0.1)

# ESG inputs
esg_h = st.sidebar.slider("Asset 1 ESG Score", 0, 100, 80)
esg_f = st.sidebar.slider("Asset 2 ESG Score", 0, 100, 55)
lambda_esg = st.sidebar.slider("ESG Preference (λ)", 0.0, 5.0, 2.0, 0.1)

# ---------------------------
# Functions
# ---------------------------
def portfolio_ret(w1, r1, r2):
    return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
    return np.sqrt(w1**2 * sd1**2 + (1 - w1)**2 * sd2**2 + 2 * rho * w1 * (1 - w1) * sd1 * sd2)

def portfolio_esg(w1, esg1, esg2):
    return w1 * esg1 + (1 - w1) * esg2

# ---------------------------
# Calculate Optimal Portfolio
# ---------------------------
weights = np.linspace(0, 1, 1000)
utility_scores = []

for w in weights:
    ret = portfolio_ret(w, r_h, r_f)
    sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
    esg = portfolio_esg(w, esg_h, esg_f)
    utility = ret - 0.5 * gamma * sd**2 + lambda_esg * (esg / 100)
    utility_scores.append(utility)

max_idx = np.argmax(utility_scores)
w1_opt = weights[max_idx]
w2_opt = 1 - w1_opt

# Portfolio characteristics
ret_opt = portfolio_ret(w1_opt, r_h, r_f)
sd_opt = portfolio_sd(w1_opt, sd_h, sd_f, rho_hf)
esg_opt = portfolio_esg(w1_opt, esg_h, esg_f)
sharpe_opt = (ret_opt - r_free) / sd_opt if sd_opt > 0 else 0

# Tangency weight scaling for risk-free inclusion
w_tangency_opt = (ret_opt - r_free) / (gamma * sd_opt**2) if sd_opt > 0 else 0
w1_optimal = w_tangency_opt * w1_opt
w2_optimal = w_tangency_opt * w2_opt
w_rf_optimal = 1 - (w1_optimal + w2_optimal)

# Clip weights to [0,1] and normalize
w1_optimal = np.clip(w1_optimal, 0, 1)
w2_optimal = np.clip(w2_optimal, 0, 1)
w_rf_optimal = np.clip(w_rf_optimal, 0, 1)
total = w1_optimal + w2_optimal + w_rf_optimal
w1_optimal /= total
w2_optimal /= total
w_rf_optimal /= total

ret_optimal = w1_optimal * r_h + w2_optimal * r_f + w_rf_optimal * r_free
sd_optimal = np.sqrt(
    (w1_optimal*sd_h)**2 + (w2_optimal*sd_f)**2 + 2*w1_optimal*w2_optimal*rho_hf*sd_h*sd_f
)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📊 ESG Portfolio", "📈 Efficient Frontier"])

# ---------------------------
# ESG Portfolio Tab
# ---------------------------
with tab1:
    st.header("Optimal Portfolio with ESG Preference")
    st.subheader("Portfolio Weights")
    st.write(f"Risk-Free Asset: {w_rf_optimal*100:.2f}%")
    st.write(f"Asset 1: {w1_optimal*100:.2f}%")
    st.write(f"Asset 2: {w2_optimal*100:.2f}%")
    st.write(f"Expected Return: {ret_optimal*100:.2f}%")
    st.write(f"Portfolio Risk (Std Dev): {sd_optimal*100:.2f}%")
    st.write(f"Sharpe Ratio: {sharpe_opt:.4f}")
    st.write(f"Portfolio ESG Score: {esg_opt:.2f}")

    # ESG Opportunity Set Plot
    weights_plot = np.linspace(0,1,200)
    esg_frontier = [portfolio_esg(w, esg_h, esg_f) for w in weights_plot]
    sharpe_frontier = [(portfolio_ret(w, r_h, r_f)-r_free)/portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(esg_frontier, sharpe_frontier, 'b-', linewidth=2, label='ESG Opportunity Set')
    ax.scatter(esg_opt, sharpe_opt, color='red', s=100, marker='*', label='Optimal ESG Portfolio')
    ax.set_xlabel('Portfolio ESG Score')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('ESG Portfolio Optimisation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ---------------------------
# Efficient Frontier Tab
# ---------------------------
with tab2:
    st.header("Efficient Frontier & Tangency Portfolio")

    weights_plot = np.linspace(0, 1, 200)
    returns_frontier = [portfolio_ret(w, r_h, r_f) for w in weights_plot]
    sds_frontier = [portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sds_frontier, returns_frontier, 'b-', linewidth=2, label='Efficient Frontier')

    # Capital Market Line
    sd_max = max(sds_frontier)*1.2
    sd_cml = np.linspace(0, sd_max, 100)
    ret_cml = r_free + (ret_opt - r_free)/sd_opt * sd_cml if sd_opt>0 else r_free*np.ones_like(sd_cml)
    ax.plot(sd_cml, ret_cml, 'g--', linewidth=2, label='Capital Market Line')

    # Tangency Portfolio (red star)
    ax.scatter(sd_opt, ret_opt, color='red', s=200, marker='*', label='Tangency Portfolio')

    # Optimal Portfolio (orange diamond)
    ax.scatter(sd_optimal, ret_optimal, color='orange', s=200, marker='D', label='Optimal Portfolio')

    # Risk-Free Asset (green square)
    ax.scatter(0, r_free, color='green', s=150, marker='s', label='Risk-Free Asset')

    ax.set_xlabel('Risk (Std Dev)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: grey;
        color: black;
    }

    /* Full Main page background (full) */
    .stApp {  /* main content container */
        background-color: tan !important;
        color: black;
    }

    /* Body background */
    .css-1d391kg {  /* outermost container */
        background-color: blue !important;
    }

    /* Header text color */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Optional: buttons */
    div.stButton > button:first-child {
        background-color: blue;
        color: black;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.title("Sustainable Finance Portfolio Optimisation App")

# ------------------------------
# Inputs from the user
# ------------------------------
st.subheader("Enter Portfolio Inputs")

r_h = st.number_input("Asset 1 Expected Return (%) [e.g., 5]:", value=5.0) / 100
sd_h = st.number_input("Asset 1 Standard Deviation (%) [e.g., 9]:", value=9.0) / 100

r_f = st.number_input("Asset 2 Expected Return (%) [e.g., 12]:", value=12.0) / 100
sd_f = st.number_input("Asset 2 Standard Deviation (%) [e.g., 20]:", value=20.0) / 100

rho_hf = st.slider("Correlation between Asset 1 and 2:", min_value=-1.0, max_value=1.0, value=-0.2, step=0.01)

r_free = st.number_input("Risk-Free Rate (%) [e.g., 2]:", value=2.0) / 100

gamma = st.number_input("Risk Aversion (γ) [e.g., 5]:", value=5.0, min_value=0.1)

esg_h = st.number_input("Asset 1 ESG Score [e.g., 80]:", value=80.0, min_value=0.0)
esg_f = st.number_input("Asset 2 ESG Score [e.g., 55]:", value=55.0, min_value=0.0)

lambda_esg = st.number_input("ESG Preference (λ) [e.g., 2]:", value=2.0, min_value=0.0)

# ------------------------------
# Functions
# ------------------------------
def portfolio_ret(w1, r1, r2):
    return w1 * r1 + (1 - w1) * r2

def portfolio_sd(w1, sd1, sd2, rho):
    return np.sqrt(w1*2 * sd12 + (1 - w1)2 * sd2*2 + 2 * rho * w1 * (1 - w1) * sd1 * sd2)

def portfolio_esg(w1, esg1, esg2):
    return w1 * esg1 + (1 - w1) * esg2

# ------------------------------
# ESG Portfolio Selection
# ------------------------------
if st.button("Calculate Optimal Portfolio"):

    weights = np.linspace(0, 1, 1000)

    sharpe_ratios = []
    utility_scores = []
    esg_scores = []

    for w in weights:
        ret = portfolio_ret(w, r_h, r_f)
        sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
        esg = portfolio_esg(w, esg_h, esg_f)

        esg_scores.append(esg)

        if sd > 0:
            sharpe = (ret - r_free) / sd
            sharpe_ratios.append(sharpe)
        else:
            sharpe_ratios.append(-np.inf)

        utility = ret - 0.5 * gamma * sd**2 + lambda_esg * (esg / 100)
        utility_scores.append(utility)

    max_idx = np.argmax(utility_scores)
    w1_esg = weights[max_idx]
    w2_esg = 1 - w1_esg

    ret_esg = portfolio_ret(w1_esg, r_h, r_f)
    sd_esg = portfolio_sd(w1_esg, sd_h, sd_f, rho_hf)
    esg_portfolio = portfolio_esg(w1_esg, esg_h, esg_f)

    if sd_esg > 0:
        sharpe_esg = (ret_esg - r_free) / sd_esg
    else:
        sharpe_esg = 0

    # ------------------------------
    # Optimal Portfolio
    # ------------------------------
    if sd_esg > 0:
        w_esg_optimal = (ret_esg - r_free) / (gamma * sd_esg**2)
    else:
        w_esg_optimal = 0

    w1_optimal = w_esg_optimal * w1_esg
    w2_optimal = w_esg_optimal * w2_esg
    w_rf_optimal = 1 - w_esg_optimal

    ret_optimal = r_free + w_esg_optimal * (ret_esg - r_free)
    sd_optimal = abs(w_esg_optimal) * sd_esg

    # ------------------------------
    # Display results
    # ------------------------------
    st.subheader("Optimal Portfolio Weights")
    st.write(f"*Risk-Free Asset:* {w_rf_optimal*100:.2f}%")
    st.write(f"*Asset 1:* {w1_optimal*100:.2f}%")
    st.write(f"*Asset 2:* {w2_optimal*100:.2f}%")
    st.write(f"*Expected Return:* {ret_optimal*100:.2f}%")
    st.write(f"*Portfolio Risk (Std Dev):* {sd_optimal*100:.2f}%")
    st.write(f"*Sharpe Ratio:* {sharpe_esg:.4f}")
    st.write(f"*Portfolio ESG Score:* {esg_portfolio:.2f}")

    # ------------------------------
    # Plot ESG Opportunity Set
    # ------------------------------
    weights_plot = np.linspace(0, 1, 200)
    esg_frontier = [portfolio_esg(w, esg_h, esg_f) for w in weights_plot]
    sharpe_frontier = []

    for w in weights_plot:
        ret = portfolio_ret(w, r_h, r_f)
        sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
        if sd > 0:
            sharpe_frontier.append((ret - r_free) / sd)
        else:
            sharpe_frontier.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ESG opportunity set
    ax.plot(esg_frontier, sharpe_frontier, 'b-', linewidth=2, label='ESG Opportunity Set')

    # ESG optimal portfolio
    ax.scatter(esg_portfolio, sharpe_esg, color='red', s=100, marker='*', label='Optimal ESG Portfolio')

    ax.set_xlabel('Portfolio ESG Score')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('ESG Portfolio Optimisation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
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

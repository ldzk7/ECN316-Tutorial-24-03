import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Custom page colors
# ---------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: grey;
        color: black;
    }
    .stApp {
        background-color: tan !important;
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    div.stButton > button:first-child {
        background-color: blue;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sustainable Finance Portfolio Optimisation App")

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Asset Data")
r_h = st.sidebar.number_input("Asset 1 Expected Return (%)", value=5.0)/100
sd_h = st.sidebar.number_input("Asset 1 Std Dev (%)", value=9.0)/100
r_f = st.sidebar.number_input("Asset 2 Expected Return (%)", value=12.0)/100
sd_f = st.sidebar.number_input("Asset 2 Std Dev (%)", value=20.0)/100
rho_hf = st.sidebar.slider("Correlation (Asset1 vs Asset2)", -1.0, 1.0, 0.2, 0.01)
r_free = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0)/100

st.sidebar.header("Preferences")
gamma = st.sidebar.slider("Risk Aversion (γ)", 0.1, 10.0, 3.0, 0.1)
esg_h = st.sidebar.number_input("Asset 1 ESG Score", 0.0, 100.0, 80.0)
esg_f = st.sidebar.number_input("Asset 2 ESG Score", 0.0, 100.0, 55.0)
lambda_esg = st.sidebar.slider("ESG Preference (λ)", 0.0, 5.0, 2.0, 0.1)

# ---------------------------
# Portfolio functions
# ---------------------------
def portfolio_ret(w1, r1, r2):
    return w1*r1 + (1-w1)*r2

def portfolio_sd(w1, sd1, sd2, rho):
    return np.sqrt(w1**2*sd1**2 + (1-w1)**2*sd2**2 + 2*rho*w1*(1-w1)*sd1*sd2)

def portfolio_esg(w1, esg1, esg2):
    return w1*esg1 + (1-w1)*esg2

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["📊 ESG Portfolio", "📈 Efficient Frontier"])

# ---------------------------
# Tab 1: ESG Portfolio
# ---------------------------
with tab1:
    st.subheader("ESG Optimized Portfolio")
    
    if st.button("Calculate ESG Portfolio"):
        weights = np.linspace(0,1,1000)
        utility_scores = []

        for w in weights:
            ret = portfolio_ret(w, r_h, r_f)
            sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
            esg = portfolio_esg(w, esg_h, esg_f)
            utility = ret - 0.5*gamma*sd**2 + lambda_esg*(esg/100)
            utility_scores.append(utility)

        max_idx = np.argmax(utility_scores)
        w1_esg = weights[max_idx]
        w2_esg = 1 - w1_esg

        ret_esg = portfolio_ret(w1_esg, r_h, r_f)
        sd_esg = portfolio_sd(w1_esg, sd_h, sd_f, rho_hf)
        esg_portfolio = portfolio_esg(w1_esg, esg_h, esg_f)
        sharpe_esg = (ret_esg - r_free)/sd_esg if sd_esg>0 else 0

        # Portfolio weights display
        st.write(f"*Asset 1:* {w1_esg*100:.2f}%")
        st.write(f"*Asset 2:* {w2_esg*100:.2f}%")
        st.write(f"*Expected Return:* {ret_esg*100:.2f}%")
        st.write(f"*Risk (Std Dev):* {sd_esg*100:.2f}%")
        st.write(f"*Sharpe Ratio:* {sharpe_esg:.4f}")
        st.write(f"*Portfolio ESG Score:* {esg_portfolio:.2f}")

        # ESG Opportunity Set plot
        weights_plot = np.linspace(0,1,200)
        esg_frontier = [portfolio_esg(w, esg_h, esg_f) for w in weights_plot]
        sharpe_frontier = [(portfolio_ret(w, r_h, r_f)-r_free)/portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(esg_frontier, sharpe_frontier, 'b-', label="ESG Opportunity Set")
        ax.scatter(esg_portfolio, sharpe_esg, color='red', s=100, marker='*', label="Optimal ESG Portfolio")
        ax.set_xlabel("Portfolio ESG Score")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ---------------------------
# Tab 2: Efficient Frontier
# ---------------------------
with tab2:
    st.subheader("Efficient Frontier & Tangency Portfolio")

    weights_plot = np.linspace(0,1,200)
    returns_frontier = [portfolio_ret(w, r_h, r_f) for w in weights_plot]
    sds_frontier = [portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

    # Tangency portfolio
    sharpe_ratios = [(portfolio_ret(w, r_h, r_f)-r_free)/portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]
    max_idx = np.argmax(sharpe_ratios)
    w_tangency = weights_plot[max_idx]
    ret_tangency = portfolio_ret(w_tangency, r_h, r_f)
    sd_tangency = portfolio_sd(w_tangency, sd_h, sd_f, rho_hf)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sds_frontier, returns_frontier, 'b-', label='Efficient Frontier')
    ax.scatter(sd_tangency, ret_tangency, color='red', s=200, marker='*', label="Tangency Portfolio")
    ax.set_xlabel("Risk (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

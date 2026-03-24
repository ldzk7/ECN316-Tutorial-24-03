# ---------------------------
# Create tabs
# ---------------------------
tab1, tab2 = st.tabs(["📊 ESG Portfolio", "📈 Efficient Frontier"])

# ---------------------------
# ESG Portfolio (Tab 1)
# ---------------------------
with tab1:
    st.subheader("ESG Optimized Portfolio")
    
    if st.button("Calculate ESG Optimal Portfolio"):
        weights = np.linspace(0, 1, 1000)
        utility_scores = []

        for w in weights:
            ret = portfolio_ret(w, r_h, r_f)
            sd = portfolio_sd(w, sd_h, sd_f, rho_hf)
            esg = portfolio_esg(w, esg_h, esg_f)
            utility = ret - 0.5 * gamma * sd**2 + lambda_esg * (esg / 100)
            utility_scores.append(utility)

        max_idx = np.argmax(utility_scores)
        w1_esg = weights[max_idx]
        w2_esg = 1 - w1_esg
        ret_esg = portfolio_ret(w1_esg, r_h, r_f)
        sd_esg = portfolio_sd(w1_esg, sd_h, sd_f, rho_hf)
        esg_portfolio = portfolio_esg(w1_esg, esg_h, esg_f)
        sharpe_esg = (ret_esg - r_free)/sd_esg if sd_esg > 0 else 0

        st.write(f"*Portfolio Weights:* Asset1={w1_esg*100:.2f}%, Asset2={w2_esg*100:.2f}%")
        st.write(f"*Portfolio ESG Score:* {esg_portfolio:.2f}")
        st.write(f"*Sharpe Ratio:* {sharpe_esg:.4f}")

        # ESG opportunity set plot
        weights_plot = np.linspace(0, 1, 200)
        esg_frontier = [portfolio_esg(w, esg_h, esg_f) for w in weights_plot]
        sharpe_frontier = [(portfolio_ret(w, r_h, r_f)-r_free)/portfolio_sd(w, sd_h, sd_f, rho_hf)
                           for w in weights_plot]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(esg_frontier, sharpe_frontier, 'b-', label="ESG Opportunity Set")
        ax.scatter(esg_portfolio, sharpe_esg, color='red', s=100, marker='*', label="Optimal ESG Portfolio")
        ax.set_xlabel("Portfolio ESG Score")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ---------------------------
# Efficient Frontier / Tangency Portfolio (Tab 2)
# ---------------------------
with tab2:
    st.subheader("Efficient Frontier & Tangency Portfolio")
    
    weights_plot = np.linspace(0, 1, 200)
    returns_frontier = [portfolio_ret(w, r_h, r_f) for w in weights_plot]
    sds_frontier = [portfolio_sd(w, sd_h, sd_f, rho_hf) for w in weights_plot]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sds_frontier, returns_frontier, 'b-', label='Efficient Frontier')

    # Tangency portfolio
    sharpe_ratios = [(portfolio_ret(w, r_h, r_f)-r_free)/portfolio_sd(w, sd_h, sd_f, rho_hf)
                     for w in weights_plot]
    max_idx = np.argmax(sharpe_ratios)
    w_tangency = weights_plot[max_idx]
    ret_tangency = portfolio_ret(w_tangency, r_h, r_f)
    sd_tangency = portfolio_sd(w_tangency, sd_h, sd_f, rho_hf)

    ax.scatter(sd_tangency, ret_tangency, color='red', s=200, marker='*', label="Tangency Portfolio")
    ax.set_xlabel("Risk (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

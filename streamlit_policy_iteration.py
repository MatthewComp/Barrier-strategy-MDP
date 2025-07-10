import streamlit as st
from gamma_loss import policy_iteration, recover_barrier
import pandas as pd

st.title("Policy Iteration for Optimal Reporting Strategy")

tab1, tab2 = st.tabs(["Model Inputs and Output",
                      "Problem Description and Method"])

with tab1:
    st.header("Problem Solver")
    risk_av = st.number_input("Risk Aversion (γ)", min_value = 0.01, max_value = 8.0, step = 0.1, value = 1.0)
    delta = st.slider("Discount factor (δ)", min_value = 0.01, max_value = 0.99, value = 0.95)
    p = st.slider("Probability of no loss (p)", min_value = 0.01, max_value = 0.99, value = 0.8)
    lam = st.number_input("Rate parameter (λ)", min_value = 0.01, step = 0.1, value = 1.0)
    shape = st.number_input("Shape parameter (β)", min_value = 0.01, step = 0.1, value = 1.0)

    num_classes = st.number_input("Number of rate classes", min_value=1, max_value=20, value=3, step=1)
    cols = st.columns(min(num_classes, 4))
    premiums = []
    previous = 0.0
    for i in range(num_classes):
        col = cols[i % len(cols)]  # cycle through columns
        with col:
            value = st.number_input(f"Premium {i+1}",
                                    key=f"premium_{i}", 
                                    format="%.2f",
                                    min_value = previous,
                                    max_value = 50.0,
                                    value = float(i+1))
            premiums.append(value)
            previous = value

    # Inject CSS to make buttons full-width
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        font-size: 18px;
        padding: 0.75em;
        border-radius: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)
    # Now the button fills the container width
    run = st.button("🚀 Run Policy Iteration")

    if run:
        policy = policy_iteration(premiums, delta, lam,
                                shape, risk_av, p, 15)
        result = recover_barrier(policy, lam, shape, p)

        df = pd.DataFrame({
            "Rate Class": [f"Class {i+1}" for i in range(len(result))],
            "Value": [round(v, 3) for v in result]
            })

        st.table(df)

with tab2:
    st.header("Problem Description and Method")
    st.write("This app uses policy iteration on a MDP-based insurance model to find an optimal reporting strategy." \
    "We have a discrete number of rate classes, and when the insured makes a claim they move to the next rate class, otherwise they move to the previous rate class." \
    "The output gives a barrier value for each rate class where the insured should claim if the loss is larger than the barrier.")

    st.markdown(r"""This is done by finding the policy matrix $\pi$ that is the maximiser of:""")
    st.latex(r"""
    \max_\pi E \biggl[ \, \sum_{n=0}^\infty \delta^n g(X_t, \pi(i)) \mid X_0 = i \, \biggr] \quad \text{for all rate classes }i,
             """)
    st.write("where:")
    st.markdown(r"""
    - $g(X_t, \pi(i))$ is the immediate reward for rate class $X_t$ and following the policy $\pi$
    - $\delta$ is the discount factor
    - $c_i$ is the premium for rate class $i$.
    """)

    st.markdown(r"""Now assume the probability the insured makes no loss is $p$,
                and the loss follows a Gamma($\lambda, \beta$) distribution when the loss is positive.
                For simplicity, assume $\pi^\leftarrow_i$ and $\pi^\rightarrow_i$
                is the probability the insured doesn't claim in class $i$ and claims in class $i$ respectively.
                Then the policy matrix $\pi$ comprises of these variables and all other values are 0.
                When $\gamma \ne \lambda$, the immediate reward function $g$ has value:""")
    st.latex(r"""
        g(i, \pi(i)) = -e^{\gamma c_i} \biggl( \pi_i^\rightarrow + p +
             \frac{1-p}{\Gamma(\beta)} \biggl(\frac{\lambda}{\lambda - \gamma}\biggr)^\beta 
             \tilde{\gamma}(\beta, (\lambda - \gamma) F^{-1}(\pi_i^\leftarrow)) \biggr),
""")
    st.markdown(r"""Where $F^{-1}$ is the quantile function of the loss, 
                and $\tilde{\gamma}$ is the lower incomplete gamma function.
                If $\gamma = \lambda$, then $g$ has value:""")
    st.latex(r"""
        g(i, \pi(i)) = -e^{\gamma c_i} \biggl( \pi_i^\rightarrow + p + \frac{1-p}{\Gamma(\beta+1)} \bigl(\gamma F^{-1}(\pi_i^\leftarrow)\bigr)^\beta  \biggr).
""")
    
    st.markdown(r"""When the policy matrix $\pi$ is found, we can recover the barrier strategy ($B$) using the following relation:""")
    st.latex(r"""
        B_i = F^{-1}(\pi_i^\leftarrow).
""")
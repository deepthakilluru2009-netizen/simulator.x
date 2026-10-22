import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAR-T Simulator", page_icon="🧬")

st.title("🧬 CAR-T Cell Simulator")
st.write("Explore how tumour obstruction and attack probability affect simulated CAR-T outcomes.")

st.warning("Educational simulation only. This does not predict real patient outcomes.")

obstruction = st.sidebar.slider("Tumour obstruction", 0.0, 0.9, 0.3, 0.1)
attack_probability = st.sidebar.slider("Attack success probability", 0.1, 1.0, 0.7, 0.1)
trials = st.sidebar.slider("Number of simulated cells", 100, 5000, 1000, 100)

if st.button("Run simulation"):
    rng = np.random.default_rng()

    reached = rng.random(trials) < (1 - obstruction)
    attacked = (rng.random(trials) < attack_probability) & reached

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Reached target", f"{reached.mean() * 100:.1f}%")
    col2.metric(
        "Attack success among cells reaching",
        f"{attacked.sum() / max(reached.sum(), 1) * 100:.1f}%"
    )
    col3.metric("Overall successful attacks", f"{attacked.mean() * 100:.1f}%")

    fig, ax = plt.subplots()
    ax.bar(
        ["Did not reach", "Reached target", "Successful attack"],
        [trials - reached.sum(), reached.sum(), attacked.sum()]
    )
    ax.set_ylabel("Number of cells")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.caption("Simplified randomized model; not validated clinical data.")

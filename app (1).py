import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="CAR-T Simulator",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 CAR-T Therapy Simulator")
st.write(
    "Explore how tumour access, antigen recognition, exhaustion, "
    "and CAR-T cell persistence can influence simulated tumour killing."
)

st.info(
    "Educational model only. These simulations are illustrative and "
    "are not calibrated to clinical data or suitable for predicting "
    "individual patient outcomes."
)


def simulate_scenario(
    obstruction,
    antigen,
    exhaustion,
    speed,
    distance,
    trials,
    steps,
    seed
):
    rng = np.random.default_rng(seed)

    all_persistence = []
    all_function = []
    all_tumour = []
    killed_counts = []

    for _ in range(trials):
        persistence = 1.0
        function = 1.0
        tumour = 100.0

        persistence_history = [persistence]
        function_history = [function]
        tumour_history = [tumour]

        for step in range(steps):
            # Movement/access: faster movement helps; obstruction hinders.
            access = speed / (speed + distance + 0.5)
            access *= (1.0 - obstruction)
            access = np.clip(access, 0.02, 0.95)

            # Cell persistence can decline over time.
            persistence *= 1.0 - (0.008 + 0.018 * exhaustion)
            persistence = np.clip(persistence, 0.0, 1.0)

            # Function can decline due to exhaustion.
            function *= 1.0 - (0.012 + 0.035 * exhaustion)
            function = np.clip(function, 0.0, 1.0)

            # Chance of a successful tumour-cell attack this step.
            attack_chance = (
                0.45 * access
                * antigen
                * persistence
                * function
            )
            attack_chance = np.clip(attack_chance, 0.0, 0.95)

            # Expected fraction of remaining tumour cells killed.
            kill_fraction = rng.binomial(
                10, attack_chance
            ) / 10.0
            tumour *= 1.0 - min(kill_fraction * 0.25, 0.8)
            tumour = max(tumour, 0.0)

            persistence_history.append(persistence)
            function_history.append(function)
            tumour_history.append(tumour)

        all_persistence.append(persistence_history)
        all_function.append(function_history)
        all_tumour.append(tumour_history)
        killed_counts.append(100.0 - tumour)

    return {
        "persistence": np.mean(all_persistence, axis=0),
        "function": np.mean(all_function, axis=0),
        "tumour": np.mean(all_tumour, axis=0),
        "mean_killed": float(np.mean(killed_counts)),
    }


def controls(prefix, defaults):
    st.subheader(prefix)
    c1, c2 = st.columns(2)

    with c1:
        obstruction = st.slider(
            f"{prefix}: Tumour obstruction",
            0.0, 0.8, defaults["obstruction"], 0.05,
            key=f"{prefix}_obstruction"
        )
        antigen = st.slider(
            f"{prefix}: Antigen recognition",
            0.1, 1.0, defaults["antigen"], 0.05,
            key=f"{prefix}_antigen"
        )
        exhaustion = st.slider(
            f"{prefix}: Exhaustion pressure",
            0.0, 1.0, defaults["exhaustion"], 0.05,
            key=f"{prefix}_exhaustion"
        )

    with c2:
        speed = st.slider(
            f"{prefix}: Movement speed",
            0.2, 2.0, defaults["speed"], 0.1,
            key=f"{prefix}_speed"
        )
        distance = st.slider(
            f"{prefix}: Distance to tumour",
            1.0, 10.0, defaults["distance"], 0.5,
            key=f"{prefix}_distance"
        )

    return {
        "obstruction": obstruction,
        "antigen": antigen,
        "exhaustion": exhaustion,
        "speed": speed,
        "distance": distance,
    }


with st.sidebar:
    st.header("Simulation settings")
    trials = st.slider("Number of trials", 10, 200, 50, 10)
    steps = st.slider("Simulation time steps", 10, 60, 30, 5)
    seed = st.number_input(
        "Random seed", min_value=0, max_value=100000,
        value=42, step=1
    )

tab1, tab2 = st.tabs(["Single experiment", "Compare two experiments"])

with tab1:
    settings = controls(
        "Experiment A",
        {
            "obstruction": 0.25,
            "antigen": 0.8,
            "exhaustion": 0.3,
            "speed": 1.0,
            "distance": 4.0,
        }
    )

    if st.button("Run Experiment A", type="primary"):
        result = simulate_scenario(
            **settings,
            trials=trials,
            steps=steps,
            seed=int(seed)
        )
        timeline = pd.DataFrame({
            "Time step": np.arange(steps + 1),
            "CAR-T persistence": result["persistence"],
            "CAR-T function": result["function"],
            "Tumour remaining (%)": result["tumour"],
        })

        st.session_state["single_result"] = result
        st.session_state["single_timeline"] = timeline

    if "single_timeline" in st.session_state:
        result = st.session_state["single_result"]
        timeline = st.session_state["single_timeline"]

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Average tumour reduction",
            f"{result['mean_killed']:.1f}%"
        )
        m2.metric(
            "Final persistence",
            f"{result['persistence'][-1] * 100:.1f}%"
        )
        m3.metric(
            "Final function",
            f"{result['function'][-1] * 100:.1f}%"
        )

        st.subheader("Persistence versus function")
        pf = timeline.melt(
            id_vars="Time step",
            value_vars=["CAR-T persistence", "CAR-T function"],
            var_name="Measure",
            value_name="Relative level"
        )
        fig = px.line(
            pf, x="Time step", y="Relative level",
            color="Measure", range_y=[0, 1],
            title="Cells can remain present while function declines"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tumour remaining over time")
        fig2 = px.line(
            timeline, x="Time step", y="Tumour remaining (%)",
            title="Simulated tumour burden"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Persistence and function are simplified relative measures, "
            "not clinical measurements."
        )

with tab2:
    st.write(
        "Change the conditions in each experiment to compare their "
        "simulated outcomes."
    )

    left, right = st.columns(2)

    with left:
        a = controls(
            "Comparison A",
            {
                "obstruction": 0.25,
                "antigen": 0.8,
                "exhaustion": 0.3,
                "speed": 1.0,
                "distance": 4.0,
            }
        )

    with right:
        b = controls(
            "Comparison B",
            {
                "obstruction": 0.25,
                "antigen": 0.8,
                "exhaustion": 0.7,
                "speed": 1.0,
                "distance": 4.0,
            }
        )

    if st.button("Run comparison", type="primary"):
        ra = simulate_scenario(
            **a, trials=trials, steps=steps, seed=int(seed)
        )
        rb = simulate_scenario(
            **b, trials=trials, steps=steps, seed=int(seed) + 1
        )

        st.session_state["comparison"] = (ra, rb)

    if "comparison" in st.session_state:
        ra, rb = st.session_state["comparison"]

        comparison = pd.DataFrame({
            "Measure": [
                "Tumour reduction (%)",
                "Final persistence (%)",
                "Final function (%)",
            ],
            "Experiment A": [
                ra["mean_killed"],
                ra["persistence"][-1] * 100,
                ra["function"][-1] * 100,
            ],
            "Experiment B": [
                rb["mean_killed"],
                rb["persistence"][-1] * 100,
                rb["function"][-1] * 100,
            ],
        })

        st.subheader("Results")
        st.dataframe(
            comparison.round(1),
            use_container_width=True,
            hide_index=True
        )

        chart_data = comparison.melt(
            id_vars="Measure",
            var_name="Experiment",
            value_name="Value"
        )
        fig3 = px.bar(
            chart_data,
            x="Measure",
            y="Value",
            color="Experiment",
            barmode="group",
            title="Compare simulated outcomes"
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("How to interpret this model")
st.markdown(
    """
    - **Persistence:** the relative proportion of cells remaining in the model.
    - **Function:** a separate simplified measure of remaining cell activity.
    - **Tumour reduction:** the simulated percentage of initial tumour burden removed.
    - **Obstruction, distance and speed:** simplified representations of access
      to the tumour.
    - **Antigen recognition:** represents how readily CAR-T cells recognise
      their target in this model.
    - **Exhaustion:** applies pressure that reduces persistence and function.

    These relationships are illustrative assumptions, not established
    quantitative predictions of real CAR-T behaviour.
    """
)

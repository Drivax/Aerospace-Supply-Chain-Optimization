from __future__ import annotations

import streamlit as st

from src.data_generation import GenerationConfig, default_component_demand, generate_supply_chain_data
from src.delay_predictor import predict_delays, train_delay_model
from src.optimizer import generate_delay_scenarios, optimize_procurement


st.set_page_config(page_title="Aerospace Supply Chain Simulator", layout="wide")
st.title("Aerospace Supply Chain What-If Simulator")
st.caption("Change assumptions and re-optimize procurement instantly.")

if "base_df" not in st.session_state:
    # Cache generated data/model once to keep interactions responsive.
    df = generate_supply_chain_data(GenerationConfig(num_suppliers=25, num_components=8, seed=42))
    model = train_delay_model(df)
    df = df.copy()
    df["predicted_delay_days"] = predict_delays(model.pipeline, df)
    st.session_state.base_df = df
    st.session_state.residual_std = float(model.metrics.get("residual_std", 1.0))

base_df = st.session_state.base_df.copy()

col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a:
    demand_multiplier = st.slider("Demand multiplier", 0.6, 1.6, 1.0, 0.05)
with col_b:
    budget = st.slider("Budget", 300000, 5000000, 1800000, 50000)
with col_c:
    deadline = st.slider("Deadline (days)", 14, 40, 28, 1)
with col_d:
    risk_weight = st.slider("Risk weight", 0.0, 1.0, 0.2, 0.05)
with col_e:
    carbon_weight = st.slider("Carbon weight", 0.0, 1.0, 0.1, 0.05)

scenario_count = st.slider("Delay scenarios", 50, 100, 80, 5)
failed_suppliers = st.multiselect(
    "Simulate supplier failure (capacity = 0)",
    options=sorted(base_df["supplier_id"].unique()),
)

if st.button("Run Re-Optimization", type="primary"):
    work_df = base_df.copy()
    if failed_suppliers:
        # Simulate disruption by disabling selected suppliers.
        work_df.loc[work_df["supplier_id"].isin(failed_suppliers), "capacity"] = 0

    demand = default_component_demand(work_df)
    # Demand shock control for rapid stress testing.
    demand = {k: int(v * demand_multiplier) for k, v in demand.items()}

    scenarios = generate_delay_scenarios(
        predicted_delay_days=work_df["predicted_delay_days"].to_numpy(),
        residual_std=float(st.session_state.residual_std),
        n_scenarios=int(scenario_count),
        seed=42,
    )

    plan, summary = optimize_procurement(
        df=work_df,
        demand=demand,
        budget=float(budget),
        deadline_days=float(deadline),
        delay_penalty=4.0,
        delay_scenarios=scenarios,
        risk_weight=float(risk_weight),
        carbon_weight=float(carbon_weight),
        scenario_confidence=0.9,
        delay_variability_cap=3.0,
        solver_time_limit=10,
    )

    st.subheader("Optimization Summary")
    st.write(summary)

    st.subheader("Selected Orders")
    st.dataframe(plan, use_container_width=True)

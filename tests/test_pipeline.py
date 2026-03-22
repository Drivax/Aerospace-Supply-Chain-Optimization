from __future__ import annotations

from src.data_generation import GenerationConfig, default_component_demand, generate_supply_chain_data
from src.delay_predictor import predict_delays, train_delay_model
from src.optimizer import optimize_procurement


def test_end_to_end_pipeline_feasible_solution() -> None:
    df = generate_supply_chain_data(GenerationConfig(num_suppliers=12, seed=7))
    model = train_delay_model(df)

    df = df.copy()
    df["predicted_delay_days"] = predict_delays(model.pipeline, df)
    demand = default_component_demand(df)

    # A large budget and moderate deadline keep the test deterministic and feasible.
    plan, summary = optimize_procurement(
        df=df,
        demand=demand,
        budget=3_000_000,
        deadline_days=30,
        delay_penalty=3.0,
    )

    assert summary["status"] == "Optimal"
    assert not plan.empty
    assert (plan["arrival_days"] <= 30).all()

    delivered = plan.groupby("component")["quantity"].sum().to_dict()
    for component, required in demand.items():
        assert delivered.get(component, 0) >= required

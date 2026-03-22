from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.data_generation import GenerationConfig, default_component_demand, generate_supply_chain_data
from src.delay_predictor import predict_delays, train_delay_model
from src.graph_visualization import (
    plot_cost_comparison,
    plot_delivery_timeline,
    plot_optimized_network,
    plot_pareto_front,
)
from src.optimizer import generate_delay_scenarios, greedy_baseline, optimize_procurement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aerospace supply chain optimization")
    parser.add_argument("--suppliers", type=int, default=25, help="Number of suppliers")
    parser.add_argument("--components", type=int, default=8, help="Number of components (max 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="Budget cap (0 = auto estimate)",
    )
    parser.add_argument("--deadline", type=float, default=28.0, help="Max arrival days")
    parser.add_argument(
        "--delay-penalty",
        type=float,
        default=4.0,
        help="Cost multiplier applied to predicted delay",
    )
    parser.add_argument("--risk-weight", type=float, default=0.2, help="Weight of supply risk in objective (0-1)")
    parser.add_argument("--scenarios", type=int, default=80, help="Monte Carlo delay scenarios (50-100 recommended)")
    parser.add_argument(
        "--scenario-confidence",
        type=float,
        default=0.9,
        help="Quantile for robust delay constraints",
    )
    parser.add_argument(
        "--delay-variability-cap",
        type=float,
        default=3.0,
        help="Cap for aggregate tail-delay exposure",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=10,
        help="CBC solver time limit in seconds",
    )
    parser.add_argument(
        "--run-pareto",
        action="store_true",
        help="Generate Pareto front by sweeping risk-weight from 0 to 1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where outputs are stored",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Optional CSV path to use instead of generating data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path) if args.data_path else Path("data") / "supply_chain_dataset.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.data_path and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # Generate a reproducible synthetic dataset when no external CSV is provided.
        df = generate_supply_chain_data(
            GenerationConfig(
                num_suppliers=args.suppliers,
                num_components=args.components,
                seed=args.seed,
            )
        )
        df.to_csv(data_path, index=False)

    model_artifacts = train_delay_model(df)
    df = df.copy()
    df["predicted_delay_days"] = predict_delays(model_artifacts.pipeline, df)
    # Scenario matrix shape: (lanes, scenarios), used by robust optimization constraints.
    delay_scenarios = generate_delay_scenarios(
        predicted_delay_days=df["predicted_delay_days"].to_numpy(dtype=float),
        residual_std=float(model_artifacts.metrics.get("residual_std", 1.0)),
        n_scenarios=max(1, int(args.scenarios)),
        seed=args.seed,
    )

    demand = default_component_demand(df)

    budget = float(args.budget)
    if budget <= 0:
        # Estimate a feasible budget from per-component cheapest valid options.
        arrival = df["lead_time_days"] + df["predicted_delay_days"]
        valid = df[arrival <= args.deadline]
        lower_bound = 0.0
        for component, req in demand.items():
            options = valid[valid["component"] == component]
            if options.empty:
                continue
            lower_bound += float(options["unit_cost"].min()) * float(req)
        budget = max(lower_bound * 1.8, 300000.0)

    baseline_plan, baseline_cost = greedy_baseline(
        df=df,
        demand=demand,
        deadline_days=args.deadline,
        delay_penalty=args.delay_penalty,
    )

    start = perf_counter()
    optimized_plan, summary = optimize_procurement(
        df=df,
        demand=demand,
        budget=budget,
        deadline_days=args.deadline,
        delay_penalty=args.delay_penalty,
        delay_scenarios=delay_scenarios,
        risk_weight=args.risk_weight,
        scenario_confidence=args.scenario_confidence,
        delay_variability_cap=args.delay_variability_cap,
        solver_time_limit=args.time_limit,
    )
    solve_time_seconds = perf_counter() - start

    optimized_cost = summary["total_cost"]
    if baseline_cost > 0:
        cost_reduction = (baseline_cost - optimized_cost) / baseline_cost
    else:
        cost_reduction = 0.0

    summary.update(
        {
            "baseline_total_cost": float(baseline_cost),
            "optimized_total_cost": float(optimized_cost),
            "cost_reduction_ratio": float(cost_reduction),
            "solve_time_seconds": float(solve_time_seconds),
            "mae_delay_model": float(model_artifacts.metrics["mae"]),
            "rmse_delay_model": float(model_artifacts.metrics["rmse"]),
            "residual_std_delay_model": float(model_artifacts.metrics["residual_std"]),
        }
    )

    pareto_df = pd.DataFrame()
    if args.run_pareto:
        # Sweep risk weight to trace the cost-risk trade-off frontier.
        rows = []
        for w in np.linspace(0, 1, 11):
            _, front_summary = optimize_procurement(
                df=df,
                demand=demand,
                budget=budget,
                deadline_days=args.deadline,
                delay_penalty=args.delay_penalty,
                delay_scenarios=delay_scenarios,
                risk_weight=float(w),
                scenario_confidence=args.scenario_confidence,
                delay_variability_cap=args.delay_variability_cap,
                solver_time_limit=args.time_limit,
            )
            if front_summary["status"] != "Optimal":
                continue
            rows.append(
                {
                    "risk_weight": float(w),
                    "total_cost": float(front_summary["total_cost"]),
                    "risk_score": float(front_summary["risk_score"]),
                }
            )
        pareto_df = pd.DataFrame(rows)

    baseline_plan.to_csv(output_dir / "baseline_plan.csv", index=False)
    optimized_plan.to_csv(output_dir / "optimized_plan.csv", index=False)
    if not pareto_df.empty:
        pareto_df.to_csv(output_dir / "pareto_front.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_cost_comparison(baseline_cost, optimized_cost, output_dir / "cost_reduction.png")
    plot_optimized_network(optimized_plan, output_dir / "optimized_network.png")
    plot_delivery_timeline(optimized_plan, output_dir / "delivery_timeline.png")
    # HTML output enables interactive zoom/hover for trade-off inspection.
    plot_pareto_front(pareto_df, output_dir / "pareto_front.html")

    print("=== Pipeline Completed ===")
    print(f"Model MAE: {model_artifacts.metrics['mae']:.3f} days")
    print(f"Delay residual std: {model_artifacts.metrics['residual_std']:.3f} days")
    print(f"Optimization status: {summary['status']}")
    print(f"Budget used: {budget:,.2f}")
    print(f"Risk weight: {args.risk_weight:.2f}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Baseline cost: {baseline_cost:,.2f}")
    print(f"Optimized cost: {optimized_cost:,.2f}")
    print(f"Cost reduction: {100 * cost_reduction:.2f}%")
    print(f"On-time rate: {100 * summary['on_time_rate']:.2f}%")
    print(f"Solve time: {solve_time_seconds:.3f}s")
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()

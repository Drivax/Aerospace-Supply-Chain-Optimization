from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pulp


def greedy_baseline(
    df: pd.DataFrame,
    demand: Dict[str, int],
    deadline_days: float,
    delay_penalty: float,
) -> Tuple[pd.DataFrame, float]:
    """Simple baseline: fill each component from cheapest valid options first."""
    records = []
    total_cost = 0.0

    for component, required_qty in demand.items():
        remaining = int(required_qty)
        subset = df[df["component"] == component].copy()
        subset = subset[
            subset["lead_time_days"] + subset["predicted_delay_days"] <= deadline_days
        ]

        if subset.empty:
            continue

        subset["unit_total"] = (
            subset["unit_cost"] + delay_penalty * subset["predicted_delay_days"]
        )
        # Intentionally naive baseline: prefers fastest arrival before cost.
        subset["arrival_days"] = subset["lead_time_days"] + subset["predicted_delay_days"]
        subset = subset.sort_values(["arrival_days", "unit_total"])

        for _, row in subset.iterrows():
            if remaining <= 0:
                break
            qty = int(min(remaining, int(row["capacity"])))
            if qty <= 0:
                continue

            total_cost += qty * float(row["unit_total"])
            total_cost += float(row["fixed_penalty"])
            remaining -= qty

            records.append(
                {
                    "supplier_id": row["supplier_id"],
                    "component": component,
                    "quantity": qty,
                    "unit_cost": float(row["unit_cost"]),
                    "predicted_delay_days": float(row["predicted_delay_days"]),
                    "lead_time_days": float(row["lead_time_days"]),
                    "arrival_days": float(row["arrival_days"]),
                }
            )

    plan = pd.DataFrame(records)
    return plan, float(total_cost)


def optimize_procurement(
    df: pd.DataFrame,
    demand: Dict[str, int],
    budget: float,
    deadline_days: float,
    delay_penalty: float,
    delay_scenarios: Optional[np.ndarray] = None,
    risk_weight: float = 0.0,
    scenario_confidence: float = 0.9,
    delay_variability_cap: float = 3.0,
    solver_time_limit: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Solve procurement with MILP and hard feasibility constraints."""
    model = pulp.LpProblem("Aerospace_Supply_Chain_Optimization", pulp.LpMinimize)

    x = {}
    y = {}

    for idx, _ in df.iterrows():
        x[idx] = pulp.LpVariable(f"x_{idx}", lowBound=0, cat="Integer")
        y[idx] = pulp.LpVariable(f"y_{idx}", lowBound=0, upBound=1, cat="Binary")

    expected_delay = df["predicted_delay_days"].to_numpy(dtype=float)
    # If Monte Carlo scenarios are available, use their moments for expected/worst delay.
    if delay_scenarios is not None and delay_scenarios.shape[0] == len(df.index):
        expected_delay = delay_scenarios.mean(axis=1)
        worst_delay = np.quantile(delay_scenarios, scenario_confidence, axis=1)
    else:
        worst_delay = expected_delay

    supplier_mean_rel = df.groupby("supplier_id")["reliability_score"].mean()
    supplier_rel_var = float(supplier_mean_rel.var()) if len(supplier_mean_rel) > 1 else 0.0
    # Risk proxy: distance from mean reliability (higher distance => higher variance contribution).
    supplier_risk = {
        sid: float((rel - supplier_mean_rel.mean()) ** 2)
        for sid, rel in supplier_mean_rel.to_dict().items()
    }

    cost_term = pulp.lpSum(
        df.loc[idx, "unit_cost"] * x[idx]
        + df.loc[idx, "fixed_penalty"] * y[idx]
        + delay_penalty * float(expected_delay[pos]) * x[idx]
        for pos, idx in enumerate(df.index)
    )
    risk_term = pulp.lpSum(
        supplier_risk.get(str(df.loc[idx, "supplier_id"]), 0.0) * x[idx]
        for idx in df.index
    )
    risk_scale = float(df["unit_cost"].mean()) if float(df["unit_cost"].mean()) > 0 else 1.0
    clipped_weight = float(np.clip(risk_weight, 0.0, 1.0))
    # Weighted-sum multi-objective: cost term + scaled risk term.
    model += (1.0 - clipped_weight) * cost_term + clipped_weight * risk_scale * risk_term

    for component, required_qty in demand.items():
        idxs = [idx for idx in df.index if df.loc[idx, "component"] == component]
        model += pulp.lpSum(x[idx] for idx in idxs) >= int(required_qty)

    for pos, idx in enumerate(df.index):
        model += x[idx] <= int(df.loc[idx, "capacity"]) * y[idx]
        expected_arrival = float(df.loc[idx, "lead_time_days"] + expected_delay[pos])
        if expected_arrival > deadline_days:
            # Prune clearly infeasible lanes in expected terms.
            model += x[idx] == 0

        # Robust constraint 1: if used, the q-quantile arrival must meet deadline.
        big_m = float(deadline_days + 40.0)
        model += (
            float(df.loc[idx, "lead_time_days"] + worst_delay[pos])
            <= deadline_days + big_m * (1 - y[idx])
        )

    # Robust constraint 2: each component must satisfy deadline in weighted worst-case average.
    for component in demand.keys():
        idxs = [idx for idx in df.index if df.loc[idx, "component"] == component]
        model += pulp.lpSum(
            float(df.loc[idx, "lead_time_days"] + worst_delay[df.index.get_loc(idx)]) * x[idx]
            for idx in idxs
        ) <= deadline_days * pulp.lpSum(x[idx] for idx in idxs)

    # Robust constraint 3: cap total tail-delay exposure above expected delays.
    total_demand = float(sum(demand.values()))
    tail_exposure = pulp.lpSum(
        max(0.0, float(worst_delay[pos] - expected_delay[pos])) * x[idx]
        for pos, idx in enumerate(df.index)
    )
    model += tail_exposure <= delay_variability_cap * total_demand

    model += pulp.lpSum(df.loc[idx, "unit_cost"] * x[idx] for idx in df.index) <= budget

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(solver_time_limit))
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        # Keep output schema stable even on infeasible/non-optimal runs.
        summary = {
            "status": status,
            "procurement_cost": 0.0,
            "fixed_cost": 0.0,
            "delay_cost": 0.0,
            "total_cost": 0.0,
            "risk_score": 0.0,
            "on_time_rate": 0.0,
            "budget": float(budget),
            "deadline_days": float(deadline_days),
            "risk_weight": clipped_weight,
            "scenario_count": int(delay_scenarios.shape[1]) if delay_scenarios is not None else 0,
            "supplier_reliability_variance": supplier_rel_var,
        }
        return pd.DataFrame(), summary

    chosen_rows = []
    for idx in df.index:
        qty = int(round(pulp.value(x[idx]) or 0))
        if qty <= 0:
            continue
        row = df.loc[idx]
        pos = df.index.get_loc(idx)
        chosen_rows.append(
            {
                "supplier_id": row["supplier_id"],
                "component": row["component"],
                "quantity": qty,
                "unit_cost": float(row["unit_cost"]),
                "fixed_penalty": float(row["fixed_penalty"]),
                "predicted_delay_days": float(row["predicted_delay_days"]),
                "expected_delay_days": float(expected_delay[pos]),
                "worst_delay_days": float(worst_delay[pos]),
                "lead_time_days": float(row["lead_time_days"]),
                "arrival_days": float(row["lead_time_days"] + expected_delay[pos]),
                "worst_arrival_days": float(row["lead_time_days"] + worst_delay[pos]),
                "supplier_risk_coeff": float(
                    supplier_risk.get(str(row["supplier_id"]), 0.0)
                ),
            }
        )

    plan = pd.DataFrame(chosen_rows)

    # Split objective into readable post-solve components for reporting.
    procurement_cost = (
        float((plan["quantity"] * plan["unit_cost"]).sum()) if not plan.empty else 0.0
    )
    fixed_cost = float(plan["fixed_penalty"].sum()) if not plan.empty else 0.0
    delay_col = "expected_delay_days" if "expected_delay_days" in plan.columns else "predicted_delay_days"
    delay_cost = float((plan["quantity"] * plan[delay_col] * delay_penalty).sum()) if not plan.empty else 0.0
    risk_score = (
        float((plan["quantity"] * plan["supplier_risk_coeff"]).sum()) if not plan.empty else 0.0
    )
    total_cost = procurement_cost + fixed_cost + delay_cost
    on_time_rate = (
        float((plan["worst_arrival_days"] <= deadline_days).mean())
        if not plan.empty and "worst_arrival_days" in plan.columns
        else 0.0
    )

    summary = {
        "status": status,
        "procurement_cost": procurement_cost,
        "fixed_cost": fixed_cost,
        "delay_cost": delay_cost,
        "total_cost": total_cost,
        "risk_score": risk_score,
        "on_time_rate": on_time_rate,
        "budget": float(budget),
        "deadline_days": float(deadline_days),
        "risk_weight": clipped_weight,
        "scenario_count": int(delay_scenarios.shape[1]) if delay_scenarios is not None else 0,
        "supplier_reliability_variance": supplier_rel_var,
    }

    return plan, summary


def generate_delay_scenarios(
    predicted_delay_days: np.ndarray,
    residual_std: float,
    n_scenarios: int = 80,
    seed: int = 42,
) -> np.ndarray:
    """Generate Monte Carlo delay scenarios from model residual uncertainty."""
    rng = np.random.default_rng(seed)
    base = np.asarray(predicted_delay_days, dtype=float)
    # Avoid near-zero variance to keep scenario set informative.
    sigma = float(max(0.05, residual_std))
    draws = rng.normal(loc=base[:, None], scale=sigma, size=(len(base), int(n_scenarios)))
    return np.clip(draws, 0.0, None)

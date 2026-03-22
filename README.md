# Aerospace Supply Chain Optimization

This repository is a practical demo of decision intelligence for critical supply chains.

It combines:

1. Delay prediction with machine learning.
2. Procurement optimization with MILP under uncertainty.

It is written for aerospace, but the same approach works for automotive, electronics, health equipment, and energy infrastructure.

## Why Aerospace?

Aerospace is a high-stakes supply-chain environment: one delayed or non-compliant part can block assembly, postpone launch windows, and trigger costly rework across tightly coupled systems. Unlike many industries, aerospace sourcing decisions must simultaneously optimize cost, reliability, certification constraints, and schedule robustness, which makes it an ideal benchmark for robust optimization.

## Aerospace Specificity

What is specific in aerospace supply chains:

- Critical components (engines, avionics, thermal protection) can have long lead times and limited qualified suppliers.
- Qualification and traceability requirements are strict, so supplier substitution is slower and more expensive.
- Integration milestones are tightly synchronized; a single late part can block test campaigns.
- Logistics are globally exposed to weather, geopolitical events, and transport mode risk.
- Failures have high safety and mission impact, so risk-aware planning is mandatory, not optional.

## Supply Chain

A supply chain is the flow of parts from suppliers to your production line.

In this project, each row in the dataset is one supplier-part lane, for example:

- Supplier `S07` can deliver `Avionics flight computer`.
- It has a cost, capacity, lead time, delay risk, and carbon intensity.

Your planning team must answer:

- Which suppliers should we select?
- How much should we buy from each?
- Can we meet demand and deadline without breaking budget?

## Real-World Application Examples

1. Aerospace final assembly
Use robust sourcing for engines, avionics, and thermal protection parts where late deliveries stop integration tests.

2. EV battery supply
Balance cost and carbon when choosing cathode suppliers across sea vs air lanes.

3. Medical device manufacturing
Prioritize reliability and anomaly alerts for sterile components with strict delivery windows.

4. Defense maintenance
Plan spare parts with uncertain logistics and geopolitical risk while preserving mission readiness.

## How the Model Works

### Step 1: Predict delay per supplier-part lane

The model estimates delay days:

$$
\hat{d}_{i,j} = f_{\text{GBDT}}(x_{i,j})
$$

where $x_{i,j}$ includes cost, lead time, reliability, transport mode, bottleneck risk, and telemetry anomaly signals.

Telemetry anomaly signals used by the model:

- `telemetry_temp_c`
- `telemetry_vibration`
- `telemetry_pressure_delta`
- `telemetry_packet_loss`
- `telemetry_anomaly_score`
- `telemetry_anomaly_flag`

### Step 2: Optimize procurement with MILP

Decision variables:

- $x_{i,j}$ = units ordered from supplier $i$ for part $j$.
- $y_{i,j}$ = binary lane activation (1 if lane used, 0 otherwise).

Objective used in this project:

$$
\min
\underbrace{\sum_{i,j} c_{i,j}x_{i,j} + \sum_{i,j} p_{i,j}y_{i,j} + \lambda\sum_{i,j}\hat{d}_{i,j}x_{i,j}}_{\text{cost + delay}}
+ \underbrace{\omega\sum_{i,j} r_i x_{i,j}}_{\text{supply risk}}
+ \underbrace{\mu\sum_{i,j} \text{carbon}_{i,j}x_{i,j}}_{\text{carbon footprint}}
$$

## Constraints, With Intuition and Examples

### 1) Demand coverage

$$
\sum_i x_{i,j} \ge D_j
$$

Meaning: each component must be purchased in sufficient quantity.

Example: if required `Raptor injector` demand is 120, selected suppliers must total at least 120 units.

### 2) Capacity linking

$$
x_{i,j} \le C_{i,j}y_{i,j}
$$

Meaning: you cannot buy from a lane that is not activated, and you cannot exceed lane capacity.

Example: if lane capacity is 40 and $y_{i,j}=1$, then $x_{i,j}\le40$. If $y_{i,j}=0$, then $x_{i,j}=0$.

### 3) Budget protection

$$
\sum_{i,j} c_{i,j}x_{i,j} \le B
$$

Meaning: total procurement spend stays under budget.

Example: with $B=2{,}000{,}000$, a plan costing 2.1M is infeasible even if fast and reliable.

### 4) Robust deadline envelope

$$
	ext{lead}_{i,j} + d^{\text{worst}}_{i,j} \le T_{\max} + M(1-y_{i,j})
$$

Meaning: if a lane is selected, even its worst-case delay must respect the deadline.

Example: if lead time is 18 days, worst delay is 12 days, and deadline is 28, lane is not allowed because $18+12=30>28$.

### 5) Tail-delay exposure cap

The model also limits aggregate delay tail exposure across all selected lanes.

Meaning: avoid plans that look cheap but are fragile under bad-delay scenarios.

## Uncertainty Handling (Stochastic + Robust)

Monte Carlo scenarios are generated from model residual uncertainty.

- Recommended scenarios: 50-100.
- Expected delay contributes to objective.
- Quantile worst delay enforces robust constraints.

Stochastic form:

$$
\min \sum_{i,j} c_{i,j}x_{i,j} + \lambda\,\mathbb{E}_{\xi}\left[\sum_{i,j} d_{i,j}(\xi)x_{i,j}\right]
$$

where $\xi$ is a scenario index.

## Multi-Objective Controls

### Risk weight

- `risk_weight = 0`: cost-first behavior.
- `risk_weight = 1`: stronger reliability diversification.

### Carbon weight

- `carbon_weight = 0`: carbon ignored.
- higher `carbon_weight`: lower-emission lanes are favored when feasible.

Per-unit carbon is modeled as `carbon_kg_per_unit`, and the output summary reports `carbon_score_kg`.

## Results Snapshot (PNG)

### Cost reduction chart

![Cost Reduction](results/cost_reduction.png)

How to read it:

- Left bar: naive baseline policy.
- Right bar: robust optimized policy.
- The gap is direct savings from better sourcing decisions.

### Optimized supplier network

![Optimized Network](results/optimized_network.png)

How to read it:

- Nodes represent suppliers and components.
- Edges show selected supplier-component lanes.
- Thicker edges indicate higher spend lanes.

### Delivery timeline

![Delivery Timeline](results/delivery_timeline.png)

How to read it:

- Each line is one selected procurement lane.
- Horizontal length represents arrival time in days.
- The distribution helps spot schedule bottlenecks quickly.

## Explain the Results

From the current run in `results/summary.json`:

- Optimization status: `Optimal`
- Baseline total cost: about `8.82M`
- Optimized total cost: about `2.76M`
- Cost reduction: about `68.74%`
- On-time rate (robust): `100%`
- Carbon score: about `10,281.64 kgCO2e`

Interpretation:

1. The optimized plan dramatically reduced cost while still meeting robust deadline constraints.
2. The model found a feasible low-risk solution under stochastic delay scenarios.
3. Carbon is not ignored: the optimizer includes it directly via `carbon_weight`.

## Added Value of This Project

### Operational value

1. Faster sourcing decisions under uncertainty.
2. Better service reliability through robust delay constraints.
3. Built-in what-if simulation for disruptions and demand shocks.

### Financial value

1. Significant spend reduction potential versus naive heuristics.
2. Budget compliance is guaranteed by formulation, not post-checks.
3. Fewer costly late-arrival escalations in critical assembly stages.

### Sustainability and risk value

1. Carbon footprint is optimized, not only reported.
2. Telemetry anomaly signals improve delay awareness earlier.
3. Risk/carbon/cost trade-offs are explicit and tunable for leadership decisions.

## Interactive What-If Dashboard

`app.py` provides a Streamlit interface to adjust:

- demand multiplier,
- budget,
- deadline,
- risk weight,
- carbon weight,
- supplier failure simulation.



## Quick Start

### 1) Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Run pipeline

```powershell
python optimize.py --run-pareto
```

### 3) Run tests

```powershell
pytest -q
```

### 4) Run dashboard locally

```powershell
streamlit run app.py
```

## Useful CLI Example

```powershell
python optimize.py --suppliers 25 --components 8 --budget 0 --deadline 28 --delay-penalty 4.0 --scenarios 80 --risk-weight 0.2 --carbon-weight 0.1 --run-pareto
```

## Notebooks

1. `01_data_preparation.ipynb`: generate and inspect data.
2. `02_delay_prediction.ipynb`: train delay model with anomaly features.
3. `03_optimization_model.ipynb`: run robust optimization with risk and carbon weights.
4. `04_results.ipynb`: generate Pareto trade-off outputs.



# Aerospace Supply Chain Optimization

This project shows, in plain terms, how to make better supplier decisions when delays are uncertain.

You get two pieces working together:

1. A delay predictor (Gradient Boosting).
2. A procurement optimizer (MILP with robust/stochastic delay handling).

The scenario is aerospace manufacturing, but the same logic applies to any critical supply chain.

## What Problem Are We Solving?

If one critical part is late, production can stop.
So the practical question is:

How do we buy enough parts, stay on budget, and still protect delivery timelines?

This repository answers that question with code you can run end to end.

## How the Model Works

### Step 1: Predict delay per supplier-part pair

The ML model estimates delay in days:

$$
\hat{d}_{i,j} = f_{\text{GBDT}}(x_{i,j})
$$

where $x_{i,j}$ includes reliability, distance, transport mode, bottleneck risk, and other features.

### Step 2: Optimize purchasing decisions (MILP)

Decision variables:

- $x_{i,j}$: how many units to order from supplier $i$ for part $j$.
- $y_{i,j}$: whether that supplier-part lane is used at all.

Base objective:

$$
\min \sum_{i,j} c_{i,j}x_{i,j} + \sum_{i,j} p_{i,j}y_{i,j} + \lambda\sum_{i,j}\hat{d}_{i,j}x_{i,j}
$$

### MILP Constraints in Human Language

These are the rules the optimizer must obey:

1. Demand coverage
Every part type must be ordered in enough quantity.

$$
\sum_i x_{i,j} \ge D_j
$$

2. Capacity linking
You cannot order from a supplier more than its capacity, and only if that lane is selected.

$$
x_{i,j} \le C_{i,j}y_{i,j}
$$

3. Budget protection
Total purchase spend cannot exceed the available budget.

$$
\sum_{i,j} c_{i,j}x_{i,j} \le B
$$

4. Deadline feasibility
If a lane is selected, its arrival must fit the deadline envelope.

$$
	ext{lead}_{i,j} + d^{\text{worst}}_{i,j} \le T_{\max} + M(1-y_{i,j})
$$

## Uncertainty Upgrade (Stochastic + Robust)

Using a single delay number can be risky. Real aerospace operations face weather, logistics interruptions, and geopolitical shocks.

This project now generates Monte Carlo delay scenarios from ML residual uncertainty:

- 50-100 scenarios recommended.
- Expected delay is used in objective terms.
- Worst-case quantile delay is used in robust constraints.

Stochastic objective:

$$
\min \sum_{i,j} c_{i,j}x_{i,j} + \lambda\,\mathbb{E}_{\xi}\left[\sum_{i,j} d_{i,j}(\xi)x_{i,j}\right]
$$

where $\xi$ is the scenario index.

Robust constraints added in code:

1. Lane-level worst-case deadline check.
2. Component-level weighted worst-case arrival bound.
3. Global tail-delay exposure cap.

## Multi-Objective Upgrade (Cost vs Risk)

The optimizer now supports a second objective: supply risk.

Risk is measured using supplier reliability variance and weighted with `risk_weight` in `[0,1]`.

- `risk_weight = 0`: pure cost focus.
- `risk_weight = 1`: maximum risk aversion.

You can sweep this parameter and generate a Pareto front.
The front is exported as an interactive Plotly HTML file.

## Interactive What-If Dashboard

`app.py` provides a Streamlit decision-support screen where you can:

- change demand multiplier,
- change budget and deadline,
- simulate supplier failure (set capacity to zero),
- re-optimize instantly.

## Scale Mode (Starship-Like Size)

The generator supports up to:

- 80 suppliers
- 15 components

Use CBC time limit in CLI. Typical command:

```powershell
python optimize.py --suppliers 80 --components 15 --time-limit 10 --scenarios 80 --run-pareto
```

Observed in this repo: the 80x15 run solved in about 0.9 seconds on the current environment.

## Project Structure

```text
Aerospace-Supply-Chain-Optimization/
├── app.py
├── data/
│   └── supply_chain_dataset.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_delay_prediction.ipynb
│   ├── 03_optimization_model.ipynb
│   └── 04_results.ipynb
├── results/
│   ├── baseline_plan.csv
│   ├── optimized_plan.csv
│   ├── summary.json
│   ├── pareto_front.csv
│   ├── pareto_front.html
│   ├── cost_reduction.png
│   ├── optimized_network.png
│   └── delivery_timeline.png
├── src/
│   ├── data_generation.py
│   ├── delay_predictor.py
│   ├── optimizer.py
│   └── graph_visualization.py
├── tests/
│   └── test_pipeline.py
├── optimize.py
├── requirements.txt
└── README.md
```

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

### 4) Run dashboard

```powershell
streamlit run app.py
```

## Useful CLI Flags

```powershell
python optimize.py --suppliers 25 --components 8 --budget 0 --deadline 28 --delay-penalty 4.0 --scenarios 80 --risk-weight 0.2 --run-pareto
```

- `--budget 0` means automatic budget estimate.
- `--scenarios` controls Monte Carlo uncertainty samples.
- `--scenario-confidence` controls robust quantile level (default 0.9).
- `--delay-variability-cap` controls tail-risk tolerance.
- `--time-limit` controls CBC solve limit.

## Tutorial Notebooks

1. `01_data_preparation.ipynb`: generate and inspect data.
2. `02_delay_prediction.ipynb`: train and evaluate delay model.
3. `03_optimization_model.ipynb`: run robust stochastic optimization.
4. `04_results.ipynb`: build cost-risk Pareto front.

## Advanced Weekend Roadmap

These are not fully implemented yet, but are the next high-impact additions:

- Carbon footprint objective term:

$$
+\mu \sum_{i,j} \text{carbon}_{i,j}x_{i,j}
$$

- Feed anomaly signals from telemetry into delay prediction.
- Deploy Streamlit app to Render or Streamlit Community Cloud.

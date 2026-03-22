from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px


def plot_cost_comparison(baseline_cost: float, optimized_cost: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Greedy baseline", "Optimized"]
    values = [baseline_cost, optimized_cost]
    colors = ["#9CA3AF", "#1D4ED8"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Total cost")
    ax.set_title("Cost Comparison")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:,.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_optimized_network(plan: pd.DataFrame, output_path: Path) -> None:
    graph = nx.Graph()

    suppliers = sorted(plan["supplier_id"].unique()) if not plan.empty else []
    components = sorted(plan["component"].unique()) if not plan.empty else []

    graph.add_nodes_from(suppliers, bipartite=0)
    graph.add_nodes_from(components, bipartite=1)

    for _, row in plan.iterrows():
        # Edge weight encodes spend on that supplier-component lane.
        weight = float(row["quantity"] * row["unit_cost"])
        graph.add_edge(row["supplier_id"], row["component"], weight=weight)

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(graph, seed=21) if graph.number_of_nodes() > 0 else {}
    # Scale widths to make high-spend lanes visually dominant.
    edge_widths = [max(0.8, d["weight"] / 50000) for _, _, d in graph.edges(data=True)]

    nx.draw_networkx_nodes(graph, pos, nodelist=suppliers, node_color="#0EA5E9", node_size=700, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=components, node_color="#22C55E", node_size=900, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    ax.set_title("Optimized Supplier-Component Network")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_delivery_timeline(plan: pd.DataFrame, output_path: Path) -> None:
    timeline = plan.copy()
    if timeline.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Delivery Timeline (No Orders)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return

    timeline["label"] = timeline["supplier_id"] + " -> " + timeline["component"]
    # Sort by arrival to read the schedule from fastest to slowest.
    timeline = timeline.sort_values("arrival_days").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = range(len(timeline))
    ax.barh(y_pos, timeline["arrival_days"], color="#F59E0B")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(timeline["label"], fontsize=7)
    ax.set_xlabel("Arrival time (days)")
    ax.set_title("Optimized Delivery Timeline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_pareto_front(pareto_df: pd.DataFrame, output_path: Path) -> None:
    if pareto_df.empty:
        output_path.write_text("<html><body><p>No Pareto data.</p></body></html>", encoding="utf-8")
        return

    fig = px.line(
        pareto_df,
        x="risk_score",
        y="total_cost",
        markers=True,
        # Color keeps the corresponding risk-weight setting visible on each point.
        color="risk_weight",
        title="Pareto Front: Cost vs Supply Risk",
        labels={
            "risk_score": "Supply risk score",
            "total_cost": "Total cost",
            "risk_weight": "Risk weight",
        },
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")

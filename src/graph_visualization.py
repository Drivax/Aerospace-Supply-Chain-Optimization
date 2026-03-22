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

    fig, ax = plt.subplots(figsize=(14, 8))

    if graph.number_of_nodes() == 0:
        ax.set_title("Optimized Supplier-Component Network (No Orders)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return

    # Deterministic two-column layout: suppliers on the left, components on the right.
    def _vertical_positions(nodes: list[str]) -> dict[str, float]:
        if not nodes:
            return {}
        if len(nodes) == 1:
            return {nodes[0]: 0.5}
        return {name: 1.0 - (idx / (len(nodes) - 1)) for idx, name in enumerate(nodes)}

    sup_y = _vertical_positions(suppliers)
    comp_y = _vertical_positions(components)
    pos = {s: (0.08, sup_y[s]) for s in suppliers}
    pos.update({c: (0.92, comp_y[c]) for c in components})

    supplier_spend = {s: 0.0 for s in suppliers}
    component_spend = {c: 0.0 for c in components}
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 0.0))
        edge_weights.append(w)
        if u in supplier_spend:
            supplier_spend[u] += w
            component_spend[v] += w
        else:
            supplier_spend[v] += w
            component_spend[u] += w

    max_edge = max(edge_weights) if edge_weights else 1.0
    edge_widths = [1.5 + 8.5 * (float(d["weight"]) / max_edge) for _, _, d in graph.edges(data=True)]
    edge_colors = [float(d["weight"]) / max_edge for _, _, d in graph.edges(data=True)]

    max_supplier_spend = max(supplier_spend.values()) if supplier_spend else 1.0
    max_component_spend = max(component_spend.values()) if component_spend else 1.0
    supplier_sizes = [700 + 2200 * (supplier_spend[s] / max_supplier_spend) for s in suppliers]
    component_sizes = [700 + 2200 * (component_spend[c] / max_component_spend) for c in components]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=suppliers,
        node_color="#0EA5E9",
        node_size=supplier_sizes,
        linewidths=1.0,
        edgecolors="#075985",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=components,
        node_color="#22C55E",
        node_size=component_sizes,
        linewidths=1.0,
        edgecolors="#166534",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        alpha=0.8,
        ax=ax,
    )

    # Keep labels readable by shortening very long component names.
    supplier_labels = {s: s for s in suppliers}
    component_labels = {
        c: (c if len(c) <= 24 else c[:21] + "...")
        for c in components
    }
    nx.draw_networkx_labels(graph, pos, labels=supplier_labels, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=component_labels, font_size=8, ax=ax)

    ax.text(0.08, 1.03, "Suppliers", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.86, 1.03, "Components", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.set_title("Optimized Supplier-Component Network (edge width/color = spend)")
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

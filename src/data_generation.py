from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


COMPONENTS: List[str] = [
    "Merlin 1D turbopump",
    "Raptor injector",
    "Starship heat shield tiles",
    "Avionics flight computer",
    "Cryogenic propellant tank",
    "Grid fin assembly",
    "Falcon interstage ring",
    "Engine controller unit",
    "Tank dome weld ring",
    "Methalox feedline valve",
    "Thermal protection blanket",
    "Pressurization manifold",
    "Payload bay actuator",
    "Composite fairing panel",
    "Navigation IMU module",
]

TRANSPORT_MODES = ["road", "air", "sea"]


@dataclass(frozen=True)
class GenerationConfig:
    num_suppliers: int = 25
    num_components: int = 8
    seed: int = 42


def _mode_delay_bias(mode: str) -> float:
    if mode == "air":
        return -1.2
    if mode == "road":
        return 0.4
    return 1.3


def _mode_carbon_factor(mode: str) -> float:
    if mode == "air":
        return 1.9
    if mode == "road":
        return 1.0
    return 0.55


def generate_supply_chain_data(config: GenerationConfig) -> pd.DataFrame:
    """Create synthetic supplier-component observations with realistic variability."""
    rng = np.random.default_rng(config.seed)
    rows = []
    # Keep generation bounded by the available component catalog.
    selected_components = COMPONENTS[: max(1, min(config.num_components, len(COMPONENTS)))]

    for supplier_id in range(1, config.num_suppliers + 1):
        supplier_reliability = np.clip(rng.normal(0.86, 0.08), 0.55, 0.99)
        geopolitical_base = float(np.clip(rng.beta(2, 8), 0.01, 0.95))
        weather_base = float(np.clip(rng.beta(2.5, 6), 0.01, 0.95))

        for component in selected_components:
            transport_mode = str(rng.choice(TRANSPORT_MODES, p=[0.45, 0.25, 0.30]))
            distance_km = float(rng.uniform(150, 12000))
            lead_time_days = float(rng.uniform(3, 18))
            unit_cost = float(rng.uniform(1200, 22000))
            fixed_penalty = float(rng.uniform(200, 4000))
            capacity = int(rng.integers(35, 170))
            demand_forecast = int(rng.integers(45, 130))
            inventory_level = int(rng.integers(5, 80))
            order_size = int(rng.integers(10, 90))
            seasonality_index = float(rng.uniform(0.7, 1.3))
            bottleneck_risk = float(np.clip(rng.beta(2, 5), 0.01, 0.95))

            # Synthetic telemetry channels commonly available from IoT gateway feeds.
            telemetry_temp_c = float(rng.normal(58, 14))
            telemetry_vibration = float(np.clip(rng.normal(5.0, 2.0), 0.0, 15.0))
            telemetry_pressure_delta = float(np.clip(rng.normal(1.2, 0.7), 0.0, 4.0))
            telemetry_packet_loss = float(np.clip(rng.beta(1.5, 12), 0.0, 1.0))

            # Telemetry anomaly score used later as a predictive signal for delays.
            telemetry_anomaly_score = float(
                np.clip(
                    0.34 * (telemetry_vibration / 15.0)
                    + 0.22 * (telemetry_pressure_delta / 4.0)
                    + 0.24 * telemetry_packet_loss
                    + 0.20 * float(abs(telemetry_temp_c - 60.0) / 40.0),
                    0.0,
                    1.0,
                )
            )
            telemetry_anomaly_flag = int(telemetry_anomaly_score >= 0.62)

            # Carbon estimate per shipped unit (kgCO2e/unit) based on distance and mode.
            carbon_kg_per_unit = float(
                np.clip(
                    8.0
                    + 0.0038 * distance_km * _mode_carbon_factor(transport_mode)
                    + 2.1 * float(order_size / 90.0)
                    + rng.normal(0, 1.2),
                    1.0,
                    180.0,
                )
            )

            historical_mean_delay = float(
                np.clip(
                    # Delay formula combines logistics and risk effects with noise.
                    5.8
                    - 4.0 * supplier_reliability
                    + _mode_delay_bias(transport_mode)
                    + 0.00009 * distance_km
                    + 1.8 * weather_base
                    + 1.6 * geopolitical_base
                    + 1.8 * bottleneck_risk
                    + 3.0 * telemetry_anomaly_score
                    + 1.1 * telemetry_anomaly_flag
                    + rng.normal(0, 0.6),
                    0.0,
                    10.0,
                )
            )

            historical_delay_std = float(np.clip(rng.normal(1.8, 0.7), 0.4, 4.8))
            # This is the realized training target for the supervised model.
            sampled_delay = float(
                np.clip(rng.normal(historical_mean_delay, historical_delay_std), 0, 20)
            )

            rows.append(
                {
                    "supplier_id": f"S{supplier_id:02d}",
                    "component": component,
                    "unit_cost": unit_cost,
                    "fixed_penalty": fixed_penalty,
                    "lead_time_days": lead_time_days,
                    "capacity": capacity,
                    "reliability_score": supplier_reliability,
                    "historical_mean_delay": historical_mean_delay,
                    "historical_delay_std": historical_delay_std,
                    "distance_km": distance_km,
                    "transportation_mode": transport_mode,
                    "inventory_level": inventory_level,
                    "demand_forecast": demand_forecast,
                    "order_size": order_size,
                    "seasonality_index": seasonality_index,
                    "weather_risk": weather_base,
                    "geopolitical_risk": geopolitical_base,
                    "bottleneck_risk": bottleneck_risk,
                    "telemetry_temp_c": telemetry_temp_c,
                    "telemetry_vibration": telemetry_vibration,
                    "telemetry_pressure_delta": telemetry_pressure_delta,
                    "telemetry_packet_loss": telemetry_packet_loss,
                    "telemetry_anomaly_score": telemetry_anomaly_score,
                    "telemetry_anomaly_flag": telemetry_anomaly_flag,
                    "carbon_kg_per_unit": carbon_kg_per_unit,
                    "delay_days": sampled_delay,
                }
            )

    return pd.DataFrame(rows)


def default_component_demand(df: pd.DataFrame) -> Dict[str, int]:
    """Derive demand target per component from dataset forecast values."""
    # Use average forecast per component as a simple planning demand proxy.
    demand = (
        df.groupby("component", as_index=True)["demand_forecast"]
        .mean()
        .round()
        .astype(int)
        .to_dict()
    )
    return {k: int(max(40, v)) for k, v in demand.items()}

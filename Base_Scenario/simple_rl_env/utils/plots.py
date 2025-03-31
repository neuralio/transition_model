import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from utils.config import geojson_path  # Import the path from config.py

def plot_rewards(output_folder, rewards, cumulative_rewards):
    """
    Plot rewards and cumulative rewards over time.
    """
    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, label="Reward")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.grid(True)
    reward_plot = os.path.join(output_folder, "rewards_over_time.png")
    plt.savefig(reward_plot)
    plt.close()
    print(f"Saved plot '{reward_plot}'")

    # Plot cumulative rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Cumulative Reward", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.legend()
    plt.grid(True)
    cumulative_plot = os.path.join(output_folder, "cumulative_rewards_over_time.png")
    plt.savefig(cumulative_plot)
    plt.close()
    print(f"Saved plot '{cumulative_plot}'")


def plot_decision_trends(output_folder, decisions_array):
    """
    Plot decision trends over time.
    """
    # Count the number of 0s (crops) and 1s (PV installations) for each timestep
    crop_counts = np.sum(decisions_array == 0, axis=1)
    pv_counts = np.sum(decisions_array == 1, axis=1)

    # Plot the decision trends over time
    plt.figure(figsize=(10, 6))
    plt.plot(crop_counts, label="Crops (0)", marker="o")
    plt.plot(pv_counts, label="PV Installations (1)", linestyle="--", marker="x")
    plt.xlabel("Timestep")
    plt.ylabel("Number of Decisions")
    plt.title("Decision Trends Over Time")
    plt.ylim(170, 240)  # Adjust the y-axis as needed
    plt.legend()
    plt.grid(True)
    decision_trend_plot = os.path.join(output_folder, "decision_trends_over_time.png")
    plt.savefig(decision_trend_plot)
    plt.close()
    print(f"Saved plot '{decision_trend_plot}'")


def plot_agent_decisions(output_folder, agent_data_csv):
    """
    Plot agent decisions on a map of Cyprus.
    """
    # Load the agent data
    agent_data = pd.read_csv(agent_data_csv)

    # Convert lat/lon to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(agent_data["lon"], agent_data["lat"])]
    gdf_agents = gpd.GeoDataFrame(agent_data, geometry=geometry, crs="EPSG:4326")

    # Load the Cyprus shapefile (GeoJSON)
    gdf_cyprus = gpd.read_file(geojson_path)

    # Clip agent points to Cyprus boundary
    gdf_clipped = gpd.sjoin(gdf_agents, gdf_cyprus, predicate="within")

    # Plot the map
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the Cyprus shapefile with light grey land
    gdf_cyprus.plot(ax=ax, color="whitesmoke", edgecolor="black")

    # Plot the decisions with custom labels and colors
    for decision, (label, color) in [(0, ("Crop cultivation", "green")), (1, ("PV installation", "orange"))]:
        gdf_clipped[gdf_clipped["decision"] == decision].plot(
            ax=ax, markersize=50, color=color, label=label
        )

    # Add legend with custom location
    plt.legend(title="Land Use Decisions", loc="lower right", fontsize=10)

    # Add title and labels
    plt.title("Agent Decisions in Cyprus", fontsize=16)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)

    # Customize grid and axes
    plt.grid(visible=True, color="lightgrey", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10, colors="grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save the map plot
    map_plot_path = os.path.join(output_folder, "agent_decisions_map.png")
    plt.savefig(map_plot_path, dpi=300)
    plt.close()
    print(f"Saved map plot to '{map_plot_path}'")

import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO
from utils.config import get_file_paths
from utils.data_import import load_pv_data, load_crop_data, merge_data, aggregate_data


def ensure_output_folder(output_folder):
    """
    Ensure the output folder exists. If not, create it.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")


class PolicyWrapper:
    """
    A wrapper for the policy network to make it compatible with SHAP.
    """
    def __init__(self, policy):
        self.policy = policy

    def predict(self, X):
        # Convert input to a PyTorch tensor
        obs = th.tensor(X, dtype=th.float32)
        with th.no_grad():  # Disable gradient computation for inference
            logits, _, _ = self.policy.forward(obs)  # Get logits from the policy
            return logits.numpy()[:, 0]  # Return only the first action's logits


def compute_shap_values(aggregated_data, model_path, output_folder):
    """
    Compute and visualize SHAP values for the RL model.
    """
    # Load the trained model
    model = PPO.load(model_path)

    # Create a wrapper for the policy network
    policy_wrapper = PolicyWrapper(model.policy)

    # Prepare the input features (select only relevant features)
    input_features = aggregated_data[["crop_suitability", "pv_suitability"]]

    # Count unique rows in the input features
    unique_rows = len(input_features.drop_duplicates())
    n_clusters = min(50, unique_rows)  # Adjust the number of clusters

    if unique_rows < 50:
        print(f"Warning: Found only {unique_rows} unique rows in input features. Adjusting clusters to {n_clusters}.")

    # Summarize the background data using k-means clustering
    background = shap.kmeans(input_features.values, n_clusters)

    # Prepare SHAP explainer
    explainer = shap.KernelExplainer(
        model=policy_wrapper.predict,
        data=background
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(input_features.values)

    # Save SHAP decision plot
    decision_plot_path = os.path.join(output_folder, "shap_decision_plot.png")
    shap.decision_plot(
        explainer.expected_value,  # Expected value from the explainer
        shap_values,               # SHAP values
        feature_names=input_features.columns.tolist(),
        show=False
    )
    plt.savefig(decision_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP decision plot to {decision_plot_path}")

    # Save SHAP waterfall plot for the first instance
    waterfall_plot_path = os.path.join(output_folder, "shap_waterfall_plot.png")
    plt.figure()  # Create a new figure for the waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],                  # SHAP values for the first instance
            base_values=explainer.expected_value,  # Expected value from the explainer
            data=input_features.values[0],         # Feature values for the first instance
            feature_names=input_features.columns.tolist()
        )
    )
    plt.savefig(waterfall_plot_path, dpi=300, bbox_inches="tight")  # Save with tight layout
    plt.close()
    print(f"Saved SHAP waterfall plot to {waterfall_plot_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain RL model decisions using SHAP.")
    parser.add_argument("--scenario", type=str, required=True, help="RCP scenario for explainability.")
    parser.add_argument("--output-folder", type=str, default="explainability_results", help="Output folder for results.")
    args = parser.parse_args()

    # Ensure the output folder exists
    ensure_output_folder(args.output_folder)

    # Load and preprocess data
    file_paths = get_file_paths(args.scenario)
    PV_, PV_Yield_ = load_pv_data(file_paths["pv_suitability_file"], file_paths["pv_yield_file"])
    crop_suitability, crop_profit = load_crop_data(
        file_paths["cs_maize_file"],
        file_paths["cs_wheat_file"],
        file_paths["cy_maize_file"],
        file_paths["cy_wheat_file"]
    )
    env_data = merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
    aggregated_data = aggregate_data(env_data)

    # Define the model path (adjusted to be one directory back from output_folder)
    model_dir = os.path.dirname(args.output_folder)
    model_path = os.path.join(model_dir, f"RCP{args.scenario}_ppo_mesa_model.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Compute SHAP values and save visualizations
    compute_shap_values(aggregated_data, model_path, args.output_folder)

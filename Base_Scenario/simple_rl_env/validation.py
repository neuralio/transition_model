import os
import time
import numpy as np
import pandas as pd
from gym_env import MesaEnv
from stable_baselines3 import PPO
from utils.config import get_file_paths
from utils.data_import import load_pv_data, load_crop_data, merge_data, aggregate_data

# Fixed parameters
NUM_ENSEMBLES = 5
MAX_STEPS = 10
TOTAL_TIMESTEPS = 5000

def ensure_output_folder(output_folder):
    """
    Ensure the output folder exists. If not, create it.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

def run_ensemble(env_data, scenario, output_folder):
    """
    Run ensemble simulations for a fixed number of runs and save results.
    """
    results = []
    seeds = list(range(NUM_ENSEMBLES))

    for seed in seeds:
        print(f"Running ensemble with seed: {seed}")
        env = MesaEnv(env_data=env_data, max_steps=MAX_STEPS)
        model = PPO('MlpPolicy', env, verbose=1, seed=seed)

        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        end_time = time.time()

        model_path = os.path.join(output_folder, f"RCP{scenario}_ensemble_seed_{seed}_model.zip")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        rewards = []
        decisions = []
        obs = env.reset(seed=seed)[0]
        for step in range(MAX_STEPS):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            decisions.extend(action)
            if terminated or truncated:
                break

        pv_count = decisions.count(1)
        crop_count = decisions.count(0)
        total_agents = len(decisions)
        pv_percentage = (pv_count / total_agents) * 100 if total_agents > 0 else 0
        crop_percentage = (crop_count / total_agents) * 100 if total_agents > 0 else 0

        results.append({
            "seed": seed,
            "training_time": end_time - start_time,
            "rewards": rewards,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "pv_count": pv_count,
            "crop_count": crop_count,
            "pv_percentage": pv_percentage,
            "crop_percentage": crop_percentage
        })

        env.close()

    results_csv = os.path.join(output_folder, f"RCP{scenario}_ensemble_results.csv")
    pd.DataFrame(results).to_csv(results_csv, index=False)
    print(f"Saved ensemble results to {results_csv}")
    return results_csv

def uncertainty_quantification(results_csv, scenario, output_folder):
    """
    Perform uncertainty quantification based on the results CSV and save results to a CSV.
    """
    # Load the ensemble results
    ensemble_results = pd.read_csv(results_csv)
    
    # Convert the 'rewards' column from a string representation of a list to an actual list
    ensemble_results["rewards"] = ensemble_results["rewards"].apply(eval)

    # Extract metrics
    pv_percentages = pd.to_numeric(ensemble_results["pv_percentage"], errors='coerce')
    crop_percentages = pd.to_numeric(ensemble_results["crop_percentage"], errors='coerce')
    mean_rewards = pd.to_numeric(ensemble_results["mean_reward"], errors='coerce')
    std_rewards = pd.to_numeric(ensemble_results["std_reward"], errors='coerce')

    # Calculate summary statistics
    stats = {
        "Metric": ["PV Percentage", "Crop Percentage", "Mean Reward", "Std Reward"],
        "Mean": [
            pv_percentages.mean(),
            crop_percentages.mean(),
            mean_rewards.mean(),
            std_rewards.mean()
        ],
        "Std Dev": [
            pv_percentages.std(),
            crop_percentages.std(),
            mean_rewards.std(),
            std_rewards.std()
        ]
    }

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats)

    # Save the results to a CSV file
    uncertainty_csv = os.path.join(output_folder, f"RCP{scenario}_uncertainty_quantification_results.csv")
    stats_df.to_csv(uncertainty_csv, index=False)
    print(f"Saved uncertainty quantification results to {uncertainty_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate and quantify uncertainty for RCP scenarios.")
    parser.add_argument("--scenario", type=str, required=True, help="RCP scenario to validate.")
    parser.add_argument("--output-folder", type=str, default="ensemble_results", help="Output folder for results.")

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

    # Run ensemble
    results_csv = run_ensemble(aggregated_data, args.scenario, args.output_folder)

    # Perform uncertainty quantification
    print("Performing uncertainty quantification...")
    uncertainty_quantification(results_csv, args.scenario, args.output_folder)

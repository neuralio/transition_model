import argparse
import os
import time
import psutil
import torch  # For MPS support
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from utils.config import get_file_paths
from utils.data_import import load_pv_data, load_crop_data, merge_data, aggregate_data
from utils.plots import plot_rewards, plot_decision_trends, plot_agent_decisions
from gym_env import MesaEnv  # Import your custom environment
import pandas as pd 
import numpy as np


def get_processing_stats():
    """
    Monitor CPU and memory usage.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_status = "Using Apple's MPS GPU" if torch.backends.mps.is_available() else "GPU not available"
    return cpu_usage, memory_usage, gpu_status


def train_rl_model(env, scenario, learning_rate=0.0009809274356741915, batch_size=64, n_epochs=6):
    """
    Train the PPO model in the provided environment.
    """
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        device="mps"  # Use MPS on Apple Silicon
    )

    print("Starting training...")
    start_time = time.time()
    model.learn(total_timesteps=5000)
    end_time = time.time()

    print("Training completed. Saving model...")
    model.save(f"RCP{scenario}_ppo_mesa_model")

    training_time = end_time - start_time
    cpu_usage, memory_usage, gpu_status = get_processing_stats()

    print(f"Training Time: {training_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage}%")
    print(f"GPU Status: {gpu_status}")


def test_check_env(scenario):
    """
    Test environment compatibility with Gym API.
    """
    file_paths = get_file_paths(scenario)

    PV_, PV_Yield_ = load_pv_data(file_paths["pv_suitability_file"], file_paths["pv_yield_file"])
    crop_suitability, crop_profit = load_crop_data(
        file_paths["cs_maize_file"],
        file_paths["cs_wheat_file"],
        file_paths["cy_maize_file"],
        file_paths["cy_wheat_file"]
    )
    env_data = merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
    aggregated_data = aggregate_data(env_data)

    env = MesaEnv(env_data=aggregated_data, max_steps=10)

    print("Checking custom environment compatibility with Gym API...")
    check_env(env, warn=True)
    print("Environment passed the check!")

def plot_phase(scenario, env):
    """
    Load the trained model, run an evaluation loop, save results, and plot.
    """
    # Create dynamic output folder
    base_folder = f"plots_{scenario}_"
    counter = 1
    while os.path.exists(f"{base_folder}{counter}"):
        counter += 1
    output_folder = f"{base_folder}{counter}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # Load the trained model
    model = PPO.load(f"RCP{scenario}_ppo_mesa_model", env=env)

    # Initialize lists to store results
    decisions_over_time = []
    rewards_over_time = []
    cumulative_rewards = []
    agent_data_over_time = []
    cumulative_reward = 0

    # Evaluate the agent
    obs, _ = env.reset()
    for step in range(env.max_steps):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Save agent data (collect from environment or model)
        agent_data = env.model.get_agent_data()  # Replace with actual function to get agent data
        agent_data["Timestep"] = step
        agent_data_over_time.append(agent_data)

        # Save rewards and decisions
        decisions_over_time.append(action)
        rewards_over_time.append(reward)
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)

        if terminated or truncated:
            break

    env.close()

    # Save agent data to CSV
    agent_data_df = pd.concat(agent_data_over_time, ignore_index=True)
    agent_data_csv = os.path.join(output_folder, "agent_data_over_time.csv")
    agent_data_df.to_csv(agent_data_csv, index=False)
    print(f"Saved agent data to '{agent_data_csv}'")

    # Save rewards and decisions
    np.save(os.path.join(output_folder, "decisions_over_time.npy"), decisions_over_time)
    np.save(os.path.join(output_folder, "rewards_over_time.npy"), rewards_over_time)
    np.save(os.path.join(output_folder, "cumulative_rewards.npy"), cumulative_rewards)

    # Plot rewards and cumulative rewards
    plot_rewards(output_folder, rewards_over_time, cumulative_rewards)

    # Plot decision trends
    decisions_array = np.array(decisions_over_time)
    plot_decision_trends(output_folder, decisions_array)

    # Plot agent decisions on the map
    plot_agent_decisions(output_folder, agent_data_csv)

    # Indicate that plotting is complete
    print("Plotting complete.")

def main():
    parser = argparse.ArgumentParser(description="Run and optionally train and evaluate the RL model.")
    parser.add_argument("--scenario", type=str, choices=["26", "45", "85"], help="Specify the RCP scenario to test/train the environment.")
    parser.add_argument("--past", action="store_true", help="Use past data instead of future RCP scenarios.")
    parser.add_argument("--check-env", action="store_true", help="Check environment compatibility with Gym.")
    parser.add_argument("--train", action="store_true", help="Train the model if this flag is provided.")
    parser.add_argument("--plot", action="store_true", help="Load trained model and generate plots.")
    args = parser.parse_args()

    if args.past:
        print("Running with past data...")
        file_paths = get_file_paths(scenario=None, past=True)
    else:
        if not args.scenario:
            raise ValueError("You must provide --scenario when not using --past.")
        print(f"Running with future scenario RCP {args.scenario}...")
        file_paths = get_file_paths(args.scenario, past=False)

    # Load data
    PV_, PV_Yield_ = load_pv_data(file_paths["pv_suitability_file"], file_paths["pv_yield_file"])
    crop_suitability, crop_profit = load_crop_data(
        file_paths["cs_maize_file"],
        file_paths["cs_wheat_file"],
        file_paths["cy_maize_file"],
        file_paths["cy_wheat_file"]
    )
    env_data = merge_data(crop_suitability, PV_, PV_Yield_, crop_profit)
    aggregated_data = aggregate_data(env_data)

    env = MesaEnv(env_data=aggregated_data, max_steps=10)

    if args.check_env:
        print("Checking environment...")
        check_env(env, warn=True)

    if args.train:
        print("Training RL model...")
        train_rl_model(env, "past" if args.past else args.scenario)

    if args.plot:
        print("Plotting results...")
        plot_phase("past" if args.past else args.scenario, env)

if __name__ == "__main__":
    main()

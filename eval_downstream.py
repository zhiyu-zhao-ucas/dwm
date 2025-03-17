import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from loguru import logger

# Set the style for the plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14})

def load_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics(data, prefix="test/"):
    """Extract episode_reward_mean and success_ratio from the data."""
    metrics = {}
    
    # Get all keys that start with the prefix
    for entry in data:
        for key, value in entry.items():
            if key.startswith(prefix) and (key.endswith("episode_reward_mean") or key.endswith("success_ratio")):
                test_key = key.split('/')[1]  # Extract test1, test2, etc.
                metric_type = "reward" if "episode_reward_mean" in key else "success"
                
                if test_key not in metrics:
                    metrics[test_key] = {"reward": [], "success": []}
                
                metrics[test_key][metric_type].append(value)
    
    return metrics

def group_by_algorithm(all_metrics):
    """Group metrics by algorithm name, averaging across different seeds."""
    algo_metrics = {}
    
    for env_name, metrics in all_metrics.items():
        # Extract algorithm name (assuming format is {algo}-{seed})
        algo_name = re.sub(r'-\d+$', '', env_name)
        
        if algo_name not in algo_metrics:
            algo_metrics[algo_name] = {}
        
        # Merge metrics from this seed into the algorithm's data
        for test_key, test_data in metrics.items():
            logger.info(f"Processing {env_name}, {test_key}, {test_data}")
            if test_key not in algo_metrics[algo_name]:
                algo_metrics[algo_name][test_key] = {"reward": [], "success": []}
            
            # If this seed has data for this test, add it
            if test_data["reward"]:
                if not algo_metrics[algo_name][test_key]["reward"]:
                    algo_metrics[algo_name][test_key]["reward"] = [test_data["reward"][0]]
                else:
                    # Average with existing values
                    algo_metrics[algo_name][test_key]["reward"].append(test_data["reward"][0])
            
            if test_data["success"]:
                if not algo_metrics[algo_name][test_key]["success"]:
                    algo_metrics[algo_name][test_key]["success"] = [test_data["success"][0]]
                else:
                    # Average with existing values
                    algo_metrics[algo_name][test_key]["success"].append(test_data["success"][0])
    
    return algo_metrics

def plot_aggregate_metrics(all_metrics, save_dir="figures/downstream"):
    """Plot aggregate metrics across different environments using bar charts."""
    os.makedirs(save_dir, exist_ok=True)
    
    env_names = list(all_metrics.keys())
    
    # Calculate average values for each environment
    avg_rewards = []
    std_rewards = []
    avg_success_ratios = []
    std_success_ratios = []
    
    for env_name in env_names:
        metrics = all_metrics[env_name]
        
        # For each environment, calculate the average across all tests
        env_avg_reward = np.mean([
            np.mean(metrics[test]["reward"]) 
            for test in metrics
        ])
        env_std_reward = np.std([
            np.mean(metrics[test]["reward"]) 
            for test in metrics
        ])
        avg_rewards.append(env_avg_reward)
        std_rewards.append(env_std_reward)
        
        env_avg_success = np.mean([
            np.mean(metrics[test]["success"]) 
            for test in metrics
        ])
        env_std_success = np.std([
            np.mean(metrics[test]["success"]) 
            for test in metrics
        ])
        avg_success_ratios.append(env_avg_success)
        std_success_ratios.append(env_std_success)
    
    env_display_names = {}
    for i, env_name in enumerate(env_names):
        if env_name == 'dwm':
            env_display_names[env_name] = 'DWM (ours)'
        elif env_name == 'ours':
            env_display_names[env_name] = 'FCDL'
        else:
            env_display_names[env_name] = env_name.upper()
    transformed_env_names = [env_display_names[env_name] for env_name in env_names]
    algorithms = list(env_display_names.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    # Plot rewards bar chart
    plt.figure(figsize=(10, 10))
    x = np.arange(len(env_names))
    bars = plt.bar(x, avg_rewards, yerr=std_rewards, width=0.6, color=colors, capsize=5)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f'{avg_rewards[i]:.2f}', ha='center', va='bottom')
    
    plt.xlabel("Algorithm")
    plt.ylabel("Average Episode Reward")
    plt.title("Zero-Shot Downstream Average Rewards")
    plt.xticks(x, transformed_env_names, rotation=45, ha="right")
    # plt.xticks(x, env_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/aggregate_rewards_bar.png")
    plt.close()
    
    # Plot success ratios bar chart
    plt.figure(figsize=(12, 7))
    bars = plt.bar(x, avg_success_ratios, yerr=std_success_ratios, width=0.6, color=colors, capsize=5)
    
    # Add values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{avg_success_ratios[i]:.2f}', ha='center', va='bottom')
    
    plt.xlabel("Algorithm")
    plt.ylabel("Average Success Ratio")
    plt.title("Zero-Shot Downstream Average Success Ratios")
    plt.xticks(x, transformed_env_names, rotation=45, ha="right")
    plt.ylim(0, 1.05)  # Success ratio is between 0 and 1
    plt.tight_layout()
    plt.savefig(f"{save_dir}/aggregate_success_bar.png")
    plt.close()

def plot_per_test_rewards(all_metrics, save_dir="figures/downstream"):
    """Plot rewards for each test (test1, test2, test3) as subplots in a single figure."""
    os.makedirs(save_dir, exist_ok=True)
    
    env_names = list(all_metrics.keys())
    test_keys = set()
    for metrics in all_metrics.values():
        test_keys.update(metrics.keys())
    test_keys = sorted(list(test_keys))  # Sort test keys (test1, test2, test3, etc.)
    
    # Set display names for algorithms
    env_display_names = {}
    for i, env_name in enumerate(env_names):
        if env_name == 'dwm':
            env_display_names[env_name] = 'DWM (ours)'
        elif env_name == 'ours':
            env_display_names[env_name] = 'FCDL'
        else:
            env_display_names[env_name] = env_name.upper()
    
    transformed_env_names = [env_display_names[env_name] for env_name in env_names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(env_names)))
    
    # Create a single figure with subplots for each test
    fig, axes = plt.subplots(1, len(test_keys), figsize=(18, 6), sharey=True)
    
    for idx, test_key in enumerate(test_keys):
        test_rewards = []
        test_std_rewards = []
        
        for env_name in env_names:
            metrics = all_metrics[env_name]
            if test_key in metrics and metrics[test_key]["reward"]:
                rewards = metrics[test_key]["reward"]
                test_rewards.append(np.mean(rewards))
                test_std_rewards.append(np.std(rewards) if len(rewards) > 1 else 0)
            else:
                test_rewards.append(0)  # No data for this test
                test_std_rewards.append(0)
        
        # Plot rewards bar chart for this test in the corresponding subplot
        ax = axes[idx]
        x = np.arange(len(env_names))
        bars = ax.bar(x, test_rewards, yerr=test_std_rewards, width=0.6, color=colors, capsize=5)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{test_rewards[i]:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f"{test_key.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(transformed_env_names, rotation=45, ha="right")
    
    # Common labels
    fig.suptitle("Zero-Shot Downstream Rewards by Test", fontsize=16)
    fig.text(0.5, 0.01, "Algorithm", ha='center', fontsize=14)
    fig.text(0.01, 0.5, "Episode Reward", va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.2)
    plt.savefig(f"{save_dir}/rewards_all_tests.png")
    plt.close()

def main():
    data_dir = "downstream/zero_shot/reward"
    save_dir = "results/downstream"
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    all_metrics = {}
    
    # Process each JSON file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('-zero_shot.json'):
            file_path = os.path.join(data_dir, filename)
            env_name = filename.split('-zero_shot.json')[0]
            
            print(f"Processing {env_name}...")
            
            data = load_json_file(file_path)
            logger.info(f"Loaded {file_path}")
            metrics = extract_metrics(data)
            logger.info(f"Extracted metrics for {env_name}, {metrics}")
            
            # Store metrics for aggregate plots
            all_metrics[env_name] = metrics
    
    # Group metrics by algorithm (averaging across seeds)
    algo_metrics = group_by_algorithm(all_metrics)
    logger.info(f"Grouped metrics by algorithm: {algo_metrics}")
    
    # Plot aggregate metrics by algorithm
    plot_aggregate_metrics(algo_metrics, save_dir=save_dir)
    
    # Plot per test rewards as subplots in a single figure
    plot_per_test_rewards(algo_metrics, save_dir=save_dir)
    
    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    main()
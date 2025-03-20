import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from pathlib import Path

def format_algorithm_name(algo_name):
    """
    Format algorithm names:
    - Change "dwm" to "DWM (OURS)"
    - Change "ours" to "FCDL"
    - Convert all names to uppercase
    """
    if algo_name.lower() == "dwm":
        return "DWM (ours)"
    elif algo_name.lower() == "ours":
        return "FCDL"
    else:
        return algo_name.upper()

def load_data_from_json_files(data_dir='ood_data'):
    """
    Load data from all JSON files in the data directory.
    Returns a dictionary mapping algorithm-seed to the data.
    """
    results = {}
    
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract algorithm and seed from filename
                # Assuming filenames are in format "algo-seed.json"
                parts = filename.split('.')[0].split('-')
                raw_algo = parts[0]
                algo = format_algorithm_name(raw_algo)
                seed = parts[1]
                key = f"{algo}-{seed}"
                
                results[key] = data
                print(f"Successfully loaded {key} data from {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return results

def extract_metrics(data):
    """Extract the final accuracy metrics for each test set."""
    # Get the last entry which should have the final results
    if not data:
        return None
    
    last_entry = data[-1]
    
    return {
        'test1': last_entry.get('test/test1/inference/accuracy', None),
        'test2': last_entry.get('test/test2/inference/accuracy', None),
        'test3': last_entry.get('test/test3/inference/accuracy', None),
        'step': last_entry.get('step', None),
        'timestamp': last_entry.get('timestamp', None)
    }

def extract_learning_curves(data):
    """Extract learning curves for each test set."""
    steps = []
    test1_acc = []
    test2_acc = []
    test3_acc = []
    
    for entry in data:
        steps.append(entry.get('step', None))
        test1_acc.append(entry.get('test/test1/inference/accuracy', None))
        test2_acc.append(entry.get('test/test2/inference/accuracy', None))
        test3_acc.append(entry.get('test/test3/inference/accuracy', None))
    
    return {
        'steps': steps,
        'test1_acc': test1_acc,
        'test2_acc': test2_acc,
        'test3_acc': test3_acc
    }

def create_summary_dataframe(results):
    """Create a DataFrame with the final accuracies from all runs."""
    summary_data = []
    
    for key, data in results.items():
        metrics = extract_metrics(data)
        if not metrics:
            continue
        
        algo, seed = key.split('-')
        # Note: algorithm names are already formatted when loaded
        
        summary_data.append({
            'algorithm': algo,
            'seed': int(seed),
            'test1_acc': metrics['test1'],
            'test2_acc': metrics['test2'],
            'test3_acc': metrics['test3'],
            'final_step': metrics['step']
        })
    
    return pd.DataFrame(summary_data)

def plot_learning_curves(results, save_dir='result/plots'):
    """
    Plot learning curves for each test set with all algorithms.
    Each test will have its own figure with all algorithms plotted together.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Group data by algorithm
    algo_data = defaultdict(list)
    for key, data in results.items():
        algo, seed = key.split('-')
        # Note: algorithm names are already formatted when loaded
        algo_data[algo].append((int(seed), extract_learning_curves(data)))
    
    # Get a list of unique algorithms for consistent colors
    algorithms = list(algo_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    algo_color_map = dict(zip(algorithms, colors))
    
    # Create three separate plots for test1, test2, test3
    test_names = ['test1', 'test2', 'test3']
    
    for test_idx, test_name in enumerate(test_names):
        plt.figure(figsize=(10, 6))
        plt.title(f'{test_name} Accuracy - All Algorithms')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot each algorithm's data
        for algo_idx, algo in enumerate(algorithms):
            runs = algo_data[algo]
            
            # Group by seed to calculate mean and std
            steps = None
            all_accs = []
            
            for seed, curves in runs:
                if steps is None:
                    steps = curves['steps']
                acc_key = f'{test_name}_acc'
                all_accs.append(curves[acc_key])
            
            if not all_accs:
                continue
                
            # Convert to numpy array for easier calculations
            all_accs = np.array(all_accs)
            mean_acc = np.mean(all_accs, axis=0)
            std_acc = np.std(all_accs, axis=0)
            
            # Plot mean line with std shading
            plt.plot(steps, mean_acc, label=algo, color=algo_color_map[algo], linewidth=2)
            plt.fill_between(steps, mean_acc - std_acc, mean_acc + std_acc, 
                            color=algo_color_map[algo], alpha=0.2)
        
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{test_name}_comparison.pdf'))
        plt.close()

def plot_comparison_bar_charts(summary_df, save_dir='result/plots'):
    """Plot bar charts comparing the performance of different algorithms with colored bars."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate mean and std for each algorithm and test
    mean_data = summary_df.groupby('algorithm')[['test1_acc', 'test2_acc', 'test3_acc']].mean()
    std_data = summary_df.groupby('algorithm')[['test1_acc', 'test2_acc', 'test3_acc']].std()
    
    # Get algorithms for consistent colors
    algorithms = mean_data.index
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    plt.figure(figsize=(15, 6))
    
    # Test 1 comparison
    plt.subplot(1, 3, 1)
    plt.title('Test1 Accuracy Comparison')
    bars = plt.bar(mean_data.index, mean_data['test1_acc'], yerr=std_data['test1_acc'], 
                  capsize=5, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Test 2 comparison
    plt.subplot(1, 3, 2)
    plt.title('Test2 Accuracy Comparison')
    bars = plt.bar(mean_data.index, mean_data['test2_acc'], yerr=std_data['test2_acc'], 
                  capsize=5, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Test 3 comparison
    plt.subplot(1, 3, 3)
    plt.title('Test3 Accuracy Comparison')
    bars = plt.bar(mean_data.index, mean_data['test3_acc'], yerr=std_data['test3_acc'], 
                  capsize=5, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.pdf'))
    plt.close()
    
    # Create a summary table
    summary_table = pd.DataFrame({
        'test1_mean': mean_data['test1_acc'],
        'test1_std': std_data['test1_acc'],
        'test2_mean': mean_data['test2_acc'],
        'test2_std': std_data['test2_acc'],
        'test3_mean': mean_data['test3_acc'],
        'test3_std': std_data['test3_acc'],
    })
    
    return summary_table

def main():
    # Create plots directory
    plots_dir = Path('result/plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Load data from JSON files
    results = load_data_from_json_files('ood_data')
    
    if not results:
        print("No data found in the ood_data directory.")
        return
    
    # Create a summary DataFrame
    summary_df = create_summary_dataframe(results)
    
    # Save summary DataFrame to CSV
    summary_df.to_csv('result/csv/ood_summary.csv', index=False)
    print(f"Saved summary data to ood_summary.csv")
    
    # Plot learning curves
    plot_learning_curves(results, 'result/plots')
    print(f"Saved learning curves to plots directory")
    
    # Plot comparison bar charts and create summary table
    summary_table = plot_comparison_bar_charts(summary_df, 'result/plots')
    summary_table.to_csv('result/csv/ood_comparison.csv')
    print(f"Saved comparison data to ood_comparison.csv")
    
    # Print the summary table
    print("\nAlgorithm Performance Summary:")
    pd.set_option('display.precision', 4)
    print(summary_table)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
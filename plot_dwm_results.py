import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths for the different code sizes
code_sizes = [1, 2, 4, 8, 16]
file_paths = [f"downstream/zero_shot/reward/dwm_fork_code_size_{size}-0-zero_shot.json" for size in code_sizes]

# Define a color map for different code sizes
color_map = {
    '1': '#1f77b4',  # blue
    '2': '#ff7f0e',  # orange
    '4': '#2ca02c',  # green
    '8': '#d62728',  # red
    '16': '#9467bd'  # purple
}

# Data structures to store results
reward_means = {
    "test1": [],
    "test2": [],
    "test3": []
}
success_ratios = {
    "test1": [],
    "test2": [],
    "test3": []
}

# Extract data from JSON files
for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            entry = data[0]  # Get the first entry
            
            # Extract reward means
            reward_means["test1"].append(entry["test/test1/policy/episode_reward_mean"])
            reward_means["test2"].append(entry["test/test2/policy/episode_reward_mean"])
            reward_means["test3"].append(entry["test/test3/policy/episode_reward_mean"])
            
            # Extract success ratios
            success_ratios["test1"].append(entry["test/test1/policy/success_ratio"])
            success_ratios["test2"].append(entry["test/test2/policy/success_ratio"])
            success_ratios["test3"].append(entry["test/test3/policy/success_ratio"])

# Print the extracted values for verification
print("Code Sizes:", code_sizes)
print("Test1 Rewards:", reward_means["test1"])
print("Test2 Rewards:", reward_means["test2"])
print("Test3 Rewards:", reward_means["test3"])
print("Test1 Success Ratios:", success_ratios["test1"])
print("Test2 Success Ratios:", success_ratios["test2"])
print("Test3 Success Ratios:", success_ratios["test3"])

# Set global font size
plt.rcParams.update({'font.size': 64})

# Create plot for Episode Reward Mean
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(42, 15), sharex=False)

# Convert code_sizes to strings for the bar chart
code_sizes_str = [str(size) for size in code_sizes]

# Bar width
bar_width = 0.6

# Test 1 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax1.bar(i, reward_means["test1"][i], width=bar_width, color=color_map[size], alpha=0.8, label=f'Size {size}')
ax1.set_title('Test 1', fontsize=72)
ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
ax1.set_xlabel('Code Size', fontsize=64)
ax1.set_xticks(range(len(code_sizes_str)))
ax1.set_xticklabels(code_sizes_str)

# Test 2 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax2.bar(i, reward_means["test2"][i], width=bar_width, color=color_map[size], alpha=0.8)
ax2.set_title('Test 2', fontsize=72)
ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
ax2.set_xlabel('Code Size', fontsize=64)
ax2.set_xticks(range(len(code_sizes_str)))
ax2.set_xticklabels(code_sizes_str)

# Test 3 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax3.bar(i, reward_means["test3"][i], width=bar_width, color=color_map[size], alpha=0.8)
ax3.set_title('Test 3', fontsize=72)
ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
ax3.set_xlabel('Code Size', fontsize=64)
ax3.set_xticks(range(len(code_sizes_str)))
ax3.set_xticklabels(code_sizes_str)

# Set y-axis tick parameters
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', labelsize=60)

# Add a common y-label for all subplots
fig.text(0.02, 0.5, 'Episode Reward Mean', va='center', rotation='vertical', fontsize=72)

# Add legend for code sizes - position it at the top for better visibility
legend_labels = [f'Size {size}' for size in code_sizes_str]
legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[size], alpha=0.8) for size in code_sizes_str]
fig.legend(legend_handles, legend_labels, loc='upper center', 
           bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=50, 
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(left=0.08, wspace=0.2, bottom=0.18)  # Increased bottom margin for legend

# Save the reward plot
plt.savefig('dwm_reward_results.png')
plt.savefig('dwm_reward_results.pdf')

# Create plot for Success Ratio
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(42, 15), sharex=False)

# Test 1 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax1.bar(i, success_ratios["test1"][i], width=bar_width, color=color_map[size], alpha=0.8, label=f'Size {size}')
ax1.set_title('Test 1', fontsize=72)
ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
ax1.set_xlabel('Code Size', fontsize=64)
ax1.set_xticks(range(len(code_sizes_str)))
ax1.set_xticklabels(code_sizes_str)

# Test 2 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax2.bar(i, success_ratios["test2"][i], width=bar_width, color=color_map[size], alpha=0.8)
ax2.set_title('Test 2', fontsize=72)
ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
ax2.set_xlabel('Code Size', fontsize=64)
ax2.set_xticks(range(len(code_sizes_str)))
ax2.set_xticklabels(code_sizes_str)

# Test 3 subplot (bar chart)
for i, size in enumerate(code_sizes_str):
    ax3.bar(i, success_ratios["test3"][i], width=bar_width, color=color_map[size], alpha=0.8)
ax3.set_title('Test 3', fontsize=72)
ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
ax3.set_xlabel('Code Size', fontsize=64)
ax3.set_xticks(range(len(code_sizes_str)))
ax3.set_xticklabels(code_sizes_str)

# Set y-axis tick parameters
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', labelsize=60)

# Add a common y-label for all subplots
fig.text(0.02, 0.5, 'Success Ratio', va='center', rotation='vertical', fontsize=72)

# Add legend for code sizes - position it at the top for better visibility
legend_labels = [f'Size {size}' for size in code_sizes_str]
legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[size], alpha=0.8) for size in code_sizes_str]
fig.legend(legend_handles, legend_labels, loc='upper center', 
           bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=50, 
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(left=0.08, wspace=0.2, bottom=0.18)  # Increased bottom margin for legend

# Save the success ratio plot
plt.savefig('dwm_success_results.png')
plt.savefig('dwm_success_results.pdf')

# Generate markdown table
with open('dwm_results.md', 'w') as f:
    # Write table header
    f.write("# DWM Results\n\n")
    f.write("## Episode Reward Mean\n\n")
    f.write("| Code Size | Test 1 | Test 2 | Test 3 |\n")
    f.write("|-----------|--------|--------|--------|\n")
    
    # Write reward mean data
    for i, size in enumerate(code_sizes):
        f.write(f"| {size} | {reward_means['test1'][i]:.2f} | {reward_means['test2'][i]:.2f} | {reward_means['test3'][i]:.2f} |\n")
    
    # Write success ratio table
    f.write("\n## Success Ratio\n\n")
    f.write("| Code Size | Test 1 | Test 2 | Test 3 |\n")
    f.write("|-----------|--------|--------|--------|\n")
    
    # Write success ratio data
    for i, size in enumerate(code_sizes):
        f.write(f"| {size} | {success_ratios['test1'][i]:.2f} | {success_ratios['test2'][i]:.2f} | {success_ratios['test3'][i]:.2f} |\n")

print("Results visualized and saved to dwm_reward_results.png/pdf and dwm_success_results.png/pdf")
print("Markdown table generated at dwm_results.md") 
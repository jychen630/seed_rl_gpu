import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')

# Define the algorithms we want to display
ALGORITHMS = [ 'SeedSampling', 'Seed TD', 'Seed LSVI', 'UCRL','Thompson']
ALGORITHMS_FULL_NAME = {
    'SeedSampling': 'Seed Sampling',
    'Seed TD': 'Seed TD',
    'Seed LSVI': 'Seed LSVI',
    'UCRL': 'UCRL',
    'Thompson': 'Thompson Resampling'
}
# Define color schemes for different implementations
implementation_colors = {
    'Torch': '#FF6B6B',    # Coral red
    'JAX': '#2E8B57',      # Sea green
    'NumPy': '#9370DB'     # Medium purple
}

# Define markers based on implementation type (Toy vs Scale)
implementation_markers = {
    'Toy': 'o',  # circle (temporarily changed from '*')
    'Scale': 'o'  # circle
}

# Define the directories and their corresponding labels
directories = {
    'parallel_toy_torch': 'Torch Toy',
    'parallel_scale_torch': 'Torch Scale',
    'parallel_toy_jax': 'JAX Toy',
    'parallel_scale_jax': 'JAX Scale',
    'parallel_toy_numpy': 'NumPy Toy',
    'parallel_scale_numpy': 'NumPy Scale'
}

# Create output directory
output_dir = Path('../combined_plots')
os.makedirs(output_dir, exist_ok=True)

def get_implementation_type(label):
    if 'Torch' in label:
        return 'Torch'
    elif 'JAX' in label:
        return 'JAX'
    else:
        return 'NumPy'

def get_marker(label):
    # Temporarily use circle for all
    return 'o'
    # Original code:
    # return '*' if 'Toy' in label else 'o'

# Function to read and combine data for memory_per_agent metric
def read_memory_data():
    all_data = []
    
    for dir_name, label in directories.items():
        file_name = f"../{dir_name}/{dir_name}_memory_per_agent.csv"
        try:
            df = pd.read_csv(file_name)
            if not df.empty:
                df['Implementation'] = label
                all_data.append(df)
        except FileNotFoundError:
            print(f"Warning: {file_name} not found")
            continue
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Function to create subplots for memory_per_agent
def create_memory_subplots(data):
    # Create a figure with 1 row and 5 columns for each algorithm
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True, sharex=True)
    
    # Get all unique K values across all datasets
    k_columns = [col for col in data.columns if col.startswith('K=')]
    k_values = [int(col.split('=')[1]) for col in k_columns]
    
    # Dictionary to collect implementation data for each algorithm
    algo_data = {algo: {} for algo in ALGORITHMS}
    
    # Group data by implementation type and algorithm
    for impl_label in data['Implementation'].unique():
        impl_type = get_implementation_type(impl_label)
        impl_data = data[data['Implementation'] == impl_label]
        
        for _, row in impl_data.iterrows():
            algo = row['Algorithm']
            if algo in ALGORITHMS:
                if impl_type not in algo_data[algo]:
                    algo_data[algo][impl_type] = []
                
                values = []
                for col in k_columns:
                    if col in row.index:
                        values.append((col, row[col]))
                    else:
                        values.append((col, np.nan))
                
                algo_data[algo][impl_type].append({
                    'label': impl_label,
                    'values': values,
                    'marker': get_marker(impl_label)
                })
    
    # Manually set the winners for each algorithm for memory_per_agent
    best_frameworks = {
        'SeedSampling': ['Torch'],
        'Seed TD': ['Torch'],
        'Seed LSVI': ['Torch'],
        'UCRL': ['Torch'],
        'Thompson': ['Torch']
    }

    # Plot each algorithm in its own subplot
    for i, algo in enumerate(ALGORITHMS):
        ax = axes[i]
        ax.set_title(ALGORITHMS_FULL_NAME[algo])
        
        # Only add y-axis label to leftmost subplot
        if i == 0:
            ax.set_ylabel('Memory per Agent (MB)')
        
        # Add data for each implementation type in specific order - NumPy first, JAX second, Torch last (to be on top)
        for impl_type in ['NumPy', 'JAX', 'Torch']:  # Changed order to make Torch appear on top
            if impl_type in algo_data[algo]:
                for impl_data in algo_data[algo][impl_type]:
                    values_dict = {k: v for k, v in impl_data['values']}
                    plot_k_vals = []
                    plot_values = []
                    
                    for k_col in k_columns:
                        if k_col in values_dict:
                            plot_k_vals.append(int(k_col.split('=')[1]))
                            plot_values.append(values_dict[k_col])
                    
                    if plot_k_vals and plot_values:
                        line, = ax.plot(
                            plot_k_vals, 
                            plot_values, 
                            marker=impl_data['marker'],
                            linestyle='-',
                            color=implementation_colors[impl_type],
                            label=impl_data['label'],
                            linewidth=2,
                            markersize=8,
                            markeredgewidth=1.5,
                            markeredgecolor='black'
                        )
        
        ax.set_xscale('log')
        ax.grid(True, which="major", linestyle='-', alpha=0.3)
        ax.grid(True, which="minor", linestyle=':', alpha=0.2)
        
        # Set the same x-axis limits and ticks for all subplots
        if k_values:
            ax.set_xticks(sorted(list(set(k_values))))
            ax.set_xticklabels([str(k) for k in sorted(list(set(k_values)))])
        
        # Add boxes for best framework(s) (lower is better)
        if algo in best_frameworks and best_frameworks[algo]:
            best_frameworks_list = best_frameworks[algo]
            
            # Display each best framework in its own box, stacked vertically
            for idx, best_framework in enumerate(best_frameworks_list):
                vertical_offset = 0.10 * idx  # Space boxes 10% apart vertically
                
                ax.text(
                    0.95, 0.95 - vertical_offset, 
                    best_framework, 
                    transform=ax.transAxes,
                    bbox=dict(
                        facecolor=implementation_colors[best_framework],
                        alpha=0.8,
                        edgecolor='black',
                        boxstyle='round,pad=0.4',
                        linewidth=1
                    ),
                    ha='right', 
                    va='top',
                    fontsize=12,
                    fontweight='bold',
                    color='white'
                )
    
    # Add a single x-axis label for all subplots
    fig.text(0.5, 0.01, 'Number of Agents (K)', ha='center', fontsize=14, fontweight='bold')
    
    # Create a legend for the entire figure with color labels
    handles, labels = [], []
    
    # Add implementation type headers with color labels - keep the original order in legend
    handles.append(plt.Line2D([0], [0], color=implementation_colors['Torch'], 
                             linewidth=3, label='Torch'))
    labels.append('Torch')
    
    handles.append(plt.Line2D([0], [0], color=implementation_colors['JAX'], 
                             linewidth=3, label='JAX'))
    labels.append('JAX')
    
    handles.append(plt.Line2D([0], [0], color=implementation_colors['NumPy'], 
                             linewidth=3, label='NumPy'))
    labels.append('NumPy')
    
    # Add explanation for colored box
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.8, ec='black'))
    labels.append('Best at largest K')
    
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.05), 
              ncol=4, frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('Parallel Environment: Memory per Agent (MB) vs Number of Agents (K)', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increased bottom margin from 0.08 to 0.1
    
    # Add a prominent text box for performance direction
    fig.text(0.98, 0.98, "↓ LOWER IS BETTER",
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=1'),
             ha='right', va='top',
             fontsize=14, fontweight='bold')
    
    # Save the plot
    plt.savefig(output_dir / 'parallel_all_algos_memory_per_agent.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

# Main execution
if __name__ == "__main__":
    data = read_memory_data()
    if data is not None:
        create_memory_subplots(data)
    else:
        print("No data available for plotting")
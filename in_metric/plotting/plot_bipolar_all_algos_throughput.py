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
    'bipolar_toy_torch': 'Torch Toy',
    'bipolar_scale_torch': 'Torch Scale',
    'bipolar_toy_jax': 'JAX Toy',
    'bipolar_scale_jax': 'JAX Scale',
    'bipolar_toy': 'NumPy Toy',
    'bipolar_scale': 'NumPy Scale'
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

# Function to read and combine data for throughput_per_agent metric
def read_throughput_data():
    all_data = []
    
    for dir_name, label in directories.items():
        file_name = f"../{dir_name}/{dir_name}_times.csv"
        try:
            df = pd.read_csv(file_name)
            if not df.empty:
                # Calculate throughput directly - throughput is 1/time
                for col in df.columns:
                    if col.startswith('K='):
                        df[col] = 1.0 / df[col]  # Convert time to throughput (agents/ms)
                
                df['Implementation'] = label
                all_data.append(df)
        except FileNotFoundError:
            print(f"Warning: {file_name} not found")
            continue
    
    if not all_data:
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Function to create subplots for throughput_per_agent
def create_throughput_subplots(data):
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
    
    # Find best performing framework for each algorithm at its own largest K
    best_frameworks = {}
    
    for algo in ALGORITHMS:
        # Find the largest K value for this algorithm specifically
        largest_k_by_algo = None
        largest_k_col_by_algo = None
        
        # Find the largest K where at least one framework has a value
        for impl_type in ['Torch', 'JAX', 'NumPy']:
            if impl_type in algo_data[algo]:
                for impl_data in algo_data[algo][impl_type]:
                    for col, val in impl_data['values']:
                        if col.startswith('K=') and not np.isnan(val):
                            k_val = int(col.split('=')[1])
                            if largest_k_by_algo is None or k_val > largest_k_by_algo:
                                largest_k_by_algo = k_val
                                largest_k_col_by_algo = col
        
        if largest_k_col_by_algo:
            # Now find the best framework at this algorithm's largest K
            # For throughput, higher is better
            best_value = float('-inf')
            best_framework = None
            
            for impl_type in ['Torch', 'JAX', 'NumPy']:
                if impl_type in algo_data[algo]:
                    for impl_data in algo_data[algo][impl_type]:
                        for col, val in impl_data['values']:
                            if col == largest_k_col_by_algo and not np.isnan(val) and val > best_value:
                                best_value = val
                                best_framework = impl_type
            
            best_frameworks[algo] = best_framework


    # Plot each algorithm in its own subplot
    for i, algo in enumerate(ALGORITHMS):
        ax = axes[i]
        ax.set_title(ALGORITHMS_FULL_NAME[algo])
        
        # Only add y-axis label to leftmost subplot
        if i == 0:
            ax.set_ylabel('Throughput (agents/ms)')
        
        # Add data for each implementation type
        for impl_type, impl_list in algo_data[algo].items():
            for impl_data in impl_list:
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
        
        # Add simple box in top right for best framework
        if algo in best_frameworks and best_frameworks[algo]:
            best_framework = best_frameworks[algo]
            ax.text(
                0.95, 0.05, 
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
                va='bottom',
                fontsize=12,
                fontweight='bold',
                color='white'
            )
    
    # Add a single x-axis label for all subplots
    fig.text(0.5, 0.01, 'Number of Agents (K)', ha='center', fontsize=14, fontweight='bold')
    
    # Create a legend for the entire figure with color labels
    handles, labels = [], []
    
    # Add implementation type headers with color labels
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
    
    plt.suptitle('Bipolar Chain: Throughput (agents/ms) vs Number of Agents (K)', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increased bottom margin from 0.08 to 0.1
    
    # Add a prominent text box for performance direction
    fig.text(0.98, 0.98, "↑ HIGHER IS BETTER",
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=1'),
             ha='right', va='top',
             fontsize=14, fontweight='bold')
    
    # Save the plot
    plt.savefig(output_dir / 'bipolar_all_algos_throughput.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

# Main execution
if __name__ == "__main__":
    data = read_throughput_data()
    if data is not None:
        create_throughput_subplots(data)
    else:
        print("No data available for plotting")
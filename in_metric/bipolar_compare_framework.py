import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')

# Control section - easily toggle which algorithms to display
SHOW_ALGORITHMS = {
    'Thompson': True,     # Hide Thompson
    'SeedSampling': True,  # Show SeedSampling
    'UCRL': True,        # Hide UCRL
    'Seed TD': True,      # Show Seed TD
    'Seed LSVI': True     # Show Seed LSVI
}

# Define color schemes for different implementations
implementation_colors = {
    'Torch': '#FF6B6B',    # Coral red
    'JAX': '#2E8B57',      # Sea green
    'NumPy': '#9370DB'     # Medium purple
}

# Define markers based on implementation type (Toy vs Scale)
implementation_markers = {
    'Toy': '*',  # star
    'Scale': 'o'  # circle
}

# Define performance direction indicators
performance_indicators = {
    'time_per_agent': '↓',      # Lower is better
    'memory_per_agent': '↓',    # Lower is better
    'gpu_per_agent': '↓',       # Lower is better
    'throughput_per_agent': '↑' # Higher is better
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

# Define the metrics and their corresponding file names
metrics = {
    'time_per_agent': '_times.csv',
    'memory_per_agent': '_memory.csv',
    'gpu_per_agent': '_gpu.csv',
    'throughput_per_agent': '_times.csv'
}

# Create output directory
output_dir = Path('combined_plots')
os.makedirs(output_dir, exist_ok=True)

def get_implementation_type(label):
    if 'Torch' in label:
        return 'Torch'
    elif 'JAX' in label:
        return 'JAX'
    else:
        return 'NumPy'

def get_marker(label):
    return '*' if 'Toy' in label else 'o'

# Function to read and combine data
def read_metric_data(metric_name):
    all_data = []
    
    for dir_name, label in directories.items():
        file_name = f"{dir_name}/{dir_name}{metrics[metric_name]}"
        try:
            df = pd.read_csv(file_name)
            # Filter out algorithms that are set to False in SHOW_ALGORITHMS
            df = df[df['Algorithm'].apply(lambda x: SHOW_ALGORITHMS.get(x, True))]
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

# Function to create the plot
def create_plot(metric_name, data, ylabel):
    plt.figure(figsize=(12, 8))
    
    # Get all unique K values
    k_columns = [col for col in data.columns if col.startswith('K=')]
    k_values = [int(col.split('=')[1]) for col in k_columns]
    
    # Plot each implementation
    for impl in data['Implementation'].unique():
        impl_data = data[data['Implementation'] == impl]
        impl_type = get_implementation_type(impl)
        base_color = implementation_colors[impl_type]
        
        for idx, (_, row) in enumerate(impl_data.iterrows()):
            values = [row[col] for col in k_columns]
            if metric_name == 'throughput_per_agent':
                values = [1/v if v != 0 else np.nan for v in values]
            
            # Create a slightly different shade for each algorithm
            color = plt.cm.colors.to_rgba(base_color, alpha=0.7 + (idx * 0.1))
            
            plt.plot(k_values, values, 
                    marker=get_marker(impl),
                    linestyle='-',
                    color=color,
                    label=f"{impl} - {row['Algorithm']}",
                    linewidth=2,
                    markersize=10,  # Slightly larger markers
                    markeredgewidth=1.5,
                    markeredgecolor='black')
    
    plt.xscale('log')
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.xlabel('Number of Concurrent Agents (K)')
    plt.ylabel(ylabel)
    
    # Add a prominent text box for performance direction
    indicator_text = "↓ LOWER IS BETTER" if performance_indicators[metric_name] == "↓" else "↑ HIGHER IS BETTER"
    plt.text(0.98, 0.98, indicator_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=1'),
             ha='right', va='top',
             fontsize=14, fontweight='bold')
    
    plt.title(f'Bipolar Chain: {ylabel} vs Number of Agents')
    
    # Add a grid with different styles for major and minor lines
    plt.grid(True, which="major", linestyle='-', alpha=0.3)
    plt.grid(True, which="minor", linestyle=':', alpha=0.2)
    
    # Create a custom legend with implementation type headers
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_elements = []
    
    # Add implementation type headers
    for impl_type in ['Torch', 'JAX', 'NumPy']:
        if any(impl_type in label for label in labels):
            legend_elements.append(plt.Line2D([0], [0], color=implementation_colors[impl_type], 
                                            label=impl_type, linewidth=3))
    
    # Add algorithm entries
    for handle, label in zip(handles, labels):
        legend_elements.append((handle, label))
    
    plt.legend(handles=[elem[0] if isinstance(elem, tuple) else elem for elem in legend_elements],
              labels=[elem[1] if isinstance(elem, tuple) else elem.get_label() for elem in legend_elements],
              bbox_to_anchor=(1.05, 1),
              loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / f'{metric_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create plots for each metric
metric_labels = {
    'time_per_agent': 'Time per Agent (ms)',
    'memory_per_agent': 'Memory Usage per Agent (MB)',
    'gpu_per_agent': 'GPU Memory Usage per Agent (MB)',
    'throughput_per_agent': 'Throughput (agents/ms)'
}

for metric_name, ylabel in metric_labels.items():
    data = read_metric_data(metric_name)
    if data is not None:
        create_plot(metric_name, data, ylabel)
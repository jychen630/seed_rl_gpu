import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style to match bipolar plots
plt.style.use('seaborn-v0_8')

# Define colors for each implementation to match bipolar plots
implementation_colors = {
    'Torch': '#FF6B6B',    # Coral red
    'JAX': '#2E8B57',      # Sea green
    'NumPy': '#9370DB'     # Medium purple
}

# Create output directory
output_dir = Path('.')
os.makedirs(output_dir, exist_ok=True)

# Function to create subplot for memory_per_agent
def create_memory_subplot(torch_data, numpy_data, jax_data=None):
    # Create a figure with single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add title and labels
    ax.set_title('CartPole: Seed TD', fontsize=16)
    ax.set_ylabel('Memory per Agent (MB)')
    ax.set_xlabel('Number of Agents (K)', fontsize=14, fontweight='bold')
    
    # Dictionary to store the best framework for each algorithm
    best_frameworks = []
    
    # Plot data for each implementation with markers in the same order as bipolar
    # Plot each implementation type in specific order - NumPy first, JAX second, Torch last (to be on top)
    
    # Plot NumPy data
    if numpy_data is not None:
        ax.plot(
            numpy_data['K'], 
            numpy_data['memory_per_agent'],
            marker='o',
            linestyle='-',
            color=implementation_colors['NumPy'],
            label='NumPy',
            linewidth=2,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='black'
        )
        
        # Find the best framework for the largest K value
        largest_k = max(numpy_data['K'])
        min_memory_at_largest_k = numpy_data[numpy_data['K'] == largest_k]['memory_per_agent'].values[0]
        best_frameworks.append('NumPy')
    
    # Plot JAX data if available
    if jax_data is not None:
        ax.plot(
            jax_data['K'],
            jax_data['memory_per_agent'],
            marker='o',
            linestyle='-',
            color=implementation_colors['JAX'],
            label='JAX',
            linewidth=2,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='black'
        )
        
        # Compare with the previous best for largest K
        largest_k_jax = max(jax_data['K'])
        min_memory_at_largest_k_jax = jax_data[jax_data['K'] == largest_k_jax]['memory_per_agent'].values[0]
        
        if 'min_memory_at_largest_k' in locals():
            if largest_k_jax >= largest_k and min_memory_at_largest_k_jax < min_memory_at_largest_k:
                min_memory_at_largest_k = min_memory_at_largest_k_jax
                largest_k = largest_k_jax
                best_frameworks = ['JAX']
            elif largest_k_jax == largest_k and min_memory_at_largest_k_jax == min_memory_at_largest_k:
                best_frameworks.append('JAX')
    
    # Plot Torch data
    if torch_data is not None:
        ax.plot(
            torch_data['K'],
            torch_data['memory_per_agent'],
            marker='o',
            linestyle='-',
            color=implementation_colors['Torch'],
            label='Torch',
            linewidth=2,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='black'
        )
        
        # Compare with the previous best for largest K
        largest_k_torch = max(torch_data['K'])
        min_memory_at_largest_k_torch = torch_data[torch_data['K'] == largest_k_torch]['memory_per_agent'].values[0]
        
        if 'min_memory_at_largest_k' in locals():
            if largest_k_torch >= largest_k and min_memory_at_largest_k_torch < min_memory_at_largest_k:
                min_memory_at_largest_k = min_memory_at_largest_k_torch
                largest_k = largest_k_torch
                best_frameworks = ['Torch']
            elif largest_k_torch == largest_k and min_memory_at_largest_k_torch == min_memory_at_largest_k:
                best_frameworks.append('Torch')
    
    # Add best framework annotation in top right corner with matching style
    for idx, framework in enumerate(best_frameworks):
        vertical_offset = 0.10 * idx  # Space boxes 10% apart vertically
        
        ax.text(
            0.95, 0.95 - vertical_offset, 
            framework, 
            transform=ax.transAxes,
            bbox=dict(
                facecolor=implementation_colors[framework],
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
    
    # Set x-axis to log scale and grid
    ax.set_xscale('log')
    ax.grid(True, which="major", linestyle='-', alpha=0.3)
    ax.grid(True, which="minor", linestyle=':', alpha=0.2)
    
    # Set the xticks to the K values from the data
    all_k_values = sorted(list(set((torch_data['K'].tolist() if torch_data is not None else []) + 
                                 (numpy_data['K'].tolist() if numpy_data is not None else []) + 
                                 (jax_data['K'].tolist() if jax_data is not None else []))))
    ax.set_xticks(all_k_values)
    ax.set_xticklabels([str(k) for k in all_k_values])
    
    # Create a legend with color labels to match bipolar style
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
    
    # Add a prominent text box for performance direction
    fig.text(0.98, 0.98, "↓ LOWER IS BETTER",
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=1'),
             ha='right', va='top',
             fontsize=14, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increased bottom margin
    plt.savefig(output_dir / "cartpole_memory_per_agent.png", dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load the data from CSV files
    try:
        torch_data = pd.read_csv("../cartpole_torch_measure.csv")
    except FileNotFoundError:
        print("Warning: cartpole_torch_measure.csv not found")
        torch_data = None
    
    try:
        numpy_data = pd.read_csv("../cartpole_numpy_measure.csv")
    except FileNotFoundError:
        print("Warning: cartpole_numpy_measure.csv not found")
        numpy_data = None
    
    try:
        jax_data = pd.read_csv("../cartpole_jax_measure.csv")
    except FileNotFoundError:
        print("Warning: cartpole_jax_measure.csv not found")
        jax_data = None
    
    if torch_data is not None or numpy_data is not None or jax_data is not None:
        create_memory_subplot(torch_data, numpy_data, jax_data)
    else:
        print("No data available for plotting")
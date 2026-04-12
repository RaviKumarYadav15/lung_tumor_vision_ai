import matplotlib.pyplot as plt
import numpy as np

def generate_confidence_heatmap(original_scan, probability_map):
    """
    Creates an Explainable AI (XAI) thermal heatmap.
    - original_scan: 2D numpy array of the CT slice
    - probability_map: 2D numpy array of AI sigmoid outputs (0.0 to 1.0)
    Returns a matplotlib Figure object.
    """
    # Create a new, clean figure
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0B1215')
    ax.set_facecolor('#0B1215')
    
    # 1. Plot the original patient CT scan in standard grayscale
    ax.imshow(original_scan, cmap='gray')
    
    # 2. Mask out the completely healthy areas (e.g., < 10% chance) 
    # so we only see colors where the AI suspects something
    masked_prob_map = np.ma.masked_where(probability_map < 0.10, probability_map)
    
    # 3. Overlay the probability map using a thermal colormap ('jet' or 'inferno')
    # Red = 100% confidence, Blue/Green = lower confidence
    heatmap = ax.imshow(masked_prob_map, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
    
    # 4. Add a colorbar legend so doctors understand the colors
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('AI Confidence Level (0% to 100%)', color='white', rotation=270, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Formatting
    ax.axis('off')
    ax.set_title("Neural Network Confidence Heatmap", color='white', pad=15)
    fig.tight_layout()
    
    return fig
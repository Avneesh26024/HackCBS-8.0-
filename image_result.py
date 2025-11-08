# image_result.py

import os
import re
import matplotlib.pyplot as plt

# Create a directory to store results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_safe_filename(query: str) -> str:
    """Converts a user query into a safe filename."""
    # Remove non-alphanumeric characters
    safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', query).strip()
    # Replace spaces with underscores
    safe_name = re.sub(r'\s+', '_', safe_name)
    # Truncate to a reasonable length
    safe_name = safe_name[:50] or "plot"
    # Add extension
    return f"{safe_name}.png"


def save_plot(fig: plt.Figure, query: str) -> str:
    """
    Saves a matplotlib figure to the 'results' directory
    and returns the full path.
    """
    try:
        filename = _get_safe_filename(query)
        save_path = os.path.join(RESULTS_DIR, filename)

        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

        print(f"✅ Plot saved to: {save_path}")
        return save_path

    except Exception as e:
        print(f"❌ Error saving plot: {e}")
        return None
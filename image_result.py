# image_result.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- NEW IMPORT ---
from upload_to_uri import upload_to_gcs

# Create a directory to store results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_safe_filename(query: str, extension: str) -> str:
    """Converts a user query into a safe filename with a given extension."""
    # Remove non-alphanumeric characters
    safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', query).strip()
    # Replace spaces with underscores
    safe_name = re.sub(r'\s+', '_', safe_name)
    # Truncate to a reasonable length
    safe_name = (safe_name[:50] or "result")

    # Ensure the extension starts with a dot
    if not extension.startswith('.'):
        extension = f".{extension}"

    return f"{safe_name}{extension}"


def save_plot(fig: plt.Figure, query: str) -> (str, str):
    """
    Saves a matplotlib figure locally, uploads to GCS,
    and returns (local_path, gcs_uri).
    """
    try:
        filename = _get_safe_filename(query, "png")
        save_path = os.path.join(RESULTS_DIR, filename)

        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"✅ Plot saved locally to: {save_path}")

        # --- NEW: Upload to GCS ---
        uri = upload_to_gcs(save_path)
        if "Error" in uri:
            print(f"❌ GCS Upload Error: {uri}")
            return save_path, None

        print(f"✅ Plot uploaded to: {uri}")
        return save_path, uri
        # --- END NEW ---

    except Exception as e:
        print(f"❌ Error saving plot: {e}")
        return None, None


def save_to_excel(df: pd.DataFrame, query: str) -> (str, str):
    """
    Saves a DataFrame to an Excel file locally, uploads to GCS,
    and returns (local_path, gcs_uri).
    """
    try:
        filename = _get_safe_filename(query, "xlsx")
        save_path = os.path.join(RESULTS_DIR, filename)

        df.to_excel(save_path, index=False)
        print(f"✅ Excel file saved locally to: {save_path}")

        # --- NEW: Upload to GCS ---
        uri = upload_to_gcs(save_path)
        if "Error" in uri:
            print(f"❌ GCS Upload Error: {uri}")
            return save_path, None

        print(f"✅ Excel file uploaded to: {uri}")
        return save_path, uri
        # --- END NEW ---

    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")
        return None, None


def save_to_pdf(df: pd.DataFrame, query: str) -> (str, str):
    """
    Saves a DataFrame to a PDF file locally, uploads to GCS,
    and returns (local_path, gcs_uri).
    """
    try:
        filename = _get_safe_filename(query, "pdf")
        save_path = os.path.join(RESULTS_DIR, filename)

        pdf = FPDF(orientation='L')  # Landscape
        pdf.add_page()
        pdf.set_font("Arial", size=8)

        # Add Title
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt=query, ln=True, align='C')
        pdf.set_font("Arial", size=8)

        # Get Headers
        headers = df.columns.tolist()
        col_widths = [len(h) + 6 for h in headers]
        for i, col in enumerate(headers):
            max_len = df[col].astype(str).str.len().max()
            col_widths[i] = max(col_widths[i], max_len + 6, 10)
        total_width = sum(col_widths)
        page_width = pdf.w - 2 * pdf.l_margin
        if total_width > page_width:
            scale = page_width / total_width
            col_widths = [w * scale for w in col_widths]

        # --- Headers ---
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Arial", 'B', 8)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, txt=str(header), border=1, fill=True)
        pdf.ln()

        # --- Data Rows ---
        pdf.set_font("Arial", size=8)
        pdf.set_fill_color(255, 255, 255)
        fill = False
        for _, row in df.iterrows():
            for i, item in enumerate(row):
                pdf.cell(col_widths[i], 6, txt=str(item), border=1, fill=fill)
            pdf.ln()
            fill = not fill

        pdf.output(save_path)
        print(f"✅ PDF file saved locally to: {save_path}")

        # --- NEW: Upload to GCS ---
        uri = upload_to_gcs(save_path)
        if "Error" in uri:
            print(f"❌ GCS Upload Error: {uri}")
            return save_path, None

        print(f"✅ PDF file uploaded to: {uri}")
        return save_path, uri
        # --- END NEW ---

    except Exception as e:
        print(f"❌ Error saving PDF file: {e}")
        return None, None
"""
Workflow diagram for the research methodology.
Two approaches converging to ecological analysis.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def add_box(ax, x, y, w, h, text, color='#dce6f1', edgecolor='#2c3e50',
            fontsize=9, textcolor='black', style='round,pad=0.1', lw=1.5):
    """Add a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style,
                          facecolor=color, edgecolor=edgecolor,
                          linewidth=lw, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            ha='center', va='center', color=textcolor, zorder=4,
            multialignment='center')
    return box


def add_arrow(ax, x1, y1, x2, y2, color='#2c3e50', lw=1.5, style='->'):
    """Add an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle=style, mutation_scale=15,
                             color=color, linewidth=lw, zorder=2)
    ax.add_patch(arrow)


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "0_exported_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # === Color scheme ===
    c_data = '#e8f4e8'       # light green — data/input
    c_method1 = '#dce6f1'    # light blue — approach 1
    c_method2 = '#fde8d0'    # light orange — approach 2
    c_result = '#f5e6f0'     # light purple — results
    c_analysis = '#fff3cd'   # light yellow — analysis
    c_edge = '#2c3e50'       # dark — edges
    c_edge2 = '#c0392b'      # red — approach 2 arrows
    c_edge1 = '#2980b9'      # blue — approach 1 arrows

    # =====================================================================
    # TOP: Input Data
    # =====================================================================
    add_box(ax, 7, 8.3, 4.5, 0.7,
            'Field Measurements\n(ASD Contact Probe Spectra + Fractional Cover)',
            color=c_data, fontsize=10)

    # =====================================================================
    # LEFT BRANCH: Approach 1 — RF / PCA+RF
    # =====================================================================
    # Title
    ax.text(3.5, 7.3, 'Approach 1: Machine Learning',
            fontsize=12, fontweight='bold', ha='center', color=c_edge1)

    # Measured mixed spectra
    add_box(ax, 3.5, 6.5, 3.5, 0.6,
            'Measured Mixed Spectra\n(240 samples, 350–2500 nm)',
            color=c_data, fontsize=8)

    # Two input options
    add_box(ax, 2, 5.5, 2.2, 0.5,
            'Reflectance Only',
            color=c_method1, fontsize=8)
    add_box(ax, 5, 5.5, 2.2, 0.5,
            'Reflectance + Indices',
            color=c_method1, fontsize=8)

    # Models
    add_box(ax, 2, 4.5, 2.2, 0.5,
            'RF / PCA+RF',
            color=c_method1, fontsize=9, edgecolor=c_edge1)
    add_box(ax, 5, 4.5, 2.2, 0.5,
            'RF / PCA+RF',
            color=c_method1, fontsize=9, edgecolor=c_edge1)

    # Nested CV
    add_box(ax, 3.5, 3.5, 4.0, 0.5,
            'Nested 5-Fold Cross-Validation\n(RandomizedSearchCV)',
            color=c_method1, fontsize=8, edgecolor=c_edge1)

    # Results
    add_box(ax, 3.5, 2.5, 3.5, 0.5,
            'Predicted Fractional Cover\n(3-class & 5-class targets)',
            color=c_result, fontsize=8)

    # Arrows — Approach 1
    add_arrow(ax, 5.5, 7.95, 3.5, 6.85, color=c_edge1)  # data → mixed spectra
    add_arrow(ax, 3.5, 6.15, 2, 5.8, color=c_edge1)
    add_arrow(ax, 3.5, 6.15, 5, 5.8, color=c_edge1)
    add_arrow(ax, 2, 5.2, 2, 4.8, color=c_edge1)
    add_arrow(ax, 5, 5.2, 5, 4.8, color=c_edge1)
    add_arrow(ax, 2, 4.2, 3.5, 3.8, color=c_edge1)
    add_arrow(ax, 5, 4.2, 3.5, 3.8, color=c_edge1)
    add_arrow(ax, 3.5, 3.2, 3.5, 2.8, color=c_edge1)

    # =====================================================================
    # RIGHT BRANCH: Approach 2 — Transfer Learning
    # =====================================================================
    ax.text(10.5, 7.3, 'Approach 2: Transfer Learning',
            fontsize=12, fontweight='bold', ha='center', color=c_edge2)

    # Spectral library
    add_box(ax, 10.5, 6.5, 3.5, 0.6,
            'Spectral Library\n(168 pure endmember spectra, 6 components)',
            color=c_data, fontsize=8)

    # Synthetic data generation
    add_box(ax, 10.5, 5.5, 3.5, 0.6,
            'Synthetic Dataset Generation\n(10,000 samples, linear + bilinear mixing)',
            color=c_method2, fontsize=8, edgecolor=c_edge2)

    # Pre-training
    add_box(ax, 10.5, 4.5, 3.0, 0.5,
            'Pre-train 1D-CNN\n(on synthetic data)',
            color=c_method2, fontsize=9, edgecolor=c_edge2)

    # Fine-tuning
    add_box(ax, 10.5, 3.5, 3.0, 0.5,
            'Fine-tune 1D-CNN\n(on measured data)',
            color=c_method2, fontsize=9, edgecolor=c_edge2)

    # Results
    add_box(ax, 10.5, 2.5, 3.5, 0.5,
            'Predicted Fractional Cover\n(3-class & 5-class targets)',
            color=c_result, fontsize=8)

    # Arrows — Approach 2
    add_arrow(ax, 8.5, 7.95, 10.5, 6.85, color=c_edge2)  # data → library
    add_arrow(ax, 10.5, 6.15, 10.5, 5.85, color=c_edge2)
    add_arrow(ax, 10.5, 5.15, 10.5, 4.8, color=c_edge2)
    add_arrow(ax, 10.5, 4.2, 10.5, 3.8, color=c_edge2)
    # Measured data also feeds into fine-tuning
    add_arrow(ax, 5.5, 7.95, 8.8, 3.5, color=c_edge2, lw=1.0)
    ax.text(6.8, 5.5, 'Measured\ndata', fontsize=7, ha='center',
            color=c_edge2, fontstyle='italic', rotation=30)
    add_arrow(ax, 10.5, 3.2, 10.5, 2.8, color=c_edge2)

    # =====================================================================
    # BOTTOM: Convergence — Ecological Analysis
    # =====================================================================
    # Feature importance
    add_box(ax, 3.5, 1.5, 3.5, 0.5,
            'Feature Importance Analysis\n(per-band & regional permutation)',
            color=c_analysis, fontsize=8)

    # Ecological analysis
    add_box(ax, 7, 0.5, 6.0, 0.7,
            'Ecological Analysis\nBiocrust Functional Type Shifts Under Warming Treatments',
            color=c_analysis, fontsize=10, edgecolor='#856404')

    # Model comparison
    add_box(ax, 10.5, 1.5, 3.5, 0.5,
            'Model Comparison\n(RF vs PCA+RF vs 1D-CNN)',
            color=c_analysis, fontsize=8)

    # Arrows — bottom
    add_arrow(ax, 3.5, 2.2, 3.5, 1.8, color=c_edge)
    add_arrow(ax, 10.5, 2.2, 10.5, 1.8, color=c_edge)
    add_arrow(ax, 3.5, 1.2, 5, 0.9, color=c_edge)
    add_arrow(ax, 10.5, 1.2, 9, 0.9, color=c_edge)

    # =====================================================================
    # Decorative: approach labels with boxes
    # =====================================================================
    # Dashed boxes around each approach
    rect1 = mpatches.FancyBboxPatch((0.8, 2.1), 5.5, 5.5,
                                     boxstyle='round,pad=0.15',
                                     facecolor='none', edgecolor=c_edge1,
                                     linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(rect1)

    rect2 = mpatches.FancyBboxPatch((8.2, 2.1), 5.0, 5.5,
                                     boxstyle='round,pad=0.15',
                                     facecolor='none', edgecolor=c_edge2,
                                     linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(rect2)

    # --- Save ---
    out_file = out_dir / 'Figure_workflow.png'
    fig.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.2,
                facecolor='white')
    plt.close(fig)
    print(f"Figure saved to: {out_file}")


if __name__ == '__main__':
    main()

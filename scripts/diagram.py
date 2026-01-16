#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_final_kdv_diagram():
    # 1. Setup Data
    L = 12
    x = np.linspace(-L/2, L/2, 1000)
    h0 = 1.5          # Undisturbed depth
    amplitude = 1.2   # Wave amplitude (eta)
    width = 1.5       # Soliton width parameter
    x0 = 0.0          # Peak position

    # Soliton profile: eta(x) = A * sech^2((x-x0)/w)
    eta = amplitude * (1.0 / np.cosh((x - x0) / width))**2
    surface_z = h0 + eta

    # 2. Plotting Setup
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'mathtext.fontset': 'cm', 
        'axes.linewidth': 1.5
    })
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 3. Draw Physical Elements
    
    # Water body
    ax.fill_between(x, 0, surface_z, color='#d1eefc', alpha=1.0, zorder=1)
    
    # Free surface line
    ax.plot(x, surface_z, color='#005b96', linewidth=3, zorder=2)
    
    # Undisturbed depth line (Still water level)
    # Drawn across the whole domain
    ax.axhline(y=h0, color='gray', linestyle='--', linewidth=2, alpha=0.6, zorder=2)

    # Rigid Bottom (z=0) with hatching
    rect_bottom = patches.Rectangle((-L/2, -0.6), L, 0.6, 
                                    facecolor='#e0e0e0', hatch='///', edgecolor='gray', zorder=1)
    ax.add_patch(rect_bottom)
    ax.axhline(y=0, color='black', linewidth=3, zorder=3)

    # 4. Annotations

    arrow_props = dict(arrowstyle='<->', color='black', lw=2)
    vector_props = dict(arrowstyle='->', color='black', lw=2.5)

    # A. Undisturbed Depth (h0) - Left side
    # Measure from bottom (0) to still water level (h0)
    ax.annotate('', xy=(-4.5, 0), xytext=(-4.5, h0), arrowprops=arrow_props)
    ax.text(-4.3, h0/2, r'$h_0$', va='center', ha='left', fontsize=18)
    
    # B. Surface Displacement (eta) - At the peak
    # Measure from still water level (h0) to peak (h0 + A)
    peak_x = x0
    peak_z = h0 + amplitude
    ax.annotate('', xy=(peak_x, h0), xytext=(peak_x, peak_z), 
                arrowprops=dict(arrowstyle='<->', color='#b30000', lw=2))
    ax.text(peak_x + 0.2, h0 + amplitude/2, r'$\eta(x,t)$', color='#b30000', va='center', fontsize=18)

    # C. Coordinates & Gravity - Far Left
    # z-axis
    ax.annotate('', xy=(-5.5, 4.5), xytext=(-5.5, 0), 
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(-5.3, 4.3, r'$z$', va='center', fontsize=18)
    
    # Gravity (g)
    ax.annotate('', xy=(-5.0, 3.5), xytext=(-5.0, 4.5), arrowprops=vector_props)
    ax.text(-4.8, 4.0, r'$g$', va='center', fontsize=18)
    
    # x-axis (Horizontal along bottom)
    ax.annotate('', xy=(5.5, -0.8), xytext=(-5.5, -0.8), 
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(5.5, -1.2, r'$x$', ha='center', fontsize=18)

    # D. Propagation Velocity (c) - Top Center
    ax.annotate('', xy=(2.5, peak_z + 0.5), xytext=(-0.5, peak_z + 0.5), arrowprops=vector_props)
    ax.text(1.0, peak_z + 0.7, r'$c$', va='center', fontsize=18)

    # E. Labels (Moved to avoid crowding)
    
    # Still water level: Moved to the right side where the wave is flat
    ax.text(4.5, h0 - 0.2, 'Still water level', fontsize=12, color='gray', style='italic', ha='right', va='top')

    # Rigid Bottom
    ax.text(3.5, -0.3, r'Rigid Bottom ($z=0$)', fontsize=14, color='black', alpha=0.9, ha='center', va='center')

    # 5. Final Layout Adjustments
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-1.5, 5.0)
    
    # Turn off the frame/ticks for a clean diagram
    ax.axis('off')

    # 6. Save
    output_dir = "../figs"
    filename = "diagram"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for fmt in ['png', 'pdf', 'eps']:
        save_path = os.path.join(output_dir, f"{filename}.{fmt}")
        plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    plt.close()

if __name__ == "__main__":
    create_final_kdv_diagram()

#!/usr/bin/env python3
"""
3D Spatiotemporal Visualization of KdV Solitons
Creates publication-quality 3D surface plots for all 4 test cases.

Generates:
- ../figs/spatiotemporal_evolution.{eps,pdf,png}
- ../stats/spatiotemporal_evolution.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration - EPS compliant, white background, BOLD labels
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")

FIGS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {'file': 'case1_single_soliton.nc', 'label': 'Single soliton'},
    {'file': 'case2_two_equal_solitons.nc', 'label': 'Two equal solitons'},
    {'file': 'case3_soliton_collision.nc', 'label': 'Soliton collision'},
    {'file': 'case4_three_solitons.nc', 'label': 'Three solitons'},
]


def load_netcdf(filepath):
    """Load NetCDF file and extract relevant data."""
    with Dataset(filepath, 'r') as nc:
        data = {
            'x': nc.variables['x'][:].data,
            't': nc.variables['t'][:].data,
            'u': nc.variables['u'][:].data,
            'mu': float(nc.mu),
            'epsilon': float(nc.epsilon),
            'scenario': str(nc.scenario),
            'nx': int(nc.nx),
            'dx': float(nc.dx),
        }
    return data


def downsample_for_plotting(x, t, u, max_x_points=80, max_t_points=60):
    """Downsample data for cleaner 3D visualization."""
    x_stride = max(1, len(x) // max_x_points)
    t_stride = max(1, len(t) // max_t_points)
    
    x_ds = x[::x_stride]
    t_ds = t[::t_stride]
    u_ds = u[::t_stride, ::x_stride]
    
    return x_ds, t_ds, u_ds


def main():
    # ========================================================================
    # Load all cases
    # ========================================================================
    all_data = []
    for case_info in CASES:
        filepath = DATA_DIR / case_info['file']
        if filepath.exists():
            data = load_netcdf(filepath)
            data['label'] = case_info['label']
            all_data.append(data)
            print(f"Loaded: {case_info['file']}")
        else:
            print(f"Warning: {filepath} not found")
            all_data.append(None)
    
    valid_data = [d for d in all_data if d is not None]
    if not valid_data:
        raise FileNotFoundError("No data files found in ../data/")
    
    # ========================================================================
    # Determine GLOBAL limits - SAME for ALL panels
    # ========================================================================
    global_x_min = min(d['x'].min() for d in valid_data)
    global_x_max = max(d['x'].max() for d in valid_data)
    global_t_min = 0
    global_t_max = max(d['t'].max() for d in valid_data)
    global_u_max = max(d['u'].max() for d in valid_data)
    
    # Round to nice values for tight limits
    global_x_lim = (-50, 50)  # Symmetric x range
    global_t_lim = (0, 80)    # Max t
    global_u_lim = (0, 9)     # Slightly above max u
    
    norm = Normalize(vmin=0, vmax=9)
    
    # ========================================================================
    # Create Figure - MORE SPACE for colorbar
    # ========================================================================
    fig = plt.figure(figsize=(14.0, 12.0))
    fig.patch.set_facecolor('white')
    
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # View angle
    elev, azim = 25, -55
    
    # Subplot positions: [left, bottom, width, height]
    # Reduced width to leave space for colorbar on right
    pw = 0.38  # panel width
    ph = 0.38  # panel height
    
    positions = [
        [0.05, 0.54, pw, ph],   # (a) top-left
        [0.48, 0.54, pw, ph],   # (b) top-right
        [0.05, 0.08, pw, ph],   # (c) bottom-left
        [0.48, 0.08, pw, ph],   # (d) bottom-right
    ]
    
    axes_list = []
    
    for idx in range(4):
        ax = fig.add_axes(positions[idx], projection='3d')
        ax.set_facecolor('white')
        axes_list.append(ax)
        
        data = all_data[idx]
        
        if data is None:
            ax.text(0.5, 0.5, 0.5, 'Data not available', 
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
            continue
        
        # Downsample
        x_ds, t_ds, u_ds = downsample_for_plotting(data['x'], data['t'], data['u'])
        
        # Meshgrid
        X, T = np.meshgrid(x_ds, t_ds)
        U = u_ds
        
        # Plot surface with HEAT colormap (hot)
        surf = ax.plot_surface(
            X, T, U,
            cmap='hot',
            norm=norm,
            linewidth=0,
            antialiased=True,
            rcount=60,
            ccount=60,
        )
        
        # BOLD axis labels
        ax.set_xlabel(r'$\mathbf{x}$ [m]', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel(r'$\mathbf{t}$ [s]', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_zlabel(r'$\mathbf{u}$ [m]', fontsize=14, fontweight='bold', labelpad=12)
        
        # SAME limits for ALL panels
        ax.set_xlim(global_x_lim)
        ax.set_ylim(global_t_lim)
        ax.set_zlim(global_u_lim)
        
        # View angle
        ax.view_init(elev=elev, azim=azim)
        
        # Pane colors
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_linewidth(0.6)
        ax.yaxis.pane.set_linewidth(0.6)
        ax.zaxis.pane.set_linewidth(0.6)
        
        # Grid
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        
        # BOLD tick labels
        ax.tick_params(axis='x', labelsize=12, pad=4)
        ax.tick_params(axis='y', labelsize=12, pad=4)
        ax.tick_params(axis='z', labelsize=12, pad=6)
        
        for label in ax.xaxis.get_ticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(11)
        for label in ax.yaxis.get_ticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(11)
        for label in ax.zaxis.get_ticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(11)
    
    # ========================================================================
    # Panel labels - CENTERED ABOVE each subplot
    # ========================================================================
    label_y_top = 0.94
    label_y_bot = 0.48
    
    label_positions = [
        (0.05 + pw/2, label_y_top),   # (a)
        (0.48 + pw/2, label_y_top),   # (b)
        (0.05 + pw/2, label_y_bot),   # (c)
        (0.48 + pw/2, label_y_bot),   # (d)
    ]
    
    for idx, (lx, ly) in enumerate(label_positions):
        fig.text(lx, ly, panel_labels[idx], fontsize=18, fontweight='bold',
                 ha='center', va='bottom')
    
    # ========================================================================
    # Colorbar - on the RIGHT with good spacing
    # ========================================================================
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    sm = ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\mathbf{u(x,t)}$ [m]', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # ========================================================================
    # Save Figure
    # ========================================================================
    for fmt in ['eps', 'pdf', 'png']:
        outpath = FIGS_DIR / f"spatiotemporal_evolution.{fmt}"
        fig.savefig(outpath, format=fmt, bbox_inches='tight', facecolor='white')
        print(f"Saved: {outpath}")
    
    plt.close(fig)
    
    # ========================================================================
    # Generate Statistics Report
    # ========================================================================
    stats_file = STATS_DIR / "spatiotemporal_evolution.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("3D SPATIOTEMPORAL VISUALIZATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Data directory: {DATA_DIR.resolve()}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VISUALIZATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Colormap: hot (heat-like)\n")
        f.write(f"  Background: white (EPS compliant)\n")
        f.write(f"  View angles: elevation = {elev} deg, azimuth = {azim} deg\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("AXIS LIMITS (SAME FOR ALL PANELS)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  x-axis: [{global_x_lim[0]}, {global_x_lim[1]}] m\n")
        f.write(f"  t-axis: [{global_t_lim[0]}, {global_t_lim[1]}] s\n")
        f.write(f"  u-axis: [{global_u_lim[0]}, {global_u_lim[1]}] m\n\n")
        
        for idx, data in enumerate(all_data):
            f.write("-" * 80 + "\n")
            f.write(f"PANEL ({chr(97+idx)}): {CASES[idx]['label'].upper()}\n")
            f.write("-" * 80 + "\n")
            
            if data is None:
                f.write("  Data not available\n\n")
                continue
            
            f.write(f"  Scenario: {data['scenario']}\n")
            f.write(f"  Grid points (nx): {data['nx']}\n")
            f.write(f"  Grid spacing (dx): {data['dx']:.6f} m\n")
            f.write(f"  Spatial domain: x in [{data['x'].min():.1f}, {data['x'].max():.1f}] m\n")
            f.write(f"  Temporal domain: t in [0, {data['t'].max():.1f}] s\n")
            f.write(f"  Time snapshots: {len(data['t'])}\n")
            f.write(f"  Physical parameters:\n")
            f.write(f"    Dispersion mu = {data['mu']}\n")
            f.write(f"    Nonlinearity epsilon = {data['epsilon']}\n\n")
            
            f.write(f"  Wave amplitude statistics:\n")
            f.write(f"    Initial max: {data['u'][0].max():.6f} m\n")
            f.write(f"    Final max: {data['u'][-1].max():.6f} m\n")
            f.write(f"    Global max: {data['u'].max():.6f} m\n")
            f.write(f"    Global min: {data['u'].min():.6f} m\n")
            
            u0 = data['u'][0]
            half_max = u0.max() / 2
            above_half = np.where(u0 > half_max)[0]
            if len(above_half) > 1:
                fwhm = (above_half[-1] - above_half[0]) * data['dx']
                f.write(f"    Initial FWHM: {fwhm:.4f} m\n")
            
            peak_x0 = data['x'][np.argmax(data['u'][0])]
            peak_xf = data['x'][np.argmax(data['u'][-1])]
            propagation = peak_xf - peak_x0
            avg_velocity = propagation / data['t'].max()
            
            f.write(f"    Peak propagation: {propagation:.4f} m\n")
            f.write(f"    Average velocity: {avg_velocity:.6f} m/s\n")
            
            A = data['u'][0].max()
            v_theo = data['epsilon'] * A / 3.0
            f.write(f"    Theoretical velocity (epsilon*A/3): {v_theo:.6f} m/s\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n")
        f.write("1. Panel (a): Single soliton propagates rightward maintaining\n")
        f.write("   its sech^2 profile shape - characteristic of KdV solitons.\n\n")
        f.write("2. Panel (b): Two equal-amplitude solitons travel at the same\n")
        f.write("   velocity, demonstrating the v ~ A relationship.\n\n")
        f.write("3. Panel (c): Soliton collision - taller soliton overtakes\n")
        f.write("   shorter one, both emerge unchanged post-collision.\n\n")
        f.write("4. Panel (d): Three-soliton interaction shows multiple\n")
        f.write("   collisions with phase shifts but preserved amplitudes.\n\n")
        f.write("5. Consistent axis limits and single colorbar enable\n")
        f.write("   direct comparison across all panels.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {stats_file}")


if __name__ == "__main__":
    main()

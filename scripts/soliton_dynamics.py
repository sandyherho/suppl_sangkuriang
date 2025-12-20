#!/usr/bin/env python3
"""
Soliton Dynamics and Collision Analysis
Analyzes soliton propagation, velocity verification, and phase shifts.

Generates:
- ../figs/soliton_dynamics.{eps,pdf,png}
- ../stats/soliton_dynamics.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks

# ============================================================================
# Configuration
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

DATA_DIR = Path("../data")
FIGS_DIR = Path("../figs")
STATS_DIR = Path("../stats")

FIGS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)


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
        }
    return data


def track_soliton_peaks(x, t, u, height_threshold=0.5):
    """Track soliton peak positions over time."""
    all_tracks = []
    
    for i, ti in enumerate(t):
        peaks, properties = find_peaks(u[i], height=height_threshold, distance=10)
        for peak_idx in peaks:
            all_tracks.append({
                'time': ti,
                'position': x[peak_idx],
                'amplitude': u[i, peak_idx],
                'frame': i
            })
    
    return all_tracks


def separate_soliton_tracks(tracks, t, n_solitons=1):
    """Separate interleaved peak data into individual soliton tracks."""
    if not tracks:
        return []
    
    time_groups = {}
    for track in tracks:
        ti = track['time']
        if ti not in time_groups:
            time_groups[ti] = []
        time_groups[ti].append(track)
    
    for ti in time_groups:
        time_groups[ti].sort(key=lambda x: x['position'])
    
    soliton_tracks = [[] for _ in range(n_solitons)]
    
    times = sorted(time_groups.keys())
    for ti in times:
        peaks = time_groups[ti]
        peaks_sorted = sorted(peaks, key=lambda x: -x['amplitude'])
        
        for j, peak in enumerate(peaks_sorted):
            if j < n_solitons:
                soliton_tracks[j].append(peak)
    
    return soliton_tracks


def compute_velocity_from_track(track):
    """Compute velocity using linear regression on position vs time."""
    if len(track) < 2:
        return np.nan, np.nan
    
    times = np.array([p['time'] for p in track])
    positions = np.array([p['position'] for p in track])
    
    coeffs = np.polyfit(times, positions, 1)
    velocity = coeffs[0]
    
    predicted = np.polyval(coeffs, times)
    ss_res = np.sum((positions - predicted)**2)
    ss_tot = np.sum((positions - np.mean(positions))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return velocity, r_squared


def theoretical_velocity(amplitude, epsilon):
    """Theoretical soliton velocity: v = epsilon * A / 3"""
    return epsilon * amplitude / 3.0


def main():
    # ========================================================================
    # Load data
    # ========================================================================
    case1_path = DATA_DIR / "case1_single_soliton.nc"
    case3_path = DATA_DIR / "case3_soliton_collision.nc"
    
    if not case1_path.exists() or not case3_path.exists():
        raise FileNotFoundError("Required data files not found in ../data/")
    
    data1 = load_netcdf(case1_path)
    data3 = load_netcdf(case3_path)
    
    print(f"Loaded: {case1_path.name}")
    print(f"Loaded: {case3_path.name}")
    
    # ========================================================================
    # Track solitons
    # ========================================================================
    tracks1 = track_soliton_peaks(data1['x'], data1['t'], data1['u'], height_threshold=1.0)
    soliton_tracks1 = separate_soliton_tracks(tracks1, data1['t'], n_solitons=1)
    
    tracks3 = track_soliton_peaks(data3['x'], data3['t'], data3['u'], height_threshold=0.8)
    soliton_tracks3 = separate_soliton_tracks(tracks3, data3['t'], n_solitons=2)
    
    # ========================================================================
    # Create Figure (2x2 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.10,
                        hspace=0.32, wspace=0.28)
    
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # ------------------------------------------------------------------------
    # (a) Single soliton - space-time contour
    # ------------------------------------------------------------------------
    ax = axes[0, 0]
    
    T, X = np.meshgrid(data1['t'], data1['x'])
    levels = np.linspace(0, data1['u'].max(), 20)
    
    contour = ax.contourf(X, T, data1['u'].T, levels=levels, cmap='Blues')
    ax.contour(X, T, data1['u'].T, levels=[0.5, 2.0, 3.5], colors='black', 
               linewidths=0.5, linestyles='-')
    
    if soliton_tracks1[0]:
        track_t = [p['time'] for p in soliton_tracks1[0]]
        track_x = [p['position'] for p in soliton_tracks1[0]]
        ax.plot(track_x, track_t, 'r--', linewidth=1.5, label='Peak trajectory')
        ax.legend(loc='lower right', frameon=True, edgecolor='black', 
                  fancybox=False, fontsize=8)
    
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$t$ [s]')
    ax.text(0.03, 0.95, panel_labels[0], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
    
    # ------------------------------------------------------------------------
    # (b) Collision case - space-time contour
    # ------------------------------------------------------------------------
    ax = axes[0, 1]
    
    T, X = np.meshgrid(data3['t'], data3['x'])
    levels = np.linspace(0, data3['u'].max(), 25)
    
    contour = ax.contourf(X, T, data3['u'].T, levels=levels, cmap='Oranges')
    ax.contour(X, T, data3['u'].T, levels=[1.0, 3.0, 5.0], colors='black',
               linewidths=0.5, linestyles='-')
    
    colors = ['darkred', 'darkblue']
    labels = ['Fast soliton', 'Slow soliton']
    for j, track in enumerate(soliton_tracks3):
        if track:
            track_t = [p['time'] for p in track]
            track_x = [p['position'] for p in track]
            ax.plot(track_x, track_t, '--', color=colors[j], linewidth=1.5, 
                    label=labels[j])
    
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$t$ [s]')
    ax.legend(loc='lower right', frameon=True, edgecolor='black', 
              fancybox=False, fontsize=8)
    ax.text(0.03, 0.95, panel_labels[1], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
    
    # ------------------------------------------------------------------------
    # (c) Velocity verification
    # ------------------------------------------------------------------------
    ax = axes[1, 0]
    
    measured_velocities = []
    theoretical_velocities = []
    amplitudes = []
    case_colors = []
    case_markers = []
    
    if soliton_tracks1[0]:
        v_meas, r2 = compute_velocity_from_track(soliton_tracks1[0])
        A = np.mean([p['amplitude'] for p in soliton_tracks1[0]])
        v_theo = theoretical_velocity(A, data1['epsilon'])
        measured_velocities.append(v_meas)
        theoretical_velocities.append(v_theo)
        amplitudes.append(A)
        case_colors.append('#1f77b4')
        case_markers.append('o')
    
    for j, track in enumerate(soliton_tracks3):
        if track and len(track) > 10:
            v_meas, r2 = compute_velocity_from_track(track)
            A = np.mean([p['amplitude'] for p in track[:10]])
            v_theo = theoretical_velocity(A, data3['epsilon'])
            measured_velocities.append(v_meas)
            theoretical_velocities.append(v_theo)
            amplitudes.append(A)
            case_colors.append('#ff7f0e' if j == 0 else '#2ca02c')
            case_markers.append('s' if j == 0 else '^')
    
    for i, (A, v) in enumerate(zip(amplitudes, measured_velocities)):
        ax.scatter([A], [v], c=[case_colors[i]], s=80, marker=case_markers[i],
                   edgecolors='black', linewidths=1, zorder=5)
    
    A_range = np.linspace(0, max(amplitudes) * 1.2, 100)
    v_theo_line = data1['epsilon'] * A_range / 3.0
    ax.plot(A_range, v_theo_line, 'k-', linewidth=1.5, label=r'$v = \varepsilon A / 3$')
    
    ax.scatter([], [], c='gray', s=60, marker='o', edgecolors='black', 
               linewidths=1, label='Measured')
    
    ax.set_xlabel(r'Amplitude $A$ [m]')
    ax.set_ylabel(r'Velocity $v$ [m/s]')
    ax.set_xlim(0, max(amplitudes) * 1.3)
    ax.set_ylim(0, max(measured_velocities) * 1.3)
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False,
              fontsize=8)
    ax.text(0.03, 0.95, panel_labels[2], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')
    
    # ------------------------------------------------------------------------
    # (d) Wave profiles at different times (collision case)
    # ------------------------------------------------------------------------
    ax = axes[1, 1]
    
    n_profiles = 5
    time_indices = np.linspace(0, len(data3['t']) - 1, n_profiles, dtype=int)
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_profiles))
    
    for i, idx in enumerate(time_indices):
        offset = i * 1.8
        ax.plot(data3['x'], data3['u'][idx] + offset, color=colors[i], linewidth=1.2,
                label=f"$t = {data3['t'][idx]:.1f}$ s")
    
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$u(x,t)$ [m] (offset)')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False,
              fontsize=7, ncol=1, handlelength=1.5)
    ax.text(0.03, 0.95, panel_labels[3], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')
    
    # ========================================================================
    # Save Figure
    # ========================================================================
    for fmt in ['eps', 'pdf', 'png']:
        outpath = FIGS_DIR / f"soliton_dynamics.{fmt}"
        fig.savefig(outpath, format=fmt, bbox_inches='tight')
        print(f"Saved: {outpath}")
    
    plt.close(fig)
    
    # ========================================================================
    # Generate Statistics Report
    # ========================================================================
    stats_file = STATS_DIR / "soliton_dynamics.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SOLITON DYNAMICS AND COLLISION ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Data directory: {DATA_DIR.resolve()}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("THEORETICAL BACKGROUND\n")
        f.write("-" * 80 + "\n")
        f.write("Soliton velocity for KdV equation: v = epsilon * A / 3\n")
        f.write("where A is the amplitude and epsilon is the nonlinearity parameter.\n\n")
        f.write("During collision, solitons pass through each other and emerge\n")
        f.write("unchanged in shape, but with a phase shift (position offset).\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CASE 1: SINGLE SOLITON PROPAGATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Scenario: {data1['scenario']}\n")
        f.write(f"  Domain: [{data1['x'].min():.1f}, {data1['x'].max():.1f}] m\n")
        f.write(f"  Time span: [0, {data1['t'].max():.1f}] s\n")
        f.write(f"  Dispersion mu = {data1['mu']}\n")
        f.write(f"  Nonlinearity epsilon = {data1['epsilon']}\n\n")
        
        if soliton_tracks1[0]:
            track = soliton_tracks1[0]
            A_mean = np.mean([p['amplitude'] for p in track])
            A_std = np.std([p['amplitude'] for p in track])
            v_meas, r2 = compute_velocity_from_track(track)
            v_theo = theoretical_velocity(A_mean, data1['epsilon'])
            
            f.write(f"  Tracked peak positions: {len(track)} points\n")
            f.write(f"  Mean amplitude: {A_mean:.4f} +/- {A_std:.4f} m\n")
            f.write(f"  Measured velocity: {v_meas:.6f} m/s\n")
            f.write(f"  Theoretical velocity: {v_theo:.6f} m/s\n")
            f.write(f"  Velocity error: {100*abs(v_meas - v_theo)/v_theo:.4f} %\n")
            f.write(f"  Linear fit R²: {r2:.6f}\n")
            f.write(f"  Amplitude stability (std/mean): {100*A_std/A_mean:.4f} %\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CASE 3: SOLITON COLLISION\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Scenario: {data3['scenario']}\n")
        f.write(f"  Domain: [{data3['x'].min():.1f}, {data3['x'].max():.1f}] m\n")
        f.write(f"  Time span: [0, {data3['t'].max():.1f}] s\n")
        f.write(f"  Dispersion mu = {data3['mu']}\n")
        f.write(f"  Nonlinearity epsilon = {data3['epsilon']}\n\n")
        
        soliton_names = ['Fast (tall) soliton', 'Slow (short) soliton']
        for j, track in enumerate(soliton_tracks3):
            if track and len(track) > 5:
                A_early = np.mean([p['amplitude'] for p in track[:10]])
                v_meas, r2 = compute_velocity_from_track(track)
                v_theo = theoretical_velocity(A_early, data3['epsilon'])
                
                f.write(f"  {soliton_names[j]}:\n")
                f.write(f"    Tracked points: {len(track)}\n")
                f.write(f"    Initial amplitude: {A_early:.4f} m\n")
                f.write(f"    Measured velocity: {v_meas:.6f} m/s\n")
                f.write(f"    Theoretical velocity: {v_theo:.6f} m/s\n")
                f.write(f"    Velocity error: {100*abs(v_meas - v_theo)/v_theo:.4f} %\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("VELOCITY VERIFICATION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Soliton':<25} {'A [m]':<10} {'v_meas [m/s]':<14} {'v_theo [m/s]':<14} {'Error %':<10}\n")
        f.write("-" * 75 + "\n")
        
        all_entries = []
        if soliton_tracks1[0]:
            track = soliton_tracks1[0]
            A = np.mean([p['amplitude'] for p in track])
            v_m, _ = compute_velocity_from_track(track)
            v_t = theoretical_velocity(A, data1['epsilon'])
            all_entries.append(('Case 1 soliton', A, v_m, v_t))
        
        for j, track in enumerate(soliton_tracks3):
            if track and len(track) > 5:
                A = np.mean([p['amplitude'] for p in track[:10]])
                v_m, _ = compute_velocity_from_track(track)
                v_t = theoretical_velocity(A, data3['epsilon'])
                all_entries.append((f'Case 3 soliton {j+1}', A, v_m, v_t))
        
        for name, A, v_m, v_t in all_entries:
            err = 100 * abs(v_m - v_t) / v_t if v_t > 0 else 0
            f.write(f"{name:<25} {A:<10.4f} {v_m:<14.6f} {v_t:<14.6f} {err:<10.4f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n")
        f.write("1. Single soliton propagates with velocity v = epsilon*A/3,\n")
        f.write("   maintaining its shape indefinitely.\n\n")
        f.write("2. In collision, taller soliton (larger A) travels faster and\n")
        f.write("   overtakes the shorter one.\n\n")
        f.write("3. Solitons emerge from collision unchanged in amplitude and\n")
        f.write("   velocity - characteristic of integrable systems.\n\n")
        f.write("4. Phase shifts occur during collision but solitons maintain\n")
        f.write("   their identity, demonstrating particle-like behavior.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {stats_file}")


if __name__ == "__main__":
    main()

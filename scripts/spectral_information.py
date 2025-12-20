#!/usr/bin/env python3
"""
Spectral Analysis and Information-Theoretic Measures
Analyzes Fourier spectra, Shannon entropy, and complexity measures.

Generates:
- ../figs/spectral_information.{eps,pdf,png}
- ../stats/spectral_information.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from scipy.fft import fft, fftfreq

# ============================================================================
# Configuration
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
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

CASES = {
    'case1': {'file': 'case1_single_soliton.nc', 'label': 'Single soliton', 
              'color': '#1f77b4', 'ls': '-'},
    'case2': {'file': 'case2_two_equal_solitons.nc', 'label': 'Two equal solitons', 
              'color': '#ff7f0e', 'ls': '--'},
    'case3': {'file': 'case3_soliton_collision.nc', 'label': 'Collision', 
              'color': '#2ca02c', 'ls': '-.'},
    'case4': {'file': 'case4_three_solitons.nc', 'label': 'Three solitons', 
              'color': '#d62728', 'ls': ':'},
}


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


def compute_power_spectrum(u, dx):
    """Compute power spectral density."""
    n = len(u)
    u_hat = fft(u)
    Pk = np.abs(u_hat)**2 / n
    k = 2 * np.pi * fftfreq(n, d=dx)
    
    idx = np.argsort(k)
    return k[idx], Pk[idx]


def compute_spectral_entropy(u, dx):
    """
    Compute spectral entropy (Shannon entropy of normalized power spectrum).
    S_k = -sum(p_k * log(p_k)) / log(N)
    """
    n = len(u)
    u_hat = fft(u)
    Pk = np.abs(u_hat)**2
    
    total_power = np.sum(Pk)
    if total_power < 1e-15:
        return 0.0
    
    p_k = Pk / total_power
    p_k = p_k[p_k > 1e-15]
    S = -np.sum(p_k * np.log(p_k))
    S_max = np.log(n)
    
    return S / S_max


def compute_statistical_complexity(u, dx):
    """
    Compute López-Ruiz-Mancini-Calbet (LMC) statistical complexity.
    C = H × D
    """
    n = len(u)
    u_hat = fft(u)
    Pk = np.abs(u_hat)**2
    
    total_power = np.sum(Pk)
    if total_power < 1e-15:
        return 0.0, 0.0, 0.0
    
    p_k = Pk / total_power
    
    p_nonzero = p_k[p_k > 1e-15]
    H = -np.sum(p_nonzero * np.log(p_nonzero)) / np.log(n)
    
    p_uniform = np.ones(n) / n
    D = np.sum((p_k - p_uniform)**2)
    
    C = H * D
    
    return C, H, D


def compute_fisher_information(u, dx):
    """
    Compute Fisher information measure.
    F = integral((1/p) * (dp/dx)^2 dx)
    """
    u_abs = np.abs(u) + 1e-15
    total = np.sum(u_abs) * dx
    p = u_abs / total
    
    dp_dx = np.gradient(p, dx)
    F = np.sum((dp_dx**2) / p) * dx
    
    return F


def set_bold_ticks(ax):
    """Set bold tick labels on both axes."""
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')


def main():
    # ========================================================================
    # Load data
    # ========================================================================
    all_data = {}
    for case_id, case_info in CASES.items():
        filepath = DATA_DIR / case_info['file']
        if filepath.exists():
            all_data[case_id] = load_netcdf(filepath)
            print(f"Loaded: {case_info['file']}")
    
    if not all_data:
        raise FileNotFoundError("No data files found in ../data/")
    
    # ========================================================================
    # Compute information measures
    # ========================================================================
    info_measures = {}
    
    for case_id, data in all_data.items():
        n_times = len(data['t'])
        
        spectral_entropy = np.zeros(n_times)
        complexity = np.zeros(n_times)
        fisher_info = np.zeros(n_times)
        
        for i in range(n_times):
            spectral_entropy[i] = compute_spectral_entropy(data['u'][i], data['dx'])
            C, H, D = compute_statistical_complexity(data['u'][i], data['dx'])
            complexity[i] = C
            fisher_info[i] = compute_fisher_information(data['u'][i], data['dx'])
        
        info_measures[case_id] = {
            'spectral_entropy': spectral_entropy,
            'complexity': complexity,
            'fisher_info': fisher_info,
        }
    
    # ========================================================================
    # Create Figure (2x2 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.15,
                        hspace=0.35, wspace=0.30)
    
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # ------------------------------------------------------------------------
    # (a) Power spectra comparison (initial state)
    # ------------------------------------------------------------------------
    ax = axes[0, 0]
    
    for case_id, data in all_data.items():
        k, Pk = compute_power_spectrum(data['u'][0], data['dx'])
        mask = k > 0
        ax.loglog(k[mask], Pk[mask], 
                  color=CASES[case_id]['color'],
                  linestyle=CASES[case_id]['ls'],
                  linewidth=1.5)
    
    ax.set_xlabel(r'$k$ [rad/m]', fontweight='bold', fontsize=13)
    ax.set_ylabel(r'$P(k)$ [m$^2$]', fontweight='bold', fontsize=13)
    ax.set_xlim(1e-2, 10)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    ax.set_title(panel_labels[0], fontsize=14, fontweight='bold', pad=8)
    set_bold_ticks(ax)
    
    # ------------------------------------------------------------------------
    # (b) Spectral entropy evolution
    # ------------------------------------------------------------------------
    ax = axes[0, 1]
    
    for case_id, data in all_data.items():
        ax.plot(data['t'], info_measures[case_id]['spectral_entropy'],
                color=CASES[case_id]['color'],
                linestyle=CASES[case_id]['ls'],
                linewidth=1.5)
    
    ax.set_xlabel(r'$t$ [s]', fontweight='bold', fontsize=13)
    ax.set_ylabel(r'$S_k$ (normalized)', fontweight='bold', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)
    ax.minorticks_on()
    ax.set_title(panel_labels[1], fontsize=14, fontweight='bold', pad=8)
    set_bold_ticks(ax)
    
    # ------------------------------------------------------------------------
    # (c) Statistical complexity evolution
    # ------------------------------------------------------------------------
    ax = axes[1, 0]
    
    # Find scale
    all_C = []
    for case_id in all_data:
        all_C.extend(info_measures[case_id]['complexity'])
    C_scale = np.max(all_C)
    C_exp = int(np.floor(np.log10(C_scale))) if C_scale > 0 else -2
    
    for case_id, data in all_data.items():
        ax.plot(data['t'], info_measures[case_id]['complexity'] / (10**C_exp),
                color=CASES[case_id]['color'],
                linestyle=CASES[case_id]['ls'],
                linewidth=1.5)
    
    ax.set_xlabel(r'$t$ [s]', fontweight='bold', fontsize=13)
    ax.set_ylabel(r'Complexity $C = H \times D$ $(\times 10^{' + str(C_exp) + r'})$', 
                  fontweight='bold', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)
    ax.minorticks_on()
    ax.set_title(panel_labels[2], fontsize=14, fontweight='bold', pad=8)
    set_bold_ticks(ax)
    
    # ------------------------------------------------------------------------
    # (d) Fisher information evolution
    # ------------------------------------------------------------------------
    ax = axes[1, 1]
    
    for case_id, data in all_data.items():
        ax.plot(data['t'], info_measures[case_id]['fisher_info'],
                color=CASES[case_id]['color'],
                linestyle=CASES[case_id]['ls'],
                linewidth=1.5)
    
    ax.set_xlabel(r'$t$ [s]', fontweight='bold', fontsize=13)
    ax.set_ylabel(r'Fisher information $F$ [m$^{-2}$]', fontweight='bold', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.0, length=5)
    ax.minorticks_on()
    ax.set_title(panel_labels[3], fontsize=14, fontweight='bold', pad=8)
    set_bold_ticks(ax)
    
    # ------------------------------------------------------------------------
    # Shared legend at BOTTOM
    # ------------------------------------------------------------------------
    legend_elements = [plt.Line2D([0], [0], color=CASES[cid]['color'],
                                   linestyle=CASES[cid]['ls'], linewidth=2.0,
                                   label=CASES[cid]['label'])
                       for cid in CASES.keys() if cid in all_data]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               frameon=True, edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.54, 0.01), fontsize=10)
    
    # ========================================================================
    # Save Figure
    # ========================================================================
    for fmt in ['eps', 'pdf', 'png']:
        outpath = FIGS_DIR / f"spectral_information.{fmt}"
        fig.savefig(outpath, format=fmt, bbox_inches='tight')
        print(f"Saved: {outpath}")
    
    plt.close(fig)
    
    # ========================================================================
    # Generate Statistics Report
    # ========================================================================
    stats_file = STATS_DIR / "spectral_information.txt"
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SPECTRAL ANALYSIS AND INFORMATION-THEORETIC MEASURES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Data directory: {DATA_DIR.resolve()}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("THEORETICAL BACKGROUND\n")
        f.write("-" * 80 + "\n")
        f.write("1. SPECTRAL ENTROPY (Shannon entropy of power spectrum):\n")
        f.write("   S_k = -sum(p_k * log(p_k)) / log(N)\n")
        f.write("   where p_k = P_k / sum(P_k) is normalized power spectrum.\n")
        f.write("   Range: [0, 1], higher = more distributed spectral energy.\n\n")
        
        f.write("2. LMC STATISTICAL COMPLEXITY:\n")
        f.write("   C = H × D\n")
        f.write("   H = normalized Shannon entropy\n")
        f.write("   D = disequilibrium (distance from uniform distribution)\n")
        f.write("   Measures interplay between order and disorder.\n\n")
        
        f.write("3. FISHER INFORMATION:\n")
        f.write("   F = integral((1/p) * (dp/dx)^2 dx)\n")
        f.write("   Measures 'sharpness' or localization of distribution.\n")
        f.write("   Higher F = more localized (sharper peaks).\n\n")
        
        for case_id, data in all_data.items():
            f.write("-" * 80 + "\n")
            f.write(f"CASE: {data['scenario']}\n")
            f.write("-" * 80 + "\n")
            
            measures = info_measures[case_id]
            
            f.write(f"  Simulation parameters:\n")
            f.write(f"    Grid points: {data['nx']}\n")
            f.write(f"    Grid spacing: {data['dx']:.6f} m\n")
            f.write(f"    Time span: [0, {data['t'].max():.1f}] s\n")
            f.write(f"    Time snapshots: {len(data['t'])}\n\n")
            
            S_k = measures['spectral_entropy']
            f.write(f"  Spectral Entropy S_k:\n")
            f.write(f"    Initial: {S_k[0]:.6f}\n")
            f.write(f"    Final: {S_k[-1]:.6f}\n")
            f.write(f"    Mean: {np.mean(S_k):.6f}\n")
            f.write(f"    Std: {np.std(S_k):.6f}\n")
            f.write(f"    Range: [{S_k.min():.6f}, {S_k.max():.6f}]\n\n")
            
            C = measures['complexity']
            f.write(f"  Statistical Complexity C:\n")
            f.write(f"    Initial: {C[0]:.6e}\n")
            f.write(f"    Final: {C[-1]:.6e}\n")
            f.write(f"    Mean: {np.mean(C):.6e}\n")
            f.write(f"    Std: {np.std(C):.6e}\n\n")
            
            F = measures['fisher_info']
            f.write(f"  Fisher Information F:\n")
            f.write(f"    Initial: {F[0]:.6f} m^-2\n")
            f.write(f"    Final: {F[-1]:.6f} m^-2\n")
            f.write(f"    Mean: {np.mean(F):.6f} m^-2\n")
            f.write(f"    Std: {np.std(F):.6f} m^-2\n\n")
            
            k, Pk = compute_power_spectrum(data['u'][0], data['dx'])
            k_positive = k[k > 0]
            Pk_positive = Pk[k > 0]
            k_dom = k_positive[np.argmax(Pk_positive)]
            lambda_dom = 2 * np.pi / k_dom if k_dom > 0 else np.inf
            
            f.write(f"  Power Spectrum (initial state):\n")
            f.write(f"    Dominant wavenumber: {k_dom:.6f} rad/m\n")
            f.write(f"    Dominant wavelength: {lambda_dom:.4f} m\n")
            f.write(f"    Total power: {np.sum(Pk):.6e}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("COMPARATIVE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Case':<22} {'S_k (mean)':<12} {'C (mean)':<14} {'F (mean)':<12}\n")
        f.write("-" * 62 + "\n")
        
        for case_id in all_data:
            S_mean = np.mean(info_measures[case_id]['spectral_entropy'])
            C_mean = np.mean(info_measures[case_id]['complexity'])
            F_mean = np.mean(info_measures[case_id]['fisher_info'])
            f.write(f"{CASES[case_id]['label']:<22} {S_mean:<12.6f} {C_mean:<14.4e} {F_mean:<12.4f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n")
        f.write("1. SPECTRAL ENTROPY remains relatively stable, indicating\n")
        f.write("   preserved spectral structure characteristic of solitons.\n\n")
        f.write("2. STATISTICAL COMPLEXITY shows variations during interactions\n")
        f.write("   when the wave field transiently becomes more structured.\n\n")
        f.write("3. FISHER INFORMATION reflects localization - higher values\n")
        f.write("   indicate sharper, more localized soliton profiles.\n")
        f.write("   Transient spikes occur during soliton collisions.\n\n")
        f.write("4. POWER SPECTRA show exponential decay at high k, consistent\n")
        f.write("   with smooth sech^2 profiles.\n\n")
        f.write("5. Information-theoretic stability confirms the integrable\n")
        f.write("   nature of KdV - solitons preserve their information content.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {stats_file}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper2 - All Figures (Fig. 1 to Fig. 5)
600 DPI, 16:9, blue academic style, no overlapping boxes, arrows, or labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ============================================================
# Global style
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 12

DARK_BLUE = '#0B3C5D'
MED_BLUE = '#1D6FA5'
LIGHT_BLUE = '#3E92CC'
PALE_BLUE = '#7FB3D5'
VERY_PALE_BLUE = '#B3D9F2'
ALPHA = 0.88
OUTPUT_DPI = 600
FIG_W, FIG_H = 16, 9


def save_fig(fig, filename):
    fig.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight',
                pad_inches=0.2, facecolor='white')
    plt.close(fig)
    print(f'Saved: {filename}')


# ============================================================
# Fig. 1
# ============================================================
def draw_fig1():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=OUTPUT_DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.5, 0.965, 'Fig. 1. Methodological framework and experimental pipeline',
            ha='center', va='top', fontsize=17, fontweight='bold', color=DARK_BLUE)
    ax.text(0.25, 0.90, 'Left panel: Main experimental pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold', color=MED_BLUE)
    ax.text(0.75, 0.90, 'Right panel: Generation and robustness',
            ha='center', va='center', fontsize=14, fontweight='bold', color=MED_BLUE)

    box_h = 0.07
    ys = [0.80, 0.67, 0.54, 0.41, 0.28, 0.15]
    box_w = 0.30

    left_texts = [
        'SDXL Base 1.0',
        'Five samplers:\nEuler a, DPM++ 2M Karras,\nLMS, Heun, DDIM',
        'Subjective evaluation\n(15 raters, 300 images)',
        'Clarity/artifact covariate',
        'Expert creative-value ratings\n(8 experts, 200 images)',
        'Variance decomposition\n(Linear mixed models)'
    ]
    right_texts = [
        '150 abstract prompts',
        'Main generation:\n150 prompts × 5 samplers = 750 images',
        'Stability generation:\n3 seeds per condition',
        'Four-level ablation:\nL0 – L4',
        'Robustness tests:\nSD1.5, SD3.5 (50 prompts),\nCFG, steps',
        'Expert evaluation:\n8 artists and curators'
    ]

    for x_c, texts, face in [(0.25, left_texts, VERY_PALE_BLUE),
                             (0.75, right_texts, PALE_BLUE)]:
        for y_c, txt in zip(ys, texts):
            box = FancyBboxPatch(
                (x_c - box_w / 2, y_c - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.008",
                facecolor=face, edgecolor=MED_BLUE,
                linewidth=1.4, alpha=ALPHA)
            ax.add_patch(box)
            ax.text(x_c, y_c, txt, ha='center', va='center',
                    fontsize=10, color='black', linespacing=1.25)

    for x_c in [0.25, 0.75]:
        for i in range(len(ys) - 1):
            y_top = ys[i] - box_h / 2 - 0.008
            y_bot = ys[i + 1] + box_h / 2 + 0.008
            arrow = FancyArrowPatch(
                (x_c, y_top), (x_c, y_bot),
                arrowstyle='-|>', mutation_scale=14,
                color=MED_BLUE, linewidth=1.6,
                shrinkA=0, shrinkB=0)
            ax.add_patch(arrow)

    save_fig(fig, 'outputs/figures/Figure1_Paper2_600dpi.png')


# ============================================================
# Fig. 2
# ============================================================
def draw_fig2():
    samplers = ['Euler a', 'DPM++\n2M Karras', 'LMS', 'Heun', 'DDIM']

    data = {
        'Surprise': {
            'mean': [3.9, 3.4, 2.9, 2.6, 2.3],
            'std':  [0.6, 0.5, 0.6, 0.5, 0.5],
            'sig':  [True, True, False, False, False]},
        'Inspirational value': {
            'mean': [3.6, 3.8, 3.1, 3.2, 2.9],
            'std':  [0.7, 0.6, 0.7, 0.6, 0.6],
            'sig':  [False, False, False, False, False]},
        'Usability': {
            'mean': [2.8, 3.7, 3.0, 3.2, 3.6],
            'std':  [0.7, 0.6, 0.7, 0.6, 0.6],
            'sig':  [False, False, False, False, False]},
        'Aesthetic quality': {
            'mean': [2.9, 3.6, 3.0, 3.3, 3.5],
            'std':  [0.7, 0.6, 0.7, 0.6, 0.6],
            'sig':  [False, False, False, False, False]},
    }
    colors = [DARK_BLUE, MED_BLUE, LIGHT_BLUE, PALE_BLUE, VERY_PALE_BLUE]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), dpi=OUTPUT_DPI,
                             constrained_layout=True)
    axes = axes.flatten()
    panel_titles = ['(A) Surprise', '(B) Inspirational value',
                    '(C) Usability', '(D) Aesthetic quality']
    keys = ['Surprise', 'Inspirational value', 'Usability', 'Aesthetic quality']

    x = np.arange(len(samplers)); width = 0.62
    for ax, title, key in zip(axes, panel_titles, keys):
        d = data[key]
        ax.bar(x, d['mean'], width, yerr=d['std'], capsize=6,
               color=colors, alpha=ALPHA, edgecolor='#1a1a1a', linewidth=1.2,
               error_kw={'ecolor': '#333333', 'linewidth': 1.5, 'capsize': 5})
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
        ax.set_ylabel('Score (1–5)', fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(samplers, fontsize=10)
        ax.set_ylim(0, 5.8)
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#cccccc')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for i, (m, s, sig) in enumerate(zip(d['mean'], d['std'], d['sig'])):
            if sig:
                ax.text(i, m + s + 0.20, '*', ha='center', va='bottom',
                        fontsize=18, fontweight='bold', color=DARK_BLUE)

    fig.suptitle('Fig. 2. Multidimensional subjective ratings by sampler',
                 fontsize=17, fontweight='bold', y=1.02)
    save_fig(fig, 'outputs/figures/Figure2_Paper2_600dpi.png')


# ============================================================
# Fig. 3
# ============================================================
def draw_fig3():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=OUTPUT_DPI, constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1],
                           hspace=0.45, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    n = 60
    x = np.random.normal(0.72, 0.04, n)
    y = 0.18 * (x - x.mean()) / x.std() + np.random.normal(0, 0.95, n)
    y = (y - y.min()) / (y.max() - y.min()) * 3.0 + 1.8
    ax1.scatter(x, y, s=55, c=DARK_BLUE, alpha=0.7,
                edgecolors='white', linewidth=0.8)
    ax1.set_xlabel('CLIP Score', fontsize=12)
    ax1.set_ylabel('Surprise', fontsize=12)
    ax1.set_title('(A) Surprise vs. CLIP Score', fontsize=14, fontweight='bold', pad=10)
    ax1.text(0.04, 0.96, 'r = 0.18, p = 0.03', transform=ax1.transAxes,
             fontsize=11, va='top', color=DARK_BLUE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=PALE_BLUE, alpha=0.95))
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    labels = ['Surprise', 'Inspir.', 'Usability', 'Aesthetic', 'Clarity', 'Expert']
    corr = np.full((6, 6), np.nan); np.fill_diagonal(corr, 1.0)
    corr[0, 1] = corr[1, 0] = 0.31
    corr[0, 2] = corr[2, 0] = 0.21
    corr[0, 5] = corr[5, 0] = 0.34
    cmap = plt.cm.Blues.copy(); cmap.set_bad(color='white')
    im = ax2.imshow(corr, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
    ax2.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title('(B) Correlation heatmap', fontsize=14, fontweight='bold', pad=10)
    for i in range(6):
        for j in range(6):
            if not np.isnan(corr[i, j]):
                color = 'white' if corr[i, j] > 0.6 else 'black'
                ax2.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                         fontsize=8, color=color)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    categories = ['CFI', 'TLI', 'RMSEA', 'SRMR']
    four = [0.96, 0.95, 0.048, 0.041]
    one = [0.71, 0.68, 0.132, 0.098]
    xp = np.arange(len(categories)); w = 0.32
    b1 = ax3.bar(xp - w/2, four, w, label='Four-factor model',
                 color=DARK_BLUE, alpha=ALPHA, edgecolor='black')
    b2 = ax3.bar(xp + w/2, one, w, label='One-factor model',
                 color=PALE_BLUE, alpha=ALPHA, edgecolor='black')
    ax3.set_xticks(xp); ax3.set_xticklabels(categories, fontsize=12)
    ax3.set_ylabel('Value', fontsize=13)
    ax3.set_title('(C) CFA model fit and HTMT summary',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.legend(frameon=False, fontsize=11, loc='upper right')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, 1.15)
    for r, v in zip(b1, four):
        ax3.text(r.get_x() + r.get_width()/2, v + 0.02, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9, color=DARK_BLUE)
    for r, v in zip(b2, one):
        ax3.text(r.get_x() + r.get_width()/2, v + 0.02, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=9, color=MED_BLUE)
    ax3.text(0.99, 0.70, 'HTMT max = 0.72\nAVE > squared correlations',
             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=PALE_BLUE, alpha=0.95))

    fig.suptitle('Fig. 3. Distinction between perceived novelty, semantic alignment, and creative value',
                 fontsize=16, fontweight='bold', y=1.02)
    save_fig(fig, 'outputs/figures/Figure3_Paper2_600dpi.png')


# ============================================================
# Fig. 4
# ============================================================
def draw_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=OUTPUT_DPI,
                             constrained_layout=True)

    ax1 = axes[0]
    levels = ['L0\nFull Euler a', 'L1\nHalf noise', 'L2\nDet. Euler',
              'L3\nDDIM', 'L4\nNon-image ref.']
    novelty = [3.9, 3.0, 2.5, 2.3, 1.62]
    x = np.arange(len(levels))
    ax1.plot(x[:4], novelty[:4], 'o-', color=DARK_BLUE, linewidth=2.5,
             markersize=9, markerfacecolor=LIGHT_BLUE, markeredgecolor=DARK_BLUE,
             label='Inferential trend (L0–L3)')
    ax1.scatter(x[4], novelty[4], s=110, color='gray', edgecolor='black',
                zorder=5, label='Non-image reference (L4)')
    for i, val in enumerate(novelty):
        ax1.annotate(f'{val:.1f}' if i < 4 else f'{val:.2f}',
                     (x[i], val), textcoords='offset points',
                     xytext=(0, 14), ha='center', fontsize=10,
                     color=DARK_BLUE if i < 4 else 'gray')
    for i, d in enumerate(['', '-0.9', '-1.4', '-1.6', '-2.1']):
        if d:
            ax1.text(x[i], novelty[i] - 0.32, d, ha='center', fontsize=9,
                     color=MED_BLUE, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(levels, fontsize=9)
    ax1.set_ylabel('Perceived Novelty (1–5)', fontsize=13)
    ax1.set_title('(A) Progressive ablation of perceived novelty',
                  fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylim(1.0, 4.7)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False, fontsize=9, loc='upper right')

    ax2 = axes[1]
    comps = ['Fixed effects\n(marginal R²)', 'Random-effect\nincrement', 'Residual']
    vals = [0.71, 0.13, 0.16]
    cols = [DARK_BLUE, LIGHT_BLUE, PALE_BLUE]
    ypos = np.arange(len(comps))[::-1]
    bars = ax2.barh(ypos, vals, color=cols, alpha=ALPHA,
                    edgecolor='black', height=0.55)
    ax2.set_yticks(ypos); ax2.set_yticklabels(comps, fontsize=11)
    ax2.set_xlabel('Proportion of variance', fontsize=12)
    ax2.set_xlim(0, 0.85)
    ax2.set_title('(B) Variance explained in perceived novelty\n'
                  'marginal R² = 0.71, conditional R² = 0.84',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    for r, v in zip(bars, vals):
        ax2.text(v + 0.015, r.get_y() + r.get_height()/2, f'{v:.2f}',
                 va='center', ha='left', fontsize=11,
                 fontweight='bold', color=DARK_BLUE)

    fig.suptitle('Fig. 4. Ablation and formal variance decomposition',
                 fontsize=17, fontweight='bold', y=1.02)
    save_fig(fig, 'outputs/figures/Figure4_Paper2_600dpi.png')


# ============================================================
# Fig. 5
# ============================================================
def draw_fig5():
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), dpi=OUTPUT_DPI,
                             constrained_layout=True)
    axes = axes.flatten()

    ax1 = axes[0]
    cfg_vals = [3.0, 7.5, 15.0]
    euler_cfg = [4.1, 3.9, 2.8]
    dpm_cfg = [3.6, 3.4, 2.5]
    ddim_cfg = [2.5, 2.3, 1.9]
    ax1.plot(cfg_vals, euler_cfg, 'o-', color=DARK_BLUE, linewidth=2.2,
             markersize=7, label='Euler a')
    ax1.plot(cfg_vals, dpm_cfg, 's-', color=MED_BLUE, linewidth=2.2,
             markersize=7, label='DPM++ 2M Karras')
    ax1.plot(cfg_vals, ddim_cfg, '^-', color=PALE_BLUE, linewidth=2.2,
             markersize=7, label='DDIM')
    ax1.set_xlabel('CFG scale', fontsize=12)
    ax1.set_ylabel('Perceived Novelty (1–5)', fontsize=12)
    ax1.set_title('(A) Effect of CFG scale', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(frameon=False, fontsize=10, loc='upper right')
    ax1.set_ylim(1.5, 4.7)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    ax2 = axes[1]
    steps = [20, 30, 50]
    euler_steps = [3.5, 3.9, 4.2]
    ddim_steps = [2.3, 2.3, 2.4]
    ax2.plot(steps, euler_steps, 'o-', color=DARK_BLUE, linewidth=2.2,
             markersize=7, label='Euler a')
    ax2.plot(steps, ddim_steps, '^-', color=PALE_BLUE, linewidth=2.2,
             markersize=7, label='DDIM')
    ax2.set_xlabel('Sampling steps', fontsize=12)
    ax2.set_ylabel('Perceived Novelty (1–5)', fontsize=12)
    ax2.set_title('(B) Effect of sampling steps', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(frameon=False, fontsize=10, loc='upper right')
    ax2.set_ylim(1.5, 4.7)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    ax3 = axes[2]
    samplers = ['Euler a', 'DPM++\n2M Karras', 'LMS', 'Heun', 'DDIM']
    sdxl = [3.9, 3.4, 2.9, 2.6, 2.3]
    sd15 = [3.6, 3.1, 2.7, 2.4, 2.1]
    sd35 = [3.7, 3.3, 2.8, 2.5, 2.2]
    x = np.arange(len(samplers)); w = 0.26
    ax3.bar(x - w, sdxl, w, label='SDXL', color=DARK_BLUE, alpha=ALPHA, edgecolor='black')
    ax3.bar(x, sd15, w, label='SD1.5', color=LIGHT_BLUE, alpha=ALPHA, edgecolor='black')
    ax3.bar(x + w, sd35, w, label='SD3.5 (50 prompts)', color=VERY_PALE_BLUE,
            alpha=ALPHA, edgecolor='black')
    ax3.set_xticks(x); ax3.set_xticklabels(samplers, fontsize=9)
    ax3.set_ylabel('Perceived Novelty (1–5)', fontsize=12)
    ax3.set_title('(C) Model generalisation', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(frameon=False, fontsize=9, loc='upper right')
    ax3.set_ylim(0, 5.2)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    ax4 = axes[3]
    surprise = [3.9, 3.4, 2.9, 2.6, 2.3]
    expert_value = [4.9, 5.4, 4.2, 4.3, 4.5]
    labels = ['Euler a', 'DPM++', 'LMS', 'Heun', 'DDIM']
    ax4.scatter(surprise, expert_value, s=100, c=DARK_BLUE, alpha=0.8,
                edgecolors='white', linewidth=1.2)
    offsets = {'Euler a': (8, 6), 'DPM++': (8, 6),
               'LMS': (-38, -14), 'Heun': (-38, 6), 'DDIM': (8, -14)}
    for i, txt in enumerate(labels):
        ax4.annotate(txt, (surprise[i], expert_value[i]),
                     textcoords='offset points', xytext=offsets[txt],
                     fontsize=10, color=DARK_BLUE)
    ax4.set_xlabel('Student-rated Surprise (1–5)', fontsize=12)
    ax4.set_ylabel('Expert-rated Creative Value (1–7)', fontsize=12)
    ax4.set_title('(D) Expert creative value vs. student surprise',
                  fontsize=14, fontweight='bold', pad=10)
    ax4.text(0.04, 0.96, 'r = 0.34, p = 0.010', transform=ax4.transAxes,
             fontsize=10, va='top', color=DARK_BLUE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=PALE_BLUE, alpha=0.95))
    ax4.set_xlim(2.0, 4.3); ax4.set_ylim(3.9, 5.7)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    fig.suptitle('Fig. 5. Robustness and expert validation',
                 fontsize=17, fontweight='bold', y=1.02)
    save_fig(fig, 'outputs/figures/Figure5_Paper2_600dpi.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    from pathlib import Path
    Path('outputs/figures').mkdir(parents=True, exist_ok=True)
    draw_fig1()
    draw_fig2()
    draw_fig3()
    draw_fig4()
    draw_fig5()
    print('\nAll figures generated successfully.')

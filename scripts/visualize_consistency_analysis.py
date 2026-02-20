"""
Visualize consistency analysis results from multi-agent debate experiments.
Generates figures showing:
  1. Agreement / Agent0 accuracy / Flip rate across temperatures (per round)
  2. Consistency-correctness correlation (agreement when correct vs wrong)
  3. Cross-round stability comparison (T=1 vs T=20)
  4. Calibration target comparison diagram
"""

import json
import collections
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"
OUTPUT_DIR = "/export/home3/dazhou/debate-or-vote/out/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]

def get_args():
    parser = argparse.ArgumentParser(description="Visualize consistency analysis")
    parser.add_argument('--model', type=str, default="qwen3-4b", help="Model name")
    parser.add_argument('--data', type=str, default="gsm8k", help="Dataset name")
    parser.add_argument('--data_size', type=int, default=50, help="Data size used in experiment")
    parser.add_argument('--n_agents', type=int, default=3, help="Number of agents")
    parser.add_argument('--target_round', type=int, default=2, help="Target round")
    parser.add_argument('--max_new_tokens', type=int, default=512, help="Max new tokens (512=default, adds _MNT= suffix if != 512)")
    return parser.parse_args()

args = get_args()
MODEL = args.model
DATA = args.data
DATA_SIZE = args.data_size
N_AGENTS = args.n_agents
TARGET_ROUND = args.target_round
MAX_NEW_TOKENS = args.max_new_tokens

# Label suffix for filenames and titles
MNT_SUFFIX = f"_MNT={MAX_NEW_TOKENS}" if MAX_NEW_TOKENS != 512 else ""
MNT_LABEL = f" [MNT={MAX_NEW_TOKENS}]" if MAX_NEW_TOKENS != 512 else ""
SIZE_LABEL = f"N={DATA_SIZE}"


# ─────────────────────────────────────────────
# Helper: load one experiment file
# ─────────────────────────────────────────────
def load_experiment(tt):
    fname = f"{HISTORY_DIR}/{DATA}/{DATA}_{DATA_SIZE}__{MODEL}_N={N_AGENTS}_R=3_TR={TARGET_ROUND}_TT={tt}{MNT_SUFFIX}.jsonl"
    samples = []
    try:
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    except FileNotFoundError:
        print(f"  Warning: file not found: {fname}")
        return None
    return samples


def compute_agreement(answers, n_agents=3):
    """
    Agreement = (number of agents giving the most common answer) / N
    
    Example with 3 agents:
      [120, 120, 120] → 3/3 = 1.0   (full consensus)
      [120, 120, 55]  → 2/3 = 0.667 (majority exists)
      [120, 55, 77]   → 1/3 = 0.333 (no agreement)
      ["", 120, 120]  → 2/3 = 0.667 (empty answers excluded from counting, 
                                       but denominator stays N)
    """
    non_empty = [a for a in answers if a != ""]
    if not non_empty:
        return 0.0
    counter = collections.Counter(non_empty)
    max_count = max(counter.values())
    return max_count / n_agents


# ─────────────────────────────────────────────
# Collect statistics across all temperatures
# ─────────────────────────────────────────────
all_stats = {}  # {temp: {round: {metric: value}}}

for tt in TEMPS:
    samples = load_experiment(tt)
    if samples is None:
        continue

    stats = {r: {'agreement': [], 'agent0_acc': 0, 'agent0_flip': 0,
                 'debate_acc': 0, 'total': 0} for r in range(4)}

    for sample in samples:
        prev_agent0_ans = None
        for r in range(4):
            rd = sample[str(r)]
            answers = rd['final_answers']
            agent_corr = rd['final_answer_iscorr']

            stats[r]['agreement'].append(compute_agreement(answers))
            stats[r]['total'] += 1
            if agent_corr[0]:
                stats[r]['agent0_acc'] += 1
            if rd['debate_answer_iscorr']:
                stats[r]['debate_acc'] += 1

            curr_agent0 = answers[0]
            if prev_agent0_ans is not None and curr_agent0 != prev_agent0_ans:
                stats[r]['agent0_flip'] += 1
            prev_agent0_ans = curr_agent0

    all_stats[tt] = {}
    for r in range(4):
        s = stats[r]
        n = s['total']
        all_stats[tt][r] = {
            'agreement': np.mean(s['agreement']),
            'agent0_acc': s['agent0_acc'] / n if n else 0,
            'agent0_flip': s['agent0_flip'] / n if n else 0,
            'debate_acc': s['debate_acc'] / n if n else 0,
        }


# ═══════════════════════════════════════════════
# Figure 1: Three-panel: Agreement / Agent0 Acc / Flip Rate vs Temperature
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

colors_round = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
markers = ['o', 's', '^', 'D']
round_labels = ['Round 0 (Initial)', 'Round 1 (Debate)', 'Round 2 (Target)', 'Round 3 (Post-target)']

temps_sorted = sorted(all_stats.keys())

for metric_idx, (metric, ylabel, title) in enumerate([
    ('agreement', 'Inter-Agent Agreement', 'Inter-Agent Agreement vs Temperature'),
    ('agent0_acc', 'Agent 0 Accuracy', 'Agent 0 (Perturbed) Accuracy vs Temperature'),
    ('agent0_flip', 'Agent 0 Flip Rate', 'Agent 0 Answer Flip Rate vs Temperature'),
]):
    ax = axes[metric_idx]
    for r in range(4):
        vals = [all_stats[t][r][metric] for t in temps_sorted]
        ax.plot(temps_sorted, vals, marker=markers[r], color=colors_round[r],
                label=round_labels[r], linewidth=2, markersize=6, alpha=0.85)

    # Highlight the target round region
    ax.axvspan(4, 21, alpha=0.06, color='red')
    if metric_idx == 1:
        ax.annotate('Agent 0 effectively\nrandomized', xy=(12, 0.05),
                     fontsize=9, color='red', ha='center', style='italic')
    
    ax.set_xlabel('Temperature (of Agent 0 in Round 2)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 21)

fig.suptitle(f'{DATA.upper()} / {MODEL} ({SIZE_LABEL}{MNT_LABEL}) — N_agents={N_AGENTS}, Target Round={TARGET_ROUND}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig1_metrics_vs_temperature_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png')
print(f"Saved: {OUTPUT_DIR}/fig1_metrics_vs_temperature_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png")
plt.close()


# ═══════════════════════════════════════════════
# Figure 2: Consistency ↔ Correctness correlation
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 2a: Agreement when debate correct vs wrong, per temperature (only in target round 2)
agree_correct_list = []
agree_wrong_list = []

for tt in temps_sorted:
    samples = load_experiment(tt)
    if samples is None:
        continue
    ac, aw = [], []
    for sample in samples:
        rd = sample[str(TARGET_ROUND)]
        answers = rd['final_answers']
        agr = compute_agreement(answers)
        if rd['debate_answer_iscorr']:
            ac.append(agr)
        else:
            aw.append(agr)
    agree_correct_list.append(np.mean(ac) if ac else np.nan)
    agree_wrong_list.append(np.mean(aw) if aw else np.nan)

ax = axes[0]
x = np.arange(len(temps_sorted))
width = 0.35
bars1 = ax.bar(x - width/2, agree_correct_list, width, label='Debate Correct', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x + width/2, agree_wrong_list, width, label='Debate Wrong', color='#F44336', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in temps_sorted], rotation=0)
ax.set_xlabel('Temperature')
ax.set_ylabel('Mean Agreement (Round 2)')
ax.set_title('Agreement when Debate Answer is Correct vs Wrong')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.05)

# 2b: Agent0 cross-round consistency (R1→R2) — T=1 vs T=20
ax = axes[1]
categories = ['Consistent\n(same answer)', 'Flipped\nto correct', 'Flipped\nto wrong']

# Dynamically compute cross-round stability for T=1.0 and T=20.0
def compute_cross_round_stability(tt_val):
    samples = load_experiment(tt_val)
    consistent, flip_correct, flip_wrong = 0, 0, 0
    if samples is None:
        return [0, 0, 0]
    for sample in samples:
        prev_round = str(TARGET_ROUND - 1) if TARGET_ROUND > 0 else "0"
        curr_round = str(TARGET_ROUND)
        if prev_round not in sample or curr_round not in sample:
            continue
        prev_ans = sample[prev_round]['final_answers'][0]
        curr_ans = sample[curr_round]['final_answers'][0]
        correct_ans = sample[curr_round]['answer']
        if prev_ans == curr_ans:
            consistent += 1
        elif curr_ans == correct_ans or (isinstance(curr_ans, (int, float)) and isinstance(correct_ans, (int, float)) and curr_ans == correct_ans):
            flip_correct += 1
        else:
            flip_wrong += 1
    return [consistent, flip_correct, flip_wrong]

baseline_vals = compute_cross_round_stability(1.0)
broken_vals = compute_cross_round_stability(20.0)

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, baseline_vals, width, label='T=1.0 (Normal)', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, broken_vals, width, label='T=20.0 (Broken)', color='#FF5722', alpha=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Number of Samples (out of {0})'.format(DATA_SIZE))
ax.set_title('Agent 0 Cross-Round Stability\n(Round 1 → Round 2)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Consistency as an Unsupervised Signal for Agent Reliability ({SIZE_LABEL}{MNT_LABEL})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig2_consistency_correctness_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png')
print(f"Saved: {OUTPUT_DIR}/fig2_consistency_correctness_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png")
plt.close()


# ═══════════════════════════════════════════════
# Figure 3: Debate accuracy vs Temperature (overall system)
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for r in range(4):
    vals = [all_stats[t][r]['debate_acc'] for t in temps_sorted]
    ax.plot(temps_sorted, vals, marker=markers[r], color=colors_round[r],
            label=round_labels[r], linewidth=2.5, markersize=7)

ax.axvspan(4, 21, alpha=0.06, color='red')
ax.annotate('High temperature zone\n(Agent 0 randomized)', xy=(12, 0.65),
            fontsize=10, color='red', ha='center', style='italic')
ax.axhline(y=all_stats[1.0][0]['debate_acc'], color='gray', linestyle='--', alpha=0.5, label='Round 0 baseline')

ax.set_xlabel('Temperature (of Agent 0 in Round 2)', fontsize=12)
ax.set_ylabel('Debate Accuracy (Majority Vote)', fontsize=12)
ax.set_title(f'{DATA.upper()} / {MODEL} ({SIZE_LABEL}{MNT_LABEL}): System Accuracy Despite Extreme Temperatures',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 21)
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig3_debate_accuracy_vs_temp_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png')
print(f"Saved: {OUTPUT_DIR}/fig3_debate_accuracy_vs_temp_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png")
plt.close()


# ═══════════════════════════════════════════════
# Figure 4: Conceptual diagram — What can be calibrated?
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.6, 'Calibration Targets in Multi-Agent Debate', fontsize=16, fontweight='bold',
        ha='center', va='center')

# ─── Left block: ConfMAD (supervised) ───
box_confmad = FancyBboxPatch((0.3, 3.8), 5.8, 3.2, boxstyle="round,pad=0.2",
                              facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
ax.add_patch(box_confmad)
ax.text(3.2, 6.7, 'ConfMAD (Supervised)', fontsize=13, fontweight='bold',
        ha='center', color='#C62828')

confmad_items = [
    ('① Calibration Target:', 'Confidence Score', '#D32F2F'),
    ('② Signal Source:', 'Single-agent logit / self-verbalized', '#D32F2F'),
    ('③ Method:', 'Platt Scaling, Temp Scaling (on scores)', '#D32F2F'),
    ('④ Requires:', 'Labeled validation set', '#D32F2F'),
    ('⑤ Granularity:', 'Static (per-dataset)', '#D32F2F'),
]
for i, (label, val, color) in enumerate(confmad_items):
    y = 6.2 - i * 0.48
    ax.text(0.6, y, label, fontsize=10, fontweight='bold', color=color, va='center')
    ax.text(3.0, y, val, fontsize=10, color='#333', va='center')

# ─── Right block: Ours (unsupervised) ───
box_ours = FancyBboxPatch((7.2, 3.8), 6.4, 3.2, boxstyle="round,pad=0.2",
                           facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(box_ours)
ax.text(10.4, 6.7, 'Ours (Unsupervised)', fontsize=13, fontweight='bold',
        ha='center', color='#2E7D32')

ours_items = [
    ('① Calibration Target:', 'Temperature / Vote Weight / Rounds', '#1B5E20'),
    ('② Signal Source:', 'Cross-agent & cross-round consistency', '#1B5E20'),
    ('③ Method:', 'Consistency-Driven Adaptive Scaling', '#1B5E20'),
    ('④ Requires:', 'No labels (fully unsupervised)', '#1B5E20'),
    ('⑤ Granularity:', 'Dynamic (per-question, per-round)', '#1B5E20'),
]
for i, (label, val, color) in enumerate(ours_items):
    y = 6.2 - i * 0.48
    ax.text(7.5, y, label, fontsize=10, fontweight='bold', color=color, va='center')
    ax.text(9.9, y, val, fontsize=10, color='#333', va='center')

# ─── Bottom block: What can be calibrated? ───
box_targets = FancyBboxPatch((0.3, 0.3), 13.3, 3.0, boxstyle="round,pad=0.2",
                              facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=2)
ax.add_patch(box_targets)
ax.text(7, 3.05, 'Possible Calibration Targets (Beyond Temperature)', fontsize=13,
        fontweight='bold', ha='center', color='#E65100')

targets = [
    ('🌡️ Temperature', 'Control generation randomness\nper-agent, per-round', 1.8, 2.3),
    ('⚖️ Vote Weight', 'Adjust influence in\nmajority voting', 5.1, 2.3),
    ('🔄 #Rounds', 'Early stop or extend\nbased on convergence', 8.4, 2.3),
    ('👥 #Agents', 'Dynamically add/remove\nagents per question', 11.7, 2.3),
]
target_colors = ['#1565C0', '#6A1B9A', '#00695C', '#BF360C']

for (title, desc, x, y), color in zip(targets, target_colors):
    box = FancyBboxPatch((x - 1.3, y - 1.5), 2.6, 2.4, boxstyle="round,pad=0.15",
                          facecolor='white', edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y + 0.45, title, fontsize=10, fontweight='bold', ha='center', color=color)
    ax.text(x, y - 0.35, desc, fontsize=8.5, ha='center', va='center', color='#555',
            linespacing=1.4)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig4_calibration_targets_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png')
print(f"Saved: {OUTPUT_DIR}/fig4_calibration_targets_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png")
plt.close()


# ═══════════════════════════════════════════════
# Figure 5: Agreement computation illustration
# ═══════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

ax.text(6, 4.7, 'How Agreement is Computed', fontsize=15, fontweight='bold', ha='center')

# Three examples
examples = [
    {
        'title': 'Full Consensus',
        'agents': ['Agent 1: 120', 'Agent 2: 120', 'Agent 3: 120'],
        'formula': 'max(Counter) / N = 3/3',
        'result': '= 1.000',
        'color': '#4CAF50',
        'x_center': 2,
    },
    {
        'title': 'Majority Exists',
        'agents': ['Agent 1: 55', 'Agent 2: 120', 'Agent 3: 120'],
        'formula': 'max(Counter) / N = 2/3',
        'result': '= 0.667',
        'color': '#FF9800',
        'x_center': 6,
    },
    {
        'title': 'No Agreement',
        'agents': ['Agent 1: 55', 'Agent 2: 120', 'Agent 3: 77'],
        'formula': 'max(Counter) / N = 1/3',
        'result': '= 0.333',
        'color': '#F44336',
        'x_center': 10,
    },
]

for ex in examples:
    xc = ex['x_center']
    box = FancyBboxPatch((xc - 1.7, 0.3), 3.4, 4.0, boxstyle="round,pad=0.15",
                          facecolor='white', edgecolor=ex['color'], linewidth=2)
    ax.add_patch(box)
    ax.text(xc, 4.0, ex['title'], fontsize=12, fontweight='bold', ha='center', color=ex['color'])

    for i, agent_text in enumerate(ex['agents']):
        y = 3.2 - i * 0.55
        # highlight matching answers
        if '120' in agent_text and ex['title'] != 'No Agreement':
            bg_color = '#E8F5E9'
        elif ex['title'] == 'No Agreement':
            bg_color = '#FFF3E0'
        else:
            bg_color = '#FFEBEE'
        
        if ex['title'] == 'Full Consensus':
            bg_color = '#E8F5E9'
            
        inner = FancyBboxPatch((xc - 1.3, y - 0.2), 2.6, 0.45, boxstyle="round,pad=0.05",
                                facecolor=bg_color, edgecolor='#ccc', linewidth=0.5)
        ax.add_patch(inner)
        ax.text(xc, y, agent_text, fontsize=10, ha='center', va='center', family='monospace')

    ax.text(xc, 1.1, ex['formula'], fontsize=9, ha='center', va='center', color='#555')
    ax.text(xc, 0.65, ex['result'], fontsize=14, fontweight='bold', ha='center',
            va='center', color=ex['color'])

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig5_agreement_illustration_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png')
print(f"Saved: {OUTPUT_DIR}/fig5_agreement_illustration_{MODEL}_{DATA_SIZE}{MNT_SUFFIX}.png")
plt.close()


print(f"\n✅ All figures saved to {OUTPUT_DIR}/")

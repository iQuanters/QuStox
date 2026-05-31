#!/usr/bin/env python3
"""Render a diagram of the 16-qubit Ising quantum reservoir → reservoir_diagram.png."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from itertools import combinations

N_GROUPS, GROUP = 4, 4
N = N_GROUPS * GROUP
groups = [list(range(g * GROUP, (g + 1) * GROUP)) for g in range(N_GROUPS)]
intra = [p for grp in groups for p in combinations(grp, 2)]
ring = [(groups[g][-1], groups[(g + 1) % N_GROUPS][0]) for g in range(N_GROUPS)]
GCOL = ["#e4572e", "#17bebb", "#ffc914", "#76b041"]

# ── positions: 4 group-centers on a big circle; 4 qubits on a small circle ────
pos = {}
R, r = 3.4, 1.0
gang = [90, 0, -90, 180]                       # top, right, bottom, left
for g in range(N_GROUPS):
    cx, cy = R * np.cos(np.radians(gang[g])), R * np.sin(np.radians(gang[g]))
    for k in range(GROUP):                     # qubit k faces "outward-rotated" so seams align
        a = np.radians(gang[g] + 135 - k * 90)
        pos[g * GROUP + k] = (cx + r * np.cos(a), cy + r * np.sin(a))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.15, 1]})

# ── panel A: topology ─────────────────────────────────────────────────────────
for (i, j) in intra:                           # intra-group: all-to-all
    g = i // GROUP
    ax.plot(*zip(pos[i], pos[j]), color=GCOL[g], lw=1.4, alpha=0.5, zorder=1)
for (i, j) in ring:                            # inter-group: ring bridges
    ax.plot(*zip(pos[i], pos[j]), color="#333", lw=2.6, ls=(0, (4, 2)), zorder=2)
for q, (x, y) in pos.items():
    g, k = q // GROUP, q % GROUP
    ax.scatter([x], [y], s=920, color=GCOL[g], edgecolors="white", linewidths=2, zorder=3)
    ax.text(x, y + 0.005, f"q{q}", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", zorder=4)
    # 2^k input multiplier
    ux, uy = x + 0.42 * np.cos(np.arctan2(y, x)), y + 0.42 * np.sin(np.arctan2(y, x))
    ax.text(ux, uy, f"×{2**k}", ha="center", va="center", fontsize=8, color=GCOL[g], zorder=4)
for g in range(N_GROUPS):
    cx, cy = R * np.cos(np.radians(gang[g])), R * np.sin(np.radians(gang[g]))
    ax.text(cx, cy, f"group {g}", ha="center", va="center", fontsize=9,
            color="#555", alpha=0.5, zorder=0)

ax.plot([], [], color="#888", lw=1.4, alpha=0.6, label="intra-group  (all-to-all, 6×4 = 24)")
ax.plot([], [], color="#333", lw=2.6, ls=(0, (4, 2)), label="inter-group ring  (4)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10)
ax.set_title("16-qubit reservoir topology  ·  4 groups × 4  ·  28 Ising couplings",
             fontsize=13, fontweight="bold")
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-5, 5); ax.set_ylim(-5.2, 5)

# ── panel B: data flow ────────────────────────────────────────────────────────
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.set_title("Per-window evolution  &  read-out", fontsize=13, fontweight="bold")


def box(x, y, w, h, text, fc, ec="#333", fs=10):
    ax2.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                  fc=fc, ec=ec, lw=1.3))
    ax2.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2):
    ax2.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                  mutation_scale=16, lw=1.6, color="#333"))


box(0.3, 8.4, 9.4, 1.2,
    "input window  [close_prev₁ … close_prevₙ]   (oldest → newest)", "#eef2f7", fs=11)
arrow(5, 8.4, 5, 7.9)
# the repeated evolution block
ax2.add_patch(FancyBboxPatch((0.6, 3.7), 8.8, 4.0, boxstyle="round,pad=0.1",
              fc="#fbfbf0", ec="#b8860b", lw=1.6, ls="--"))
ax2.text(5, 7.4, "repeat for each step v in the window  (no reset, no mid-measure)",
         ha="center", fontsize=9.5, style="italic", color="#7a5c00")
box(1.1, 5.9, 7.8, 1.0, "encode:  RX(π · 2ᵏ · v)  on qubit k of every group", "#fdecea", fs=10.5)
arrow(5, 5.9, 5, 5.5)
box(1.1, 4.0, 7.8, 1.4,
    "Ising evolve (fixed random):\n∏ RX(hₓ)RY(h_y)RZ(h_z)   ·   ∏ RZZ(J_z) RXX(Jₓ)", "#e8f6f5", fs=10)
arrow(5, 3.7, 5, 3.2)
box(0.7, 2.1, 8.6, 1.0,
    "measure all 16 qubits · 1000 shots  →  ⟨Zᵢ⟩ (16) + ⟨ZᵢZⱼ⟩ intra (24) = 40", "#eef7ea", fs=10)
arrow(5, 2.1, 5, 1.6)
box(0.3, 0.4, 9.4, 1.0,
    "40 reservoir features  ⊕  Open/High/Low (3)  →  NN [43, 6, 1]  →  BUY / SELL",
    "#e7eefc", fs=10.5)

fig.suptitle("HQRC — Quantum Reservoir (16-qubit transverse-field Ising)",
             fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("reservoir_diagram.png", dpi=150, bbox_inches="tight")
print("wrote reservoir_diagram.png")

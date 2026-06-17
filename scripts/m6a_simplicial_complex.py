"""
m6a_simplicial_complex.py
=========================
Module 6a - Building a simplicial complex from a graph (the clique complex).

Proof verified:
    The clique complex of a graph promotes every triangle of the 1-skeleton
    to a filled 2-simplex. The number of 2-simplices produced by our
    intersection-based enumeration matches an independent count (networkx
    triangle count divided by 3), and the Euler characteristic
        chi = #nodes - #edges + #triangles
    is reproduced by the per-dimension simplex counts.

    VERIFY: triangle count from src.simplicial.triangles equals the
    independent networkx count, exactly.

Output: media/figures/m6a_simplicial_complex.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, GOLD, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.simplicial import triangles, simplex_counts, random_simplicial_complex

np.random.seed(42)
setup_light_theme()

# ── A small graph to visualise the filled complex ───────────────────────────
G_small = nx.erdos_renyi_graph(18, 0.28, seed=5)
tris_small = triangles(G_small)
counts_small = simplex_counts(G_small)
pos = nx.spring_layout(G_small, seed=2)

# Independent triangle count for verification
nx_tri = sum(nx.triangles(G_small).values()) // 3
chi = counts_small[0] - counts_small[1] + counts_small[2]

# ── A larger random simplicial complex: simplex spectrum ─────────────────────
G_big, tris_big = random_simplicial_complex(300, k1=10, k_delta=8, seed=7)
nodes_per_dim = [G_big.number_of_nodes(), G_big.number_of_edges(), len(tris_big)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")

# Filled 2-simplices
ax1.set_facecolor("#F8FAFC")
for (a, b, c) in tris_small:
    poly = Polygon([pos[a], pos[b], pos[c]], closed=True, facecolor=TEAL,
                   alpha=0.22, edgecolor='none', zorder=1)
    ax1.add_patch(poly)
nx.draw_networkx_edges(G_small, pos, ax=ax1, edge_color=SLATE, width=1.4, alpha=0.7)
nx.draw_networkx_nodes(G_small, pos, ax=ax1, node_color=NAVY, node_size=180,
                       edgecolors='white', linewidths=1.2)
nx.draw_networkx_labels(G_small, pos, ax=ax1, font_color='white', font_size=8)
ax1.set_title(f'Clique complex: {counts_small[0]} nodes, {counts_small[1]} edges, '
              f'{counts_small[2]} triangles\n(filled = 2-simplices)',
              color=NAVY, fontweight='bold')
ax1.axis('off')

# Simplex spectrum (counts per dimension) of the larger complex
apply_axes_style(ax2)
dims = ['0-simplices\n(nodes)', '1-simplices\n(edges)', '2-simplices\n(triangles)']
bars = ax2.bar(dims, nodes_per_dim, color=[NAVY, TEAL, GOLD], alpha=0.85,
               edgecolor='white', width=0.62)
for b, v in zip(bars, nodes_per_dim):
    ax2.text(b.get_x() + b.get_width() / 2, v + max(nodes_per_dim) * 0.01,
             str(v), ha='center', va='bottom', fontsize=11, color=NAVY, fontweight='bold')
ax2.set_ylabel('Count', color=NAVY)
ax2.set_title(f'Random simplicial complex (N=300)\n'
              rf'$\langle k\rangle\approx10$, $\langle k_\Delta\rangle\approx8$',
              color=NAVY, fontweight='bold')

fig.suptitle('Module 6a — Simplicial complexes: promoting triangles to 2-simplices',
             fontsize=16, color=NAVY, fontweight='bold', y=1.0)
plt.tight_layout()

# ── VERIFY ────────────────────────────────────────────────────────────────────
ok = (len(tris_small) == nx_tri)
print("=" * 70)
print("VERIFY — clique complex triangle enumeration:")
print(f"  src.simplicial.triangles count = {len(tris_small)}")
print(f"  independent networkx count     = {nx_tri}")
print(f"  Euler characteristic chi = V - E + F = {counts_small[0]} - "
      f"{counts_small[1]} + {counts_small[2]} = {chi}")
print(f"  triangle counts match exactly: {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m6a_simplicial_complex")
plt.close()

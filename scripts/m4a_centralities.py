"""
m4a_centralities.py
===================
Module 4a: Node centralities on Erdos-Renyi vs Barabasi-Albert graphs.

Proof verified:
    On a scale-free (Barabasi-Albert) network the leading eigenvector of the
    adjacency matrix localises on the high-degree hubs, tightening the
    relationship between degree centrality and eigenvector centrality. On a
    homogeneous Erdos-Renyi network there are no hubs, so the same two
    centralities are less tightly coupled.

    VERIFY: Pearson correlation r(degree, eigenvector) is larger on BA than on
    ER (r_BA > r_ER, with r_BA > 0.8) - hub localisation strengthens the link.

Output: media/figures/m4a_centralities.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.viz import (NAVY, TEAL, RED, SLATE, PURPLE,
                     setup_light_theme, apply_axes_style, save_figure)
from src.networks import (er_graph, ba_graph, centralities, centrality_vector,
                          giant_component)

np.random.seed(42)
setup_light_theme()

# Build graphs (matched mean degree); use giant components
# Eigenvector centrality is only well defined on a connected graph, so we work
# on the largest connected component (BA is connected; ER may not be).
N = 300
m = 3
G_er = giant_component(er_graph(N, p=2 * m / (N - 1), seed=42))   # <k> ~ 2m
G_ba = giant_component(ba_graph(N, m=m, seed=42))

names = ['degree', 'betweenness', 'eigenvector', 'pagerank']
cmaps = ['viridis', 'plasma', 'magma', 'cividis']


def centrality_table(G):
    c = centralities(G)
    nodes = sorted(G.nodes())
    return {k: centrality_vector(c[k], nodes) for k in c}, nodes


tab_er, nodes_er = centrality_table(G_er)
tab_ba, nodes_ba = centrality_table(G_ba)

# Figure: layouts colored by centrality (top BA) + correlation panels
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.25)
fig.patch.set_facecolor("#F8FAFC")

pos_ba = nx.spring_layout(G_ba, seed=7)
deg_ba = tab_ba['degree']
for j, (cname, cmap) in enumerate(zip(names, cmaps)):
    ax = fig.add_subplot(gs[0, j])
    vals = tab_ba[cname]
    nx.draw_networkx_edges(G_ba, pos_ba, ax=ax, alpha=0.12, edge_color=SLATE)
    sizes = 20 + 380 * (deg_ba - deg_ba.min()) / (np.ptp(deg_ba) + 1e-12)
    sc = nx.draw_networkx_nodes(G_ba, pos_ba, ax=ax, node_color=vals,
                                cmap=cmap, node_size=sizes, linewidths=0.4,
                                edgecolors='white')
    ax.set_title(f'BA - {cname}', color=NAVY, fontweight='bold', fontsize=12)
    ax.axis('off')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

# Correlation matrices ER vs BA
keys = ['degree', 'closeness', 'betweenness', 'eigenvector', 'katz', 'pagerank']


def corr_matrix(tab):
    M = np.vstack([tab[k] for k in keys])
    return np.corrcoef(M)


for col, (tab, label) in enumerate([(tab_er, 'Erdos-Renyi'), (tab_ba, 'Barabasi-Albert')]):
    ax = fig.add_subplot(gs[1, col])
    C = corr_matrix(tab)
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_title(f'{label}: centrality correlations', color=NAVY, fontweight='bold', fontsize=12)
    for i in range(len(keys)):
        for k in range(len(keys)):
            ax.text(k, i, f'{C[i, k]:.2f}', ha='center', va='center',
                    color='white' if abs(C[i, k]) > 0.5 else NAVY, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# Scatter: degree vs eigenvector for both topologies
ax = fig.add_subplot(gs[1, 2:])
apply_axes_style(ax)
r_er = np.corrcoef(tab_er['degree'], tab_er['eigenvector'])[0, 1]
r_ba = np.corrcoef(tab_ba['degree'], tab_ba['eigenvector'])[0, 1]
ax.scatter(tab_er['degree'], tab_er['eigenvector'], s=22, color=TEAL,
           alpha=0.6, edgecolors='white', linewidths=0.3, label=f'ER (r={r_er:.2f})')
ax.scatter(tab_ba['degree'], tab_ba['eigenvector'], s=22, color=PURPLE,
           alpha=0.6, edgecolors='white', linewidths=0.3, label=f'BA (r={r_ba:.2f})')
ax.set_xlabel('Degree centrality', color=NAVY)
ax.set_ylabel('Eigenvector centrality', color=NAVY)
ax.set_title('Degree vs eigenvector centrality', color=NAVY, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, facecolor='white', edgecolor=SLATE)

fig.suptitle('Module 4a: Centralities: hub localisation on scale-free vs homogeneous graphs',
             fontsize=16, color=NAVY, fontweight='bold', y=0.98)

# VERIFY
ok = (r_ba > r_er) and (r_ba > 0.8)
print("=" * 70)
print("VERIFY: degree vs eigenvector centrality correlation:")
print(f"  Erdos-Renyi:     r(degree, eigenvector) = {r_er:.4f}")
print(f"  Barabasi-Albert: r(degree, eigenvector) = {r_ba:.4f}")
print(f"  Hub localisation (r_BA > r_ER and r_BA > 0.8): {'PASS' if ok else 'FAIL'}")
print("=" * 70)

save_figure(fig, "m4a_centralities")
plt.close()

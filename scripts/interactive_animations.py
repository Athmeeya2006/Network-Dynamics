"""
interactive_animations.py
==========================
Self-contained, interactive (auto-play, play/pause, scrub-slider, rotatable-3D)
Plotly animations for the Nonlinear-Dynamics-on-Networks ladder. Each HTML is
written to media/interactive/ and shares one local plotly.min.js (offline).

The animations sample time finely for smooth playback at a moderate speed, and
import the dynamics from src/ so they use the same equations as the static
figures.

Run:  python scripts/interactive_animations.py
"""

from __future__ import annotations

import os
import sys
import colorsys

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.viz import (TEAL, RED, GOLD, PURPLE, CYAN, BG, CARD, DIM,        # noqa: E402
                     get_media_dir)
from src.integrators import rk4_step                                      # noqa: E402
from src.flows import lorenz, rossler                                     # noqa: E402
from src.oscillators import StuartLandau                                  # noqa: E402
from src.games import replicator_rhs, rps_payoff, simplex_projection      # noqa: E402
from src.kuramoto import (mean_field_rhs, network_rhs, order_parameter,   # noqa: E402
                          sample_lorentzian)
from src.networks import watts_strogatz, adjacency_matrix                 # noqa: E402

OUT = get_media_dir("interactive")
LIGHTTXT = "#E2E8F0"
MUTED = "#94A3B8"

# Cyclic colorscale for phases (red, yellow, green, cyan, blue, magenta, red)
CYCLIC = [[0.0, "#FF3B3B"], [0.166, "#FFD166"], [0.333, "#06D6A0"],
          [0.5, "#1BE3E3"], [0.666, "#4D7CFF"], [0.833, "#C77DFF"],
          [1.0, "#FF3B3B"]]


# HELPERS

def integrate(rhs, x0, T, dt, args=()):
    """Fixed-step RK4 trajectory; returns array (n+1, dim)."""
    n = int(round(T / dt))
    xs = np.empty((n + 1, len(x0)))
    x = np.array(x0, dtype=float)
    xs[0] = x
    for i in range(n):
        x = rk4_step(rhs, x, i * dt, dt, *args)
        xs[i + 1] = x
    return xs


def rainbow(n):
    """n evenly-spaced rainbow hex colors."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 0.82, 1.0)
        out.append(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
    return out


def base_layout(title, subtitle="", is3d=False):
    lay = dict(
        title=dict(text=f"<b>{title}</b>" + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
                   x=0.5, xanchor="center", y=0.96, font=dict(color=LIGHTTXT, size=20)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=LIGHTTXT, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=55, r=45, t=95, b=95),
        showlegend=False,
    )
    if is3d:
        ax = dict(backgroundcolor=BG, gridcolor=DIM, zerolinecolor=DIM,
                  showbackground=True, color=MUTED)
        lay["scene"] = dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor=BG)
    else:
        lay["xaxis"] = dict(gridcolor=DIM, zerolinecolor=DIM, color=MUTED)
        lay["yaxis"] = dict(gridcolor=DIM, zerolinecolor=DIM, color=MUTED)
    return lay


def finalize(fig, names, labels, filename, frame_ms, prefix=""):
    """Attach auto-play, Play/Pause buttons and a scrub-slider; write HTML."""
    play = dict(label="▶  Play", method="animate",
                args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                 fromcurrent=True, mode="immediate",
                                 transition=dict(duration=0))])
    pause = dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate", transition=dict(duration=0))])
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", showactive=False,
        x=0.0, y=-0.08, xanchor="left", yanchor="top",
        bgcolor=CARD, bordercolor=DIM, borderwidth=1,
        font=dict(color=LIGHTTXT, size=13), pad=dict(l=8, r=8, t=7, b=7),
        buttons=[play, pause])])
    steps = [dict(method="animate", label=lab,
                  args=[[nm], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))])
             for nm, lab in zip(names, labels)]
    fig.update_layout(sliders=[dict(
        active=0, x=0.20, y=-0.04, len=0.78, xanchor="left", yanchor="top",
        bgcolor=CARD, bordercolor=DIM, borderwidth=1, tickcolor=DIM,
        font=dict(color=MUTED, size=10), pad=dict(t=2, b=2),
        currentvalue=dict(prefix=prefix, font=dict(color=LIGHTTXT, size=13)),
        steps=steps)])
    path = os.path.join(OUT, filename)
    fig.write_html(path, include_plotlyjs="directory", auto_play=True,
                   animation_opts=dict(frame=dict(duration=frame_ms, redraw=True),
                                       transition=dict(duration=0), mode="immediate"),
                   full_html=True,
                   config={"displayModeBar": True, "displaylogo": False,
                           "scrollZoom": True})
    print(f"  saved  media/interactive/{filename}")
    return path


# 1 + 2  ATTRACTOR COMETS (Lorenz, Rossler), rotatable 3D, animated draw

def attractor_comet(rhs, x0, T, dt, title, subtitle, color, filename,
                    F=300, trail=120, frame_ms=26):
    traj = integrate(rhs, x0, T, dt)
    n = len(traj)
    idx = np.linspace(trail, n - 1, F).astype(int)

    faded = go.Scatter3d(x=traj[::2, 0], y=traj[::2, 1], z=traj[::2, 2],
                         mode="lines", line=dict(color=DIM, width=2), hoverinfo="skip")
    i0 = idx[0]
    tr = go.Scatter3d(x=traj[i0 - trail:i0 + 1, 0], y=traj[i0 - trail:i0 + 1, 1],
                      z=traj[i0 - trail:i0 + 1, 2], mode="lines",
                      line=dict(color=color, width=6), hoverinfo="skip")
    head = go.Scatter3d(x=[traj[i0, 0]], y=[traj[i0, 1]], z=[traj[i0, 2]], mode="markers",
                        marker=dict(size=6, color="#FFFFFF", line=dict(color=color, width=2)))
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        s = slice(i - trail, i + 1)
        frames.append(go.Frame(name=str(k), traces=[1, 2], data=[
            go.Scatter3d(x=traj[s, 0], y=traj[s, 1], z=traj[s, 2]),
            go.Scatter3d(x=[traj[i, 0]], y=[traj[i, 1]], z=[traj[i, 2]])]))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    fig = go.Figure(data=[faded, tr, head],
                    layout=base_layout(title, subtitle, is3d=True), frames=frames)
    return finalize(fig, names, labels, filename, frame_ms, prefix="time  ")


# 3  BUTTERFLY EFFECT, two Lorenz trajectories, tiny initial gap

def butterfly():
    T, dt, trail, F, frame_ms = 45.0, 0.01, 90, 320, 26
    a = integrate(lorenz, [1.0, 1.0, 1.0], T, dt)
    b = integrate(lorenz, [1.0 + 1e-3, 1.0, 1.0], T, dt)
    n = len(a)
    idx = np.linspace(trail, n - 1, F).astype(int)

    def faded(tr, c):
        return go.Scatter3d(x=tr[::3, 0], y=tr[::3, 1], z=tr[::3, 2], mode="lines",
                            line=dict(color=c, width=1.5), opacity=0.22, hoverinfo="skip")

    def seg(tr, i, c, w):
        s = slice(i - trail, i + 1)
        return go.Scatter3d(x=tr[s, 0], y=tr[s, 1], z=tr[s, 2], mode="lines",
                            line=dict(color=c, width=w))

    def pt(tr, i, c):
        return go.Scatter3d(x=[tr[i, 0]], y=[tr[i, 1]], z=[tr[i, 2]], mode="markers",
                            marker=dict(size=6, color=c))

    i0 = idx[0]
    data = [faded(a, TEAL), faded(b, RED), seg(a, i0, TEAL, 5), seg(b, i0, RED, 5),
            pt(a, i0, "#7FE3F0"), pt(b, i0, "#FF8A8A")]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        sep = float(np.linalg.norm(a[i] - b[i]))
        frames.append(go.Frame(name=str(k), traces=[2, 3, 4, 5], data=[
            seg(a, i, TEAL, 5), seg(b, i, RED, 5), pt(a, i, "#7FE3F0"), pt(b, i, "#FF8A8A")],
            layout=dict(title=dict(text="<b>Butterfly effect: two Lorenz trajectories</b>"
                        f"<br><sup>start 0.001 apart, separation now = {sep:6.2f}</sup>"))))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    fig = go.Figure(data=data, layout=base_layout(
        "Butterfly effect: two Lorenz trajectories",
        "start 0.001 apart, they track, then diverge", is3d=True), frames=frames)
    return finalize(fig, names, labels, "anim_butterfly_effect.html", frame_ms, prefix="time  ")


# 4  DOUBLE PENDULUM, 8 near-identical copies, slow-motion sensitive dependence

def double_pendulum_rhs(s, t, g=9.81, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    th1, w1, th2, w2 = s
    d = th1 - th2
    den = 2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)
    a1 = (-g * (2 * m1 + m2) * np.sin(th1) - m2 * g * np.sin(th1 - 2 * th2)
          - 2 * np.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * np.cos(d))) / (L1 * den)
    a2 = (2 * np.sin(d) * (w1**2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1)
          + w2**2 * L2 * m2 * np.cos(d))) / (L2 * den)
    return np.array([w1, a1, w2, a2])


def double_pendulum():
    P = 8                      # number of near-identical pendulums
    dt = 0.003                 # fine integration (accurate, smooth)
    T = 14.0                   # seconds of physics
    stride = 8                 # frame every 0.024 s (smooth)
    trail = 40                 # bob trail length (frames)
    frame_ms = 40              # moderate slow motion, easy to follow
    L1 = L2 = 1.0
    th1_0, th2_0 = 1.9, 1.9    # both arms raised, richly chaotic
    cols = rainbow(P)

    trajs = []
    for p in range(P):
        s0 = [th1_0 + p * 5e-4, 0.0, th2_0, 0.0]   # micro-spread on θ1
        s = integrate(double_pendulum_rhs, s0, T, dt)
        x1 = L1 * np.sin(s[:, 0]); y1 = -L1 * np.cos(s[:, 0])
        x2 = x1 + L2 * np.sin(s[:, 2]); y2 = y1 - L2 * np.cos(s[:, 2])
        trajs.append((x1, y1, x2, y2))
    n = trajs[0][0].size
    idx = np.arange(0, n, stride)

    def rod(p, i):
        x1, y1, x2, y2 = trajs[p]
        return go.Scatter(x=[0, x1[i], x2[i]], y=[0, y1[i], y2[i]], mode="lines+markers",
                          line=dict(color=cols[p], width=2.5),
                          marker=dict(size=[4, 6, 11], color=cols[p]), hoverinfo="skip")

    def tail(p, i):
        x1, y1, x2, y2 = trajs[p]
        s = slice(max(0, i - trail), i + 1)
        return go.Scatter(x=x2[s], y=y2[s], mode="lines",
                          line=dict(color=cols[p], width=1.6), opacity=0.45, hoverinfo="skip")

    data = [rod(p, 0) for p in range(P)] + [tail(p, 0) for p in range(P)]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        d = [rod(p, i) for p in range(P)] + [tail(p, i) for p in range(P)]
        frames.append(go.Frame(name=str(k), traces=list(range(2 * P)), data=d))
        names.append(str(k)); labels.append(f"t={i*dt:.2f}")

    lay = base_layout("Double pendulum: 8 copies started almost identically",
                      "they overlap at first, then chaos fans them apart")
    lay["xaxis"].update(range=[-2.3, 2.3], scaleanchor="y", showgrid=False, zeroline=False,
                        visible=False)
    lay["yaxis"].update(range=[-2.4, 1.5], showgrid=False, zeroline=False, visible=False)
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_double_pendulum.html", frame_ms, prefix="time  ")


# 5  LOGISTIC COBWEB, animated over the parameter r

def logistic_cobweb():
    rs = np.linspace(2.5, 3.99, 130)
    xg = np.linspace(0, 1, 200)

    def cobweb_path(r, x0=0.2, n=80):
        xs, ys = [x0], [0.0]
        x = x0
        for _ in range(n):
            fx = r * x * (1 - x)
            xs += [x, fx]; ys += [fx, fx]
            x = fx
        return xs, ys

    diag = go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                      line=dict(color=DIM, width=1.5, dash="dash"), hoverinfo="skip")
    para = go.Scatter(x=xg, y=rs[0] * xg * (1 - xg), mode="lines",
                      line=dict(color=GOLD, width=3), hoverinfo="skip")
    cx, cy = cobweb_path(rs[0])
    web = go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=TEAL, width=1.2), hoverinfo="skip")

    def regime(r):
        if r < 3.0: return "fixed point"
        if r < 3.449: return "period-2"
        if r < 3.544: return "period-4"
        if r < 3.5699: return "period-8 and up"
        return "CHAOS"

    frames, names, labels = [], [], []
    for k, r in enumerate(rs):
        cx, cy = cobweb_path(r)
        frames.append(go.Frame(name=str(k), traces=[1, 2], data=[
            go.Scatter(x=xg, y=r * xg * (1 - xg)), go.Scatter(x=cx, y=cy)],
            layout=dict(title=dict(text="<b>Logistic map cobweb</b>"
                        f"<br><sup>r = {r:.3f}, {regime(r)}</sup>"))))
        names.append(str(k)); labels.append(f"{r:.2f}")

    lay = base_layout("Logistic map cobweb", f"r = {rs[0]:.3f}, {regime(rs[0])}")
    lay["xaxis"].update(range=[0, 1], title="xₙ")
    lay["yaxis"].update(range=[0, 1], title="xₙ₊₁", scaleanchor="x")
    fig = go.Figure(data=[diag, para, web], layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_logistic_cobweb.html", 75, prefix="parameter r = ")


# 6  STUART-LANDAU LIMIT CYCLE, trajectories converge onto the cycle

def limit_cycle():
    osc = StuartLandau(mu=1.0, omega=1.6)   # limit-cycle radius = 1
    T, dt, trail, F, frame_ms = 24.0, 0.008, 60, 320, 26
    ics = [[0.05, 0.0], [-0.04, 0.03], [1.9, 0.2], [-1.6, -1.4]]
    cols = [TEAL, CYAN, RED, GOLD]
    trajs = [integrate(osc.rhs, ic, T, dt) for ic in ics]
    n = trajs[0].shape[0]
    idx = np.linspace(0, n - 1, F).astype(int)

    th = np.linspace(0, 2 * np.pi, 200)
    cyc = go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines",
                     line=dict(color=DIM, width=2, dash="dash"), hoverinfo="skip")
    paths = [go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines",
                        line=dict(color=c, width=1), opacity=0.25, hoverinfo="skip")
             for t, c in zip(trajs, cols)]

    def tail(j, i):
        s = slice(max(0, i - trail), i + 1)
        return go.Scatter(x=trajs[j][s, 0], y=trajs[j][s, 1], mode="lines",
                          line=dict(color=cols[j], width=3), hoverinfo="skip")

    def head(j, i):
        return go.Scatter(x=[trajs[j][i, 0]], y=[trajs[j][i, 1]], mode="markers",
                          marker=dict(size=9, color=cols[j], line=dict(color="#FFF", width=1)))

    nseed = 1 + len(paths)
    data = [cyc] + paths + [tail(j, 0) for j in range(4)] + [head(j, 0) for j in range(4)]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        d = [tail(j, i) for j in range(4)] + [head(j, i) for j in range(4)]
        frames.append(go.Frame(name=str(k), traces=list(range(nseed, nseed + 8)), data=d))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    lay = base_layout("Stuart-Landau: every start spirals onto the same limit cycle",
                      "μ = 1, cycle radius √μ = 1 (dashed)")
    lay["xaxis"].update(range=[-2.1, 2.1], title="x")
    lay["yaxis"].update(range=[-2.1, 2.1], title="y", scaleanchor="x")
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_limit_cycle.html", frame_ms, prefix="time  ")


# 7  REPLICATOR RPS, orbits on the simplex (neutral, stable, unstable)

def rps_simplex():
    T, dt, trail, F, frame_ms = 120.0, 0.02, 55, 340, 26
    x0 = [0.42, 0.34, 0.24]
    regimes = [("Neutral (closed orbit)", 0.0, GOLD),
               ("Stable (spiral in)", 0.5, TEAL),
               ("Unstable (spiral out)", -0.5, RED)]
    trajs, cols, names_reg = [], [], []
    for nm, eps, c in regimes:
        tr = integrate(replicator_rhs, x0, T, dt, args=(rps_payoff(eps),))
        tr = np.clip(tr, 1e-9, 1.0)
        trajs.append(simplex_projection(tr)); cols.append(c); names_reg.append(nm)
    n = trajs[0].shape[0]
    idx = np.linspace(0, n - 1, F).astype(int)

    V = simplex_projection(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], float))
    tri = go.Scatter(x=V[:, 0], y=V[:, 1], mode="lines",
                     line=dict(color=MUTED, width=2), hoverinfo="skip")
    paths = [go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines",
                        line=dict(color=c, width=1), opacity=0.22, hoverinfo="skip")
             for t, c in zip(trajs, cols)]

    def tail(j, i):
        s = slice(max(0, i - trail), i + 1)
        return go.Scatter(x=trajs[j][s, 0], y=trajs[j][s, 1], mode="lines",
                          line=dict(color=cols[j], width=3), hoverinfo="skip")

    def head(j, i):
        return go.Scatter(x=[trajs[j][i, 0]], y=[trajs[j][i, 1]], mode="markers",
                          marker=dict(size=10, color=cols[j], line=dict(color="#FFF", width=1)))

    nseed = 1 + 3
    data = [tri] + paths + [tail(j, 0) for j in range(3)] + [head(j, 0) for j in range(3)]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        d = [tail(j, i) for j in range(3)] + [head(j, i) for j in range(3)]
        frames.append(go.Frame(name=str(k), traces=list(range(nseed, nseed + 6)), data=d))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    lay = base_layout("Rock-Paper-Scissors replicator dynamics on the simplex",
                      "gold = closed orbit, teal = spiral in, red = spiral out")
    lay["xaxis"].update(range=[-0.1, 1.1], showgrid=False, zeroline=False, visible=False)
    lay["yaxis"].update(range=[-0.12, 1.0], showgrid=False, zeroline=False, visible=False,
                        scaleanchor="x")
    lay["annotations"] = [
        dict(x=0, y=0, text="<b>Rock</b>", showarrow=False, yshift=-16, font=dict(color=LIGHTTXT)),
        dict(x=1, y=0, text="<b>Paper</b>", showarrow=False, yshift=-16, font=dict(color=LIGHTTXT)),
        dict(x=0.5, y=np.sqrt(3)/2, text="<b>Scissors</b>", showarrow=False, yshift=16,
             font=dict(color=LIGHTTXT))]
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_rps_simplex.html", frame_ms, prefix="time  ")


# 8  KURAMOTO ON THE CIRCLE, synchronization in time

def kuramoto_circle():
    rng = np.random.default_rng(7)
    N, T, dt, F, frame_ms = 80, 32.0, 0.01, 320, 28
    K = 2.4
    omega = np.clip(sample_lorentzian(0.5, N, rng), -3, 3)
    theta = np.empty((int(T/dt) + 1, N))
    theta[0] = rng.uniform(-np.pi, np.pi, N)
    for i in range(int(T/dt)):
        theta[i + 1] = rk4_step(mean_field_rhs, theta[i], i*dt, dt, omega, K)
    idx = np.linspace(0, theta.shape[0] - 1, F).astype(int)

    circ_t = np.linspace(0, 2*np.pi, 150)
    circle = go.Scatter(x=np.cos(circ_t), y=np.sin(circ_t), mode="lines",
                        line=dict(color=DIM, width=2), hoverinfo="skip")

    def dots(i):
        th = theta[i]
        return go.Scatter(x=np.cos(th), y=np.sin(th), mode="markers",
                          marker=dict(size=11, color=omega, colorscale="Turbo", cmin=-2, cmax=2,
                                      line=dict(color="rgba(0,0,0,0.5)", width=0.5),
                                      colorbar=dict(title="ωᵢ", x=1.02, len=0.7,
                                                    tickfont=dict(color=MUTED))))

    def vec(i):
        r, psi = order_parameter(theta[i])
        return go.Scatter(x=[0, r*np.cos(psi)], y=[0, r*np.sin(psi)], mode="lines+markers",
                          line=dict(color="#FFFFFF", width=3),
                          marker=dict(size=[1, 13], color="#FFFFFF"), hoverinfo="skip")

    data = [circle, dots(0), vec(0)]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        r, _ = order_parameter(theta[i])
        frames.append(go.Frame(name=str(k), traces=[1, 2], data=[dots(i), vec(i)],
            layout=dict(title=dict(text="<b>Kuramoto oscillators synchronising</b>"
                        f"<br><sup>N={N}, K={K}, order parameter r = {r:.2f}</sup>"))))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    lay = base_layout("Kuramoto oscillators synchronising",
                      f"N={N}, K={K}, the white arrow is the order parameter r")
    lay["xaxis"].update(range=[-1.3, 1.3], visible=False)
    lay["yaxis"].update(range=[-1.3, 1.3], visible=False, scaleanchor="x")
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_kuramoto_circle.html", frame_ms, prefix="time  ")


# 9  KURAMOTO ON A NETWORK, nodes coloured by phase

def kuramoto_network():
    rng = np.random.default_rng(3)
    N, T, dt, F, K, frame_ms = 45, 30.0, 0.01, 300, 1.4, 32
    G = watts_strogatz(N, 4, 0.18, seed=3)
    A = adjacency_matrix(G)
    pos = nx.spring_layout(G, seed=3, k=0.7)
    px = np.array([pos[i][0] for i in range(N)])
    py = np.array([pos[i][1] for i in range(N)])

    omega = rng.normal(0, 0.6, N)
    theta = np.empty((int(T/dt) + 1, N))
    theta[0] = rng.uniform(-np.pi, np.pi, N)
    for i in range(int(T/dt)):
        theta[i + 1] = rk4_step(network_rhs, theta[i], i*dt, dt, omega, K, A)
    idx = np.linspace(0, theta.shape[0] - 1, F).astype(int)

    ex, ey = [], []
    for u, v in G.edges():
        ex += [px[u], px[v], None]; ey += [py[u], py[v], None]
    edges = go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=DIM, width=1),
                       opacity=0.5, hoverinfo="skip")

    def nodes(i):
        c = (np.mod(theta[i] + np.pi, 2*np.pi)) / (2*np.pi)
        return go.Scatter(x=px, y=py, mode="markers",
                          marker=dict(size=18, color=c, colorscale=CYCLIC, cmin=0, cmax=1,
                                      line=dict(color="rgba(0,0,0,0.5)", width=1)),
                          hovertext=[f"node {j}" for j in range(N)], hoverinfo="text")

    data = [edges, nodes(0)]
    frames, names, labels = [], [], []
    for k, i in enumerate(idx):
        r, _ = order_parameter(theta[i])
        frames.append(go.Frame(name=str(k), traces=[1], data=[nodes(i)],
            layout=dict(title=dict(text="<b>Kuramoto on a small-world network</b>"
                        f"<br><sup>colour = phase, order parameter r = {r:.2f}</sup>"))))
        names.append(str(k)); labels.append(f"t={i*dt:.1f}")

    lay = base_layout("Kuramoto on a small-world network",
                      "node colour = phase, watch clusters lock together")
    lay["xaxis"].update(visible=False)
    lay["yaxis"].update(visible=False, scaleanchor="x")
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_kuramoto_network.html", frame_ms, prefix="time  ")


# 10  VECTOR-FIELD FLOW, particles advected in a 2D phase portrait

def vector_field_flow():
    A = np.array([[-0.30, -1.1], [1.1, -0.30]])   # stable spiral

    def field(s, t):
        return A @ s

    gx, gy = np.meshgrid(np.linspace(-3, 3, 17), np.linspace(-3, 3, 17))
    u = A[0, 0] * gx + A[0, 1] * gy
    v = A[1, 0] * gx + A[1, 1] * gy
    quiv = ff.create_quiver(gx, gy, u, v, scale=0.11, arrow_scale=0.35,
                            line=dict(color=DIM, width=1))
    quiv_tr = quiv.data[0]
    quiv_tr.hoverinfo = "skip"

    rng = np.random.default_rng(1)
    M, F, dt, frame_ms = 180, 300, 0.035, 28
    P = rng.uniform(-3, 3, (M, 2))
    P = P[np.argsort(np.arctan2(P[:, 1], P[:, 0]))]   # order colours by angle
    cols = rainbow(M)
    pos_hist = [P.copy()]
    for _ in range(F):
        for j in range(M):
            P[j] = rk4_step(field, P[j], 0.0, dt)
        out = ((np.abs(P[:, 0]) > 3.2) | (np.abs(P[:, 1]) > 3.2)
               | (np.hypot(P[:, 0], P[:, 1]) < 0.05))
        if out.any():
            P[out] = rng.uniform(-3, 3, (int(out.sum()), 2))
        pos_hist.append(P.copy())

    def dots(k):
        return go.Scatter(x=pos_hist[k][:, 0], y=pos_hist[k][:, 1], mode="markers",
                          marker=dict(size=6, color=cols), hoverinfo="skip")

    data = [quiv_tr, dots(0)]
    frames, names, labels = [], [], []
    for k in range(F + 1):
        frames.append(go.Frame(name=str(k), traces=[1], data=[dots(k)]))
        names.append(str(k)); labels.append(f"{k}")

    lay = base_layout("Phase-flow: particles riding a stable-spiral vector field",
                      "every particle drifts along the arrows toward the fixed point")
    lay["xaxis"].update(range=[-3.2, 3.2], title="x")
    lay["yaxis"].update(range=[-3.2, 3.2], title="y", scaleanchor="x")
    fig = go.Figure(data=data, layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_vector_field_flow.html", frame_ms, prefix="step ")


# 11  PITCHFORK POTENTIAL, a ball in a changing double-well

def pitchfork_potential():
    rs = np.concatenate([np.linspace(-1.0, 2.0, 110), np.linspace(2.0, -1.0, 110)])
    xg = np.linspace(-1.8, 1.8, 240)

    def V(x, r):
        return -0.5 * r * x**2 + 0.25 * x**4

    def ball(r):
        return np.sqrt(r) if r > 0 else 0.0   # follow the + branch

    cv = go.Scatter(x=xg, y=V(xg, rs[0]), mode="lines", line=dict(color=TEAL, width=3),
                    hoverinfo="skip")
    bx = ball(rs[0])
    bp = go.Scatter(x=[bx], y=[V(bx, rs[0])], mode="markers",
                    marker=dict(size=18, color=GOLD, line=dict(color="#FFF", width=1.5)))

    frames, names, labels = [], [], []
    for k, r in enumerate(rs):
        bx = ball(r)
        state = "one stable rest state" if r <= 0 else "symmetry broken, two wells"
        frames.append(go.Frame(name=str(k), traces=[0, 1], data=[
            go.Scatter(x=xg, y=V(xg, r)), go.Scatter(x=[bx], y=[V(bx, r)])],
            layout=dict(title=dict(text="<b>Supercritical pitchfork as a potential well</b>"
                        f"<br><sup>r = {r:+.2f}, {state}</sup>"))))
        names.append(str(k)); labels.append(f"{r:+.2f}")

    lay = base_layout("Supercritical pitchfork as a potential well",
                      f"r = {rs[0]:+.2f}, one stable rest state")
    lay["xaxis"].update(range=[-1.8, 1.8], title="state x")
    lay["yaxis"].update(range=[-1.1, 0.9], title="potential V(x)")
    fig = go.Figure(data=[cv, bp], layout=lay, frames=frames)
    return finalize(fig, names, labels, "anim_pitchfork_potential.html", 55, prefix="parameter r = ")


# INDEX PAGE

INTERACTIVE = [
    ("Module 1", "1D bifurcations", "Saddle-node, transcritical and both pitchforks, with the phase line and diagram.", "sim_1d_bifurcations.html"),
    ("Module 1", "Hopf bifurcation (2D)", "Soft and hard onset of a limit cycle, with the amplitude diagram and hysteresis.", "sim_2d_bifurcations.html"),
    ("Module 1", "Van der Pol limit cycle", "Slide mu from a near-circular orbit to stiff relaxation spikes.", "sim_van_der_pol.html"),
    ("Chaos", "Double pendulum", "Sliders for count, start angles, gravity, spread and speed.", "sim_double_pendulum.html"),
    ("Chaos", "Lorenz attractor", "Tune sigma, rho, beta live; drag to rotate the 3-D view.", "sim_lorenz.html"),
    ("Module 5", "Kuramoto sync", "Change coupling K, N and frequency spread; r updates live.", "sim_kuramoto_circle.html"),
    ("Maps", "Logistic cobweb", "Slide r through the period-doubling route to chaos.", "sim_logistic.html"),
    ("Module 1", "Hopf limit cycle", "Slide mu across the Hopf bifurcation (point vs cycle).", "sim_limit_cycle.html"),
    ("Module 1", "Pitchfork potential", "Slide r and watch the single well split in two.", "sim_pitchfork.html"),
    ("Module 3", "RPS replicator", "Slide the payoff: closed orbit, spiral-in, spiral-out.", "sim_rps.html"),
    ("Module 1", "Phase portrait", "Tune the matrix [[a,b],[c,d]] and classify the flow.", "sim_vector_field.html"),
]

CINEMATIC = [
    ("Chaos", "Lorenz attractor", "A comet traces the butterfly attractor in 3-D, rotatable.", "anim_lorenz_comet.html"),
    ("Chaos", "Rossler attractor", "The folded-band attractor that period-doubles to chaos.", "anim_rossler_comet.html"),
    ("Chaos", "Butterfly effect", "Two trajectories 0.001 apart track, then diverge wildly.", "anim_butterfly_effect.html"),
    ("Chaos", "Double pendulum", "Eight near-identical pendulums fan apart in slow motion.", "anim_double_pendulum.html"),
    ("Maps", "Logistic cobweb", "Sweep r and watch period-doubling 1, 2, 4, chaos.", "anim_logistic_cobweb.html"),
    ("Module 1", "Stuart-Landau limit cycle", "Every start spirals onto the same Hopf limit cycle.", "anim_limit_cycle.html"),
    ("Module 3", "RPS on the simplex", "Replicator orbits: closed, spiral-in, spiral-out.", "anim_rps_simplex.html"),
    ("Module 5", "Kuramoto on a circle", "Phases clump up and the order parameter r grows.", "anim_kuramoto_circle.html"),
    ("Module 5", "Kuramoto on a network", "Nodes coloured by phase lock into clusters.", "anim_kuramoto_network.html"),
    ("Module 1", "Phase-flow particles", "Particles ride a stable-spiral vector field inward.", "anim_vector_field_flow.html"),
    ("Module 1", "Pitchfork potential", "A ball in a well that splits in two as r increases.", "anim_pitchfork_potential.html"),
]


def build_index():
    def cards(items):
        return "\n".join(
            f'''      <a class="card" href="{fn}">
        <div class="badge">{mod}</div>
        <h3>{title}</h3>
        <p>{desc}</p>
        <span class="go">open</span>
      </a>''' for mod, title, desc, fn in items)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nonlinear Dynamics - Interactive Simulations</title>
<style>
  :root {{ --bg:{BG}; --card:{CARD}; --dim:{DIM}; --txt:#E2E8F0; --muted:#94A3B8;
           --teal:{TEAL}; --gold:{GOLD}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
          font-family:Inter,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:46px 24px 4px; text-align:center; }}
  header h1 {{ margin:0; font-size:2.1rem; letter-spacing:.3px; }}
  header p {{ color:var(--muted); margin:.5rem 0 0; font-size:1.02rem; }}
  h2.sec {{ max-width:1100px; margin:30px auto 0; padding:0 22px; font-size:1.15rem;
            color:var(--txt); }}
  h2.sec span {{ color:var(--muted); font-size:.92rem; font-weight:400; }}
  .grid {{ max-width:1100px; margin:14px auto 0; padding:0 20px;
           display:grid; gap:18px; grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--dim); border-radius:14px;
           padding:20px 20px 16px; text-decoration:none; color:var(--txt);
           transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease;
           display:flex; flex-direction:column; }}
  .card:hover {{ transform:translateY(-4px); border-color:var(--teal);
                 box-shadow:0 10px 30px rgba(14,116,144,.25); }}
  .badge {{ align-self:flex-start; font-size:.72rem; font-weight:700; letter-spacing:.5px;
            color:var(--bg); background:var(--gold); padding:3px 9px; border-radius:20px;
            text-transform:uppercase; }}
  .card h3 {{ margin:.7rem 0 .4rem; font-size:1.15rem; }}
  .card p {{ margin:0; color:var(--muted); font-size:.92rem; line-height:1.45; flex:1; }}
  .go {{ margin-top:12px; color:var(--teal); font-weight:600; font-size:.9rem; }}
  footer {{ text-align:center; color:var(--muted); padding:24px; font-size:.85rem; }}
</style></head><body>
  <header>
    <h1>Nonlinear Dynamics on Networks</h1>
    <p>Interactive labs you control, plus auto-playing animation clips.</p>
  </header>
  <h2 class="sec">Interactive simulations <span>move a slider and the simulation re-runs live</span></h2>
  <main class="grid">
{cards(INTERACTIVE)}
  </main>
  <h2 class="sec">Animation clips <span>auto-play; drag the slider, rotate the 3-D scenes</span></h2>
  <main class="grid">
{cards(CINEMATIC)}
  </main>
  <footer>Self-contained Plotly, drag to rotate 3-D, scroll to zoom, double-click to reset</footer>
</body></html>"""
    path = os.path.join(OUT, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print("  saved  media/interactive/index.html")


# MAIN

def main():
    print("Building interactive animations in media/interactive/")
    attractor_comet(lorenz, [1.0, 1.0, 1.0], 50.0, 0.01,
                    "Lorenz strange attractor", "σ=10, β=8/3, ρ=28, drag to rotate",
                    TEAL, "anim_lorenz_comet.html")
    attractor_comet(rossler, [1.0, 1.0, 1.0], 180.0, 0.03,
                    "Rössler attractor", "a=b=0.2, c=5.7, drag to rotate",
                    PURPLE, "anim_rossler_comet.html")
    butterfly()
    double_pendulum()
    logistic_cobweb()
    limit_cycle()
    rps_simplex()
    kuramoto_circle()
    kuramoto_network()
    vector_field_flow()
    pitchfork_potential()

    build_index()
    print("Done.")


if __name__ == "__main__":
    main()

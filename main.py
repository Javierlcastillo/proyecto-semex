import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
import agentpy as ap
from route import Route, Point

# NEW: import generated routes (created by svg_to_routes.py)
from generated_routes import load_routes

def build_example_routes() -> list[Route]:
    # load generated ROUTE_POLYLINES -> Route objects
    # samples_per_segment controls internal polyline resolution (keep low e.g. 2 for straight segments)
    return load_routes(samples_per_segment=2)

def _draw_arrow_for_segment(ax: Axes, p0: np.ndarray, p1: np.ndarray, color: str):
    dx, dy = (p1 - p0)
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    # arrow tail at p0 + 0.7*(p1-p0)
    tail = p0 + 0.7 * (p1 - p0)
    ax.arrow(tail[0], tail[1], ux * (L * 0.12), uy * (L * 0.12),
             head_width=3.0, head_length=6.0, fc=color, ec=color, length_includes_head=True)

def draw_routes(ax: Axes, routes: list[Route], colors: list[str]|None = None, linewidth: int = 2, draw_end_arrows: bool = True):
    if colors is None:
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']
    for i, r in enumerate(routes):
        pts = r.sample_even(400)
        ax.plot(pts[:,0], pts[:,1], color=colors[i % len(colors)], linewidth=linewidth)
        if draw_end_arrows and len(pts) > 2:
            # arrow at end following route direction
            _draw_arrow_for_segment(ax, pts[-3], pts[-1], colors[i % len(colors)])

class RouteDemoModel(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    This uses plain numpy arrays for positions but is wrapped into an AgentPy Model
    so you can call draw_routes(ax, routes) inside your AgentPy visualization.
    """
    def setup(self):
        self.routes = build_example_routes()
        self.n_agents = 30
        # assign each agent a random route
        self.agent_route = np.random.choice(len(self.routes), size=self.n_agents)
        # s position along route (distance)
        self.s = np.zeros(self.n_agents, dtype=float)
        # random speeds (world units per frame)
        self.speed = np.random.uniform(0.6, 2.0, size=self.n_agents)

    def step(self):
        # advance agents and wrap at end
        for i in range(self.n_agents):
            r = self.routes[self.agent_route[i]]
            self.s[i] += self.speed[i]
            if self.s[i] > r.length:
                self.s[i] -= r.length
        # store last computed xy for plotting convenience
        self.pos = np.vstack([ self.routes[self.agent_route[i]].pos_at(self.s[i]) for i in range(self.n_agents) ])

def run_matplotlib_demo():
    routes = build_example_routes()
    model = RouteDemoModel({'steps': 200})
    model.setup()

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    ax.set_xlim(-100, 500)
    ax.set_ylim(-100, 500)
    draw_routes(ax, routes)

    scat = ax.scatter([], [], s=30, c='k')

    def update(frame):
        model.step()
        scat.set_offsets(model.pos)
        return scat,

    ani = FuncAnimation(fig, update, frames=400, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    run_matplotlib_demo()
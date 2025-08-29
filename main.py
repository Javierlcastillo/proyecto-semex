import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
import agentpy as ap
from route import Route
from car import Car

from defined_routes import routes

class Model(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    """
    fig: plt.Figure
    ax: Axes
    
    routes: list[Route]
    agents: list[Car]

    def setup(self):
        self.routes = routes
        self.agents = [Car(self, r) for r in self.routes]

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(100, 700)
        self.ax.set_ylim(0, 600)

        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'cyan', 'magenta']

        for i, r in enumerate(self.routes):
            pts = r.sample_even(400)
            self.ax.plot(pts[:,0], pts[:,1], color=colors[i % len(colors)], linewidth=2, zorder=1)

        pos = np.array([a.position for a in self.agents])
        self.scat = self.ax.scatter(pos[:,0], pos[:,1], s=40, c='k', zorder=2)

        plt.ion()
        self.fig.show()

    def plot(self):
        pos = np.array([a.position for a in self.agents])
        self.scat.set_offsets(pos)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def step(self):
        # Advance cars, then repaint them
        for a in self.agents:
            if hasattr(a, "step"):
                a.step()
        self.plot()

model = Model({ 'steps': 200 })
model.run()
plt.ioff()
plt.show()
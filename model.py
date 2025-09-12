import agentpy as ap
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import json
import random
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import patches
from route import Route
from car import Car

from defined_routes import routes, route_colors
from defined_tlconnections import generate_tlconnections

from network_manager import NetManager
import os
import atexit
from datetime import datetime
from typing import Optional
from matplotlib.animation import PillowWriter

from traffic_light import TLConnection
from traffic_light_controller import TrafficLightController, TLCtrlMode

class Model(ap.Model):
    """
    Minimal model that keeps arrays of agents moving along assigned routes.
    """
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None
    step_text = None

    routes: list[Route]
    car_patches: list[patches.Rectangle] = []

    render_every: int = 10

    net: NetManager

    collision_flags: Dict[Car, bool] = {}

    ### TRAFFIC LIGHT CONTROLLER ###
    TlcCtrl: TrafficLightController
    TlcConn: list[TLConnection]

    # Flow-based spawning
    flow_data: Dict[str, Any] = {}
    car_spawn_queues: Dict[int, List[float]] = {}  # route_idx -> [spawn_times]
    active_cars: List[Car] = []
    steps_per_interval: int = 0
    minimum_spawn_distance: float = 50.0
    current_interval: int = 0

    # --- Lightweight GIF recording (Pillow) ---
    record_gif: bool = False
    gif_path: str = ""
    gif_fps: int = 20
    gif_dpi: int = 100
    _gif_writer: Optional[PillowWriter] = None
    # --- end GIF options ---

    def setup(self):
        self.net = NetManager()
        self.net.start()

        self.finished_cars = 0

        # --- OFFLINE EXPORT (single JSON after run) ---
        self.offline_export: bool = bool(getattr(self, "p", {}).get("offline_export", False))
        export_dir = str(getattr(self, "p", {}).get("offline_export_dir", "exports"))
        os.makedirs(export_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.offline_export_path: str = os.path.join(export_dir, f"run-{ts}.json")
        self._recorded_frames: list[dict] = []

        # make sure we always dump on shutdown
        atexit.register(self._dump_offline_json)
        # ------------------------------------------------
        # Mode switches (with defaults)
        p = getattr(self, "p", {})
        print("Model parameters:", p)

        # Training enabled?
        self.training_enabled: bool = bool(p.get('train', True))

        ### RENDERING ###
        self.render_every: int = int(p.get('render_every', 0)) 
        rendering = self.render_every > 0

        print("Rendering is", "enabled" if rendering else "disabled", " with interval", self.render_every)

        # Set up plotting if rendering OR recording GIF
        if rendering or self.record_gif:
            self.fig, self.ax = plt.subplots(figsize=(8, 8), squeeze=True)  # type: ignore
            self.ax.set_aspect('equal')  # type: ignore
            self.ax.set_xlim(100, 700)   # type: ignore
            self.ax.set_ylim(0, 600)     # type: ignore
            self.step_text = self.ax.text(105, 595, f"Step: {self.t}", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))  # type: ignore

            plt.ion() # type: ignore
            plt.show(block=False) # type: ignore

        self.routes = routes
        self.TlcConn = generate_tlconnections(self.ax)

        for tlc in self.TlcConn:
            tlc.traffic_light.plot()

        for i, route in enumerate(self.routes):
            route.plot(self.ax, color=route_colors[i] if i < len(route_colors) else "black")

        ### END RENDERING ###

        mode_key = str(getattr(self, "p", {}).get("tl_mode", "fixed")).strip().lower()
        if mode_key in ("qlearning", "ql", "q"):
            tl_mode = TLCtrlMode.QLEARNING
        elif mode_key in ("fixed", "f", "static"):
            tl_mode = TLCtrlMode.FIXED
        else:
            print(f"[TL] Unknown tl_mode='{mode_key}', falling back to FIXED")
            tl_mode = TLCtrlMode.FIXED

        self.TlcCtrl = TrafficLightController(self, self.TlcConn, tl_mode)

        self.policy_dir: str = str(p.get('policy_dir', 'checkpoints'))

        # Only autosave during training
        self.autosave_interval: int = int(p.get('autosave_interval', (300 if self.training_enabled else 0)))

        # Load flow data
        flow_path = p.get('flow_path', 'data/traffic_flow.json')
        self.load_flow_data(flow_path)
        
        # Initialize active cars list and spawn queues as well as cars finished
        self.active_cars = []
        self.car_spawn_queues = {i: [] for i in range(len(self.routes))}
        self.list_finished_cars = []
        
        # Calculate steps per interval based on time_interval_minutes
        minutes_per_step = 1 / 60  # assuming 1 step = 1 second
        self.steps_per_interval = int(self.flow_data.get("time_interval_minutes", 15) / minutes_per_step)
        self.minimum_spawn_distance = float(self.flow_data.get("minimum_spawn_distance", 50.0))
        
        # Schedule initial spawns
        self.schedule_spawns_for_interval(0)

        # GIF params (no heavy deps)
        self.record_gif = bool(p.get('record_gif', False))
        self.gif_fps = int(p.get('gif_fps', 20))
        self.gif_dpi = int(p.get('gif_dpi', 100))
        default_video_dir = os.path.join(os.getcwd(), "videos")
        os.makedirs(default_video_dir, exist_ok=True)
        default_gif_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.t}_steps.gif"
        self.gif_path = str(p.get('gif_path', os.path.join(default_video_dir, default_gif_name)))

        # Initialize Pillow GIF writer
        if self.record_gif:
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8), squeeze=True)  # type: ignore
                self.ax.set_aspect('equal')  # type: ignore
                self.ax.set_xlim(100, 700)   # type: ignore
                self.ax.set_ylim(0, 600)     # type: ignore
            self._gif_writer = PillowWriter(fps=self.gif_fps)
            self._gif_writer.setup(self.fig, self.gif_path, dpi=self.gif_dpi)
            atexit.register(self._finalize_gif)

    def step(self):
        # 0) initialize cars_finished counter per step to 0
        # 0) Check for new cars to spawn
        for route_idx, spawn_times in self.car_spawn_queues.items():
            while spawn_times and spawn_times[0] <= self.t:
                if self.can_spawn_safely(route_idx):
                    self.spawn_car(route_idx)
                    spawn_times.pop(0)
                else:
                    # Can't spawn yet, try again next step
                    break

        # 1) Update traffic light states
        self.TlcCtrl.step()

        # 2) Decision + movement for all active cars
        for car in list(self.active_cars):
            car.step()

        # 3) Remove cars that have completed their routes
        self.remove_completed_cars()

        # 4) Check if we need to move to next interval
        self.check_new_interval()

        # 5) Shared caches (collisions, TTC) 
        self._compute_collision_flags()

        if self.offline_export:
            self._record_frame()

        # 6) Capture every frame to GIF (offscreen)
        if self.record_gif:
            self._capture_gif_frame()

        # 7) Render sparsely
        if self.render_every > 0 and (self.t % self.render_every == 0):
            self.plot()

        # 8) Push state to clients
        # 9) append to list of finished cars
        self.list_finished_cars.append(self.finished_cars)
        self.push_state()

    def plot(self):
        if not self.ax:
            return
        
        if self.step_text:
            self.step_text.set_text(f"Step: {self.t}")

        for car in self.active_cars:
            car.plot(self.ax)

        self.fig.canvas.draw_idle()      # type: ignore[union-attr]
        self.fig.canvas.flush_events()   # type: ignore[union-attr]

    # --- helpers for drawing/capture ---
    def _draw_frame(self) -> None:
        if not self.ax:
            return
        
        if self.step_text:
            self.step_text.set_text(f"Step: {self.t}")

        for car in self.active_cars:
            car.plot(self.ax)

        if self.fig:
            self.fig.canvas.draw() # type: ignore

    def _capture_gif_frame(self) -> None:
        if self._gif_writer is None:
            return
        self._draw_frame()
        self._gif_writer.grab_frame() # type: ignore

    def load_flow_data(self, flow_path: str):
        """Load traffic flow data from JSON file."""
        try:
            with open(flow_path, 'r') as f:
                self.flow_data = json.load(f)
            print(f"Successfully loaded flow data from {flow_path}")
        except Exception as e:
            print(f"Failed to load flow data: {e}")
            # Create default flow data
            self.flow_data = {
                "routes": {f"route{i}": [3] * 8 for i in range(len(self.routes))},
                "time_interval_minutes": 15,
                "minimum_spawn_distance": 50,
                "spawn_delay_seconds": 5
            }
            print("Using default flow data")
    
    def schedule_spawns_for_interval(self, interval_idx: int):
        """Schedule car spawns for the given time interval."""
        if interval_idx >= len(next(iter(self.flow_data["routes"].values()))):
            print(f"No more flow data available after interval {interval_idx}")
            return
            
        self.current_interval = interval_idx
        
        # Clear existing spawn queues
        for queue in self.car_spawn_queues.values():
            queue.clear()
            
        # Schedule spawns for each route
        for route_idx, _route in enumerate(self.routes):
            route_key = f"route{route_idx}"
            if route_key in self.flow_data["routes"]:
                cars_to_spawn = self.flow_data["routes"][route_key][interval_idx]
                
                # Distribute cars evenly within the interval
                if cars_to_spawn > 0:
                    spawn_delay = self.flow_data.get("spawn_delay_seconds", 5)
                    steps_between_spawns = min(
                        self.steps_per_interval // cars_to_spawn,
                        spawn_delay  # minimum delay between spawns
                    )
                    
                    for i in range(cars_to_spawn):
                        # Add some randomness to avoid synchronized spawns across routes
                        spawn_step = self.t + i * steps_between_spawns + random.randint(0, min(5, steps_between_spawns//2))
                        self.car_spawn_queues[route_idx].append(spawn_step)
    
    def can_spawn_safely(self, route_idx: int) -> bool:
        """
        Check if it's safe to spawn a car by creating a "ghost" car and checking for overlaps.
        """
        target_route = self.routes[route_idx]
        
        # --- Create a "ghost" car's geometry ---
        x, y = target_route.pos_at(0)
        p1 = target_route.pos_at(1.0) # A point slightly ahead to get heading
        heading_rad = np.arctan2(p1[1] - y, p1[0] - x)
        
        w, h = 20.0, 10.0 # Standard car dimensions from car.py
        c, s = np.cos(heading_rad), np.sin(heading_rad)
        hw, hh = w/2.0, h/2.0
        
        local_corners = np.array([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        ])
        R = np.array([[c, -s], [s, c]])
        ghost_corners = local_corners @ R.T + np.array([x, y])
        # --- End of ghost geometry ---

        # Check for overlap with all active cars
        for car in self.active_cars:
            if car._sat_overlap(ghost_corners, car.corners):
                # Overlap detected, not safe to spawn
                return False
                    
        # No overlaps found
        return True
    
    def spawn_car(self, route_idx: int) -> None:
        """Create a new car on the specified route."""
        route = self.routes[route_idx]
        relevant_tlconnections = [tlc for tlc in self.TlcConn if tlc.route == route]
        
        new_car = Car(
            self,
            route=route,
            tlconnections=relevant_tlconnections
        )
        
        self.active_cars.append(new_car)
        print(f"Spawned car on route {route_idx}, total active cars: {len(self.active_cars)}")

    def _compute_collision_flags(self) -> None:
        """Broadphase (grid) + SAT narrowphase once per step for all cars."""
        cars = self.active_cars
        n = len(cars)
        if n == 0:
            self.collision_flags = {}
            return

        # Precompute corners and AABBs
        corners: List[np.ndarray] = [c.corners for c in cars]
        aabbs: List[Tuple[float, float, float, float]] = []
        for C in corners:
            xs = C[:, 0]; ys = C[:, 1]
            aabbs.append((float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())))

        # Uniform grid hashing on AABBs
        cell = max(max(c.width, c.height) for c in cars) * 2.0
        if cell <= 0:
            cell = 10.0
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i, (xmin, xmax, ymin, ymax) in enumerate(aabbs):
            gx0, gx1 = int(np.floor(xmin / cell)), int(np.floor(xmax / cell))
            gy0, gy1 = int(np.floor(ymin / cell)), int(np.floor(ymax / cell))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    grid.setdefault((gx, gy), []).append(i)

        # Candidate pairs from shared cells
        pairs: set[Tuple[int, int]] = set()
        for ids in grid.values():
            k = len(ids)
            if k < 2:
                continue
            for a in range(k):
                ia = ids[a]
                for b in range(a + 1, k):
                    ib = ids[b]
                    # Quick AABB overlap test to prune
                    ax0, ax1, ay0, ay1 = aabbs[ia]
                    bx0, bx1, by0, by1 = aabbs[ib]
                    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                        continue
                    pairs.add((min(ia, ib), max(ia, ib)))

        # Narrowphase SAT on candidates
        collided = [False] * n
        for ia, ib in pairs:
            if cars[ia]._sat_overlap(corners[ia], corners[ib]):
                collided[ia] = True
                collided[ib] = True

        # Publish per-car flags
        self.collision_flags = {cars[i]: collided[i] for i in range(n)}

    def remove_completed_cars(self):
        """Remove cars that have completed their routes."""
        completed_cars = [car for car in self.active_cars if car.s >= car.route.length]
        for car in completed_cars:
            if car in self.active_cars:
                self.active_cars.remove(car)
        
        if completed_cars:
            print(f"Removed {len(completed_cars)} completed cars. Active cars: {len(self.active_cars)}")

    def check_new_interval(self):
        """Check if we need to move to the next time interval."""
        interval_idx = self.t // self.steps_per_interval
        if interval_idx > self.current_interval:
            self.schedule_spawns_for_interval(interval_idx)


    def _record_frame(self) -> None:
        """Capture one frame in a Unity-friendly schema."""
        # cars
        cars_payload = []
        for car in self.active_cars:
            x, y = float(car.pos[0]), float(car.pos[1])
            yaw_deg = float(np.degrees(car.heading_in_radians))
            # best-effort extras (fallbacks if attributes don’t exist)
            speed = float(getattr(car, "v", getattr(car, "speed", 0.0)))
            s_param = float(getattr(car, "s", 0.0))
            length = float(getattr(car, "length", 4.0))
            width  = float(getattr(car, "width", 1.8))
            route_name = getattr(getattr(car, "route", None), "name", "")
            cars_payload.append({
                "id": int(id(car)),
                "position": [x, y],
                "heading": yaw_deg,      # degrees; the Unity player maps this for you
                "route_id": int(id(car.route)) if getattr(car, "route", None) else 0,
                "speed": speed,
                "s": s_param,
                "route_name": route_name,
                "length": length,
                "width": width
            })
        

        # traffic lights (unique set)
        seen = set()
        lights_payload = []
        for tlc in self.TlcConn:
            tl = tlc.traffic_light
            if tl in seen:
                continue
            seen.add(tl)
            # best-effort rotation; 0 if not present
            
            rot_deg = float(getattr(tl, "rotation", getattr(tl, "rotation_deg", 0.0)))
            lights_payload.append({
                "light_id": str(getattr(tl, "name", id(tl))),
                "state": str(tl.state.value),
                "x": float(getattr(tl, "x", 0.0)),
                "y": float(getattr(tl, "y", 0.0)),
                "rotation_deg": rot_deg
            })

        self._recorded_frames.append({
            "t": float(self.t),
            "cars": cars_payload,
            "lights": lights_payload
        })

    def _dump_offline_json(self) -> None:
            """Write the recorded frames to a single JSON file (once)."""
            if not getattr(self, "offline_export", False):
                return
            if not getattr(self, "_recorded_frames", None):
                return
            try:
                payload = {
                    "meta": {
                        "created_at": datetime.now().isoformat(),
                        "total_frames": len(self._recorded_frames),
                        "dt_hint_seconds": 1.0,  # tweak if your step isn’t 1s
                    },
                    "frames": self._recorded_frames
                }
                with open(self.offline_export_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"[EXPORT] Offline run saved to: {self.offline_export_path}")
            except Exception as e:
                print(f"[EXPORT] Failed to write offline JSON: {e}")
            finally:
                # avoid double-writing if atexit fires twice
                self._recorded_frames = []
                
    def push_state(self):
        state: Any = {
            "cars": [
            {
                "id": id(car),
                "position": car.pos,
                "heading": np.degrees(car.heading_in_radians),
                "route_id": id(car.route)
            } for car in self.active_cars
            ],
            "traffic_lights": [
                {
                    "id": id(tl),
                    "state": tl.state.value,
                    "x": tl.x,
                    "y": tl.y,
                    "rotation_deg": float(getattr(tl, "rotation", getattr(tl, "rotation_deg", 0.0)))
                } for tl in set([tlc.traffic_light for tlc in self.TlcConn])
            ]
        }

        self.net.push_state(state)

    def _finalize_gif(self) -> None:
        if self._gif_writer is not None:
            try:
                self._gif_writer.finish()
                print(f"GIF saved to: {self.gif_path}")
            except Exception as e:
                print(f"Error finalizing GIF: {e}")
            finally:
                self._gif_writer = None

    def end(self):
        """
        Called at the end of the simulation run.
        Calculates and prints the total number of finished cars.
        """
        print(f"Total cars finished: {self.finished_cars}")
        df = pd.DataFrame(self.list_finished_cars, columns=["finished_per_stp"])
        valor_tl = "h"
        df.to_csv(f"cars_per_step{valor_tl}.csv", index=False)

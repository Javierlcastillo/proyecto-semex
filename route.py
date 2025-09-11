import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Sequence, Optional
import math
from matplotlib.axes import Axes

Point = Tuple[float, float]
ParamFunc = Callable[[np.float64], Point] # t in [0, 1] -> (x, y)

def _calculate_cumulative_lengths(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the cumulative lengths of a series of 2D points.
    """
    if points.shape[0] == 0:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))

def _interpolate_points(points: npt.NDArray[np.float64], s_vals: npt.NDArray[np.float64], cumlen: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Interpolate points along the route at specified arc lengths.
    """
    if points.shape[0] == 0:
        return np.zeros((len(s_vals), 2), dtype=float)
    s = np.clip(s_vals, 0.0, cumlen[-1])
    idx = np.searchsorted(cumlen, s, side="right") - 1
    idx = np.clip(idx, 0, points.shape[0] - 2)
    seg_len = cumlen[idx + 1] - cumlen[idx]
    alpha = np.where(seg_len > 0.0, (s - cumlen[idx]) / seg_len, 0.0)
    p0 = points[idx]
    p1 = points[idx + 1]
    return (1 - alpha)[:, None] * p0 + alpha[:, None] * p1


@dataclass
class Route:
    """
    A route composed from one or more parametric segments.
    Internally we build a high-resolution polyline and compute cumulative length; all public sampling is done by arc-length.
    """
    segments: Sequence[ParamFunc]
    _cumulative_lengths: npt.NDArray[np.float64] = field(init=False)
    samples_per_segment: int = 600 # Controls arc-length approximation accuracy
    name: str = "Generic"
    
    length: np.float32 = np.float32(0.0)

    def __post_init__(self):
        if not self.segments:
            self._points = np.zeros((0, 2), dtype=float)
            self._cumulative_lengths = np.array([0.0], dtype=float)
            self.length = np.float32(0.0)
            return
        
        pts_list: List[npt.NDArray[np.float64]] = []
        for i, f in enumerate(self.segments):
            t = np.linspace(0.0, 1.0, self.samples_per_segment, endpoint=(i == len(self.segments) - 1))
            seg_pts = np.asarray([f(tt) for tt in t], dtype=np.float64)
            pts_list.append(seg_pts)
        all_pts = np.vstack(pts_list)

        # Remove duplicate consecutive points that can appear at segment boundaries
        diffs = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
        keep = np.concatenate(([True], diffs > 1e-9))
        self._points = all_pts[keep]
        self._cumulative_lengths = _calculate_cumulative_lengths(self._points)
        self.length = np.float32(self._cumulative_lengths[-1])

    def pos_at(self, s: float) -> Point:
        """Return (x, y) at distance s along the route (clipped to [0, length])"""
        if self.length == 0.0:
            return (0.0, 0.0)
        p = _interpolate_points(self._points, np.asarray([s], dtype=float), self._cumulative_lengths)[0]
        return float(p[0]), float(p[1])

    def sample_even(self, n: int) -> npt.NDArray[np.float64]:
        """Return n points evenly spaced by world-distance along the route."""
        if n <= 0:
            return np.zeros((0, 2), dtype=float)
        if self.length == 0.0:
            return np.tile(self._points[:1], (n, 1))
        s = np.linspace(0.0, self.length, n, dtype=np.float64)
        return _interpolate_points(self._points, s, self._cumulative_lengths)
    
    def plot(self, ax: Axes, color: str = 'black'):
        pts = self.sample_even(500)
        ax.plot(pts[:,0], pts[:,1], color=color, linewidth=2, zorder=1) # type: ignore

    @classmethod
    def from_polyline(cls, points: Sequence[Point], samples_per_segment: int = 2) -> "Route":
        """Construct a route from straight segments (polyline)."""
        def make_line(p0: Point, p1: Point) -> ParamFunc:
            def f(t: float) -> Point:
                return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            return f
        segs: List[ParamFunc] = []
        pts = list(points)
        if len(pts) < 2:
            return cls([], samples_per_segment=samples_per_segment)
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append(make_line(a, b))
        return cls(segs, samples_per_segment=max(2, samples_per_segment))

    @classmethod
    def from_arc(cls, center: Point, radius: float, start_angle: float, end_angle: float, clockwise: bool = False, samples_per_segment: int = 600) -> "Route":
        """Circular arc (angles in radians). If clockwise=True, angles are interpreted decreasingly."""
        if clockwise:
            start_angle, end_angle = end_angle, start_angle
        def f(t: float) -> Point:
            theta = start_angle + t * (end_angle - start_angle)
            return (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta))
        return cls([f], samples_per_segment=samples_per_segment)
    
    @classmethod
    def from_quadratic_bezier(cls, p0: Point, p1: Point, p2: Point, samples_per_segment: int = 800) -> "Route":
        """Construct a route from a quadratic Bezier curve."""
        def f(t: float) -> Point:
            u = 1 - t
            x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
            y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
            return (x, y)
        return cls([f], samples_per_segment=samples_per_segment)
    
    @classmethod
    def from_cubic_bezier(cls, p0: Point, p1: Point, p2: Point, p3: Point, samples_per_segment: int = 1200) -> "Route":
        """Construct a route from a cubic Bezier curve."""
        def f(t: float) -> Point:
            u = 1 - t
            x = (u**3)*p0[0] + 3*(u**2)*t*p1[0] + 3*u*(t**2)*p2[0] + (t**3)*p3[0]
            y = (u**3)*p0[1] + 3*(u**2)*t*p1[1] + 3*u*(t**2)*p2[1] + (t**3)*p3[1]
            return (x, y)
        return cls([f], samples_per_segment=samples_per_segment)
    
    @classmethod
    def join(cls, routes: Sequence["Route"], samples_per_segment: Optional[int] = None) -> "Route":
        """Join multiple Route instances into one composite route. Uses their sampled polylines."""
        segs: List[ParamFunc] = []
        for r in routes:
            # convert each route's piecewise polyline into parametric straight segments
            pts = r._points
            for a, b in zip(pts[:-1], pts[1:]):
                    def make_line(a: npt.NDArray[np.float64] = a, b: npt.NDArray[np.float64] = b):
                        def f(t: float) -> Point:
                            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                        return f
                    segs.append(make_line())
        return cls(segs, samples_per_segment=(samples_per_segment or 2))

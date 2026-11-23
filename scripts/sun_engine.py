"""Clear-sky direct sunlight on a local east/north/up Cartesian model (metres).

NOAA/Meeus solar coordinates use the geometric solar centre, without a weather-
dependent refraction correction. All input clock times must carry a UTC offset.
The ray tracer checks both sides of every opaque triangle, including the target.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import asin, atan2, cos, degrees, isfinite, radians, sin, tan

import numpy as np


@dataclass(frozen=True)
class SunPosition:
    altitude: float
    azimuth: float
    direction: np.ndarray
    equation_of_time_minutes: float


def solar_position(moment, latitude, longitude):
    """Return geometric altitude, clockwise-from-north azimuth, and unit ENU."""
    if moment.utcoffset() is None:
        raise ValueError("solar_position needs a timezone-aware datetime")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("latitude or longitude outside geographical bounds")
    utc = moment.astimezone(timezone.utc)
    # The reference epoch is Julian date 2451545 (2000-01-01 12:00 UTC).
    days = (utc - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400
    century = days / 36525
    mean_longitude = radians((280.46646 + century * (36000.76983 + .0003032 * century)) % 360)
    mean_anomaly = radians(357.52911 + century * (35999.05029 - .0001537 * century))
    eccentricity = .016708634 - century * (.000042037 + .0000001267 * century)
    centre = ((1.914602 - century * (.004817 + .000014 * century)) * sin(mean_anomaly)
              + (.019993 - .000101 * century) * sin(2 * mean_anomaly)
              + .000289 * sin(3 * mean_anomaly))
    omega = radians(125.04 - 1934.136 * century)
    apparent_lon = mean_longitude + radians(centre - .00569 - .00478 * sin(omega))
    obliquity_seconds = 21.448 - century * (46.815 + century * (.00059 - century * .001813))
    obliquity = radians(23 + (26 + obliquity_seconds / 60) / 60 + .00256 * cos(omega))
    declination = asin(sin(obliquity) * sin(apparent_lon))
    y = tan(obliquity / 2) ** 2
    minutes = 4 * degrees(
        y * sin(2 * mean_longitude) - 2 * eccentricity * sin(mean_anomaly)
        + 4 * eccentricity * y * sin(mean_anomaly) * cos(2 * mean_longitude)
        - .5 * y**2 * sin(4 * mean_longitude) - 1.25 * eccentricity**2 * sin(2 * mean_anomaly)
    )
    utc_minutes = utc.hour * 60 + utc.minute + (utc.second + utc.microsecond / 1e6) / 60
    hour_angle = radians(((utc_minutes + minutes + 4 * longitude) % 1440) / 4 - 180)
    latitude_rad = radians(latitude)
    east = -cos(declination) * sin(hour_angle)
    north = sin(declination) * cos(latitude_rad) - cos(declination) * cos(hour_angle) * sin(latitude_rad)
    up = sin(declination) * sin(latitude_rad) + cos(declination) * cos(hour_angle) * cos(latitude_rad)
    return SunPosition(degrees(asin(np.clip(up, -1, 1))), degrees(atan2(east, north)) % 360,
                       np.array([east, north, up]), minutes)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersects(a, b, c, d, length_tol, area_tol):
    if np.any(np.maximum(a, b) + length_tol < np.minimum(c, d)) or np.any(
            np.maximum(c, d) + length_tol < np.minimum(a, b)):
        return False
    turns = np.array([_cross(b-a, c-a), _cross(b-a, d-a), _cross(d-c, a-c), _cross(d-c, b-c)])
    sides = np.where(abs(turns) <= area_tol, 0, np.sign(turns))
    return sides[0] * sides[1] <= 0 and sides[2] * sides[3] <= 0


def _triangulate(polygon):
    """Ear-clip a simple polygon of either winding, preserving concave notches."""
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("expected a simple polygon with >=3 finite XY pairs")
    scale = max(float(np.ptp(points, axis=0).max()), 1.)
    length_tol = 1e-10 * scale
    area_tol = 1e-12 * scale**2
    # Ignore a repeated closing point, duplicate consecutive vertices and points
    # on straight boundary runs. A reversal or self-crossing is an input error.
    keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > length_tol
    points = points[keep]
    while len(points) >= 3:
        prev = points - np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0) - points
        straight = abs(_cross(prev, following)) <= area_tol
        if not straight.any():
            break
        if np.any(straight & (np.einsum("ij,ij->i", prev, following) < 0)):
            raise ValueError("polygon doubles back on an edge")
        points = points[~straight]
    if len(points) < 3:
        raise ValueError("degenerate footprint")
    for i in range(len(points)):
        for j in range(i + 2, len(points)):
            if i == 0 and j == len(points) - 1:
                continue
            if _intersects(points[i], points[i+1], points[j], points[(j+1) % len(points)],
                           length_tol, area_tol):
                raise ValueError("footprint boundary crosses or touches itself")
    local = points - points[0]
    area_twice = _cross(local, np.roll(local, -1, axis=0)).sum()
    if abs(area_twice) <= area_tol:
        raise ValueError("zero-area footprint")
    if area_twice < 0:
        points = points[::-1].copy()
    remaining = list(range(len(points)))
    triangles = []
    while len(remaining) > 3:
        for index, current in enumerate(remaining):
            before = remaining[index - 1]
            after = remaining[(index + 1) % len(remaining)]
            a, b, c = points[[before, current, after]]
            if _cross(b-a, c-a) <= area_tol:
                continue
            others = points[[p for p in remaining if p not in (before, current, after)]]
            inside = ((_cross(b-a, others-a) >= -area_tol) & (_cross(c-b, others-b) >= -area_tol)
                      & (_cross(a-c, others-c) >= -area_tol))
            if not inside.any():
                triangles.append([before, current, after])
                remaining.pop(index)
                break
        else:
            raise ValueError("could not triangulate footprint; check for intersecting edges")
    triangles.append(remaining)
    return points, np.array(triangles, dtype=int)


def prism_triangles(polygon, bottom, top):
    """Closed triangular surface between absolute elevations bottom and top."""
    if not isfinite(bottom) or not isfinite(top) or top <= bottom:
        raise ValueError("prism top must exceed its base elevation")
    points, caps = _triangulate(polygon)
    lower = np.column_stack([points, np.full(len(points), bottom)])
    upper = np.column_stack([points, np.full(len(points), top)])
    next_lower = np.roll(lower, -1, axis=0)
    next_upper = np.roll(upper, -1, axis=0)
    return np.concatenate([upper[caps], lower[caps[:, ::-1]], np.stack([lower, next_lower, next_upper], axis=1),
                           np.stack([lower, next_upper, upper], axis=1)])


class Mesh:
    """3D scene of opaque prisms. Each label can identify multiple closed parts."""

    def __init__(self, triangles, owners, ids):
        self.ids = tuple(ids)
        self.owners = np.asarray(owners, dtype=np.int32)
        self.triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        self.vertex = self.triangles[:, 0]
        self.edge1 = self.triangles[:, 1] - self.vertex
        self.edge2 = self.triangles[:, 2] - self.vertex
        self.normal = np.cross(self.edge1, self.edge2)
        self.area_twice = np.linalg.norm(self.normal, axis=1)

    @classmethod
    def from_prisms(cls, prisms):
        """Build from (owner_id, polygon_xy, bottom_z_absolute, top_z_absolute)."""
        labels = {}
        triangles, owners = [], []
        for label, polygon, bottom, top in prisms:
            try:
                part = prism_triangles(polygon, bottom, top)
            except ValueError as error:
                raise ValueError(f"Invalid prism {label}: {error}") from error
            index = labels.setdefault(label, len(labels))
            triangles.append(part)
            owners.extend([index] * len(part))
        return cls(np.concatenate(triangles) if triangles else [], owners, labels)

    def classify(self, origins, normals, direction):
        """Nearest owner index for each ray, -1 for sunlight, -2 for no incidence.

        All rays share a direction, pointing *towards* the sun. Parallel batch
        Möller–Trumbore uses scalar triple products and bounds peak matrix size.
        """
        origins = np.asarray(origins, dtype=float)
        normals = np.asarray(normals, dtype=float)
        ray = np.asarray(direction, dtype=float)
        if (origins.ndim != 2 or origins.shape[1] != 3 or normals.ndim != 2
                or normals.shape not in ((len(origins), 2), origins.shape) or ray.shape != (3,)):
            raise ValueError("expected Nx3 points, Nx2 or Nx3 normals and one 3-vector")
        norm = np.linalg.norm(ray)
        if (not np.isfinite(origins).all() or not np.isfinite(normals).all()
                or np.any(np.linalg.norm(normals, axis=1) == 0) or not np.isfinite(norm) or norm == 0):
            raise ValueError("invalid vector or coordinates")
        ray = ray / norm
        result = np.full(len(origins), -2, dtype=np.int32)
        if ray[2] <= 0:
            return result
        front = np.flatnonzero(normals @ ray[:normals.shape[1]] > 1e-12)
        result[front] = -1
        if not len(front) or not len(self.vertex):
            return result
        det = -(self.normal @ ray)
        valid = abs(det) > 1e-12 * self.area_twice
        if not valid.any():
            return result
        vertex, det = self.vertex[valid], det[valid, None]
        coeff_u = np.cross(ray, self.edge2[valid]) / det
        coeff_v = np.cross(self.edge1[valid], ray) / det
        coeff_t = self.normal[valid] / det
        u_origin = np.einsum("ij,ij->i", vertex, coeff_u)
        v_origin = np.einsum("ij,ij->i", vertex, coeff_v)
        t_origin = np.einsum("ij,ij->i", vertex, coeff_t)
        owners = self.owners[valid]
        chunk = max(1, 200000 // len(vertex))
        for start in range(0, len(front), chunk):
            indices = front[start:start+chunk]
            points = origins[indices]
            u = points @ coeff_u.T - u_origin
            v = points @ coeff_v.T - v_origin
            distance = points @ coeff_t.T - t_origin
            hit = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance > 1e-7)
            distance[~hit] = np.inf
            closest = distance.argmin(axis=1)
            blocked = np.isfinite(distance[np.arange(len(points)), closest])
            result[indices[blocked]] = owners[closest[blocked]]
        return result


def time_bins(start, end, step_minutes=5):
    """Half-open clock bins and midpoint samples. Last bin may be shorter."""
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("time windows need timezone-aware datetimes")
    if not isfinite(step_minutes) or step_minutes <= 0 or end <= start:
        raise ValueError("expected a positive step and a nonempty window")
    step = timedelta(minutes=step_minutes)
    if step == timedelta(0):
        raise ValueError("step smaller than datetime resolution")
    bins = []
    while start < end:
        stop = min(start + step, end)
        bins.append((start, stop, start + (stop-start)/2))
        start = stop
    return bins


def lit_intervals(lit, bins):
    """Merge adjacent lit bins, keeping darkness and unsampled gaps separate."""
    if len(lit) != len(bins):
        raise ValueError("one visibility flag is required for each time bin")
    intervals = []
    for visible, (start, end, _mid) in zip(lit, bins):
        if visible:
            if intervals and intervals[-1][1] == start:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
    return intervals

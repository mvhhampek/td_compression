import numpy as np
import math
from typing import Dict, Tuple

def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Computes a 3x3 rotation matrix from an axis and an angle."""
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle)
    b = math.sin(angle)
    c = 1 - a
    x, y, z = axis
    
    R = np.array([
        [a + x*x*c,   x*y*c - z*b, x*z*c + y*b],
        [y*x*c + z*b, a + y*y*c,   y*z*c - x*b],
        [z*x*c - y*b, z*y*c + x*b, a + z*z*c]
    ], dtype=float)
    return R

def ransac_plane(
    points: np.ndarray,
    iters: int = 200,
    dist_thresh: float = 0.2,
    min_inliers: int = 500,
    seed: int = 0,
    z_normal_thresh: float = 0.85
) -> Dict[str, np.ndarray]:
    """Plane RANSAC. Plane: n*x + d = 0, n is unit."""
    rng = np.random.default_rng(seed)
    P = np.asarray(points, dtype=float)
    N = int(P.shape[0])
    if N < 3:
        raise ValueError("Need at least 3 points")

    best = (-1, None, None, None)
    for _ in range(int(iters)):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = P[idx]
        n = np.cross(p2 - p1, p3 - p1)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn
        
        if abs(n[2]) < z_normal_thresh:
            continue 

        d = -float(np.dot(n, p1))
        dist = np.abs(P @ n + d)
        inliers = dist < float(dist_thresh)
        cnt = int(inliers.sum())
        if cnt > best[0] and cnt >= int(min_inliers):
            best = (cnt, n, d, inliers)

    if best[1] is None:
        n = np.array([0.0, 0.0, 1.0])
        d = 0.0
        inliers = np.zeros(N, dtype=bool)
    else:
        _, n, d, inliers = best

    if n[2] < 0:
        n = -n
        d = -d

    point = -d * n
    return {"normal": n, "d": np.array([d], dtype=float), "point": point, "inliers": inliers}


def align_ground(
    points: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps plane_normal -> +Z and translates so plane passes through z=0."""
    pts = np.asarray(points, dtype=float)
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    z = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(n, z), -1.0, 1.0))

    if abs(dot - 1.0) < 1e-9:
        R = np.eye(3)
    elif abs(dot + 1.0) < 1e-9:
        axis = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(n, np.array([0.0, 1.0, 0.0]))
        R = _rodrigues(axis, math.pi)
    else:
        axis = np.cross(n, z)
        angle = math.atan2(np.linalg.norm(axis), dot)
        R = _rodrigues(axis, angle)

    p0 = np.asarray(plane_point, dtype=float)
    p0r = R @ p0
    t = np.array([0.0, 0.0, -p0r[2]], dtype=float)

    pts_a = (R @ pts.T).T + t
    return pts_a, R, t
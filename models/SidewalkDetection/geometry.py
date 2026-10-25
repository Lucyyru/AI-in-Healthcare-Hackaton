"""NumPy ray/ground intersection with arbitrary camera pitch and roll."""
import numpy as np


def ground_basis(up_camera):
    down = -np.asarray(up_camera, dtype=float)
    down /= np.linalg.norm(down)
    forward = np.array([0., 0., 1.])
    forward -= np.dot(forward, down) * down
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    return down, right, forward


def pixels_to_ground(points, k, up_camera, height_m, horizon_margin_px=20):
    """Return (ground_XZ, valid), preserving point order and using meters.

    Camera axes: x right, y down, z forward. up_camera is GeoCalib's UP
    vector. The ground satisfies down_camera . P = camera_height_m.
    Invalid/near-horizon rays are NaN, not invented intersections.
    """
    if height_m <= 0 or not np.isfinite(height_m):
        raise ValueError("Camera height must be positive and finite, in meters")
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    k = np.asarray(k, dtype=float)
    down, right, forward = ground_basis(up_camera)
    rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(k).T
    denominators = rays @ down
    line = np.linalg.solve(k.T, down)
    horizon_distance_px = denominators / np.linalg.norm(line[:2])
    valid = (np.isfinite(points).all(axis=1) & (denominators > 0)
             & (horizon_distance_px > horizon_margin_px))
    intersections = rays[valid] * (height_m / denominators[valid, None])
    ground = np.full((len(points), 2), np.nan)
    ground[valid] = np.column_stack((intersections @ right, intersections @ forward))
    return ground, valid


def _sorted_edge(points):
    depths, inverse, counts = np.unique(points[:, 1], return_inverse=True, return_counts=True)
    x = np.bincount(inverse, weights=points[:, 0], minlength=len(depths)) / counts
    return depths, x


def measure_width(left, right, calibration, height_m, horizon_margin_px=20,
                  frame_margin_px=10):
    """Median perpendicular width of the selected region, not minimum clearance.

    Excludes image-frame endpoints because a clipped floor mask need not mark
    a real wall/sidewalk edge. Returns result metadata and per-depth samples.
    """
    width, height = calibration["image_size"]
    ground_edges, kept, warnings = [], [], []
    for name, edge in [("left", left), ("right", right)]:
        edge = np.asarray(edge).reshape(-1, 2)
        ground, valid = pixels_to_ground(edge, calibration["K"], calibration["up_camera"],
                                         height_m, horizon_margin_px)
        interior = ((edge[:, 0] >= frame_margin_px) & (edge[:, 0] < width - frame_margin_px)
                    & (edge[:, 1] >= frame_margin_px) & (edge[:, 1] < height - frame_margin_px))
        if np.any(~interior):
            warnings.append(f"{name} boundary touches image frame; those points were excluded")
        valid &= interior & (ground[:, 1] > 0)
        ground_edges.append(ground[valid])
        kept.append(int(valid.sum()))
    report = {"width_m": None, "camera_height_m": float(height_m),
              "valid_left_points": kept[0], "valid_right_points": kept[1],
              "horizon_margin_px": horizon_margin_px, "frame_margin_px": frame_margin_px,
              "warnings": warnings}
    if min(kept) < 2:
        warnings.append("Insufficient ground boundary points")
        return report, np.empty((0, 5))
    zl, xl = _sorted_edge(ground_edges[0])
    zr, xr = _sorted_edge(ground_edges[1])
    near, far = max(zl[0], zr[0]), min(zl[-1], zr[-1])
    if near >= far:
        warnings.append("No overlapping ground-depth range")
        return report, np.empty((0, 5))
    depths = np.linspace(near, far, 50)
    left_x, right_x = np.interp(depths, zl, xl), np.interp(depths, zr, xr)
    centered = depths - depths.mean()
    denom = centered @ centered
    slope_l = centered @ (left_x - left_x.mean()) / denom
    slope_r = centered @ (right_x - right_x.mean()) / denom
    slope = (slope_l + slope_r) / 2
    correction = 1 / np.hypot(slope, 1)
    horizontal = np.abs(right_x - left_x)
    widths = horizontal * correction
    valid = np.isfinite(widths) & (widths > 0)
    if not valid.any():
        warnings.append("No positive finite width samples")
        return report, np.empty((0, 5))
    median = float(np.median(widths[valid]))
    p10, p90 = np.percentile(widths[valid], [10, 90])
    angle_difference = float(abs(np.degrees(np.arctan(slope_l) - np.arctan(slope_r))))
    if angle_difference > 10:
        warnings.append("Selected boundaries are not approximately parallel")
    if p90 - p10 > median * 0.5:
        warnings.append("Large width variation across depths; inspect the mask and boundary selection")
    report.update(width_m=median, width_p10_m=float(p10), width_p90_m=float(p90),
                  ground_depth_range_m=[float(near), float(far)],
                  direction_slope=float(slope), angle_correction=float(correction),
                  boundary_direction_difference_deg=angle_difference,
                  measurement_note="Median width of the selected visible region; not verified physical passage width")
    samples = np.column_stack((depths, left_x, right_x, horizontal, widths))
    return report, samples[valid]

"""CPU-testable orbit geometry and straight-alpha PNG conversion."""
import math
import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Cannot normalize a zero or non-finite vector')
    return vector / length


def infer_orbit_frame(cameras, center, up_override=None):
    """Fit capture plane; orient its normal toward the elevated drone cameras.

    DUSt3R has no gravity reference. The fitted plane assumes an approximately
    level full orbit, and --up-vector can override it for a tilted flight path.
    """
    positions = np.asarray([c['position'] for c in cameras], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if positions.shape[0] < 3 or positions.shape[1:] != (3,):
        raise ValueError('At least three cameras are needed to infer the orbit')
    if not np.isfinite(positions).all():
        raise ValueError('Non-finite camera positions')
    if up_override is not None:
        up = unit(up_override)
    else:
        _, singular, vh = np.linalg.svd(positions - positions.mean(axis=0), full_matrices=False)
        if singular[1] < 1e-6:
            raise ValueError('Camera path is nearly a line; supply --up-vector')
        up = vh[-1]
        toward_cameras = positions.mean(axis=0) - center
        sign = float(np.dot(up, toward_cameras))
        if abs(sign) < 1e-5:
            camera_up = -np.mean([np.asarray(c['rotation'])[:, 1] for c in cameras], axis=0)
            sign = float(np.dot(up, camera_up))
        if sign < 0:
            up = -up
        up = unit(up)
    first = positions[0] - center
    horizontal = first - np.dot(first, up) * up
    if np.linalg.norm(horizontal) < 1e-7:
        horizontal = np.eye(3)[np.argmin(np.abs(up))]
        horizontal -= np.dot(horizontal, up) * up
    axis_x = unit(horizontal)
    axis_y = unit(np.cross(up, axis_x))
    offsets = positions - center
    elevations = np.degrees(np.arcsin(np.clip(offsets @ up / np.linalg.norm(offsets, axis=1), -1, 1)))
    return axis_x, axis_y, up, float(np.median(elevations))


def orbit_camera(center, axis_x, axis_y, up, radius, azimuth_deg, elevation_deg):
    """Return OpenCV c2w (x right, y down, z forward), including at 90 degrees.

    At exactly 90 degrees azimuth changes camera roll; all centers coincide.
    This avoids a singular look-at matrix while retaining a meaningful top view.
    """
    azimuth, elevation = np.radians([azimuth_deg, elevation_deg])
    radial = np.cos(azimuth) * axis_x + np.sin(azimuth) * axis_y
    tangent = -np.sin(azimuth) * axis_x + np.cos(azimuth) * axis_y
    position = np.asarray(center) + radius * (np.cos(elevation) * radial + np.sin(elevation) * up)
    forward = unit(np.asarray(center) - position)
    right = unit(tangent)
    down = unit(np.cross(forward, right))
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, down, forward))
    c2w[:3, 3] = position
    return c2w


def azimuths(start, count):
    if count < 1:
        raise ValueError('Frame count must be positive')
    return np.mod(float(start) + np.arange(count) * 360.0 / count, 360.0)


def safe_radius(bound_radius, fov_x, fov_y, margin=1.2):
    if bound_radius <= 0 or margin < 1 or not 0 < min(fov_x, fov_y) < math.pi:
        raise ValueError('Invalid radius, field of view, or framing margin')
    return bound_radius * margin / math.sin(min(fov_x, fov_y) / 2)


def rgba_from_accumulated(rgb, alpha):
    """Unpremultiply a BLACK-background splat render using native opacity.

    RGB values never determine alpha; an opaque black tire remains opaque.
    Input RGB HWC, alpha HW (or HW1); output PNG-ready straight RGBA uint8.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or alpha.shape != rgb.shape[:2]:
        raise ValueError('Expected RGB HxWx3 and alpha HxW')
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError('Renderer returned non-finite RGB/alpha')
    alpha = np.clip(alpha, 0, 1)
    straight = np.divide(rgb, alpha[..., None], out=np.zeros_like(rgb), where=alpha[..., None] > 1e-8)
    rgba = np.concatenate((np.clip(straight, 0, 1), alpha[..., None]), axis=-1)
    return np.rint(rgba * 255).astype(np.uint8)


def alpha_bounds(alpha, threshold=1 / 255):
    alpha = np.asarray(alpha)
    ys, xs = np.where(alpha >= threshold)
    if not len(xs):
        raise ValueError('Render has no visible foreground opacity')
    height, width = alpha.shape
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    touches = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 2 or bbox[3] >= height - 2
    return bbox, bool(touches)


def elevation_directory(elevation):
    if float(elevation).is_integer():
        return f'elev_{int(elevation):03d}'
    return f'elev_{elevation:07.3f}'.rstrip('0').rstrip('.').replace('.', 'p')

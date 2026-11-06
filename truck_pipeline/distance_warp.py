"""Independent RGB + camera-Z depth backward-camera reprojection (CPU only)."""
import numpy as np


def mast3r_geometry(width, height, long_edge=512):
    """size!=224, square_ok=False geometry in dust3r.utils.image.load_images."""
    rw, rh = (round(n * long_edge / max(width, height)) for n in (width, height))
    cx, cy = rw // 2, rh // 2
    hw, hh = (2 * cx // 16) * 8, (2 * cy // 16) * 8
    if rw == rh:
        hh = 3 * hw / 4
    box = tuple(round(n) for n in (cx-hw, cy-hh, cx+hw, cy+hh))
    return (rw, rh), box


def align_depth(depth, size, camera, legacy=False, assume_center=False, long_edge=512):
    """Return original-grid depth and principal point; never stretch a cropped depth."""
    w, h = size
    if depth.dtype.kind not in 'fiu':
        raise ValueError('Depth must contain real numeric camera-Z values')
    if (camera['width'], camera['height']) != (w, h):
        raise ValueError('Camera dimensions do not match RGB (check EXIF orientation)')
    if legacy:
        resized, box = mast3r_geometry(w, h, long_edge)
        dw, dh = box[2]-box[0], box[3]-box[1]
        if depth.size != dw * dh or (depth.ndim == 2 and depth.shape != (dh, dw)):
            raise ValueError(f'MASt3R depth must have shape {(dh, dw)} or be flattened')
        depth = depth.reshape(dh, dw)
        # Pixel centers: PIL resize maps (u+.5)*scale-.5; crop subtracts box origin.
        x = (np.arange(w)+.5)*resized[0]/w-.5-box[0]
        y = (np.arange(h)+.5)*resized[1]/h-.5-box[1]
        valid_x = (x >= -.5) & (x < dw-.5)
        valid_y = (y >= -.5) & (y < dh-.5)
        ix = np.clip(np.floor(x+.5).astype(int), 0, dw-1)
        iy = np.clip(np.floor(y+.5).astype(int), 0, dh-1)
        aligned = depth[iy[:, None], ix[None, :]].astype(np.float64)
        aligned[~(valid_y[:, None] & valid_x[None, :])] = np.nan
        # Actual MASt3R intrinsics live on the cropped depth grid.
        if 'cx' in camera and 'cy' in camera:
            cx, cy = camera['cx'], camera['cy']
        elif assume_center:
            cx, cy = dw/2, dh/2
        else:
            raise ValueError('Legacy export omits optimized principal point. Supply cx/cy on the depth grid, or explicitly --assume-centered-principal-point')
        principal = ((cx+box[0]+.5)*w/resized[0]-.5,
                     (cy+box[1]+.5)*h/resized[1]-.5)
    else:
        if camera.get('intrinsics_grid') == 'mast3r-preprocessed':
            raise ValueError('Camera principal point is on MASt3R grid; use --layout mast3r-legacy')
        if depth.shape != (h, w):
            raise ValueError('Aligned depth must match RGB resolution; use --layout mast3r-legacy for old exports')
        if 'cx' not in camera or 'cy' not in camera:
            raise ValueError('Aligned camera requires explicit cx and cy in RGB pixel coordinates')
        aligned, principal = depth.astype(np.float64), (camera['cx'], camera['cy'])
    if not np.isfinite(principal).all():
        raise ValueError('Invalid principal point')
    return aligned, principal


def reference_depth(depth, fraction=.1):
    if not 0 < fraction <= 1:
        raise ValueError('Reference fraction must be in (0, 1]')
    h, w = depth.shape
    rw, rh = max(1, round(w*fraction)), max(1, round(h*fraction))
    x, y = (w-rw)//2, (h-rh)//2
    roi = depth[y:y+rh, x:x+rw]
    values = roi[np.isfinite(roi) & (roi > 0)]
    if not values.size:
        raise ValueError('No valid positive depth in central reference region')
    return float(np.median(values))


def warp(rgb, depth, principal, factor, z_ref, supersample=2):
    """Area splat pixel footprints, nearest-Z visibility per subpixel, area resolve.

    Each source pixel is a constant-Z square; no triangles bridge depth edges.
    Memory is O(H*W*supersample**2); source contributions are processed in chunks.
    Returns RGB uint8, binary validity and fractional coverage uint8.
    """
    if not np.isfinite(factor) or factor < 1 or not np.isfinite(z_ref) or z_ref <= 0:
        raise ValueError('factor must be finite >=1 and reference depth positive')
    if supersample not in (1, 2, 4):
        raise ValueError('supersample must be 1, 2 or 4')
    h, w = depth.shape
    if rgb.shape != (h, w, 3):
        raise ValueError('RGB/depth shape mismatch')
    valid = np.isfinite(depth) & (depth > 0)
    if factor == 1:
        return np.where(valid[..., None], rgb, 0).astype(np.uint8), valid.astype(np.uint8)*255, valid.astype(np.uint8)*255
    a = supersample
    oh, ow = h*a, w*a
    zbuffer = np.full(oh*ow, np.inf, np.float32)
    weight = np.zeros(oh*ow, np.float32)
    color = np.zeros((oh*ow, 3), np.float32)
    ids = np.flatnonzero(valid)
    delta = (factor-1)*z_ref
    if not np.isfinite(delta):
        raise ValueError('Retreat distance overflow')
    cx, cy = principal

    def contributions():
        for begin in range(0, len(ids), 65536):
            src = ids[begin:begin+65536]
            y, x = np.divmod(src, w)
            z = depth.ravel()[src]
            s = z/(z+delta)
            # Grid edges in supersampled coordinates; original centers at integer u,v.
            left = (cx+s*(x-.5-cx)+.5)*a
            top = (cy+s*(y-.5-cy)+.5)*a
            right, bottom = left+s*a, top+s*a
            bx, by = np.floor(left).astype(int), np.floor(top).astype(int)
            for dy in range(a+1):
                yy = by+dy
                wy = np.maximum(0, np.minimum(bottom, yy+1)-np.maximum(top, yy))
                for dx in range(a+1):
                    xx = bx+dx
                    area = wy*np.maximum(0, np.minimum(right, xx+1)-np.maximum(left, xx))
                    ok = (area > 1e-12) & (xx >= 0) & (xx < ow) & (yy >= 0) & (yy < oh)
                    yield yy[ok]*ow+xx[ok], (z[ok]+delta).astype(np.float32), area[ok], src[ok]

    for dst, z, _, _ in contributions():
        np.minimum.at(zbuffer, dst, z)
    for dst, z, area, src in contributions():
        front = z == zbuffer[dst]
        dst, area, src = dst[front], area[front], src[front]
        np.add.at(weight, dst, area)
        for c in range(3):
            np.add.at(color[:, c], dst, area*rgb.reshape(-1, 3)[src, c])
    coverage = np.minimum(weight, 1)
    color /= np.maximum(weight[:, None], 1e-20)
    color *= coverage[:, None]
    cov = coverage.reshape(h, a, w, a).mean(axis=(1, 3))
    resolved = color.reshape(h, a, w, a, 3).mean(axis=(1, 3))
    resolved /= np.maximum(cov[..., None], 1e-20)
    mask = cov > 1e-12
    return np.rint(resolved).clip(0, 255).astype(np.uint8), mask.astype(np.uint8)*255, np.rint(cov*255).astype(np.uint8)

"""CPU geometry for full-frame DUSt3R input and masked initialization.

DUSt3R sees a resized, symmetrically padded copy. Original photographs and
masks are never cropped or written. Intrinsics are mapped back per axis.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageTransform:
    original_width: int
    original_height: int
    width: int
    height: int
    left: int
    top: int
    network_width: int
    network_height: int

    @classmethod
    def for_size(cls, width, height, long_edge=512):
        if min(width, height) < 2 or long_edge < 32:
            raise ValueError('Image dimensions must be >=2; DUSt3R size must be >=32')
        scale = long_edge / max(width, height)
        rw = max(2, int(round(width * scale / 2)) * 2)
        rh = max(2, int(round(height * scale / 2)) * 2)
        nw, nh = (rw + 15) // 16 * 16, (rh + 15) // 16 * 16
        return cls(width, height, rw, rh, (nw-rw)//2, (nh-rh)//2, nw, nh)

    def image(self, image, mask=False):
        mode = 'L' if mask else 'RGB'
        interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
        image = image.convert(mode).resize((self.width, self.height), interpolation)
        out = Image.new(mode, (self.network_width, self.network_height), 0)
        out.paste(image, (self.left, self.top))
        return np.asarray(out)

    def unpad(self, array):
        a = np.asarray(array)
        if a.shape[:2] != (self.network_height, self.network_width):
            raise ValueError(f'Wrong network map shape: {a.shape}')
        return a[self.top:self.top+self.height, self.left:self.left+self.width].copy()

    def original_intrinsics(self, network_focal):
        # Symmetric padding and even sizes keep both principal points centered.
        return {'fx': float(network_focal) * self.original_width / self.width,
                'fy': float(network_focal) * self.original_height / self.height,
                'cx': self.original_width / 2., 'cy': self.original_height / 2.}


def filter_by_mask_support(points, poses, focals, masks, min_support=.5, min_views=2):
    """Keep points supported by masks in enough positive-depth, in-frame views.

    Missing observations outside a camera's field of view do not vote against
    a point. No all-mask intersection is imposed on imperfect inferred poses.
    """
    points = np.asarray(points, dtype=np.float64)
    supported = np.zeros(len(points), dtype=np.int32)
    observed = np.zeros(len(points), dtype=np.int32)
    for pose, focal, mask in zip(poses, focals, masks):
        cam = (points - pose[:3, 3]) @ pose[:3, :3]
        z = cam[:, 2]
        h, w = mask.shape
        valid = np.isfinite(cam).all(1) & (z > 1e-6)
        indices = np.flatnonzero(valid)
        uv = cam[indices, :2] / z[indices, None] * float(focal) + [w/2., h/2.]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        indices, uv = indices[inside], uv[inside].astype(np.int64)
        observed[indices] += 1
        supported[indices] += (mask[uv[:, 1], uv[:, 0]] > 0)
    return (supported >= min_views) & (supported >= min_support * observed)


def normalize_reconstruction(points, poses, depths):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 16 or not np.isfinite(points).all():
        raise ValueError('Masked reconstruction has fewer than 16 finite points; inspect masks, poses and confidence')
    center = np.median(points, axis=0)
    scale = np.max(np.abs(points-center))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError('Degenerate truck point cloud; reconstruction has no spatial extent')
    poses = np.asarray(poses).copy()
    poses[:, :3, 3] = (poses[:, :3, 3]-center) / scale
    return ((points-center)/scale).astype(np.float32), poses, np.asarray(depths, dtype=np.float32)/scale, center, scale


def write_rgb_ply(path, points, colors):
    colors = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
    with open(path, 'w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\n' % len(points))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(points, colors):
            f.write('%.8g %.8g %.8g %d %d %d\n' % (*p, *c))

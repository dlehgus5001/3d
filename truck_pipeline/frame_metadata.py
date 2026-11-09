"""Export per-frame MASt3R results without masks, mesh or a fused model."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from truck_pipeline.distance_warp import align_depth, mast3r_geometry


def discover_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Image directory does not exist: {folder}')
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if len(paths) < 2:
        raise ValueError('MASt3R metadata estimation requires at least two images of the same object')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Images must have unique filename stems')
    return paths


def export_frames(paths, depths, intrinsics, output, image_size=512):
    """Copy orientation-normalized full RGB, align camera-Z to that grid, save K.

    `depths` are MASt3R get_dense_pts3d camera-Z arrays (not Euclidean range).
    `intrinsics` are the actual optimized K on the preprocessed grid.
    """
    output = Path(output)
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    if len(paths) != len(depths) or len(paths) != len(intrinsics):
        raise ValueError('Image/depth/intrinsics counts differ')
    output.mkdir(parents=True)
    frames = []
    for path, raw, K in zip(paths, depths, intrinsics):
        path = Path(path)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            rgb = np.array(image)
        w, h = image.size
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f'{path.name}: invalid optimized intrinsics')
        if not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0):
            raise ValueError('Only zero-skew pinhole intrinsics are supported')
        camera = dict(width=w, height=h, cx=float(K[0, 2]), cy=float(K[1, 2]))
        depth, principal = align_depth(np.asarray(raw), (w, h), camera, legacy=True, long_edge=image_size)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise ValueError(f'{path.name}: no valid positive camera-Z depth')
        depth[~valid] = np.nan
        resized, box = mast3r_geometry(w, h, image_size)
        name = path.stem
        camera = dict(img_name=f'{name}.png', width=w, height=h,
                      fx=float(K[0, 0]*w/resized[0]), fy=float(K[1, 1]*h/resized[1]),
                      cx=float(principal[0]), cy=float(principal[1]), intrinsics_grid='rgb',
                      depth_semantics='camera_Z', source=str(path.resolve()),
                      preprocessing=dict(resized_size=list(resized), crop_box=list(box),
                                         intrinsics_depth_grid=K.tolist()))
        for subdir in ['rgb', 'depth', 'cameras']:
            (output/subdir).mkdir(exist_ok=True)
        Image.fromarray(rgb).save(output/'rgb'/f'{name}.png')
        np.save(output/'depth'/f'{name}.npy', depth.astype(np.float32), allow_pickle=False)
        (output/'cameras'/f'{name}.json').write_text(json.dumps(camera, indent=2)+'\n')
        frames.append(dict(rgb=f'rgb/{name}.png', depth=f'depth/{name}.npy', camera=f'cameras/{name}.json'))
    # Written last: a partial failed export is not a consumable metadata bundle.
    manifest = dict(version=1, layout='aligned', depth_semantics='camera_Z', frames=frames)
    (output/'frames.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def read_bundle(folder, frame_name=None):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'frames.json').read_text())
    if manifest.get('version') != 1 or manifest.get('layout') != 'aligned' or manifest.get('depth_semantics') != 'camera_Z':
        raise ValueError('Unsupported metadata manifest')
    frames = manifest['frames']
    if not frames:
        raise ValueError('Empty metadata bundle')
    result = []
    for frame in frames:
        rgb = (folder/frame['rgb']).resolve()
        if not rgb.is_relative_to(folder):
            raise ValueError(f'Invalid bundle rgb path: {frame["rgb"]}')
        if frame_name is not None and rgb.name != frame_name:
            continue
        # RGB may intentionally be a subset after filtering/segmentation.
        # Keep the manifest association: never reindex depth by remaining order.
        if not rgb.exists():
            continue
        files = []
        for key in ['rgb', 'depth', 'camera']:
            path = (folder/frame[key]).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                raise ValueError(f'Invalid bundle {key} path: {frame[key]}')
            files.append(path)
        rgb, depth, camera_path = files
        camera = json.loads(camera_path.read_text())
        if camera['img_name'] != rgb.name or camera.get('depth_semantics') != 'camera_Z':
            raise ValueError('Metadata RGB/camera mismatch')
        result.append((rgb, depth, camera))
    if len({rgb.stem for rgb, _, _ in result}) != len(result):
        raise ValueError('Duplicate metadata frame names')
    if not result:
        raise ValueError('No matching RGB files remain in the metadata bundle')
    return result

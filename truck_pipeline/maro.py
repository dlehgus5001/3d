"""Offline adapter from the truck DUSt3R artifacts to official MaRO-GS."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from truck_pipeline.workflow import check_pose_artifacts, write_json
from truck_pipeline.object_inputs import resolve_object_images

UPSTREAM = 'https://github.com/kgh1234/MaRO-GS'
REVISION = '901e18cd35a6d86aff84b257f7f4181f5422d438'


def export_dataset(pose_dir, selection, output, objects_dir=None, rgb_source='original',
                   max_points=100000, min_confidence=3., seed=0):
    """Write selected RGB/masks and COLMAP text cameras without running COLMAP.

    Initialization unprojects full-frame DUSt3R Z, before any mask rejection.
    Camera centers and depths are already in the same normalized world units.
    """
    root, output = Path(pose_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'MaRO dataset already exists: {output}; choose a fresh --data-dir')
    if max_points < 16 or not np.isfinite(min_confidence) or min_confidence < 0:
        raise ValueError('max-points must be >=16 and min-confidence finite and >=0')
    if rgb_source not in ('original', 'object'):
        raise ValueError('rgb-source must be original or object')
    cameras = check_pose_artifacts(root, selection)
    objects = resolve_object_images(selection['frames'], objects_dir) if objects_dir else None
    if rgb_source == 'object' and objects is None:
        raise ValueError('--rgb-source object requires --objects-dir')
    n = selection['count']
    depths = np.load(root/f'dust3r_depth_{n}.npy', mmap_mode='r')
    confidences = np.load(root/f'dust3r_confidence_{n}.npy', mmap_mode='r')
    rng = np.random.default_rng(seed)
    points, colors, mappings = [], [], []
    # Bounded candidate count per view; final deterministic sample below.
    per_view = max(16, int(np.ceil(max_points/n)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.maro-', dir=output.parent) as temporary:
        dst = Path(temporary)
        for folder in ('images', 'mask', 'sparse/0'):
            (dst/folder).mkdir(parents=True)
        intrinsics, extrinsics = [], []
        for i, camera in enumerate(cameras):
            name = camera['img_name']
            target = f'{i:06d}.png'
            with Image.open(root/'images'/name) as im:
                original = im.convert('RGB')
            source = objects[name] if rgb_source == 'object' else root/'images'/name
            with Image.open(source) as im:
                im.convert('RGB').save(dst/'images'/target)
            with Image.open(root/'masks'/(Path(name).stem+'.png')) as im:
                im.convert('L').save(dst/'mask'/target)
            rotation = np.asarray(camera['rotation'])
            center = np.asarray(camera['position'])
            world_to_camera = rotation.T
            q = Rotation.from_matrix(world_to_camera).as_quat()  # xyzw -> COLMAP wxyz
            q = q[[3, 0, 1, 2]]
            translation = -world_to_camera @ center
            intrinsics.append(f"{i+1} PINHOLE {camera['width']} {camera['height']} " +
                              ' '.join(str(camera[k]) for k in ('fx', 'fy', 'cx', 'cy'))+'\n')
            extrinsics.append(f'{i+1} '+ ' '.join(map(str, [*q, *translation]))+f' {i+1} {target}\n\n')
            z = depths[i]
            h, w = z.shape
            valid = (z > 0) & np.isfinite(z) & (confidences[i] > min_confidence)
            candidates = np.flatnonzero(valid)
            if len(candidates) > per_view:
                candidates = rng.choice(candidates, per_view, replace=False)
            v, u = np.unravel_index(candidates, z.shape)
            # K is in original RGB pixels; maps cover the complete original FOV.
            u_rgb = u * camera['width']/w
            v_rgb = v * camera['height']/h
            xyz = np.column_stack(((u_rgb-camera['cx'])/camera['fx'],
                                   (v_rgb-camera['cy'])/camera['fy'], np.ones(len(u)))) * z[v,u,None]
            points.append(xyz @ rotation.T + center)
            color_map = np.asarray(original.resize((w,h), Image.Resampling.BILINEAR))
            colors.append(color_map[v,u])
            mappings.append(dict(exported=target, original=name, rgb=str(source),
                                 object=str(objects[name]) if objects else None))
        xyz, rgb = np.concatenate(points), np.concatenate(colors)
        if len(xyz) < 16 or not np.isfinite(xyz).all():
            raise ValueError('Need at least 16 finite confident depth points; inspect DUSt3R outputs')
        if len(xyz) > max_points:
            ids = rng.choice(len(xyz), max_points, replace=False)
            xyz, rgb = xyz[ids], rgb[ids]
        # Official fetchPly requires normals as well as XYZ and RGB.
        from plyfile import PlyData, PlyElement
        vertices = np.zeros(len(xyz), dtype=[(k,'f4') for k in ('x','y','z','nx','ny','nz')]+
                            [(k,'u1') for k in ('red','green','blue')])
        for j,k in enumerate(('x','y','z')): vertices[k] = xyz[:,j]
        for j,k in enumerate(('red','green','blue')): vertices[k] = rgb[:,j]
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(dst/'sparse/0/points3D.ply')
        (dst/'sparse/0/cameras.txt').write_text(''.join(intrinsics))
        (dst/'sparse/0/images.txt').write_text(''.join(extrinsics))
        manifest = dict(upstream=UPSTREAM, revision=REVISION, selection=selection,
                        rgb_source=rgb_source, frames=mappings, initial_point_count=len(xyz),
                        initialization='full-frame confident DUSt3R Z, no mask filtering',
                        min_confidence=min_confidence, seed=seed)
        write_json(dst/'maro_inputs.json', manifest)
        write_json(dst/'capture_cameras.json', cameras)
        Path(temporary).rename(output)
    return manifest


def training_command(python, backend, data, output, iterations=30000, resolution=4):
    return [str(python), str(Path(backend)/'train.py'), '-s', str(data), '-m', str(output),
            '--mask_dir', str(Path(data)/'mask'), '--iterations', str(iterations),
            '--resolution', str(resolution), '--save_iterations', str(iterations),
            '--checkpoint_iterations', str(iterations), '--disable_viewer',
            '--geometric_filtering', 'True', '--region_filtering', 'True',
            '--gaussian_merge', 'True', '--object_weight_loss', 'True', '--mask_iou_loss', 'True']

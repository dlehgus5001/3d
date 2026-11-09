#!/usr/bin/env python3
"""Estimate COLMAP-free truck cameras and masked points using local DUSt3R.

No models or data are downloaded. Run the independent frame extraction command,
place external SAM masks in masks/, then use this entry point or train_truck.py.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

from truck_pipeline.data import validate_dataset, DatasetValidationError
from truck_pipeline.pose_geometry import (ImageTransform, filter_by_mask_support,
                                         normalize_reconstruction, write_rgb_ply)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-s', '--source-path', '--source_path', default='data/truck')
    p.add_argument('--sparse_num', '--sparse-num', type=int, default=50)
    p.add_argument('--model-path', default='models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--scene-graph', default='swin-4', help='swin-N (loop closure) or complete')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--align-iterations', type=int, default=300)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--min-confidence', type=float, default=3.)
    p.add_argument('--min-mask-support', type=float, default=.5)
    p.add_argument('--min-mask-views', type=int, default=2)
    p.add_argument('--max-init-points', type=int, default=100000)
    p.add_argument('--seed', type=int, default=0)
    return p


def selected_records(source, records, count):
    if count < 3 or count > len(records):
        raise ValueError(f'--sparse_num must be between 3 and {len(records)} (got {count})')
    indices_path = source / f'sparse_{count}.txt'
    if indices_path.exists():
        ids = np.atleast_1d(np.loadtxt(indices_path, dtype=np.int64))
        if len(ids) != count or len(set(ids.tolist())) != count or np.any(ids < 0) or np.any(ids >= len(records)):
            raise ValueError(f'{indices_path}: expected {count} unique zero-based image indices in range')
    else:
        ids = np.floor(np.arange(count) * len(records) / count).astype(np.int64)
        np.savetxt(indices_path, ids, fmt='%d')
    return [records[int(i)] for i in ids], ids


def run(args):
    source = Path(args.source_path).resolve()
    records = validate_dataset(source, paired_only=True)
    records, ids = selected_records(source, records, args.sparse_num)
    if not Path(args.model_path).is_file():
        raise ValueError(f'Local DUSt3R checkpoint missing: {args.model_path}; copy it from the connected preparation machine')
    if args.batch_size < 1 or args.align_iterations < 1 or args.max_init_points < 16:
        raise ValueError('batch-size/align-iterations must be positive; max-init-points must be >=16')
    if not 0 <= args.min_mask_support <= 1 or not 1 <= args.min_mask_views <= args.sparse_num:
        raise ValueError('min-mask-support must be in [0,1]; min-mask-views must be between 1 and frame count')
    if args.scene_graph != 'complete':
        if not args.scene_graph.startswith('swin-'):
            raise ValueError('--scene-graph must be complete or swin-N')
        try:
            window = int(args.scene_graph.split('-')[1])
        except ValueError as e:
            raise ValueError('--scene-graph must be complete or swin-N') from e
        if not 2 <= window <= args.sparse_num:
            raise ValueError('swin window must be between 2 and frame count')
    selection = {'count': len(records), 'filenames': [r['filename'] for r in records], 'indices': ids.tolist()}
    print('DUSt3R actual input selection: ' + json.dumps(selection, ensure_ascii=False), flush=True)
    # Import GPU dependencies after inexpensive input/checkpoint validation.
    import torch
    from dust3r.inference import inference, load_model
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA GPU/PyTorch CUDA runtime unavailable; camera estimation and GS training require a suitable GPU')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    loaded, masks, colors, transforms = [], [], [], []
    for i, record in enumerate(records):
        name = record['filename']
        with Image.open(source/'images'/name) as im:
            transform = ImageTransform.for_size(*im.size, args.image_size)
            rgb = transform.image(im)
        with Image.open(source/'masks'/name) as im:
            mask = transform.image(im, mask=True) > 0
        if not mask.any():
            raise ValueError(f'{name}: truck disappears at DUSt3R inference resolution; increase --image-size or capture closer')
        tensor = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/127.5-1
        loaded.append(dict(img=tensor[None], true_shape=np.int32([[rgb.shape[0], rgb.shape[1]]]), idx=i, instance=str(i)))
        masks.append(mask)
        colors.append(rgb.astype(np.float32)/255.)
        transforms.append(transform)
    model = load_model(str(Path(args.model_path).resolve()), args.device)
    pairs = make_pairs(loaded, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    # Vendored swin graph includes self-pairs; remove those and duplicate edges.
    unique = {(a['idx'], b['idx']): (a,b) for a,b in pairs if a['idx'] != b['idx']}
    pairs = list(unique.values())
    print(f'DUSt3R graph={args.scene_graph}, directed pairs={len(pairs)}, input count={len(loaded)}', flush=True)
    output = inference(pairs, model, args.device, batch_size=args.batch_size)
    del model, pairs, loaded
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer, min_conf_thr=args.min_confidence)
    scene.compute_global_alignment(init='mst', niter=args.align_iterations, schedule='cosine', lr=.01)
    def numpy(x):
        return x.detach().cpu().numpy()
    poses = numpy(scene.get_im_poses())
    focals = numpy(scene.get_focals()).reshape(-1)
    depths = [numpy(x) for x in scene.get_depthmaps()]
    confidences = [numpy(x) for x in scene.get_conf(mode='id')]
    clouds = [numpy(x) for x in scene.get_pts3d()]
    candidates, candidate_colors = [], []
    for name, cloud, depth, confidence, mask, rgb in zip(selection['filenames'], clouds, depths, confidences, masks, colors):
        valid = mask & (confidence > args.min_confidence) & np.isfinite(depth) & (depth > 0) & np.isfinite(cloud).all(-1)
        print(f'{name}: confident masked initial candidates={int(valid.sum())}', flush=True)
        candidates.append(cloud[valid]); candidate_colors.append(rgb[valid])
    points, point_colors = np.concatenate(candidates), np.concatenate(candidate_colors)
    if len(points) > args.max_init_points:
        sample = rng.choice(len(points), args.max_init_points, replace=False)
        points, point_colors = points[sample], point_colors[sample]
    keep = filter_by_mask_support(points, poses, focals, masks, args.min_mask_support, args.min_mask_views)
    points, point_colors = points[keep], point_colors[keep]
    # Unpad first: stored depth/confidence maps cover full original field of view,
    # at inference resolution. Camera loader resizes these aligned maps for GS.
    depths = np.stack([t.unpad(d) for t,d in zip(transforms, depths)])
    confidences = np.stack([t.unpad(c) for t,c in zip(transforms, confidences)])
    points, poses, depths, center, scale = normalize_reconstruction(points, poses, depths)
    cameras = []
    for i, (record, pose, focal, transform) in enumerate(zip(records, poses, focals, transforms)):
        cameras.append({'id': i, 'img_name': record['filename'], 'width': record['width'], 'height': record['height'],
                        'position': pose[:3,3].tolist(), 'rotation': pose[:3,:3].tolist(),
                        **transform.original_intrinsics(focal)})
    if not np.isfinite(poses).all() or not np.isfinite(focals).all() or np.any(focals <= 0):
        raise ValueError('DUSt3R returned nonfinite cameras or invalid focal lengths; inspect input orbit and alignment')
    count = args.sparse_num
    np.save(source/f'dust3r_depth_{count}.npy', depths.astype(np.float32))
    np.save(source/f'dust3r_confidence_{count}.npy', confidences.astype(np.float32))
    with open(source/f'dust3r_{count}.json', 'w') as f:
        json.dump(cameras, f, indent=2)
    write_rgb_ply(source/f'dust3r_{count}.ply', points, point_colors)
    manifest = {**selection, 'checkpoint': str(Path(args.model_path).resolve()), 'scene_graph': args.scene_graph,
                'directed_pairs': len(unique), 'point_count': len(points), 'center_before_normalization': center.tolist(),
                'scale_before_normalization': float(scale), 'image_transforms': [asdict(t) for t in transforms],
                'min_confidence': args.min_confidence, 'min_mask_support': args.min_mask_support,
                'min_mask_views': args.min_mask_views, 'seed': args.seed}
    with open(source/f'dust3r_manifest_{count}.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Saved {count} cameras and {len(points)} masked truck points to {source}', flush=True)


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, DatasetValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

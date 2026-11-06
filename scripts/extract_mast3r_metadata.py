#!/usr/bin/env python3
"""Estimate per-frame Z depth/intrinsics for one object image folder, offline."""
import argparse
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from truck_pipeline.frame_metadata import discover_images, export_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True, help='One object, ordered images of the same static scene')
    p.add_argument('--model', type=Path, required=True, help='Trusted LOCAL MASt3R checkpoint; no auto-download')
    p.add_argument('--output', type=Path, required=True, help='New metadata directory')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--scene-graph', choices=['complete', 'swin'], default='swin')
    p.add_argument('--window', type=int, default=4, help='Forward temporal neighbors, with loop closure')
    p.add_argument('--image-size', type=int, choices=[64, 128, 256, 512], default=512)
    p.add_argument('--coarse-iterations', type=int, default=300)
    p.add_argument('--refine-iterations', type=int, default=300)
    p.add_argument('--cache-parent', type=Path, help='Writable temporary storage; per-run cache removed on exit')
    args = p.parse_args()
    paths = discover_images(args.images)
    if not args.model.is_file() or not args.model.stat().st_size:
        p.error('Missing/empty trusted local MASt3R checkpoint')
    if args.output.exists():
        p.error('Output exists; choose a new directory')
    if args.window < 1 or args.coarse_iterations < 1 or args.refine_iterations < 0:
        p.error('Window/coarse iterations must be positive; refine iterations must be >=0')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use a GPU machine or explicitly --device cpu')
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.utils.image import load_images
    from dust3r.utils.device import to_numpy
    from dust3r.image_pairs import make_pairs
    loaded = load_images([str(path) for path in paths], size=args.image_size)
    if args.scene_graph == 'complete':
        pairs = make_pairs(loaded, scene_graph='complete', symmetrize=True)
    else:
        # Existing make_pairs(swin-N) includes self-pairs; create only distinct edges.
        edges = set()
        for i in range(len(paths)):
            for step in range(1, min(args.window, len(paths)-1)+1):
                j = (i+step) % len(paths)
                edges.add((min(i, j), max(i, j)))
        pairs = [(loaded[a], loaded[b]) for i, j in sorted(edges) for a, b in [(i, j), (j, i)]]
    print(f'{len(paths)} frames, {len(pairs)} directed pairs; no masks or fused model output')
    model = AsymmetricMASt3R.from_pretrained(str(args.model.resolve())).to(args.device).eval()
    # All caches are generated in this private run; never ingest external pickle caches.
    with tempfile.TemporaryDirectory(prefix='mast3r-metadata-', dir=args.cache_parent) as cache:
        scene = sparse_global_alignment([str(path) for path in paths], pairs, cache, model,
                                        device=args.device, niter1=args.coarse_iterations,
                                        niter2=args.refine_iterations)
        # Dense points are an internal MASt3R intermediate; do not fuse or save them.
        with torch.no_grad():
            _, depths, _ = scene.get_dense_pts3d(clean_depth=False)
        export_frames(paths, to_numpy(depths), to_numpy(scene.intrinsics), args.output, args.image_size)
    print(f'Metadata saved: {args.output}/frames.json')


if __name__ == '__main__':
    main()

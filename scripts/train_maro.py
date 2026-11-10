#!/usr/bin/env python3
"""Use official MaRO-GS with existing images/mask/object folders, offline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from truck_pipeline.workflow import prepare_dataset_view, select_frames
from truck_pipeline.object_inputs import resolve_object_images
from truck_pipeline.maro import export_dataset, training_command
from scripts.train_truck import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-dir', type=Path, required=True)
    p.add_argument('--masks-dir', '--mask-dir', type=Path, required=True)
    p.add_argument('--objects-dir', '--object-dir', type=Path)
    p.add_argument('--rgb-source', choices=['original', 'object'], default='original')
    p.add_argument('--pose-dir', type=Path, required=True, help='Existing/new train_truck DUSt3R output')
    p.add_argument('--data-dir', type=Path, required=True, help='New exported MaRO dataset directory')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=50)
    p.add_argument('--stage', choices=['all','poses','prepare','train'], default='all')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--dust3r-model', type=Path, default=ROOT/'models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    p.add_argument('--maro-python', default=sys.executable, help='Python with CUDA GS dependencies (can be a separate venv)')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--resolution', type=int, default=4)
    p.add_argument('--max-init-points', type=int, default=100000)
    p.add_argument('--min-confidence', type=float, default=3.)
    args = p.parse_args()
    try:
        if args.iterations < 5100 or args.resolution < 1:
            raise ValueError('Use iterations >=5100 so MaRO modules run; resolution must be positive')
        pose = prepare_dataset_view(args.pose_dir, args.images_dir, args.masks_dir)
        selection = select_frames(pose, args.frames)
        if args.objects_dir:
            resolve_object_images(selection['frames'], args.objects_dir)
        if args.rgb_source == 'object' and not args.objects_dir:
            raise ValueError('--rgb-source object requires --objects-dir')
        print(json.dumps(dict(selected=selection['count'], filenames=selection['filenames'],
                              rgb_source=args.rgb_source, pose_dir=str(pose)), indent=2), flush=True)
        if args.validate_only: return
        data, output = args.data_dir.resolve(), args.model_dir.resolve()
        if args.stage in ('all','train') and output.exists() and any(output.iterdir()):
            raise ValueError(f'Model directory is not empty: {output}; use a fresh --model-dir')
        if args.stage in ('all','poses'):
            subprocess.run([sys.executable, str(ROOT/'scripts/train_truck.py'), '--data-dir', str(pose),
                            '--frames', str(args.frames), '--stage', 'poses',
                            '--dust3r-model', str(args.dust3r_model.resolve())], check=True, cwd=ROOT)
        if args.stage == 'poses': return
        if args.stage in ('all','prepare'):
            previous = json.loads((pose/f"pose_inputs_{selection['count']}.json").read_text())
            if previous['input_sha256'] != selection['input_sha256'] or previous['filenames'] != selection['filenames']:
                raise ValueError('Pose inputs changed; rerun --stage poses')
            export_dataset(pose, selection, data, args.objects_dir, args.rgb_source,
                           args.max_init_points, args.min_confidence)
            print(f'MaRO dataset prepared: {data}', flush=True)
        if args.stage == 'prepare': return
        manifest = json.loads((data/'maro_inputs.json').read_text())
        if manifest['selection']['input_sha256'] != selection['input_sha256'] or manifest['rgb_source'] != args.rgb_source:
            raise ValueError('Exported inputs differ; prepare a fresh --data-dir')
        backend = ROOT/'third_party/maro_gs'
        output.mkdir(parents=True, exist_ok=True)
        command = training_command(args.maro_python, backend, data, output, args.iterations, args.resolution)
        with (output/'maro_train.log').open('w') as log:
            run(command, log)
        ply = output/f'point_cloud/iteration_{args.iterations}/point_cloud.ply'
        if not ply.is_file() or ply.stat().st_size == 0:
            raise RuntimeError(f'Training did not save {ply}')
        # Keep original ordered capture cameras for the independent orbit renderer.
        import shutil
        shutil.copy2(data/'capture_cameras.json', output/'capture_cameras.json')
        shutil.copy2(data/'maro_inputs.json', output/'maro_inputs.json')
        print(f'Trained MaRO-GS: {ply}\nRender: python scripts/render_truck.py --model-path {output} '
              f'--camera-json {output}/capture_cameras.json --output renders/maro')
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        p.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()

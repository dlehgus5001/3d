#!/usr/bin/env python3
"""Validate SAM masks, estimate DUSt3R poses, then train the existing 3DGS path."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from truck_pipeline.workflow import select_frames, write_json, check_pose_artifacts


def run(command, log):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    log.write('+ ' + shlex.join(map(str, command)) + '\n')
    log.flush()
    with subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f'Command failed with exit status {status}; see {log.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/truck')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'output/truck')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--dust3r-model', type=Path, default=ROOT / 'models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument('--stage', choices=['all', 'poses', 'train'], default='all')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--resolution', type=int, default=4, help='Training downsample 1/2/4/8, or explicit width; originals never modified')
    parser.add_argument('--sh-degree', type=int, choices=range(4), default=2)
    parser.add_argument('--scene-graph', default='swin-4', help='Temporal loop graph (lower memory than complete)')
    parser.add_argument('--align-iterations', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--port', type=int, default=6009)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    output = args.model_dir.resolve()
    selection = select_frames(data, args.frames)
    print(json.dumps({'selected_count': selection['count'], 'selected_filenames': selection['filenames'],
                      'input_sha256': selection['input_sha256']}, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return
    if args.iterations < args.frames or args.resolution == 0 or args.resolution < -1:
        raise ValueError('iterations must be >= frames; resolution must be -1 or positive')
    if args.stage != 'train' and not args.dust3r_model.is_file():
        raise ValueError(f'Local DUSt3R checkpoint missing: {args.dust3r_model}; see docs/OFFLINE.md')
    if args.stage != 'poses' and output.exists() and any(output.iterdir()):
        raise ValueError(f'Model directory is not empty: {output}; choose a new --model-dir')
    # Fail before modifying sparse selections when CUDA is unavailable.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('DUSt3R and Gaussian training require a working CUDA GPU')
    pose_manifest = data / f'pose_inputs_{args.frames}.json'
    if args.stage == 'train':
        previous = json.loads(pose_manifest.read_text())
        if previous['input_sha256'] != selection['input_sha256'] or previous['filenames'] != selection['filenames']:
            raise ValueError('Images/masks/selection changed since poses; run --stage poses again')
        check_pose_artifacts(data, selection)
    else:
        np_text = ''.join(f'{i}\n' for i in selection['indices'])
        (data / f'sparse_{args.frames}.txt').write_text(np_text)
        with (data / 'truck_poses.log').open('w') as log:
            run([sys.executable, ROOT / 'pred_poses.py', '-s', data,
                 '--sparse_num', args.frames, '--model-path', args.dust3r_model.resolve(),
                 '--scene-graph', args.scene_graph, '--align-iterations', args.align_iterations,
                 '--batch-size', args.batch_size], log)
        check_pose_artifacts(data, selection)
        write_json(pose_manifest, selection)
    if args.stage == 'poses':
        print(f'Pose estimation complete: {pose_manifest}')
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'selected_frames.json', selection)
    with (output / 'truck_train.log').open('w') as log:
        log.write(json.dumps(selection, ensure_ascii=False) + '\n')
        run([sys.executable, ROOT / 'train_gs.py', '-s', data, '-m', output,
             '--use_dust3r', '--skip_render_cameras', '--sparse_view_num', args.frames,
             '--init_pcd_name', f'dust3r_{args.frames}', '--mono_loss_type', 'dust3r',
             '--random_background', '--white_background', '--sh_degree', args.sh_degree,
             '--iterations', args.iterations, '--resolution', args.resolution,
             '--port', args.port, '--test_iterations', args.iterations,
             '--checkpoint_iterations', args.iterations], log)
    ply = output / 'point_cloud' / f'iteration_{args.iterations}' / 'point_cloud.ply'
    if not ply.is_file() or ply.stat().st_size == 0:
        raise RuntimeError(f'Training exited without saved model: {ply}')
    checkpoint = output / f'ckpt{args.iterations}.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f'Training exited without saved checkpoint: {checkpoint}')
    metadata = json.loads((output / 'truck_metadata.json').read_text())
    if metadata['training_frame_count'] != args.frames or set(metadata['training_frame_names']) != {Path(n).stem for n in selection['filenames']}:
        raise RuntimeError('Actual training cameras differ from selection; inspect training_frames.json')
    visits = metadata.get('frame_visit_counts', {})
    if set(visits) != set(metadata['training_frame_names']) or any(v < 1 for v in visits.values()) or sum(visits.values()) != args.iterations:
        raise RuntimeError('Some selected training views were not used or visit counts differ from iterations')
    print(f'Trained model saved: {ply}\nRender independently with scripts/render_truck.py --model-path {output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)

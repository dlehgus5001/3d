"""Actual CPU PyTorch integration for the vendored DUSt3R path.

Uses a tiny random-weight DUSt3R checkpoint and synthetic frames. This verifies
loading/inference/global optimization/artifact wiring, not pretrained model
quality, real drone camera accuracy, or CUDA compatibility.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TORCH_PRESENT = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_PRESENT, 'CPU PyTorch unavailable; actual DUSt3R integration not executed')
class Dust3rCpuRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from dust3r.model import AsymmetricCroCo3DStereo
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.directory = tempfile.TemporaryDirectory(prefix='dust3r-runtime-')
        cls.root = Path(cls.directory.name)
        cls.model_expression = (
            "AsymmetricCroCo3DStereo(img_size=(32,32),patch_size=16,"
            "enc_embed_dim=32,enc_depth=1,enc_num_heads=4,dec_embed_dim=32,"
            "dec_depth=1,dec_num_heads=4,mlp_ratio=2,pos_embed='RoPE100',"
            "head_type='linear',landscape_only=False)")
        cls.model = eval(cls.model_expression)
        cls.checkpoint = cls.root / 'tiny_trusted_local.pth'
        torch.save({'args': Namespace(model=cls.model_expression),
                    'model': cls.model.state_dict()}, cls.checkpoint)
        (cls.root / 'images').mkdir()
        (cls.root / 'masks').mkdir()
        rng = np.random.default_rng(7)
        for index in range(3):
            image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            Image.fromarray(image).save(cls.root / 'images' / f'{index+1:04d}.png')
            mask = np.zeros((48,64), dtype=np.uint8)
            mask[2:-2,2:-2] = 255
            Image.fromarray(mask).save(cls.root / 'masks' / f'{index+1:04d}.png')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_trusted_checkpoint_loads_eval_and_pair_inference_runs(self):
        import torch
        from dust3r.inference import load_model, inference
        model = load_model(str(self.checkpoint), 'cpu')
        self.assertFalse(model.training)
        images = [dict(img=torch.rand(1,3,32,32), true_shape=np.int32([[32,32]]),
                       idx=i, instance=str(i)) for i in range(3)]
        pairs = [(images[0], images[1]), (images[1], images[0]),
                 (images[1], images[2]), (images[2], images[1])]
        output = inference(pairs, model, 'cpu', batch_size=2)
        self.assertEqual(output['pred1']['pts3d'].shape, (4,32,32,3))
        self.assertTrue(torch.isfinite(output['pred1']['pts3d']).all())
        self.assertTrue(torch.isfinite(output['pred2']['pts3d_in_other_view']).all())
        # The actual optimizer must expose one known-focal flag per input.
        from dust3r.cloud_opt import global_aligner
        scene = global_aligner(output, 'cpu', verbose=False)
        self.assertEqual(scene.get_known_focal_mask().tolist(), [False]*3)
        np.testing.assert_array_equal(scene._get_msk_indices(np.array([True, False, True])), [0,2])
        np.testing.assert_array_equal(scene._get_msk_indices(torch.tensor([0,2])), [0,2])
        np.testing.assert_array_equal(scene._get_msk_indices(torch.tensor([True, False, True])), [0,2])
        scene.preset_focal([48.,48.,48.])
        self.assertEqual(scene.get_known_focal_mask().tolist(), [True]*3)
        torch.testing.assert_close(scene.get_focals().flatten(), torch.full((3,),48.))
        scene._set_focal(200.)
        torch.testing.assert_close(scene.get_focals().flatten(), torch.full((3,),48.))
        with self.assertRaises(ValueError):
            scene.preset_focal([48.,49.,48.])

    def test_pose_cli_runs_model_mst_optimizer_and_saves_all_artifacts(self):
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = '1'
        environment['MKL_NUM_THREADS'] = '1'
        result = subprocess.run([
            sys.executable, str(ROOT/'pred_poses.py'), '-s', str(self.root),
            '--model-path', str(self.checkpoint), '--device', 'cpu',
            '--sparse_num', '3', '--scene-graph', 'swin-3', '--image-size', '32',
            '--align-iterations', '3', '--min-confidence', '0',
            '--min-mask-support', '0', '--min-mask-views', '1',
            '--max-init-points', '2000', '--seed', '7'], cwd=ROOT,
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-16000:])
        cameras = json.loads((self.root/'dust3r_3.json').read_text())
        self.assertEqual([c['img_name'] for c in cameras], ['0001.png','0002.png','0003.png'])
        self.assertEqual([c['id'] for c in cameras], [0,1,2])
        for camera in cameras:
            self.assertEqual((camera['width'],camera['height']), (64,48))
            self.assertGreater(camera['fx'],0)
            self.assertGreater(camera['fy'],0)
            self.assertTrue(np.isfinite(camera['position']).all())
            self.assertTrue(np.isfinite(camera['rotation']).all())
        for kind in ['depth','confidence']:
            array = np.load(self.root/f'dust3r_{kind}_3.npy')
            self.assertEqual(array.shape, (3,24,32))
            self.assertTrue(np.isfinite(array).all())
        manifest = json.loads((self.root/'dust3r_manifest_3.json').read_text())
        self.assertEqual(manifest['count'],3)
        self.assertEqual(manifest['directed_pairs'],6)
        self.assertGreaterEqual(manifest['point_count'],16)
        ply = (self.root/'dust3r_3.ply').read_text()
        self.assertIn(f"element vertex {manifest['point_count']}",ply)
        self.assertEqual((self.root/'sparse_3.txt').read_text().split(), ['0','1','2'])
        from truck_pipeline.workflow import check_pose_artifacts, select_frames
        self.assertEqual(len(check_pose_artifacts(self.root, select_frames(self.root,3))),3)

    def test_fifty_view_cli_keeps_every_frame_through_actual_inference_and_alignment(self):
        fifty = self.root / 'fifty_views'
        (fifty / 'images').mkdir(parents=True)
        (fifty / 'masks').mkdir()
        rng = np.random.default_rng(19)
        mask = np.zeros((48,64), dtype=np.uint8)
        mask[2:-2,2:-2] = 255
        for index in range(50):
            name = f'{index+1:04d}.png'
            Image.fromarray(rng.integers(0,256,(48,64,3),dtype=np.uint8)).save(fifty/'images'/name)
            Image.fromarray(mask).save(fifty/'masks'/name)
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = '1'
        environment['MKL_NUM_THREADS'] = '1'
        result = subprocess.run([
            sys.executable, str(ROOT/'pred_poses.py'), '-s', str(fifty),
            '--model-path', str(self.checkpoint), '--device', 'cpu',
            '--sparse_num', '50', '--scene-graph', 'swin-4', '--image-size', '32',
            '--batch-size', '4', '--align-iterations', '2', '--min-confidence', '0',
            '--min-mask-support', '0', '--min-mask-views', '1',
            '--max-init-points', '2000', '--seed', '7'], cwd=ROOT,
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-16000:])
        cameras = json.loads((fifty/'dust3r_50.json').read_text())
        self.assertEqual([c['img_name'] for c in cameras], [f'{i+1:04d}.png' for i in range(50)])
        self.assertEqual([c['id'] for c in cameras], list(range(50)))
        self.assertEqual(np.load(fifty/'dust3r_depth_50.npy').shape, (50,24,32))
        manifest = json.loads((fifty/'dust3r_manifest_50.json').read_text())
        self.assertEqual(manifest['count'],50)
        self.assertEqual(manifest['directed_pairs'],300)
        self.assertGreaterEqual(manifest['point_count'],16)
        from truck_pipeline.workflow import check_pose_artifacts, select_frames
        self.assertEqual(len(check_pose_artifacts(fifty, select_frames(fifty,50))),50)

    def test_local_checkpoint_preflight_executes_real_stereo_forward(self):
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = '1'
        result = subprocess.run([
            sys.executable, str(ROOT/'scripts/check_dust3r_checkpoint.py'),
            '--checkpoint', str(self.checkpoint), '--device', 'cpu',
            '--image-size', '32'], cwd=ROOT, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout[-16000:])
        self.assertIn('stereo forward passed', result.stdout)

    def test_dpt_head_checkpoint_load_and_forward_match_official_head_path(self):
        import torch
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.inference import load_model, inference
        # The shipped pretrained filename selects the DPT branch. Keep the
        # actual DPT decoder/heads, but reduce transformer widths and image
        # size so their interfaces can be checked without multi-GB weights.
        expression = (
            "AsymmetricCroCo3DStereo(img_size=(32,32),patch_size=16,"
            "enc_embed_dim=32,enc_depth=1,enc_num_heads=4,dec_embed_dim=32,"
            "dec_depth=10,dec_num_heads=4,mlp_ratio=2,pos_embed='RoPE100',"
            "head_type='dpt',landscape_only=False)")
        model = eval(expression)
        checkpoint = self.root / 'tiny_dpt_trusted_local.pth'
        torch.save({'args': Namespace(model=expression), 'model': model.state_dict()}, checkpoint)
        del model
        loaded = load_model(str(checkpoint), 'cpu')
        images = [dict(img=torch.rand(1,3,32,32),true_shape=np.int32([[32,32]]),
                       idx=i,instance=str(i)) for i in range(2)]
        output = inference([(images[0],images[1])],loaded,'cpu',batch_size=1)
        self.assertFalse(loaded.training)
        self.assertEqual(output['pred1']['pts3d'].shape,(1,32,32,3))
        self.assertEqual(output['pred2']['pts3d_in_other_view'].shape,(1,32,32,3))
        self.assertTrue(torch.isfinite(output['pred1']['pts3d']).all())
        self.assertTrue(torch.isfinite(output['pred2']['conf']).all())


if __name__ == '__main__':
    unittest.main()

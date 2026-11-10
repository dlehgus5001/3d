"""Real CUDA rasterizer smoke test; skipped only when CUDA PyTorch is absent.

This uses a saved synthetic black Gaussian, not a trained truck. On the offline
GPU machine it checks native opacity, PLY reload, orbit CLI, PNGs and CSV. It does
not validate reconstruction quality, truck coverage, or DUSt3R weights.
"""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
else:
    CUDA_AVAILABLE = False


@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA-enabled PyTorch/GPU unavailable; actual rasterization not executed')
class NativeOpacityGpuSmoke(unittest.TestCase):
    def test_pose_refinement_silhouette_has_nonzero_opacity_gradient(self):
        from types import SimpleNamespace
        from scene.gaussian_model import GaussianModel
        from scene.cameras import MiniCam
        from gaussian_renderer import render, render_w_pose
        from utils.graphics_utils import getProjectionMatrix
        from utils.sh_utils import RGB2SH
        from scripts.truck_render_utils import orbit_camera
        from truck_pipeline.training_render import render_training_view, silhouette_loss

        gaussian = GaussianModel(0)
        gaussian._xyz = torch.zeros((1, 3), device='cuda', requires_grad=True)
        gaussian._features_dc = RGB2SH(torch.zeros((1, 3), device='cuda'))[:, None, :].requires_grad_()
        gaussian._features_rest = torch.empty((1, 0, 3), device='cuda', requires_grad=True)
        gaussian._scaling = torch.full((1, 3), float(np.log(.15)), device='cuda', requires_grad=True)
        gaussian._rotation = torch.tensor([[1., 0, 0, 0]], device='cuda', requires_grad=True)
        gaussian._opacity = torch.zeros((1, 1), device='cuda', requires_grad=True)
        c2w = orbit_camera(np.zeros(3), *np.eye(3), 3, 0, 45)
        view = torch.tensor(np.linalg.inv(c2w).T.copy(), dtype=torch.float32, device='cuda')
        fov = float(np.radians(60))
        projection = getProjectionMatrix(.01, 100., fov, fov).T.cuda()
        camera = MiniCam(64, 64, fov, fov, .01, 100., view, view @ projection)
        camera.cam_rot_delta = torch.zeros(3, device='cuda', requires_grad=True)
        camera.cam_trans_delta = torch.zeros(3, device='cuda', requires_grad=True)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        package = render_training_view(camera, gaussian, pipe, torch.zeros(3, device='cuda'),
            refine_pose=True, need_silhouette=True, render_fixed=render, render_pose=render_w_pose)
        self.assertTrue(torch.isfinite(package['silhouette_alpha']).all())
        target = (package['silhouette_alpha'].detach() > .01).float()
        silhouette_loss(package, target).backward()
        self.assertIsNotNone(gaussian._opacity.grad)
        self.assertTrue(torch.isfinite(gaussian._opacity.grad).all())
        self.assertGreater(float(gaussian._opacity.grad.abs().sum()), 0)

    def test_saved_black_gaussian_orbit_rgba(self):
        # Missing/broken CUDA extensions on a GPU host must fail, not be skipped.
        from scene.gaussian_model import GaussianModel
        from scripts.render_truck import main
        from scripts.truck_render_utils import azimuths, orbit_camera
        from utils.sh_utils import RGB2SH

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / 'model'
            ply = model_path / 'point_cloud' / 'iteration_1' / 'point_cloud.ply'
            gaussian = GaussianModel(0)
            gaussian._xyz = torch.zeros((1, 3), device='cuda')
            gaussian._features_dc = RGB2SH(torch.zeros((1, 3), device='cuda'))[:, None, :]
            gaussian._features_rest = torch.empty((1, 0, 3), device='cuda')
            gaussian._scaling = torch.full((1, 3), float(np.log(.15)), device='cuda')
            gaussian._rotation = torch.tensor([[1., 0, 0, 0]], device='cuda')
            gaussian._opacity = torch.full((1, 1), float(np.log(.99 / .01)), device='cuda')
            gaussian.save_ply(str(ply))
            cameras = []
            for azimuth in azimuths(0, 8):
                cam = orbit_camera(np.zeros(3), *np.eye(3), 3, azimuth, 45)
                cameras.append(dict(position=cam[:3, 3].tolist(), rotation=cam[:3, :3].tolist(),
                                    width=64, height=48, fx=48, fy=48))
            (model_path / 'truck_metadata.json').write_text(json.dumps({'cameras': cameras}))
            output = Path(tmp) / 'rgba'
            main(['--model-path', str(model_path), '--elevations', '40', '60', '90',
                  '--frames-per-elevation', '2', '--width', '64', '--height', '48',
                  '--output', str(output)])
            with (output / 'views.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row['file'] for row in rows], [f'{i}.png' for i in range(1,7)])
            for row in rows:
                path = output / row['file']
                with Image.open(path) as image:
                    self.assertEqual(image.mode, 'RGBA')
                    self.assertEqual(image.size, (64, 48))
                    rgba = np.asarray(image)
                # Every rendered RGB pixel is black, but alpha must be nonzero
                # on the Gaussian and zero on its surrounding background.
                self.assertLessEqual(int(rgba[..., :3].max()), 1)
                self.assertGreater(int(rgba[..., 3].max()), 200)
                self.assertTrue(np.any(rgba[..., 3] == 0))
                self.assertTrue(np.any(rgba[..., 3] > 0))
                self.assertTrue(np.all(rgba[0, :, 3] == 0))
                self.assertTrue(np.all(rgba[-1, :, 3] == 0))
                self.assertTrue(np.all(rgba[:, 0, 3] == 0))
                self.assertTrue(np.all(rgba[:, -1, 3] == 0))
                self.assertEqual(row['touches_border'], 'False')


if __name__ == '__main__':
    unittest.main()

"""Actual PLY/model tensor round trips on CPU; no CUDA kernels are simulated.

Import-only stand-ins reject any attempted CUDA KNN operation. They let the
real GaussianModel serializer run with CPU torch without compiled extensions.
These checks do not establish rasterizer or KNN correctness.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager

READY = all(importlib.util.find_spec(name) is not None for name in ['torch', 'plyfile', 'scipy'])
if READY:
    import torch

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def import_aliases(aliases):
    # Restore only aliases, preserving native cv2/scipy modules loaded inside.
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def load_model_class():
    def reject_cuda(*args, **kwargs):
        raise AssertionError('CPU serialization test must not call CUDA KNN')
    knn = types.ModuleType('simple_knn')
    knn.__path__ = []
    knn_c = types.ModuleType('simple_knn._C')
    knn_c.distCUDA2 = reject_cuda
    p3d = types.ModuleType('pytorch3d')
    p3d.__path__ = []
    p3d_ops = types.ModuleType('pytorch3d.ops')
    p3d_ops.knn_points = reject_cuda
    p3d.ops = p3d_ops
    with import_aliases({'simple_knn': knn, 'simple_knn._C': knn_c,
                                 'pytorch3d': p3d, 'pytorch3d.ops': p3d_ops}):
        spec = importlib.util.spec_from_file_location('_serialization_model', ROOT / 'scene/gaussian_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.GaussianModel


@unittest.skipUnless(READY, 'CPU torch/plyfile/scipy unavailable; serialization not executed')
class GaussianSerializationCpuTests(unittest.TestCase):
    def test_saved_coefficients_and_inference_grad_flag_roundtrip(self):
        GaussianModel = load_model_class()
        for degree in [0, 2, 3]:
            with self.subTest(degree=degree), tempfile.TemporaryDirectory() as temporary:
                original = GaussianModel(degree)
                original._xyz = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
                original._features_dc = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
                original._features_rest = torch.arange(6 * ((degree + 1) ** 2 - 1), dtype=torch.float32).reshape(2, -1, 3)
                original._opacity = torch.tensor([[.3], [-.5]])
                original._scaling = torch.full((2, 3), -.2)
                original._rotation = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
                target = Path(temporary) / 'model.ply'
                original.save_ply(str(target))
                restored = GaussianModel(degree)
                restored.load_ply(str(target), requires_grad=False, device='cpu')
                self.assertEqual(restored.active_sh_degree, degree)
                for name in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']:
                    tensor = getattr(restored, name)
                    torch.testing.assert_close(tensor, getattr(original, name))
                    self.assertFalse(tensor.requires_grad, name)
                self.assertEqual(restored.max_radii2D.device.type, 'cpu')

    def test_training_load_preserves_parameter_gradients(self):
        GaussianModel = load_model_class()
        original = GaussianModel(0)
        original._xyz = torch.zeros((1, 3))
        original._features_dc = torch.zeros((1, 1, 3))
        original._features_rest = torch.zeros((1, 0, 3))
        original._opacity = torch.zeros((1, 1))
        original._scaling = torch.zeros((1, 3))
        original._rotation = torch.tensor([[1., 0., 0., 0.]])
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'model.ply'
            original.save_ply(str(target))
            restored = GaussianModel(0)
            restored.load_ply(str(target), device='cpu')
            self.assertTrue(restored._xyz.requires_grad)
            self.assertTrue(restored._opacity.requires_grad)

@unittest.skipUnless(READY and importlib.util.find_spec('cv2') is not None, 'CPU camera dependencies unavailable')
class CameraLoaderCpuTests(unittest.TestCase):
    def test_external_mask_depth_and_confidence_reach_training_camera(self):
        import numpy as np
        from PIL import Image
        captured = []
        class CaptureCamera:
            def __init__(self, **kwargs):
                self.arguments = kwargs
                captured.append(self)
        package = types.ModuleType('scene')
        package.__path__ = []
        cameras = types.ModuleType('scene.cameras')
        cameras.Camera = CaptureCamera
        cameras.Camera_w_pose = CaptureCamera
        with import_aliases({'scene': package, 'scene.cameras': cameras}):
            spec = importlib.util.spec_from_file_location('_camera_loader_audit', ROOT / 'utils/camera_utils.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        args = types.SimpleNamespace(resolution=2, data_device='cpu', white_background=False)
        mask = np.zeros((10, 16), dtype=np.float32)
        mask[2:8, 4:12] = 1
        info = module.CameraInfo(uid=0, R=np.eye(3), T=np.zeros(3), FovY=1., FovX=1.,
            image=Image.new('RGB', (16, 10), (10, 20, 30)), image_path='unused/0001.png',
            image_name='0001', width=16, height=10, mask=mask,
            mono_depth=np.full((5, 8), 3., dtype=np.float32),
            confidence=np.full((5, 8), .7, dtype=np.float32), is_dust3r=True)
        camera = module.loadCam(args, 0, info, 1., mode='test')
        values = camera.arguments
        self.assertEqual(tuple(values['image'].shape), (3, 5, 8))
        self.assertEqual(tuple(values['gt_alpha_mask'].shape), (1, 5, 8))
        torch.testing.assert_close(values['mono_depth'], torch.full((1, 5, 8), 3.))
        torch.testing.assert_close(values['confidence'], torch.full((1, 5, 8), .7))
        rgba = Image.new('RGBA', (16, 10), (0, 0, 0, 128))
        info = info._replace(image=rgba, mask=None, mono_depth=None, confidence=None)
        camera = module.loadCam(args, 0, info, 1., mode='test')
        self.assertEqual(tuple(camera.arguments['gt_alpha_mask'].shape), (1, 5, 8))
        torch.testing.assert_close(camera.arguments['gt_alpha_mask'], torch.full((1, 5, 8), 128/255))


if __name__ == '__main__':
    unittest.main()

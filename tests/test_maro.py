"""Exercise exported geometry and the actual vendored CPU readers."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from truck_pipeline.workflow import select_frames, write_json
from truck_pipeline.pose_geometry import write_rgb_ply
from truck_pipeline.maro import export_dataset

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT/'third_party/maro_gs'


class MaroTests(unittest.TestCase):
    def test_export_roundtrip_geometry_and_original_vs_object_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('images','masks','object'): (root/folder).mkdir()
            cameras = []
            rotation = Rotation.from_euler('y', 27, degrees=True).as_matrix()
            for i, stem in enumerate(('1','10','100')):
                Image.new('RGB',(16,16),(120,80,40)).save(root/'images'/f'{stem}.jpg')
                Image.new('RGB',(16,16),(9,8,7)).save(root/'object'/f'{stem}.png')
                mask = np.zeros((16,16),np.uint8); mask[2:8,4:13]=255
                Image.fromarray(mask).save(root/'masks'/f'{stem}.png')
                cameras.append(dict(id=i,img_name=f'{stem}.jpg',width=16,height=16,
                                    rotation=rotation.tolist(),position=[i,0.,0.],fx=10.,fy=12.,cx=8.,cy=8.))
            selection = select_frames(root)
            write_json(root/'pose_inputs_3.json', selection)
            write_json(root/'dust3r_3.json', cameras)
            (root/'sparse_3.txt').write_text('0\n1\n2\n')
            np.save(root/'dust3r_depth_3.npy',np.full((3,4,4),2.,np.float32))
            np.save(root/'dust3r_confidence_3.npy',np.full((3,4,4),5.,np.float32))
            write_rgb_ply(root/'dust3r_3.ply',np.arange(60).reshape(20,3),np.ones((20,3)))
            export_dataset(root,selection,root/'export',root/'object')
            with Image.open(root/'export/images/000000.png') as im:
                self.assertGreater(np.asarray(im)[0,0,0],100)
            export_dataset(root,selection,root/'export_object',root/'object',rgb_source='object')
            with Image.open(root/'export_object/images/000000.png') as im:
                np.testing.assert_array_equal(np.asarray(im)[0,0],[9,8,7])
            spec=importlib.util.spec_from_file_location('maro_colmap',BACKEND/'scene/colmap_loader.py')
            loader=importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
            extr=loader.read_extrinsics_text(root/'export/sparse/0/images.txt')
            intr=loader.read_intrinsics_text(root/'export/sparse/0/cameras.txt')
            self.assertEqual(len(extr),3)
            np.testing.assert_allclose(intr[1].params,[10,12,8,8])
            for i in range(3):
                e=extr[i+1]; R=loader.qvec2rotmat(e.qvec)
                np.testing.assert_allclose(R,rotation.T,atol=1e-7)
                np.testing.assert_allclose(-R.T@e.tvec,cameras[i]['position'],atol=1e-7)
            p=PlyData.read(root/'export/sparse/0/points3D.ply')['vertex']
            xyz=np.column_stack([p[k] for k in ('x','y','z')])
            # First pixel lies outside the mask but must remain in initialization.
            expected=rotation@np.array([-1.6,-8/12*2,2])
            np.testing.assert_allclose(xyz[0],expected,atol=1e-6)
            self.assertEqual(len(xyz),48)
            self.assertIn('nx',p.data.dtype.names)
            with self.assertRaisesRegex(ValueError,'already exists'):
                export_dataset(root,selection,root/'export')
            result = subprocess.run([sys.executable, str(ROOT/'scripts/train_maro.py'),
                '--images-dir', str(root/'images'), '--masks-dir', str(root/'masks'),
                '--objects-dir', str(root/'object'), '--pose-dir', str(root),
                '--data-dir', str(root/'cli_export'), '--model-dir', str(root/'model'),
                '--stage', 'prepare'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root/'cli_export/maro_inputs.json').is_file())
            self.assertFalse((root/'model').exists())

    def test_official_renderer_accepts_repository_four_output_rasterizer(self):
        # Execute the actual renderer with CPU stand-ins for CUDA tensors/extensions.
        import torch
        import math
        tree = ast.parse((BACKEND/'gaussian_renderer/__init__.py').read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render')
        class TorchCPU:
            def zeros_like(self, x, **kw):
                kw.pop('device',None)
                return torch.zeros_like(x,**kw)
        camera = SimpleNamespace(FoVx=1.,FoVy=1.,image_height=4,image_width=6,
            world_view_transform=torch.eye(4),full_proj_transform=torch.eye(4),camera_center=torch.zeros(3))
        model = SimpleNamespace(get_xyz=torch.zeros(2,3),get_opacity=torch.ones(2,1),
            get_scaling=torch.ones(2,3),get_rotation=torch.ones(2,4),get_features=torch.zeros(2,1,3),active_sh_degree=0)
        for count in (2,3,4):
            outputs=(torch.zeros(3,4,6),torch.ones(2),torch.ones(1,4,6),torch.ones(1,4,6))[:count]
            ns=dict(torch=TorchCPU(), math=math, GaussianModel=object,
                    GaussianRasterizationSettings=lambda **kw: kw,
                    GaussianRasterizer=lambda **kw: lambda **args: outputs)
            # Annotation torch.Tensor needs the actual class.
            ns['torch'].Tensor=torch.Tensor
            exec(compile(ast.Module(body=[node],type_ignores=[]),'maro_renderer','exec'),ns)
            result=ns['render'](camera,model,SimpleNamespace(),torch.zeros(3))
            self.assertEqual(tuple(result['render'].shape),(3,4,6))
            self.assertEqual(result['visibility_filter'].dtype,torch.bool)

    def test_actual_mask_loader_keeps_square_orientation_and_exact_numeric_stem(self):
        spec=importlib.util.spec_from_file_location('maro_masks',BACKEND/'scene/mask_readers.py')
        loader=importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mask=np.zeros((16,16),np.uint8);mask[2:5,8:13]=255
            Image.fromarray(mask).save(root/'1.png')
            Image.fromarray(255-mask).save(root/'10.png')
            self.assertEqual(loader._find_mask_paths(str(root),'1.jpg'),[str(root/'1.png')])
            actual=loader._load_binary_mask_from_dir(str(root),'1.jpg',16,16,device='cpu')
            np.testing.assert_array_equal(actual.numpy(),mask/255)


if __name__ == '__main__': unittest.main()

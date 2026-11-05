import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from truck_pipeline.pose_geometry import ImageTransform, filter_by_mask_support, normalize_reconstruction, write_rgb_ply
from pred_poses import selected_records


class PoseGeometryTests(unittest.TestCase):
    def test_full_frame_transform_preserves_all_edges_and_center(self):
        # Typical drone sensor frame with a non-16-aligned short side.
        t = ImageTransform.for_size(3840, 2160)
        self.assertEqual((t.width, t.height), (512, 288))
        t2 = ImageTransform.for_size(4056, 3040)
        self.assertEqual(t2.network_width % 16, 0)
        self.assertEqual(t2.network_height % 16, 0)
        self.assertEqual(t2.width/2+t2.left, t2.network_width/2)
        self.assertEqual(t2.height/2+t2.top, t2.network_height/2)
        image = Image.new('RGB', (4056,3040), (23,42,61))
        padded = t2.image(image)
        unpadded = t2.unpad(padded)
        self.assertEqual(unpadded.shape, (t2.height,t2.width,3))
        np.testing.assert_array_equal(unpadded[0,0], [23,42,61])
        np.testing.assert_array_equal(unpadded[-1,-1], [23,42,61])
        k = t2.original_intrinsics(500)
        # Pixel projection from original space lands at same network location.
        x,y,z = .3,-.7,4.
        original = np.array([x/z*k['fx']+k['cx'],y/z*k['fy']+k['cy']])
        mapped = original * [t2.width/4056,t2.height/3040]+[t2.left,t2.top]
        expected = np.array([x/z*500+t2.network_width/2,y/z*500+t2.network_height/2])
        np.testing.assert_allclose(mapped,expected)

    def test_mask_support_rejects_background_and_negative_depth(self):
        poses = np.repeat(np.eye(4)[None],3,axis=0)
        masks = [np.zeros((100,100), bool) for _ in range(3)]
        for mask in masks:
            mask[40:60,40:60]=True
        pts=np.array([[0,0,5],[2,0,5],[0,0,-5],[100,0,5]])
        valid=filter_by_mask_support(pts, poses, [100]*3,masks,min_views=2)
        np.testing.assert_array_equal(valid,[True,False,False,False])
        masks[0][:]=False
        valid=filter_by_mask_support(pts, poses, [100]*3,masks,min_views=2)
        self.assertTrue(valid[0])
        self.assertFalse(filter_by_mask_support(pts,poses,[100]*3,masks,min_support=1,min_views=2)[0])

    def test_normalization_keeps_geometry_and_depth_in_same_units(self):
        points=np.arange(60,dtype=float).reshape(-1,3)
        poses=np.repeat(np.eye(4)[None],3,axis=0)
        poses[:,:3,3]=[1,2,3]
        depths=np.full((3,4,4),10.)
        normalized, cams, depth, center, scale=normalize_reconstruction(points,poses,depths)
        np.testing.assert_allclose(normalized*scale+center,points,rtol=1e-6)
        np.testing.assert_allclose(cams[:,:3,3]*scale+center,poses[:,:3,3])
        np.testing.assert_allclose(depth*scale,depths)
        with self.assertRaises(ValueError):
            normalize_reconstruction(points[:3],poses,depths)
        with self.assertRaises(ValueError):
            normalize_reconstruction(np.zeros_like(points),poses,depths)

    def test_selection_respects_saved_indices_and_all_50(self):
        records=[{'filename':f'{i+1:04d}.png'} for i in range(50)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            selected,ids=selected_records(path,records,50)
            self.assertEqual(selected,records)
            self.assertEqual(ids.tolist(),list(range(50)))
            np.savetxt(path/'sparse_3.txt',[1,7,49],fmt='%d')
            selected,ids=selected_records(path,records,3)
            self.assertEqual([r['filename'] for r in selected],['0002.png','0008.png','0050.png'])
            np.savetxt(path/'sparse_3.txt',[1,1,49],fmt='%d')
            with self.assertRaises(ValueError):
                selected_records(path,records,3)

    def test_ply_preserves_black_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'truck.ply'
            write_rgb_ply(p,[[0,0,0],[1,2,3]],[[0,0,0],[1,.5,0]])
            lines=p.read_text().splitlines()
            self.assertIn('element vertex 2',lines)
            self.assertEqual(lines[-2],'0 0 0 0 0 0')
            self.assertEqual(lines[-1],'1 2 3 255 128 0')

if __name__ == '__main__':
    unittest.main()

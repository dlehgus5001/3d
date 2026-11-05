"""Real CPU autograd tests for dispatch/loss glue, not CUDA rasterizer tests."""
import importlib.util
import unittest

TORCH_PRESENT = importlib.util.find_spec('torch') is not None
if TORCH_PRESENT:
    import torch
    from truck_pipeline.training_render import render_training_view, silhouette_loss


@unittest.skipUnless(TORCH_PRESENT, 'PyTorch unavailable; CPU autograd tests not executed')
class TrainingRenderCpuTests(unittest.TestCase):
    def test_pose_refinement_mask_loss_uses_base_alpha_gradient(self):
        opacity = torch.tensor([0.2], requires_grad=True)
        calls = []
        def pose(*args):
            calls.append('pose')
            return {'rendered_alpha': opacity.sigmoid().detach(), 'render': opacity * 2}
        def fixed(*args):
            calls.append('fixed')
            return {'rendered_alpha': opacity.sigmoid(), 'render': opacity * 3}
        result = render_training_view(None, None, None, None, refine_pose=True,
            need_silhouette=True, render_fixed=fixed, render_pose=pose)
        self.assertEqual(calls, ['pose', 'fixed'])
        self.assertFalse(result['rendered_alpha'].requires_grad)
        loss = silhouette_loss(result, torch.ones(1))
        loss.backward()
        self.assertLess(float(opacity.grad), 0)
        self.assertAlmostEqual(float(result['render'].detach()), .4, places=6)

    def test_fixed_pose_needs_only_one_base_render_with_alpha_gradients(self):
        opacity = torch.tensor([.1], requires_grad=True)
        calls = []
        def fixed(*args):
            calls.append('fixed')
            return {'rendered_alpha': opacity.sigmoid()}
        def pose(*args):
            self.fail('Pose renderer must not be used after refinement')
        result = render_training_view(None, None, None, None, refine_pose=False,
            need_silhouette=True, render_fixed=fixed, render_pose=pose)
        silhouette_loss(result, torch.zeros(1)).backward()
        self.assertEqual(calls, ['fixed'])
        self.assertGreater(float(opacity.grad), 0)

    def test_no_extra_base_render_when_silhouette_disabled(self):
        result = {'rendered_alpha': torch.zeros(1)}
        def fixed(*args):
            self.fail('Disabled silhouette should not allocate a second rasterization')
        self.assertIs(render_training_view(None, None, None, None, refine_pose=True,
            need_silhouette=False, render_fixed=fixed, render_pose=lambda *args: result), result)

    def test_mse_and_invalid_mode(self):
        result = {'rendered_alpha': torch.tensor([.5])}
        self.assertEqual(float(silhouette_loss(result, torch.zeros(1), 'mse')), .25)
        with self.assertRaises(ValueError):
            silhouette_loss(result, torch.zeros(1), 'unknown')


if __name__ == '__main__':
    unittest.main()

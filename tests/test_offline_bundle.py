"""CPU checks for fail-fast setup and non-conflicting pose wheel staging."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.stage_pose_rasterizer import stage_pose_rasterizer, OLD_PACKAGE, POSE_PACKAGE

ROOT = Path(__file__).resolve().parents[1]


class OfflineBundleTests(unittest.TestCase):
    def test_stages_distribution_package_and_extension_without_touching_source(self):
        source = ROOT / 'submodules/diff-gaussian-rasterization-w-pose'
        if not (source / 'setup.py').is_file():
            self.skipTest('Pinned pose submodule not initialized')
        original_setup = (source / 'setup.py').read_bytes()
        original_module = (source / OLD_PACKAGE / '__init__.py').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            staged = stage_pose_rasterizer(source, Path(directory) / 'pose')
            self.assertFalse((staged / OLD_PACKAGE).exists())
            self.assertEqual((staged / POSE_PACKAGE / '__init__.py').read_bytes(), original_module)
            tree = ast.parse((staged / 'setup.py').read_text())
            setup = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'setup')
            keywords = {node.arg: node.value for node in setup.keywords}
            self.assertEqual(ast.literal_eval(keywords['name']), POSE_PACKAGE)
            self.assertEqual(ast.literal_eval(keywords['packages']), [POSE_PACKAGE])
            extension = keywords['ext_modules'].elts[0]
            extension_args = {node.arg: node.value for node in extension.keywords}
            self.assertEqual(ast.literal_eval(extension_args['name']), POSE_PACKAGE + '._C')
            self.assertTrue((staged / 'third_party/glm/glm/glm.hpp').is_file())
            self.assertFalse((staged / '.git').exists())
            with self.assertRaises(ValueError):
                stage_pose_rasterizer(source, staged)
        self.assertEqual((source / 'setup.py').read_bytes(), original_setup)
        self.assertEqual((source / OLD_PACKAGE / '__init__.py').read_bytes(), original_module)
        self.assertFalse((source / POSE_PACKAGE).exists())

    def test_rejects_incomplete_sources_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                stage_pose_rasterizer(Path(directory), Path(directory) / 'staged')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_dust3r_checkpoint.py'),
                                     '--checkpoint', str(Path(directory) / 'missing.pth'), '--device', 'cpu'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing/empty trusted local DUSt3R checkpoint', result.stderr)

    def test_shell_syntax_and_missing_weight_fail_before_environment_mutation(self):
        result = subprocess.run(['bash', '-n', str(ROOT/'scripts/prepare_offline_bundle.sh'),
                                 str(ROOT/'scripts/install_offline.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bundle'
            result = subprocess.run(['bash', str(ROOT/'scripts/prepare_offline_bundle.sh'),
                                     str(bundle), str(Path(directory)/'missing.pth')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing/empty local checkpoint', result.stderr)
            self.assertFalse(bundle.exists())


if __name__ == '__main__':
    unittest.main()

import unittest
from pathlib import Path
import importlib.util
import ast


class SystemTests(unittest.TestCase):
    def test_requirements_exists(self):
        p = Path("requirements.txt")
        self.assertTrue(p.exists(), "requirements.txt is missing")
        self.assertGreater(p.stat().st_size, 0, "requirements.txt is empty")

    def test_venv_exists(self):
        p = Path('.venv')
        self.assertTrue(p.exists() and p.is_dir(), ".venv not found in project root")
        # Windows uses Scripts, Unix uses bin
        self.assertTrue((p / 'Scripts').exists() or (p / 'bin').exists(), ".venv appears incomplete (no Scripts/bin)")

    def test_core_trade_analysis_importable(self):
        path = Path('core') / 'trade_analysis.py'
        self.assertTrue(path.exists(), f"{path} not found")
        spec = importlib.util.spec_from_file_location('core_trade_analysis', str(path))
        self.assertIsNotNone(spec, 'Could not create module spec for core.trade_analysis')
        mod = importlib.util.module_from_spec(spec)
        import sys
        # Ensure module is available in sys.modules while executing (dataclasses expect it)
        sys.modules[spec.name] = mod
        try:
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                self.fail(f"Importing core.trade_analysis raised an exception: {e}")
        finally:
            # Clean up inserted module
            if spec.name in sys.modules and sys.modules[spec.name] is mod:
                del sys.modules[spec.name]
        # Check key functions exist
        for name in ('load_trades', 'compute_metrics', 'aggregate', 'plot_equity_curve', 'plot_drawdown', 'write_summary'):
            self.assertTrue(hasattr(mod, name), f"{name} not found in core.trade_analysis")

    def test_run_analysis_parseable(self):
        path = Path('run_analysis.py')
        self.assertTrue(path.exists(), 'run_analysis.py missing')
        src = path.read_text(encoding='utf-8')
        try:
            ast.parse(src)
        except Exception as e:
            self.fail(f"run_analysis.py is not valid Python: {e}")

    def test_readme_exists(self):
        self.assertTrue(Path('README.md').exists(), 'README.md missing')


if __name__ == '__main__':
    unittest.main()

import unittest
from pathlib import Path
import importlib.util


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('core_trade_analysis', str(path))
    if spec is None:
        raise ImportError(f'Could not create spec from {path}')
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # keep module in sys.modules for debugging and re-raise
        raise
    import unittest
    from pathlib import Path
    import importlib.util


    def load_module(path: Path):
        spec = importlib.util.spec_from_file_location('core_trade_analysis', str(path))
        if spec is None:
            raise ImportError(f'Could not create spec from {path}')
        mod = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # keep module in sys.modules for debugging and re-raise
            raise
        return mod


    class FunctionalIOTests(unittest.TestCase):
        def setUp(self):
            self.data_dir = Path('mnt') / 'data'
            self.assertTrue(self.data_dir.exists(), f"Data directory not found: {self.data_dir}")
            self.mod = load_module(Path('core') / 'trade_analysis.py')

        def test_csv_file_loads_and_metrics(self):
            csvs = list(self.data_dir.glob('*.csv'))
            self.assertTrue(csvs, 'No CSV files found in mnt/data to test')
            p = csvs[0]
            # The broker CSV uses 'Date' and 'Amount' column names; map them to expected names
            column_map = {"Date": "date", "Amount": "pnl"}
            df = self.mod.load_trades(p, column_map=column_map)
            # basic sanity
            self.assertIn('timestamp', df.columns)
            self.assertIn('pnl', df.columns)
            self.assertGreaterEqual(len(df), 1, 'CSV produced empty DataFrame')
            metrics = self.mod.compute_metrics(df)
            self.assertEqual(metrics.total_trades, len(df))

        def test_json_file_loads_and_metrics(self):
            jsons = list(self.data_dir.glob('*.json'))
            self.assertTrue(jsons, 'No JSON files found in mnt/data to test')
            p = jsons[0]
            # These JSON files contain a top-level 'BrokerageTransactions' list of dicts.
            # Extract it and write a temp CSV that load_trades can read.
            import json, tempfile
            data = json.loads(p.read_text(encoding='utf-8'))
            records = data.get('BrokerageTransactions') or data.get('brokerageTransactions')
            self.assertTrue(records, f'JSON file {p} did not contain BrokerageTransactions')
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
                import pandas as _pd
                _pd.DataFrame(records).to_csv(tmp.name, index=False)
                tmp_path = tmp.name
            try:
                column_map = {"Date": "date", "Amount": "pnl"}
                df = self.mod.load_trades(tmp_path, column_map=column_map)
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
            self.assertIn('timestamp', df.columns)
            self.assertIn('pnl', df.columns)
            self.assertGreaterEqual(len(df), 1, 'JSON produced empty DataFrame')
            metrics = self.mod.compute_metrics(df)
            self.assertEqual(metrics.total_trades, len(df))


    if __name__ == '__main__':
        unittest.main()
    return mod

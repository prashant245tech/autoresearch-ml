import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        self.addCleanup(self._restore_cwd)

    def _restore_cwd(self):
        os.chdir(self._old_cwd)

    def abs_path(self, relative_path: str) -> Path:
        return Path(self._tmpdir.name) / relative_path

    def write_text(self, relative_path: str, content: str) -> Path:
        path = self.abs_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_json(self, relative_path: str, payload: Any) -> Path:
        path = self.abs_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def write_parquet(self, relative_path: str, df: pd.DataFrame) -> Path:
        path = self.abs_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def create_prepared_data(self, train_df=None, test_df=None, meta=None):
        if train_df is None:
            x_train = np.arange(1, 41, dtype=float)
            train_df = pd.DataFrame(
                {
                    "x": x_train,
                    "price_msf": (3.0 * x_train) + 5.0,
                }
            )
        if test_df is None:
            x_test = np.arange(41, 51, dtype=float)
            test_df = pd.DataFrame(
                {
                    "x": x_test,
                    "price_msf": (3.0 * x_test) + 5.0,
                }
            )
        if meta is None:
            meta = {
                "version": "test-suite",
                "model_features": ["x"],
                "ref_cols": [],
                "encoding": {},
                "feature_config": [],
                "filter_spec_config": [],
            }

        self.write_parquet("data/train.parquet", train_df)
        self.write_parquet("data/test.parquet", test_df)
        self.write_json("data/columns.json", meta)
        return train_df, test_df, meta

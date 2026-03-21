import pandas as pd

import prepare
from tests.support import WorkspaceTestCase


class PrepareValidationTests(WorkspaceTestCase):
    def test_validate_spec_rejects_unknown_top_level_keys(self):
        spec = {
            "data_file": "source.csv",
            "target_column": "price_msf",
            "features": [{"column": "Quantity", "type": "numeric"}],
            "unexpected_key": True,
        }

        with self.assertRaises(SystemExit) as exc:
            prepare.validate_spec(spec)

        self.assertIn("unsupported keys", str(exc.exception))
        self.assertIn("unexpected_key", str(exc.exception))

    def test_validate_spec_requires_filter_columns_to_be_declared(self):
        spec = {
            "data_file": "source.csv",
            "target_column": "price_msf",
            "features": [{"column": "Quantity", "type": "numeric"}],
            "train_row_filters": [
                {"column": "Account Type", "op": "eq", "value": "FREE HOUSE"}
            ],
        }

        with self.assertRaises(SystemExit) as exc:
            prepare.validate_spec(spec)

        self.assertIn("Each filtered column must be declared", str(exc.exception))
        self.assertIn("Account Type", str(exc.exception))

    def test_validate_spec_rejects_feature_and_filter_spec_overlap(self):
        spec = {
            "data_file": "source.csv",
            "target_column": "price_msf",
            "features": [{"column": "Account Type", "type": "categorical"}],
            "filter_specs": [{"column": "Account Type", "type": "categorical"}],
        }

        with self.assertRaises(SystemExit) as exc:
            prepare.validate_spec(spec)

        self.assertIn("must not appear in both `features` and `filter_specs`", str(exc.exception))

    def test_filter_specs_are_applied_before_row_filters(self):
        raw_df = pd.DataFrame(
            {
                "Account Type": ["Free House ", "FREE HOUSE", "FREEHOUSE", "Merchant"],
                "Quantity": [10, 20, 30, 40],
                "price_msf": [100, 110, 120, 130],
            }
        )
        features = [{"column": "Quantity", "type": "numeric"}]
        filter_specs = [
            {
                "column": "Account Type",
                "type": "categorical",
                "normalise": "uppercase_strip",
                "categorical_consolidation": {"FREEHOUSE": "FREE HOUSE"},
            }
        ]

        prepared_df = prepare.prepare_dataset(
            raw_df=raw_df,
            features=features,
            filter_specs=filter_specs,
            target_col="price_msf",
            spec={"outlier_iqr_k": 0},
            dataset_label="train source",
            row_filters=[{"column": "Account Type", "op": "eq", "value": "FREE HOUSE"}],
            apply_outlier_filter=False,
        )

        self.assertEqual(len(prepared_df), 3)
        self.assertEqual(sorted(prepared_df["Account Type"].unique().tolist()), ["FREE HOUSE"])

import pandas as pd

import prepare
from tests.support import WorkspaceTestCase


class PrepareValidationTests(WorkspaceTestCase):
    def test_validate_spec_rejects_unknown_top_level_keys(self):
        spec = {
            "data_file": "source.csv",
            "target_column": "target_value",
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
            "target_column": "target_value",
            "features": [{"column": "Quantity", "type": "numeric"}],
            "train_row_filters": [
                {"column": "Cohort", "op": "eq", "value": "INTERNAL"}
            ],
        }

        with self.assertRaises(SystemExit) as exc:
            prepare.validate_spec(spec)

        self.assertIn("Each filtered column must be declared", str(exc.exception))
        self.assertIn("Cohort", str(exc.exception))

    def test_validate_spec_rejects_feature_and_filter_spec_overlap(self):
        spec = {
            "data_file": "source.csv",
            "target_column": "target_value",
            "features": [{"column": "Cohort", "type": "categorical"}],
            "filter_specs": [{"column": "Cohort", "type": "categorical"}],
        }

        with self.assertRaises(SystemExit) as exc:
            prepare.validate_spec(spec)

        self.assertIn("must not appear in both `features` and `filter_specs`", str(exc.exception))

    def test_filter_specs_are_applied_before_row_filters(self):
        raw_df = pd.DataFrame(
            {
                "Cohort": ["Internal ", "INTERNAL", "INHOUSE", "External"],
                "Quantity": [10, 20, 30, 40],
                "target_value": [100, 110, 120, 130],
            }
        )
        features = [{"column": "Quantity", "type": "numeric"}]
        filter_specs = [
            {
                "column": "Cohort",
                "type": "categorical",
                "normalise": "uppercase_strip",
                "categorical_consolidation": {"INHOUSE": "INTERNAL"},
            }
        ]

        prepared_df = prepare.prepare_dataset(
            raw_df=raw_df,
            features=features,
            filter_specs=filter_specs,
            target_col="target_value",
            output_target_col="target_value",
            spec={"outlier_iqr_k": 0},
            dataset_label="train source",
            row_filters=[{"column": "Cohort", "op": "eq", "value": "INTERNAL"}],
            apply_outlier_filter=False,
        )

        self.assertEqual(len(prepared_df), 3)
        self.assertEqual(sorted(prepared_df["Cohort"].unique().tolist()), ["INTERNAL"])

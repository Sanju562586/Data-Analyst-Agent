import unittest
import pandas as pd
import numpy as np
import json
import tools
from tool_registry import build_openai_tools, get_tool, TOOL_MAP as REGISTRY_TOOL_MAP
from agent import TOOL_MAP as AGENT_TOOL_MAP


class TestDataCleaningTools(unittest.TestCase):

    def setUp(self):
        # Reset tools.df before each test
        tools.df = None
        tools.last_query_df = None

    def test_handle_missing_values_no_dataset(self):
        result = tools.handle_missing_values(strategy="mean")
        self.assertEqual(result, "No dataset loaded.")

    def test_handle_missing_values_invalid_strategy(self):
        tools.df = pd.DataFrame({"a": [1, 2, 3]})
        result = tools.handle_missing_values(strategy="invalid_strat")
        self.assertIn("Error: Unsupported strategy", result)

    def test_handle_missing_values_mean(self):
        tools.df = pd.DataFrame({
            "age": [20.0, 40.0, np.nan, 60.0],
            "salary": [1000.0, np.nan, 3000.0, 4000.0],
            "name": ["Alice", "Bob", "Charlie", "David"]
        })
        summary = tools.handle_missing_values(strategy="mean", columns=["age", "salary"])
        self.assertIn("Missing Values Handling Summary (`mean`)", summary)
        # Expected mean age: (20+40+60)/3 = 40.0
        self.assertEqual(tools.df["age"].iloc[2], 40.0)
        # Expected mean salary: (1000+3000+4000)/3 = 2666.6667
        self.assertAlmostEqual(tools.df["salary"].iloc[1], 2666.6667, places=3)
        self.assertEqual(tools.df["age"].isnull().sum(), 0)
        self.assertEqual(tools.df["salary"].isnull().sum(), 0)

    def test_handle_missing_values_mean_non_numeric(self):
        tools.df = pd.DataFrame({
            "category": ["A", np.nan, "B"]
        })
        result = tools.handle_missing_values(strategy="mean", columns=["category"])
        self.assertIn("Error: Strategy 'mean' requires numeric columns", result)

    def test_handle_missing_values_median(self):
        tools.df = pd.DataFrame({
            "score": [10.0, 20.0, np.nan, 100.0, 200.0]
        })
        summary = tools.handle_missing_values(strategy="median", columns="score")
        # Median of [10, 20, 100, 200] is 60.0
        self.assertEqual(tools.df["score"].iloc[2], 60.0)
        self.assertEqual(tools.df["score"].isnull().sum(), 0)

    def test_handle_missing_values_mode(self):
        tools.df = pd.DataFrame({
            "city": ["NY", "LA", "NY", np.nan, "SF"],
            "code": [1, 2, 1, np.nan, 3]
        })
        summary = tools.handle_missing_values(strategy="mode", columns=["city", "code"])
        self.assertEqual(tools.df["city"].iloc[3], "NY")
        self.assertEqual(tools.df["code"].iloc[3], 1)
        self.assertEqual(tools.df.isnull().sum().sum(), 0)

    def test_handle_missing_values_fill(self):
        tools.df = pd.DataFrame({
            "status": ["active", np.nan, "inactive"],
            "count": [5, np.nan, 12]
        })
        # Missing fill_value error check
        err = tools.handle_missing_values(strategy="fill", columns=["status"])
        self.assertIn("Error: 'fill_value' must be specified", err)

        # Successful fill
        summary = tools.handle_missing_values(strategy="fill", columns=["status"], fill_value="unknown")
        self.assertEqual(tools.df["status"].iloc[1], "unknown")

    def test_handle_missing_values_drop(self):
        tools.df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, np.nan, 40.0],
            "c": ["x", "y", "z", "w"]
        })
        # Drop only when 'a' is null
        summary = tools.handle_missing_values(strategy="drop", columns=["a"])
        self.assertEqual(len(tools.df), 3)
        self.assertNotIn(np.nan, tools.df["a"].values)
        self.assertEqual(tools.df.index.tolist(), [0, 1, 2])

    def test_handle_missing_values_no_nulls(self):
        tools.df = pd.DataFrame({"a": [1, 2, 3]})
        result = tools.handle_missing_values(strategy="drop")
        self.assertIn("No missing values found", result)

    def test_detect_and_handle_outliers_iqr_clip(self):
        # 10 values, normal range 10-20, one extreme 500
        tools.df = pd.DataFrame({
            "val": [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 500.0]
        })
        summary = tools.detect_and_handle_outliers(columns=["val"], method="iqr", action="clip")
        self.assertIn("### Outlier Detection & Handling Summary", summary)
        self.assertIn("Clipped 1 values", summary)
        self.assertLess(tools.df["val"].max(), 100.0)
        self.assertEqual(len(tools.df), 10)

    def test_detect_and_handle_outliers_iqr_drop(self):
        tools.df = pd.DataFrame({
            "val": [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 500.0]
        })
        summary = tools.detect_and_handle_outliers(columns="val", method="iqr", action="drop")
        self.assertEqual(len(tools.df), 9)
        self.assertEqual(tools.df["val"].max(), 20.0)

    def test_detect_and_handle_outliers_iqr_flag(self):
        tools.df = pd.DataFrame({
            "val": [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 500.0]
        })
        summary = tools.detect_and_handle_outliers(columns=["val"], method="iqr", action="flag")
        self.assertIn("val_outlier", tools.df.columns)
        self.assertTrue(tools.df["val_outlier"].iloc[-1])
        self.assertFalse(tools.df["val_outlier"].iloc[0])
        self.assertEqual(len(tools.df), 10)

    def test_detect_and_handle_outliers_zscore(self):
        data = [10.0] * 50 + [1000.0]  # extreme outlier
        tools.df = pd.DataFrame({"metric": data})
        summary = tools.detect_and_handle_outliers(columns=["metric"], method="zscore", action="drop")
        self.assertEqual(len(tools.df), 50)

    def test_detect_and_handle_outliers_invalid_inputs(self):
        tools.df = pd.DataFrame({"txt": ["hello", "world"]})
        res = tools.detect_and_handle_outliers(columns=["txt"])
        self.assertIn("Error: Outlier detection requires numeric columns", res)

        res2 = tools.detect_and_handle_outliers(method="unknown_method")
        self.assertIn("Error: Unsupported outlier method", res2)

        res3 = tools.detect_and_handle_outliers(action="unknown_action")
        self.assertIn("Error: Unsupported outlier action", res3)

    def test_convert_column_types_dates_and_currencies(self):
        tools.df = pd.DataFrame({
            "order_date": ["2024-01-15", "02/16/2024", "March 17, 2024", "invalid_date"],
            "price": ["$1,250.50", "€50.25", "£99.00", "(25.00)"],
            "quantity": ["10", "20", "30", np.nan],
            "in_stock": ["yes", "no", "true", "FALSE"]
        })

        summary = tools.convert_column_types({
            "order_date": "datetime",
            "price": "numeric",
            "quantity": "integer",
            "in_stock": "boolean"
        })

        self.assertIn("### Column Type Conversion Summary", summary)
        # Check datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(tools.df["order_date"]))
        self.assertEqual(tools.df["order_date"].iloc[0].year, 2024)
        self.assertTrue(pd.isna(tools.df["order_date"].iloc[3]))

        # Check price (numeric, currency and comma stripped, parentheses converted to negative)
        self.assertTrue(pd.api.types.is_numeric_dtype(tools.df["price"]))
        self.assertAlmostEqual(tools.df["price"].iloc[0], 1250.50)
        self.assertAlmostEqual(tools.df["price"].iloc[1], 50.25)
        self.assertAlmostEqual(tools.df["price"].iloc[2], 99.00)
        self.assertAlmostEqual(tools.df["price"].iloc[3], -25.00)

        # Check nullable integer
        self.assertEqual(str(tools.df["quantity"].dtype), "Int64")
        self.assertEqual(tools.df["quantity"].iloc[0], 10)
        self.assertTrue(pd.isna(tools.df["quantity"].iloc[3]))

        # Check boolean
        self.assertTrue(tools.df["in_stock"].iloc[0])
        self.assertFalse(tools.df["in_stock"].iloc[1])
        self.assertTrue(tools.df["in_stock"].iloc[2])
        self.assertFalse(tools.df["in_stock"].iloc[3])

    def test_convert_column_types_json_string_input(self):
        tools.df = pd.DataFrame({
            "revenue": ["$500", "$600"]
        })
        json_str = json.dumps({"revenue": "float"})
        summary = tools.convert_column_types(json_str)
        self.assertIn("Success", summary)
        self.assertEqual(tools.df["revenue"].iloc[0], 500.0)

    def test_tool_registries_and_definitions(self):
        # Verify tools.json loads and includes all 3 tools
        with open("tools.json", "r", encoding="utf-8") as f:
            defs = json.load(f)
        tool_names = [d["name"] for d in defs]
        self.assertIn("handle_missing_values", tool_names)
        self.assertIn("detect_and_handle_outliers", tool_names)
        self.assertIn("convert_column_types", tool_names)

        # Verify tool_registry TOOL_MAP
        self.assertIn("handle_missing_values", REGISTRY_TOOL_MAP)
        self.assertIn("detect_and_handle_outliers", REGISTRY_TOOL_MAP)
        self.assertIn("convert_column_types", REGISTRY_TOOL_MAP)
        self.assertTrue(callable(get_tool("handle_missing_values")))
        self.assertTrue(callable(get_tool("detect_and_handle_outliers")))
        self.assertTrue(callable(get_tool("convert_column_types")))

        # Verify agent TOOL_MAP
        self.assertIn("handle_missing_values", AGENT_TOOL_MAP)
        self.assertIn("detect_and_handle_outliers", AGENT_TOOL_MAP)
        self.assertIn("convert_column_types", AGENT_TOOL_MAP)

        # Verify OpenAI tools schema generation
        openai_tools = build_openai_tools()
        openai_tool_names = [t["function"]["name"] for t in openai_tools]
        self.assertIn("handle_missing_values", openai_tool_names)
        self.assertIn("detect_and_handle_outliers", openai_tool_names)
        self.assertIn("convert_column_types", openai_tool_names)


if __name__ == "__main__":
    unittest.main()

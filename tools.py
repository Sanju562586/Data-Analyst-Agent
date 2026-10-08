import pandas as pd
import numpy as np
import io
import contextlib
import os
import json
import ast
import matplotlib.pyplot as plt

df = None
last_query_df = None


def load_csv(path=None, file=None):
    global df

    target = path if path is not None else file

    if target is not None:
        try:
            if hasattr(target, "read") or (isinstance(target, str) and os.path.exists(target)):
                df = pd.read_csv(target)
            elif df is not None:
                # Target path does not exist on disk, but dataset is already loaded in memory
                pass
            else:
                return f"Error: File '{target}' does not exist and no dataset is currently loaded."
        except Exception as e:
            if df is None:
                return f"Failed to load CSV: {e}"

    if df is not None:
        return {
            "status": "success",
            "message": "Dataset is loaded in memory.",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "preview": df.head().to_dict(orient="records")
        }

    return "No dataset loaded. Please provide a valid file path or upload a CSV."


def analyze_data():
    global df

    if df is None:
        return "No dataset loaded."

    stats = df.describe(include="all").to_string()
    missing = df.isnull().sum().to_string()
    dtypes = df.dtypes.to_string()
    duplicates = len(df) - len(df.drop_duplicates())
    unique = df.nunique().to_string()

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        correlation = numeric_df.corr().to_string()
    else:
        correlation = "No numeric columns available."

    summary = f"""
    Shape: {df.shape}

    ------------------------------
    Data Types
    ------------------------------
    {dtypes}

    ------------------------------
    Missing Values
    ------------------------------
    {missing}

    ------------------------------
    Duplicate Rows
    ------------------------------
    {duplicates}

    ------------------------------
    Unique Values
    ------------------------------
    {unique}

    ------------------------------
    Statistics
    ------------------------------
    {stats}

    ------------------------------
    Correlation Matrix
    ------------------------------
    {correlation}
    """

    return summary


def run_pandas_code(code):
    global df, last_query_df

    if df is None:
        return "No dataset loaded."

    local_vars = {
        "df": df,
        "pd": pd,
        "np": np
    }

    # Ensure pandas displays full content without premature ellipsis truncation
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 1000)

    output = io.StringIO()

    try:
        cleaned_code = code.strip()
        if cleaned_code.startswith("```"):
            lines = cleaned_code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_code = "\n".join(lines).strip()

        with contextlib.redirect_stdout(output):
            # Check if code is a single expression that can be evaluated directly
            expr_val = None
            try:
                expr_val = eval(cleaned_code, {}, local_vars)
            except Exception:
                expr_val = None

            if expr_val is not None:
                if isinstance(expr_val, (pd.DataFrame, pd.Series)):
                    if isinstance(expr_val, pd.Series):
                        expr_val = expr_val.to_frame()
                    last_query_df = expr_val
                    print(expr_val.to_string())
                else:
                    print(expr_val)
            else:
                exec(cleaned_code, {}, local_vars)

        # Inspect local_vars to see if any new/filtered DataFrame was created
        new_dfs = [
            v for k, v in local_vars.items()
            if k not in ("df", "pd", "np") and isinstance(v, (pd.DataFrame, pd.Series))
        ]
        if new_dfs:
            candidate_df = new_dfs[-1]
            if isinstance(candidate_df, pd.Series):
                candidate_df = candidate_df.to_frame()
            last_query_df = candidate_df

        printed_output = output.getvalue().strip()

        # If code didn't print anything but produced/filtered a DataFrame, provide a preview
        if not printed_output and last_query_df is not None:
            preview_limit = 50
            row_count = len(last_query_df)
            trimmed = last_query_df.head(preview_limit)
            try:
                tbl = trimmed.to_markdown(index=False)
            except Exception:
                tbl = trimmed.to_string(index=False)
            if row_count > preview_limit:
                printed_output = f"Produced {row_count} rows (showing first {preview_limit}):\n\n{tbl}"
            else:
                printed_output = f"Produced {row_count} rows:\n\n{tbl}"

        return printed_output or "Code executed successfully."

    except Exception as e:
        return f"Error: {e}"


def run_sql_query(query: str):
    """
    Executes a SQL query on the loaded dataset using DuckDB.
    The dataset is available as the table 'df'.
    """
    global df, last_query_df

    if df is None:
        return "No dataset loaded. Please provide a valid file path or upload a CSV."

    try:
        import duckdb
    except ImportError:
        return "Error: DuckDB is not installed. Please install duckdb using 'pip install duckdb'."

    cleaned_query = query.strip()
    if cleaned_query.startswith("```"):
        lines = cleaned_query.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_query = "\n".join(lines).strip()
    cleaned_query = cleaned_query.rstrip(";")

    try:
        conn = duckdb.connect(database=":memory:")
        conn.register("df", df)
        result = conn.execute(cleaned_query).df()
        last_query_df = result

        if result is None or result.empty:
            return "Query executed successfully, but returned 0 rows."

        row_count = len(result)
        preview_limit = 50
        trimmed_result = result.head(preview_limit)

        try:
            table_str = trimmed_result.to_markdown(index=False)
        except Exception:
            table_str = trimmed_result.to_string(index=False)

        if row_count > preview_limit:
            return f"Showing first {preview_limit} of {row_count} rows:\n\n{table_str}"

        return f"Query returned {row_count} row(s):\n\n{table_str}"

    except Exception as e:
        return f"SQL Error: {e}"


def remove_duplicates():
    """
    Remove rows that are exact duplicates across all columns.
    Keeps the first occurrence and drops the rest.
    """
    global df

    if df is None:
        return "No dataset loaded."

    total = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = total - len(df)

    if removed == 0:
        return "No duplicate rows found. Dataset unchanged."

    return f"Removed {removed} duplicate rows. {len(df)} rows remain."


def handle_missing_values(strategy="drop", columns=None, fill_value=None):
    """
    Automatically imputes or cleans null columns in the dataset and provides a before/after summary.

    Parameters:
        strategy (str): 'mean', 'median', 'mode', 'drop', or 'fill'.
        columns (list or str, optional): Column name(s) to process. If None, targets all applicable columns.
        fill_value (any, optional): Constant value to use when strategy='fill'.
    """
    global df, last_query_df

    if df is None:
        return "No dataset loaded."

    if not strategy:
        strategy = "drop"
    strategy_clean = str(strategy).strip().lower()

    valid_strategies = ["mean", "median", "mode", "drop", "fill"]
    if strategy_clean not in valid_strategies:
        return f"Error: Unsupported strategy '{strategy}'. Supported strategies: {', '.join(valid_strategies)}."

    # Parse columns argument
    target_cols = []
    if columns is not None:
        if isinstance(columns, str):
            cleaned_str = columns.strip()
            if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
                try:
                    parsed = json.loads(cleaned_str)
                    if isinstance(parsed, list):
                        target_cols = [str(c).strip() for c in parsed]
                except Exception:
                    target_cols = [c.strip() for c in cleaned_str.strip("[]").split(",") if c.strip()]
            else:
                target_cols = [c.strip() for c in cleaned_str.split(",") if c.strip()]
        elif isinstance(columns, (list, tuple)):
            target_cols = [str(c).strip() for c in columns if str(c).strip()]
        else:
            target_cols = [str(columns)]

    if target_cols:
        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            return f"Error: Column(s) {missing_cols} not found in dataset. Available columns: {list(df.columns)}"
    else:
        # Automatically determine target columns with missing values
        if strategy_clean in ("mean", "median"):
            target_cols = [
                c for c in df.select_dtypes(include="number").columns
                if df[c].isnull().any()
            ]
        else:
            target_cols = [c for c in df.columns if df[c].isnull().any()]

    # If no target columns found
    if not target_cols:
        total_nulls = int(df.isnull().sum().sum())
        if total_nulls == 0:
            return "No missing values found in the dataset. Dataset unchanged."
        if strategy_clean in ("mean", "median"):
            non_num_nulls = [c for c in df.columns if df[c].isnull().any()]
            return (
                f"No numeric columns with missing values found for '{strategy_clean}' strategy. "
                f"Columns with nulls ({non_num_nulls}) are non-numeric. Use 'mode', 'fill', or 'drop' instead."
            )
        return "No matching columns found with missing values. Dataset unchanged."

    # Validate fill_value for 'fill'
    if strategy_clean == "fill" and fill_value is None:
        return "Error: 'fill_value' must be specified when using strategy 'fill'."

    # Validate numeric types for mean/median
    if strategy_clean in ("mean", "median"):
        non_numeric = [c for c in target_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            return (
                f"Error: Strategy '{strategy_clean}' requires numeric columns. "
                f"Non-numeric column(s): {non_numeric}."
            )

    rows_before = len(df)
    total_nulls_before = int(df.isnull().sum().sum())
    col_nulls_before = {c: int(df[c].isnull().sum()) for c in target_cols}
    imputation_details = {}

    if strategy_clean == "drop":
        df = df.dropna(subset=target_cols).reset_index(drop=True)
        rows_after = len(df)
        rows_dropped = rows_before - rows_after
        for c in target_cols:
            imputation_details[c] = "Dropped rows with nulls"
    elif strategy_clean == "mean":
        for c in target_cols:
            mean_val = df[c].mean()
            if pd.isna(mean_val):
                imputation_details[c] = "Skipped (all values null)"
            else:
                formatted_val = round(float(mean_val), 4)
                df[c] = df[c].fillna(mean_val)
                imputation_details[c] = f"Mean ({formatted_val})"
    elif strategy_clean == "median":
        for c in target_cols:
            median_val = df[c].median()
            if pd.isna(median_val):
                imputation_details[c] = "Skipped (all values null)"
            else:
                formatted_val = round(float(median_val), 4)
                df[c] = df[c].fillna(median_val)
                imputation_details[c] = f"Median ({formatted_val})"
    elif strategy_clean == "mode":
        for c in target_cols:
            mode_series = df[c].mode(dropna=True)
            if mode_series.empty:
                imputation_details[c] = "Skipped (no mode found)"
            else:
                mode_val = mode_series.iloc[0]
                df[c] = df[c].fillna(mode_val)
                imputation_details[c] = f"Mode ('{mode_val}')"
    elif strategy_clean == "fill":
        for c in target_cols:
            df[c] = df[c].fillna(fill_value)
            imputation_details[c] = f"Constant ('{fill_value}')"

    rows_after = len(df)
    total_nulls_after = int(df.isnull().sum().sum())

    # Build summary markdown table
    summary_rows = []
    summary_rows.append(f"### Missing Values Handling Summary (`{strategy_clean}`)")
    if strategy_clean == "drop":
        summary_rows.append(f"- **Rows Before:** {rows_before} | **Rows After:** {rows_after} (Dropped: {rows_dropped} rows)")
    else:
        summary_rows.append(f"- **Total Rows:** {rows_after}")
    summary_rows.append(f"- **Total Nulls in Dataset:** {total_nulls_before} -> {total_nulls_after}")
    summary_rows.append("")
    summary_rows.append("| Column | Nulls Before | Imputed Value / Action | Nulls After |")
    summary_rows.append("|---|---|---|---|")

    for c in target_cols:
        n_before = col_nulls_before[c]
        n_after = int(df[c].isnull().sum()) if c in df.columns else 0
        action_desc = imputation_details.get(c, "Processed")
        summary_rows.append(f"| `{c}` | {n_before} | {action_desc} | {n_after} |")

    return "\n".join(summary_rows)


def detect_and_handle_outliers(columns=None, method="iqr", action="clip"):
    """
    Detects statistical outliers in numeric columns and handles them (clip, drop, or flag).

    Parameters:
        columns (list or str, optional): Numeric column(s) to process. If None, selects all numeric columns.
        method (str): 'iqr' (Interquartile Range) or 'zscore'. Default: 'iqr'.
        action (str): 'clip' (cap bounds), 'drop' (remove outlier rows), or 'flag' (add boolean column). Default: 'clip'.
    """
    global df, last_query_df

    if df is None:
        return "No dataset loaded."

    method_clean = str(method).strip().lower() if method else "iqr"
    action_clean = str(action).strip().lower() if action else "clip"

    if method_clean not in ["iqr", "zscore"]:
        return f"Error: Unsupported outlier method '{method}'. Supported methods: 'iqr', 'zscore'."

    if action_clean not in ["clip", "drop", "flag"]:
        return f"Error: Unsupported outlier action '{action}'. Supported actions: 'clip', 'drop', 'flag'."

    # Parse columns argument
    target_cols = []
    if columns is not None:
        if isinstance(columns, str):
            cleaned_str = columns.strip()
            if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
                try:
                    parsed = json.loads(cleaned_str)
                    if isinstance(parsed, list):
                        target_cols = [str(c).strip() for c in parsed]
                except Exception:
                    target_cols = [c.strip() for c in cleaned_str.strip("[]").split(",") if c.strip()]
            else:
                target_cols = [c.strip() for c in cleaned_str.split(",") if c.strip()]
        elif isinstance(columns, (list, tuple)):
            target_cols = [str(c).strip() for c in columns if str(c).strip()]
        else:
            target_cols = [str(columns)]

    if target_cols:
        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            return f"Error: Column(s) {missing_cols} not found in dataset. Available columns: {list(df.columns)}"
        non_numeric = [c for c in target_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            return f"Error: Outlier detection requires numeric columns. Non-numeric column(s): {non_numeric}."
    else:
        target_cols = df.select_dtypes(include="number").columns.tolist()

    if not target_cols:
        return "No numeric columns found in dataset to detect outliers."

    rows_before = len(df)
    stats_per_col = {}
    combined_outlier_mask = pd.Series(False, index=df.index)

    for c in target_cols:
        series = df[c].dropna()
        if series.empty:
            stats_per_col[c] = {
                "lower": np.nan, "upper": np.nan, "count": 0, "pct": 0.0
            }
            continue

        if method_clean == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        else:  # zscore
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=0))
            if std_val == 0 or np.isnan(std_val):
                lower = mean_val
                upper = mean_val
            else:
                lower = mean_val - 3.0 * std_val
                upper = mean_val + 3.0 * std_val

        col_outlier_mask = (df[c] < lower) | (df[c] > upper)
        col_outliers_count = int(col_outlier_mask.sum())
        col_outliers_pct = round((col_outliers_count / rows_before) * 100, 2) if rows_before > 0 else 0.0

        stats_per_col[c] = {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "count": col_outliers_count,
            "pct": col_outliers_pct,
            "mask": col_outlier_mask
        }
        combined_outlier_mask |= col_outlier_mask

    # Execute action
    total_outliers_detected = sum(s["count"] for s in stats_per_col.values() if "count" in s)

    if action_clean == "clip":
        for c in target_cols:
            if c in stats_per_col and not np.isnan(stats_per_col[c]["lower"]):
                lower = stats_per_col[c]["lower"]
                upper = stats_per_col[c]["upper"]
                df[c] = df[c].clip(lower=lower, upper=upper)
        rows_after = len(df)
        action_note = "Capped values outside calculated boundaries."

    elif action_clean == "drop":
        rows_to_drop = int(combined_outlier_mask.sum())
        df = df[~combined_outlier_mask].reset_index(drop=True)
        rows_after = len(df)
        action_note = f"Dropped {rows_to_drop} rows containing outliers across target columns."

    elif action_clean == "flag":
        for c in target_cols:
            if c in stats_per_col and "mask" in stats_per_col[c]:
                flag_col = f"{c}_outlier"
                df[flag_col] = stats_per_col[c]["mask"]
        rows_after = len(df)
        action_note = "Added boolean flag columns (`<column>_outlier`) marking outlier records."

    # Build markdown summary
    summary_lines = []
    summary_lines.append("### Outlier Detection & Handling Summary")
    summary_lines.append(f"- **Method:** `{method_clean.upper()}` | **Action:** `{action_clean}`")
    summary_lines.append(f"- **Rows Before:** {rows_before} | **Rows After:** {rows_after}")
    summary_lines.append(f"- **Total Outliers Detected:** {total_outliers_detected}")
    summary_lines.append(f"- **Action Impact:** {action_note}")
    summary_lines.append("")
    summary_lines.append("| Column | Lower Bound | Upper Bound | Outliers Count | % of Rows | Action Result |")
    summary_lines.append("|---|---|---|---|---|---|")

    for c in target_cols:
        st = stats_per_col[c]
        low = st["lower"]
        up = st["upper"]
        cnt = st.get("count", 0)
        pct = st.get("pct", 0.0)
        if action_clean == "clip":
            res = f"Clipped {cnt} values" if cnt > 0 else "No clipping needed"
        elif action_clean == "drop":
            res = "Contributed to dropped rows" if cnt > 0 else "No outliers"
        elif action_clean == "flag":
            res = f"Flagged in `{c}_outlier`"
        else:
            res = "Processed"

        summary_lines.append(f"| `{c}` | {low} | {up} | {cnt} | {pct}% | {res} |")

    return "\n".join(summary_lines)


def convert_column_types(column_mapping):
    """
    Parses messy dates, strips currency symbols ($, €, £, ¥, ₹) and formatting commas,
    and cleanly converts column data types in the active dataset.

    Parameters:
        column_mapping (dict or str): e.g. {"order_date": "datetime", "price": "numeric", "in_stock": "boolean"}
    """
    global df, last_query_df

    if df is None:
        return "No dataset loaded."

    # Parse column_mapping if passed as string or JSON
    mapping = column_mapping
    if isinstance(mapping, str):
        mapping_str = mapping.strip()
        try:
            mapping = json.loads(mapping_str)
        except Exception:
            try:
                mapping = ast.literal_eval(mapping_str)
            except Exception as e:
                return f"Error: Could not parse column_mapping string. Please provide a valid dictionary: {e}"

    if not isinstance(mapping, dict) or not mapping:
        return "Error: 'column_mapping' must be a non-empty dictionary mapping column names to target types."

    missing_cols = [c for c in mapping.keys() if c not in df.columns]
    if missing_cols:
        return f"Error: Column(s) {missing_cols} not found in dataset. Available columns: {list(df.columns)}"

    results = []

    for col, target_raw in mapping.items():
        target = str(target_raw).strip().lower()
        orig_dtype = str(df[col].dtype)
        nulls_before = int(df[col].isnull().sum())

        try:
            if target in ["numeric", "float", "float64", "number", "int", "integer", "int64"]:
                series = df[col]
                # If object or string, strip currency, formatting, percentage, etc.
                if series.dtype == "object" or isinstance(series.dtype, pd.StringDtype):
                    s = series.astype(str).str.strip()
                    # Strip currency symbols: $, €, £, ¥, ₹
                    s = s.str.replace(r"[\$€£¥₹]", "", regex=True)
                    # Strip commas used as thousand separators (e.g. 1,000.50 -> 1000.50)
                    s = s.str.replace(",", "", regex=False)
                    # Strip percent signs
                    s = s.str.replace("%", "", regex=False)
                    # Convert accounting parentheses e.g. (100.50) to -100.50
                    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                    # Handle common null string representations
                    s = s.replace(["nan", "null", "none", "n/a", "na", ""], np.nan)
                    series = pd.to_numeric(s, errors="coerce")
                else:
                    series = pd.to_numeric(series, errors="coerce")

                if target in ["int", "integer", "int64"]:
                    # Round and cast to nullable integer Int64 to gracefully handle any NaN
                    series = series.round().astype("Int64")
                elif target in ["float", "float64"]:
                    series = series.astype(float)

                df[col] = series

            elif target in ["datetime", "date", "timestamp"]:
                df[col] = pd.to_datetime(df[col], errors="coerce")

            elif target in ["string", "str", "text"]:
                df[col] = df[col].astype("string")

            elif target in ["boolean", "bool"]:
                if df[col].dtype == "bool":
                    converted = df[col]
                else:
                    s = df[col].astype(str).str.strip().str.lower()
                    bool_map = {
                        "true": True, "1": True, "yes": True, "y": True, "t": True,
                        "false": False, "0": False, "no": False, "n": False, "f": False
                    }
                    mapped = s.map(bool_map)
                    converted = mapped.astype("boolean")
                df[col] = converted

            elif target in ["categorical", "category"]:
                df[col] = df[col].astype("category")

            else:
                df[col] = df[col].astype(target)

            new_dtype = str(df[col].dtype)
            nulls_after = int(df[col].isnull().sum())
            coerced_count = max(0, nulls_after - nulls_before)

            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(repr(v) for v in sample_vals) if sample_vals else "Empty"

            results.append({
                "column": col,
                "orig_dtype": orig_dtype,
                "target": target,
                "new_dtype": new_dtype,
                "coerced": coerced_count,
                "sample": sample_str,
                "status": "Success"
            })

        except Exception as e:
            results.append({
                "column": col,
                "orig_dtype": orig_dtype,
                "target": target,
                "new_dtype": orig_dtype,
                "coerced": 0,
                "sample": "-",
                "status": f"Failed: {e}"
            })

    summary_lines = []
    summary_lines.append("### Column Type Conversion Summary")
    summary_lines.append(f"- **Columns Processed:** {len(mapping)}")
    summary_lines.append("")
    summary_lines.append("| Column | Original Type | Target Type | Converted Type | Coerced to Null | Sample Values | Status |")
    summary_lines.append("|---|---|---|---|---|---|---|")

    for r in results:
        summary_lines.append(
            f"| `{r['column']}` | `{r['orig_dtype']}` | `{r['target']}` | `{r['new_dtype']}` | {r['coerced']} | {r['sample']} | {r['status']} |"
        )

    return "\n".join(summary_lines)


def plot_chart(chart_type, x=None, y=None, title=None, xlabel=None, ylabel=None):
    global df

    if df is None:
        return "No dataset loaded."

    if chart_type not in ["bar", "line", "scatter", "hist", "pie"]:
        return "Unsupported chart type."

    plt.close("all")

    fig, ax = plt.subplots(figsize=(10, 6))

    # ── BAR ──
    if chart_type == "bar":
        bars = ax.bar(df[x], df[y])
        ax.set_title(title or f"{y} vs {x}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel or x, fontsize=12)
        ax.set_ylabel(ylabel or y, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # ── LINE ──
    elif chart_type == "line":
        ax.plot(df[x], df[y], marker="o", linewidth=2, label=y)
        ax.set_title(title or f"{y} over {x}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.xticks(rotation=45)

    # ── SCATTER ──
    elif chart_type == "scatter":
        ax.scatter(df[x], df[y])
        ax.set_title(title or f"{y} vs {x}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.grid(True, linestyle="--", alpha=0.5)

    # ── HISTOGRAM ──
    elif chart_type == "hist":
        ax.hist(df[x], bins=20)
        ax.set_title(title or f"Distribution of {x}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or "Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ── PIE ──
    elif chart_type == "pie":
        ax.pie(df[y], labels=df[x], autopct="%1.1f%%", startangle=90)
        ax.set_title(title or f"{y} Distribution", fontsize=16, fontweight="bold")
        ax.legend(title=x, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    filename = f"outputs/{chart_type}_chart.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename


def export_csv(filename="exported_dataset"):
    """
    Export the current DataFrame to a CSV file.
    Returns the file path of the saved CSV.
    """
    global df

    if df is None:
        return "No dataset loaded."

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    safe_name = filename.strip().replace(" ", "_")
    if not safe_name.endswith(".csv"):
        safe_name += ".csv"

    filepath = f"outputs/{safe_name}"
    df.to_csv(filepath, index=False)

    return filepath
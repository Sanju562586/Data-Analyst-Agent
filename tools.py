import pandas as pd
import numpy as np
import io
import contextlib
import os
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
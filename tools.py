import pandas as pd
import io
import contextlib
import os
import matplotlib.pyplot as plt

df = None

def load_csv(file):
    global df

    df = pd.read_csv(file)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "preview": df.head()
    }


def analyze_data():
    global df

    if df is None:
        return "No dataset loaded."

    stats = df.describe(include="all").to_string()
    missing = df.isnull().sum().to_string()
    dtypes = df.dtypes.to_string()
    duplicates = df.duplicated().sum()
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
    global df

    if df is None:
        return "No dataset loaded."
    
    local_vars = {
        "df": df,
        "pd": pd
    }

    output = io.StringIO()

    try:
        with contextlib.redirect_stdout(output):
            exec(code, {}, local_vars)
        
        return output.getvalue() or "Code executed successfully."

    except Exception as e:
        return f"Error: {e}"



def plot_chart(chart_type, x=None, y=None, title=None, xlabel=None, ylabel=None):
    global df

    if df is None:
        return "No dataset loaded."

    if chart_type not in ["bar", "line", "scatter", "hist", "pie"]:
        return "Unsupported chart type."

    plt.close("all")

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------- BAR ---------------- #
    if chart_type == "bar":

        bars = ax.bar(df[x], df[y])

        ax.set_title(title or f"{y} vs {x}", fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel or x, fontsize=12)
        ax.set_ylabel(ylabel or y, fontsize=12)

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.xticks(rotation=45, ha="right")

        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # ---------------- LINE ---------------- #
    elif chart_type == "line":

        ax.plot(
            df[x],
            df[y],
            marker="o",
            linewidth=2,
            label=y
        )

        ax.set_title(title or f"{y} over {x}",
                     fontsize=16,
                     fontweight="bold")

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        ax.grid(True, linestyle="--", alpha=0.5)

        ax.legend()

        plt.xticks(rotation=45)

    # ---------------- SCATTER ---------------- #
    elif chart_type == "scatter":

        ax.scatter(df[x], df[y])

        ax.set_title(title or f"{y} vs {x}",
                     fontsize=16,
                     fontweight="bold")

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)

        ax.grid(True, linestyle="--", alpha=0.5)

    # ---------------- HISTOGRAM ---------------- #
    elif chart_type == "hist":

        ax.hist(df[x], bins=20)

        ax.set_title(title or f"Distribution of {x}",
                     fontsize=16,
                     fontweight="bold")

        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or "Frequency")

        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # ---------------- PIE ---------------- #
    elif chart_type == "pie":

        ax.pie(
            df[y],
            labels=df[x],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(title or f"{y} Distribution",
                     fontsize=16,
                     fontweight="bold")

        ax.legend(
            title=x,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

    plt.tight_layout()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    filename = f"outputs/{chart_type}_chart.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return filename
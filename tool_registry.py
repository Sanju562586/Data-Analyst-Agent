import json

from tools import (
    load_csv,
    analyze_data,
    run_pandas_code,
    run_sql_query,
    plot_chart,
    remove_duplicates,
    export_csv
)

with open("tools.json", "r", encoding="utf-8") as f:
    TOOL_DEFINITIONS = json.load(f)

TOOL_MAP = {
    "load_csv":          load_csv,
    "analyze_data":      analyze_data,
    "run_pandas_code":   run_pandas_code,
    "run_sql_query":     run_sql_query,
    "plot_chart":        plot_chart,
    "remove_duplicates": remove_duplicates,
    "export_csv":        export_csv
}


def get_tool(name):
    return TOOL_MAP.get(name)


def get_tool_definition(name):
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == name:
            return tool
    return None


def get_all_tool_definitions():
    return TOOL_DEFINITIONS


def build_openai_tools():
    """
    Builds standard OpenAI-compatible tool definitions
    compatible with Groq, OpenRouter, and any OpenAI-style provider.
    """
    tools = []
    for tool in TOOL_DEFINITIONS:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        })
    return tools


# Backward compatibility alias
build_function_declarations = build_openai_tools
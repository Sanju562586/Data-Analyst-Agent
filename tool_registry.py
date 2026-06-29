import json
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

from tools import (
    load_csv,
    analyze_data,
    run_pandas_code,
    plot_chart
)

with open("tools.json", "r", encoding="utf-8") as f:
    TOOL_DEFINITIONS = json.load(f)

TOOL_MAP = {
    "load_csv": load_csv,
    "analyze_data": analyze_data,
    "run_pandas_code": run_pandas_code,
    "plot_chart": plot_chart
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

# UNDERSTAND THIS
def build_function_declarations():

    declarations = []

    for tool in TOOL_DEFINITIONS:
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"]
            )
        )

    return declarations
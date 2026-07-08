
from tool_registry import get_tool
from tool_registry import build_function_declarations
from tools import (
    load_csv,
    analyze_data,
    run_pandas_code,
    plot_chart,
    remove_duplicates,
    export_csv
)
import os
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

import dotenv
import json

dotenv.load_dotenv()
gemini_key = os.getenv("GEMINI")
client = genai.Client(api_key=gemini_key)

gemini_tools = [
    types.Tool(
        function_declarations=build_function_declarations()
    )
]

TOOL_MAP = {
    "load_csv":          load_csv,
    "analyze_data":      analyze_data,
    "run_pandas_code":   run_pandas_code,
    "plot_chart":        plot_chart,
    "remove_duplicates": remove_duplicates,
    "export_csv":        export_csv
}

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("tools.json", "r", encoding="utf-8") as f:
    TOOL_DESCRIPTIONS = json.load(f)


class DataAnalysisAgent:

    def __init__(self):
        self.client = client
        self.gemini_tools = gemini_tools

    def _generate_response(self, contents):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=self.gemini_tools
            )
        )
        return response

    def _generate_response_stream(self, contents):
        stream = self.client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=self.gemini_tools
            )
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _execute_tool(self, function_call):
        tool_name = function_call.name
        args = function_call.args

        if tool_name not in TOOL_MAP:
            return tool_name, f"Tool '{tool_name}' not found."

        try:
            result = TOOL_MAP[tool_name](**args)
        except Exception as e:
            result = f"Tool execution failed: {e}"

        return tool_name, result

    def _build_tool_response(self, tool_name, tool_output):
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=tool_name,
                    response={
                        "result": str(tool_output)
                    }
                )
            ]
        )

    def run_agent_stream(self, user_query, chat_history=None):
        """
        Streaming agent loop — yields typed events in real time:
          {"type": "tool_start", "name": <str>}
          {"type": "tool_done",  "name": <str>}
          {"type": "text",       "content": <str>}   ← streamed word-by-word
          {"type": "chart",      "content": <path>}
          {"type": "csv",        "content": <path>}
        """
        messages = []

        if chat_history:
            for msg in chat_history:
                role = "model" if msg["role"] == "assistant" else "user"
                messages.append(
                    {
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    }
                )
        else:
            messages.append(
                {
                    "role": "user",
                    "parts": [{"text": user_query}]
                }
            )

        last_chart_filename = None
        last_csv_filename   = None

        while True:

            response = self._generate_response(messages)

            if response.function_calls:

                messages.append({"role": "model", "parts": response.parts})

                for function_call in response.function_calls:

                    yield {"type": "tool_start", "name": function_call.name}

                    try:
                        tool_name, tool_result = self._execute_tool(function_call)

                        if function_call.name == "plot_chart" and isinstance(tool_result, str) and tool_result.endswith(".png"):
                            last_chart_filename = tool_result

                        if function_call.name == "export_csv" and isinstance(tool_result, str) and tool_result.endswith(".csv"):
                            last_csv_filename = tool_result

                    except Exception as e:
                        tool_name  = function_call.name
                        tool_result = f"Tool execution failed: {e}"

                    yield {"type": "tool_done", "name": tool_name}

                    messages.append(self._build_tool_response(tool_name, tool_result))

                continue   # next model turn

            # ── No more tool calls → emit final response ──────────────────────
            if last_csv_filename:
                yield {"type": "csv", "content": last_csv_filename}
                return

            if last_chart_filename:
                yield {"type": "chart", "content": last_chart_filename}
                return

            # Stream text chunks
            for chunk in self._generate_response_stream(messages):
                yield {"type": "text", "content": chunk}
            return
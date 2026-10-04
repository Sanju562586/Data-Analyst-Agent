
import tools
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
import time
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

    def __init__(self, model: str = None):
        self.client = client
        self.gemini_tools = gemini_tools
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    def _get_system_instruction(self):
        instruction = SYSTEM_PROMPT
        if hasattr(tools, "df") and tools.df is not None:
            instruction += (
                f"\n\nCURRENT DATASET STATUS: A dataset is ALREADY loaded in memory with "
                f"{tools.df.shape[0]} rows and {tools.df.shape[1]} columns. "
                f"Columns: {list(tools.df.columns)}. "
                f"Do NOT call load_csv."
            )
        return instruction

    def _generate_response(self, contents, enable_tools=True):
        config_kwargs = {
            "system_instruction": self._get_system_instruction()
        }
        if enable_tools:
            config_kwargs["tools"] = self.gemini_tools

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return response

    def _execute_tool(self, function_call):
        tool_name = function_call.name
        args = function_call.args or {}

        if tool_name not in TOOL_MAP:
            return tool_name, f"Tool '{tool_name}' not found."

        try:
            result = TOOL_MAP[tool_name](**args)
        except Exception as e:
            result = f"Tool execution failed: {e}"

        return tool_name, result

    def _build_tool_response(self, tool_name, tool_output):
        return types.Content(
            role="user",
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
        max_iterations = 6
        iteration = 0
        executed_calls = set()
        final_text = ""

        while iteration < max_iterations:
            iteration += 1

            response = self._generate_response(messages, enable_tools=True)

            if response.function_calls:

                # Check if this exact batch of tool calls was already executed to prevent infinite ping-pong
                is_repeating = True
                for fc in response.function_calls:
                    call_sig = (fc.name, json.dumps(dict(fc.args or {}), sort_keys=True))
                    if call_sig not in executed_calls:
                        is_repeating = False
                        break

                if is_repeating:
                    # Model is stuck repeating the exact same tool calls; break loop
                    break

                if response.candidates and response.candidates[0].content:
                    messages.append(response.candidates[0].content)
                else:
                    messages.append({"role": "model", "parts": response.parts})

                tool_parts = []

                for function_call in response.function_calls:
                    call_sig = (function_call.name, json.dumps(dict(function_call.args or {}), sort_keys=True))
                    executed_calls.add(call_sig)

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

                    tool_parts.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={
                                "result": str(tool_result)
                            }
                        )
                    )

                messages.append(
                    types.Content(
                        role="user",
                        parts=tool_parts
                    )
                )

                continue   # next model turn

            # Model produced a natural language answer without function calls
            if hasattr(response, "text") and response.text:
                final_text = response.text
            break

        # If loop reached max iterations or ended without text, request a final answer without tools
        if not final_text:
            try:
                final_response = self._generate_response(messages, enable_tools=False)
                if hasattr(final_response, "text") and final_response.text:
                    final_text = final_response.text
            except Exception:
                pass

        # ── Emit final outputs ──────────────────────────────────────────
        if last_csv_filename:
            yield {"type": "csv", "content": last_csv_filename}

        if last_chart_filename:
            yield {"type": "chart", "content": last_chart_filename}

        if final_text:
            words = final_text.split(" ")
            for i, word in enumerate(words):
                space = " " if i < len(words) - 1 else ""
                yield {"type": "text", "content": word + space}
                time.sleep(0.012)
        elif not last_csv_filename and not last_chart_filename:
            yield {"type": "text", "content": "Analysis complete."}

        return
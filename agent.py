import os
import time
import json
import re
import dotenv
from typing import List, Dict, Any, Optional

try:
    # pyrefly: ignore [missing-import]
    from openai import OpenAI
except ImportError:
    OpenAI = None

import tools
from tool_registry import build_openai_tools, get_tool
from tools import (
    load_csv,
    analyze_data,
    run_pandas_code,
    run_sql_query,
    plot_chart,
    remove_duplicates,
    handle_missing_values,
    detect_and_handle_outliers,
    convert_column_types,
    export_csv
)

dotenv.load_dotenv()

TOOL_MAP = {
    "load_csv":                   load_csv,
    "analyze_data":               analyze_data,
    "run_pandas_code":            run_pandas_code,
    "run_sql_query":              run_sql_query,
    "plot_chart":                 plot_chart,
    "remove_duplicates":          remove_duplicates,
    "handle_missing_values":      handle_missing_values,
    "detect_and_handle_outliers": detect_and_handle_outliers,
    "convert_column_types":       convert_column_types,
    "export_csv":                 export_csv
}

SYSTEM_PROMPT_PATH = "system_prompt.txt"
if os.path.exists(SYSTEM_PROMPT_PATH):
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = "You are an expert AI Data Analysis Assistant."


class ModelCandidate:
    """Represents a specific LLM model endpoint with its provider and credentials."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        headers: Optional[Dict[str, str]] = None
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.name = f"{provider} - {model}"

    def create_client(self) -> Any:
        if OpenAI is None:
            raise ImportError(
                "The 'openai' Python library is required. Please install it using: pip install openai groq"
            )
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.headers,
            timeout=60.0
        )


class DataAnalysisAgent:
    """
    Intelligent Data Analysis Agent with automated multi-tier model and provider fallback.
    Supported Providers:
      - Groq (ultra-fast inference, high intelligence)
      - OpenRouter (broad model catalog & redundancy)
    """

    def __init__(self):
        self.tools = build_openai_tools()
        self._client_cache = {}

    def _build_candidates(self) -> List[ModelCandidate]:
        """Dynamically builds the fallback cascade based on environment configuration."""
        dotenv.load_dotenv(override=True)

        groq_key = (os.getenv("GROQ_API_KEY") or os.getenv("GROQ") or "").strip()
        openrouter_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPEN_ROUTER")
            or os.getenv("OPENROUTER")
            or ""
        ).strip()

        # Groq model configurations
        groq_primary = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        groq_fallback = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant").strip()

        # OpenRouter model configurations
        openrouter_primary = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct").strip()
        openrouter_fallback = os.getenv("OPENROUTER_FALLBACK_MODEL", "google/gemini-2.0-flash-001").strip()

        openrouter_headers = {
            "HTTP-Referer": "https://github.com/Data-Analyst-Agent",
            "X-Title": "DataSense AI"
        }

        order_str = os.getenv("PROVIDER_ORDER", "groq,openrouter")
        provider_order = [p.strip().lower() for p in order_str.split(",") if p.strip()]

        candidates: List[ModelCandidate] = []

        for provider in provider_order:
            if provider == "groq" and groq_key:
                if groq_primary:
                    candidates.append(
                        ModelCandidate("Groq", groq_primary, "https://api.groq.com/openai/v1", groq_key)
                    )
                if groq_fallback and groq_fallback != groq_primary:
                    candidates.append(
                        ModelCandidate("Groq", groq_fallback, "https://api.groq.com/openai/v1", groq_key)
                    )
            elif provider == "openrouter" and openrouter_key:
                if openrouter_primary:
                    candidates.append(
                        ModelCandidate(
                            "OpenRouter", openrouter_primary, "https://openrouter.ai/api/v1",
                            openrouter_key, openrouter_headers
                        )
                    )
                if openrouter_fallback and openrouter_fallback != openrouter_primary:
                    candidates.append(
                        ModelCandidate(
                            "OpenRouter", openrouter_fallback, "https://openrouter.ai/api/v1",
                            openrouter_key, openrouter_headers
                        )
                    )

        # In case a provider wasn't listed in PROVIDER_ORDER but has a valid key:
        if "groq" not in provider_order and groq_key:
            candidates.append(
                ModelCandidate("Groq", groq_primary, "https://api.groq.com/openai/v1", groq_key)
            )
            if groq_fallback and groq_fallback != groq_primary:
                candidates.append(
                    ModelCandidate("Groq", groq_fallback, "https://api.groq.com/openai/v1", groq_key)
                )

        if "openrouter" not in provider_order and openrouter_key:
            candidates.append(
                ModelCandidate(
                    "OpenRouter", openrouter_primary, "https://openrouter.ai/api/v1",
                    openrouter_key, openrouter_headers
                )
            )
            if openrouter_fallback and openrouter_fallback != openrouter_primary:
                candidates.append(
                    ModelCandidate(
                        "OpenRouter", openrouter_fallback, "https://openrouter.ai/api/v1",
                        openrouter_key, openrouter_headers
                    )
                )

        return candidates

    def _get_client_for_candidate(self, candidate: ModelCandidate) -> Any:
        cache_key = (candidate.base_url, candidate.api_key)
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = candidate.create_client()
        return self._client_cache[cache_key]

    def _get_system_instruction(self) -> str:
        instruction = SYSTEM_PROMPT
        if hasattr(tools, "df") and tools.df is not None:
            cols = list(tools.df.columns)
            special_cols = [c for c in cols if not str(c).isidentifier()]
            instruction += (
                f"\n\nCURRENT DATASET STATUS: A dataset is ALREADY loaded in memory with "
                f"{tools.df.shape[0]} rows and {tools.df.shape[1]} columns.\n"
                f"Available Columns: {cols}.\n"
                f"Do NOT call load_csv."
            )
            if special_cols:
                quoted = ", ".join(f'"{c}"' for c in special_cols)
                instruction += (
                    f"\nCRITICAL COLUMN QUOTING IN SQL: The following columns contain spaces or special characters: {special_cols}. "
                    f"When writing SQL queries (`run_sql_query`), you MUST enclose them in ANSI double quotes (e.g. {quoted}). "
                    f"NEVER use backticks (` `), which cause DuckDB syntax errors."
                )
        return instruction

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]):
        if tool_name not in TOOL_MAP:
            return tool_name, f"Tool '{tool_name}' not found."

        try:
            result = TOOL_MAP[tool_name](**args)
        except Exception as e:
            result = f"Tool execution failed: {e}"

        return tool_name, result

    def run_agent_stream(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """
        Streaming agent loop with automated multi-tier provider & model fallback:
          Yields typed events in real time:
            {"type": "tool_start", "name": <str>}
            {"type": "tool_done",  "name": <str>}
            {"type": "fallback",   "content": <str>}
            {"type": "text",       "content": <str>}   (streamed word-by-word)
            {"type": "chart",      "content": <path>}
            {"type": "csv",        "content": <path>}
        """
        candidates = self._build_candidates()

        if not candidates:
            yield {
                "type": "text",
                "content": (
                    "⚠️ **No LLM API Keys Configured**\n\n"
                    "Please configure at least one API key in your `.env` file:\n\n"
                    "- `GROQ_API_KEY=your_groq_api_key_here` (Get a free key at https://console.groq.com)\n"
                    "- `OPENROUTER_API_KEY=your_openrouter_api_key_here` (Get a key at https://openrouter.ai)\n\n"
                    "DataSense AI will automatically switch between providers and models whenever an outage or rate limit occurs."
                )
            }
            return

        # Prepare messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._get_system_instruction()}
        ]

        if chat_history:
            for msg in chat_history:
                role = "assistant" if msg["role"] == "assistant" else "user"
                content = msg.get("content")
                if content:
                    messages.append({"role": role, "content": str(content)})
        else:
            messages.append({"role": "user", "content": user_query})

        last_chart_filename = None
        last_csv_filename = None
        last_table_df = None
        last_query_sql = None
        max_iterations = 8
        iteration = 0
        executed_calls = set()
        final_text = ""

        active_candidate_idx = 0
        total_candidates = len(candidates)

        while iteration < max_iterations:
            iteration += 1

            # Attempt model generation with automated fallback cascade
            response_msg = None
            errors_encountered = []

            while active_candidate_idx < total_candidates:
                candidate = candidates[active_candidate_idx]
                try:
                    client = self._get_client_for_candidate(candidate)
                    completion = client.chat.completions.create(
                        model=candidate.model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.2
                    )
                    response_msg = completion.choices[0].message
                    break  # Success!

                except Exception as e:
                    err_summary = str(e)
                    # Shorten verbose HTTP response text if present
                    if len(err_summary) > 160:
                        err_summary = err_summary[:160] + "..."

                    errors_encountered.append(f"{candidate.name}: {err_summary}")
                    failed_candidate_name = candidate.name
                    active_candidate_idx += 1

                    if active_candidate_idx < total_candidates:
                        next_candidate = candidates[active_candidate_idx]
                        yield {
                            "type": "fallback",
                            "content": f"⚠️ {failed_candidate_name} failed. Falling back to {next_candidate.name}..."
                        }
                    else:
                        break

            if response_msg is None:
                # All candidates in the cascade failed
                all_errors = "\n".join(f"- {e}" for e in errors_encountered)
                yield {
                    "type": "text",
                    "content": (
                        f"❌ **All LLM providers and fallback models failed.**\n\n"
                        f"Details:\n{all_errors}\n\n"
                        "Please check your API keys or rate limits in `.env`."
                    )
                }
                return

            # Check for tool calls
            tool_calls = getattr(response_msg, "tool_calls", None)

            if tool_calls:
                # Loop prevention check
                is_repeating = True
                for tc in tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments or ""
                    call_sig = (fn_name, fn_args)
                    if call_sig not in executed_calls:
                        is_repeating = False
                        break

                if is_repeating:
                    # Model repeated the exact same tool calls; break to get final answer
                    break

                # Append assistant tool invocation message to conversation history
                assistant_msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response_msg.content or ""
                }
                assistant_msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg_dict)

                # Execute each tool call
                for tc in tool_calls:
                    fn_name = tc.function.name
                    call_sig = (fn_name, tc.function.arguments or "")
                    executed_calls.add(call_sig)

                    yield {"type": "tool_start", "name": fn_name}

                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}

                    tool_name, tool_result = self._execute_tool(fn_name, args)

                    if fn_name in ("run_sql_query", "run_pandas_code"):
                        if hasattr(tools, "last_query_df") and tools.last_query_df is not None:
                            new_df = tools.last_query_df
                            if last_table_df is None or len(new_df) > 1 or len(last_table_df) <= 1:
                                last_table_df = new_df
                                if fn_name == "run_sql_query":
                                    last_query_sql = args.get("query", "")
                                else:
                                    last_query_sql = args.get("code", "")

                    if fn_name == "plot_chart" and isinstance(tool_result, str) and tool_result.endswith(".png"):
                        last_chart_filename = tool_result

                    if fn_name == "export_csv" and isinstance(tool_result, str) and tool_result.endswith(".csv"):
                        last_csv_filename = tool_result

                    yield {"type": "tool_done", "name": tool_name}

                    tool_content = str(tool_result)
                    if "error" in tool_content.lower() or "failed" in tool_content.lower():
                        tool_content += (
                            "\n[SYSTEM DIRECTIVE: Silently correct the query or code and execute the tool again immediately. "
                            "Do NOT explain the error, your attempts, or debugging steps to the user. "
                            "Output only the direct, concise analytical answer once the result is obtained.]"
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": tool_content
                    })

                continue  # Next model turn with tool outputs

            # If the model didn't call tools, check if it accidentally dumped unexecuted code in text
            if response_msg.content:
                content = response_msg.content
                has_code_dump = (
                    ("run_pandas_code" in content or "run_sql_query" in content or "import pandas" in content)
                    and ("```" in content or "SELECT " in content or "df[" in content)
                )
                user_wants_code = any(w in user_query.lower() for w in ["show code", "write code", "python code", "sql query", "provide code", "give code"])

                if has_code_dump and not user_wants_code and iteration < max_iterations:
                    py_match = re.search(r'```(?:python|py)?\s*(.*?df\[.*?)```', content, re.DOTALL)
                    sql_match = re.search(r'```(?:sql)?\s*(SELECT.*?)```', content, re.DOTALL | re.IGNORECASE)

                    if py_match:
                        extracted_code = py_match.group(1).strip()
                        yield {"type": "tool_start", "name": "run_pandas_code"}
                        t_name, t_res = self._execute_tool("run_pandas_code", {"code": extracted_code})
                        yield {"type": "tool_done", "name": "run_pandas_code"}
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[SYSTEM DIRECTIVE: Code was executed with output:\n{t_res}\n\n"
                                "Now present ONLY the direct, concise final answer to the user. "
                                "Do NOT include any code, explanations of errors, tool names, or internal narration.]"
                            )
                        })
                        continue
                    elif sql_match:
                        extracted_sql = sql_match.group(1).strip()
                        yield {"type": "tool_start", "name": "run_sql_query"}
                        t_name, t_res = self._execute_tool("run_sql_query", {"query": extracted_sql})
                        yield {"type": "tool_done", "name": "run_sql_query"}
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[SYSTEM DIRECTIVE: Query was executed with output:\n{t_res}\n\n"
                                "Now present ONLY the direct, concise final answer to the user. "
                                "Do NOT include any code, explanations of errors, tool names, or internal narration.]"
                            )
                        })
                        continue

                final_text = content
            break

        # If loop reached max iterations or ended without text, request final summary without tools
        if not final_text:
            while active_candidate_idx < total_candidates:
                candidate = candidates[active_candidate_idx]
                try:
                    client = self._get_client_for_candidate(candidate)
                    completion = client.chat.completions.create(
                        model=candidate.model,
                        messages=messages,
                        temperature=0.2
                    )
                    final_text = completion.choices[0].message.content or ""
                    break
                except Exception:
                    active_candidate_idx += 1

        # Emit final artifacts and streamed text
        if last_table_df is not None:
            yield {
                "type": "table",
                "content": last_table_df,
                "row_count": len(last_table_df),
                "query": last_query_sql
            }

        if last_csv_filename:
            yield {"type": "csv", "content": last_csv_filename}

        if last_chart_filename:
            yield {"type": "chart", "content": last_chart_filename}

        if final_text:
            cleaned_text = self._sanitize_response_text(
                final_text,
                has_table_artifact=(last_table_df is not None),
                user_query=user_query
            )
            if cleaned_text:
                words = cleaned_text.split(" ")
                for i, word in enumerate(words):
                    space = " " if i < len(words) - 1 else ""
                    yield {"type": "text", "content": word + space}
                    time.sleep(0.012)
            elif last_table_df is not None:
                yield {"type": "text", "content": "Here are the query results displayed in the table below."}
            else:
                yield {"type": "text", "content": "Analysis complete."}
        elif not last_csv_filename and not last_chart_filename and last_table_df is None:
            yield {"type": "text", "content": "Analysis complete."}

        return

    @staticmethod
    def _sanitize_response_text(text: str, has_table_artifact: bool = False, user_query: str = "") -> str:
        """
        Cleans the agent response text:
        1. Strips out dummy markdown tables that only contain ellipsis (...) or dot placeholders.
        2. Strips out internal meta-commentary, debugging narration, and retry chatter.
        3. Strips out unprompted raw code blocks if the user did not ask for code.
        """
        if not text:
            return text

        # 1. Regex matching markdown tables where all rows below header are ellipses / dots / dashes
        table_pattern = re.compile(
            r'(\n|^)(\|[^\n]+\|\s*\n\|[-:\s|]+\|\s*\n(?:\|[\s\.\…\-\*|]+\|\s*(?:\n|$))+)',
            re.MULTILINE
        )

        def is_dummy_table(match):
            table_block = match.group(2)
            lines = [line.strip() for line in table_block.strip().splitlines() if line.strip()]
            if len(lines) < 3:
                return match.group(0)
            body_rows = lines[2:]
            is_all_dummy = True
            for row in body_rows:
                content = row.replace("|", "").strip()
                if re.search(r'[a-zA-Z0-9]', content):
                    is_all_dummy = False
                    break
            if is_all_dummy:
                if has_table_artifact:
                    return "\n\n*The complete matching records are displayed in the interactive table below.*"
                return ""
            return match.group(0)

        cleaned = table_pattern.sub(is_dummy_table, text)

        # 2. Strip internal debugging narration, retries, and trial-and-error explanations
        meta_patterns = [
            r'(?i)(?:to find [^,\n]+, )?we (?:first )?need to ensure that the [^\n]+\n*',
            r'(?i)however,\s+the original dataset description does not explicitly state this[^\n]*\n*',
            r'(?i)assuming (?:that )?the [\'"][^\'"]+[\'"] column does indeed represent[^\n]*\n*',
            r'(?i)(?:since|because)\s+the\s+(?:sql\s+)?query\s+failed[^\n]*\n*',
            r'(?i)let[\'’]s\s+try\s+again\s+with\s+(?:a\s+)?modified\s+query[^\n]*:?\n*',
            r'(?i)however,\s+the\s+error\s+message\s+indicates[^\n]*\n*',
            r'(?i)we\s+can\s+use\s+the\s+(?:run_pandas_code|run_sql_query)\s+tool\s+to\s+achieve\s+the\s+same\s+result:?\n*',
            r'(?i)the\s+sql\s+query\s+failed[^\n]*\n*',
            r'(?i)error message indicates that the sql parser is not able to handle[^\n]*\n*'
        ]
        for mp in meta_patterns:
            cleaned = re.sub(mp, "", cleaned)

        # 3. Clean up dangling introductions followed by the replaced notice or empty lines
        cleaned = re.sub(
            r'(?:(?:Here are the details|Below are the records|Here is the list)[^\n:]*:\s*\n*)(\*The complete matching records are displayed in the interactive table below\.\*)',
            r'\1',
            cleaned,
            flags=re.IGNORECASE
        )

        # 4. If the user didn't request code, strip raw code blocks that resulted from tool narration
        user_wants_code = any(w in user_query.lower() for w in ["show code", "write code", "python code", "sql query", "provide code", "give code"])
        if not user_wants_code:
            cleaned = re.sub(r'```(?:python|py)?\s*(?:import pandas|# Assuming df).*?```', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'SELECT\s+AVG\([^)]+\)\s+FROM\s+df\s+WHERE[^\n]+(?:\n|$)', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'```(?:sql)?\s*SELECT.*?```', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned
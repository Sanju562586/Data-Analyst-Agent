# DataSense AI — Deployment Guide

> **Intelligent data analysis powered by Gemini 2.5 Flash**  
> Upload any CSV → ask questions → get instant analysis, charts, and insights.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Deploying to Hugging Face Spaces](#3-deploying-to-hugging-face-spaces) ⭐ Recommended
4. [Running Locally](#4-running-locally)
5. [Environment Variables](#5-environment-variables)
6. [File Structure](#6-file-structure)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Project Overview

DataSense AI is a Streamlit application that wraps **Google Gemini 2.5 Flash** with a set of data analysis tools. The agent autonomously decides which tools to call based on user questions, streams responses in real time, and renders charts directly in the chat.

### Tools Available

| Tool | Description |
|------|-------------|
| `load_csv` | Loads CSV into memory as a pandas DataFrame |
| `analyze_data` | Full EDA — stats, nulls, duplicates, correlations |
| `run_pandas_code` | Executes custom pandas code on the dataset |
| `plot_chart` | Generates bar / line / scatter / hist / pie charts |
| `remove_duplicates` | Deduplicates rows |
| `export_csv` | Exports the current (cleaned) dataset |

---

## 2. Architecture

```
User Browser
    │
    ▼
┌─────────────────────────────────┐
│         app.py (Streamlit)      │
│  - File upload UI               │
│  - Chat message rendering       │
│  - Live streaming event loop    │
└────────────┬────────────────────┘
             │  run_agent_stream()
             ▼
┌─────────────────────────────────┐
│         agent.py                │
│  - DataAnalysisAgent            │
│  - Gemini 2.5 Flash API calls   │
│  - Tool execution loop          │
│  - Yields events: tool / text   │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 tools.py      Google GenAI API
 (6 tools)     (gemini-2.5-flash)
```

---

## 3. Deploying to Hugging Face Spaces

> **Estimated time: ~10 minutes**

### Step 1 — Create a Hugging Face Account

Go to [huggingface.co](https://huggingface.co) and sign up for a free account.

---

### Step 2 — Create a New Space

1. Click your profile icon → **New Space**
2. Fill in the form:

   | Field | Value |
   |-------|-------|
   | **Space name** | `datasense-ai` (or any name you like) |
   | **License** | MIT |
   | **SDK** | **Streamlit** ← important |
   | **SDK Version** | 1.35.0 or latest |
   | **Hardware** | CPU Basic (free) |
   | **Visibility** | Public or Private |

3. Click **Create Space**

---

### Step 3 — Add Your Gemini API Key as a Secret

> This keeps your key safe — never hardcode it in source files.

1. Inside your Space, go to **Settings** → **Variables and secrets**
2. Click **New secret**
3. Set:
   - **Name:** `GEMINI`
   - **Value:** your Google AI Studio API key
4. Click **Save**

> Get your free API key at [aistudio.google.com](https://aistudio.google.com)

---

### Step 4 — Upload the Project Files

You need to push these files to your Space's Git repository:

```
datasense-ai/
├── app.py               ← Streamlit UI + streaming chat
├── agent.py             ← Gemini agent loop
├── tools.py             ← 6 analysis tool functions
├── tool_registry.py     ← Tool → function mapping
├── tools.json           ← Tool schemas for Gemini
├── system_prompt.txt    ← Agent system instructions
├── requirements.txt     ← Python dependencies
└── .gitignore           ← (exclude .env and venv)
```

**Option A — Git push (recommended)**

```bash
# Clone your empty Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/datasense-ai
cd datasense-ai

# Copy your project files in (everything except venv/ and .env)
# Then commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option B — Web UI upload**

1. In your Space, click **Files** → **Add file** → **Upload files**
2. Drag and drop all the files listed above
3. Click **Commit changes**

---

### Step 5 — Verify the Space Builds

After pushing, Hugging Face will automatically:

1. Detect `requirements.txt` and install all dependencies
2. Detect `app.py` and launch Streamlit on port 7860
3. Show a **Building** → **Running** status badge

> ⏱ Build usually takes **2–4 minutes** on first deploy.

Once status shows 🟢 **Running**, your app is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/datasense-ai
```

---

### Step 6 — Test the Deployment

1. Open your Space URL
2. Upload any `.csv` file from the sidebar
3. Ask: *"Summarize this dataset"*
4. Watch the tool step pills appear and text stream in real time ✦

---

### Important: `.gitignore` for Hugging Face

Make sure your `.gitignore` contains at minimum:

```gitignore
.env
venv/
__pycache__/
*.pyc
outputs/
*.png
```

> ⚠️ **Never push `.env`** — your API key lives in Space Secrets, not in the repo.

---

## 4. Running Locally

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/datasense-ai.git
cd datasense-ai

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create the .env file
echo GEMINI=your_api_key_here > .env

# 5. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 5. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI` | ✅ Yes | Your Google AI Studio API key |

### Local: `.env` file

```env
GEMINI=AIza...your_key_here
```

### Hugging Face: Space Secrets

Set `GEMINI` in **Settings → Variables and secrets → New secret**.

The app reads the key with:

```python
import dotenv, os
dotenv.load_dotenv()           # reads .env locally
gemini_key = os.getenv("GEMINI")  # reads HF secret on cloud
```

`dotenv.load_dotenv()` is a safe no-op on Hugging Face (no `.env` file present), so `os.getenv("GEMINI")` reads the injected secret directly — no code change needed between local and deployed environments.

---

## 6. File Structure

```
datasense-ai/
│
├── app.py
│   ├── CSS injection        Premium dark glassmorphism theme
│   ├── Sidebar              File upload, stats, column chips
│   ├── Hero header          Branded title with Live badge
│   ├── Chat history         Replays previous messages
│   └── Event loop           Consumes run_agent_stream() in real time
│
├── agent.py
│   ├── DataAnalysisAgent    Main class
│   ├── _generate_response   Blocking Gemini call (for tool rounds)
│   ├── _generate_response_stream   Streaming Gemini call (final reply)
│   ├── _execute_tool        Dispatches named tool calls
│   └── run_agent_stream()   Generator — yields typed events live
│
├── tools.py
│   ├── load_csv()
│   ├── analyze_data()
│   ├── run_pandas_code()
│   ├── plot_chart()
│   ├── remove_duplicates()
│   └── export_csv()
│
├── tool_registry.py         Builds Gemini FunctionDeclaration objects
│
├── tools.json               JSON schema for all 6 tools
│
├── system_prompt.txt        System instructions for the agent
│
└── requirements.txt         Python package dependencies
```

---

## 7. Troubleshooting

### ❌ API key not found

```
google.api_core.exceptions.Unauthenticated: API key not valid
```

**Fix:** Ensure the Space secret is named exactly **`GEMINI`** (all caps, no quotes).

---

### ❌ Build fails: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'google.genai'
```

**Fix:** Ensure `requirements.txt` contains `google-genai>=0.8.0` (not `google-generativeai` — these are different packages).

---

### ❌ Sidebar not visible

The sidebar expand arrow is hidden when `header { visibility: hidden; }` is present in CSS.  
**Fix:** Only hide `#MainMenu` and `footer` — never `header`:

```css
/* Correct */
#MainMenu, footer { visibility: hidden; }
```

---

### ❌ Responses appear all at once (no streaming)

**Fix:** The `for event in agent.run_agent_stream(...)` loop must be inside a live Streamlit rendering context. Ensure it is within a `with st.chat_message("assistant"):` block and the loop runs during the active request, not inside a `@st.cache` or background thread.

---

### ❌ Chart not displaying

**Fix:** The `outputs/` directory is created automatically by `plot_chart()` on first use. If it persists as missing, check that the app has write permission in its working directory (HF Spaces sandbox allows this by default).

---

## Quick Deploy Checklist

```
☐ Created Hugging Face Space  (SDK: Streamlit)
☐ Added GEMINI secret         (Settings → Variables and secrets)
☐ Uploaded all 8 project files
☐ Confirmed .env is NOT committed to the repo
☐ Waited for build status → 🟢 Running
☐ Tested with a sample CSV upload
☐ Verified streaming responses work end-to-end
```

---

*Built with Streamlit · Google Gemini 2.5 Flash · Python*

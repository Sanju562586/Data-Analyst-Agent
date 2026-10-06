# pyrefly: ignore [missing-import]
import os
import time
import dotenv
import streamlit as st
from agent import DataAnalysisAgent
from tools import load_csv

dotenv.load_dotenv()

st.set_page_config(
    page_title="DataSense AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-base:        #080c18;
    --bg-card:        rgba(255,255,255,0.035);
    --bg-card-hover:  rgba(255,255,255,0.06);
    --border:         rgba(255,255,255,0.07);
    --accent-1:       #6366f1;
    --accent-2:       #8b5cf6;
    --accent-3:       #06b6d4;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --radius:         16px;
    --radius-sm:      10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 90% 55% at 50% -10%, rgba(99,102,241,0.20) 0%, transparent 65%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(6,182,212,0.07) 0%, transparent 60%),
        var(--bg-base) !important;
}

/* Hide only the menu and footer — NOT the header (sidebar toggle lives there) */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 5rem !important; max-width: 1200px !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 99px; }

/* ════════════════════════════════
   HERO HEADER
════════════════════════════════ */
.hero-wrap {
    display: flex; align-items: center; gap: 1.1rem;
    padding: 1.8rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #06b6d4 100%);
    border-radius: 15px; display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.3), 0 0 32px rgba(99,102,241,0.45);
}
.hero-text h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem; font-weight: 700;
    background: linear-gradient(90deg, #f1f5f9 0%, #a5b4fc 55%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.15;
}
.hero-text p { font-size: 0.85rem; color: var(--text-secondary); margin: 0.3rem 0 0; letter-spacing: 0.01em; }
.hero-badge {
    margin-left: auto; flex-shrink: 0;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.28);
    color: #34d399; font-size: 0.7rem; font-weight: 700;
    padding: 0.28rem 0.8rem; border-radius: 99px;
    letter-spacing: 0.08em; text-transform: uppercase;
    box-shadow: 0 0 12px rgba(16,185,129,0.15);
    animation: pulse-badge 2.5s ease-in-out infinite;
}
@keyframes pulse-badge {
    0%,100% { box-shadow: 0 0 8px rgba(16,185,129,0.15); }
    50%      { box-shadow: 0 0 18px rgba(16,185,129,0.3); }
}

/* ════════════════════════════════
   SIDEBAR
════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, rgba(11,14,25,0.99) 0%, rgba(8,10,20,0.99) 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.4rem 1.1rem; }

.sidebar-logo {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0 0 1.3rem; border-bottom: 1px solid var(--border); margin-bottom: 1.3rem;
}
.sidebar-logo .dot {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.25), 0 0 18px rgba(99,102,241,0.4);
}
.sidebar-logo span {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem;
    background: linear-gradient(90deg, #e2e8f0, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.sidebar-section {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.13em;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 0.65rem; margin-top: 0.5rem;
}

.stat-row { display: flex; gap: 0.5rem; margin: 0.85rem 0; }
.stat-card {
    flex: 1; background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: var(--radius-sm); padding: 0.7rem 0.5rem; text-align: center;
    transition: background .2s;
}
.stat-card:hover { background: rgba(99,102,241,0.13); }
.stat-card .val {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-card .lbl { font-size: 0.63rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.15rem; }

.col-chip-wrap { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; }
.col-chip {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.22);
    color: #a5b4fc; font-size: 0.68rem; font-weight: 500;
    padding: 0.18rem 0.5rem; border-radius: 99px;
    transition: background .15s;
}
.col-chip:hover { background: rgba(99,102,241,0.2); }

[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.04) !important;
    border: 1.5px dashed rgba(99,102,241,0.3) !important;
    border-radius: var(--radius) !important;
    transition: all .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
    background: rgba(99,102,241,0.08) !important;
}

/* ════════════════════════════════
   CHAT MESSAGES
════════════════════════════════ */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-bottom: 0.5rem !important;
    animation: msgIn .3s cubic-bezier(.22,.68,0,1.2) both;
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0)  scale(1); }
}

/* ── PREMIUM CUSTOM AVATARS ── */
[data-testid="chatAvatarIcon-user"] img,
[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-assistant"] img,
[data-testid="chatAvatarIcon-assistant"] svg {
    display: none !important;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    border-radius: 12px !important;
    overflow: visible !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    flex-shrink: 0 !important;
}

[data-testid="chatAvatarIcon-user"]::before,
[data-testid="chatAvatarIcon-assistant"]::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 12px;
    z-index: 0;
    animation: avatarIn .4s cubic-bezier(.22,.68,0,1.2) both;
}
@keyframes avatarIn {
    from { opacity: 0; transform: scale(0.6) rotate(-10deg); }
    to   { opacity: 1; transform: scale(1) rotate(0deg); }
}

[data-testid="chatAvatarIcon-user"]::after,
[data-testid="chatAvatarIcon-assistant"]::after {
    position: relative;
    z-index: 1;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    line-height: 1;
    letter-spacing: -0.02em;
}

[data-testid="chatAvatarIcon-user"]::before {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 45%, #818cf8 100%);
    box-shadow:
        0 0 0 1.5px rgba(99,102,241,0.5),
        0 0 0 4px rgba(99,102,241,0.12),
        0 6px 20px rgba(99,102,241,0.45),
        inset 0 1px 0 rgba(255,255,255,0.18);
}
[data-testid="chatAvatarIcon-user"]::after {
    content: 'U';
    color: #fff;
    text-shadow: 0 1px 4px rgba(0,0,0,0.25);
}

[data-testid="chatAvatarIcon-assistant"]::before {
    background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 40%, #06b6d4 100%);
    box-shadow:
        0 0 0 1.5px rgba(139,92,246,0.55),
        0 0 0 4px rgba(139,92,246,0.12),
        0 6px 22px rgba(109,40,217,0.5),
        inset 0 1px 0 rgba(255,255,255,0.15);
    animation: avatarIn .4s cubic-bezier(.22,.68,0,1.2) both,
               assistantPulse 3.5s ease-in-out 1s infinite;
}
@keyframes assistantPulse {
    0%,100% { box-shadow: 0 0 0 1.5px rgba(139,92,246,0.55), 0 0 0 4px rgba(139,92,246,0.12), 0 6px 22px rgba(109,40,217,0.5), inset 0 1px 0 rgba(255,255,255,0.15); }
    50%      { box-shadow: 0 0 0 1.5px rgba(139,92,246,0.7),  0 0 0 7px rgba(139,92,246,0.08), 0 8px 30px rgba(109,40,217,0.65), inset 0 1px 0 rgba(255,255,255,0.2); }
}
[data-testid="chatAvatarIcon-assistant"]::after {
    content: '✦';
    color: #fff;
    font-size: 1.1rem;
    text-shadow: 0 0 10px rgba(6,182,212,0.6), 0 1px 4px rgba(0,0,0,0.3);
}

/* ── Message content area ── */
[data-testid="stChatMessageContent"] {
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    backdrop-filter: blur(10px);
    transition: box-shadow .25s;
}

/* User bubble */
.stChatMessage[data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg,
        rgba(79,82,213,0.18) 0%,
        rgba(99,102,241,0.12) 100%) !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    box-shadow: 0 2px 20px rgba(99,102,241,0.10), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* Assistant bubble */
.stChatMessage[data-testid="stChatMessage-assistant"] [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:nth-child(even) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg,
        rgba(139,92,246,0.12) 0%,
        rgba(30,27,75,0.35) 100%) !important;
    border: 1px solid rgba(139,92,246,0.22) !important;
    box-shadow: 0 2px 20px rgba(139,92,246,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
}

[data-testid="stChatMessageContent"]:hover {
    box-shadow: 0 4px 30px rgba(99,102,241,0.18) !important;
}

[data-testid="stChatMessageContent"] p {
    color: var(--text-primary) !important;
    margin: 0 !important;
    font-size: 0.92rem;
    line-height: 1.7;
}
[data-testid="stChatMessageContent"] strong { color: #c7d2fe !important; }
[data-testid="stChatMessageContent"] code {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 5px !important;
    padding: 0.1rem 0.4rem !important;
    font-size: 0.82rem !important;
    color: #a5b4fc !important;
}
[data-testid="stChatMessageContent"] pre {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol {
    padding-left: 1.4rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stChatMessageContent"] li { margin-bottom: 0.2rem; }
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #e0e7ff !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.3rem !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.09) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15), 0 4px 24px rgba(99,102,241,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-size: 0.91rem !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
    transition: all .2s !important; letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.55) !important;
}

.stDownloadButton > button {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    color: #34d399 !important; border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; transition: all .2s !important;
    box-shadow: 0 2px 12px rgba(16,185,129,0.12) !important;
}
.stDownloadButton > button:hover {
    background: rgba(16,185,129,0.2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(16,185,129,0.25) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    backdrop-filter: blur(8px) !important;
}
.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.28) !important;
    border-radius: var(--radius-sm) !important; color: #6ee7b7 !important;
    box-shadow: 0 0 20px rgba(16,185,129,0.06) !important;
}
.stWarning {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.28) !important;
    border-radius: var(--radius-sm) !important; color: #fcd34d !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Charts ── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid rgba(99,102,241,0.15);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.1);
    transition: transform .3s, box-shadow .3s;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.01);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 30px rgba(99,102,241,0.15);
}

/* ── WELCOME CARD ── */
.welcome-card {
    background: linear-gradient(145deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px; padding: 2.8rem 2.2rem; text-align: center;
    margin: 2.5rem auto; max-width: 580px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: cardIn .5s cubic-bezier(.22,.68,0,1) both;
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.welcome-card .icon {
    font-size: 3rem; margin-bottom: 1.1rem;
    filter: drop-shadow(0 0 20px rgba(99,102,241,0.5));
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0);   }
    50%      { transform: translateY(-6px); }
}
.welcome-card h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700; margin: 0 0 0.6rem;
    background: linear-gradient(90deg, #e0e7ff, #a5b4fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.welcome-card p {
    color: var(--text-secondary); font-size: 0.87rem;
    margin: 0 0 1.6rem; line-height: 1.65;
}
.suggestion-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; }
.suggestion-pill {
    background: rgba(99,102,241,0.09);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 99px; padding: 0.38rem 0.9rem;
    font-size: 0.77rem; font-weight: 500;
    color: #a5b4fc; cursor: default;
    transition: all .2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.suggestion-pill:hover {
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.45);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99,102,241,0.2);
}

/* ── TOOL STEP PILLS ── */
.tool-step {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 99px;
    padding: 0.28rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a5b4fc;
    margin: 0.2rem 0;
    letter-spacing: 0.02em;
    transition: all .25s;
}
.tool-step .ring {
    width: 12px; height: 12px;
    border: 2px solid rgba(99,102,241,0.3);
    border-top-color: #6366f1;
    border-radius: 50%;
    display: inline-block;
    animation: spin .7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.tool-step.done {
    background: rgba(16,185,129,0.07);
    border-color: rgba(16,185,129,0.22);
    color: #6ee7b7;
}
.tool-steps-wrap { display: flex; flex-direction: column; gap: 0.15rem; margin-bottom: 0.6rem; }

/* ── STREAMING CURSOR ── */
.stream-cursor {
    display: inline-block;
    width: 2px; height: 1em;
    background: #818cf8;
    margin-left: 1px;
    vertical-align: middle;
    border-radius: 1px;
    animation: blink .9s step-end infinite;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = DataAnalysisAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="dot">✦</div>
        <span>DataSense AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📁 Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        result = load_csv(uploaded_file)
        st.session_state.dataset_loaded = True
        st.success(f"✓ **{uploaded_file.name}** loaded")

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="val">{result['rows']:,}</div>
                <div class="lbl">Rows</div>
            </div>
            <div class="stat-card">
                <div class="val">{result['columns']}</div>
                <div class="lbl">Cols</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">🗂 Columns</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="col-chip">{c}</span>' for c in result["column_names"])
        st.markdown(f'<div class="col-chip-wrap">{chips}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section" style="margin-top:1.2rem">👁 Preview</div>', unsafe_allow_html=True)
        st.dataframe(result["preview"], width='stretch', height=200)
    else:
        st.markdown("""
        <div style="color:#475569;font-size:0.8rem;text-align:center;padding:1.5rem 0.5rem;line-height:1.6;">
            Upload a <strong style="color:#6366f1">CSV file</strong> to unlock<br>
            AI-powered analysis and insights.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">🤖 AI Engines & Fallback</div>', unsafe_allow_html=True)
    groq_on = bool((os.getenv("GROQ_API_KEY") or os.getenv("GROQ") or "").strip())
    or_on = bool((os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER") or os.getenv("OPENROUTER") or "").strip())
    
    groq_badge = '<span style="color:#34d399;font-weight:600;">● Active</span>' if groq_on else '<span style="color:#ef4444;font-weight:500;">○ Missing Key</span>'
    or_badge = '<span style="color:#34d399;font-weight:600;">● Active</span>' if or_on else '<span style="color:#ef4444;font-weight:500;">○ Missing Key</span>'

    st.markdown(f"""
    <div style="font-size:0.75rem;color:#94a3b8;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:0.6rem 0.7rem;margin-bottom:0.8rem;line-height:1.7;">
        <div>⚡ <strong>Groq:</strong> {groq_badge}</div>
        <div>🌐 <strong>OpenRouter:</strong> {or_badge}</div>
        <div style="font-size:0.68rem;color:#64748b;margin-top:0.3rem;">Auto-fallback enabled on failure</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">💡 Try asking</div>', unsafe_allow_html=True)
    for hint in ["Summarize this dataset", "Show column distributions", "Find missing values", "Run a SQL query", "Plot a correlation heatmap", "Export cleaned data"]:
        st.markdown(f"<div style='color:#64748b;font-size:0.78rem;padding:0.22rem 0;'>→ {hint}</div>", unsafe_allow_html=True)


# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">✦</div>
    <div class="hero-text">
        <h1>DataSense AI</h1>
        <p>Intelligent data analysis, powered by AI &nbsp;&bull;&nbsp; Ask anything about your data</p>
    </div>
    <div class="hero-badge">&#9679; Live</div>
</div>
""", unsafe_allow_html=True)

# ─── Welcome / Chat ─────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.dataset_loaded:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">🔮</div>
        <h3>Welcome to DataSense AI</h3>
        <p>Upload a CSV dataset from the sidebar to get started.<br>
        Then ask me anything — I will analyze, visualize, and explain your data.</p>
        <div class="suggestion-grid">
            <span class="suggestion-pill">📊 Summarize</span>
            <span class="suggestion-pill">⚡ SQL Query</span>
            <span class="suggestion-pill">📈 Visualize</span>
            <span class="suggestion-pill">🔍 Find outliers</span>
            <span class="suggestion-pill">🧹 Clean data</span>
            <span class="suggestion-pill">📥 Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages and st.session_state.dataset_loaded:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">✅</div>
        <h3>Dataset ready — ask me anything!</h3>
        <p>Your data is loaded. Try asking for a summary, a chart,<br>or any custom analysis you have in mind.</p>
    </div>
    """, unsafe_allow_html=True)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])

        if message.get("table") is not None:
            df_hist = message["table"]
            rows_hist = message.get("table_rows", len(df_hist))
            query_hist = message.get("query", "")
            st.markdown(f"📊 **Query Results** ({rows_hist:,} rows):")
            if query_hist:
                with st.expander("🔍 View Executed SQL Query", expanded=False):
                    st.code(query_hist, language="sql")
            st.dataframe(df_hist, width='stretch')
            st.download_button(
                label="⬇️  Download Query Results (CSV)",
                data=df_hist.to_csv(index=False).encode("utf-8"),
                file_name="query_results.csv",
                mime="text/csv",
                key=f"hist_table_{idx}"
            )

        if message.get("chart"):
            st.image(message["chart"], width='stretch')

        if message.get("csv"):
            filepath = message["csv"]
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.download_button(
                        label=f"⬇️  Download {os.path.basename(filepath)}",
                        data=f.read(),
                        file_name=os.path.basename(filepath),
                        mime="text/csv",
                        key=f"hist_csv_{idx}"
                    )


# ─── Chat Input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything about your dataset…")

if prompt:

    if not st.session_state.dataset_loaded:
        st.warning("⚠️  Please upload a CSV file from the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        steps_ph = st.empty()   # tool pills
        text_ph  = st.empty()   # streaming text
        table_ph = st.empty()   # query results table
        chart_ph = st.empty()   # chart image
        csv_ph   = st.empty()   # csv download button

        tool_steps = []
        streamed_text = ""
        last_table_obj = None
        last_table_rows = 0
        last_chart_file = None
        last_csv_file = None

        def render_steps():
            if not tool_steps:
                steps_ph.empty()
                return
            pills = ""
            for s in tool_steps:
                icon   = '✓' if s["done"] else '<span class="ring"></span>'
                status = "done" if s["done"] else ""
                label  = s["name"].replace("_", " ").title()
                pills += f'<div class="tool-step {status}">{icon} {label}</div>\n'
            steps_ph.markdown(
                f'<div class="tool-steps-wrap">{pills}</div>',
                unsafe_allow_html=True
            )

        for event in st.session_state.agent.run_agent_stream(prompt, st.session_state.messages):

            etype = event["type"]

            if etype == "tool_start":
                tool_steps.append({"name": event["name"], "done": False})
                render_steps()

            elif etype == "tool_done":
                for s in tool_steps:
                    if s["name"] == event["name"] and not s["done"]:
                        s["done"] = True
                        break
                render_steps()

            elif etype == "fallback":
                st.toast(event.get("content", "Switching to fallback model..."), icon="🔄")

            elif etype == "table":
                steps_ph.empty()
                last_table_obj = event["content"]
                last_table_rows = event.get("row_count", len(last_table_obj))
                last_query_sql = event.get("query", "")
                with table_ph.container():
                    st.markdown(f"📊 **Query Results** ({last_table_rows:,} rows):")
                    if last_query_sql:
                        with st.expander("🔍 View Executed SQL Query", expanded=False):
                            st.code(last_query_sql, language="sql")
                    st.dataframe(last_table_obj, width='stretch')
                    st.download_button(
                        label="⬇️  Download Query Results (CSV)",
                        data=last_table_obj.to_csv(index=False).encode("utf-8"),
                        file_name="query_results.csv",
                        mime="text/csv",
                        key=f"dl_table_live_{time.time()}"
                    )

            elif etype == "chart":
                steps_ph.empty()
                last_chart_file = event["content"]
                chart_ph.image(last_chart_file, width='stretch')

            elif etype == "csv":
                steps_ph.empty()
                last_csv_file = event["content"]
                with open(last_csv_file, "rb") as f:
                    csv_bytes = f.read()
                filename = os.path.basename(last_csv_file)
                with csv_ph.container():
                    st.success(f"✓ Dataset exported as **{filename}**")
                    st.download_button(
                        label="⬇️  Download CSV",
                        data=csv_bytes,
                        file_name=filename,
                        mime="text/csv",
                        key=f"dl_csv_live_{time.time()}"
                    )

            elif etype == "text":
                if tool_steps:
                    steps_ph.empty()
                    tool_steps = []
                streamed_text += event["content"]
                text_ph.markdown(
                    streamed_text + '<span class="stream-cursor"></span>',
                    unsafe_allow_html=True
                )

        if streamed_text:
            text_ph.markdown(streamed_text)

        message_record = {
            "role": "assistant",
            "content": streamed_text or "Analysis complete."
        }
        if last_table_obj is not None:
            message_record["table"] = last_table_obj
            message_record["table_rows"] = last_table_rows
            message_record["query"] = last_query_sql
        if last_chart_file:
            message_record["chart"] = last_chart_file
        if last_csv_file:
            message_record["csv"] = last_csv_file

        st.session_state.messages.append(message_record)
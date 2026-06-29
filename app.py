
import streamlit as st
from agent import DataAnalysisAgent
from tools import load_csv

st.set_page_config(
    page_title="DataSense",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DataSense")
st.markdown("### AI Powered Data Analysis Agent")

# ---------------- Session State ---------------- #

if "agent" not in st.session_state:
    st.session_state.agent = DataAnalysisAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False


# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        result = load_csv(uploaded_file)

        st.session_state.dataset_loaded = True

        st.success(f"{uploaded_file.name} uploaded successfully!")

        st.write("### Dataset Information")
        st.write(f"Rows : {result['rows']}")
        st.write(f"Columns : {result['columns']}")

        st.write("### Column Names")
        st.write(result["column_names"])

        st.write("### Preview")
        st.dataframe(result["preview"])


# ---------------- Chat History ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- Chat Input ---------------- #

prompt = st.chat_input("Ask anything about your dataset...")


if prompt:

    if not st.session_state.dataset_loaded:

        st.warning("Please upload a CSV file first.")

        st.stop()

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = st.session_state.agent.run_agent(
                prompt,
                st.session_state.messages
            )

        if response["type"] == "stream":
            full_text = st.write_stream(response["content"])
            display_content = full_text or ""

        elif response["type"] == "text":
            content = response["content"] or ""
            st.markdown(content if content else "*(No response)*")
            display_content = content

        elif response["type"] == "chart":
            st.image(response["content"], use_container_width=True)
            display_content = f"[Chart saved: {response['content']}]"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": display_content
        }
    )
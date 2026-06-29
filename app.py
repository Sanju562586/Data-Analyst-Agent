import streamlit as st
from tools import load_csv, analyze_data, run_pandas_code, plot_chart


st.set_page_config(
    page_title="DataSense",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DataSense")
st.markdown("### AI Powered Data Analysis Agent")

with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV File",
        type=["csv"]
    )

if uploaded_file:
        result = load_csv(uploaded_file)
        st.success(f"Uploaded: {uploaded_file.name}")

        st.write("### Dataset Information")
        st.write(f"Rows : {result['rows']}")
        st.write(f"Columns : {result['columns']}")

        st.write("### Columns")
        st.write(result["column_names"])

        st.write("### Preview")
        st.dataframe(result["preview"])

        if st.button("Analyze Data"):
            summary = analyze_data()
            st.write("### Analysis Summary")
            st.code(summary)

        result = run_pandas_code("""print(df.head())""")
        st.code(result)

        chart_filename = plot_chart(
            chart_type="bar",
            x="Age",
            y='Annual Income (k$)',
            title="Annual Income by Age",
            xlabel="Age",
            ylabel = "Annual Income"
        )
        st.image(chart_filename)
    

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input(
    "Ask anything about your dataset..."
)

if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = "I received your question."

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)
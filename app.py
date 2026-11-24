import os
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from ingestion import parse_uploaded_file
from rag_engine import HybridRAGStore
from analyst_agent import run_analyst_query

load_dotenv()

st.set_page_config(
    page_title="Analyze IT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap](https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap)');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #18181B;
    }

    h1, h2, h3, .editorial-title {
        font-family: 'Newsreader', Georgia, serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: #141413;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .masthead {
        border-bottom: 1.5px solid #18181B;
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }

    .masthead-sub {
        font-size: 0.82rem;
        color: #57534E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    .paper-card {
        background-color: #FFFFFF;
        border: 1px solid #DCD7CD;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(24, 24, 27, 0.04);
    }

    .source-pill {
        display: inline-block;
        background-color: #EFECE6;
        border: 1px solid #D6D1C7;
        color: #3F3F46;
        font-size: 0.74rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    section[data-testid="stSidebar"] {
        background-color: #EFECE6;
        border-right: 1px solid #DCD7CD;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "rag_store" not in st.session_state:
    st.session_state.rag_store = HybridRAGStore()
if "parsed_files" not in st.session_state:
    st.session_state.parsed_files = {}
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_fig" not in st.session_state:
    st.session_state.active_fig = None
if "active_chart_desc" not in st.session_state:
    st.session_state.active_chart_desc = ""

with st.sidebar:
    st.markdown("### Workspace Sources")
    st.caption("Upload CSV, Excel, JSON, Parquet, PDF, DOCX, TXT, or Markdown.")

    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        default_key = os.getenv("GROQ_API_KEY", "")

    if default_key:
        groq_key = default_key
        st.caption("✓ Groq API connected via server environment")
    else:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free API key from console.groq.com",
        )

    uploaded_files = st.file_uploader(
        "Drop datasets & reference docs",
        accept_multiple_files=True,
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "pdf", "docx", "txt", "md", "log"],
    )

    if uploaded_files:
        new_files_added = False
        for f in uploaded_files:
            if f.name not in st.session_state.parsed_files:
                with st.spinner(f"Indexing {f.name}..."):
                    try:
                        parsed = parse_uploaded_file(f)
                        st.session_state.parsed_files[f.name] = parsed
                        if parsed["is_tabular"] and parsed["df"] is not None:
                            st.session_state.dfs[f.name] = parsed["df"]
                        new_files_added = True
                    except Exception as err:
                        st.error(f"Failed to parse {f.name}: {err}")

        if new_files_added:
            all_docs = []
            for p in st.session_state.parsed_files.values():
                all_docs.extend(p["text_blocks"])
            st.session_state.rag_store.build_index(all_docs)

    if st.session_state.parsed_files:
        st.divider()
        st.markdown("**Indexed Files**")
        for fname, meta in st.session_state.parsed_files.items():
            st.markdown(
                f"📄 **{fname}**  \n<small style='color:#57534E'>{meta['summary']}</small>",
                unsafe_allow_html=True,
            )

        if st.button("Clear Workspace", use_container_width=True):
            st.session_state.parsed_files = {}
            st.session_state.dfs = {}
            st.session_state.messages = []
            st.session_state.active_fig = None
            st.session_state.active_chart_desc = ""
            st.session_state.rag_store = HybridRAGStore()
            st.rerun()

st.markdown(
    """
    <div class="masthead">
        <div>
            <span class="editorial-title" style="font-size: 1.85rem;">Ledger &amp; Lens</span>
            <span style="margin-left: 12px; color: #57534E; font-size: 0.95rem;">RAG-Grounded Data &amp; Document Workbench</span>
        </div>
        <div class="masthead-sub">Engine: Groq OSS-120B · Local FAISS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.parsed_files:
    st.markdown(
        """
        <div class="paper-card">
            <h3 style="margin-top:0;">Ready for your files</h3>
            <p style="color:#3F3F46; margin-bottom:0.5rem;">
                Upload one or more files in the sidebar to begin. You can combine structured tables
                (<b>CSV, Excel, Parquet, JSON</b>) with unstructured context (<b>PDF reports, DOCX memos, TXT notes</b>)
                in a single RAG index.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


def handle_question(prompt_text: str):
    if not groq_key:
        st.error("Enter your free Groq API key in the sidebar first.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    retrieved = st.session_state.rag_store.retrieve(prompt_text, top_k=6)

    with st.spinner("Analyzing data & retrieved context..."):
        response = run_analyst_query(
            api_key=groq_key,
            user_query=prompt_text,
            rag_chunks=retrieved,
            dfs=st.session_state.dfs,
            active_chart_context=st.session_state.active_chart_desc,
            chat_history=st.session_state.messages[:-1],
        )

    if response["fig"] is not None:
        st.session_state.active_fig = response["fig"]
        st.session_state.active_chart_desc = (
            f"Chart generated for query: '{prompt_text}'. Summary: {response['exec_output'][:400]}"
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "code": response["code"],
        "sources": response["sources"],
    })


viz_col, chat_col = st.columns([1.35, 1], gap="large")

with viz_col:
    st.markdown("### Visualization & Data Canvas")

    if st.session_state.dfs:
        with st.expander(
            "Manual Chart Builder (or ask the AI to plot any chart)",
            expanded=st.session_state.active_fig is None,
        ):
            selected_df_name = st.selectbox("Dataset", list(st.session_state.dfs.keys()))
            df_curr = st.session_state.dfs[selected_df_name]
            c1, c2, c3 = st.columns(3)
            chart_type = c1.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram", "Box"])
            x_col = c2.selectbox("X-Axis", df_curr.columns)
            y_opts = ["(Count)"] + list(df_curr.columns)
            y_col = c3.selectbox("Y-Axis", y_opts)

            if st.button("Render Chart"):
                palette = ["#C84B31", "#2D6A4F", "#D97706", "#1E3A5F", "#6B7280"]
                if chart_type == "Histogram" or y_col == "(Count)":
                    fig = px.histogram(
                        df_curr, x=x_col, color_discrete_sequence=palette, template="plotly_white"
                    )
                elif chart_type == "Bar":
                    fig = px.bar(
                        df_curr, x=x_col, y=y_col, color_discrete_sequence=palette, template="plotly_white"
                    )
                elif chart_type == "Line":
                    fig = px.line(
                        df_curr, x=x_col, y=y_col, color_discrete_sequence=palette, template="plotly_white"
                    )
                elif chart_type == "Scatter":
                    fig = px.scatter(
                        df_curr, x=x_col, y=y_col, color_discrete_sequence=palette, template="plotly_white"
                    )
                else:
                    fig = px.box(
                        df_curr, x=x_col, y=y_col, color_discrete_sequence=palette, template="plotly_white"
                    )

                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Inter, sans-serif", color="#18181B"),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.session_state.active_fig = fig
                st.session_state.active_chart_desc = (
                    f"{chart_type} chart of '{selected_df_name}' showing X='{x_col}' vs Y='{y_col}'."
                )

    if st.session_state.active_fig is not None:
        st.markdown('<div class="paper-card">', unsafe_allow_html=True)
        st.plotly_chart(st.session_state.active_fig, use_container_width=True)

        with st.form("chart_qa_form", clear_on_submit=True):
            q_col, b_col = st.columns([4, 1])
            chart_question = q_col.text_input(
                "Ask about this visualization",
                placeholder="e.g., Why does Q3 spike here, or what is the correlation?",
                label_visibility="collapsed",
            )
            submitted = b_col.form_submit_button("Ask Chart", use_container_width=True)
            if submitted and chart_question.strip():
                handle_question(f"[Regarding the current chart] {chart_question.strip()}")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.dfs:
        st.markdown("#### Raw Table Inspector")
        tabs = st.tabs(list(st.session_state.dfs.keys()))
        for tab, (name, df_item) in zip(tabs, st.session_state.dfs.items()):
            with tab:
                st.dataframe(df_item.head(100), use_container_width=True, height=260)

with chat_col:
    st.markdown("### Analyst Notebook (RAG)")

    chat_container = st.container(height=540)
    with chat_container:
        if not st.session_state.messages:
            st.caption("Ask any question across your uploaded datasets and documents, or request a chart.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("code"):
                    with st.expander("View Executed Python"):
                        st.code(msg["code"], language="python")
                if msg.get("sources"):
                    pills = "".join([f'<span class="source-pill">{s}</span>' for s in msg["sources"]])
                    st.markdown(pills, unsafe_allow_html=True)

    notebook_prompt = st.chat_input("Ask a question or request a visualization...")
    if notebook_prompt:
        handle_question(notebook_prompt)
        st.rerun()
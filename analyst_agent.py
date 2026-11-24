import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq


MODEL_ID = "openai/gpt-oss-120b"


def run_analyst_query(
    api_key: str,
    user_query: str,
    rag_chunks: list[dict],
    dfs: dict[str, pd.DataFrame],
    active_chart_context: str = "",
    chat_history: list[dict] = None,
) -> dict:
    client = Groq(api_key=api_key)

    rag_context_str = "\n\n---\n\n".join(
        [f"[{c['source']} | relevance={c['score']:.2f}]\n{c['content']}" for c in rag_chunks]
    )

    df_schemas = []
    for name, df in dfs.items():
        cols_info = ", ".join([f"{col} ({dtype})" for col, dtype in df.dtypes.items()])
        df_schemas.append(f"dfs['{name}'] -> shape={df.shape}, columns=[{cols_info}]")
    df_schema_str = "\n".join(df_schemas) if df_schemas else "No tabular DataFrames loaded."

    system_prompt = f"""You are a senior quantitative data analyst. You answer questions using retrieved RAG evidence AND, when tabular data is available, executable Python (pandas + plotly.express).

AVAILABLE DATAFRAMES IN PYTHON ENVIRONMENT (`dfs` dictionary):
{df_schema_str}

CURRENT ACTIVE VISUALIZATION CONTEXT (what the user is looking at right now):
{active_chart_context or "None currently displayed."}

RETRIEVED RAG CONTEXT:
{rag_context_str}

INSTRUCTIONS:
1. Ground every answer strictly in the retrieved RAG context and DataFrame facts. Cite source names naturally.
2. Whenever the user asks for a chart, visualization, comparison, trend, or exact numerical calculation on a DataFrame, write clean Python code inside a single ```python block.
3. In your Python code:
   - Access DataFrames via `dfs['exact_filename']`.
   - Store any computed metric, summary, or table in a variable named `result_summary`.
   - If creating a visualization, assign a Plotly figure to a variable named `fig`.
   - Style Plotly charts cleanly with warm editorial colors: `color_discrete_sequence=["#C84B31", "#2D6A4F", "#D97706", "#1E3A5F", "#6B7280"]` and `template="plotly_white"`.
   - Update layout with `fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#18181B"))`.
   - Do NOT call `fig.show()`.
4. Keep prose sharp, analytical, and free of filler."""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.15,
        max_tokens=2048,
    )

    raw_reply = completion.choices[0].message.content
    code_match = re.search(r"```python\n(.*?)```", raw_reply, re.DOTALL)

    fig = None
    executed_code = None
    exec_output = ""

    if code_match and dfs:
        executed_code = code_match.group(1).strip()
        local_env = {"dfs": dfs, "pd": pd, "np": np, "px": px, "go": go}
        try:
            exec(executed_code, {}, local_env)
            fig = local_env.get("fig", None)
            if "result_summary" in local_env:
                exec_output = str(local_env["result_summary"])
        except Exception as e:
            exec_output = f"Code execution note: {e}"

    clean_answer = re.sub(r"```python\n.*?```", "", raw_reply, flags=re.DOTALL).strip()
    if exec_output and not clean_answer:
        clean_answer = exec_output

    sources = list(dict.fromkeys([c["source"] for c in rag_chunks]))

    return {
        "answer": clean_answer,
        "fig": fig,
        "code": executed_code,
        "exec_output": exec_output,
        "sources": sources,
    }
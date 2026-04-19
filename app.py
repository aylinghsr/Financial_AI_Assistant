import streamlit as st
import pandas as pd
import altair as alt
import re

from src.loader import load_all_documents
from src.chunker import split_into_chunks
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import HybridRetriever
from src.generator import Generator
from src.data_qa import DataQA
from src.router import QueryRouter

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="wide"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_number(x):
    if pd.isna(x):
        return x
    if isinstance(x, (int, float)):
        abs_x = abs(x)
        if abs_x >= 1_000_000_000:
            return f"{x / 1_000_000_000:,.2f}B"
        if abs_x >= 1_000_000:
            return f"{x / 1_000_000:,.2f}M"
        return f"{x:,.2f}"
    return x


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["float", "int"]).columns:
        df_copy[col] = df_copy[col].map(format_number)
    return df_copy


def infer_chart_type(df: pd.DataFrame):
    cols = df.columns.tolist()
    if len(df) == 0:
        return None
    if "period" in cols and any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        return "line"
    if len(cols) >= 2:
        if not pd.api.types.is_numeric_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]):
            return "bar"
    return None


def render_kpis(df: pd.DataFrame):
    if df is None or df.empty:
        return
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(df) == 1 and len(numeric_cols) == 1:
        col = numeric_cols[0]
        st.metric(label=col.replace("_", " ").title(), value=format_number(df.iloc[0][col]))
    elif "entity" in df.columns and numeric_cols:
        metric_col = numeric_cols[0]
        cols = st.columns(min(len(df), 4))
        for i, (_, row) in enumerate(df.head(4).iterrows()):
            with cols[i]:
                st.metric(label=str(row["entity"]), value=format_number(row[metric_col]))


def render_chart(df: pd.DataFrame):
    if df is None or df.empty:
        return
    chart_type = infer_chart_type(df)
    if chart_type is None:
        return

    st.markdown("### 📈 Visual")

    if chart_type == "line":
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return
        y_col = numeric_cols[0]
        color_col = "entity" if "entity" in df.columns else None
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("period:N", title="Period", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title(),
                    axis=alt.Axis(format="~s")),
            color=color_col if color_col else alt.value("#4C78A8"),
            tooltip=list(df.columns)
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)

    elif chart_type == "bar":
        x_col = df.columns[0]
        y_col = df.columns[1]
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{x_col}:N", sort="-y", title=x_col.replace("_", " ").title()),
            y=alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title(),
                    axis=alt.Axis(format="~s")),
            tooltip=list(df.columns)
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)


def render_document_sources(sources):
    st.markdown("### 📚 Sources")
    for i, source in enumerate(sources, start=1):
        with st.container():
            st.markdown(
                f"**Source {i}**  \n"
                f"File: `{source['source']}`  \n"
                f"Page: **{source['page']}**  \n"
                f"Score: **{source['score']}**"
            )
            st.divider()


def parse_document_answer(answer: str) -> dict:
    parsed = {
        "direct_answer": "",
        "explanation": "",
        "insight": "",
    }

    # split by emoji markers wherever they appear — works even on one line
    direct      = re.search(r'📊[^📝🧠]*', answer)
    explanation = re.search(r'📝[^📊🧠]*', answer)
    insight     = re.search(r'🧠[^📊📝]*', answer)

    if direct:
        parsed["direct_answer"] = re.sub(r'📊\s*(Direct Answer)?', '', direct.group()).strip()
    if explanation:
        parsed["explanation"] = re.sub(r'📝\s*(Explanation)?', '', explanation.group()).strip()
    if insight:
        parsed["insight"] = re.sub(r'🧠\s*(Insight)?', '', insight.group()).strip()

    # fallback — if nothing found, show full answer in direct_answer
    if not any(parsed.values()):
        parsed["direct_answer"] = answer.strip()

    return parsed


def parse_data_answer(answer: str) -> dict:
    parsed = {"summary": answer, "insight": ""}
    for line in answer.splitlines():
        line = line.strip()
        if line.lower().startswith("summary:"):
            parsed["summary"] = line.replace("Summary:", "").strip()
        elif line.lower().startswith("insight:"):
            parsed["insight"] = line.replace("Insight:", "").strip()
    return parsed


# ── Pipeline loaders ──────────────────────────────────────────────────────────

@st.cache_resource
def load_doc_pipeline():
    docs = load_all_documents("documents")
    chunks = split_into_chunks(docs)
    embedder = Embedder()
    indexer = Indexer()
    retriever = HybridRetriever(chunks, embedder, indexer)
    generator = Generator()
    return docs, retriever, generator


@st.cache_resource
def load_data_pipeline():
    return DataQA()


@st.cache_resource
def load_router():
    return QueryRouter()


docs, retriever, generator = load_doc_pipeline()
data_qa = load_data_pipeline()
router = load_router()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Financial AI")
    st.caption("Hybrid RAG + SQL assistant")

    show_debug  = st.toggle("Show routing details",  value=True)
    show_sql    = st.toggle("Show generated SQL",    value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.divider()
    st.markdown("### System")
    st.markdown("- Hybrid RAG + SQL")
    st.markdown("- Model: Mistral 7B")
    st.markdown("- Vector DB: Qdrant")
    st.markdown("- Embeddings: all-MiniLM-L6-v2")

    st.divider()
    st.markdown("### Example questions")
    with st.expander("📄 Document questions"):
        st.markdown("- What is the minimum CET1 capital ratio?")
        st.markdown("- What is the Liquidity Coverage Ratio?")
        st.markdown("- Why was Basel III introduced?")
        st.markdown("- How does Basel III define the leverage ratio?")

    with st.expander("🗄️ Data questions"):
        st.markdown("- What is the total amount by entity in gl_balances?")
        st.markdown("- How many transactions are there for Solaris SE?")
        st.markdown("- Show monthly balances for gl_number 1300")
        st.markdown("- List all income accounts")

    st.divider()
    st.markdown("### Loaded documents")
    sources = sorted(set(d.metadata["source"] for d in docs))
    for s in sources:
        st.markdown(f"- `{s}`")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Financial AI Assistant")
st.caption("Ask a question about financial regulations or structured financial data.")

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the minimum CET1 capital ratio?"
)

if query:
    route = router.route(query)

    if show_debug:
        st.info(f"Routed to: {route}")

    with st.spinner("Processing your question..."):

        if route == "DATA":
            result = data_qa.ask(query)
            parsed = parse_data_answer(result["answer"])

            st.markdown("## 📊 Summary")
            st.success(parsed["summary"])

            if parsed["insight"]:
                st.markdown("### 🧠 Insight")
                st.info(parsed["insight"])

            if result["rows"] is not None:
                render_kpis(result["rows"])
                render_chart(result["rows"])
                st.markdown("### 📋 Result Table")
                st.dataframe(
                    format_dataframe_for_display(result["rows"]),
                    use_container_width=True
                )

            if show_sql:
                with st.expander("🧾 Generated SQL"):
                    st.code(result["sql"], language="sql")

        else:
            results  = retriever.retrieve(query, top_k=5)
            response = generator.generate(query, results)
            parsed   = parse_document_answer(response["answer"])

            st.markdown("## 📘 Direct Answer")
            st.success(f"**{parsed['direct_answer']}**")

            if parsed["explanation"]:
                st.markdown("### 📝 Explanation")
                st.write(parsed["explanation"])

            if parsed["insight"]:
                st.markdown("### 🧠 Insight")
                st.info(parsed["insight"])

            render_document_sources(response["sources"])

            if show_chunks:
                with st.expander("🔍 Retrieved Chunks"):
                    for i, r in enumerate(results, start=1):
                        st.markdown(
                            f"**Chunk {i}** — `{r['metadata']['source']}` | "
                            f"Page **{r['metadata']['page']}** | "
                            f"Score **{r['score']:.3f}**"
                        )
                        st.text(r["content"][:1000])
                        st.divider()

else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📄 Ask about documents")
        st.markdown("- Basel III definitions\n- Capital requirements\n- Leverage ratio\n- Liquidity coverage ratio")
    with c2:
        st.markdown("### 🗄️ Ask about data")
        st.markdown("- Totals and averages\n- Account balances\n- Transaction counts\n- Top accounts")

    st.divider()
    st.markdown("### Suggested prompts")
    for p in [
        "What is the minimum Common Equity Tier 1 capital ratio?",
        "What is the Liquidity Coverage Ratio?",
        "What is the total amount by entity in gl_balances?",
        "How many transactions are there for Solaris SE?",
        "Show monthly balances for gl_number 1300",
        "List all income accounts"
    ]:
        st.markdown(f"- {p}")
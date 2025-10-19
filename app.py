# app.py
# IR Demo — Extended Boolean vs Fuzzy (D1–D15)
# Fully computed from corpus (no mocks). Supports duplicates in documents.
# Pages:
#   1) Ranking — Two horizontal bar charts (Extended Boolean / Fuzzy) + Top-k
#   2) Extended Boolean — Step 1–3 details with TF, IDF, weights, and ranking
#   3) Fuzzy — Step 1–3 details with frequency-aware membership, correlation, completion

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# 0) CORPUS — tokens (supports duplicates)
# -----------------------------------------------------------------------------
DOCS: Dict[str, List[str]] = {
    "D1":  ["bird", "cat", "bird", "cat", "dog", "dog", "bird"],
    "D2":  ["cat", "dog", "tiger"],
    "D3":  ["bird", "dog", "bird"],
    "D4":  ["cat", "tiger"],
    "D5":  ["tiger", "cat", "dog", "tiger", "dog"],
    "D6":  ["cat", "tiger", "cat"],
    "D7":  ["bird", "cat", "dog", "bird"],
    "D8":  ["bird", "cat", "dog", "dog"],
    "D9":  ["cat", "dog", "tiger"],
    "D10": ["tiger", "cat", "tiger"],
    "D11": ["bird", "cat", "dog", "tiger", "bird"],
    "D12": ["cat", "dog", "tiger", "tiger"],
    "D13": ["bird", "bird", "bird", "cat"],
    "D14": ["dog", "dog", "tiger", "cat"],
    "D15": ["tiger", "cat", "bird", "tiger"],
}
TERMS: List[str] = ["bird", "cat", "dog", "tiger"]


# -----------------------------------------------------------------------------
# 1) Utilities — counts, TF/IDF, memberships, correlations, parser
# -----------------------------------------------------------------------------
def term_count_table(docs: Dict[str, List[str]], terms: List[str]) -> pd.DataFrame:
    """Return per-document counts for given terms + Max column."""
    rows = []
    for d, toks in docs.items():
        counts = {t: toks.count(t) for t in terms}
        counts["Max"] = max(counts.values()) if counts else 0
        rows.append(pd.Series(counts, name=d))
    df = pd.DataFrame(rows)
    df.index.name = "Doc"
    return df.loc[sorted(df.index)]


def binary_membership(docs: Dict[str, List[str]], terms: List[str]) -> pd.DataFrame:
    """Binary membership μ∈{0,1} indicating presence/absence of each term per doc."""
    mu = {d: [1 if t in toks else 0 for t in terms] for d, toks in docs.items()}
    df = pd.DataFrame(mu, index=terms).T
    df.index.name = "Doc"
    df.columns = terms
    return df.loc[sorted(df.index)]


def tf_normalized(counts: pd.DataFrame) -> pd.DataFrame:
    """TF normalized by the per-document max count: TF = count / Max."""
    tf = counts.copy()
    for t in [c for c in tf.columns if c != "Max"]:
        tf[t] = tf.apply(lambda r: (r[t] / r["Max"]) if r["Max"] else 0.0, axis=1)
    return tf.drop(columns=["Max"])


def idf_norm_from_df(mu_bin: pd.DataFrame) -> pd.Series:
    """IDF = log10(N/df) then normalized by the maximum IDF."""
    N = mu_bin.shape[0]
    dfreq = mu_bin.sum(axis=0).astype(float)
    idf = pd.Series(
        {t: math.log10(N / dfreq[t]) if dfreq[t] > 0 else 0.0 for t in mu_bin.columns}
    )
    max_idf = idf.max() if idf.max() > 0 else 1.0
    return idf / max_idf


def weighted_jaccard_correlation(mu_freq: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted Jaccard correlation across documents:
        c(i,j) = sum_d min(mu_i(d), mu_j(d)) / sum_d max(mu_i(d), mu_j(d))
    Diagonal is set to 1.
    """
    terms = list(mu_freq.columns)
    M = mu_freq.to_numpy(dtype=float)  # D x T
    C = np.zeros((len(terms), len(terms)))
    for i in range(len(terms)):
        for j in range(len(terms)):
            num = np.minimum(M[:, i], M[:, j]).sum()
            den = np.maximum(M[:, i], M[:, j]).sum()
            C[i, j] = (num / den) if den > 0 else 0.0
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=terms, columns=terms)


# ------------------------ Query parser (AND/OR/NOT) --------------------------
class Lexer:
    def __init__(self, s: str):
        s = s.replace("(", " ( ").replace(")", " ) ")
        self.tokens = [tok for tok in s.strip().split() if tok]
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos].lower() if self.pos < len(self.tokens) else None

    def pop(self) -> str | None:
        tok = self.peek()
        self.pos += 1
        return tok


def parse_query(s: str):
    """
    Grammar:
      expr   := term {(OR) term}*
      term   := factor {(AND) factor}*
      factor := NOT factor | '(' expr ')' | SYMBOL
    Accepts: and/&/&&, or/|/||, not/!
    """
    lex = Lexer(s)

    def parse_expr():
        node = parse_term()
        while True:
            tok = lex.peek()
            if tok in ("or", "|", "||"):
                lex.pop()
                right = parse_term()
                node = ("OR", node, right)
            else:
                break
        return node

    def parse_term():
        node = parse_factor()
        while True:
            tok = lex.peek()
            if tok in ("and", "&", "&&"):
                lex.pop()
                right = parse_factor()
                node = ("AND", node, right)
            else:
                break
        return node

    def parse_factor():
        tok = lex.peek()
        if tok is None:
            return None
        if tok in ("not", "!"):
            lex.pop()
            child = parse_factor()
            return ("NOT", child)
        if tok == "(":
            lex.pop()
            node = parse_expr()
            if lex.peek() != ")":
                raise ValueError("Missing ')'")
            lex.pop()
            return node
        # symbol
        sym = lex.pop().lower()
        if sym not in TERMS:
            # basic plural mapping: cats -> cat, dogs -> dog
            if sym.endswith("s") and sym[:-1] in TERMS:
                sym = sym[:-1]
            else:
                raise ValueError(f"Unknown term: {sym}")
        return ("TERM", sym)

    ast = parse_expr()
    if lex.peek() is not None:
        raise ValueError("Extra tokens in query")
    return ast


# ------------------------ Evaluators (EB / Fuzzy) ----------------------------
def and_demorgan(x: float, y: float, p: float) -> float:
    return 1.0 - (((1.0 - x) ** p + (1.0 - y) ** p) / 2.0) ** (1.0 / p)


def and_pnorm(x: float, y: float, p: float) -> float:
    return ((x**p + y**p) / 2.0) ** (1.0 / p)


def eval_ast_ext(ast, row: pd.Series, p: float = 2.0, and_mode: str = "demorgan") -> float:
    """Evaluate Extended Boolean AST on a row of weights."""
    op = ast[0]
    if op == "TERM":
        return float(row[ast[1]])
    if op == "NOT":
        return 1.0 - eval_ast_ext(ast[1], row, p, and_mode)
    if op in ("AND", "OR"):
        left = eval_ast_ext(ast[1], row, p, and_mode)
        right = eval_ast_ext(ast[2], row, p, and_mode)
        AND = and_demorgan if and_mode == "demorgan" else and_pnorm
        if op == "AND":
            return AND(left, right, p)
        return 1.0 - AND(1.0 - left, 1.0 - right, p)
    raise ValueError("Invalid AST")


def eval_ast_fuzzy(ast, row: pd.Series) -> float:
    """Evaluate Fuzzy AST on a row of memberships."""
    op = ast[0]
    if op == "TERM":
        return float(row[ast[1]])
    if op == "NOT":
        return 1.0 - eval_ast_fuzzy(ast[1], row)
    if op == "AND":
        return min(eval_ast_fuzzy(ast[1], row), eval_ast_fuzzy(ast[2], row))
    if op == "OR":
        return max(eval_ast_fuzzy(ast[1], row), eval_ast_fuzzy(ast[2], row))
    raise ValueError("Invalid AST")


# -----------------------------------------------------------------------------
# 2) Extended Boolean pipeline
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extended_boolean_pipeline(
    docs: Dict[str, List[str]],
    terms: List[str],
    p: float = 2.0,
    and_mode: str = "demorgan",
):
    counts = term_count_table(docs, terms)
    tf = tf_normalized(counts)
    mu_bin = binary_membership(docs, terms)
    idf_norm = idf_norm_from_df(mu_bin)
    weights = tf.mul(idf_norm, axis=1)
    return {
        "counts": counts,
        "tf": tf,
        "idf_norm": idf_norm.rename("idf_norm"),
        "weights": weights,
        "p": p,
        "and_mode": and_mode,
    }


def extended_boolean_similarity_from_ast(
    ast, weights: pd.DataFrame, p: float = 2.0, and_mode: str = "demorgan"
) -> pd.Series:
    return weights.apply(lambda r: eval_ast_ext(ast, r, p=p, and_mode=and_mode), axis=1).rename(
        "Extended Boolean"
    )


# -----------------------------------------------------------------------------
# 3) Fuzzy pipeline (frequency-aware + weighted Jaccard)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fuzzy_pipeline(docs: Dict[str, List[str]], terms: List[str]):
    counts = term_count_table(docs, terms)
    mu_raw = tf_normalized(counts)                 # μ(d,t) ∈ [0,1] from frequency
    C = weighted_jaccard_correlation(mu_raw)       # correlation across docs

    # completion by correlation (only where μ==0)
    mu_completed = mu_raw.copy()
    for d in mu_raw.index:
        row = mu_raw.loc[d]
        present = [t for t in terms if row[t] > 0]
        for t in terms:
            if row[t] == 0:
                mu_completed.loc[d, t] = max((row[x] * C.loc[t, x] for x in present), default=0.0)

    return {
        "counts": counts,
        "mu_raw": mu_raw,
        "correlation": C,
        "mu_completed": mu_completed,
    }


def fuzzy_similarity_from_ast(ast, mu_completed: pd.DataFrame) -> pd.Series:
    return mu_completed.apply(lambda r: eval_ast_fuzzy(ast, r), axis=1).rename("Fuzzy")


# -----------------------------------------------------------------------------
# 4) App — pages
# -----------------------------------------------------------------------------
st.set_page_config(page_title="IR: Extended Boolean vs Fuzzy (D1–D15)", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    (
        "📊 Page 1 — Ranking (Bar Charts)",
        "📘 Page 2 — Extended Boolean (Step 1–3)",
        "📗 Page 3 — Fuzzy Model (Step 1–3)",
    ),
    index=0,
)

# Global query controls
st.sidebar.markdown("---")
st.sidebar.markdown("**Query (supports AND / OR / NOT / parentheses)**")
default_query = "(cat AND dog) AND NOT tiger"
query_str = st.sidebar.text_input("Enter a query", value=default_query)

and_mode_label = st.sidebar.selectbox(
    "Extended Boolean AND definition",
    ["DeMorgan (slide-style)", "Direct p-norm"],
    index=0,
)
and_mode = "demorgan" if and_mode_label.startswith("DeMorgan") else "pnorm"
p_val = st.sidebar.slider("p value (Extended Boolean)", 1.0, 8.0, 2.0, step=1.0)

# Run pipelines
ext = extended_boolean_pipeline(DOCS, TERMS, p=p_val, and_mode=and_mode)
fuz = fuzzy_pipeline(DOCS, TERMS)

# Parse query
parse_error = None
try:
    ast = parse_query(query_str)
except Exception as e:
    parse_error = str(e)


# -----------------------------------------------------------------------------
# PAGE 1 — Ranking (Top-k)
# -----------------------------------------------------------------------------
if page.startswith("📊"):
    st.title("📊 Top-k Similarity Scores — Extended Boolean vs Fuzzy")

    if parse_error:
        st.error(f"Query parse error: {parse_error}")
        st.stop()

    max_docs = len(DOCS)
    k = st.sidebar.slider("Top-k", min_value=1, max_value=max_docs, value=5, step=1)

    sim_ext = extended_boolean_similarity_from_ast(ast, ext["weights"], p=ext["p"], and_mode=ext["and_mode"])
    sim_fuz = fuzzy_similarity_from_ast(ast, fuz["mu_completed"])

    # sort and clip
    top_ext = sim_ext.sort_values(ascending=False).head(k)
    top_fuz = sim_fuz.sort_values(ascending=False).head(k)

    # combined table (all docs)
    scores_all = pd.concat([sim_ext, sim_fuz], axis=1).loc[sorted(DOCS.keys())]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📘 Extended Boolean — Top {k}")
        fig_ext = px.bar(
            x=top_ext.values,
            y=top_ext.index,
            orientation="h",
            text=[f"{v:.3f}" for v in top_ext.values],
            color=top_ext.values,
            color_continuous_scale="Blues",
            labels={"x": "Similarity", "y": "Document"},
            title=f"Extended Boolean (Top {k})",
        )
        fig_ext.update_layout(
            yaxis=dict(autorange="reversed"),
            height=460,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ext, use_container_width=True)

    with col2:
        st.subheader(f"📗 Fuzzy — Top {k}")
        fig_fuz = px.bar(
            x=top_fuz.values,
            y=top_fuz.index,
            orientation="h",
            text=[f"{v:.3f}" for v in top_fuz.values],
            color=top_fuz.values,
            color_continuous_scale="Greens",
            labels={"x": "Similarity", "y": "Document"},
            title=f"Fuzzy (Top {k})",
        )
        fig_fuz.update_layout(
            yaxis=dict(autorange="reversed"),
            height=460,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fuz, use_container_width=True)

    st.subheader("📄 Scores — All Documents")
    st.dataframe(scores_all.style.format("{:.4f}"), use_container_width=True)

    st.download_button(
        "Download scores (CSV)",
        data=scores_all.to_csv(index=True).encode("utf-8"),
        file_name="scores_comparison.csv",
        mime="text/csv",
    )

    with st.expander("Show corpus (document tokens)"):
        st.write(DOCS)


# -----------------------------------------------------------------------------
# PAGE 2 — Extended Boolean (details)
# -----------------------------------------------------------------------------
elif page.startswith("📘"):
    st.title("📘 Extended Boolean — Steps 1–3")
    st.caption(
        f"AND = {'DeMorgan' if ext['and_mode']=='demorgan' else 'Direct p-norm'}   |   p = {int(ext['p'])}"
    )

    st.markdown("### Step 1 — TF–IDF (Counts → TF, IDF, Weights)")
    st.markdown("- **TF** per doc: `count / MaxCount(doc)`")
    st.markdown("- **IDF_norm**: `log10(N/df)` then normalized by the maximum IDF")
    st.markdown("- **Weight**: `TF × IDF_norm`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Counts + Max (per document)**")
        st.dataframe(ext["counts"], use_container_width=True)

        st.markdown("**TF (count / Max)**")
        st.dataframe(ext["tf"].style.format("{:.3f}"), use_container_width=True)

    with c2:
        st.markdown("**IDF_norm (normalized)**")
        st.dataframe(ext["idf_norm"].to_frame(), use_container_width=True)

        st.markdown("**Weights = TF × IDF_norm**")
        st.dataframe(ext["weights"].style.format("{:.3f}"), use_container_width=True)

    st.markdown("### Step 2 — Similarity (p-norm)")
    st.write("DeMorgan AND:")
    st.latex(r"AND(x,y)=1-\left(\frac{(1-x)^p+(1-y)^p}{2}\right)^{1/p}")
    st.write("Direct p-norm AND:")
    st.latex(r"AND(x,y)=\left(\frac{x^p+y^p}{2}\right)^{1/p}")
    st.write("OR and NOT:")
    st.latex(r"OR(x,y)=1-\;AND(1-x,\,1-y)")
    st.latex(r"NOT(x)=1-x")

    if parse_error:
        st.error(f"Query parse error: {parse_error}")
    else:
        sim_ext = extended_boolean_similarity_from_ast(ast, ext["weights"], p=ext["p"], and_mode=ext["and_mode"])
        st.subheader("Similarity per document")
        st.dataframe(sim_ext.to_frame().style.format("{:.4f}"), use_container_width=True)

        st.subheader("Ranking (High → Low)")
        ranking_ext = sim_ext.sort_values(ascending=False).rename("Extended Boolean")
        st.dataframe(ranking_ext.to_frame().style.format("{:.4f}"), use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 3 — Fuzzy (details)
# -----------------------------------------------------------------------------
else:
    st.title("📗 Fuzzy Model — Steps 1–3")

    st.markdown("### Step 1 — Membership from frequency (per document)")
    st.latex(r"\mu_t(D)=\frac{\mathrm{count}(t,D)}{\max_x \mathrm{count}(x,D)}\in[0,1]")
    st.write("Weighted Jaccard correlation across documents:")
    st.latex(r"c(i,j)=\frac{\sum_D \min(\mu_i(D),\mu_j(D))}{\sum_D \max(\mu_i(D),\mu_j(D))}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Counts + Max**")
        st.dataframe(fuz["counts"], use_container_width=True)

        st.markdown("**μ_raw (TF normalized)**")
        st.dataframe(fuz["mu_raw"].style.format("{:.3f}"), use_container_width=True)

    with c2:
        st.markdown("**Correlation (Weighted Jaccard)**")
        st.dataframe(fuz["correlation"].style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Step 2 — Membership completion (fill missing terms by correlation)")
    st.write("If a term is missing in a document (i.e., $\\mu_t(D)=0$), fill it using the strongest correlation with present terms:")
    st.latex(r"\mu'_t(D)=\max_{x\in T(D)}\big(\mu_x(D)\cdot c(t,x)\big)")
    st.write("If the term already exists ($t\in T(D)$), keep its value:")
    st.latex(r"\mu'_t(D)=\mu_t(D)")

    st.dataframe(fuz["mu_completed"].style.format("{:.3f}"), use_container_width=True)
    st.download_button(
        "Download μ_completed (CSV)",
        data=fuz["mu_completed"].to_csv(index=True).encode("utf-8"),
        file_name="fuzzy_mu_completed.csv",
        mime="text/csv",
    )

    st.markdown("### Step 3 — Similarity (Fuzzy AND / OR / NOT)")
    st.write("Fuzzy operators:")
    st.latex(r"AND(a,b)=\min(a,b),\quad OR(a,b)=\max(a,b),\quad NOT(x)=1-x")
    st.write("Works with the same user query (AND/OR/NOT/parentheses).")

    if parse_error:
        st.error(f"Query parse error: {parse_error}")
    else:
        sim_fuz = fuzzy_similarity_from_ast(ast, fuz["mu_completed"])
        st.subheader("Similarity per document")
        st.dataframe(sim_fuz.to_frame().style.format("{:.4f}"), use_container_width=True)

        st.subheader("Ranking (High → Low)")
        ranking_fuz = sim_fuz.sort_values(ascending=False).rename("Fuzzy")
        st.dataframe(ranking_fuz.to_frame().style.format("{:.4f}"), use_container_width=True)

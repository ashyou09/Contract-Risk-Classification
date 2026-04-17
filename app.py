import json
import os
import re
from io import BytesIO, StringIO

import pandas as pd
import plotly.express as px
import PyPDF2
import streamlit as st
from dotenv import load_dotenv

from contract_agent.core.domain import DOMAIN_META, detect_domain, get_domain_badge_html
from contract_agent.utils.text import clean_text, get_summary, segment_clauses

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — premium dark-accent design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    padding: 14px 18px;
    border-radius: 12px;
    background: var(--secondary-background-color);
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
}

/* ── Upload dropzone ── */
[data-testid="stFileUploadDropzone"] {
    border-radius: 16px;
    border: 2px dashed #3b82f6;
    background: rgba(59,130,246,.05);
    padding: 32px 20px;
    transition: all .25s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb;
    background: rgba(59,130,246,.1);
    transform: scale(1.01);
}

/* ── Expander header ── */
.streamlit-expanderHeader { font-weight: 600 !important; border-radius: 8px !important; }

/* ── Clause risk cards ── */
.risk-card {
    padding: 18px 20px; border-radius: 10px; margin-bottom: 10px;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 4px rgba(0,0,0,.1);
    border: 1px solid rgba(128,128,128,.15);
    transition: box-shadow .18s;
}
.risk-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.18); }
.risk-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.18); }

/* ── Mode badge ── */
.badge-online  { display:inline-block; padding:3px 12px; border-radius:20px;
                 font-size:.78rem; font-weight:700;
                 background:linear-gradient(135deg,#1e40af,#3b82f6); color:#fff; }
.badge-offline { display:inline-block; padding:3px 12px; border-radius:20px;
                 font-size:.78rem; font-weight:700;
                 background:linear-gradient(135deg,#14532d,#22c55e); color:#fff; }

/* ── Action pill ── */
.action-pill {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:.78rem; font-weight:600; margin-left:8px;
}
.pill-remove    { background:#fee2e2; color:#b91c1c; }
.pill-negotiate { background:#fef3c7; color:#92400e; }
.pill-legal     { background:#ede9fe; color:#5b21b6; }
.pill-accept    { background:#d1fae5; color:#065f46; }

/* ── Welcome card ── */
.welcome-card {
    border-radius: 16px;
    padding: 32px 36px;
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(99,102,241,.06));
    border: 1px solid rgba(99,102,241,.2);
    margin-bottom: 24px;
}

/* ── Trigger card (Step C) ── */
.trigger-card {
    border-radius: 16px;
    padding: 28px 32px;
    background: linear-gradient(135deg, rgba(129,140,248,.07), rgba(56,189,248,.05));
    border: 1px solid rgba(129,140,248,.25);
    margin: 24px 0;
    text-align: center;
}

/* ── Domain badge row ── */
.domain-row {
    display:flex; align-items:center; gap:12px;
    margin-bottom: 16px;
}

/* ── RAG status pill ── */
.rag-chroma { display:inline-block; padding:3px 10px; border-radius:12px;
              font-size:.75rem; font-weight:600;
              background:#d1fae5; color:#065f46; }
.rag-tfidf  { display:inline-block; padding:3px 10px; border-radius:12px;
              font-size:.75rem; font-weight:600;
              background:#fef3c7; color:#92400e; }

/* ── Circular checkboxes for the filter popover ── */
div[data-testid="stPopoverBody"] div[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {
    border-radius: 50% !important;
}

/* ── Pill styled popover button ── */
button[data-testid="stPopoverButton"] {
    border-radius: 20px !important;
    padding: 2px 20px !important;
}

/* ── Custom HTML Expander for Clauses ── */
details.custom-expander {
    margin-bottom: 12px;
    border-radius: 10px;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 4px rgba(0,0,0,.1);
    border: 1px solid rgba(128,128,128,.15);
    transition: box-shadow .18s;
}
details.custom-expander:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,.18);
}
summary.custom-summary {
    padding: 16px 20px;
    cursor: pointer;
    font-weight: 500;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 12px;
}
summary.custom-summary::-webkit-details-marker {
    display: none;
}
details.custom-expander[open] summary .expand-icon {
    transform: rotate(90deg);
}
.expand-icon {
    display: inline-block;
    transition: transform 0.2s;
    font-size: 0.8rem;
    opacity: 0.5;
}
.risk-pill {
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
    display: inline-block;
}
.pill-high { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.pill-medium { background-color: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.pill-low { background-color: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.preview-text {
    opacity: 0.8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.95rem;
}
.details-content {
    padding: 16px 20px;
    border-top: 1px solid rgba(128,128,128,.1);
    font-size: 0.95rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from contract_agent.utils.ml import load_sklearn_pipeline
    return load_sklearn_pipeline()


@st.cache_data(ttl=30)
def _ollama_health():
    from contract_agent.llm.local import check_ollama_health
    return check_ollama_health()


@st.cache_data(ttl=120)
def _detect_domain_cached(text_sample: str) -> str:
    """Domain detection is cached per unique text sample."""
    return detect_domain(text_sample, llm_fallback=True)


def _rag_status_html() -> str:
    """Return an HTML pill showing whether ChromaDB or TF-IDF is active."""
    from contract_agent.retrieval.chroma import _get_chroma_collection
    col = _get_chroma_collection()
    if col is not None:
        count = col.count()
        return f'<span class="rag-chroma">ChromaDB · {count} guidelines</span>'
    return '<span class="rag-tfidf">TF-IDF fallback — run python rag_setup.py for ChromaDB</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Step B — Milestone 1 ML Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _render_ml_dashboard(df: pd.DataFrame, domain: str):
    """Render the classical ML risk classification results with domain badge."""

    # Domain header
    badge_html = get_domain_badge_html(domain)
    st.markdown(
        f'<div class="domain-row">'
        f'<span style="font-weight:600;font-size:.95rem;">Contract Type Detected:</span> '
        f'{badge_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    high = len(df[df["Risk Level"] == "High"])
    med  = len(df[df["Risk Level"] == "Medium"])
    low  = len(df[df["Risk Level"] == "Low"])
    m1.metric("High Risk",   high)
    m2.metric("Medium Risk", med)
    m3.metric("Low Risk",    low)
    m4.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Risk Distribution")
        st.plotly_chart(px.pie(df, names="Risk Level", color="Risk Level",
            color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"}),
            use_container_width=True)
    with c2:
        st.markdown("#### Confidence Score Distribution")
        st.plotly_chart(px.histogram(df, x="Confidence", color="Risk Level", nbins=20,
            color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"}),
            use_container_width=True)

    df = df.copy()
    df["Risk_Score"] = df["Risk Level"].map({"High": 3, "Medium": 2, "Low": 1})
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Risk Intensity Across Document")
        st.plotly_chart(px.scatter(df, x=df.index, y="Risk_Score", color="Risk Level",
            color_discrete_map={"High": "#ef4444", "Medium": "#f97316", "Low": "#22c55e"},
            labels={"index": "Clause Position", "Risk_Score": "Intensity"}),
            use_container_width=True)
    with c4:
        st.markdown("#### Top Keywords (High Risk)")
        from collections import Counter
        words = re.findall(r"\b\w+\b",
            " ".join(df[df["Risk Level"] == "High"]["Clause Text"].tolist()).lower())
        stops = {"the","and","to","of","in","a","is","that","for","it","on","be",
                 "as","by","or","this","an","are","with","from","at","not","will"}
        filtered = Counter(w for w in words if w not in stops and len(w) > 3).most_common(10)
        if filtered:
            kw = pd.DataFrame(filtered, columns=["Keyword", "Count"])
            st.plotly_chart(px.bar(kw, x="Keyword", y="Count", color="Count",
                color_continuous_scale="Reds"), use_container_width=True)
        else:
            st.info("No high-risk clauses detected.")

    st.markdown("---")
    st.markdown("### Clause Detail")
    
    with st.popover("Filter Clauses", use_container_width=False):
        high_check = st.checkbox("High Risk", value=True)
        med_check  = st.checkbox("Medium Risk", value=True)
        low_check  = st.checkbox("Low Risk", value=True)
        
    picks = []
    if high_check: picks.append("High")
    if med_check: picks.append("Medium")
    if low_check: picks.append("Low")
    
    filtered_df = df[df["Risk Level"].isin(picks)]
    
    # ── Pagination logic ──
    items_per_page = 10
    total_items = len(filtered_df)
    total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
    
    if "clause_page" not in st.session_state:
        st.session_state.clause_page = 1
        
    if st.session_state.clause_page > total_pages:
        st.session_state.clause_page = max(1, total_pages)
        
    start_idx = (st.session_state.clause_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    for _, row in page_df.iterrows():
        risk = row["Risk Level"]
        cls  = risk.lower() + "-risk"
        pill_cls = f"pill-{risk.lower()}"
        preview = get_summary(row['Clause Text'], 85)
        conf = row['Confidence']
        
        html = f"""
        <details class="custom-expander {cls}">
            <summary class="custom-summary">
                <span class="expand-icon">▶</span>
                <span class="risk-pill {pill_cls}">{risk} Risk</span>
                <span class="preview-text">{preview} </span>
                <span style="margin-left:auto; opacity:0.5; font-size:0.85rem;">{conf:.1%}</span>
            </summary>
            <div class="details-content">
                <p style="margin:0;">{row['Clause Text']}</p>
            </div>
        </details>
        """
        st.markdown(html, unsafe_allow_html=True)
        
    # ── Render Paginator ──
    if total_pages > 1:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Page {st.session_state.clause_page} of {total_pages}**")
        
        max_cols = min(total_pages, 10)
        # Use smaller weights for buttons to keep them tighter together
        btn_cols = st.columns([0.5] * max_cols + [10], gap="small") 
        
        for p in range(1, max_cols + 1):
            if p == st.session_state.clause_page:
                # Active page: Match Streamlit button dimensions and styling
                btn_cols[p-1].markdown(f"""
                <div style='
                    height: 38.4px;
                    display: flex; justify-content: center; align-items: center;
                    background-color: #d1fae5; color: #065f46;
                    border-radius: 0.5rem; font-size: 14px; font-weight: 500;
                    border: 1px solid #34d399; cursor: default;
                    margin: 0; padding: 0;'>
                    {p}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Inactive page
                if btn_cols[p-1].button(f"{p}", key=f"page_btn_{p}_{min(total_pages,10)}", use_container_width=True):
                    st.session_state.clause_page = p
                    st.rerun()

    st.markdown("---")
    st.download_button(
        "📥 Download CSV",
        df.drop(columns=["Risk_Score"], errors="ignore").to_csv(index=False).encode(),
        "contract_risk.csv", "text/csv", key="dl_m1_csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step C + D — Milestone 2 Agentic Panel (Streaming Master-Detail)
# ─────────────────────────────────────────────────────────────────────────────
def _render_agentic_panel(
    raw_text:  str,
    threshold: float,
    mode:      str,
    domain:    str,
    df_ml:     pd.DataFrame,
):
    """
    Implements the UI flow:
      Step C        — "Analyze with AI" trigger button
      Step D        — Interactive streaming master-detail workflow
    """
    from contract_agent.retrieval.chroma import LegalPracticeRetriever
    from contract_agent.utils.ml import load_sklearn_pipeline
    from contract_agent.llm.prompting import safe_parse_analysis

    is_online = mode == "online"
    badge_cls = "badge-online" if is_online else "badge-offline"
    badge_lbl = "🌐 Online"   if is_online else "🖥 Offline"

    # ── Guard checks (Online always passes via Free API fallback) ────────────
    if not is_online:
        h = _ollama_health()
        if not h["reachable"]:
            st.error("❌ **Ollama is not reachable.** Run `ollama serve` in a terminal, then refresh.")
            return

    state_key = f"state_{mode}"
    queue_key = f"q_{mode}"
    res_key   = f"res_{mode}"
    idx_key   = f"idx_{mode}"

    cur_state = st.session_state.get(state_key, "initial")

    # ── Step C: Trigger button ──────────────────────────────────────────────
    if cur_state == "initial":
        from contract_agent.llm.cloud import check_cloud_health
        health   = check_cloud_health()
        groq_tag = " + Groq fallback" if health.get("groq") else ""

        st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px;">
            <h2 style="font-size: 2.2rem; margin-bottom: 15px;">Deep AI Legal Review</h2>
            <span class="{badge_cls}" style="margin-bottom: 20px;">{badge_lbl}{groq_tag}</span>
            <p style="font-size: 1.1rem; opacity: 0.8; max-width: 600px; margin: 20px auto 30px auto; line-height: 1.6;">
                Unleash the full power of our Legal LangGraph AI. We will interactively analyze <strong>every single clause</strong> against established legal precedence for your contract's domain (<strong>{domain}</strong>), providing negotiation tips and safer rewrites.
            </p>
        </div>
        """, unsafe_allow_html=True)



        _, btn_col, _ = st.columns([1, 2, 1])
        if btn_col.button(f"Start Deep Analysis ({domain} Domain)", type="primary", use_container_width=True, key=f"run_{mode}"):
            ml_model = load_model()
            if ml_model is None:
                st.error("⚠️ ML model not found. Run training first.")
                return

            with st.spinner("Preparing Document Clauses..."):
                from contract_agent.utils.text import segment_clauses, clean_text
                from contract_agent.retrieval.chroma import DomainAwareRetriever

                clauses  = segment_clauses(raw_text)
                retriever = DomainAwareRetriever(top_k=3)
                
                queue = []
                for clause in clauses:
                    cleaned  = clean_text(clause)
                    risk     = ml_model.predict([cleaned])[0]
                    conf     = float(max(ml_model.predict_proba([cleaned])[0]))
                    bp       = retriever.retrieve(clause, domain=domain)
                    
                    queue.append({
                        "clause_text": clause,
                        "risk_level":  risk,
                        "confidence":  conf,
                        "best_practices": bp
                    })
                
                st.session_state[queue_key] = queue
                st.session_state[res_key] = []
                st.session_state[idx_key] = 0
                st.session_state[state_key] = "processing"
                st.rerun()

    # ── Step D: Processing or Finished UI ──────────────────────────────────────
    else:
        queue = st.session_state.get(queue_key, [])
        results = st.session_state.get(res_key, [])
        
        # Guard against orphaned state if user reloads
        if not queue and not results and cur_state == "processing":
            st.session_state[state_key] = "initial"
            st.rerun()
            
        total = len(queue) + len(results)
        pct = len(results) / max(total, 1)

        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;">'
            f'<span style="font-weight:600;">Domain Context:</span> {get_domain_badge_html(domain)} &nbsp; {_rag_status_html()}</div>',
            unsafe_allow_html=True,
        )

        if cur_state == "processing":
            st.progress(pct, text=f"Analyzing clause {len(results)+1} of {total} in background... You can begin reading now.")
        else:
            st.success(f"Analysis Complete — Reviewed all {total} clauses.")
            col1, col2, _ = st.columns([1, 1, 3])
            
            if col1.button("Restart Analysis", use_container_width=True):
                st.session_state[state_key] = "initial"
                st.rerun()
                
            import json
            json_blob = json.dumps(results, indent=2).encode('utf-8')
            with col2:
                st.download_button("Download JSON", json_blob, "report.json", "application/json", use_container_width=True)
        
        st.markdown("---")
        
        # ── Visual Layout: Two Columns ──
        left_col, right_col = st.columns([1.3, 2.7], gap="medium")
        
        with left_col:
            st.markdown("##### 📄 Clauses List")
            if not results:
                st.caption("Waiting for first clause...")
            
            # Scrollable wrapper
            st.markdown("<div style='max-height: 65vh; overflow-y: auto; padding-right: 15px;'>", unsafe_allow_html=True)
            for i, res in enumerate(results):
                risk = res["risk_level"]
                icon = "🔴" if risk=="High" else "🟠" if risk=="Medium" else "🟢"
                label = f"{icon} Clause {i+1} — {risk} Risk"
                
                is_selected = (st.session_state.get(idx_key, 0) == i)
                btn_type = "primary" if is_selected else "secondary"
                
                if st.button(label, key=f"sel_{mode}_{i}", use_container_width=True, type=btn_type):
                    st.session_state[idx_key] = i
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            
        with right_col:
            if results:
                sel_idx = st.session_state.get(idx_key, 0)
                if sel_idx >= len(results):
                    sel_idx = 0
                    st.session_state[idx_key] = 0
                
                cur = results[sel_idx]
                r_num = sel_idx + 1
                an = cur["analysis"]
                risk = cur["risk_level"]
                conf = cur["confidence"]
                
                st.markdown(f"### Clause {r_num} Breakdown")
                
                st.markdown(f"**Risk Bearer:** {an.get('who_bears_the_risk', 'Unknown')} &nbsp;|&nbsp; **ML Risk:** {risk} &nbsp;|&nbsp; **Action:** {an.get('action_required', 'Review')}")
                st.markdown("---")
                
                from contract_agent.utils.text import get_summary
                summary_text = get_summary(cur["clause_text"], 150)
                
                st.markdown("##### What It Says")
                st.info(an.get('plain_english_summary', '—'))
                
                if risk in ("High", "Medium"):
                    cl1, cl2 = st.columns(2)
                    with cl1:
                        st.markdown("##### Why It's Risky")
                        st.warning(an.get('what_makes_it_risky', '—'))
                    with cl2:
                        st.markdown(f"##### {domain} Practice")
                        st.success(an.get('industry_standard_practice', '—'))
                        
                    st.markdown("##### Negotiation & Mitigation")
                    st.write(an.get('negotiation_tips', '—'))
                    
                    st.markdown("##### Safer Rewrite")
                    st.code(an.get('safer_rewrite', '—'), language=None)
                else:
                    st.markdown("##### Assessment")
                    st.success("This clause is standard and lower risk. No immediate modifications required.")

                with st.expander("Show Original Clause Text"):
                    st.write(cur["clause_text"])
            else:
                st.info("The first clause is currently being analyzed by the AI...")

        # ── Background Process Chunk ──
        if cur_state == "processing" and queue:
            import time
            row = queue.pop(0)
            
            if is_online:
                from contract_agent.llm.cloud import analyze_clause_with_cloud as _analyze
            else:
                from contract_agent.llm.local import analyze_clause_with_ollama as _analyze

            try:
                analysis = _analyze(
                    row["clause_text"],
                    str(row["risk_level"]),
                    float(row["confidence"]),
                    row.get("best_practices", []),
                    domain=domain,
                )
            except Exception as exc:
                analysis = safe_parse_analysis("{}", fallback_text=f"LLM error: {exc}")
            
            results.append({**row, "analysis": analysis})
            
            st.session_state[queue_key] = queue
            st.session_state[res_key]   = results
            
            time.sleep(0.05) # Tiny beat
            st.rerun()
            
        elif cur_state == "processing" and not queue:
            st.session_state[state_key] = "finished"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Session state init
    for k in ("file_data", "file_name", "file_type", "detected_domain"):
        st.session_state.setdefault(k, None)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Quick Settings")

        # AI mode toggle
        st.markdown("#### Processing Mode")
        mode_choice = st.radio(
            "How should we analyze your document?",
            ["Online", "Offline"],
            index=0, key="mode_radio",
            help="Cloud AI uses our fast online servers. Local PC uses your own computer for complete privacy.",
        )
        selected_mode = "online" if "Cloud" in mode_choice else "offline"

        st.markdown("---")

        # File update & Clear
        if st.session_state.file_data is not None:
            st.markdown("#### Document Actions")
            nf = st.file_uploader("Replace current document", type=["txt", "pdf"], key="sidebar_uploader")
            if nf:
                st.session_state.file_data    = nf.getvalue()
                st.session_state.file_name    = nf.name
                st.session_state.file_type    = nf.type
                st.session_state.detected_domain = None
                for k in list(st.session_state.keys()):
                    if k.startswith("report_"):
                        del st.session_state[k]
                st.rerun()

            if st.button("🗑️ Clear Current File", use_container_width=True):
                for k in ["file_data", "file_name", "file_type", "detected_domain"]:
                    st.session_state[k] = None
                for k in list(st.session_state.keys()):
                    if k.startswith("report_"):
                        del st.session_state[k]
                st.rerun()
            st.markdown("---")

        with st.expander("Advanced Controls"):
            st.caption("For power users and technical debugging.")
            
            st.markdown("**System Health Checks**")
            if selected_mode == "online":
                from contract_agent.llm.cloud import check_cloud_health
                health = check_cloud_health()
                if health.get("pollinations"):
                    st.success("Cloud Server: Active")
                elif health["openrouter"]:
                    st.success("Cloud Server: Active")
                else:
                    st.warning("Cloud Server: Down / Missing Keys")
            else:
                h = _ollama_health()
                if h["reachable"]:
                    st.success("Local Engine: Linked")
                    avail = h.get("available_models", [])
                    if len(avail) > 1:
                        chosen = st.selectbox("Local model:", avail,
                            index=avail.index(h["model"]) if h["model"] in avail else 0,
                            key="ollama_model_select")
                        os.environ["OLLAMA_MODEL"] = chosen
                else:
                    st.error("Local Engine Not Found")

            st.markdown("---")
            st.markdown("**Legal Knowledge Base**")
            from contract_agent.retrieval.chroma import _get_chroma_collection
            col = _get_chroma_collection()
            
            if col is not None:
                st.markdown(_rag_status_html(), unsafe_allow_html=True)
                if st.button("Rebuild Database", use_container_width=True):
                    from rag_setup import build_vector_db
                    with st.spinner("Processing..."):
                        try:
                            build_vector_db(reset=True)
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Rebuild failed: {_e}")
            else:
                st.info("Initializing Knowledge Base for the first time...")
                from rag_setup import build_vector_db
                with st.spinner("Downloading embedding models & building database... (this may take 60s)"):
                    try:
                        build_vector_db()
                        st.rerun()
                    except Exception as _e:
                        st.error(
                            f"ChromaDB setup failed: {_e}\n\n"
                            "The app will use TF-IDF retrieval as a fallback."
                        )

            st.markdown("---")
            st.markdown("**Risk Sensitivity**")
            confidence_threshold = st.slider(
                "Filter by confidence", 0.0, 1.0, 0.5, 0.05,
                help="Only show results if AI is highly confident.",
            )

        st.markdown("---")
        st.caption("Contract Risk Management System\nPrototype Build")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1rem;'>
        <h1 style='font-size:3rem; font-weight:800;
            background:linear-gradient(90deg,#3B82F6,#818CF8,#3B82F6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            margin-bottom:.5rem;'>
            Intelligent Contract Risk Analysis
        </h1>
        <p style='opacity:.75; font-size:1.15rem; font-weight:400;'>
            Uncover hidden risks in legal documents instantly using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='width:50%;margin:6px auto 28px;border:none;border-top:1px solid rgba(128,128,128,.25);'>",
        unsafe_allow_html=True)

    # ── Step A: File Upload ───────────────────────────────────────────────────
    if st.session_state.file_data is None:
        _, col, _ = st.columns([1, 2, 1])
        with col:
            mf = st.file_uploader(
                "Upload a Contract (PDF or TXT) to begin",
                type=["txt", "pdf"], key="main_uploader",
            )
            if mf:
                st.session_state.file_data = mf.getvalue()
                st.session_state.file_name = mf.name
                st.session_state.file_type = mf.type
                st.session_state.detected_domain = None
                st.rerun()
        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; padding:70px 40px; opacity:.6;'>
            <h3 style='font-weight:500;'>Awaiting document</h3>
            <p style='font-size:1.05rem;'>Upload a PDF or TXT file to initiate analysis workflow</p>
        </div>""", unsafe_allow_html=True)
        return

    # ── ML model check ────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("ML model not initialized. Contact your administrator.")
        return

    # ── Parse file ────────────────────────────────────────────────────────────
    text = ""
    try:
        if st.session_state.file_type == "application/pdf":
            reader = PyPDF2.PdfReader(BytesIO(st.session_state.file_data))
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    text += t + "\n"
        else:
            text = StringIO(st.session_state.file_data.decode("utf-8")).read()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    if not text.strip():
        st.warning("No extractable text found in this file.")
        return

    # ── Domain detection (cached, runs once per document) ─────────────────────
    if st.session_state.detected_domain is None:
        with st.spinner("Analyzing document properties..."):
            st.session_state.detected_domain = _detect_domain_cached(text[:3000])
    detected_domain = st.session_state.detected_domain

    # ── Step B: ML Classification ─────────────────────────────────────────────
    clauses_ml = segment_clauses(text)
    data_ml    = []
    for clause in clauses_ml:
        cleaned = clean_text(clause)
        risk    = model.predict([cleaned])[0]
        probs   = model.predict_proba([cleaned])[0]
        conf    = max(probs)
        if conf >= confidence_threshold:
            data_ml.append({
                "Clause Text": clause,
                "Risk Level":  risk,
                "Confidence":  conf,
            })
    df = pd.DataFrame(data_ml)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "Quick Screening",
        "AI Deep Analysis",
    ])

    with tab1:
        if df.empty:
            st.warning("No clauses above the confidence threshold.")
        else:
            _render_ml_dashboard(df, domain=detected_domain)

    with tab2:
        if df.empty:
            st.warning("No clauses above the confidence threshold — nothing to analyse.")
        else:
            _render_agentic_panel(
                raw_text=text,
                threshold=confidence_threshold,
                mode=selected_mode,
                domain=detected_domain,
                df_ml=df,
            )


if __name__ == "__main__":
    main()

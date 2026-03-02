import streamlit as st
import pandas as pd
import pickle
import os
import re
import plotly.express as px
from io import StringIO, BytesIO
import PyPDF2

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
/* Main App Background and Typography */
.stApp {
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

/* Metric Cards Styling */
[data-testid="stMetric"] {
    padding: 15px 20px;
    border-radius: 12px;
    border: none;
    background: var(--secondary-background-color);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
}

/* Risk Clause Cards Typography and Spacing */
.risk-card {
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 12px;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid rgba(128,128,128,0.2);
    transition: all 0.2s ease;
}
.risk-card:hover {
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
.high-risk { border-left: 6px solid #ef4444 !important; }
.medium-risk { border-left: 6px solid #f97316 !important; }
.low-risk { border-left: 6px solid #22c55e !important; }

/* Upload Dropzone Styling */
[data-testid="stFileUploadDropzone"] {
    border-radius: 16px;
    border: 2px dashed #3b82f6;
    background-color: rgba(59, 130, 246, 0.05);
    padding: 30px 20px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb;
    background-color: rgba(59, 130, 246, 0.1);
    transform: scale(1.01);
}

/* Customizing Streamlit Expander Header */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Load Pipeline Model
# =============================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.pkl")

    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


# =============================
# Helper Functions
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-z0-9\s.,;:\-\'\"()/]', '', text)
    return text


def segment_clauses(text):
    paragraphs = text.split('\n\n')
    clauses = []

    for para in paragraphs:
        split_para = re.split(r'(?:\n|^)\s*(?:\d+\.|[a-z]\))\s+', para)
        for p in split_para:
            p = p.strip()
            if len(p) > 50:
                clauses.append(p)

    return clauses


def get_summary(text, limit=200):
    return text if len(text) <= limit else text[:limit] + "..."


# =============================
# Main App
# =============================
def main():
    if "file_data" not in st.session_state:
        st.session_state.file_data = None
        st.session_state.file_name = None
        st.session_state.file_type = None

    # Sidebar
    with st.sidebar:
        if st.session_state.file_data is not None:
            st.markdown("### Update Document")
            new_file = st.file_uploader("Upload a New Contract", type=["txt", "pdf"], help="Drag and drop or click to select.", key="sidebar_uploader")
            if new_file:
                st.session_state.file_data = new_file.getvalue()
                st.session_state.file_name = new_file.name
                st.session_state.file_type = new_file.type
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Settings")
            confidence_threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, 0.05)
            st.markdown("---")
            
            if st.button("Clear Current File", use_container_width=True):
                st.session_state.file_data = None
                st.session_state.file_name = None
                st.session_state.file_type = None
                st.rerun()
        else:
            st.markdown("### Settings")
            confidence_threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, 0.05)

    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='
                font-size: 3rem; 
                font-weight: 800; 
                background: linear-gradient(90deg, #3B82F6, #60A5FA, #3B82F6); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                margin-bottom: 0.5rem;'>
                Intelligent Contract Risk Analysis
            </h1>
            <p style='color: var(--text-color); opacity: 0.8; font-size: 1.25rem; font-weight: 400;'>
                Uncover hidden risks in legal documents instantly using AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='width: 50%; margin: 10px auto 30px auto; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    if st.session_state.file_data is None:
        # Center File Upload
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            main_file = st.file_uploader("Upload a Contract (PDF or TXT)", type=["txt", "pdf"], help="Drag and drop or click to select.", key="main_uploader")
            if main_file:
                st.session_state.file_data = main_file.getvalue()
                st.session_state.file_name = main_file.name
                st.session_state.file_type = main_file.type
                st.rerun()
        
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    else:
        st.success(f"Analysis Ready: **{st.session_state.file_name}** - Scroll down for results")
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Load Model
    model = load_model()

    if model is None:
        st.error("⚠️ Model missing! Run training first.")
        return

    if st.session_state.file_data is not None:

        # Read File
        text = ""
        try:
            if st.session_state.file_type == "application/pdf":
                reader = PyPDF2.PdfReader(BytesIO(st.session_state.file_data))
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            else:
                stringio = StringIO(st.session_state.file_data.decode("utf-8"))
                text = stringio.read()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        if text:
            clauses = segment_clauses(text)

            data = []

            for clause in clauses:
                cleaned = clean_text(clause)

                # 🔥 Direct prediction (NO manual TF-IDF)
                risk = model.predict([cleaned])[0]
                probs = model.predict_proba([cleaned])[0]
                confidence = max(probs)

                if confidence >= confidence_threshold:
                    data.append({
                        "Clause Text": clause,
                        "Risk Level": risk,
                        "Confidence": confidence
                    })

            df = pd.DataFrame(data)

            if df.empty:
                st.warning("No clauses found matching criteria.")
                return

            # =============================
            # Dashboard
            # =============================

            # -----------------------------
            # Metrics Row
            # -----------------------------
            st.markdown("### 📊 Executive Summary")

            m1, m2, m3, m4 = st.columns(4)

            high_count = len(df[df["Risk Level"] == "High"])
            med_count = len(df[df["Risk Level"] == "Medium"])
            low_count = len(df[df["Risk Level"] == "Low"])
            avg_conf = df["Confidence"].mean()

            m1.metric("High Risk Clauses", high_count,
                    delta="Attention Needed" if high_count > 0 else "Safe",
                    delta_color="inverse")

            m2.metric("Medium Risk Clauses", med_count)
            m3.metric("Low Risk Clauses", low_count)
            m4.metric("Avg Confidence", f"{avg_conf:.1%}")

            # -----------------------------
            # Row 1: Pie + Histogram
            # -----------------------------
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Risk Distribution")
                fig_pie = px.pie(
                    df,
                    names="Risk Level",
                    color="Risk Level",
                    color_discrete_map={
                       "High": "#085566",
                        "Medium": "#440866",
                        "Low": "#085566"
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.markdown("#### Confidence Score Distribution")
                fig_hist = px.histogram(
                    df,
                    x="Confidence",
                    color="Risk Level",
                    nbins=20,
                    color_discrete_map={
                       "High": "#085566",
                        "Medium": "#440866",
                        "Low": "#085566"
                    }
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # -----------------------------
            # Row 2: Risk Flow + Keywords
            # -----------------------------
            c3, c4 = st.columns(2)

            with c3:
                st.markdown("#### Risk Intensity Across Document")

                df["Risk_Score"] = df["Risk Level"].map({
                    "High": 3,
                    "Medium": 2,
                    "Low": 1
                })

                fig_scatter = px.scatter(
                    df,
                    x=df.index,
                    y="Risk_Score",
                    color="Risk Level",
                    color_discrete_map={
                        "High": "#085566",
                        "Medium": "#440866",
                        "Low": "#085566"
                    },
                    labels={
                        "index": "Clause Position",
                        "Risk_Score": "Risk Intensity"
                    }
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

            with c4:
                st.markdown("#### Top Keywords (High Risk Clauses)")

                from collections import Counter

                high_text = " ".join(
                    df[df["Risk Level"] == "High"]["Clause Text"].tolist()
                ).lower()

                words = re.findall(r"\b\w+\b", high_text)

                stop_words = {
                    "the","and","to","of","in","a","is","that","for","it",
                    "on","be","as","by","or","this","an","are","with",
                    "from","at","not","will"
                }

                filtered = [w for w in words if w not in stop_words and len(w) > 3]

                common_words = Counter(filtered).most_common(10)

                if common_words:
                    kw_df = pd.DataFrame(common_words, columns=["Keyword", "Count"])

                    fig_bar = px.bar(
                        kw_df,
                        x="Keyword",
                        y="Count",
                        color="Count",
                        color_continuous_scale="Reds"
                    )

                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No high risk clauses detected.")
            # =============================
            # Detailed Clause Review
            # =============================
            st.markdown("---")
            st.markdown("### 🔍 Detailed Clause Review")

            filter_risk = st.multiselect(
                "Filter Risk",
                ["High", "Medium", "Low"],
                default=["High", "Medium"]
            )

            df_filtered = df[df["Risk Level"].isin(filter_risk)]

            for _, row in df_filtered.iterrows():
                risk = row["Risk Level"]
                color_class = risk.lower() + "-risk"
                icon = "🔴" if risk == "High" else "🟠" if risk == "Medium" else "🟢"

                summary_text = get_summary(row["Clause Text"], 100)

                with st.expander(f"{icon} {risk}: {summary_text} ({row['Confidence']:.1%})"):
                    st.markdown(f"""
                    <div class="risk-card {color_class}">
                        <p><strong>Full Clause:</strong></p>
                        <p>{row['Clause Text']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Download CSV
            st.markdown("---")
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Analysis Report (CSV)",
                data=csv,
                file_name="contract_risk_analysis.csv",
                mime="text/csv"
            )

    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px; font-family: 'Inter', sans-serif;">
            <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.4;">📄</div>
            <h3 style="font-weight: 500; color: var(--text-color); opacity: 0.8;">Awaiting document upload</h3>
            <p style="font-size: 1.1rem; color: var(--text-color); opacity: 0.6;">Please upload a contract to begin the analysis.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
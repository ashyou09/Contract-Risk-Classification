import streamlit as st
import pandas as pd
import pickle
import os
import re
import plotly.express as px
from io import StringIO
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
[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.3);
}
.risk-card {
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 5px solid #ddd;
    border: 1px solid rgba(128,128,128,0.25);
}
.high-risk { border-left: 5px solid #d32f2f !important; }
.medium-risk { border-left: 5px solid #f57c00 !important; }
.low-risk { border-left: 5px solid #388e3c !important; }
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

    # Sidebar
    with st.sidebar:
        st.markdown("## 📂 Upload Contract")
        uploaded_file = st.file_uploader("Choose a PDF or TXT", type=["txt", "pdf"])

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, 0.05)

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("⚖️ Intelligent Contract Risk Analysis")
        st.markdown("Identify hidden risks in legal documents instantly.")
    with col2:
        if uploaded_file:
            st.success("Analysis Ready")

    # Load Model
    model = load_model()

    if model is None:
        st.error("⚠️ Model missing! Run training first.")
        return

    if uploaded_file is not None:

        # Read File
        text = ""
        try:
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
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
                        "High": "#d32f2f",
                        "Medium": "#f57c00",
                        "Low": "#388e3c"
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
                        "High": "#d32f2f",
                        "Medium": "#f57c00",
                        "Low": "#388e3c"
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
                        "High": "#d32f2f",
                        "Medium": "#f57c00",
                        "Low": "#388e3c"
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
        <div style="text-align: center; padding: 50px; color: #888;">
            <h3>👋 Welcome to Intelligent Contract Analysis</h3>
            <p>Upload a contract to get started.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
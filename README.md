# Intelligent Contract Risk Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

**Phase: Milestone 1 - ML-Based Contract Risk Classification**

An AI-driven legal document analysis system that evaluates contractual risk. This project applies classical machine learning and Natural Language Processing (NLP) techniques to analyze contract text, segment clauses, and identify potential risk patterns **without the use of Large Language Models (LLMs) or Generative AI**.

---

## Project Overview & Problem Statement

This repository contains the **Milestone 1** implementation of the _Intelligent Contract Risk Analysis_ project.

**Problem Statement:** Legal professionals spend countless hours manually reviewing lengthy contracts to identify risky clauses, liability caps, and termination conditions. This manual process is prone to human error and inefficiency.
**Solution:** Our system automates this process by parsing raw contract documents (PDFs/Text), splitting them into evaluable clauses, and utilizing supervised Machine Learning models to classify each clause's inherent risk (High, Medium, Low).

### Core Declaration

**This project strictly adheres to the "No GenAI" rule.** All parsing, feature engineering (TF-IDF), model training, and logic implementation are explicitly written and engineered by the team using traditional Python data science libraries (Scikit-learn, Pandas, NLTK/SpaCy) without reliance on generative AI outputs for core inference.

## Technical Depth & Sub-Features

Our system incorporates three distinct sub-features beyond basic classification:

1. **Custom Data Pipeline & Preprocessing**: We developed a custom clause segmentation pipeline that processes the complex `CUAD_v1` JSON dataset, extracts relevant clauses, tokenizes text, and maps 41 distinct legal clause types into 3 standardized risk categories for model consumption.
2. **Confidence-Scored Real-Time Inference**: The Streamlit interface performs real-time inference on uploaded documents. It not only classifies text but calculates and displays the model's _probability confidence score_, allowing human operators to set dynamic "Confidence Thresholds".
3. **Interactive Analytical Dashboard**: A comprehensive UI built with Plotly Express that provides a visual executive summary, tracking 'Risk Intensity' across the document's positional timeline, and extracting Top Keywords natively from high-risk clauses.

## Dataset & EDA

**Data Source**: We utilized the **CUAD (Contract Understanding Atticus Dataset)** `CUAD_v1.json`.

- **EDA Insights**: During exploration, we identified 41 highly specific clause types across 510 commercial contracts. We observed severe class imbalances and varying text lengths. This insight informed our decision to map these into overarching 'High/Medium/Low' risk buckets to maintain robust classification boundaries. Our processed dataset yielded balanced training features representing thousands of individual clauses.

## Methodology & Optimisation

1. **Preprocessing**: Handled raw contract text through formatting, lowercasing, stop-word removal, and regex-based punctuation stripping. Paragraphs are segmented into logical individual clauses.
2. **Feature Engineering**: We utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** set to extract up to 5000 maximum features involving unigrams and bigrams (`ngram_range=(1,2)`). This numerical vectorization maps text frequency against corpus ubiquity.
3. **Models Evaluated**:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
4. **Optimisation & Selection**: To prevent data leakage, we employed `Scikit-Learn Pipelines` paired with **5-Fold Stratified Cross-Validation**.
   - _Design Justification_: **Logistic Regression** was ultimately selected as our final production model due to its superior F1-Score (0.8859), efficiency in high-dimensional TF-IDF spaces, and explainability via probabilistic outputs, outperforming tree-based ensemble methods.

## Evaluation Metrics

Based on our 5-Fold Cross Validation on the training data:

- **Logistic Regression (Deployed Model)**:
  - Accuracy: 0.8856
  - F1-Score (Weighted): 0.8859
- **Decision Tree**: F1-Score: 0.8084
- **Random Forest**: F1-Score: 0.8058

## Technology Stack & Tool Deep-Dive

- **UI Framework**: [Streamlit](https://streamlit.io/) (Rapid frontend deployment for Python data apps).
- **Machine Learning**: `Scikit-Learn` (Core library for TF-IDF vectorization, Pipeline creation, classification algorithms, and cross-validation).
- **Data Manipulation**: `Pandas` (Dataframe structuring and CSV I/O) & `NumPy`.
- **Visualizations**: `Plotly Express` (Interactive, web-native charting).
- **PDF Extraction**: `PyPDF2` (Parsing text systematically from raw contract PDFs).
- **Serialization**: `pickle` (Exporting and loading the production pipeline state).

## System Architecture / Code Structure

Data flows from the user interface to the inference engine and back as structured analytical metrics.

```text
Contract-Risk-Classification/
├── CUAD_Dataset/          # Raw CUAD v1 dataset
├── data/                  # Processed datasets and tabular outputs
├── models/
│   └── best_model.pkl     # Deployed ML pipeline (TF-IDF + Logistic Regression)
├── src/                   # Jupyter Notebooks detailing the data pipeline
│   ├── 1_inspect.ipynb
│   ├── 2_data_preprocessing.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_train.ipynb
│   └── 5_evaluate.ipynb
├── app.py                 # Main Streamlit web application & Inference Engine
└── requirements.txt       # Project python dependencies
```

## Installation & Setup

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Contract-Risk-Classification.git
cd Contract-Risk-Classification
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application

The pre-trained model is saved within the `models/` directory.

```bash
streamlit run app.py
```

_Note: A live deployed version link will be generated via HuggingFace Spaces/Streamlit Community Cloud for final submission._

## 👥 Team Contribution

_(Please update these fields with your exact team members and responsibilities before submission)_

- **Member 1 (Ashutosh Singh - 2401010109)**: Developed the custom data preprocessing pipeline, CUAD JSON parsing, and risk mapping logic, Streamlit UI.
- **Member 2 (Ranvendra Pratap Singh - 2401010373)**: Implemented the TF-IDF feature engineering and setup Scikit-learn Pipeline architecture,Streamlit UI.
- **Member 3 (Shreya Suman - 2401010)**: Trained and evaluated ML models (Logistic Regression, Decision Trees), performed Cross-validation,Streamlit UI.
---

**Disclaimer:** This tool is an academic project designed for educational purposes to demonstrate ML-based analysis. It does NOT constitute professional legal advice.
`
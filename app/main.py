import sys
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add root folder to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.data_service import load_dataset, generate_dataset_summary_full
from app.services.ai_service import generate_insights

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="AI Dataset Insight Generator",
    layout="wide"
)

st.title("🧠 AI Dataset Insight Generator (Open Source)")
st.write("Upload a CSV dataset to get structured summaries, visual distributions, and AI-generated insights explained simply.")

# ----------------------------
# File Upload
# ----------------------------
file = st.file_uploader("Upload CSV dataset", type=["csv"])

TOP_N = 10  # show only top 10 values for visuals & AI insights

if file:
    try:
        # Load dataset
        df = load_dataset(file)
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(5))

        # ----------------------------
        # Generate Dataset Summary
        # ----------------------------
        summary = generate_dataset_summary_full(df)

        st.subheader("📋 Dataset Overview")
        st.write(f"Rows: {summary['metadata']['rows']}")
        st.write(f"Columns: {summary['metadata']['columns']}")

        # Missing values
        st.subheader("⚠️ Missing Values")
        missing_df = pd.DataFrame.from_dict(summary['metadata']['missing_values'], orient='index', columns=['Missing Count'])
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])

        # ----------------------------
        # Numeric Summary & Visuals
        # ----------------------------
        st.subheader("🔢 Numeric Summary & Distributions")
        numeric_summary = summary.get('numeric_summary', {})
        numeric_cols = [col for col in numeric_summary.keys() if 'id' not in col.lower()]

        if numeric_cols:
            st.dataframe(pd.DataFrame(numeric_summary)[numeric_cols])

            # Split numeric columns into discrete (categorical-like) and continuous
            discrete_numeric_cols = [col for col in numeric_cols if df[col].nunique() <= TOP_N]
            continuous_numeric_cols = [col for col in numeric_cols if df[col].nunique() > TOP_N]

            # Discrete numeric columns
            for col in discrete_numeric_cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                value_counts = df[col].value_counts().nlargest(TOP_N)
                sns.barplot(x=value_counts.values, y=value_counts.index.astype(str), ax=ax, palette='Blues_r')
                ax.set_title(f"Discrete numeric (treated as categorical): {col}")
                st.pyplot(fig)

            # Continuous numeric columns
            for col in continuous_numeric_cols:
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(df[col], kde=True, ax=ax, color='skyblue', bins=TOP_N*2)
                ax.set_title(f"Continuous numeric: {col}")
                st.pyplot(fig)
        else:
            st.write("No numeric columns detected (or all are ID-like columns).")

        # ----------------------------
        # Correlations
        # ----------------------------
        st.subheader("🔗 Strong Correlations (|r| ≥ 0.7)")
        correlations = summary.get('correlations', {})
        if correlations:
            corr_df = pd.DataFrame(correlations).fillna(0)
            if not corr_df.empty and corr_df.shape[0] > 1 and corr_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
                ax.set_title("Strong Numeric Correlations")
                st.pyplot(fig)
            else:
                st.write("Not enough numeric columns with strong correlations to plot heatmap.")
        else:
            st.write("No strong correlations found.")

        # ----------------------------
        # Categorical Summary & Visuals
        # ----------------------------
        st.subheader("📝 Categorical Summary & Top Values")
        cat_summary = summary.get('categorical_summary', {})
        if cat_summary:
            cat_df = pd.DataFrame([
                {
                    "Column": col,
                    "Unique Values": info["unique_count"],
                    "Top Values": ", ".join([str(k) for k in list(info["top_values"].keys())[:TOP_N]])
                }
                for col, info in cat_summary.items()
            ])
            st.dataframe(cat_df)

            # Bar plots for top categorical values
            for col, info in cat_summary.items():
                top_vals = dict(list(info["top_values"].items())[:TOP_N])
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.barplot(x=list(top_vals.values()), y=list(top_vals.keys()), ax=ax, palette='viridis')
                ax.set_title(f"Top {TOP_N} values for {col}")
                st.pyplot(fig)
        else:
            st.write("No categorical columns detected.")

        # ----------------------------
        # AI Insights
        # ----------------------------
        st.subheader("🤖 AI Insights (Explained Simply)")
        if st.button("Generate AI Insights"):
            with st.spinner("Analyzing dataset with Mistral..."):
                # Pass TOP_N to limit numeric/categorical examples in AI prompt
                insights = generate_insights(summary, top_n=TOP_N)
            st.markdown(insights)

        # ----------------------------
        # Optional: Free-text dataset question
        # ----------------------------
        question = st.text_input("Ask a question about your dataset (optional):")
        if question:
            with st.spinner("Analyzing your question..."):
                response = generate_insights(summary, question, top_n=TOP_N)
            st.subheader("🗨️ Answer")
            st.markdown(response)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
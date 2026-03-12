import sys
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(
    page_title="AI Dataset Insight Generator",
    layout="wide"
)

# Add root folder to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.data_service import load_dataset, generate_dataset_summary_full
from app.services.ai_service import generate_insights

# ----------------------------
# App Title
# ----------------------------
st.title("🧠 AI Dataset Insight Generator")
st.write(
    "Upload a CSV dataset to automatically generate summaries, visualizations, "
    "and AI-powered insights."
)

st.info(
    "⚠️ **AI Insights require a local Ollama server.** "
    "The online demo shows dataset analysis and visualizations. "
    "Run the app locally with Ollama for full AI insights."
)

# ----------------------------
# File Upload
# ----------------------------
file = st.file_uploader("Upload CSV dataset", type=["csv"])
TOP_N = 10

# ----------------------------
# When File Uploaded
# ----------------------------
if file:
    try:
        # Load dataset
        df = load_dataset(file)
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # ----------------------------
        # Dataset Summary
        # ----------------------------
        summary = generate_dataset_summary_full(df)

        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", summary["metadata"]["rows"])
        with col2:
            st.metric("Columns", summary["metadata"]["columns"])

        # ----------------------------
        # Missing Values
        # ----------------------------
        st.subheader("⚠️ Missing Values")
        missing_df = pd.DataFrame.from_dict(
            summary["metadata"]["missing_values"],
            orient="index",
            columns=["Missing Count"]
        )
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        if not missing_df.empty:
            st.dataframe(missing_df)
        else:
            st.success("No missing values detected.")

        # ----------------------------
        # Numeric Analysis
        # ----------------------------
        st.subheader("🔢 Numeric Summary & Distributions")
        numeric_summary = summary.get("numeric_summary", {})
        numeric_cols = [c for c in numeric_summary if "id" not in c.lower()]

        if numeric_cols:
            st.dataframe(pd.DataFrame(numeric_summary)[numeric_cols])

            discrete_cols = [c for c in numeric_cols if df[c].nunique() <= TOP_N]
            continuous_cols = [c for c in numeric_cols if df[c].nunique() > TOP_N]

            # Discrete numeric columns
            for col in discrete_cols:
                fig, ax = plt.subplots(figsize=(6,4))
                counts = df[col].value_counts().nlargest(TOP_N)
                sns.barplot(
                    x=counts.index.astype(str),
                    y=counts.values,
                    ax=ax,
                    palette='Blues_r'
                )
                ax.set_title(f"{col} (Discrete Numeric)", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            # Continuous numeric columns
            for col in continuous_cols:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.histplot(
                    df[col],
                    kde=True,
                    bins=TOP_N*2,
                    ax=ax,
                    color='skyblue'
                )
                ax.set_title(f"{col} Distribution", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)

        else:
            st.write("No numeric columns detected.")

        # ----------------------------
        # Correlations
        # ----------------------------
        st.subheader("🔗 Strong Correlations (|r| ≥ 0.7)")
        correlations = summary.get("correlations", {})
        if correlations:
            corr_df = pd.DataFrame(correlations).fillna(0)
            if corr_df.shape[0] > 1 and corr_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(
                    corr_df,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    ax=ax
                )
                ax.set_title("Strong Numeric Correlations", fontsize=14)
                ax.set_xlabel("Columns", fontsize=12)
                ax.set_ylabel("Columns", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("Not enough correlations to visualize.")
        else:
            st.write("No strong correlations detected.")

        # ----------------------------
        # Categorical Analysis
        # ----------------------------
        st.subheader("📝 Categorical Summary")
        cat_summary = summary.get("categorical_summary", {})
        if cat_summary:
            cat_df = pd.DataFrame([
                {
                    "Column": col,
                    "Unique Values": info["unique_count"],
                    "Top Values": ", ".join(
                        list(map(str, list(info["top_values"].keys())[:TOP_N]))
                    )
                }
                for col, info in cat_summary.items()
            ])
            st.dataframe(cat_df)

            for col, info in cat_summary.items():
                top_vals = dict(list(info["top_values"].items())[:TOP_N])
                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(
                    x=list(top_vals.keys()),
                    y=list(top_vals.values()),
                    ax=ax,
                    palette='viridis'
                )
                ax.set_title(f"{col} (Top {TOP_N} Values)", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.write("No categorical columns detected.")

        # ----------------------------
        # AI Insights
        # ----------------------------
        st.subheader("🤖 AI Insights")
        if st.button("Generate AI Insights"):
            with st.spinner("Generating AI insights..."):
                insights = generate_insights(summary, top_n=TOP_N)
            st.markdown(insights)

        # ----------------------------
        # Ask Questions
        # ----------------------------
        st.subheader("💬 Ask a Question About the Dataset")
        question = st.text_input("Example: What trends exist in this dataset?")
        if question:
            with st.spinner("Analyzing question..."):
                answer = generate_insights(summary, question, top_n=TOP_N)
            st.markdown(answer)

    except Exception as e:
        st.error(f"❌ Error processing dataset: {e}")

else:
    st.info("Upload a CSV file to start exploring your dataset.")
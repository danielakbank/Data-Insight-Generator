# app/services/ai_service.py

try:
    import ollama
except ImportError:
    ollama = None

MODEL = "mistral"
DEFAULT_TOP_N = 5  # reduce prompt size for faster AI generation


def summary_to_prompt(summary: dict, question: str = None, top_n: int = DEFAULT_TOP_N) -> str:
    """
    Convert dataset summary into a structured prompt for the LLM.
    The explanation should be simple and beginner-friendly.
    """

    numeric = summary.get("numeric_summary", {})
    categorical = summary.get("categorical_summary", {})
    metadata = summary.get("metadata", {})
    correlations = summary.get("correlations", {})
    missing = metadata.get("missing_values", {})

    # Remove ID-like numeric columns
    numeric_cols = [col for col in numeric if "id" not in col.lower()]

    prompt = f"""
You are a senior data scientist explaining a dataset to a beginner.

Explain the dataset in simple language (like teaching a 16-year-old).

Structure your answer into these sections:

1. Dataset Overview
2. Numeric Column Insights
3. Categorical Column Insights
4. Strong Correlations
5. Missing Values / Data Quality
6. Interesting Observations
7. Possible Business Insights

Dataset Overview:
Rows: {metadata.get("rows")}
Columns: {metadata.get("columns")}
"""

    # -------- Numeric Columns --------
    if numeric_cols:
        prompt += "\nNumeric Column Insights:\n"
        for col in numeric_cols[:top_n]:
            stats = numeric[col]
            prompt += (
                f"- {col}: "
                f"min={stats.get('min')}, "
                f"max={stats.get('max')}, "
                f"mean={stats.get('mean')}, "
                f"std={stats.get('std')}\n"
            )

    # -------- Categorical Columns --------
    if categorical:
        prompt += "\nCategorical Column Insights:\n"
        for col, info in list(categorical.items())[:top_n]:
            top_vals = list(info.get("top_values", {}).keys())[:top_n]
            top_vals_str = ", ".join(map(str, top_vals))
            prompt += f"- {col}: {info.get('unique_count')} unique values (top: {top_vals_str})\n"

    # -------- Correlations --------
    if correlations:
        prompt += "\nStrong Correlations (|r| ≥ 0.7):\n"
        count = 0
        for col, others in correlations.items():
            for other_col, val in others.items():
                prompt += f"- {col} and {other_col}: r={val}\n"
                count += 1
                if count >= top_n:
                    break
            if count >= top_n:
                break

    # -------- Missing Values --------
    if missing:
        missing_summary = {k: v for k, v in missing.items() if v > 0}
        if missing_summary:
            prompt += "\nMissing Values:\n"
            for col, val in list(missing_summary.items())[:top_n]:
                prompt += f"- {col}: {val} missing values\n"

    prompt += """
Explain insights clearly using bullet points where helpful.
Focus only on the most important findings.
Keep the explanation concise.
"""

    if question:
        prompt += f"\nUser Question: {question}"

    return prompt


def generate_insights(summary: dict, question: str = None, top_n: int = DEFAULT_TOP_N):
    """
    Generate AI insights using Ollama.
    If Ollama is unavailable (e.g., Streamlit Cloud), return a helpful message.
    """

    if ollama is None:
        return """
⚠️ **AI insights are available only in the local version of this app.**

This online demo still provides:

• Dataset summary  
• Visualizations  
• Correlation analysis  

To enable AI insights locally:

1. Install **Ollama**
2. Run `ollama pull mistral`
3. Start Ollama and run the Streamlit app locally.
"""

    prompt = summary_to_prompt(summary, question, top_n)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 350,   # faster generation
                "temperature": 0.2,
                "top_p": 0.9
            }
        )

        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ AI generation error: {e}"
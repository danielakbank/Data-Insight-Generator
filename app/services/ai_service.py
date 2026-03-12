import ollama

MODEL = "mistral"  # Ollama local model
DEFAULT_TOP_N = 10  # Limit numeric/categorical/strong correlation examples

def summary_to_prompt(summary: dict, question: str = None, top_n: int = DEFAULT_TOP_N) -> str:
    """
    Build a structured prompt for Mistral AI:
    - Teen-friendly explanations
    - Subtopic-separated for clarity
    - Only include top_n numeric/categorical/correlation examples
    """
    numeric = summary.get("numeric_summary", {})
    cat = summary.get("categorical_summary", {})
    metadata = summary.get("metadata", {})
    correlations = summary.get("correlations", {})
    missing = metadata.get("missing_values", {})

    # Filter out ID-like numeric columns
    numeric_cols = [col for col in numeric.keys() if "id" not in col.lower()]

    prompt = f"""
You are a senior data scientist. Analyze this dataset and explain it in simple, easy-to-understand terms, like teaching a 16-year-old. Structure your explanation into the following subtopics:

1. Dataset Overview
2. Numeric Column Insights
3. Categorical Column Insights
4. Strong Correlations
5. Missing Values / Data Quality Issues
6. Interesting Observations
7. Business Insights / Suggestions for further analysis

Dataset Overview:
- Rows: {metadata.get('rows')}
- Columns: {metadata.get('columns')}
"""

    # Numeric summary (top N columns only)
    if numeric_cols:
        prompt += "\nNumeric Column Insights (showing examples of min, max, mean, std):\n"
        for col in numeric_cols[:top_n]:
            stats = numeric[col]
            prompt += f"- {col}: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, std={stats.get('std')}\n"

    # Categorical summary (top N values only)
    if cat:
        prompt += "\nCategorical Column Insights (top values):\n"
        for col, info in cat.items():
            top_vals = list(info.get("top_values", {}).keys())[:top_n]
            top_vals_str = ", ".join([str(v) for v in top_vals])
            prompt += f"- {col}: {info.get('unique_count')} unique values, top values = {top_vals_str}\n"

    # Strong correlations (top N only per column)
    if correlations:
        prompt += "\nStrong Correlations (|r| >= 0.7, showing top examples):\n"
        for col, others in correlations.items():
            for i, (other_col, val) in enumerate(others.items()):
                if i >= top_n:
                    break
                prompt += f"- {col} and {other_col}: correlation = {val}\n"

    # Missing values
    if missing:
        missing_summary = {k: v for k, v in missing.items() if v > 0}
        if missing_summary:
            prompt += "\nMissing Values / Data Quality Issues:\n"
            for col, val in missing_summary.items():
                prompt += f"- {col}: {val} missing entries\n"

    prompt += """
Explain each subtopic clearly and simply, using bullet points if necessary. 
Keep it concise, easy to read, and focus on insights that are most important for understanding the dataset.
"""

    if question:
        prompt += f"\nUser Question: {question}"

    return prompt


def generate_insights(summary: dict, question: str = None, top_n: int = DEFAULT_TOP_N):
    """
    Generate structured AI insights using Mistral.
    Returns: string explanation divided into subtopics.
    """
    prompt = summary_to_prompt(summary, question, top_n=top_n)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 800,   # increased from 200 → allows full dataset explanation
                "temperature": 0.2,
                "top_p": 0.9
            }
        )
        return response["message"]["content"]

    except Exception as e:
        return f"Ollama AI Error: {e}"
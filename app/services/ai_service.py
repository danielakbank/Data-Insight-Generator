# app/services/ai_service.py

import ollama

MODEL = "mistral"  # Ollama local model

def summary_to_prompt(summary: dict, question: str = None) -> str:
    """
    Build a structured prompt for Mistral AI:
    - Teen-friendly explanations
    - Separate subtopics for clarity
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

    # Numeric summary
    if numeric_cols:
        prompt += "\nNumeric Column Insights (showing examples of min, max, mean, std):\n"
        for col in numeric_cols[:10]:
            stats = numeric[col]
            prompt += f"- {col}: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean')}, std={stats.get('std')}\n"

    # Categorical summary
    if cat:
        prompt += "\nCategorical Column Insights (top values):\n"
        for col, info in cat.items():
            top_vals = ", ".join([str(v) for v in info.get("top_values", {}).keys()])
            prompt += f"- {col}: {info.get('unique_count')} unique values, top values = {top_vals}\n"

    # Strong correlations
    if correlations:
        prompt += "\nStrong Correlations (|r| >= 0.7):\n"
        for col, others in correlations.items():
            for other_col, val in others.items():
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


def generate_insights(summary: dict, question: str = None):
    """
    Generate structured AI insights using Mistral.
    Returns: string explanation divided into subtopics.
    """
    prompt = summary_to_prompt(summary, question)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 200,   # slightly longer for full subtopic coverage
                "temperature": 0.2,
                "top_p": 0.9
            }
        )
        return response["message"]["content"]

    except Exception as e:
        return f"Ollama AI Error: {e}"
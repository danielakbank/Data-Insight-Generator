# 🧠 AI Dataset Insight Generator

A **Streamlit app** that generates structured dataset summaries, visualizations, and AI-driven insights explained simply. Powered by **Mistral (Ollama local AI)** for fast, human-friendly analysis.

---

## Features

- Upload CSV datasets and get:
  - Numeric and categorical summaries
  - Missing value analysis
  - Visual distributions (histograms, bar plots)
  - Strong correlations heatmap (|r| ≥ 0.7)
- AI insights explained in **simple language** for easy understanding
- Optional **free-text questions** about your dataset
- Filters out ID-like numeric columns automatically
---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/<USERNAME>/ai-dataset-insight-generator.git
cd ai-dataset-insight-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run app/main.py
```

---

## Deployment

- Deploy easily on **[Streamlit Community Cloud](https://share.streamlit.io/)**.  
- Make sure the repo is public or private with Streamlit access.  
- Set `app/main.py` as the main file.

---

## Folder Structure

```
AI Data Insight Generator App/
├── app/
│   ├── main.py
│   └── services/
│       ├── ai_service.py
│       └── data_service.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Usage

1. Upload a CSV file.
2. View dataset preview and numeric/categorical summaries.
3. Inspect distributions and correlations visually.
4. Click **Generate AI Insights** to get explanations in simple terms.
5. Ask dataset-specific questions in the text box.

---

## Requirements

- Python 3.10+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Ollama (local AI)

---

## License

MIT License


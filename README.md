
# 🧠 AI Dataset Insight Generator

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-success)](https://data-insight-generator-bfktkwkcrhcf5hmcciziam.streamlit.app/)

An open-source Streamlit application that automatically analyzes CSV datasets and generates structured summaries, visualizations, and AI-powered insights explained in simple language.

The tool helps users quickly understand datasets without writing complex analysis code.

AI insights are powered by Mistral running locally via Ollama, enabling fast and private AI analysis without external API costs.

---

## 🌐 Live Demo

Try the app online:

https://data-insight-generator-bfktkwkcrhcf5hmcciziam.streamlit.app/

⚠️ Note

The cloud demo currently supports:

- Dataset upload
- Data preview
- Automated visualizations
- Dataset summaries

The AI Insights feature works only when running the app locally because it requires Ollama with the Mistral model running on your machine.

---

## 🚀 Demo Workflow

Upload CSV → Preview Data → Visualizations → AI Insights → Ask Questions

---

## ✨ Features

## 📊 Automated Dataset Analysis

Upload any CSV dataset and automatically generate:

- Dataset overview (rows and columns)
- Missing value analysis
- Numeric column statistics
- Categorical column summaries

---

## 📈 Automatic Visualizations

The application generates visual insights automatically:

- Histogram distributions for numeric columns
- Bar charts for categorical variables
- Strong correlation heatmap (|r| ≥ 0.7)

All charts include clear titles and labeled axes for readability.

---

## 🤖 AI-Powered Insights

Using Mistral via Ollama, the app can:

- Explain dataset trends in plain English
- Identify patterns and correlations
- Summarize key findings

---

## 💬 Interactive Dataset Q&A

Users can ask questions about the dataset such as:

"What trends exist in this dataset?"

"Which features have strong correlations?"

The AI will generate a response based on the dataset summary.

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Ollama (Local AI Runtime)
- Mistral LLM

---

## ⚙️ Installation

#### 1️⃣ Clone the repository

git clone https://github.com/danielakbank/ai-dataset-insight-generator.git

cd ai-dataset-insight-generator

---

#### 2️⃣ Install dependencies

pip install -r requirements.txt

---

#### 3️⃣ Install Ollama

Download Ollama:

https://ollama.com/download

---

#### 4️⃣ Pull the Mistral model

ollama pull mistral

---

#### 5️⃣ Run the application

streamlit run app/main.py

The application will open in your browser.

---

## ☁️ Deployment

This app can be deployed using Streamlit Community Cloud.

https://share.streamlit.io/

Steps:

1. Push the project to GitHub
2. Create a new Streamlit Cloud app
3. Set the entry file as:

app/main.py

⚠️ Note:

AI insights require Ollama running locally, so the Streamlit cloud demo only supports dataset analysis and visualizations.

---

## 📁 Project Structure

**ai-dataset-insight-generator/**
- **app/**
  - `main.py`
  - **services/**
    - `data_service.py`
    - `ai_service.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

---

## 📊 Example Use Cases

This project can be used for:

- Exploratory Data Analysis (EDA)
- Dataset understanding
- Data science learning
- AI-assisted data interpretation
- Quick dataset inspection

---

## 🔮 Future Improvements

Possible enhancements:

- Automated dataset profiling reports
- Feature importance detection
- Anomaly detection
- Exportable insight reports
- Support for additional AI models

---

## 📜 License

MIT License

You are free to use, modify, and distribute this project.

---

## ⭐ Support the Project

If you found this project useful:

- Star the repository
- Contribute improvements
- Share feedback

# ⚽ Saudi Pro League Analyzer

A machine learning project that predicts Saudi Pro League match outcomes using real season statistics — built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This project analyzes the **2025/26 Saudi Pro League season** across 18 teams and 105 matches. It predicts whether a match will end in a Home Win, Draw, or Away Win — using only pre-match team statistics (no score leakage).

The model achieves **52% accuracy**, outperforming a random baseline of 33% by 57%.

> "Draws remain the hardest outcome to predict in football analytics — consistent with professional sports prediction literature."

---

## 📊 Key Findings

- 🏠 **Home teams win 45%** of Saudi Pro League matches
- ✈️ **Away teams win 36%** — unusually high compared to European leagues  
- 🤝 **Draws account for 19%** of results
- 🥇 **Top 3 teams:** Al Ittihad · Al Hilal · Al Nassr
- 📈 Al Hilal leads in goals scored (95) across the season

---

## 🖥️ Live Demo

👉 **[Try the app here](https://your-app.streamlit.app)** ← 

The app includes:
- 🔮 **Match Predictor** — select any two teams and get Win/Draw/Loss probabilities
- 📊 **Team Stats** — explore individual team performance with charts
- 📈 **League Overview** — standings, result distribution, top scorers

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and cleaning |
| `scikit-learn` | Random Forest classifier |
| `matplotlib` / `seaborn` | Data visualization |
| `streamlit` | Interactive web app |

---

## 📁 Project Structure

```
saudi-pro-league-analyzer/
│
├── data/
│   └── spl_matches.csv          ← season data (standings + results)
│
├── notebooks/
│   ├── 01_data_collection.ipynb ← data scraping
│   ├── 02_eda.ipynb             ← exploratory analysis
│   └── 03_model.ipynb           ← model building
│
├── src/
│   └── predict.py               ← standalone prediction script
│
├── app.py                       ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/saudi-pro-league-analyzer.git
cd saudi-pro-league-analyzer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the prediction script**
```bash
python src/predict.py
```

**4. Launch the web app**
```bash
streamlit run app.py
```

---

## 🤖 Model Details

- **Algorithm:** Random Forest Classifier (100 trees)
- **Features used:** Team points, win percentage, average goals scored/conceded (home & away)
- **Train/Test split:** 80/20
- **Accuracy:** 52% — beats random chance (33%) by 57%

### Why not higher accuracy?
Football is inherently unpredictable. The best models in the world top out at ~60-65%. This model uses season-level aggregates only — adding match-by-match form, head-to-head history, and player availability would improve it further.

---

## 🔮 Future Improvements

- [ ] Add rolling form (last 5 matches) as a feature
- [ ] Use home/away specific stats instead of overall
- [ ] Add head-to-head historical data
- [ ] Deploy with automatic weekly data updates

---

## 👤 Author

**Abdulmajeed Salmeen** — Data Science enthusiast based in Jeddah 🇸🇦

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/abdulmajeed-salmeen/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/AbdulmajeedSalmeen)

---

## 📄 License

MIT License — feel free to use, modify, and share.

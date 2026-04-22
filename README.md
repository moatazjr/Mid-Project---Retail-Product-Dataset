# 🛒 E-Commerce Product Data — Mid-Year Data Science Project

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Project Overview

This project fulfils the **Mid-Year Data Science** requirements:

| Requirement | Status |
|---|---|
| Dataset gathered & assessed | ✅ |
| Data quality documented | ✅ |
| Data cleaning pipeline | ✅ |
| EDA with 6+ variables | ✅ |
| 5+ plot types | ✅ |
| Jupyter Notebook | ✅ |
| Streamlit Dashboard | ✅ |
| Clean CSV exported | ✅ |

---

## 🎯 Research Questions

1. Which product category commands the highest prices?
2. How does customer rating vary across price tiers?
3. Does discount percentage impact stock availability?
4. What is the correlation between Price, Rating, and Discount?
5. Which category offers the best value (high rating + high discount)?

---

## 📂 Project Structure

```
midyear_project/
│
├── data/
│   ├── synthetic_dataset.csv      ← Raw (dirty) dataset
│   └── clean_dataset.csv          ← Cleaned dataset (output)
│
├── notebooks/
│   └── midyear_project.ipynb      ← Full Jupyter Notebook
│
├── app/
│   └── streamlit_app.py           ← Interactive Streamlit dashboard
│
├── assets/                        ← Saved chart images
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| Attribute | Value |
|---|---|
| **Rows** | 4,362 (raw) → 4,347 (clean) |
| **Columns** | 5 (raw) → 8 (clean, +3 derived) |
| **Domain** | E-Commerce / Retail |
| **Source** | Synthetic (simulates real-world retail data) |

### Columns

| Column | Type | Description | Issues Found |
|---|---|---|---|
| `Category` | String | Product category (A/B/C/D) | 63% missing, abbreviations |
| `Price` | Float | Product price ($) | 4% missing |
| `Rating` | Float | Customer rating (1–5) | 47% missing |
| `Stock` | String | In Stock / Out of Stock | 31% missing |
| `Discount` | Float | Discount percentage | 9% missing |

### Why This Dataset is NOT Clean
- Over **6,700+ total missing values** across all columns
- Category column uses **single-letter codes** (A, B, C, D) — not human-readable
- **15 duplicate rows** present
- Multiple columns have **>30% missingness** — requiring careful imputation strategy

---

## 🧹 Cleaning Steps

1. Removed **15 duplicate rows**
2. Expanded category codes → readable names (A→Electronics, B→Clothing, C→Home & Garden, D→Sports)
3. Filled missing `Category` with **mode**
4. Filled missing `Price` with **per-category median**
5. Filled missing `Rating` with **per-category median**
6. Filled missing `Stock` with **mode**
7. Filled missing `Discount` with **per-category median**
8. Rounded numeric columns to 2 decimal places
9. Engineered 3 new features: `Final_Price`, `Price_Category`, `Rating_Label`

---

## 📈 Visualizations Included

| Plot Type | Used For |
|---|---|
| **Pie chart** | Category distribution |
| **Bar chart** | Stock status, avg price/rating/discount by category |
| **Histogram** | Price, Rating, Discount distributions |
| **Scatter plot** | Price vs. Rating (with regression line) |
| **Box plot** | Discount by Stock Status |
| **Violin plot** | Rating by Price Tier |
| **Heatmap** | Correlation matrix, Rating × Category × Price Tier |
| **Grouped bar** | Final Price by Category & Stock |
| **3D Scatter** | Price × Rating × Discount (Plotly) |
| **Pair plot** | All numeric variables |

---

## 🚀 How to Run

### Option A — Streamlit Dashboard

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/midyear-ds-project.git
cd midyear-ds-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
cd app
streamlit run streamlit_app.py
```

### Option B — Jupyter Notebook

```bash
cd notebooks
jupyter notebook midyear_project.ipynb
```

---

## 💡 Key Findings

- **Price does NOT predict Rating** — Pearson r ≈ 0.00 (customers rate all prices similarly)
- **Electronics** commands the highest average price (~$5,100)
- **Sports** enjoys the highest average discounts (~25.5%)
- **Discounting doesn't clear stock** — Out-of-stock products are actually discounted *more*
- Rating is uniformly distributed (1–5) across ALL category and price-tier combinations

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas** — data manipulation & cleaning
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — static visualizations
- **Plotly** — interactive charts
- **SciPy** — statistical tests (Pearson correlation)
- **Streamlit** — interactive web dashboard
- **Jupyter Notebook** — documented analysis

---

## 📎 References

- Main Epsilon AI Repo: https://github.com/epsilon-ai
- Dataset: Synthetic E-Commerce Dataset (provided)

---

*Project submitted as part of the Mid-Year Data Science curriculum.*

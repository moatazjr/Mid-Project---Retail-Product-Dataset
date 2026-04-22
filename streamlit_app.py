"""
=============================================================================
Mid-Year Data Science Project — Streamlit Dashboard
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Data Science Project",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #764ba2;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 0.8rem;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.93rem;
    }
    .step-box {
        background: #fff8f0;
        border-left: 4px solid #f6a623;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.92rem;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #667eea33;
        border-radius: 10px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    """Load the original (dirty) dataset."""
    try:
        return pd.read_csv("data/synthetic_dataset.csv")
    except FileNotFoundError:
        return pd.read_csv("../data/synthetic_dataset.csv")


@st.cache_data
def clean_data(df_raw):
    """
    Full data cleaning pipeline.
    Returns: cleaned DataFrame and a list of step descriptions.
    """
    df = df_raw.copy()
    steps = []

    # 1. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    steps.append(f"✅ Removed **{removed}** duplicate rows. Rows: {before} → {len(df)}")

    # 2. Expand category abbreviations → readable labels
    cat_map = {"A": "Electronics", "B": "Clothing", "C": "Home & Garden", "D": "Sports"}
    df["Category"] = df["Category"].map(cat_map)
    steps.append("✅ Expanded Category codes (A→Electronics, B→Clothing, C→Home & Garden, D→Sports)")

    # 3. Fill missing Category with mode
    mode_cat = df["Category"].mode()[0]
    missing_cat = df["Category"].isna().sum()
    df["Category"] = df["Category"].fillna(mode_cat)
    steps.append(f"✅ Filled **{missing_cat}** missing Category values with mode → '{mode_cat}'")

    # 4. Fill missing Price with per-category median
    missing_price = df["Price"].isna().sum()
    df["Price"] = df.groupby("Category")["Price"].transform(lambda x: x.fillna(x.median()))
    steps.append(f"✅ Filled **{missing_price}** missing Price values with per-category median")

    # 5. Fill missing Rating with per-category median
    missing_rating = df["Rating"].isna().sum()
    df["Rating"] = df.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.median()))
    steps.append(f"✅ Filled **{missing_rating}** missing Rating values with per-category median")

    # 6. Fill missing Stock with mode
    missing_stock = df["Stock"].isna().sum()
    mode_stock = df["Stock"].mode()[0]
    df["Stock"] = df["Stock"].fillna(mode_stock)
    steps.append(f"✅ Filled **{missing_stock}** missing Stock values with mode → '{mode_stock}'")

    # 7. Fill missing Discount with per-category median
    missing_disc = df["Discount"].isna().sum()
    df["Discount"] = df.groupby("Category")["Discount"].transform(lambda x: x.fillna(x.median()))
    steps.append(f"✅ Filled **{missing_disc}** missing Discount values with per-category median")

    # 8. Round numerical columns for readability
    df["Price"] = df["Price"].round(2)
    df["Rating"] = df["Rating"].round(2)
    df["Discount"] = df["Discount"].round(2)
    steps.append("✅ Rounded Price, Rating, and Discount to 2 decimal places")

    # 9. Derive new features
    df["Final_Price"] = (df["Price"] * (1 - df["Discount"] / 100)).round(2)
    df["Price_Category"] = pd.cut(
        df["Price"],
        bins=[0, 2500, 5000, 7500, 10000],
        labels=["Budget", "Mid-Range", "Premium", "Luxury"],
    )
    df["Rating_Label"] = pd.cut(
        df["Rating"],
        bins=[0, 2, 3, 4, 5],
        labels=["Poor", "Average", "Good", "Excellent"],
    )
    steps.append("✅ Added **3 derived features**: Final_Price, Price_Category, Rating_Label")

    return df, steps


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=80)
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to section:",
        [
            "🏠 Overview",
            "🔍 Data Quality",
            "🧹 Data Cleaning",
            "📊 Univariate Analysis",
            "🔗 Bivariate Analysis",
            "📈 Multivariate Analysis",
            "💡 Insights & Conclusions",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🎛️ Filters (Clean Data)")

# Load data
df_raw = load_raw_data()
df_clean, cleaning_steps = clean_data(df_raw)

# Sidebar filters (applied globally to clean data)
with st.sidebar:
    selected_cats = st.multiselect(
        "Category",
        df_clean["Category"].unique(),
        default=df_clean["Category"].unique(),
    )
    price_range = st.slider(
        "Price Range",
        float(df_clean["Price"].min()),
        float(df_clean["Price"].max()),
        (float(df_clean["Price"].min()), float(df_clean["Price"].max())),
    )
    stock_filter = st.multiselect(
        "Stock Status",
        df_clean["Stock"].unique(),
        default=df_clean["Stock"].unique(),
    )

df_filtered = df_clean[
    (df_clean["Category"].isin(selected_cats))
    & (df_clean["Price"] >= price_range[0])
    & (df_clean["Price"] <= price_range[1])
    & (df_clean["Stock"].isin(stock_filter))
]

PALETTE = ["#667eea", "#764ba2", "#f6a623", "#2dcca7", "#ff6b6b"]

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🛒 E-Commerce Product Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Mid-Year Data Science Project** | Data Wrangling · EDA · Visualization")
    st.divider()

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Products", f"{len(df_clean):,}")
    c2.metric("Categories", df_clean["Category"].nunique())
    c3.metric("Avg Price", f"${df_clean['Price'].mean():,.0f}")
    c4.metric("Avg Rating", f"{df_clean['Rating'].mean():.2f} ⭐")
    c5.metric("Avg Discount", f"{df_clean['Discount'].mean():.1f}%")

    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🎯 Research Questions")
        st.markdown("""
        1. **Which product category commands the highest prices?**
        2. **How does customer rating vary across price tiers?**
        3. **Does discount percentage impact stock availability?**
        4. **What is the correlation between Price, Rating, and Discount?**
        5. **Which category offers the best value (high rating + high discount)?**
        """)

    with col_right:
        st.markdown("### 📋 Dataset Summary")
        info = pd.DataFrame({
            "Attribute": ["Original Rows", "Columns", "Missing Values", "Duplicate Rows", "Clean Rows", "New Features"],
            "Value": [
                f"{len(df_raw):,}",
                str(df_raw.shape[1]),
                f"{df_raw.isnull().sum().sum():,}",
                str(df_raw.duplicated().sum()),
                f"{len(df_clean):,}",
                "3 (Final_Price, Price_Category, Rating_Label)",
            ],
        })
        st.dataframe(info, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📂 Clean Dataset Preview")
    st.dataframe(df_filtered.head(20), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DATA QUALITY
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Data Quality":
    st.markdown('<div class="section-header">🔍 Data Quality Assessment</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(7, 4))
        missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).reset_index()
        missing_pct.columns = ["Column", "Missing %"]
        bars = ax.barh(missing_pct["Column"], missing_pct["Missing %"], color=PALETTE)
        for bar, val in zip(bars, missing_pct["Missing %"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Data per Column", fontweight="bold")
        ax.set_xlim(0, 80)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Missing Value Summary Table")
        mv = pd.DataFrame({
            "Column": df_raw.columns,
            "Missing Count": df_raw.isnull().sum().values,
            "Missing %": (df_raw.isnull().sum() / len(df_raw) * 100).round(1).values,
            "Data Type": df_raw.dtypes.values,
        })
        st.dataframe(mv, use_container_width=True, hide_index=True)

        st.markdown("#### Duplicate Rows")
        dup_count = df_raw.duplicated().sum()
        st.info(f"🔁 Found **{dup_count}** duplicate rows ({dup_count/len(df_raw)*100:.2f}% of dataset)")

    st.markdown("---")
    st.markdown("#### Raw Dataset — First 10 Rows (Unclean)")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("#### Issues Identified")
    issues = [
        ("Category", "~63% missing; uses abbreviations (A, B, C, D) instead of readable labels"),
        ("Price", "4% missing; no outlier capping needed (102–9999 is a valid range)"),
        ("Rating", "~47% missing; non-integer float values"),
        ("Stock", "~31% missing; only 2 valid string categories"),
        ("Discount", "~9% missing"),
        ("Duplicates", "15 duplicate rows identified and need removal"),
    ]
    for col, issue in issues:
        st.markdown(f'<div class="step-box">⚠️ <b>{col}</b>: {issue}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: DATA CLEANING
# ─────────────────────────────────────────────────────────────
elif page == "🧹 Data Cleaning":
    st.markdown('<div class="section-header">🧹 Data Cleaning Pipeline</div>', unsafe_allow_html=True)

    st.markdown("#### Step-by-Step Cleaning Process")
    for step in cleaning_steps:
        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Before Cleaning — Missing Values")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df_raw.isnull().sum().plot(kind="bar", ax=ax, color=PALETTE[4], edgecolor="white")
        ax.set_title("Raw Data — Missing Count per Column", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### After Cleaning — Missing Values")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df_clean.isnull().sum().plot(kind="bar", ax=ax, color=PALETTE[2], edgecolor="white")
        ax.set_title("Clean Data — Missing Count per Column", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Clean Dataset Info")
    info_df = pd.DataFrame({
        "Column": df_clean.columns,
        "Non-Null": df_clean.notnull().sum().values,
        "Dtype": df_clean.dtypes.values,
        "Unique Values": [df_clean[c].nunique() for c in df_clean.columns],
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# PAGE: UNIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "📊 Univariate Analysis":
    st.markdown('<div class="section-header">📊 Univariate Analysis (1D)</div>', unsafe_allow_html=True)
    st.caption("Exploring each variable individually.")

    # Category distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Product Category Distribution")
        cat_counts = df_filtered["Category"].value_counts()
        fig = px.pie(values=cat_counts.values, names=cat_counts.index,
                     color_discrete_sequence=PALETTE, hole=0.4)
        fig.update_layout(margin=dict(t=30, b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 The four categories are fairly balanced, with <b>Electronics</b> being the largest segment.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("##### Stock Status Distribution")
        stock_counts = df_filtered["Stock"].value_counts()
        fig = px.bar(x=stock_counts.index, y=stock_counts.values,
                     color=stock_counts.index, color_discrete_sequence=[PALETTE[0], PALETTE[4]],
                     labels={"x": "Status", "y": "Count"})
        fig.update_layout(showlegend=False, margin=dict(t=30, b=0), height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Stock is nearly evenly split between In-Stock and Out-of-Stock products.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_filtered["Price"], bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
        ax.axvline(df_filtered["Price"].mean(), color=PALETTE[4], linestyle="--", linewidth=1.8, label=f"Mean: ${df_filtered['Price'].mean():,.0f}")
        ax.axvline(df_filtered["Price"].median(), color=PALETTE[2], linestyle="--", linewidth=1.8, label=f"Median: ${df_filtered['Price'].median():,.0f}")
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Count")
        ax.set_title("Price Distribution", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">💡 Price is <b>roughly uniform</b> across the $100–$10,000 range — suggesting diverse product tiers.</div>', unsafe_allow_html=True)

    with col4:
        st.markdown("##### Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df_filtered["Rating"], bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
        ax.axvline(df_filtered["Rating"].mean(), color=PALETTE[4], linestyle="--", linewidth=1.8, label=f"Mean: {df_filtered['Rating'].mean():.2f}")
        ax.set_xlabel("Rating (1–5)")
        ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">💡 Ratings are <b>uniformly distributed</b> between 1 and 5 — an interesting characteristic of this synthetic dataset.</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("##### Discount Distribution")
        fig = px.histogram(df_filtered, x="Discount", nbins=30,
                           color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(margin=dict(t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("##### Price Category Breakdown")
        pc = df_filtered["Price_Category"].value_counts().sort_index()
        fig = px.bar(x=pc.index.astype(str), y=pc.values,
                     color=pc.index.astype(str), color_discrete_sequence=PALETTE,
                     labels={"x": "Tier", "y": "Count"})
        fig.update_layout(showlegend=False, margin=dict(t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Descriptive Statistics")
    st.dataframe(df_filtered[["Price", "Rating", "Discount", "Final_Price"]].describe().round(2),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────
# PAGE: BIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "🔗 Bivariate Analysis":
    st.markdown('<div class="section-header">🔗 Bivariate Analysis (2D)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Average Price by Category")
        avg_price = df_filtered.groupby("Category")["Price"].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_price.index, y=avg_price.values,
                     color=avg_price.index, color_discrete_sequence=PALETTE,
                     labels={"x": "Category", "y": "Avg Price ($)"})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Electronics</b> commands the highest average price, while <b>Clothing</b> is most affordable.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("##### 2. Average Rating by Category")
        avg_rating = df_filtered.groupby("Category")["Rating"].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_rating.index, y=avg_rating.values,
                     color=avg_rating.index, color_discrete_sequence=PALETTE,
                     labels={"x": "Category", "y": "Avg Rating"})
        fig.update_layout(showlegend=False, height=320, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 All categories have similar average ratings (~3.0), indicating <b>no category-based rating bias</b>.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 3. Price vs. Rating (Scatter)")
        fig = px.scatter(df_filtered.sample(min(1000, len(df_filtered))),
                         x="Price", y="Rating", color="Category",
                         color_discrete_sequence=PALETTE, opacity=0.6,
                         trendline="ols")
        fig.update_layout(height=340, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        corr, pval = stats.pearsonr(df_filtered["Price"].dropna(), df_filtered["Rating"].dropna())
        st.markdown(f'<div class="insight-box">💡 Pearson r = <b>{corr:.3f}</b> (p = {pval:.3f}) — essentially <b>no linear correlation</b> between Price and Rating.</div>', unsafe_allow_html=True)

    with col4:
        st.markdown("##### 4. Discount by Stock Status")
        fig = px.box(df_filtered, x="Stock", y="Discount", color="Stock",
                     color_discrete_sequence=[PALETTE[0], PALETTE[4]])
        fig.update_layout(showlegend=False, height=340, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 Out-of-Stock products have a <b>slightly higher median discount</b>, suggesting heavy discounting doesn\'t always clear inventory.</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("##### 5. Rating by Price Category (Violin)")
        fig, ax = plt.subplots(figsize=(6, 4))
        order = ["Budget", "Mid-Range", "Premium", "Luxury"]
        valid = df_filtered[df_filtered["Price_Category"].notna()]
        sns.violinplot(data=valid, x="Price_Category", y="Rating",
                       order=order, palette=PALETTE[:4], ax=ax, inner="quartile")
        ax.set_title("Rating Distribution by Price Tier", fontweight="bold")
        ax.set_xlabel("Price Tier")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">💡 Rating spread is consistent across price tiers — customers rate cheap and expensive products similarly.</div>', unsafe_allow_html=True)

    with col6:
        st.markdown("##### 6. Avg Discount by Category")
        avg_disc = df_filtered.groupby("Category")["Discount"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(avg_disc.index, avg_disc.values, color=PALETTE[:4])
        for bar, val in zip(bars, avg_disc.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Avg Discount %")
        ax.set_title("Average Discount by Category", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">💡 <b>Sports</b> products enjoy the highest average discounts, while <b>Electronics</b> have the lowest.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: MULTIVARIATE ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "📈 Multivariate Analysis":
    st.markdown('<div class="section-header">📈 Multivariate Analysis (3D+)</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("##### Correlation Matrix")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        corr_matrix = df_filtered[["Price", "Rating", "Discount", "Final_Price"]].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlBu_r",
                    mask=mask, ax=ax, linewidths=0.5,
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title("Correlation Heatmap", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("##### Key Correlations")
        for pair, corr_val, note in [
            ("Price ↔ Final Price", corr_matrix.loc["Price", "Final_Price"], "Strong positive — expected"),
            ("Price ↔ Rating", corr_matrix.loc["Price", "Rating"], "Negligible — price ≠ quality"),
            ("Discount ↔ Final Price", corr_matrix.loc["Discount", "Final_Price"], "Negative — discounts reduce final price"),
            ("Rating ↔ Discount", corr_matrix.loc["Rating", "Discount"], "Negligible — ratings are independent"),
        ]:
            color = "🟢" if abs(corr_val) > 0.3 else "🟡" if abs(corr_val) > 0.1 else "🔴"
            st.markdown(f'<div class="insight-box">{color} <b>{pair}</b>: r = {corr_val:.3f}<br><small>{note}</small></div>', unsafe_allow_html=True)

    st.divider()

    # 3D scatter
    st.markdown("##### 3D Scatter: Price × Rating × Discount (by Category)")
    sample = df_filtered.sample(min(1500, len(df_filtered)))
    fig3d = px.scatter_3d(sample, x="Price", y="Rating", z="Discount",
                           color="Category", color_discrete_sequence=PALETTE,
                           opacity=0.65, height=500)
    fig3d.update_traces(marker_size=3)
    fig3d.update_layout(margin=dict(t=30, b=0))
    st.plotly_chart(fig3d, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Avg Final Price: Category × Stock")
        pivot = df_filtered.pivot_table(values="Final_Price", index="Category",
                                         columns="Stock", aggfunc="mean").round(1)
        fig = px.bar(pivot.reset_index().melt(id_vars="Category"),
                     x="Category", y="value", color="Stock",
                     barmode="group", color_discrete_sequence=[PALETTE[0], PALETTE[4]],
                     labels={"value": "Avg Final Price ($)"})
        fig.update_layout(height=350, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("##### Pair Plot: Price, Rating, Discount")
        fig, axes = plt.subplots(3, 3, figsize=(7, 6))
        cols = ["Price", "Rating", "Discount"]
        colors_cat = {"Electronics": PALETTE[0], "Clothing": PALETTE[1],
                      "Home & Garden": PALETTE[2], "Sports": PALETTE[3]}
        sample_pp = df_filtered.sample(min(800, len(df_filtered)))
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                ax = axes[i][j]
                if i == j:
                    ax.hist(sample_pp[c1], bins=20, color=PALETTE[i], alpha=0.7, edgecolor="white")
                else:
                    for cat, grp in sample_pp.groupby("Category"):
                        ax.scatter(grp[c2], grp[c1], alpha=0.35, s=6,
                                   color=colors_cat.get(cat, "gray"), label=cat)
                if i == 2:
                    ax.set_xlabel(c2, fontsize=8)
                if j == 0:
                    ax.set_ylabel(c1, fontsize=8)
                ax.tick_params(labelsize=7)
        plt.suptitle("Pair Plot", fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("##### Heatmap: Avg Rating per Category & Price Tier")
    pivot2 = df_filtered.pivot_table(values="Rating", index="Category",
                                      columns="Price_Category", aggfunc="mean")
    pivot2 = pivot2.reindex(columns=["Budget", "Mid-Range", "Premium", "Luxury"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot2, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, annot_kws={"size": 11})
    ax.set_title("Average Rating: Category × Price Tier", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">💡 Ratings are remarkably consistent (~3.0) across all combinations of category and price tier — confirming price does not drive customer satisfaction in this market.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: INSIGHTS & CONCLUSIONS
# ─────────────────────────────────────────────────────────────
elif page == "💡 Insights & Conclusions":
    st.markdown('<div class="section-header">💡 Insights & Conclusions</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🔑 Key Findings")
        findings = [
            ("Price & Rating are uncorrelated", "Customers rate products consistently regardless of price — a higher price tag does not guarantee better reviews."),
            ("All categories have similar ratings", "No single category dominates in customer satisfaction. Every category clusters around 3.0 stars."),
            ("Discounts don't clear stock", "Out-of-Stock products show slightly higher discounts — aggressive discounting alone may not resolve supply issues."),
            ("Sports has the highest discounts", "Sports products average the highest discounts, possibly to compete with seasonal alternatives."),
            ("Electronics are most expensive", "Electronics dominate the Luxury and Premium price tiers, consistent with real-world market patterns."),
        ]
        for title, desc in findings:
            st.markdown(f'<div class="insight-box">🔹 <b>{title}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### ✅ Recommendations")
        recs = [
            "Focus quality improvement in all categories — price alone doesn't earn better reviews.",
            "Investigate why high discounts still result in stock shortages — supply chain issues may be a factor.",
            "Sports & Clothing: target promotions at loyal buyers rather than broad discounting.",
            "Electronics: premium pricing is justified — maintain brand narrative around innovation.",
            "Collect more granular data (e.g., product age, reviews text) to deepen analysis.",
        ]
        for i, r in enumerate(recs, 1):
            st.markdown(f'<div class="step-box">✔️ <b>Rec {i}:</b> {r}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Summary Dashboard")
    c3, c4, c5, c6 = st.columns(4)
    best_cat = df_filtered.groupby("Category")["Rating"].mean().idxmax()
    cheapest = df_filtered.groupby("Category")["Price"].mean().idxmin()
    best_disc = df_filtered.groupby("Category")["Discount"].mean().idxmax()
    in_stock_pct = (df_filtered["Stock"] == "In Stock").mean() * 100
    c3.metric("Best Rated Category", best_cat, "⭐")
    c4.metric("Most Affordable", cheapest, "💰")
    c5.metric("Best Avg Discount", best_disc, "🏷️")
    c6.metric("In-Stock Rate", f"{in_stock_pct:.1f}%", "📦")

    st.divider()
    st.markdown("#### 📁 Download Clean Data")
    csv_bytes = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Clean Dataset (CSV)",
        data=csv_bytes,
        file_name="clean_dataset.csv",
        mime="text/csv",
    )

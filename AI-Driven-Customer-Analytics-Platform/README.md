# 📊 AI-Driven Customer Analytics Platform

> **Full-stack Python analytics platform** — from raw CSV to ML models, SHAP explainability, BI dashboards, and cloud deployment.

### 🌐 Live Deployment & Resources
- **GitHub Repository**: [Data-Analytics-with-AI](https://github.com/pavan12-dotcom/Data-Analytics-with-AI)
- **AWS ECS Fargate Streamlit App**: [Streamlit App Live (AWS)](http://43.204.220.219:8501)

### 📊 Dashboard Preview
![Dashboard Preview](3_WhatsApp%20Image%202026-06-13%20at%208.49.00%20PM.jpeg)

---


## 🧩 Tech Stack (as per project specification)

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas · NumPy |
| **Statistics** | SciPy (ANOVA, Chi-Square, Pearson, Spearman) · Statsmodels (Tukey HSD) |
| **ML Models** | Scikit-learn (K-Means, Random Forest, Gradient Boosting) |
| **Explainability** | SHAP · LIME |
| **Visualization** | Matplotlib · Seaborn · Plotly |
| **BI Dashboard** | Streamlit (Tableau/Power BI style) |
| **AI Narratives** | Prompt Engineering (Jinja2 templates + OpenAI GPT-4 optional) |
| **Database** | SQLAlchemy + PyMySQL |
| **Deployment** | Docker · Cloud-ready (AWS / Azure / GCP) |

---

## 📁 Project Structure

```
AI-Driven-Customer-Analytics-Platform/
│
├── customer_shopping_behavior.csv    # Dataset (3,900 records, 18 features)
├── customer_shopping_behavior.ipynb  # Original EDA notebook
│
├── analytics_pipeline.py             # ✅ Full ML pipeline (all tech stack)
├── streamlit_app.py                  # ✅ Interactive BI Dashboard
│
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker/Cloud deployment
├── .env.example                      # Environment variables template
├── README.md                         # This file
│
└── outputs/                          # Generated on run
    ├── 02a_distributions.png
    ├── 02b_scatter_diagrams.png
    ├── 02c_categoricals.png
    ├── 03_statistical_analysis.png
    ├── 04a_elbow_silhouette.png
    ├── 04b_cluster_visualization.png
    ├── 05_regression_results.png
    ├── 06_classification_results.png
    ├── 07a_shap_bar_regression.png
    ├── 07b_shap_beeswarm_regression.png
    ├── 07c_shap_dependence_prev_purchases.png
    ├── 07d_shap_bar_classifier.png
    ├── 08_lime_sample_*.png
    ├── 09a_bi_revenue_dashboard.png
    ├── 09b_segment_intelligence.png
    ├── statistical_results.json
    ├── ai_narrative_report.txt
    └── openai_prompt.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline (generates all outputs)
```bash
python analytics_pipeline.py
```

### 3. Launch the Streamlit BI Dashboard
```bash
streamlit run streamlit_app.py
```
Open → `http://localhost:8501`

### 4. Run with Docker (Cloud deployment)
```bash
# Build
docker build -t customer-analytics .

# Run
docker run -p 8501:8501 customer-analytics

# With MySQL connection
docker run -p 8501:8501 \
  -e DB_HOST=your-host \
  -e DB_USER=root \
  -e DB_PASS=yourpassword \
  -e DB_NAME=customer_analytics \
  customer-analytics
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Records | 3,900 customers |
| Features | 18 columns |
| Missing | 37 nulls in `Review Rating` → imputed by category median |
| Target (Regression) | `Purchase Amount (USD)` |
| Target (Classification) | `Subscription Status` |

**Key Features:** Age, Gender, Item Purchased, Category, Purchase Amount, Location, Size, Color, Season, Review Rating, Subscription Status, Shipping Type, Discount Applied, Previous Purchases, Payment Method, Frequency of Purchases

---

## 🧪 Pipeline Sections

### §1 Data Loading & Cleaning
- Load CSV with Pandas
- Impute missing `Review Rating` using category-group median
- Normalize column names (lowercase, underscores)
- Engineer features: `age_group`, `loyalty_tier`, `purchase_frequency_days`

### §2 EDA (Scatter Diagrams + Distributions)
- Histograms + KDE for all numerical features
- Scatter plots with OLS regression lines
- Categorical bar charts

### §3 Statistical Analysis
- **ANOVA**: Purchase Amount ~ Season (F=3.87, p=0.009 ✅)
- **Tukey HSD**: Post-hoc pairwise season comparison
- **Chi-Square**: Gender × Category, Subscription × Discount
- **Pearson/Spearman**: Correlation matrix + heatmap

### §4 Customer Segmentation (K-Means)
- Elbow + Silhouette method to select k=4
- Final clustering with StandardScaler preprocessing
- PCA 2D visualization
- Segment profiles: Champions, Loyalists, Regulars, New Customers

### §5 Regression (Random Forest)
- Target: Purchase Amount
- Features: 12 columns (categorical encoded + numerical)
- **R²=0.847, MAE=$4.23, RMSE=$6.71**
- 5-fold cross-validation

### §6 Classification (Gradient Boosting)
- Target: Subscription Status (binary)
- **Accuracy=78.3%, F1=0.77, AUC=0.86**
- Confusion matrix + ROC curve

### §7 SHAP Explainability
- Global feature importance (bar plot)
- Beeswarm summary plot
- Dependence plot: Previous Purchases
- SHAP for classifier model

### §8 LIME (Local Explanations)
- 3 local sample explanations
- Per-feature contribution for individual predictions

### §9 BI Visualizations (Tableau/Power BI style)
- Dark-themed executive revenue dashboard
- Segment intelligence board
- All publication-quality plots

### §10 AI Narrative Insights (Prompt Engineering)
- Template-based auto-narrative generation (Jinja2)
- OpenAI GPT-4 prompt engineering (optional)
- Exported as `.txt` and `.md`

### §11 Database Export (SQLAlchemy + PyMySQL)
- Push cleaned + enriched DataFrame to MySQL
- Configurable via environment variables

---

## 🌐 Cloud Deployment

### AWS EC2 / ECS
```bash
# ECR push
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag customer-analytics:latest <account>.dkr.ecr.<region>.amazonaws.com/customer-analytics:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/customer-analytics:latest
```

### Azure Container Instances
```bash
az container create --resource-group myRG \
  --name customer-analytics \
  --image customer-analytics:latest \
  --ports 8501 --ip-address public
```

### GCP Cloud Run
```bash
gcloud run deploy customer-analytics \
  --image gcr.io/PROJECT_ID/customer-analytics \
  --port 8501 --allow-unauthenticated
```

---

## 🔑 Environment Variables

```env
DB_USER=root
DB_PASS=yourpassword
DB_HOST=localhost
DB_PORT=3306
DB_NAME=customer_analytics
OPENAI_API_KEY=sk-...
```

---

## 📈 Key Findings

| Finding | Insight |
|---------|---------|
| **Top Revenue Season** | Fall ($61.56 avg) — ANOVA significant |
| **Best Category** | Clothing (most orders) · Footwear (highest avg $60.26) |
| **Subscription Rate** | Only 27% — biggest retention opportunity |
| **Top Segment** | Champions (19.6%) drive disproportionate LTV |
| **#1 SHAP Driver** | Previous Purchases → strongest purchase predictor |

---

*Built with ❤️ — Python · Scikit-learn · SHAP · LIME · Streamlit · Docker*

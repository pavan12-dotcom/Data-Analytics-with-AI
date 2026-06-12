import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency
from lime.lime_tabular import LimeTabularExplainer
import google.generativeai as genai

# Page Configuration
st.set_page_config(
    page_title="AI Customer Behavior Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
    <style>
    /* Dark Mode Sleek Styling */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #8892b0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00f2fe;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0aec0;
    }
    button[data-baseweb="tab"]:hover {
        color: #00f2fe;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe;
        border-bottom-color: #00f2fe;
    }
    </style>
""", unsafe_allow_html=True)

# Load Datasets and Models
@st.cache_data
def load_data():
    if os.path.exists("customer_segments.csv"):
        return pd.read_csv("customer_segments.csv")
    return pd.DataFrame()

df = load_data()

@st.cache_resource
def load_ml_assets():
    model, scaler = None, None
    if os.path.exists("rf_reg_model.pkl"):
        model = joblib.load("rf_reg_model.pkl")
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_ml_assets()

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #00f2fe;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go to:",
    [
        "Dashboard Overview",
        "Exploratory Data Analysis",
        "Statistical Significance Tests",
        "XAI Prediction Model",
        "AI Business Strategy",
        "System Deployment"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Dataset Details")
if not df.empty:
    st.sidebar.info(f"""
    - **Total Customers**: {len(df)}
    - **Features**: {df.shape[1]}
    - **Segments (Clusters)**: {df['Cluster'].nunique() if 'Cluster' in df.columns else 'N/A'}
    """)
else:
    st.sidebar.warning("Dataset not found!")

# --- 1. DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.markdown("<h1 class='main-header'>AI-Driven Customer Behavior Analytics Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Perform customer segmentation, predict purchasing behavior, and extract statistical insights.</p>", unsafe_allow_html=True)
    
    if not df.empty:
        # Key Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val'>{len(df):,}</div>
                <div class='metric-label'>Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            avg_income = df['Income'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val'>${avg_income:,.2f}</div>
                <div class='metric-label'>Average Income</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            avg_spending = df['Total_Spending'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val'>${avg_spending:,.2f}</div>
                <div class='metric-label'>Average Spending</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            avg_purchases = df['Total_Purchases'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val'>{avg_purchases:.1f}</div>
                <div class='metric-label'>Avg Purchases</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Segment Distribution
        st.header("Customer Segment (Cluster) Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            # Bar chart of cluster counts
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1a1f2c')
            
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues_d", ax=ax)
            ax.set_title("Customer Counts per Segment", color="white", fontsize=12, fontweight='bold')
            ax.set_xlabel("Segment ID (Cluster)", color="white")
            ax.set_ylabel("Number of Customers", color="white")
            ax.tick_params(colors="white")
            ax.spines['bottom'].set_color('#2d3748')
            ax.spines['left'].set_color('#2d3748')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        with col_table:
            # Table of cluster percentages
            cluster_pct = (df["Cluster"].value_counts(normalize=True) * 100).sort_index()
            summary_tbl = pd.DataFrame({
                "Customers": cluster_counts.values,
                "Percentage (%)": cluster_pct.values
            }, index=[f"Segment {i}" for i in cluster_counts.index])
            st.dataframe(summary_tbl.style.format({"Percentage (%)": "{:.1f}%"}))
            
        st.divider()
        
        # Raw Data Sample
        st.header("Dataset Overview (First 10 Customers)")
        st.dataframe(df.head(10), use_container_width=True)

# --- 2. EXPLORATORY DATA ANALYSIS ---
elif page == "Exploratory Data Analysis":
    st.markdown("<h1 class='main-header'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore relationships between customer demographics, behaviors, and segment attributes.</p>", unsafe_allow_html=True)
    
    if not df.empty:
        col_controls, col_plot = st.columns([1, 2])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID, Z_CostContact, Z_Revenue as they are uninformative
        numeric_cols = [c for c in numeric_cols if c not in ['ID', 'Z_CostContact', 'Z_Revenue']]
        
        with col_controls:
            st.subheader("Configure Scatter Diagram")
            x_feature = st.selectbox("X-Axis Feature", numeric_cols, index=numeric_cols.index("Income"))
            y_feature = st.selectbox("Y-Axis Feature", numeric_cols, index=numeric_cols.index("Total_Spending"))
            color_by = st.checkbox("Color by Cluster (Segments)", value=True)
            
            st.divider()
            st.info("""
            **Scatter Diagrams** help identify patterns, trends, and outliers:
            - Look for linear/non-linear relationships between variables.
            - Color-coding by Cluster reveals if segmentation matches specific feature boundaries.
            """)
            
        with col_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1a1f2c')
            
            if color_by and 'Cluster' in df.columns:
                sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Cluster', palette='viridis', alpha=0.7, ax=ax)
                legend = ax.legend(title="Segment", facecolor='#1a1f2c', edgecolor='#2d3748')
                plt.setp(legend.get_texts(), color='white')
                plt.setp(legend.get_title(), color='white')
            else:
                sns.scatterplot(data=df, x=x_feature, y=y_feature, color='#00f2fe', alpha=0.7, ax=ax)
                
            ax.set_title(f"{y_feature} vs {x_feature}", color="white", fontsize=14, fontweight='bold')
            ax.set_xlabel(x_feature, color="white")
            ax.set_ylabel(y_feature, color="white")
            ax.tick_params(colors="white")
            ax.spines['bottom'].set_color('#2d3748')
            ax.spines['left'].set_color('#2d3748')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        st.divider()
        
        # Correlation Matrix Section
        st.header("Feature Correlation Heatmap")
        
        selected_corr_features = st.multiselect(
            "Select Features for Heatmap:",
            numeric_cols,
            default=['Income', 'Age', 'Children', 'Recency', 'Total_Purchases', 'Total_Spending', 'Customer_Days']
        )
        
        if len(selected_corr_features) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            fig_corr.patch.set_facecolor('#0e1117')
            
            corr_matrix = df[selected_corr_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, 
                        annot_kws={"size": 10, "weight": "bold"}, cbar=True)
            
            ax_corr.set_title("Correlation Matrix of Customer Metrics", color="white", fontsize=14, fontweight='bold')
            ax_corr.tick_params(colors="white")
            st.pyplot(fig_corr)
        else:
            st.warning("Please select at least 2 features to render the correlation heatmap.")

# --- 3. STATISTICAL SIGNIFICANCE TESTS ---
elif page == "Statistical Significance Tests":
    st.markdown("<h1 class='main-header'>Statistical Significance Testing</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Verify whether differences between customer groups are statistically significant or occurred by chance.</p>", unsafe_allow_html=True)
    
    if not df.empty:
        # ANOVA SECTION
        st.header("1. One-Way ANOVA Test (Analysis of Variance)")
        st.write("""
        An **ANOVA test** determines if there is a statistically significant difference in a continuous numerical feature 
        (e.g., *Income*, *Spending*) across the different customer segments (clusters).
        """)
        
        anova_feature = st.selectbox(
            "Select Numerical Feature to Test across Segments:",
            ['Income', 'Total_Spending', 'Age', 'Total_Purchases', 'Recency'],
            index=0
        )
        
        # Group data by Cluster
        clusters = sorted(df['Cluster'].unique())
        groups = [df[df['Cluster'] == c][anova_feature].dropna() for c in clusters]
        
        f_stat, p_val_anova = f_oneway(*groups)
        
        col_anova_res, col_anova_box = st.columns([1, 1])
        
        with col_anova_res:
            st.subheader("ANOVA Results")
            st.metric("F-Statistic", f"{f_stat:.4f}")
            
            # Formatting P-value
            p_str = f"{p_val_anova:.4e}" if p_val_anova < 0.001 else f"{p_val_anova:.4f}"
            st.metric("P-value", p_str)
            
            if p_val_anova < 0.05:
                st.success("Significant Difference Found (p < 0.05)")
                st.markdown(f"""
                **Interpretation:** The average **{anova_feature}** differs significantly across the customer segments. 
                We reject the null hypothesis ($H_0$), confirming that these clusters represent distinct behaviors/demographics 
                rather than random variations.
                """)
            else:
                st.warning("No Significant Difference (p >= 0.05)")
                st.markdown(f"""
                **Interpretation:** There is no statistically significant difference in average **{anova_feature}** 
                across the customer segments. The variation within groups is greater than the variation between groups.
                """)
                
        with col_anova_box:
            # Boxplot of feature by Cluster
            fig_box, ax_box = plt.subplots(figsize=(8, 5))
            fig_box.patch.set_facecolor('#0e1117')
            ax_box.set_facecolor('#1a1f2c')
            
            sns.boxplot(data=df, x='Cluster', y=anova_feature, palette='viridis', ax=ax_box)
            ax_box.set_title(f"{anova_feature} Distribution per Segment", color="white", fontsize=12, fontweight='bold')
            ax_box.set_xlabel("Segment ID (Cluster)", color="white")
            ax_box.set_ylabel(anova_feature, color="white")
            ax_box.tick_params(colors="white")
            ax_box.spines['bottom'].set_color('#2d3748')
            ax_box.spines['left'].set_color('#2d3748')
            ax_box.spines['top'].set_visible(False)
            ax_box.spines['right'].set_visible(False)
            st.pyplot(fig_box)
            
        st.divider()
        
        # CHI-SQUARE SECTION
        st.header("2. Chi-Square Test of Independence")
        st.write("""
        A **Chi-Square test** determines if there is an association between customer segment membership (Cluster) 
        and categorical variables like campaign acceptance (**Response**).
        """)
        
        categorical_cols = ['Response', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain']
        selected_cat = st.selectbox("Select Categorical Campaign Variable:", categorical_cols, index=0)
        
        # Contingency Table
        contingency_table = pd.crosstab(df['Cluster'], df[selected_cat])
        
        chi2, p_val_chi, dof, expected = chi2_contingency(contingency_table)
        
        col_chi_res, col_chi_tbl = st.columns([1, 1])
        
        with col_chi_res:
            st.subheader("Chi-Square Results")
            st.metric("Chi-Square Value", f"{chi2:.4f}")
            
            p_str_chi = f"{p_val_chi:.4e}" if p_val_chi < 0.001 else f"{p_val_chi:.4f}"
            st.metric("P-value", p_str_chi)
            
            if p_val_chi < 0.05:
                st.success("Significant Association Found (p < 0.05)")
                st.markdown(f"""
                **Interpretation:** There is a significant association between customer segment (Cluster) 
                and campaign **{selected_cat}**. Certain segments are significantly more likely to respond positively 
                to this campaign than others.
                """)
            else:
                st.warning("No Significant Association (p >= 0.05)")
                st.markdown(f"""
                **Interpretation:** Segment membership and campaign **{selected_cat}** are independent. 
                Customer segments do not display statistically distinct response rates for this variable.
                """)
                
        with col_chi_tbl:
            st.subheader("Contingency Table (Observed)")
            # style contingency table
            st.dataframe(contingency_table.style.highlight_max(axis=0, color="#1e3a5f"))

# --- 4. XAI PREDICTION MODEL ---
elif page == "XAI Prediction Model":
    st.markdown("<h1 class='main-header'>Predictive Analytics & Explainable AI (XAI)</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Predict customer spending behavior using a Random Forest Regressor and explain decision criteria locally with LIME.</p>", unsafe_allow_html=True)
    
    if model is None:
        st.error("ML model not loaded! Make sure rf_reg_model.pkl is present.")
    else:
        # Feature columns: ['Income', 'Age', 'Children', 'Recency', 'Total_Purchases']
        col_inputs, col_pred = st.columns([1, 1])
        
        with col_inputs:
            st.header("Enter Customer Profile Attributes")
            
            # Default values from dataset
            def_income = float(df['Income'].mean()) if not df.empty else 50000.0
            def_age = int(df['Age'].mean()) if not df.empty else 40
            def_recency = int(df['Recency'].mean()) if not df.empty else 30
            def_purchases = int(df['Total_Purchases'].mean()) if not df.empty else 10
            
            income = st.number_input("Annual Income ($)", min_value=0.0, max_value=500000.0, value=def_income, step=1000.0)
            age = st.number_input("Age (Years)", min_value=18, max_value=110, value=def_age, step=1)
            children = st.number_input("Number of Children/Dependents", min_value=0, max_value=10, value=0, step=1)
            recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=def_recency, step=1)
            total_purchases = st.number_input("Total Number of Purchases", min_value=0, max_value=100, value=def_purchases, step=1)
            
            submit_btn = st.button("Calculate Predicted Spending & Run XAI")
            
        with col_pred:
            st.header("Model Output & Explanation")
            
            if submit_btn:
                input_df = pd.DataFrame({
                    "Income": [income],
                    "Age": [age],
                    "Children": [children],
                    "Recency": [recency],
                    "Total_Purchases": [total_purchases]
                })
                
                # Model Prediction
                pred_spending = model.predict(input_df)[0]
                
                st.markdown(f"""
                <div class='metric-card' style='border-color: #00f2fe; margin-bottom: 2rem;'>
                    <div class='metric-val' style='font-size: 2.8rem;'>${pred_spending:,.2f}</div>
                    <div class='metric-label' style='font-size: 1.1rem; color: #00f2fe;'>Estimated Total Spending</div>
                </div>
                """, unsafe_allow_html=True)
                
                # LIME EXPLANATION
                with st.spinner("Generating LIME explainability metrics..."):
                    feature_cols = ['Income', 'Age', 'Children', 'Recency', 'Total_Purchases']
                    X_train = df[feature_cols].values
                    
                    explainer = LimeTabularExplainer(
                        training_data=X_train,
                        feature_names=feature_cols,
                        class_names=['Total_Spending'],
                        mode='regression',
                        random_state=42
                    )
                    
                    exp = explainer.explain_instance(
                        data_row=input_df.iloc[0],
                        predict_fn=model.predict,
                        num_features=5
                    )
                    
                    # Process weights
                    exp_list = exp.as_list()
                    exp_df = pd.DataFrame(exp_list, columns=['Feature Condition', 'Contribution'])
                    exp_df = exp_df.sort_values(by='Contribution', ascending=True)
                    
                    # Plot horizontal bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#1a1f2c')
                    
                    # Colors: Green for positive contribution, Red for negative
                    colors = ['#ff4d4d' if x < 0 else '#2ecc71' for x in exp_df['Contribution']]
                    
                    sns.barplot(
                        x='Contribution', 
                        y='Feature Condition', 
                        data=exp_df, 
                        palette=colors,
                        ax=ax
                    )
                    
                    ax.set_title("LIME Explainer: Feature Contributions", color="white", fontsize=12, fontweight='bold')
                    ax.set_xlabel("Contribution Value ($)", color="white")
                    ax.set_ylabel("Decision Boundaries", color="white")
                    ax.tick_params(colors="white")
                    ax.spines['bottom'].set_color('#2d3748')
                    ax.spines['left'].set_color('#2d3748')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    st.pyplot(fig)
                    
                    st.info("""
                    **Explainable AI (LIME) Interpretation:**
                    - **Green bars (positive)** show feature conditions that increased this customer's predicted spending relative to the base average.
                    - **Red bars (negative)** show features that decreased their predicted spending.
                    """)
            else:
                st.write("Submit the form to generate spending predictions and explain the model's logic.")

# --- 5. AI BUSINESS STRATEGY ---
elif page == "AI Business Strategy":
    st.markdown("<h1 class='main-header'>AI Business Strategy & Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Generate narrative business insights and segment strategies using the Gemini LLM.</p>", unsafe_allow_html=True)
    
    if not df.empty:
        # Prepare segment-level details for prompt
        cluster_summary = df.groupby("Cluster")[["Income", "Age", "Total_Spending", "Total_Purchases", "Recency"]].mean()
        
        st.subheader("Aggregated Customer Segment Profiles")
        st.dataframe(cluster_summary.style.format("${:,.2f}", subset=['Income', 'Total_Spending']).format("{:.1f}", subset=['Age', 'Total_Purchases', 'Recency']))
        
        st.divider()
        
        # Load API Key securely from environment or local .env file
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
                            
        # If still not found, check Streamlit Secrets
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass
                
        # Fallback to manual text input in sidebar
        if not api_key:
            api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
            
        if api_key:
            genai.configure(api_key=api_key)
            try:
                ai_model = genai.GenerativeModel("gemini-2.5-flash")
                
                if st.button("Generate Senior Analyst Report"):
                    with st.spinner("Senior Business Analyst is writing the strategy report..."):
                        prompt = f"""
                        You are a senior business analyst.
                        The customer analytics dataset has the following statistics:
                        - Total Customer Base: {len(df)}
                        - Average overall Income: {df['Income'].mean():.2f}
                        - Average overall Spending: {df['Total_Spending'].mean():.2f}
                        - Number of Customer Segments: {df['Cluster'].nunique()}
                        
                        Here is the mean profiles of each cluster/segment:
                        {cluster_summary.to_string()}
                        
                        Provide a professional report including:
                        1. Executive Summary
                        2. Key Customer Profiles (Profile description for Cluster 0, 1, 2, 3)
                        3. Behavior Insights
                        4. Actionable Marketing Recommendations & Campaign Strategies tailored to each segment.
                        """
                        response = ai_model.generate_content(prompt)
                        st.markdown(response.text)
            except Exception as e:
                st.error(f"Error calling Gemini API: {e}")
        else:
            st.warning("Gemini API Key not found. Please provide an API key in the sidebar configuration to unlock the Business Strategy generator.")
    else:
        st.warning("Dataset not found!")

# --- 6. SYSTEM DEPLOYMENT ---
elif page == "System Deployment":
    st.markdown("<h1 class='main-header'>Containerization & Deployment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Inspect deployment status, build the Docker image, and run it locally or on the cloud.</p>", unsafe_allow_html=True)
    
    col_desc, col_docker = st.columns([1, 1])
    
    with col_desc:
        st.subheader("Cloud Architecture Overview")
        st.write("""
        The platform is fully containerized using **Docker** to ensure reproducibility, scaling, and easy cloud deployments (AWS ECS, Google Cloud Run, Azure Container Instances).
        
        ### How to Build & Run locally:
        1. Open your terminal in the root project folder.
        2. **Build the Docker Image**:
           ```bash
           docker build -t customer-analytics .
           ```
        3. **Run the Container**:
           ```bash
           docker run -d -p 8501:8501 --name analytics-platform customer-analytics
           ```
        4. Open [http://localhost:8501](http://localhost:8501) in your browser.
        """)
        
    with col_docker:
        st.subheader("Dockerfile Preview")
        dockerfile_content = """
FROM python:3.12

COPY . .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
        """
        st.code(dockerfile_content, language="dockerfile")

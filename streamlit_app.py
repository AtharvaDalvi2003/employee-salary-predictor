import streamlit as st 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import random
import time

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== Load Model ==========
@st.cache_resource
def load_model():
    model = joblib.load("final_salary_model.pkl")
    columns = joblib.load("input_columns_config.pkl")
    X_eval, y_true, y_guess = joblib.load("model_test_output.pkl")
    return model, columns, X_eval, y_true, y_guess

model, columns_required, X_eval, y_true, y_guess = load_model()
employee_df = pd.read_csv("employee_income_data.csv").dropna()

# ========== Custom CSS ==========
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .title-box {
        text-align: center;
        margin-bottom: 1rem;
    }
    .predict-btn button {
        background-color: #4CAF50;
        color: #ffffff;
        padding: 0.6rem 2rem;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
        margin: 0 auto;
        display: block;
    }
    .predict-btn button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .social-container {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    .social-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
    }
    .social-icon:hover {
        transform: scale(1.1);
    }
    .social-icon img {
        width: 20px;
        height: 20px;
        filter: invert(0);
    }
    .divider {
        border: 0;
        height: 1px;
        background: #333;
        margin: 1.5rem 0;
    }
    .developer-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem auto;
        max-width: 500px;
        text-align: center;
    }
    .form-section {
        background-color: #161a1f;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .chart-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    .insight-box h4 {
        color: #FFA500;
        margin-bottom: 0.5rem;
    }
    .stSelectbox, .stRadio, .stTextInput {
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Header Image ==========
def load_header_image():
    image_url = "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-1.2.1&auto=format&fit=crop&w=900&h=220&q=80"
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

header_img = load_header_image()
st.image(header_img, use_container_width=True)

# ========== Title ==========
st.markdown("""
<div class="title-box">
    <h1>📊 AI Salary Predictor</h1>
    <h4>Predict employee salaries with Machine Learning</h4>
</div>
""", unsafe_allow_html=True)

# ========== Input Form ==========
with st.form("salary_form"):
    st.markdown("""
    <div class="form-section">
        <h3 style="color:#ffffff; margin-bottom:1rem;">🔍 Enter Candidate Details</h3>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<h4>Personal Information</h4>""", unsafe_allow_html=True)
        emp_age = st.selectbox("Age", ["Enter Age"] + list(range(18, 66)))
        emp_gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    with col2:
        st.markdown("""<h4>Professional Information</h4>""", unsafe_allow_html=True)
        emp_title = st.selectbox("Job Title", ["Enter Job Title"] + sorted(employee_df['Job Title'].unique()))
        emp_experience = st.selectbox("Experience (Years)", ["Enter Experience"] + list(range(0, 51)))

    col3, col4 = st.columns(2)
    with col3:
        emp_location = st.selectbox("Work Location", ["Enter Location", "Urban", "Suburban", "Rural"])
    with col4:
        emp_edu = st.selectbox("Education", ["Enter Education", "Bachelor's", "Master's", "PhD"])

    industry_sector = st.selectbox("Industry Sector", ["Select Sector", "Technology", "Finance", "Healthcare", "Manufacturing", "Retail"])

    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Predict", type="primary")

# ========== Prediction Logic ==========
if submitted:
    if emp_edu == "Enter Education" or emp_location == "Enter Location" or industry_sector == "Select Sector" or emp_age == "Enter Age" or emp_experience == "Enter Experience" or emp_title == "Enter Job Title":
        st.error("❌ Please fill in all the details before predicting your salary.")
    elif isinstance(emp_experience, int) and isinstance(emp_age, int) and (emp_experience >= emp_age or emp_experience > (emp_age - 18)):
        st.error("❌ Invalid age/experience combination. Experience cannot exceed (Age - 18).")
    elif emp_edu == "Master's" and isinstance(emp_age, int) and emp_age < 23:
        st.error("❌ Too young for a Master's degree. Minimum age required: 23.")
    elif emp_edu == "PhD" and isinstance(emp_age, int) and emp_age < 26:
        st.error("❌ Too young for a PhD. Minimum age required: 26.")
    else:
        with st.spinner("Predicting salary..."):
            time.sleep(1)

        input_vector = {feature: 0 for feature in columns_required}
        input_vector["Age"] = emp_age if isinstance(emp_age, int) else 0
        input_vector["Years of Experience"] = emp_experience if isinstance(emp_experience, int) else 0

        for cat in [f"Gender_{emp_gender}", f"Education Level_{emp_edu}", f"Job Title_{emp_title}", f"Location_{emp_location}", f"Industry Sector_{industry_sector}"]:
            if cat in columns_required:
                input_vector[cat] = 1

        final_input = pd.DataFrame([input_vector])[columns_required]
        estimated_salary = model.predict(final_input)[0]

        st.empty()
        st.success(f"🎯 **Predicted Annual Salary: ₹{estimated_salary:,.2f}**")

        # ========== Salary Distribution ==========
        st.markdown("---")
        st.header("📊 Salary Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        random_data = np.random.normal(estimated_salary, estimated_salary * 0.2, 50)
        ax.hist(random_data, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(estimated_salary, color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f"Salary Distribution for {emp_title}")
        ax.set_xlabel("Salary Range (₹)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # ========== Horizontal Bar Chart (Simplified Salary Comparison) ==========
        # ========== Animated Salary Journey ==========
        st.markdown("---")
        st.header("🎨 Your Salary Journey: Past, Present & Future")

        if submitted:
        # Create timeline data
            timeline_data = []
        current_year = 2024
    
    # Past experience
        for i in range(emp_experience):
            timeline_data.append({
            'Year': current_year - emp_experience + i,
            'Experience': i,
            'Salary': estimated_salary * (0.6 + (i * 0.4 / emp_experience)),  # Growth curve
            'Phase': 'Career Journey',
            'Status': 'Past'
        })
    
    # Current position
        timeline_data.append({
        'Year': current_year,
        'Experience': emp_experience,
        'Salary': estimated_salary,
        'Phase': 'Current Position',
        'Status': 'Present'
        })
    
    # Future projections (next 10 years)
        for i in range(1, 11):
            future_salary = estimated_salary * (1.08 ** i)  # 8% annual growth
            timeline_data.append({
            'Year': current_year + i,
            'Experience': emp_experience + i,
            'Salary': future_salary,
            'Phase': 'Future Projection',
            'Status': 'Future'
        })
    
        timeline_df = pd.DataFrame(timeline_data)
    
    # Create gradient line chart
        fig_journey = go.Figure()
    
    # Past (blue gradient)
        past_data = timeline_df[timeline_df['Status'] == 'Past']
        if not past_data.empty:
            fig_journey.add_trace(go.Scatter(
            x=past_data['Year'],
            y=past_data['Salary'],
            mode='lines+markers',
            line=dict(color='#87CEEB', width=4),
            marker=dict(size=8, color='#4682B4'),
            name='Career History',
            hovertemplate='<b>%{x}</b><br>Salary: ₹%{y:,.0f}<extra></extra>'
        ))
    
    # Present (red highlight)
        present_data = timeline_df[timeline_df['Status'] == 'Present']
        fig_journey.add_trace(go.Scatter(
        x=present_data['Year'],
        y=present_data['Salary'],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='Current Position',
        hovertemplate='<b>NOW</b><br>Predicted Salary: ₹%{y:,.0f}<extra></extra>'
        ))
    
    # Future (green gradient)
        future_data = timeline_df[timeline_df['Status'] == 'Future']
        fig_journey.add_trace(go.Scatter(
        x=future_data['Year'],
        y=future_data['Salary'],
        mode='lines+markers',
        line=dict(color='#98FB98', width=4, dash='dot'),
        marker=dict(size=8, color='#32CD32'),
        name='Future Projection',
        hovertemplate='<b>%{x}</b><br>Projected Salary: ₹%{y:,.0f}<extra></extra>'
        ))
    
    # Styling
        fig_journey.update_layout(
        title="🚀 Your Complete Salary Journey Timeline",
        xaxis_title="Year",
        yaxis_title="Annual Salary (₹)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=600,
        hovermode="x unified",
        showlegend=True
        )
    
        st.plotly_chart(fig_journey, use_container_width=True)


        # ========== Density Plot ==========
        # ========== Market Volatility Simulator ==========
st.markdown("---")
st.header("📈 Salary Market Trends & Economic Impact")

if submitted:
    # Simulate market conditions
    years = list(range(2020, 2031))
    scenarios = {
        'Optimistic Growth': [1.12, 1.10, 1.08, 1.09, 1.11, 1.13, 1.12, 1.10, 1.09, 1.11, 1.12],
        'Conservative Growth': [1.05, 1.04, 1.03, 1.04, 1.05, 1.06, 1.05, 1.04, 1.05, 1.06, 1.05],
        'Economic Recession': [0.98, 0.95, 1.01, 1.03, 1.08, 1.10, 1.07, 1.05, 1.06, 1.08, 1.09]
    }
    
    # Create trend data
    trend_data = []
    base_salary = estimated_salary * 0.8  # Starting point
    
    for scenario, multipliers in scenarios.items():
        current_salary = base_salary
        for year, multiplier in zip(years, multipliers):
            current_salary *= multiplier
            trend_data.append({
                'Year': year,
                'Salary': current_salary,
                'Scenario': scenario,
                'Growth_Rate': (multiplier - 1) * 100
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    # Create trend chart with confidence intervals
    fig_trends = px.line(
        trend_df,
        x='Year',
        y='Salary',
        color='Scenario',
        title="📊 Salary Trends Under Different Economic Scenarios",
        labels={'Salary': 'Expected Salary (₹)', 'Year': 'Year'},
        hover_data={'Growth_Rate': ':.1f%'}
    )
    
    # Add current year marker
    fig_trends.add_vline(
        x=2024,
        line_dash="dot",
        line_color="yellow",
        annotation_text="Current Year"
    )
    
    # Styling
    fig_trends.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=600,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)

# ========== Career Insights ==========
st.markdown("---")
st.header("💡 Career Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="insight-box">
        <h4>💼 Career Tip</h4>
        <p>{random.choice([
            "Research industry standards before salary discussions.",
            "A Master's degree can increase earnings by 20-30% over a Bachelor's.",
            "Switching roles every 2-3 years can accelerate salary growth.",
            "70% of jobs are filled through referrals. Build your network!",
            "Learn Python or SQL—these skills can boost salaries by 15-25%."
        ])}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="insight-box">
        <h4>✨ Motivation</h4>
        <p>{random.choice([
            "Believe in yourself. You are capable of more than you imagine.",
            "Success is not final, failure is not fatal: It is the courage to continue that counts.",
            "Your salary is a reflection of your value. Keep growing, and the numbers will follow.",
            "The only limit to your earnings is your willingness to learn and adapt."
        ])}</p>
    </div>
    """, unsafe_allow_html=True)

# ========== Developer Footer ==========
github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
email_icon = "https://cdn-icons-png.flaticon.com/512/281/281769.png"

st.markdown(f"""
<div class="developer-box">
    <h4 style="margin: 0 0 1rem 0;">Profile of the Developer - Atharva Dalvi</h4>
    <div class="social-container">
        <a href="https://github.com/AtharvaDalvi2003" target="_blank" class="social-icon">
            <img src="{github_icon}" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/atharva-dalvi-01a281316/" target="_blank" class="social-icon">
            <img src="{linkedin_icon}" alt="LinkedIn">
        </a>
        <a href="mailto:atharva@example.com" class="social-icon">
            <img src="{email_icon}" alt="Contact">
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

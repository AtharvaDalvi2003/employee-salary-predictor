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
st.markdown("""<style>
.stApp { background-color: #0e1117; color: #ffffff; }
.title-box { text-align: center; margin-bottom: 1rem; }
.predict-btn button {
    background-color: #4CAF50; color: #ffffff; padding: 0.6rem 2rem;
    font-weight: bold; border-radius: 12px; transition: 0.3s;
    margin: 0 auto; display: block;
}
.predict-btn button:hover { background-color: #45a049; transform: scale(1.02); }
.social-container { display: flex; justify-content: center; gap: 1.5rem; margin: 1.5rem 0; }
.social-icon { width: 40px; height: 40px; border-radius: 50%; background: #ffffff;
    display: flex; align-items: center; justify-content: center; transition: all 0.3s; }
.social-icon:hover { transform: scale(1.1); }
.social-icon img { width: 20px; height: 20px; filter: invert(0); }
.divider { border: 0; height: 1px; background: #333; margin: 1.5rem 0; }
.developer-box {
    background-color: #1e1e1e; border: 1px solid #333; border-radius: 10px;
    padding: 1rem; margin: 1rem auto; max-width: 500px; text-align: center;
}
.form-section {
    background-color: #161a1f; padding: 1rem; border-radius: 12px;
    border: 1px solid #333; margin-bottom: 1rem;
}
.stDeployButton {display:none;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
.chart-box {
    background-color: #1e1e1e; padding: 1rem; border-radius: 10px;
    border: 1px solid #333; margin: 1rem 0;
}
.insight-box {
    background-color: #1e1e1e; padding: 1rem; border-radius: 10px;
    border: 1px solid #333; margin: 1rem 0;
}
.insight-box h4 { color: #FFA500; margin-bottom: 0.5rem; }
.welcome-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px; text-align: center;
    margin: 2rem 0; color: white;
}
</style>""", unsafe_allow_html=True)

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
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

with st.form("salary_form"):
    st.markdown("""
    <div class="form-section">
        <h3 style="color:#ffffff; margin-bottom:1rem;">🔍 Enter Candidate Details</h3>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4>Personal Information</h4>", unsafe_allow_html=True)
        emp_age = st.selectbox("Age", ["Enter Age"] + list(range(18, 66)))
        emp_gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    with col2:
        st.markdown("<h4>Professional Information</h4>", unsafe_allow_html=True)
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
        st.session_state.prediction_made = False
    elif isinstance(emp_experience, int) and isinstance(emp_age, int) and (emp_experience >= emp_age or emp_experience > (emp_age - 18)):
        st.error("❌ Invalid age/experience combination.")
        st.session_state.prediction_made = False
    elif emp_edu == "Master's" and emp_age < 23:
        st.error("❌ Too young for a Master's degree. Minimum age: 23.")
        st.session_state.prediction_made = False
    elif emp_edu == "PhD" and emp_age < 26:
        st.error("❌ Too young for a PhD. Minimum age: 26.")
        st.session_state.prediction_made = False
    else:
        with st.spinner("🔮 Analyzing your profile and predicting salary..."):
            time.sleep(2)

        input_vector = {feature: 0 for feature in columns_required}
        input_vector["Age"] = emp_age
        input_vector["Years of Experience"] = emp_experience
        for cat in [f"Gender_{emp_gender}", f"Education Level_{emp_edu}", f"Job Title_{emp_title}", f"Location_{emp_location}", f"Industry Sector_{industry_sector}"]:
            if cat in columns_required:
                input_vector[cat] = 1

        final_input = pd.DataFrame([input_vector])[columns_required]
        estimated_salary = model.predict(final_input)[0]

        st.session_state.prediction_made = True
        st.session_state.estimated_salary = estimated_salary
        st.session_state.emp_title = emp_title
        st.session_state.emp_experience = emp_experience

        st.success(f"🎯 **Predicted Annual Salary: ₹{estimated_salary:,.2f}**")

# ========== Show After Prediction ==========
if st.session_state.prediction_made:
    estimated_salary = st.session_state.estimated_salary
    emp_title = st.session_state.emp_title
    emp_experience = st.session_state.emp_experience

    # Salary Distribution
    st.markdown("---")
    st.header("📊 Salary Distribution Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    random_data = np.random.normal(estimated_salary, estimated_salary * 0.2, 100)
    ax.hist(random_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(estimated_salary, color='red', linestyle='dashed', linewidth=3, label=f'Your Predicted Salary: ₹{estimated_salary:,.0f}')
    ax.set_title(f"Salary Distribution for {emp_title}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Salary Range (₹)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Timeline
    st.markdown("---")
    st.header("🎨 Your Salary Journey: Past, Present & Future")
    current_year = 2024
    timeline_data = []

    for i in range(emp_experience):
        timeline_data.append({
            'Year': current_year - emp_experience + i,
            'Experience': i,
            'Salary': estimated_salary * (0.6 + (i * 0.4 / emp_experience)),
            'Status': 'Past'
        })

    timeline_data.append({
        'Year': current_year,
        'Experience': emp_experience,
        'Salary': estimated_salary,
        'Status': 'Present'
    })

    for i in range(1, 11):
        future_salary = estimated_salary * (1.08 ** i)
        timeline_data.append({
            'Year': current_year + i,
            'Experience': emp_experience + i,
            'Salary': future_salary,
            'Status': 'Future'
        })

    timeline_df = pd.DataFrame(timeline_data)
    fig_journey = go.Figure()

    past = timeline_df[timeline_df['Status'] == 'Past']
    if not past.empty:
        fig_journey.add_trace(go.Scatter(x=past['Year'], y=past['Salary'], mode='lines+markers', name='Past', line=dict(color='#87CEEB', width=4)))

    present = timeline_df[timeline_df['Status'] == 'Present']
    fig_journey.add_trace(go.Scatter(x=present['Year'], y=present['Salary'], mode='markers', name='Now', marker=dict(size=18, color='red', symbol='star')))

    future = timeline_df[timeline_df['Status'] == 'Future']
    fig_journey.add_trace(go.Scatter(x=future['Year'], y=future['Salary'], mode='lines+markers', name='Future', line=dict(color='#98FB98', width=4, dash='dot')))

    fig_journey.update_layout(title="📈 Career Salary Timeline", xaxis_title="Year", yaxis_title="Salary (₹)", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", height=600)
    st.plotly_chart(fig_journey, use_container_width=True, key="salary_journey_chart")

    # Salary Trends
    st.markdown("---")
    st.header("📈 Salary Market Trends & Economic Impact")
    years = list(range(2020, 2031))
    scenarios = {
        'Optimistic Growth': [1.12, 1.10, 1.08, 1.09, 1.11, 1.13, 1.12, 1.10, 1.09, 1.11, 1.12],
        'Conservative Growth': [1.05]*11,
        'Economic Recession': [0.98, 0.95, 1.01, 1.03, 1.08, 1.10, 1.07, 1.05, 1.06, 1.08, 1.09]
    }

    trend_data = []
    base_salary = estimated_salary * 0.8
    for scenario, multipliers in scenarios.items():
        current_salary = base_salary
        for year, multiplier in zip(years, multipliers):
            current_salary *= multiplier
            trend_data.append({
                'Year': year,
                'Salary': current_salary,
                'Scenario': scenario
            })

    trend_df = pd.DataFrame(trend_data)
    fig_trends = px.line(trend_df, x='Year', y='Salary', color='Scenario', title="📊 Salary Trends", labels={'Salary': '₹ Salary'}, height=600)
    fig_trends.add_vline(x=2024, line_dash="dot", line_color="yellow", annotation_text="Current Year")
    fig_trends.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white", hovermode="x unified")
    st.plotly_chart(fig_trends, use_container_width=True, key="salary_trend_chart")

    # Career Insights
    st.markdown("---")
    st.header("💡 Career Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="insight-box"><h4>💼 Career Tip</h4><p>{random.choice([
            "Research industry standards before salary discussions.",
            "A Master's degree can increase earnings by 20-30%.",
            "Switching roles every 2-3 years can accelerate growth.",
            "70% of jobs are filled through referrals. Network smart.",
            "Learn Python or SQL to boost salaries by 15-25%."
        ])}</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="insight-box"><h4>✨ Motivation</h4><p>{random.choice([
            "Believe in yourself. You're capable of more than you imagine.",
            "Success isn't final, failure isn't fatal. Keep going!",
            "Keep growing, and the salary will follow.",
            "Learn. Adapt. Grow. Earn more."
        ])}</p></div>""", unsafe_allow_html=True)

    # Developer Footer
    github_icon = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
    linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
    email_icon = "https://cdn-icons-png.flaticon.com/512/281/281769.png"
    st.markdown(f"""
    <div class="developer-box">
        <h4 style="margin: 0 0 1rem 0;">Profile of the Developer - Atharva Dalvi</h4>
        <div class="social-container">
            <a href="https://github.com/AtharvaDalvi2003" target="_blank" class="social-icon"><img src="{github_icon}" alt="GitHub"></a>
            <a href="https://www.linkedin.com/in/atharva-dalvi-01a281316/" target="_blank" class="social-icon"><img src="{linkedin_icon}" alt="LinkedIn"></a>
            <a href="mailto:atharva@example.com" class="social-icon"><img src="{email_icon}" alt="Contact"></a>
        </div>
    </div>
    """, unsafe_allow_html=True)

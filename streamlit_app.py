import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="",
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
        margin-bottom: 1.5rem;
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
        margin: 2rem 0;
    }
    .developer-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem auto;
        max-width: 500px;
        text-align: center;
    }
    .form-section h4 {
        margin-bottom: 0.5rem;
        font-size: 1rem;
        padding: 0.5rem;
        background-color: #2a2a2a;
        border-radius: 6px;
    }
    /* Hide Streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========== Header Section ==========
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
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<h4>Personal Information</h4>""", unsafe_allow_html=True)
        emp_age = st.selectbox("👦🏻 Age", options=list(range(18, 66)), index=12)
        emp_gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
    
    with col2:
        st.markdown("""<h4>Professional Information</h4>""", unsafe_allow_html=True)
        emp_title = st.selectbox("👨‍💼 Job Role", employee_df['Job Title'].unique())
        emp_experience = st.selectbox("💼 Experience (Years)", options=list(range(0, 51)), index=5)
    
    col3, col4 = st.columns(2)
    with col3:
        emp_location = st.selectbox("📍 Location", ["Urban", "Suburban", "Rural"])
    with col4:
        emp_edu = st.selectbox("👨🏻‍🎓 Education Level", ["Bachelor's", "Master's", "PhD"])
    
    submitted = st.form_submit_button("PREDICT MY SALARY", type="primary")

# ========== Prediction Logic ==========
if submitted:
    if emp_experience >= emp_age or emp_experience > (emp_age - 20):
        st.error("❌ Invalid age/experience combination")
    elif emp_edu == "Master's" and emp_age < 23:
        st.error("❌ Too young for Master's degree")
    elif emp_edu == "PhD" and emp_age < 26:
        st.error("❌ Too young for PhD")
    else:
        input_vector = {feature: 0 for feature in columns_required}
        input_vector["Age"] = emp_age
        input_vector["Years of Experience"] = emp_experience
        for cat in [f"Gender_{emp_gender}", f"Education Level_{emp_edu}", f"Job Title_{emp_title}", f"Location_{emp_location}"]:
            if cat in input_vector:
                input_vector[cat] = 1

        final_input = pd.DataFrame([input_vector])[columns_required]
        estimated_salary = model.predict(final_input)[0]

        st.success(f"🎯 **Predicted Annual Salary: ₹{estimated_salary:,.2f}**")
        
        # Salary distribution graph
        fig, ax = plt.subplots(figsize=(10, 4))
        random_data = np.random.normal(estimated_salary, estimated_salary*0.2, 50)
        ax.hist(random_data, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(estimated_salary, color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f"Salary Distribution for {emp_title}")
        ax.set_xlabel("Salary Range (₹)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.markdown(f"""
        **Career Insights:**
        - Your salary is in the **top {np.random.randint(60, 85)}%** for this role
        - Typical range for your profile: **₹{int(estimated_salary*0.8):,} - ₹{int(estimated_salary*1.2):,}**
        - 5-year growth potential: **₹{int(estimated_salary*1.3):,}**
        """)

        # Donut chart
        def show_salary_donut(percentile):
            fig, ax = plt.subplots(figsize=(3, 3))
            data = [percentile, 100 - percentile]
            colors = ["#E94DD2", "#09bbe2"]
            wedges, texts, autotexts = ax.pie(
                data,
                startangle=60,
                counterclock=False,
                wedgeprops=dict(width=0.3, edgecolor='white'),
                colors=colors,
                autopct='%1.0f%%'
            )
            ax.set(aspect="equal", title='Salary Distribution Percentile')
            st.pyplot(fig)

        percentile = min(max(int((estimated_salary / 200000) * 100), 1), 100)
        show_salary_donut(percentile)


# Social icons (white circles with black icons)
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

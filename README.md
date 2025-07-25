# 📊 AI Salary Predictor

A sophisticated machine learning-powered web application that predicts employee salaries based on various factors including age, experience, education, job title, location, and industry sector.

## 🌟 Features

- **Real-time Salary Prediction**: Uses machine learning to predict salaries based on multiple parameters
- **Interactive Data Visualization**: 
  - Salary distribution charts
  - Career timeline projections
  - Market trend analysis with multiple economic scenarios
- **Smart Validation**: Built-in logic to validate age/experience combinations and education requirements
- **Responsive Design**: Clean, modern UI with dark theme
- **Career Insights**: Provides personalized career tips and motivational messages

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Science**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (joblib for model persistence)
- **Visualization**: 
  - Matplotlib
  - Plotly Express
  - Plotly Graph Objects
- **Image Processing**: PIL (Python Imaging Library)
- **HTTP Requests**: Requests library

## 📋 Prerequisites

Before running this application, make sure you have Python 3.7+ installed on your system.

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/AtharvaDalvi2003/ai-salary-predictor.git
cd ai-salary-predictor
---
```
2.Install required packages
```
pip install -r requirements.txt
```
## 📁 Project Structure
```
ai-salary-predictor/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── final_salary_model.pkl          # Trained ML model
├── input_columns_config.pkl        # Model configuration
├── model_test_output.pkl           # Evaluation data
├── employee_income_data.csv        # Dataset
└── README.md                       # Project documentation
```
## 🏃‍♂️ Running the Application
1.Start the Streamlit server
```
streamlit run app.py
```
2. Open your browser Navigate to [http://localhost:8501 ](https://employee-salary-predictor-31.streamlit.app/)(Streamlit will usually open this automatically)

## 💻 Usage

1.Fill in the candidate details:

- Age (18-65)
- Gender (Male/Female)
- Job Title (from available options)
- Years of Experience (0-50)
- Work Location (Urban/Suburban/Rural)
- Education Level (Bachelor's/Master's/PhD)
- Industry Sector (Technology/Finance/Healthcare/Manufacturing/Retail)

2.Click "🚀 Predict" to get the salary prediction

3.Explore the results:

- View predicted annual salary
- Analyze salary distribution charts
- Explore career timeline projections
- Review market trend scenarios
- Get personalized career insights

## 🎨 Features Breakdown
Input Validation

- Age and experience compatibility checks
- Education level and age requirements validation
- Complete form submission validation
  
## Visualizations
- Salary Distribution: Histogram showing where your salary stands relative to others
- Career Timeline: Interactive chart showing past, present, and future salary projections
- Market Trends: Multiple economic scenarios (Optimistic, Conservative, Recession)

## Career Insights
- Randomized career tips
- Motivational messages
- Industry-specific advice

## 📊 Model Information
The application uses a pre-trained machine learning model that considers:

- Numerical Features: Age, Years of Experience
- Categorical Features: Gender, Education Level, Job Title, Location, Industry Sector
 
The model has been trained on employee income data and provides predictions with visualization of confidence intervals.

## 🚀 Live Demo

You can view the live demo of this application here https://employee-salary-predictor-31.streamlit.app/

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.



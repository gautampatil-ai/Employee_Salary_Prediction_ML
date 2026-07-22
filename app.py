# app.py
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING & ANIMATIONS
# ==========================================
st.markdown("""
<style>
    /* Gradient Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Card Styling */
    .css-1r6slb0, .css-12w0qpk {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Custom Header Banner */
    .header-box {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5);
    }
    
    .header-box h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-box p {
        color: #e2e8f0;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Animated Prediction Card */
    .result-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.3);
        animation: pulse 2s infinite, fadeIn 0.8s ease-in-out;
    }

    .result-title {
        color: #ecfdf5;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
        font-weight: 600;
    }

    .result-value {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }

    /* Pulse & Fade Animations */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 14px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }

    /* Custom Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0f172a;
        color: #94a3b8;
        text-align: center;
        padding: 10px;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource
def load_ml_model(model_path="Employee_Salary_model.pkl"):
    """
    Loads trained machine learning pipeline/model safely with proper error handling.
    Checks for both secondary file name (Employee_Salary_model.pkl) and model.pkl.
    """
    target_path = model_path
    if not os.path.exists(target_path):
        if os.path.exists("model.pkl"):
            target_path = "model.pkl"
        else:
            return None, f"Model file '{model_path}' or 'model.pkl' not found."
    
    try:
        with open(target_path, "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model, None
    except Exception as e:
        return None, f"Error loading model file: {str(e)}"


# ==========================================
# MAIN STREAMLIT APP
# ==========================================
def main():
    # --------------------------------------
    # SIDEBAR SECTION
    # --------------------------------------
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title("Project Center")
        st.markdown("---")
        
        st.subheader("📌 Project Info")
        st.write("""
        **AI Employee Salary Predictor** predicts industry-aligned salaries based on worker demographics, role, education, and career experience.
        """)
        
        st.markdown("---")
        st.subheader("⚙️ Model Details")
        st.markdown("""
        * **Algorithm:** XGBoost Regressor
        * **Framework:** Scikit-learn / XGBoost
        * **Features Processed:** 5
        * **Serialization:** Pickle (`.pkl`)
        """)
        
        st.markdown("---")
        st.subheader("🛠️ Technologies")
        st.markdown("""
        * Python
        * Streamlit
        * Scikit-Learn / XGBoost
        * Pandas & NumPy
        """)

    # --------------------------------------
    # MAIN HEADER SECTION
    # --------------------------------------
    st.markdown("""
    <div class="header-box">
        <h1>💼 AI Employee Salary Predictor</h1>
        <p>Machine Learning Based Salary Estimation System</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model, error_msg = load_ml_model()

    if error_msg:
        st.error(f"⚠️ {error_msg}")
        st.info("Please ensure your `.pkl` model file is uploaded to the root directory.")
        st.stop()

    # --------------------------------------
    # INPUT FEATURE FORM
    # --------------------------------------
    st.subheader("📋 Enter Employee Profile Details")
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", 
            min_value=18, 
            max_value=70, 
            value=30, 
            step=1,
            help="Employee's age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            help="Employee's self-identified gender"
        )

        education_level = st.selectbox(
            "Education Level",
            options=["Bachelor's", "Master's", "PhD", "High School"],
            help="Highest completed degree level"
        )

    with col2:
        job_title = st.selectbox(
            "Job Title",
            options=[
                "Data Scientist",
                "Data Analyst",
                "Software Engineer",
                "Machine Learning Engineer",
                "Manager",
                "Developer",
                "Project Manager",
                "Business Analyst",
                "Other"
            ],
            help="Current or prospective professional role"
        )

        years_experience = st.slider(
            "Years of Experience",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5,
            help="Total work experience in relevant fields"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------
    # ENCODING & PREDICTION LOGIC
    # --------------------------------------
    if st.button("🚀 Predict Salary"):
        # Map categorical variables into ordinal/numerical formats to match model training setup
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        
        education_map = {
            "High School": 0,
            "Bachelor's": 1,
            "Master's": 2,
            "PhD": 3
        }

        job_title_map = {
            "Software Engineer": 0,
            "Data Analyst": 1,
            "Manager": 2,
            "Data Scientist": 3,
            "Developer": 4,
            "Business Analyst": 5,
            "Project Manager": 6,
            "Machine Learning Engineer": 7,
            "Other": 8
        }

        # Encode input inputs
        encoded_gender = gender_map.get(gender, 0)
        encoded_edu = education_map.get(education_level, 1)
        encoded_job = job_title_map.get(job_title, 8)

        # Prepare feature DataFrame matching exact feature names in the trained model
        input_data = pd.DataFrame([{
            'Age': float(age),
            'Gender': encoded_gender,
            'Education Level': encoded_edu,
            'Job Title': encoded_job,
            'Years of Experience': float(years_experience)
        }])

        try:
            # Predict salary
            raw_prediction = model.predict(input_data)[0]
            predicted_salary = max(0, float(raw_prediction))  # Prevent negative salary estimations

            # Display prediction card with success state
            st.balloons()
            st.success("Analysis Complete! Here is your salary estimation:")

            formatted_salary = f"${predicted_salary:,.2f}"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Estimated Annual Salary</div>
                <div class="result-value">{formatted_salary} / year</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while calculating prediction: {str(e)}")

    # --------------------------------------
    # PROJECT & MODEL DETAILS SECTIONS
    # --------------------------------------
    st.markdown("---")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.subheader("ℹ️ About the Project")
        st.write("""
        This application utilizes advanced predictive analytics to estimate standard market compensation for prospective employee profiles. 
        It minimizes manual benchmarking efforts and streamlines HR salary decision-making workflows.
        """)

    with info_col2:
        st.subheader("📊 Model Capabilities")
        st.write("""
        * **Non-linear Modelling:** Accounted by XGBoost ensemble decision trees.
        * **Multi-variable Input:** Factors career tenure, educational background, age, and operational role.
        * **Production Ready:** Pre-compiled pipeline for instant online inference.
        """)

    # --------------------------------------
    # FOOTER
    # --------------------------------------
    st.markdown("""
    <div class="footer">
        Developed using Machine Learning | AI Employee Salary Predictor
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

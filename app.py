# app.py
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS & VISUAL STYLING
# ==========================================
st.markdown("""
<style>
    /* Main Theme & Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Hero Banner Header */
    .hero-banner {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 50%, #6366f1 100%);
        border-radius: 16px;
        padding: 32px 20px;
        text-align: center;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-banner p {
        color: #e2e8f0;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Clean Card Containers */
    .input-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }

    /* Streamlit Input Labels Visibility Fix */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        margin-bottom: 8px !important;
    }

    /* Custom Salary Result Box */
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        animation: fadeIn 0.6s ease-in-out;
    }

    .result-card .title {
        color: #d1fae5;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .result-card .value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 6px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Primary Action Button */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 14px 28px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }

    /* Custom Footer */
    .footer {
        margin-top: 40px;
        padding: 20px;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL PIPELINE
# ==========================================
@st.cache_resource
def load_ml_model():
    """
    Safely locates and loads the trained `.pkl` model.
    """
    possible_paths = ["Employee_Salary_model.pkl", "model.pkl", "salary_model.pkl"]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not model_path:
        return None, "No trained model file (`.pkl`) found in project root."

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model, None
    except Exception as e:
        return None, f"Error loading model file: {str(e)}"


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # --------------------------------------
    # SIDEBAR
    # --------------------------------------
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=85)
        st.title("Control Panel")
        st.markdown("---")

        st.subheader("⚙️ System Status")
        st.success("Model Status: Online")

        st.subheader("📌 Key Input Attributes")
        st.markdown("""
        * **Age**: Employee's current age
        * **Gender**: Candidate demographic
        * **Education Level**: Highest completed degree
        * **Job Title**: Target / Current role
        * **Experience**: Total career tenure
        """)

        st.markdown("---")
        if st.button("🔄 Reset Inputs", use_container_width=True):
            st.rerun()

    # --------------------------------------
    # HERO HEADER
    # --------------------------------------
    st.markdown("""
    <div class="hero-banner">
        <h1>💼 Employee Salary Predictor</h1>
        <p>Enter employee details below to receive an instant machine learning-driven salary estimate.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Trained Model
    model, error_msg = load_ml_model()

    if error_msg:
        st.error(f"⚠️ {error_msg}")
        st.info("Ensure your trained `.pkl` model file is saved in the same directory as `app.py`.")
        st.stop()

    # --------------------------------------
    # INPUT SECTION (ORGANIZED LAYOUT)
    # --------------------------------------
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📋 Profile Information")
    st.write("Adjust the fields below to customize the profile details.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 1: Demographic & General Inputs
    row1_col1, row1_col2, row1_col3 = st.columns(3, gap="large")

    with row1_col1:
        age = st.number_input(
            "👤 Candidate Age",
            min_value=18,
            max_value=75,
            value=30,
            step=1,
            help="Select employee age in years"
        )

    with row1_col2:
        gender = st.selectbox(
            "⚧ Gender Identity",
            options=["Male", "Female", "Other"],
            index=0,
            help="Select demographic identity"
        )

    with row1_col3:
        education_level = st.selectbox(
            "🎓 Education Level",
            options=["High School", "Bachelor's", "Master's", "PhD"],
            index=1,
            help="Select highest degree earned"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: Professional Details
    row2_col1, row2_col2 = st.columns([1, 1], gap="large")

    with row2_col1:
        job_title = st.selectbox(
            "💼 Job Title / Designation",
            options=[
                "Software Engineer",
                "Data Analyst",
                "Data Scientist",
                "Machine Learning Engineer",
                "Project Manager",
                "Business Analyst",
                "Developer",
                "Manager",
                "Other"
            ],
            index=0,
            help="Primary functional role"
        )

    with row2_col2:
        years_experience = st.slider(
            "⏳ Total Work Experience (Years)",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5,
            help="Total professional experience"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Validation Warning
    if years_experience > (age - 16):
        st.warning("⚠️ Work experience appears unusually high relative to the candidate's age. Please verify the entries.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------
    # PREDICTION LOGIC & VISUALIZATION
    # --------------------------------------
    if st.button("🚀 Calculate Estimated Salary", use_container_width=True):
        # Mappings matching categorical encoders
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
        job_map = {
            "Software Engineer": 0, "Data Analyst": 1, "Manager": 2,
            "Data Scientist": 3, "Developer": 4, "Business Analyst": 5,
            "Project Manager": 6, "Machine Learning Engineer": 7, "Other": 8
        }

        # Build feature DataFrame matching trained model format
        input_data = pd.DataFrame([{
            'Age': float(age),
            'Gender': gender_map.get(gender, 0),
            'Education Level': education_map.get(education_level, 1),
            'Job Title': job_map.get(job_title, 8),
            'Years of Experience': float(years_experience)
        }])

        try:
            # Generate Prediction
            raw_pred = model.predict(input_data)[0]
            predicted_salary = max(18000.0, float(raw_pred))

            st.balloons()

            # Output Banner
            st.markdown(f"""
            <div class="result-card">
                <div class="title">Estimated Annual Compensation</div>
                <div class="value">${predicted_salary:,.2f} / year</div>
            </div>
            """, unsafe_allow_html=True)

            # Key Metric Breakdown Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Monthly Estimated Base", f"${predicted_salary/12:,.2f}")
            m2.metric("Bi-Weekly Paycheck", f"${predicted_salary/26:,.2f}")
            m3.metric("Experience Tier", "Senior" if years_experience >= 8 else ("Mid-Level" if years_experience >= 3 else "Entry-Level"))
            m4.metric("Role Selected", job_title)

            st.markdown("---")

            # Compensation Position & Trend Analysis
            col_chart1, col_chart2 = st.columns([1, 1], gap="large")

            with col_chart1:
                st.subheader("📊 Market Compensation Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_salary,
                    number={'prefix': "$", 'valueformat': ",.0f"},
                    gauge={
                        'axis': {'range': [20000, 220000]},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [20000, 70000], 'color': '#1e293b'},
                            {'range': [70000, 130000], 'color': '#334155'},
                            {'range': [130000, 220000], 'color': '#475569'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_chart2:
                st.subheader("📈 Projected Growth Trajectory")
                exp_range = np.linspace(0, 25, 25)
                trend_df = pd.DataFrame({
                    'Years of Experience': exp_range,
                    'Estimated Growth Curve': predicted_salary * (1 + (exp_range - years_experience) * 0.04)
                })

                fig_trend = px.line(
                    trend_df,
                    x='Years of Experience',
                    y='Estimated Growth Curve',
                    labels={'Estimated Growth Curve': 'Salary ($)'},
                    template="plotly_dark",
                    color_discrete_sequence=['#10b981']
                )
                fig_trend.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

    # --------------------------------------
    # FOOTER
    # --------------------------------------
    st.markdown("""
    <div class="footer">
        AI Employee Salary Predictor • Powered by Streamlit & Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

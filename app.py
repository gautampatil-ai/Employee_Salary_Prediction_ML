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
    page_title="Enterprise Salary Predictor AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #172033 100%);
        color: #f1f5f9;
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Main Container Cards */
    .input-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }

    /* Professional Banner */
    .hero-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #6366f1 100%);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .hero-banner p {
        color: #e2e8f0;
        font-size: 1rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Prediction Display Card */
    .salary-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25);
    }

    .salary-box .title {
        color: #ecfdf5;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    .salary-box .amount {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Button Custom Styling */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }

    /* Custom Footer */
    .footer {
        margin-top: 50px;
        padding: 15px;
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
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
    Safely locates and loads the trained model executable.
    """
    possible_paths = ["Employee_Salary_model.pkl", "model.pkl", "salary_model.pkl"]
    model_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not model_path:
        return None, "Model file (`.pkl`) not found in project directory."

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    # --------------------------------------
    # SIDEBAR
    # --------------------------------------
    with st.sidebar:
        st.title("💼 System Overview")
        st.markdown("---")
        
        st.subheader("⚙️ System Status")
        st.success("Model Status: Online & Ready")
        
        st.subheader("📋 Input Features")
        st.markdown("""
        * **Age:** Candidate age (18–70)
        * **Gender:** Demographic feature
        * **Education:** Highest degree level
        * **Job Title:** Operational role
        * **Experience:** Tenure in years
        """)

        st.markdown("---")
        if st.button("🔄 Reset Parameters", use_container_width=True):
            st.rerun()

    # --------------------------------------
    # HERO BANNER
    # --------------------------------------
    st.markdown("""
    <div class="hero-banner">
        <h1>AI Employee Salary Predictor</h1>
        <p>Enterprise Compensation Analytics & Market Salary Benchmark Tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model, error_msg = load_ml_model()

    if error_msg:
        st.error(f"⚠️ {error_msg}")
        st.info("Please upload your `.pkl` model file to the root workspace folder to enable predictions.")
        st.stop()

    # --------------------------------------
    # WORKSPACE TABS
    # --------------------------------------
    tab_single, tab_batch, tab_analytics = st.tabs([
        "👤 Individual Prediction", 
        "📂 Batch Processing (CSV)", 
        "📊 Salary Trajectory Analytics"
    ])

    # ======================================
    # TAB 1: INDIVIDUAL PREDICTION
    # ======================================
    with tab_single:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("📌 Enter Employee Attributes")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input(
                "Age", 
                min_value=18, 
                max_value=70, 
                value=30, 
                step=1,
                help="Employee's age"
            )

            gender = st.selectbox(
                "Gender",
                options=["Male", "Female", "Other"],
                help="Gender identity"
            )

        with c2:
            education_level = st.selectbox(
                "Education Level",
                options=["High School", "Bachelor's", "Master's", "PhD"],
                index=1,
                help="Highest completed degree"
            )

            job_title = st.selectbox(
                "Job Title / Role",
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
                help="Current job designation"
            )

        with c3:
            years_experience = st.slider(
                "Years of Experience",
                min_value=0.0,
                max_value=40.0,
                value=5.0,
                step=0.5,
                help="Total professional experience in years"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Validation Logic
        if years_experience > (age - 16):
            st.warning("⚠️ Work experience seems unrealistically high relative to candidate age. Please verify inputs.")

        # Predict Button
        if st.button("🚀 Calculate Estimated Salary", use_container_width=True):
            # Numeric Feature Encodings
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            job_map = {
                "Software Engineer": 0, "Data Analyst": 1, "Manager": 2,
                "Data Scientist": 3, "Developer": 4, "Business Analyst": 5,
                "Project Manager": 6, "Machine Learning Engineer": 7, "Other": 8
            }

            # Prepare Data Frame
            input_data = pd.DataFrame([{
                'Age': float(age),
                'Gender': gender_map.get(gender, 0),
                'Education Level': education_map.get(education_level, 1),
                'Job Title': job_map.get(job_title, 8),
                'Years of Experience': float(years_experience)
            }])

            try:
                # Predict
                raw_prediction = model.predict(input_data)[0]
                predicted_salary = max(18000.0, float(raw_prediction))  # Floor value

                st.balloons()

                # Salary Result Card
                st.markdown(f"""
                <div class="salary-box">
                    <div class="title">Estimated Annual Compensation</div>
                    <div class="amount">${predicted_salary:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics Summary Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Estimated Monthly Pay", f"${predicted_salary/12:,.2f}")
                m2.metric("Tenure Class", "Senior" if years_experience >= 8 else ("Mid-Level" if years_experience >= 3 else "Entry-Level"))
                m3.metric("Role Selected", job_title)

                # Interactive Gauge Chart
                st.subheader("📈 Compensation Gauge Benchmark")
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
                    height=260,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    # ======================================
    # TAB 2: BATCH CSV PROCESSING
    # ======================================
    with tab_batch:
        st.subheader("📁 Bulk Prediction Tool")
        st.write("Upload a CSV dataset with columns: `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`.")

        uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("📋 **Preview Loaded Dataset:**", df.head(3))

            if st.button("⚡ Run Batch Predictions"):
                try:
                    df_encoded = df.copy()
                    
                    gender_map = {"Male": 0, "Female": 1, "Other": 2}
                    education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
                    job_map = {
                        "Software Engineer": 0, "Data Analyst": 1, "Manager": 2,
                        "Data Scientist": 3, "Developer": 4, "Business Analyst": 5,
                        "Project Manager": 6, "Machine Learning Engineer": 7, "Other": 8
                    }

                    if "Gender" in df_encoded.columns and df_encoded["Gender"].dtype == 'object':
                        df_encoded["Gender"] = df_encoded["Gender"].map(gender_map).fillna(0)
                    if "Education Level" in df_encoded.columns and df_encoded["Education Level"].dtype == 'object':
                        df_encoded["Education Level"] = df_encoded["Education Level"].map(education_map).fillna(1)
                    if "Job Title" in df_encoded.columns and df_encoded["Job Title"].dtype == 'object':
                        df_encoded["Job Title"] = df_encoded["Job Title"].map(job_map).fillna(8)

                    predictions = model.predict(df_encoded)
                    df["Predicted Salary ($)"] = [round(max(18000.0, float(p)), 2) for p in predictions]

                    st.success("✅ Batch predictions complete!")
                    st.dataframe(df, use_container_width=True)

                    # Export Option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Export CSV",
                        data=csv,
                        file_name="salary_predictions_export.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error processing CSV file: {str(e)}")

    # ======================================
    # TAB 3: INDUSTRY ANALYTICS
    # ======================================
    with tab_analytics:
        st.subheader("📊 Industry Salary Trajectory Chart")

        exp_range = np.linspace(0, 25, 30)
        trend_df = pd.DataFrame({
            'Years of Experience': exp_range,
            'Bachelor Degree': 48000 + (exp_range ** 1.25) * 3500,
            'Master Degree': 60000 + (exp_range ** 1.25) * 4000,
            'PhD Degree': 75000 + (exp_range ** 1.25) * 4600
        })

        fig_trend = px.line(
            trend_df,
            x='Years of Experience',
            y=['Bachelor Degree', 'Master Degree', 'PhD Degree'],
            title="Estimated Salary Growth Curve by Education Tier",
            template="plotly_dark",
            color_discrete_sequence=['#3b82f6', '#8b5cf6', '#10b981']
        )
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trend, use_container_width=True)

    # --------------------------------------
    # FOOTER
    # --------------------------------------
    st.markdown("""
    <div class="footer">
        Enterprise AI Salary Predictor System • Built with Streamlit & Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

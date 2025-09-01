import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from bertopic import BERTopic
from transformers import pipeline as hf_pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import json
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Proactive Customer Retention Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Modern Aesthetic UI ---
st.markdown("""
<style>
    /* Background and Main Container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Sidebar Styling */
    .st-emotion-cache-16txtl3 {
        background: #2c3e50 !important;
        color: white !important;
        border-radius: 12px;
        padding: 2rem 1rem;
    }
    .st-emotion-cache-16txtl3 h1, 
    .st-emotion-cache-16txtl3 h2, 
    .st-emotion-cache-16txtl3 h3, 
    .st-emotion-cache-16txtl3 h4 {
        color: white !important;
    }

    /* Metric Cards */
    .st-emotion-cache-1r6slb0 {
        background: #ffffffdd;
        border: none;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .st-emotion-cache-1r6slb0:hover {
        box-shadow: 0 12px 24px rgba(0,0,0,0.25);
        transform: translateY(-6px);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2a5298, #1e3c72);
        transform: scale(1.05);
    }

    /* Expanders */
    .st-emotion-cache-p5msec {
        border-radius: 12px;
        background: #ffffffee;
        border: none;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        padding: 0.8rem;
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 600;
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching Core Models (Lighter ones) ---
@st.cache_resource
def load_core_models():
    """Load the lighter, essential models for prediction and explanation."""
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        model = joblib.load('churn_model.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        df_original = pd.read_csv('telco_churn_with_all_feedback.csv')
        return preprocessor, model, explainer, df_original
    except FileNotFoundError as e:
        st.error(f"Error loading core model files: {e}. Please ensure preprocessor.pkl, churn_model.pkl, and shap_explainer.pkl are present.")
        return None, None, None, None

# --- Caching Heavy NLP Models (Lazy Loading) ---
@st.cache_resource
def get_nlp_models():
    """Load the heavy NLP models only when needed."""
    st.info("Loading NLP models for the first time... this may take a moment.")
    try:
        topic_model = joblib.load('bertopic_model.pkl')
        sentiment_pipeline = hf_pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return topic_model, sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading NLP models: {e}. Please ensure bertopic_model.pkl is present and you have an internet connection.")
        return None, None

# --- Load Core Models at Start ---
preprocessor, model, explainer, df_original = load_core_models()

# --- Text Preprocessing Function for Topic Modeling ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_for_topic_modeling(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Generative AI Recommender Function ---
def generate_recommendation(churn_prob, shap_values, processed_df, topic):
    # 🔐 Replace with your actual Gemini API key
    API_KEY = ""  # <-- Use your working key
    
    # ✅ Choose from available models
    MODEL = "gemini-2.0-flash"  # or "gemini-2.0-pro"
    
    if not API_KEY:
        return "Error: Gemini API key is missing. Please add your key to the script."

    feature_names = processed_df.columns
    shap_abs = np.abs(shap_values)
    top_indices = np.argsort(shap_abs)[-3:]
    drivers = [
        f"- **{feature_names[i].replace('_', ' ').title()}**: "
        f"{'increases' if shap_values[i] > 0 else 'decreases'} churn risk"
        for i in top_indices
    ]
    drivers_text = "\n".join(drivers)

    prompt = f"""
As a Customer Retention Expert, analyze the following customer profile and generate a proactive retention strategy.

**Customer Analysis:**
- **Churn Probability:** {churn_prob:.1%} (High Risk)
- **Primary Feedback Topic:** {topic}
- **Top 3 Churn Drivers:**
{drivers_text}

**Your Task:**
Provide a concise, actionable retention plan in Markdown with two sections:
1. **Root Cause Analysis**
2. **Recommended Action**
"""

    # ✅ Correct endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.HTTPError as e:
        st.error(f"[HTTP ERROR] {e}\nResponse: {response.text}")
        return "❌ HTTP Error occurred while calling Gemini API."
    except Exception as e:
        st.error(f"[ERROR] {e}")
        return "❌ Unexpected error occurred while calling Gemini API."


# --- UI Layout ---
st.title("🚀 Proactive Customer Retention Engine")
st.markdown("Enter customer details to predict churn, understand the drivers, and analyze feedback.")

st.sidebar.header("👤 Customer Profile Input")

if all([preprocessor, model, explainer, df_original is not None]):
    # --- Input Widgets ---
    col1, col2 = st.sidebar.columns(2)
    tenure = col1.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = col2.slider("Monthly Charges ($)", 0.0, 120.0, 75.0, 0.01)
    total_charges = col1.slider("Total Charges ($)", 0.0, 9000.0, 900.0, 0.01)

    senior_citizen_option = col2.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    senior_citizen_value = 1 if senior_citizen_option == "Yes" else 0

    categorical_features = df_original.select_dtypes(include=['object']).drop(columns=['customerID', 'Churn', 'PromptInput', 'CustomerFeedback']).columns
    input_data = {}
    for i, feature in enumerate(categorical_features):
        options = df_original[feature].unique().tolist()
        default_index = 0
        if feature == 'Contract':
            try: default_index = options.index('Month-to-month')
            except ValueError: default_index = 0
        
        if i % 2 == 0:
            input_data[feature] = col1.selectbox(feature, options, index=default_index)
        else:
            input_data[feature] = col2.selectbox(feature, options, index=default_index)

    customer_feedback = st.sidebar.text_area("Customer Feedback", "The internet speed has been inconsistent lately, and the price seems a bit high for the service I'm getting.")
    
    if st.sidebar.button("Analyze Customer", use_container_width=True):
        topic_model, sentiment_pipeline = get_nlp_models()
        if not all([topic_model, sentiment_pipeline]):
            st.stop()

        with st.spinner('🔍 Analyzing Customer Data...'):
            input_df = pd.DataFrame([input_data])
            input_df['tenure'] = tenure
            input_df['MonthlyCharges'] = monthly_charges
            input_df['TotalCharges'] = total_charges
            input_df['SeniorCitizen'] = senior_citizen_value

            sentiment_result = sentiment_pipeline(customer_feedback[:512])[0]
            st.session_state.sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
            st.session_state.sentiment_label = sentiment_result['label']
            input_df['SentimentScore'] = st.session_state.sentiment_score

            expected_cols = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
            input_df_ordered = input_df[expected_cols]

            processed_input = preprocessor.transform(input_df_ordered)
            st.session_state.prediction_proba = model.predict_proba(processed_input)[0][1]
            st.session_state.prediction = (st.session_state.prediction_proba > 0.5)

            processed_feedback = preprocess_for_topic_modeling(customer_feedback)
            if processed_feedback:
                topic_num, _ = topic_model.transform([processed_feedback])
                topic_name_raw = topic_model.get_topic_info(topic_num[0])['Name'].iloc[0]
                st.session_state.topic_name = topic_name_raw.split('_', 1)[1].replace('_', ' ').title()
            else:
                st.session_state.topic_name = "N/A"

            feature_names = np.concatenate([preprocessor.transformers_[0][2], preprocessor.named_transformers_['cat'].get_feature_names_out()])
            st.session_state.processed_input_df = pd.DataFrame(processed_input, columns=feature_names)
            st.session_state.shap_values = explainer.shap_values(st.session_state.processed_input_df)
            st.session_state.explainer_expected_value = explainer.expected_value
            st.session_state.analysis_run = True

    if 'analysis_run' in st.session_state and st.session_state.analysis_run:
        st.header("📊 Analysis Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Churn Probability", f"{st.session_state.prediction_proba:.1%}", "High Risk" if st.session_state.prediction else "Low Risk", delta_color="inverse")
        res_col2.metric("Predicted Sentiment", st.session_state.sentiment_label, f"{st.session_state.sentiment_score:.2f} Score")
        res_col3.metric("Feedback Topic", st.session_state.topic_name)

        st.markdown("---")
        
        with st.expander("🔬 View Detailed Churn Drivers (SHAP Analysis)", expanded=True):
            # FIX: Render the SHAP plot as a static matplotlib image for reliability
            # Convert shap values and input to shap.Explanation format
            shap_exp = shap.Explanation(
            values=st.session_state.shap_values[0],
            base_values=st.session_state.explainer_expected_value,
            data=st.session_state.processed_input_df.iloc[0],
            feature_names=st.session_state.processed_input_df.columns.tolist()
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_exp, show=False)
            st.pyplot(fig)
            plt.close(fig)
            
            st.info("Red features increase churn risk; blue features decrease it.")
        
        if st.session_state.prediction:
            st.markdown("---")
            st.subheader("💡 AI-Powered Retention Strategy")
            if st.button("Generate Retention Suggestion", use_container_width=True):
                with st.spinner("🧠 Consulting with Retention Expert AI..."):
                    recommendation = generate_recommendation(
                        st.session_state.prediction_proba, 
                        st.session_state.shap_values[0,:], 
                        st.session_state.processed_input_df, 
                        st.session_state.topic_name
                    )
                    st.markdown(recommendation)

else:
    st.warning("Could not load core model files. The application cannot run.")


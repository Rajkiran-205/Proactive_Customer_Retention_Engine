# Proactive_Customer_Retention_Engine
Proactive Customer Retention Engine 🚀
An end-to-end data science project that predicts, explains, and recommends actions for customer churn using a multi-layered AI approach. This repository contains the full workflow, from data preprocessing and model training to a final interactive web application built with Streamlit.

(This is a sample image. It's highly recommended you replace it with a screenshot of your actual running application.)

📋 Table of Contents
Overview

✨ Key Features

🛠️ Tech Stack

📂 Project Structure

🚀 Getting Started

📈 Methodology

🔎 Overview
Customer churn is a critical problem for subscription-based businesses. This project moves beyond simple churn prediction by building a holistic "retention engine" that not only identifies at-risk customers with high accuracy but also provides deep, actionable insights for retention teams.

The core of this project is a Streamlit web application that integrates several AI components:

A high-performance XGBoost model to predict churn probability.

SHAP (SHapley Additive exPlanations) to explain the "why" behind each prediction.

NLP models for real-time sentiment analysis and topic modeling on customer feedback.

A Generative AI (powered by Google's Gemini) to synthesize all this information and create personalized retention strategies on the fly.

✨ Key Features
High-Accuracy Prediction: Utilizes an XGBoost classifier trained on telco data, achieving ~97% accuracy.

Explainable AI (XAI): Generates dynamic SHAP waterfall plots to show which features (e.g., contract type, tenure) are driving the churn risk for each customer.

Real-time NLP Analysis:

Sentiment Analysis: Instantly classifies customer feedback as positive or negative.

Topic Modeling: Uses BERTopic to categorize feedback into underlying themes (e.g., "Billing Issues," "Service Speed").

AI-Powered Retention Strategies: The final layer uses a generative model to craft unique, context-aware retention scripts based on the churn probability, SHAP values, and NLP insights.

Interactive Dashboard: A clean, modern, and user-friendly interface built with Streamlit for easy interaction.

🛠️ Tech Stack
Backend & ML: Python, Scikit-learn, XGBoost, SHAP

NLP: BERTopic, Transformers (Hugging Face), NLTK

Web Framework: Streamlit

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Development Environment: Jupyter Notebook

📂 Project Structure
The repository is structured to show the complete end-to-end machine learning lifecycle.

.
├── 📄 Telco.ipynb                   # 1. Data Cleaning, Preprocessing & Feature Engineering
├── 📄 TextPreprocessing_TopicModeling.ipynb # 2. NLP: Text Cleaning and BERTopic model training
├── 📄 ModelTraining_Evaluation.ipynb    # 3. XGBoost model training and performance evaluation
├── 📄 interpretation_SHAP.ipynb       # 4. SHAP explainer creation and analysis
├── 🚀 app.py                        # 5. The final Streamlit web application
├── 📦 *.pkl                         # Saved artifacts (models, preprocessor, explainer)
└── 📄 README.md                     # This file

🚀 Getting Started
To run this project locally, follow these steps:

1. Clone the Repository

git clone [https://github.com/your-username/proactive-customer-retention-engine.git](https://github.com/your-username/proactive-customer-retention-engine.git)
cd proactive-customer-retention-engine

2. Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
First, create a requirements.txt file with all the necessary libraries. Then run:

pip install -r requirements.txt

Note: You will need to generate a requirements.txt file listing all libraries like pandas, streamlit, xgboost, shap, bertopic, etc.

4. Run the Notebooks (Optional)
To understand the workflow, you can run the Jupyter Notebooks in the specified order (1-4). This will regenerate the .pkl model files.

5. Launch the Streamlit App
Make sure all the .pkl artifact files are in the same directory as app.py. Then run:

streamlit run app.py

Open your web browser to http://localhost:8501 to view the application.

📈 Methodology
The project follows a structured machine learning pipeline:

Data Preprocessing: The initial telco dataset is cleaned, with missing values handled and data types corrected. A ColumnTransformer pipeline is used to scale numerical features and one-hot encode categorical features.

NLP Pipeline: Customer feedback text is cleaned, lemmatized, and used to train a BERTopic model to identify recurring themes.

Model Training: An XGBoost classifier is trained on the preprocessed structured data. Class imbalance is handled using scale_pos_weight for better performance on the minority (churn) class.

Model Interpretation: A shap.Explainer is created using the trained model and the training data as a background dataset to enable explanations for new predictions.

Deployment: All trained artifacts (the preprocessor, XGBoost model, BERTopic model, and SHAP explainer) are serialized using joblib and loaded into the app.py script to serve real-time predictions and insights.

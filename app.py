import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import io
from datetime import datetime



# PAGE CONFIG

st.set_page_config(
    page_title="Breast Cancer Coimbra | ML App",
    page_icon="🩺",
    layout="wide"
)

# LOAD CSS

def load_css(file):
    with open(file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# SESSION STATE INITIALIZATION
 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# LOAD DATA & MODELS

@st.cache_data
def load_data():
    return pd.read_csv("breast_data.csv")

df = load_data()

FEATURE_COLS = [
    "Age","BMI","Glucose","Insulin","HOMA",
    "Leptin","Adiponectin","Resistin","MCP.1"
]
TARGET_COL = "Classification"

scaler = joblib.load("scaler.pkl")

ALL_MODELS = {
    "Logistic Regression": joblib.load("lr_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "MLP": joblib.load("mlp_model.pkl"),
    "Decision Tree": joblib.load("tree_model.pkl"),
}

corpus = joblib.load("corpus.pkl")

# for pdf report
def generate_pdf_report(inputs, prediction, prob, model, risk):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Breast Cancer Prediction Report")

    c.setFont("Helvetica", 9)
    c.drawString(50, y-20, datetime.now().strftime("%d-%m-%Y %H:%M"))

    y -= 50
    for f, v in zip(FEATURE_COLS, inputs):
        c.drawString(50, y, f"{f}: {v}")
        y -= 12

    y -= 10
    c.drawString(50, y, f"Model: {model}")
    c.drawString(50, y-12, f"Result: {prediction}")
    c.drawString(50, y-24, f"Prob: {prob:.2f}%")
    c.drawString(50, y-36, f"Risk: {risk}")

    c.drawString(50, y-60, "Educational use only. Not a medical diagnosis.")
    c.save()
    buf.seek(0)
    return buf


# TRAIN MODELS ONCE

@st.cache_resource
def train_models():
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    for model in ALL_MODELS.values():
        model.fit(X_train, y_train)

    return X_test, y_test

X_test, y_test = train_models()

class_counts = df[TARGET_COL].value_counts().sort_index().values


# SESSION STATE

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# HEADER

st.markdown(
    "<h1 style='margin-top:40px;'>🩺 Breast Cancer Prediction System</h1>",
    unsafe_allow_html=True
)

# NAVIGATION BAR
if st.session_state.logged_in:
    page = option_menu(
    None,
    ["Home", "Prediction", "Dataset Summary", "Model Visualization", "Chatbot", "About"],
    icons=["house", "activity", "table", "bar-chart", "chat-dots", "info-circle"],
    orientation="horizontal"
)    
else:
    page = "Home"

# HOME
if page == "Home":
    if not st.session_state.logged_in:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        st.subheader("🔐 Login Required")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        credentials = {
            "karishma": "kari123",
            "pawan": "paw123",
            "kanika": "kan123"
        }

        if st.button("Login"):
            if username in credentials and credentials[username] == password:
                st.success("Login Successful ✅")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.container():
          st.subheader("Welcome")
          st.write(
        "This application predicts **Breast Cancer vs Healthy** "
        "using machine learning models trained on the "
        "**Breast Cancer Coimbra Dataset**."
    )

        st.markdown("</div>", unsafe_allow_html=True)

# PREDICTION

elif page == "Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Breast Cancer Prediction")

    st.write("Patient Input")
    st.write(" Adjust the sliders based on patient blood test values")
    age = st.slider("Age", 18, 100, 45)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    glucose = st.slider("Glucose", 40, 300, 100)
    insulin = st.slider("Insulin", 0.0, 900.0, 100.0)
    homa = st.slider("HOMA", 0.0, 80.0, 2.0)
    leptin = st.slider("Leptin", 0.0, 200.0, 20.0)
    adiponectin = st.slider("Adiponectin", 0.0, 50.0, 10.0)
    resistin = st.slider("Resistin", 0.0, 50.0, 10.0)
    mcp1 = st.slider("MCP-1", 0.0, 1000.0, 300.0)

# Model input (order matters)
    inputs = [age, bmi, glucose, insulin, homa,leptin, adiponectin, resistin, mcp1]

    model_name = st.selectbox("Select Model", list(ALL_MODELS.keys()))

    # NOW USE pred
    if st.button("Predict"):
     X = scaler.transform(np.array(inputs).reshape(1, -1))
     model = ALL_MODELS[model_name]

     pred = int(model.predict(X)[0])
     prob = model.predict_proba(X)[0][1] * 100 if hasattr(model, "predict_proba") else 0

     if pred == 1:
        prediction_text = "Breast Cancer Detected"
        st.error("❌ Breast Cancer Detected")
     else:
        prediction_text = "Healthy"
        st.success("✅ Healthy")

     if prob > 70:
        risk = "High Risk"
     elif prob > 40:
        risk = "Moderate Risk"
     else:
        risk = "Low Risk"

     st.info(f"Probability of Cancer: {prob:.2f}%")

     pdf = generate_pdf_report(
        inputs,
        prediction_text,
        prob,
        model_name,
        risk
    )
     st.download_button(
        "📄 Download Report",
        pdf,
        "Breast_Cancer_Report.pdf",
        "application/pdf"
    )

# DATASET SUMMARY

elif page == "Dataset Summary":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Summary")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Dataset Shape")
    st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.markdown("### Numerical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# MODEL VISUALIZATION

elif page == "Model Visualization":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Model Visualization")

    scores = []
    for name, model in ALL_MODELS.items():
        acc = accuracy_score(y_test, model.predict(X_test))
        scores.append([name, acc])

    acc_df = pd.DataFrame(scores, columns=["Model", "Accuracy"]) \
                .sort_values(by="Accuracy", ascending=False)

    st.dataframe(acc_df, use_container_width=True)

    # BAR CHART (STRUCTURE KEPT)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(acc_df["Model"], acc_df["Accuracy"], color="#c77d9b")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}", ha="center")

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        ["Healthy (0)", "Breast Cancer (1)"],
        class_counts,
        color=["#c77d9b", "#9d4edd"],
        alpha=0.85
    )

    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(bar.get_height()), ha="center")

    ax.set_title("Dataset Class Distribution")
    st.pyplot(fig)

    st.markdown("### 🔍 Feature Importance (Random Forest)")

    rf = ALL_MODELS["Random Forest"]
    if hasattr(rf, "feature_importances_"):
     imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf.feature_importances_
     }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.invert_yaxis()
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
 

    st.markdown("### 📊 Confusion Matrix")

    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",xticklabels=["Healthy", "Cancer"],yticklabels=["Healthy", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# CHATBOT

elif page == "Chatbot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💬 Dataset Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("💬 Chat History")
        if st.session_state.chat_history:
            for u, b in st.session_state.chat_history:
                st.write(f"**You:** {u}")
                st.write(f"**Bot:** {b}")
        else:
            st.write("No history yet.")
        
        if st.button(" Clear History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    def chatbot_response(user_text):
        user_words = set(re.findall(r'\w+', user_text.lower()))
        best_answer, best_score = None, 0

        for q, a in corpus.items():
            q_words = set(re.findall(r'\w+', q.lower()))
            score = len(user_words & q_words)
            if score > best_score:
                best_score = score
                best_answer = a

        if best_score >= 2:
            return best_answer
        return corpus.get("default", "Sorry, I don't understand your question.")

    user_input = st.text_input("Ask about Breast Cancer, Glucose, Insulin, etc.")

    if user_input:
        reply = chatbot_response(user_input)
        st.session_state.chat_history.append((user_input, reply))

    for u, b in st.session_state.chat_history:
        st.markdown(f" **You:** {u}")
        st.markdown(f" **Bot:** {b}")
        st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ABOUT

elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About Project")

    st.write("""
    ### 📂 Dataset
    Breast Cancer Coimbra Dataset  
    Blood biomarkers (Glucose, Insulin, Leptin, etc.) classify patients as:
    - **0 → Healthy**
    - **1 → Breast Cancer**

    ###  Objective
    - Real-time prediction
    - Multi-model performance comparison
    - Dataset-based chatbot
    - Classification reports

    ###  ML Models Used
    - KNN
    - Logistic Regression
    - Random Forest
    - SVM
    - MLP
    - Decision Tree

    ### Tech Stack
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas & Numpy
    - Matplotlib
    - Joblib

    """)

    st.write("""
             Developer
             - DEVEOP BY :**Karishma**
             """)
    st.markdown("</div>", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page Config
st.set_page_config(page_title="📧 Spam Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>📧 Spam Mail Detection App</h1>", unsafe_allow_html=True)

# Load & preprocess
@st.cache_data
def load_data():
    df = pd.read_csv('mail_data.csv')
    df = df.where((pd.notnull(df)), '')
    df.loc[df['Category'] == 'spam', 'Category'] = 0
    df.loc[df['Category'] == 'ham', 'Category'] = 1
    return df

@st.cache_data
def preprocess(df):
    def clean_text(text):
        text = re.sub(r'\W', ' ', text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        return text
    df['Message'] = df['Message'].apply(clean_text)
    return df

@st.cache_resource
def train_model(data):
    X = data['Message']
    Y = data['Category'].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_feat = vectorizer.fit_transform(X_train)
    X_test_feat = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_feat, Y_train)

    acc_train = accuracy_score(Y_train, model.predict(X_train_feat))
    acc_test = accuracy_score(Y_test, model.predict(X_test_feat))
    cm = confusion_matrix(Y_test, model.predict(X_test_feat))

    return model, vectorizer, acc_train, acc_test, cm

df = load_data()
df = preprocess(df)
model, vectorizer, acc_train, acc_test, cm = train_model(df)

# Sidebar input
st.sidebar.header("🛠 Input Options")
option = st.sidebar.radio("Choose Input Type:", ["Single Message", "Upload CSV"])

# Main prediction area
if option == "Single Message":
    st.subheader("📨 Predict a Single Message")
    message = st.text_area("Enter your email or message:")

    if st.button("🔍 Predict"):
        vect = vectorizer.transform([message])
        prediction = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0]

        st.success("✅ Prediction: **Ham**" if prediction == 1 else "🚫 Prediction: **Spam**")
        st.info(f"Confidence Score → Ham: {prob[1]:.2%}, Spam: {prob[0]:.2%}")

elif option == "Upload CSV":
    st.subheader("📁 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload a CSV with a 'Message' column", type=['csv'])

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        df_uploaded['Cleaned'] = df_uploaded['Message'].apply(lambda x: re.sub(r'\W', ' ', str(x)).lower())
        transformed = vectorizer.transform(df_uploaded['Cleaned'])
        preds = model.predict(transformed)
        probs = model.predict_proba(transformed)
        df_uploaded['Prediction'] = ["Ham" if p == 1 else "Spam" for p in preds]
        df_uploaded['Spam Probability'] = [f"{p[0]:.2%}" for p in probs]
        df_uploaded['Ham Probability'] = [f"{p[1]:.2%}" for p in probs]

        st.write(df_uploaded[['Message', 'Prediction', 'Ham Probability', 'Spam Probability']])
        csv = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

# Metrics
st.subheader("📊 Model Metrics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{acc_train:.2%}")
with col2:
    st.metric("Testing Accuracy", f"{acc_test:.2%}")

with st.expander("📉 Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'], ax=ax)
    st.pyplot(fig)

with st.expander("📋 Classification Report"):
    X = df['Message']
    Y = df['Category'].astype(int)
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    X_test_feat = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_feat)
    report = classification_report(Y_test, y_pred, target_names=["Spam", "Ham"], output_dict=False)
    st.text(report)

# Word Cloud
st.subheader("🌥 Word Cloud of Messages")
spam_words = ' '.join(df[df['Category'] == 0]['Message'])
ham_words = ' '.join(df[df['Category'] == 1]['Message'])

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🚫 Spam Words")
    spam_wc = WordCloud(width=400, height=200).generate(spam_words)
    st.image(spam_wc.to_array())
with col2:
    st.markdown("#### ✅ Ham Words")
    ham_wc = WordCloud(width=400, height=200).generate(ham_words)
    st.image(ham_wc.to_array())

# Footer
st.markdown("""---  
<p style='text-align: center;'>Built by Zain Ali — <a href='https://github.com/ZainAli0o/spam-detector' target='_blank'>View on GitHub</a></p>""", unsafe_allow_html=True)

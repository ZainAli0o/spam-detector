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

st.set_page_config(page_title="Spam Detector", layout="wide")
st.title("📧 Spam Mail Detection App")

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
    Y = data['Category'].astype('int')
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

st.sidebar.header("Enter Message or Upload CSV")
option = st.sidebar.radio("Choose Input Type:", ["Single Message", "Upload CSV"])

if option == "Single Message":
    message = st.text_area("Enter your email/message:")
    if st.button("Predict"):
        vect = vectorizer.transform([message])
        prediction = model.predict(vect)
        st.success("Prediction: 📬 Ham" if prediction[0] == 1 else "Prediction: 🚫 Spam")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV with a 'Message' column", type=['csv'])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        df_uploaded['Cleaned'] = df_uploaded['Message'].apply(lambda x: re.sub(r'\W', ' ', str(x)).lower())
        transformed = vectorizer.transform(df_uploaded['Cleaned'])
        preds = model.predict(transformed)
        df_uploaded['Prediction'] = ["Ham" if p == 1 else "Spam" for p in preds]
        st.write(df_uploaded[['Message', 'Prediction']])

st.subheader("📊 Model Accuracy")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Accuracy", f"{acc_train:.2%}")
with col2:
    st.metric("Testing Accuracy", f"{acc_test:.2%}")

st.subheader("🔍 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'], ax=ax)
st.pyplot(fig)

st.subheader("🌥 Word Clouds")

spam_words = ' '.join(df[df['Category'] == 0]['Message'])
ham_words = ' '.join(df[df['Category'] == 1]['Message'])

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Spam Messages")
    spam_wc = WordCloud(width=400, height=200).generate(spam_words)
    st.image(spam_wc.to_array())
with col2:
    st.markdown("#### Ham Messages")
    ham_wc = WordCloud(width=400, height=200).generate(ham_words)
    st.image(ham_wc.to_array())

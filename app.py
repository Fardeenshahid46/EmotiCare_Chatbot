#Required libraries
import streamlit as st
import pandas as pd
import torch.nn as nn
from datetime import datetime
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

#To load and prepare dataset
@st.cache_data
def load_dataset():
    df=pd.read_csv("Emotion_final.csv")
    return df

#To Load dataset
emotion_df=load_dataset()

def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z\s]","",text)
    return text

emotion_df["CleanText"]=emotion_df["Text"].apply(clean_text)
#To encode traget labels
le=LabelEncoder()
emotion_df["EmotionEncoded"]=le.fit_transform(emotion_df["Emotion"])

#To Vectorize text
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(emotion_df['CleanText'])
y=emotion_df["EmotionEncoded"]

#To train a simple model
model=LogisticRegression(max_iter=1000)
model.fit(X,y)

#Save for reuse
joblib.dump(model,"model.joblib")
joblib.dump(vectorizer,"vectorizer.joblib")
joblib.dump(le,"label_encoder.joblib")

#Rule-based override (temporary fix)
def keyword_override(text):
    text=text.lower()
    if "not feeling well" in text or "depressed" in text:
        return "sad"
    if "anxious" in text or "worried" in text:
        return "fear"
    if "excited" in text or "wow" in text:
        return "surprise"
    if "angry" in text or "furious" in text:
        return "angry"
    if "happy" in text or "glad" in text:
        return "happy"
    return None
    
#To predict from real model
@st.cache_data(show_spinner=False)
def predict_emotion(text):
    rule_emotion = keyword_override(text)
    if rule_emotion:
        return rule_emotion
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

#To initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

#Use streamlit for User Interface
st.set_page_config(page_title="EmotiCare Chatbot",page_icon="🧠",layout="centered")

with st.sidebar:
    st.markdown("👤 User Panel")
    st.write("Welcome to EmotiCare! Track your emotions below.")
    
    if st.button("📤 Export Chat History"):
        if not os.path.exists("logs"):
           os.makedirs("logs")
        df=pd.DataFrame(st.session_state.chat_history,columns=["Time","User","Bot","Emotion"])
        df.to_csv("logs/chat_logs.csv",index=False)
        st.success("✅ Exported chat to logs/chat_logs.csv ")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history.clear()
        log_path="logs/chat_logs.csv"
        if os.path.exists(log_path):
            with open(log_path,"w") as f:
                f.write("Time,User,Bot,Emotion\n")
        st.success("🧹 Chat history cleared!")
               
#App Title
st.markdown("""
        <div style="text-align:center; padding:20px 0px;">
        <h1 style="font-size=3em;color:#4A90E2;">🧠 EmotiCare Chatbot</h1>
        <p style="font-size:1.3em; color:#555">Understand Your Emotions with AI-Powered Converstaions</p>    
        """,unsafe_allow_html=True)
user_input=st.text_input("You:",key="user_input")
if user_input:
    predicted_emotion=predict_emotion(user_input)
    responses={
        'happy': "😊 I'm glad you're feeling happy today!",
        'sad': "😢 I'm here for you. Want to talk about it?",
        'angry': "😠 Let's take a deep breath together.",
        'fear': "😨 It's okay to be scared. You're not alone.",
        'surprise': "😲 Oh! That sounds unexpected!",
        'neutral': "🙂 Got it! I'm listening."
    }    
    bot_reply=responses.get(predicted_emotion,"I'm here to support you")
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append((timestamp,user_input,bot_reply,predicted_emotion))
    
st.markdown("---")
st.markdown("🗨️ Conversation")
for time,user,bot,emotion in reversed(st.session_state.chat_history):
    st.markdown(f"""
        <div style="background-color:gray;padding:10px;border-radius:10px;margin:10px 0px;">
        <b>You:</b>{user}<br> 
        <b>Bot({emotion}):</b>{bot}<br>
        <small>{time}</small>
        </div>       
            """,unsafe_allow_html=True)
st.markdown("""
<div style="background-color:blue; padding:15px; border-radius:10px; border: 1px solid #c3ddfd;">
    <strong>✨ AI Engine:</strong> Trained on Real Emotions Dataset<br>
</div>
""", unsafe_allow_html=True)
        
import os
import json
import datetime
import csv
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from JSON
file_path = "intents.json"
if not os.path.exists(file_path):
    st.error("Error: 'intents.json' file not found.")
    st.stop()

with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Prepare training data
tags, patterns = [], []
for intent in intents:
    for pattern in intent.get("patterns", []):
        tags.append(intent["tag"])
        patterns.append(pattern)

# Ensure there is data before training
if not patterns or not tags:
    st.error("Error: No training data found in 'intents.json'.")
    st.stop()

# Train the vectorizer and classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, tags)

# Emoji mapping based on chatbot responses
emoji_reactions = {
    "greeting": "👋",
    "goodbye": "👋😊",
    "age": "⏳",
    "thankyou": "🙏",
    "help": "💡",
    "unknown": "🤔",
}

# Chatbot function with error handling
def chatbot(input_text):
    if not input_text.strip():
        return "I'm here to chat! Please enter a message.", "unknown"

    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent.get("responses", ["Sorry, I don't have an answer for that."])), tag

    return "I'm not sure how to respond to that.", "unknown"

# Function to save chat history
def save_chat_log(user_input, response):
    file_name = "chat_log.csv"
    if not os.path.exists(file_name):
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, "a", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Function to load chat history
def load_chat_history():
    file_name = "chat_log.csv"
    history = []
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header row
            history = list(csv_reader)
    return history

# Streamlit App
def main():
    st.set_page_config(page_title="AI Chatbot 🤖", page_icon="💬", layout="wide")

    st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot with Emoji Reactions 💬</h1>", unsafe_allow_html=True)

    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Chat":
        st.write("### Welcome! Type your message below and chat with the AI 🤖")

        # Chat history session
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You:", key="user_input")

        if user_input:
            response, tag = chatbot(user_input)
            emoji = emoji_reactions.get(tag, "💬")

            # Store messages in session
            st.session_state.chat_history.append(("You", user_input, "🧑‍💻"))
            st.session_state.chat_history.append(("Chatbot", response, emoji))

            # Save chat log
            save_chat_log(user_input, response)

        # Display chat history dynamically
        st.write("### 🗨️ Chat History")
        for sender, message, emoji in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"<div style='text-align: right; background: #dcf8c6; padding: 10px; border-radius: 10px;'>{emoji} **{message}**</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background: #f1f0f0; padding: 10px; border-radius: 10px;'>{emoji} **{message}**</div>", unsafe_allow_html=True)

        # Quick reply buttons
        st.write("### ⚡ Quick Replies:")
        col1, col2, col3 = st.columns(3)
        if col1.button("Hello 👋"):
            st.session_state.chat_history.append(("You", "Hello", "🧑‍💻"))
            response, tag = chatbot("Hello")
            st.session_state.chat_history.append(("Chatbot", response, emoji_reactions.get(tag, "💬")))
        if col2.button("Help 💡"):
            st.session_state.chat_history.append(("You", "I need help", "🧑‍💻"))
            response, tag = chatbot("I need help")
            st.session_state.chat_history.append(("Chatbot", response, emoji_reactions.get(tag, "💬")))
        if col3.button("Goodbye 👋"):
            st.session_state.chat_history.append(("You", "Goodbye", "🧑‍💻"))
            response, tag = chatbot("Goodbye")
            st.session_state.chat_history.append(("Chatbot", response, emoji_reactions.get(tag, "💬")))

    elif choice == "Conversation History":
        st.header("🗂️ Conversation History")
        history = load_chat_history()

        if not history:
            st.write("No conversation history found.")
        else:
            for row in history:
                st.markdown(f"🧑‍💻 **User:** {row[0]}")
                st.markdown(f"🤖 **Chatbot:** {row[1]}")
                st.markdown(f"🕒 **Timestamp:** {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.subheader("ℹ️ About This Chatbot")
        st.write("""
        - 🤖 **AI-powered chatbot** using **Natural Language Processing (NLP)**.
        - 💡 **Trained with TF-IDF & Logistic Regression** for intent recognition.
        - 🌟 **Supports Emoji Reactions** to make conversations fun.
        - 📝 **Logs conversations for review**.
        - 🎨 **Beautiful interactive UI with Streamlit**.
        """)

if __name__ == "__main__":
    main()

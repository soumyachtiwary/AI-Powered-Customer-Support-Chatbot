from flask import Flask, render_template, request, jsonify
import sqlite3
import re
import math
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

app = Flask(__name__)
DB_NAME = "chatbot.db"
stemmer = PorterStemmer()

FAQS = [
    {
        "question": "How can I reset my password?",
        "answer": "You can reset your password from the account settings page by selecting 'Forgot Password' and following the instructions."
    },
    {
        "question": "How do I create an account?",
        "answer": "Click the Sign Up button, enter your details, verify your email address, and then log in."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact customer support through the support page or by sending an email to support@example.com."
    },
    {
        "question": "What are your business hours?",
        "answer": "Customer support is available Monday to Friday, 9:00 AM to 6:00 PM."
    },
    {
        "question": "How do I update my profile?",
        "answer": "Log in, open your Profile or Account Settings, edit the required information, and click Save."
    },
    {
        "question": "How can I check my order status?",
        "answer": "Open the Orders section after logging in and select the relevant order to view its current status."
    },
    {
        "question": "How do I cancel an order?",
        "answer": "Open your Orders section and select Cancel Order if cancellation is still available for that order."
    },
    {
        "question": "What payment methods are accepted?",
        "answer": "The system supports common online payment methods such as cards and digital payments, depending on the service configuration."
    }
]

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def preprocess(text):
    tokens = wordpunct_tokenize(text.lower())
    cleaned = []
    for token in tokens:
        if re.fullmatch(r"[a-zA-Z]+", token):
            cleaned.append(stemmer.stem(token))
    return cleaned

def cosine_similarity(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0

    a = Counter(tokens_a)
    b = Counter(tokens_b)
    vocabulary = set(a) | set(b)

    dot = sum(a[t] * b[t] for t in vocabulary)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

FAQ_TOKENS = [(faq, preprocess(faq["question"])) for faq in FAQS]

def get_response(message):
    user_tokens = preprocess(message)
    best_faq = None
    best_score = 0.0

    for faq, faq_tokens in FAQ_TOKENS:
        score = cosine_similarity(user_tokens, faq_tokens)
        if score > best_score:
            best_score = score
            best_faq = faq

    if best_faq and best_score >= 0.20:
        return best_faq["answer"]

    return (
        "I'm sorry, I couldn't find a suitable answer for that question. "
        "Please try rephrasing it or contact customer support."
    )

def save_log(user_message, bot_response):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO chat_logs (user_message, bot_response) VALUES (?, ?)",
        (user_message, bot_response)
    )
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"response": "Please enter a question."})

    response = get_response(message)
    save_log(message, response)
    return jsonify({"response": response})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)

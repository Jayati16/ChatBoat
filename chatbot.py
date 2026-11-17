"""
College Helpdesk Chatbot - intent classification with scikit-learn
------------------------------------------------------------------
Idea: the user types a sentence -> the model predicts the INTENT
(what the user wants) -> the bot replies with a matching answer.

Pipeline:  text cleaning -> TF-IDF vectorizer -> Logistic Regression
Install:   pip install scikit-learn
Run:       python chatbot.py
"""

import random
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------
# 1. DATASET: each intent has example sentences (patterns) and replies
# ----------------------------------------------------------------------
INTENTS = {
    "greeting": {
        "patterns": [
            "hi", "hello", "hey there", "good morning", "good evening",
            "hello bot", "hey", "hi there", "greetings", "is anyone there",
        ],
        "responses": ["Hello! How can I help you today?", "Hi there! Ask me anything."],
    },
    "goodbye": {
        "patterns": [
            "bye", "goodbye", "see you later", "talk to you soon", "ok bye",
            "i have to go", "see you", "catch you later", "bye bye", "good night",
        ],
        "responses": ["Goodbye! Have a great day.", "See you later!"],
    },
    "thanks": {
        "patterns": [
            "thanks", "thank you", "thanks a lot", "that was helpful",
            "great thank you", "many thanks", "appreciate it", "thank you so much",
            "awesome thanks", "cheers",
        ],
        "responses": ["You're welcome!", "Happy to help!"],
    },
    "admission": {
        "patterns": [
            "how can i apply for admission", "admission process", "how to get admission",
            "when do admissions start", "what is the last date to apply",
            "admission form", "i want to join the college", "how do i enroll",
            "eligibility for admission", "documents required for admission",
        ],
        "responses": [
            "Admissions are done through the online form on the college website. "
            "Keep your marksheets and ID proof ready."
        ],
    },
    "fees": {
        "patterns": [
            "what is the fee", "course fees", "how much is the tuition", "fee structure",
            "how much do i have to pay", "what are the charges", "fees for mca",
            "is there any scholarship", "can i pay fees in installments", "total cost of the course",
        ],
        "responses": [
            "Fee details depend on the course. Please check the fee structure on the "
            "website or contact the accounts office."
        ],
    },
    "timetable": {
        "patterns": [
            "show me the timetable", "class schedule", "when is my next lecture",
            "what are the class timings", "lecture timetable", "college timing",
            "what time does college start", "when do classes begin", "weekly schedule",
            "today's lectures",
        ],
        "responses": ["Classes usually run from morning to afternoon. Please check the timetable on the student portal."],
    },
    "exam": {
        "patterns": [
            "when are the exams", "exam date", "exam schedule", "semester exam dates",
            "when does the exam start", "exam time table", "when is the final exam",
            "hall ticket", "when will the exam be held", "internal exam dates",
        ],
        "responses": ["Exam dates are announced on the notice board and student portal. Hall tickets are issued a few days before."],
    },
    "library": {
        "patterns": [
            "library timings", "when does the library open", "how can i borrow a book",
            "library rules", "is the library open on sunday", "issue a book",
            "how many books can i take", "library card", "return a book", "reading room",
        ],
        "responses": ["The library is open on working days. You can borrow books with your library card."],
    },
    "contact": {
        "patterns": [
            "contact number", "how can i contact the office", "phone number of the college",
            "email address", "where can i reach the administration", "office contact",
            "who do i call", "helpline number", "college address", "talk to a staff member",
        ],
        "responses": ["You can contact the college office through the details on the Contact Us page of the website."],
    },
    "bot_identity": {
        "patterns": [
            "who are you", "what is your name", "are you a robot", "what can you do",
            "tell me about yourself", "are you a human", "who made you",
            "what are you", "how do you work", "what can i ask you",
        ],
        "responses": ["I'm a helpdesk chatbot built with Python and scikit-learn. Ask me about admissions, fees, exams and more."],
    },
}

FALLBACK = "Sorry, I didn't understand that. Could you rephrase? I can help with admissions, fees, timetable, exams, library and contact info."
CONFIDENCE_THRESHOLD = 0.60  # below this, the bot says it does not understand (tune this!)


# ----------------------------------------------------------------------
# 2. PREPROCESSING: make text uniform so the model sees clean words
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()                      # "Hello" -> "hello"
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


# ----------------------------------------------------------------------
# 3. BUILD TRAINING DATA (X = sentences, y = intent labels)
# ----------------------------------------------------------------------
def build_dataset():
    X, y = [], []
    for intent, data in INTENTS.items():
        for pattern in data["patterns"]:
            X.append(clean_text(pattern))
            y.append(intent)
    return X, y


# ----------------------------------------------------------------------
# 4. MODEL: TF-IDF turns text into numbers, Logistic Regression classifies
# ----------------------------------------------------------------------
def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),   # words + word pairs
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])


def evaluate(X, y):
    """Hold out 20% of the data to check the model on sentences it has not seen."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.2f} "
          f"({len(X_test)} test sentences, {len(X_train)} training sentences)")
    print(classification_report(y_test, pred, zero_division=0))


# ----------------------------------------------------------------------
# 5. RESPOND: predict intent, check confidence, pick a reply
# ----------------------------------------------------------------------
def reply(model, user_text: str) -> str:
    cleaned = clean_text(user_text)
    if not cleaned:
        return FALLBACK
    probs = model.predict_proba([cleaned])[0]
    best = probs.argmax()
    if probs[best] < CONFIDENCE_THRESHOLD:
        return FALLBACK
    intent = model.classes_[best]
    return random.choice(INTENTS[intent]["responses"])


def main():
    X, y = build_dataset()

    print("=== Evaluating model ===")
    evaluate(X, y)

    model = make_model().fit(X, y)  # final model trained on all data

    print("=== Chatbot ready! Type 'quit' to exit ===")
    while True:
        try:
            user = input("You: ")
        except EOFError:
            break
        if user.strip().lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        print("Bot:", reply(model, user))


if __name__ == "__main__":
    main()

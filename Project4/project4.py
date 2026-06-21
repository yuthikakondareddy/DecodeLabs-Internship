import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

df.columns = [
    "Sentiment",
    "ID",
    "Date",
    "Query",
    "User",
    "Text"
]

print(df.head())

df = df[["Sentiment", "Text"]]

df["Sentiment"] = df["Sentiment"].replace(4, 1)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["Text"] = df["Text"].apply(clean_text)

X = df["Text"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

review = ["I really loved this product. It is amazing!"]

review = [clean_text(review[0])]

review_vector = vectorizer.transform(review)

prediction = model.predict(review_vector)

if prediction[0] == 1:
    print("\nPrediction: Positive Review 😀")
else:
    print("\nPrediction: Negative Review 😞")
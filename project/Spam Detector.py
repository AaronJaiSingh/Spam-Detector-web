import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and clean the dataset
df = pd.read_csv('spam.csv')
data = df.where(pd.notnull(df), '')

# Convert 'spam' to 1 and 'ham' to 0
data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})

# Split into features and labels
x_train = data['Message']
y_train = data['Category'].astype(int)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
x_train_idf = vectorizer.fit_transform(x_train)

# Train Logistic Regression model
model = LogisticRegression(class_weight='balanced')
model.fit(x_train_idf, y_train)

# Test messages
x_test = [
    "Congratulations! You've been selected for a $1000 gift card. Click here to claim your prize.",
    "Hey, are we still meeting for lunch at 1?",
    "URGENT! Your mobile number has won £2000 cash! Call 09061701461 to claim.",
    "Don’t forget to bring your charger to class tomorrow.",
    "Winner!! You have won a free entry to a holiday trip to Bahamas. Reply YES to claim.",
    "Let me know if you need help with the assignment.",
    "Get your free ringtone now! Text WIN to 80085. Offer ends soon.",
    "I'll be home around 7. Want me to pick up dinner?"
]

# Vectorize and predict
x_test_idf = vectorizer.transform(x_test)
predictions = model.predict(x_test_idf)
prediction1 = ["ham" if p == 0 else "spam" for p in predictions]

# Display results
for msg, label in zip(x_test, prediction1):
    print(f"{label.upper()}: {msg}")

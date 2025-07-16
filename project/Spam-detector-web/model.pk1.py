import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load and clean the dataset
df = pd.read_csv('spam.csv')
df = df.where(pd.notnull(df), '')
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

# Features and labels
x_train = df['Message']
y_train = df['Category'].astype(int)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
x_train_idf = vectorizer.fit_transform(x_train)

# Train the model
model = LogisticRegression(class_weight='balanced')
model.fit(x_train_idf, y_train)

# Save the model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

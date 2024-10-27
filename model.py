import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import joblib

# Load and prepare the dataset
def load_data():
    spam = pd.read_csv('spam.csv')  # Ensure you have a spam.csv file
    z = spam['EmailText']
    y = spam['Label']
    return train_test_split(z, y, test_size=0.2)

# Train the model
def train_model():
    z_train, z_test, y_train, y_test = load_data()
    cv = CountVectorizer()
    features = cv.fit_transform(z_train)
    model = svm.SVC()
    model.fit(features, y_train)
    
    # Save the model and vectorizer
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(cv, 'count_vectorizer.pkl')

# Load the model
def load_model():
    model = joblib.load('spam_model.pkl')
    cv = joblib.load('count_vectorizer.pkl')
    return model, cv

# Classify email
def classify_email(email_text):
    model, cv = load_model()
    features = cv.transform([email_text])
    prediction = model.predict(features)
    return prediction[0]
import os
from model import train_model

def main():
    # Check if the model has already been trained
    if not os.path.exists('spam_model.pkl') or not os.path.exists('count_vectorizer.pkl'):
        print("Training the model...")
        train_model()
        print("Model trained and saved successfully.")
    else:
        print("Model already trained. You can start the Flask app.")

if __name__ == '__main__':
    main()
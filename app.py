from flask import Flask, render_template, request
from model import classify_email  # Corrected import

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    email_text = request.form['email_text']
    result = classify_email(email_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
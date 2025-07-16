from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        message = request.form['message']
        data = vectorizer.transform([message])
        result = model.predict(data)[0]
        prediction = "SPAM" if result == 1 else "HAM"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import numpy as np
import joblib

# +
app = Flask(__name__)

# Load the saved model from the file
model = joblib.load('C:\\Users\\user\\Desktop\\internship\\internship.pkl')


# -

@app.route('/')
def home():
    return render_template('index.html')


# +
# Define the route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = request.form['user_input']

    # Preprocess user input if needed
    # Make a prediction using your loaded model
    # Replace this with your actual prediction logic
    prediction = model.predict(user_input)

    # Render the result in the 'result.html' template
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
# -



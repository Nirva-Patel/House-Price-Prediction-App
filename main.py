from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load data and model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    # Get unique, sorted locations
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Fetch form data
    location = request.form.get('location')
    bhk = request.form.get('bhk')  # Fixed variable name typo
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    # Create input dataframe
    input_df = pd.DataFrame([[location, sqft, bhk, bath]],
                            columns=['location', 'total_sqft', 'bhk', 'bath'])

    # Make prediction
    prediction = pipe.predict(input_df)[0]*1e5

    # Return the prediction as a response
    return str(np.round(prediction,2))

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

app = app


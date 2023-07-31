import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the pickled model and scaler
with open('regmodel.pkl', 'rb') as file:
    regmodel = pickle.load(file)

with open('scaling.pkl', 'rb') as file:
    scalar = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Ensure that the data sent in the request is in JSON format and contains the required input features
    data = request.json['data']
    print(data)
    
    # Transform the data using the scaler
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    
    # Make the prediction using the model
    output = regmodel.predict(new_data)
    print(output[0])
    
    # Return the prediction as JSON response
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)

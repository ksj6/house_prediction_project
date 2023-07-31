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


# for creating a form for giving the inputs by iser in html page and then calculating the result
@app.route('/predict' , methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()] # capturing th values in the form which are in input given  
    final_input = scalar.transform(np.array(data).reshape(1 , -1))
    print(final_input)
    output = regmodel.predict(final_input)[0] # the final output of the regression model
    # rendering a specific html page 
    return render_template("home.html" , prediction_text="The final House Price Prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

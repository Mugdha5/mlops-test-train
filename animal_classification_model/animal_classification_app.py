########## Flask ##############

from flask import Flask, request,jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('animal_classification_model.joblib')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        try:
            # Get input data from the request
            data = request.get_json()

            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])

            # Make predictions
            prediction = model.predict(input_df)[0]

            # Return the prediction as JSON response
            return jsonify({'prediction': int(prediction)})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
         return jsonify({'message': 'This endpoint only accepts POST requests.'})


if __name__ == '__main__':
     app.run(debug=True,host='127.0.0.1')



from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
# import pickle
import joblib
import numpy as np
import json
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('dataset_ID')
parser.add_argument('date_start')
parser.add_argument('date_end')
parser.add_argument('payload')

# Define how the api will respond to the post requests
class FallsClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        # model = joblib.load('ml_model_decision_tree_fall_system_Entropy_sklearn.pkl')
        # X = np.array(json.loads(args['data']))
        payload = eval(args['payload'])
        # dataset_id = args['dataset_ID']
        # print(data, dataset_id)
        X = pd.DataFrame().from_dict(payload)
        
        prediction = model.predict(X)[0]
        
        if prediction == 1:
            prediction = "fall"
        elif prediction == 0:
            prediction = "not fall"
        
        # return jsonify(list(json.dumps(prediction)))
        response = {"response": "OK","prediction": prediction}
        
        
        # response = {"response": "OK", "value": json.dumps(payload)}
        return jsonify(response)

api.add_resource(FallsClassifier, '/falls')

if __name__ == '__main__':
    # Load model
    
    model_name = 'ml_model_decision_tree_fall_system_Entropy_sklearn.pkl'
    
    with open('model.'+model_name, 'rb') as f:
        model = joblib.load(f)

    app.run(debug=True)
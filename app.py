from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import pandas as pd
import time
from datetime import datetime
import json

fall = Flask(__name__)
api = Api(fall)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('dataset_ID')
parser.add_argument('time')
parser.add_argument('payload')

# Define how the api will respond to the post requests
class FallsClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        payload = eval(args['payload'])
        
        X = pd.DataFrame().from_dict(payload)
        
        # dt_before_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        dt_before_pred = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        # dt_before_pred = time.time()*1000

        prediction = model.predict(X)[0]
        dt_after_pred = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        # dt_after_pred = time.time()*1000
        
        # dt_after_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        
        
        # response = {"prediction": str(prediction),"dt_before_pred":str(dt_before_pred),"dt_after_pred":str(dt_after_pred)}
        response = {"prediction": str(prediction),"dt_before_pred":dt_before_pred,"dt_after_pred":dt_after_pred}
        
        return jsonify(response)

api.add_resource(FallsClassifier, '/falls')

if __name__ == '__main__':
    # Load model
    
    # model_name = 'complementary_filter_26juni2022_8labelclass.h5'
    # model_name = 'complementary_filter_27juni2022_5labelclass.h5'
    # model_name = 'kfall_complementary_filter_30juni2022_2label.h5'
    model_name = 'kfall_complementary_filter_7juli2022_3label.h5'

    
    with open('model/'+model_name, 'rb') as f:
        model = joblib.load(f)

    fall.run(debug=True)
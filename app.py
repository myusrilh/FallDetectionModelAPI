from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import pandas as pd
import time

app = Flask(__name__)
api = Api(app)

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
        
        dt_before_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        prediction = model.predict(X)[0]
        dt_after_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        
        # 0 = standing
        # 1 = walking
        # 2 = free fall
        # 3 = lying down
        # 4 = wake up
        # 5 = slowly goes down
        # 6 = jogging
        # 7 = wake to sit
        # 8 = sit
        if prediction == 0:
            prediction = "Berdiri"
        elif prediction == 1:
            prediction = "Berjalan"
        elif prediction == 2:
            prediction = "Terjatuh!"
        elif prediction == 3:
            prediction = "Posisi di lantai"
        elif prediction == 4:
            prediction = "Bangkit"
        elif prediction == 5:
            prediction = "Turun perlahan"
        elif prediction == 6:
            prediction = "Berlari pelan"
        elif prediction == 7:
            prediction = "Bangkit ke posisi duduk"
        elif prediction == 8:
            prediction = "Duduk"
        
        response = {"prediction": prediction,"dt_before_pred":dt_before_pred,"dt_after_pred":dt_after_pred}
        
        return jsonify(response)

api.add_resource(FallsClassifier, '/falls')

if __name__ == '__main__':
    # Load model
    
    model_name = 'complementary_filter_21juni2022_8labelclass.h5'
    
    with open('model/'+model_name, 'rb') as f:
        model = joblib.load(f)

    app.run(debug=True)
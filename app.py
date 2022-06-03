from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
# import pickle
import joblib
import pandas as pd
import time,datetime
import json

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
        # model = joblib.load('ml_model_decision_tree_fall_system_Entropy_sklearn.pkl')
        # X = np.array(json.loads(args['data']))
        payload = eval(args['payload'])
        # dataset_id = args['dataset_ID']
        # print(data, dataset_id)
        X = pd.DataFrame().from_dict(payload)
        
        dt_before_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        # dt_before_pred = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S.%f")
        prediction = model.predict(X)[0]
        dt_after_pred = time.strftime('%A, %d %B %Y %H:%M:%S')
        # dt_after_pred = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S.%f")
        
        if prediction == 1:
            prediction = "fall"
        elif prediction == 0:
            prediction = "not fall"
        
        # return jsonify(list(json.dumps(prediction)))
        # response = {"response": "OK","prediction": prediction,"seconds":args['time'],"datetime":time.strftime('%A %B, %d %Y %H:%M:%S')}
        response = {"prediction": prediction,"dt_before_pred":dt_before_pred,"dt_after_pred":dt_after_pred}
        
        
        # response = {"response": "OK", "value": json.dumps(payload)}
        return jsonify(response)

api.add_resource(FallsClassifier, '/falls')

if __name__ == '__main__':
    # Load model
    
    # model_name = 'decision_tree_fall_system_kalman_filter_30mei2022_entropy_sklearn.sav'
    # model_name = 'decision_tree_fall_system_kalman_filter_31mei2022_entropy_sklearn.h5'
    model_name = 'decision_tree_fall_system_complementary_filter_2juni2022_entropy_sklearn.h5'
    
    with open('model/'+model_name, 'rb') as f:
        model = joblib.load(f)

    app.run(debug=True)
from flask import Flask, jsonify, render_template
from flask_restful import Api, Resource, reqparse
import joblib
import pandas
from datetime import datetime

fall = Flask(__name__)
api = Api(fall)

parser = reqparse.RequestParser()
parser.add_argument('dataset_ID')
parser.add_argument('time')
parser.add_argument('payload')

class FallsClassifier(Resource):
    def post(self):
        
        model_name = 'kfall_complementary_filter_8juli2022_3label.h5'
    
        with open('model/'+model_name, 'rb') as f:
            model = joblib.load(f)
        
        args = parser.parse_args()
        payload = eval(args['payload'])
        
        X = pandas.DataFrame().from_dict(payload)
        
        dt_before_pred = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        prediction = model.predict(X)[0]
        dt_after_pred = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        
        
        response = {"prediction":str(prediction),"dt_before_pred":dt_before_pred,"dt_after_pred":dt_after_pred}
        
        return jsonify(response)

class Home(Resource):
    def index():
        # A welcome message to test our server
        return render_template("index.html")

api.add_resource(FallsClassifier, '/falls')
api.add_resource(Home, '/')

if __name__ == '__main__':
    # model_name = 'kfall_complementary_filter_7juli2022_3label.h5'
    

    fall.run()

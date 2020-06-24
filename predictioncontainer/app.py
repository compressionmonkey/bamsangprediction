from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
# from pricepredictor import predict_price

app = Flask(__name__)
CORS(app)

api = Api(app)

class PricePrediction(Resource):
    def get(self):
        return "hello world"
    def post(self):
        return "yo"

api.add_resource(PricePrediction, '/priceprediction')
if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

from service.recommender import Recommender

app = Flask(__name__)

ratings_df = pd.read_csv('data/ml-100k/u.data', sep='\t',
                         names=['user_id', 'item_id',
                                'rating', 'timestamp'])
recommender = Recommender(ratings_df, num_factors=100)
recommender.train_user_item_matrix(num_iter=100)


@app.route('/')
def hello():
    return 'hi'


@app.route('/health')
def health():
    return jsonify({'status': 'UP'})


@app.route('/recommend/<user_id>', methods=['GET'])
def predict():
    return


@app.route('/criteria', methods=["POST"])
def create_user():
    data = request.json
    criteria = Criteria(data)

    user_id = str(uuid.uuid4())
    INVENTORY_SERVICE.create_user(user_id, criteria)

    return jsonify({"userId": user_id})


@app.route('/inquiry/<user_id>', methods=["GET"])
def find_inquiry(user_id):
    return INVENTORY_SERVICE.compute_inquiry(user_id).to_json()


@app.route('/inquiry/<user_id>', methods=["POST"])
def send_inquiry_response(user_id):
    data = request.json
    inquiry = Inquiry(data)

    return json.dumps(INVENTORY_SERVICE.process_inquiry(user_id, inquiry))


@app.route('/recommendation/<user_id>', methods=["GET"])
def send_recommendations(user_id):
    return json.dumps(INVENTORY_SERVICE.recommend_vehicles(user_id))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

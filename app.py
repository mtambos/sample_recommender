#! /usr/bin/env python
from gevent import monkey
monkey.patch_all()

import os
import json

from flask import Flask, request, jsonify
from flasgger import Swagger
from flask_httpauth import HTTPBasicAuth


app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))
app.config['SWAGGER'] = {
    # set to True so instead of
    # $ref: '#/definitions/alert'
    # we get
    # $ref: '#/definitions/index_post_alert'
    'prefix_ids': True
}
Swagger(app)

APP_ROOT = os.path.dirname(os.path.realpath(__file__))

# AUTHENTICATION
auth = HTTPBasicAuth()
users = {
    "mtambos": "agtd%dfgkjhRE85$§XXC6"
}


@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None


@app.before_request
def before_request():
    print(f"Request received to {request.path}.")


@auth.login_required
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Given a document, returns a list of recommendations.
    ---
    tags:
      - recommend
    parameters:
      - in: body
        name: body
        schema:
          id: doc
          required:
            - content
            - num
          properties:
            content:
              type: string
              description: list of characteristics important to the user.
            num:
              type: integer
              description: number of recommendations to return.
              default: 10
    responses:
      200:
        description: recommendations
        schema:
            type: array
            items:
                $ref: '#/definitions/Recommendation'
    """
    from recommender import content_engine
    content = request.json['content']
    num_predictions = request.json.get('num', 10)
    return jsonify(content_engine.recommend(content, num_predictions))


@auth.login_required
@app.route('/train/<string:data_url>/', methods=['GET'])
def train(data_url):
    """
    Train the recommender with the given data.
    ---
    tags:
      - train
    parameters:
      - name: data_url
        in: path
        type: string
        required: true
    responses:
      200:
        description: OK if successfully finished.
        type: string
    """
    from recommender import content_engine
    content_engine.train(data_url)
    return "OK"


# API ENDPOINTS
@auth.login_required
@app.route('/')
def index():
    status_message = json.dumps({'status': "200"})
    return str(status_message)


if __name__ == '__main__':
    app.run(debug=True)


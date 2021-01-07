"""
# run web service
set FLASK_APP=web-service.py
python -m flask run
"""

from flask import Flask, request
import tensorflow.keras as krs

# create a new web app
app = Flask(__name__)

# add root route
@app.route("/")
def home():
    return app.send_static_file('index.html')


@app.route("/prediction/power", methods=["POST"])
def powerproduction():
    speed = float(request.get_json()["speed"])
    model = krs.models.load_model("mp2.h5")
    prediction = model.predict([speed])
    preds = prediction.tolist()
    return {'prediction': preds[0]}


if __name__ == '__main__':
    app.run(debug=True)

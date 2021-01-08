"""
# run web service
set FLASK_APP=webService.py
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

# method to respond with prediction
@app.route("/prediction/power", methods=["POST"])
def powerproduction():
    speed = float(request.get_json()["speed"])
    model = krs.models.load_model("newmodel.h5")
    prediction = model.predict([speed])
    preds = prediction.tolist()

    print(preds[0])

    speed1 = float(request.get_json()["speed"])
    model1 = krs.models.load_model("newmodel3.h5")
    prediction1 = model1.predict([speed])
    preds1 = prediction1.tolist()
    
    print(preds1[0])
    
    speed2 = float(request.get_json()["speed"])
    model2 = krs.models.load_model("newmodel2.h5")
    prediction2 = model2.predict([speed])
    preds2 = prediction2.tolist()
    
    print(preds2[0])
    
    res = "Model :%s  Model 2: %s  Model 3 :%s" % (preds[0], preds1[0], preds2[0])
    
    return res


if __name__ == '__main__':
    app.run(debug=True)

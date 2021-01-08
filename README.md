# Emerging-Technology-Progect

Introduction
This project objective is to predict power generated with input of wind speed. Models are generated using Jupyter notebook. Python libraries used are Pandas, Tensorflow, numpy,matplotlib. The genereated models are utilized by api developed with Flask to service api requests.


## Technologies/Libraries

- Python
- Pandas
- Matplotlib
- Tensorflow
- Numpy
- Flask
- IDE (Jupyter Notebook)
- Html
- Javascript

# Project Overview
The project contians data set which shall be trained with the models developed with Jupyter Notebook.
Below are the models used in the application
- Building a Sequential Model that has a Sigmoid and Linear Layer
- Building a Sequential Model that has Linear Layer.
- set the data into Numpy arrays

The models are trained ,predictive model is saved . The models are revised and comporaed with the graphical representation and the suitable model is selected to be used in the application. The three models are used  by the API service to provide the predicted value to the requestor.


## Accuracy to Data Frame
model3Acc = pd.DataFrame(m3history.history['accuracy'], columns=['Model Accuracy ='])
 
print("Over 500 Epochs the Model has an accuracy of:")
model3Acc.mean()

## Plotting 
plt.plot(m3history.history['accuracy'])
plt.title('Comaparing Model Accuracy')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')

## Identifying Loss
plt.plot(m3history.history['loss'])
plt.title('Model 3 Losses')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')

## Web Service

@app.route("/prediction/power", methods=["POST"])
def powerproduction():
    speed = float(request.get_json()["speed"])
    model = krs.models.load_model("newmodel.h5")
    prediction = model.predict([speed])
    preds = prediction.tolist() 

    speed1 = float(request.get_json()["speed"])
    model1 = krs.models.load_model("newmodel3.h5")
    prediction1 = model1.predict([speed])
    preds1 = prediction1.tolist() 
    
    speed2 = float(request.get_json()["speed"])
    model2 = krs.models.load_model("newmodel2.h5")
    prediction2 = model2.predict([speed])
    preds2 = prediction2.tolist() 
    
    res = "Model :%s  Model 2: %s  Model 3 :%s" % (preds[0], preds1[0], preds2[0])
    
    return res
    
## How to run Jupyter NoteBook
```
cd ../project
$ jupyter notebook
```

## How to Execute
Models are generated using the data sets derived with the sample data and trained models.

Exeucte the web service with conosle python command.

Webservice shold have started on port 5000.  In webbrowser use the url http://localhost:5000/

This shall open the index.html with a text field to enter the wind speed and button to submit

After submitting the request will route to the webservice with the input wind speed.

Web service shall use the three models to fetch the predicted value.

All the three predicted values are sent back to the browser to dispaly

Finally shall show result of predicted power generated predicted with the 3 models.

### How to run the Web Service and Docker Image
```
$ git clone https://github.com/ika25/Emerging-Technology-Progect
$ cd ../Emerging-Technology-Progect\Project Folder
$ set FLASK_APP=webService.py
  python -m flask run
```
```
$ docker build -t webService
$ docker run -d -p 5000:5000 webService
```

## How to install Keras, tensorflow and Flask
```
pip install keras
pip install tensorflow
pip install Flask

```

# Conclusion
Project was challenging i traied best of my ability to complete minimum requremetns, when i started project few weeks back i realised structure of project was not according to the project requirement and It was not goint in right direction and had to change few things because of that having not much time left for this project was stressful. I really had to do lot of research online to understand project requirements.

# References

* https://stackabuse.com/deep-learning-in-keras-building-a-deep-learning-model/
* https://keras.io/guides/sequential_model/
* https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
* https://www.datacamp.com/community/tutorials/deep-learning-python
* https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
* https://towardsdatascience.com/step-by-step-guide-building-a-prediction-model-in-python-ac441e8b9e8b
* https://machinelearningmastery.com/make-predictions-scikit-learn/
* https://www.dataquest.io/blog/numpy-tutorial-python/
* https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
* https://keras.io/getting_started/intro_to_keras_for_engineers/
* https://keras.io/examples/vision/mnist_convnet/
* https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
* https://www.tensorflow.org/guide/keras/train_and_evaluate
* https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
* https://machinelearningmastery.com/save-load-keras-deep-learning-models/

### Flask

* https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/
* https://towardsdatascience.com/building-a-web-application-to-deploy-machine-learning-models-e224269c1331
* https://flask.palletsprojects.com/en/1.1.x/
* https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask
* https://opensource.com/article/18/4/flask

### LearnOnline

* https://web.microsoftstream.com/video/bf7aaf56-d65f-47fb-aec9-e0c2af49534a?referrer=https:%2F%2Flearnonline.gmit.ie%2F
* https://web.microsoftstream.com/video/8ceee222-ebdf-4234-b05b-9090155f1747
* https://web.microsoftstream.com/video/ed1d0b97-441b-4e60-9ecf-b2e775d6bb7c




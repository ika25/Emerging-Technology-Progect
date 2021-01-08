# Emerging-Technology-Progect

##Introduction
This project objective is to predict power generated with input of wind speed. Models are generated using Jupyter notebook. Python libraries used are Pandas, Tensorflow, numpy,matplotlib. The genereated models are utilized by api developed with Flask to service api requests.


##Technologies/Libraries

-Python
-Pandas
-Matplotlib
-Tensorflow
-Numpy
-Flask
-IDE (Jupyter Notebook/Pycharm)
-Html
-Javascript

# Project Overview
The project contians data set which shall be trained with the models developed with Jupyter Notebook.
Below are the models used in the application
-Building a Sequential Model that has a Sigmoid and Linear Layer
-Building a Sequential Model that has Linear Layer.
-set the data into Numpy arrays

The models are trained ,predictive model is saved . The models are revised and comporaed with the graphical representation and the suitable model is selected to be used in the application. The three models are used  by the API service to provide the predicted value to the requestor.


# Accuracy to Data Frame
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

# Web Service

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

# References
https://stackabuse.com/deep-learning-in-keras-building-a-deep-learning-model/
https://keras.io/guides/sequential_model/
https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
https://www.datacamp.com/community/tutorials/deep-learning-python
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
https://towardsdatascience.com/step-by-step-guide-building-a-prediction-model-in-python-ac441e8b9e8b
https://machinelearningmastery.com/make-predictions-scikit-learn/
https://www.dataquest.io/blog/numpy-tutorial-python/
https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
https://keras.io/getting_started/intro_to_keras_for_engineers/
https://keras.io/examples/vision/mnist_convnet/
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
https://www.tensorflow.org/guide/keras/train_and_evaluate
https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
https://machinelearningmastery.com/save-load-keras-deep-learning-models/ ### 




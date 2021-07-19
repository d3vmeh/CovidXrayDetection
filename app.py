from flask import Flask, render_template, request, redirect
import os
import numpy as np
import cv2
from keras.models import load_model


app = Flask(__name__)


model = load_model("model1.h5")


@app.route("/",methods=["GET","POST"])
def index():
    if request.method == "GET":
        return render_template("index.html",img="")
    else:
        givenimage = request.files["img"].read()

        imagearray = np.fromstring(givenimage,np.uint8)
        imagearray = cv2.imdecode(imagearray,cv2.IMREAD_COLOR)
        imagearraygray = cv2.cvtColor(imagearray,cv2.COLOR_BGR2GRAY)

        imagearraygray = cv2.resize(imagearraygray,(50,50))
        predictionarray = np.array(imagearraygray)
        predictionarray = predictionarray.reshape(1,50,50,1)
        predictions = model.predict(predictionarray)
        prediction = np.argmax(predictions[0])
        message = ""
        if prediction == 0:
            message = ("COVID-19 Pneumonia")

        if prediction == 1:
            message = ("ARDS Pneumonia")

        if prediction == 2:
            message = ("SARS Pneumonia")

        if prediction == 3:
            message = ("Pneumocystis Pneumonia")

        if prediction == 4:
            message = ("Streptococcus Pneumonia")

          #No Finding does not work very well, as the dataset only has 1 or 2 cases of
          #it
        if prediction == 5:
            message = ("no infection")

        return render_template("index.html",img="",msg=message)



if __name__ == "__main__":
    app.run()
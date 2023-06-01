from typing import Dict
import rasterio
import streamlit as stream
from PIL import Image
import numpy as np
from keras.models import load_model
import os
import pickle

def main():
    #Welcome message
    stream.title("Tumor Delineation Web app")
    stream.write("Hello! welcome to T-Classify, a Tumor delineation application that allows you to classify a tumor image as bening or malgnant.")
    stream.write("Please click the 'Browse files' button or drag and drop an image file below to classify your image")
    file = stream.file_uploader("Upload",type=["png","jpg","jpeg", "tif", "dcm"])

    if file is not None:
        path = os.path.join("tiff_images/" + file.name)
        with rasterio.open(path) as img:
            arr = img.read()
            arr = arr.reshape((512,512,1))
            arr = np.array(arr) / 255
            im =[arr.flatten()]
            stream.image(img.read().reshape(512,512,1), caption="CT Scan", clamp=True)
            arr = arr.reshape(1,512,512,1)
        main_button = stream.button("Classify")
        if main_button:
            direct = os.getcwd()
            with open(direct + "/" + "mlp.pkl", "rb") as f:
                model = pickle.load(f)
            model2 = load_model(direct +  "/" + "model.h5") 
            clss = model.predict(im)
            age = model2.predict(arr)
            print(age)

            if int(clss[0]) < 0.5 and int(age[0][-1]) * 100 > 50:
                txt = "The tumor scan is malignant, however it has been there for a long time. A surgery is suggested to get it out as soon as possible"
            elif int(clss[0]) < 0.5 and int(age[0][-1]) * 100 < 50:
                txt = "The tumor scan is malignant and has not been present for long. a treatment is advised"
            elif int(clss[0]) >= 0.5 and int(age[0][-1]) * 100 > 50:
                txt = "The tumor scan is benign and has been there for a long time. it is quite a dangerous situation, treatment is advised immediately."
            elif int(clss[0]) >= 0.5 and int(age[0][-1]) * 100 < 50:
                txt = "The tumor is benign, however, it is in its early stage. treatment advised"
            else:
                txt = "This is an unknown image, please check well"
            stream.write(txt)



if __name__ == "__main__":
    main()
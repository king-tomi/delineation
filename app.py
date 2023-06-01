import rasterio
import streamlit as stream
import numpy as np
from keras.models import load_model

def main():
    #Welcome message
    stream.title("Tumor Delineation Web app")
    stream.write("Hello! welcome to T-Classify, a Tumor delineation application that allows you to classify a tumor image as bening or malgnant.")
    stream.write("Please click the 'Browse files' button or drag and drop an image file below to classify your image")
    file = stream.file_uploader("Upload",type=["png","jpg","jpeg", "tif", "dcm"])

    if file is not None:
        path = file.name
        with rasterio.open(path) as img:
            arr = img.read()
            arr = arr.reshape((512,512,1))
            arr = np.array(arr) / 255
            im =[arr.flatten()]
            stream.image(img.read().reshape(512,512,1), caption="CT Scan", clamp=True)
            arr = arr.reshape(1,512,512,1)
        main_button = stream.button("Classify")
        if main_button:
            model = load_model("model.h5") 
            clss = model.predict(arr)

            if int(clss[0]) < 0.5:
                txt = "The tumor is malignant"
            elif int(clss[0]) >= 0.5:
                txt = "The tumor is benign"
            else:
                txt = "This is an unknown image, please check well"
            stream.write(txt)



if __name__ == "__main__":
    main()
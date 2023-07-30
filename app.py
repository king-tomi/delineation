import rasterio
from PIL import Image
import streamlit as stream
import numpy as np
from keras.models import load_model
import os

def main():
    #Welcome message
    stream.title("Tumor Delineation Web app")
    stream.write("Hello! welcome to T-Classify, a Tumor delineation application that allows you to classify a tumor image as bening or malgnant.")
    stream.write("Please click the 'Browse files' button or drag and drop an image file below to classify your image")
    file = stream.file_uploader("Upload",type=["png","jpg","jpeg", "tif", "dcm"])
    
    pathf = os.getcwd()

    if file is not None:
        if "tif" in file.name:
            path = os.path.join(pathf, f"tiff_images/{file.name}")
        else:
            path = os.path.join(pathf, f"dicom_dir/{file.name}")

        with rasterio.open(path) as img:
            arr = img.read()
            arr = arr.reshape((512,512,1))
            arr = np.array(arr) / 255
            im =[arr.flatten()]
            stream.image(img.read().reshape(512, 512, 1), caption="CT Scan", clamp=True)
            arr = arr.reshape(1,512,512,1)
        main_button = stream.button("Classify")
        if main_button:
            model = load_model(os.path.join(pathf, "delineation/model.h5")) 
            clss = model.predict(arr)

            if int(clss[0]) < 0.5:
                txt = "The image is one without tumor"
            elif int(clss[0]) >= 0.5:
                txt = "The image is one with tumor"
            else:
                txt = "This is an unknown image, please check well"
            stream.write(txt)



if __name__ == "__main__":
    main()

pip install -r requirements.txt

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import os
from PIL import Image
global file

with st.sidebar:
        choice = st.radio("Navigation",["Home","Pizza Testing"])


if choice == "Home":
    st.title("This Website uses Deep Learning to Multi-Classifies Pizza based on how cooked it is ")
    st.subheader("Contributions: Omar , Thomas, and Collin")

    pass

if choice == "Pizza Testing":


    st.title("Please provide an image of your pizza!")
    
  
    import base64

    # Load the image file
    cooked = open('cooked.jpg', 'rb')
    cooked_bytes = cooked.read()

    # Encode the image bytes as base64
    cooked_image = base64.b64encode(cooked_bytes).decode()

    # Display the image in the Streamlit app
    #st.image(cooked_bytes, caption='Cooked Example')
    st.caption("Picture Example of a Cooked Pizza ")
    # Create a download button for the encoded image
    href = f'<a href="data:image/jpg;base64,{cooked_image}" download="Cooked.jpg">Download Cooked</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    
    # Load the image file
    Uncooked = open('Uncooked.jpg', 'rb')
    Uncooked_bytes = Uncooked.read()

    # Encode the image bytes as base64
    uncooked_image = base64.b64encode(Uncooked_bytes).decode()

    # Display the image in the Streamlit app
    #st.image(Uncooked_bytes, caption='UnCooked Example')
    st.caption("Picture Example of an UnCooked Pizza ")
    # Create a download button for the encoded image
    href = f'<a href="data:image/jpg;base64,{uncooked_image}" download="Uncooked.jpg">Download UnCooked</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.subheader("Upload your pizza Image here:")
    file = st.file_uploader("Upload your Image Here", type=["jpg", "jpeg", "png"])
    
    if file:
        st.image(file,caption='BEFORE RESIZING')
        
    button = st.button('Click me and find out if your pizza is BUSSIN or DISGUSTING')
        
    if button:
         import requests

         url = 'https://drive.google.com/uc?id=1k0McXeXNYT-rvDt1wmPcDpP4f_HR1Kps'
         r = requests.get(url)

         with open('Pizza_model.h5', 'wb') as f:
            f.write(r.content)
            
         from keras.models import Sequential, load_model
         new_model = load_model('Pizza_Model.h5')
        

         image = Image.open(file)
        
         # Resize the image
         st.subheader("Resized Image:")
         resize = tf.image.resize(image, (256,256))
         st.image(resize.numpy().astype(int))

    
         yhat = new_model.predict(np.expand_dims(resize/255, 0))
         
         import time

         progress_text = "Watch as our AI technology leaps into action and paves the way for a smarter future!"

         my_bar = st.progress(0, text=progress_text)

         for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1, text=progress_text)
        
                
         if 0 <= yhat <= 0.20: 
            st.subheader(f'THIS IS COOKED PIZZA!') 
            st.balloons()   
            
         elif 0.21 <= yhat <= 0.40:
            st.subheader('its cooked but not crispy')
            st.balloons()
         elif 0.41 <= yhat <= 0.60:
            st.subheader('Needs more time')
            st.balloons()
         elif 0.61 <= yhat <= 1.0:
        
            st.subheader(f'ITS RAWW - Ramsey')
            st.balloons()
    pass
    


                                    

#Importing Libraries
import streamlit as st 
import requests
from streamlit_lottie import st_lottie
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

#Page Configuration
st.set_page_config(page_title="Alzheimer Disease Detection",
layout="wide",page_icon="pngegg.png")

#All Functions Used
@st.cache(allow_output_mutation=True)
def load_model():
    '''It will load the keras CNN Model'''
    model = tf.keras.models.load_model("MyModel_h5.h5")
    return model

def import_and_predict(image_data, model):
    '''Importing the image from streamlit interface and sending it to the Model'''
    size = (128,128)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)   
    return prediction

def load_lottieurl(url):
    '''It will load the lottie animation'''
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Main
model = load_model()
lottie_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_33asonmr.json")
lottie_animation2 = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_bcop55ma.json")
st.markdown("<h1 style='font-style:Helvetica; font-size:80px;'>ALZHEIMER'S DISEASE DETECTION USING MRI IMAGES</h1>", unsafe_allow_html=True)
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("<h2 style='font-style:Helvetica; '>WHAT IS ALZHEIMER'S DISEASE?</h2>", unsafe_allow_html=True)
        st.markdown("""<h4 style='font-style:Helvetica; text-align: justify; '>Alzheimer's disease is a brain disorder that slowly destroys memory and thinking skills and, eventually, the ability to carry out the simplest tasks. In most people with the disease — those with the late-onset type symptoms first appear in their mid-60s. Early-onset Alzheimer's occurs between a person's 30s and mid-60s and is very rare. 
        Alzheimer's disease is the most common cause of dementia among older adults. Approximately 5.8 million people in the United States age 65 and older live with Alzheimer's disease. Of those, 80% are 75 years old and older. Out of the approximately 50 million people worldwide with dementia, between 60% and 70% are estimated to have Alzheimer's disease.
        The early signs of the disease include forgetting recent events or conversations. As the disease progresses, a person with Alzheimer's disease will develop severe memory impairment and lose the ability to carry out everyday tasks.
        Medications may temporarily improve or slow progression of symptoms. 
        </h4>""", unsafe_allow_html=True)
        st.markdown("""<h4 style='font-style:Helvetica; text-align: justify; '>These treatments can sometimes help people with Alzheimer's disease maximize function and maintain independence for a time. Different programs and services can help support people with Alzheimer's disease and their caregivers.
        There is no treatment that cures Alzheimer's disease or alters the disease process in the brain. In advanced stages of the disease, complications from severe loss of brain function — such as dehydration, malnutrition or infection — result in death.
        </h4>""", unsafe_allow_html=True)
    with right_column:
        st_lottie(lottie_animation, height=800, key="coding")
with st.container():
    st.markdown("---")
    left_column, right_column = st.columns(2)
    with left_column:
        file = st.file_uploader("", type=("jpg","png","jpeg"))
        if file is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            score = tf.nn.softmax(predictions[0])
            class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
            result = class_names[np.argmax(score)]
            st.markdown("""<h4 style='font-style:Helvetica; text-align: justify; '>RESULT:    </h4>""", unsafe_allow_html=True) 
            st.markdown(
            f"""<h4 style='font-style:Helvetica; text-align: justify; '>This Image is most likely belongs to {result}. </h4>""", unsafe_allow_html=True)
    with right_column:
        st_lottie(lottie_animation2, height=800, key="coding2")
        

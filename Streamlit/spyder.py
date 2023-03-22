# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:50:40 2023

@author: AYOBAMI SHOYEMI
"""

#importing dependencies
import numpy as np
import pandas as pd
from joblib import load
import streamlit as st

# loading in the data(The dumped model)
model = load('RF_model.joblib')


#Backend
def predictions(sepallength, sepalwidth, petallength, petalwidth):
    prediction = model.predict(np.array([[sepallength, sepalwidth, petallength, petalwidth]]))
    
    return prediction



# fuction to create the UI
def main():
    st.title('Iris flower model')
    
    
    

    sepallength = st.text_input('Enter sepal length: ')
    sepalwidth = st.text_input('Enter sepal width: ')
    petallength = st.text_input('Enter petal length: ')
    petalwidth = st.text_input('Enter petal width: ')
    
    
    button = st.button('predict')
    
    result = ''

    if(button):
       result = predictions(sepallength, sepalwidth, petallength, petalwidth)
       if result == 0:
           st.success('Setosa')
       if result == 1:
           st.success('Versicolor')
       else:
           st.success('Virginica')
    
    
    
    
if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow
import pickle

scaler = pickle.load(open('scal.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title(" *House Price Prediction Application*")
st.write("*This app predicts the market value of **Houses***")

st.header("*Enter the informations of the House below*")

def house_input():
    c1,c2 = st.columns(2)
    
    with c1:
        area = st.number_input('*What is the area of the house?*', 1000,18000,4000)
        bedrooms = st.number_input('*How many bedrooms are in the house?*',1,6,5)
        bathrooms = st.number_input('*How many bathrooms are in the house?*',1,4,3)
        stories = st.number_input('*How many storey building is the house?*',1,4,3)
        mainroad = st.selectbox('*Is the house close to a mainroad?*',(['Yes','No']))
        guestroom = st.selectbox('*Does the house have a Guestroom?*',(['Yes','No']))
     
    with c2:
        basement = st.selectbox('*Does the house have a basement?*',(['Yes','No']))
        hotwaterheating = st.selectbox('*Does the house have a heating component installed?*',(['Yes','No']))
        airconditioning = st.selectbox('*Is the house airconditioned?*',(['Yes','No']))
        parking = st.number_input('*How many parking lots does the house have?*',0,3,2)
        prefarea = st.selectbox('*Does the house have a preferred area or a lounge?*',(['Yes','No']))
        furnishingstatus = st.selectbox('*Select the status of the house furnishing*',(['furnished','semi-furnished','unfurnished']))
                 
     
    feat = np.array([area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]).reshape(1,-1)
    cols = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']
    feat1 = pd.DataFrame(feat, columns=cols)
    return feat1

df = house_input()
#st.write(df)

df.replace({'Yes':1,
            'No':0},inplace = True)
#st.write(df)

df['area'] = pd.to_numeric(df['area'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['stories'] = pd.to_numeric(df['stories'], errors='coerce')


df['area_per_bathroom'] = df['area']/df['bathrooms']
df['area_per_bedrooms'] = df['area']/df['bedrooms']
df['area_per_stories'] = df['area']/df['stories']
df['bathrooms_per_bedrooms'] = df['bathrooms']/df['bedrooms']
df['stories_per_bedrooms'] = df['stories']/df['bedrooms']
df['area_times_bedrooms'] = df['area']*df['bedrooms']
df['bathroom_times_stories'] = df['bathrooms']*df['stories']
df['area_times_stories'] = df['area']*df['stories']

def prepare(df):
    
    enc_data =pd.DataFrame(encoder.transform(df[['furnishingstatus']]))#.toarray())
    #enc_data.columns = encoder.get_feature_names_out()
    enc_data.columns = encoder.get_feature_names_out(['furnishingstatus'])
    df = df.join(enc_data)

    df.drop(['furnishingstatus'],
           axis=1,
           inplace = True)
    
    cols = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df,columns=cols)
    return df
df = prepare(df)
#st.write(df)


model = pickle.load(open('model.pkl','rb'))
predictions = model.predict(df)

st.subheader('*House Price*')
if st.button('*Click here to get the price of the **House***'):
    #st.write(predictions)
    st.write(np.exp(predictions))
    
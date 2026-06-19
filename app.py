import streamlit as st 
import pandas as pd 
import pickle 

with open(r'pkl_files/scaler_for_cluster.pkl','rb') as f:
    scaler_for_cluster = pickle.load(f)
    
with open(r'pkl_files/scaler_for_temp.pkl','rb') as f:
    scaler_for_temp = pickle.load(f)
    
with open(r'pkl_files/cluster_prediction_model.pkl','rb') as f:
    cluster_prediction_model = pickle.load(f)
    
with open(r'pkl_files/temp_prediction_model.pkl','rb') as f:
    temp_prediction_model = pickle.load(f)
        
st.sidebar.header("Input Features")
date = st.sidebar.date_input("Enter the date")
if date:
    dataframe = pd.DataFrame({'date':[date]})

    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['day'] = dataframe['date'].dt.day
    dataframe.drop('date',axis=1,inplace=True)

    latitude = st.sidebar.number_input("Enter the Latitude")
    dataframe['latitude'] = latitude 

    longitude = st.sidebar.number_input("Enter the Longitude")
    dataframe['longitude'] = longitude
    
predict = st.sidebar.button("Predict Temperature")

if predict:
    scaler_data_for_cluster = scaler_for_cluster.transform(dataframe)

    cluster = cluster_prediction_model.predict(scaler_data_for_cluster)
    dataframe['cluster'] = cluster 

    scaler_data_for_prediction_model = scaler_for_temp.transform(dataframe)

    temp_prediction = temp_prediction_model.predict(dataframe)
    
    map_data = pd.DataFrame({"lat": [latitude],"lon": [longitude]})
    
    st.markdown(
    f"""
    <h2 style='text-align:center; color:red;'>
    Temperature: {temp_prediction[0]:.2f} °C
    </h2>""",unsafe_allow_html=True)

    st.map(map_data)
    
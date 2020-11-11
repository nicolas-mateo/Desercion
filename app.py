import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

DATA_URL = ("grad_desert.csv")

st.sidebar.title("Tablero Deserción Estudiantil ETITC")
st.sidebar.write("A continuación se muestran las diferentes opciones de visualización de datos:")
functionality = st.sidebar.radio('¿Qué visualización desea?',('Información Histórica','Predicción', 'Calculadora')) 

if functionality=='Información Histórica':
    
    @st.cache(persist=True)
    def load_data(nrows):
        data = pd.read_csv(DATA_URL)
        return data    
    data = load_data(100000)

    st.title("Información Histórica Académica y Sociodemográfica")
    st.header("1. Distribución Estudiantes por Estrato")

    #location_bog = pd.read_csv("georeferencia_localidad_bog.csv", sep=";")


    #px.set_mapbox_access_token(open(".mapbox_token").read())
    #fig2 = px.scatter_mapbox(data, lat="centroid_latlatitude", lon="centroid_lon", color="peak_hour", size="car_hours", color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

    #A partir de aqui escribir ale y nico
    st.write("Grafico por Filtros")
    estado=st.multiselect(label='Estado de Estudiante',options=['DESERTOR','GRADUADO'],default=['DESERTOR','GRADUADO'])
    ciclo=st.multiselect(label='Ciclos Propedeuticos',options=['TECNICO','TECNOLOGIA','PROFESIONAL'],default=['TECNICO','TECNOLOGIA','PROFESIONAL'])
    to_plot=data[(data['ESTADO'].isin(estado)) & (data['CICLO'].isin(ciclo))].groupby(['ESTRATO','ESTADO'])['key'].count().reset_index()

    fig = px.bar(to_plot,x='ESTRATO', y='key', color='ESTADO',labels={'ESTRATO':'ESTRATO','key':'Total Estudiantes'} , title='Estudiantes por Estrato y Estado')
    st.plotly_chart(fig)
    
if functionality=='Calculadora':
    st.write("""
    # Predicción de la deserción *estudiantil* en IETC
    """)

    def user_input_features():
        sepal_length = st.slider('Notas definitivas del estudiante', 0, 3, 5)
        sepal_width = st.slider('Notas promedio', 0, 3, 5)
        petal_length = st.slider('Edad del estudiante', 15, 30, 45)
        petal_width = st.slider('Estrato', 1, 3, 5)
        data = {'Notas_definitivas': sepal_length,
                'Notas_Promedio': sepal_width,
                'Edad': petal_length,
                'Estrato': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features



    df=user_input_features()
    st.subheader('User Input parameters')
    st.write(df)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    clf = RandomForestClassifier()
    clf.fit(X, Y)

    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Estado de los estudiantes')
    st.write(iris.target_names)

    st.subheader('Prediction')
    st.write(iris.target_names[prediction])
    #st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

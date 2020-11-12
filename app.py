import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

DATA_URL = ("grad_desert.csv")
@st.cache(persist=True)

def load_csv(file_name):
    data = pd.read_csv(file_name)
    return data

st.sidebar.title("Tablero Deserción Estudiantil ETITC")
st.sidebar.write("A continuación se muestran las diferentes opciones de visualización de datos:")
functionality = st.sidebar.radio('¿Qué visualización desea?',('Información Histórica','Predicción', 'Calculadora')) 

if functionality=='Información Histórica':
    
    
    data = pd.read_csv("grad_desert.csv")
    st.title("Información Histórica Académica y Sociodemográfica")
    st.header("1. Distribución Estudiantes por Localidad y Estado Académico")
    estado1 = st.multiselect(label='Estado de Estudiante', options=['DESERTOR', 'GRADUADO'], default=['DESERTOR', 'GRADUADO']) 

    #limpiar datos localidad en archivo principal
    data['LOCALIDAD'] = data['LOCALIDAD'].fillna('')
    data['LOCALIDAD'] = data['LOCALIDAD'].apply(lambda x: x.replace('LA CANDELARIA','CANDELARIA').replace('RAFAEL URIBE','RAFAEL URIBE URIBE'))

    location_bog = pd.read_csv("georeferencia_localidad_bog.csv")
    location_bog
    data_map = data.merge(location_bog, how="left", on="LOCALIDAD").drop(columns=["CODIGO", "gp"], axis=1).rename(columns={"LONGITUD":"long_localidad", "LATITUD":"lat_localidad"})

    #Mapa por localidad
    to_map = data_map[data_map['ESTADO'].isin(estado1)].groupby(["LOCALIDAD", "long_localidad", "lat_localidad"])['key'].count().reset_index().rename(columns={"key":"Num_estudiantes", "LOCALIDAD":"Localidad"})  

    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(to_map, lat="lat_localidad", lon="long_localidad", hover_name="LOCALIDAD", size="Num_estudiantes", size_max=20, zoom=10)
    st.plotly_chart(fig)
    
    if st.checkbox("Mostrar datos"):
        st.table(to_map[["Localidad", "Num_estudiantes"]].sort_values(by="Num_estudiantes", ascending=False).set_index("Localidad"))

    
    #A partir de aqui escribir ale y nico
    st.write("Grafico por Filtros")
    estado=st.multiselect(label='Estado de Estudiante',options=['DESERTOR','GRADUADO'],default=['DESERTOR','GRADUADO'])
    estado
    ciclo=st.multiselect(label='Ciclos Propedeuticos',options=['TECNICO','TECNOLOGIA','PROFESIONAL'],default=['TECNICO','TECNOLOGIA','PROFESIONAL'])
    ciclo
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

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
DATA_URL = ("caracterizacion_estudiantes_clean.csv")
functionality = st.sidebar.radio('Que Visualizacion Desea',('Mapa Demografico','Prediccion')) 

if functionality=='Mapa Demografico':
    @st.cache(persist=True,allow_output_mutation=True)
    def load_data():
        types={'TIPO':str,'DOCUMENTO':str,'GENERO':str,'DEPARTAMENTO':str,
               'FECHA NACIMIENTO':str,'TITULO_BACHILLER':str,'EDAD':str,
               'ESTADO CIVIL':str,'EPS':str,'INGRESO FAMILIAR':str,
               'EGRESOS FAMILIAR':str,'SITUACION LABORAL':str,'ESTRATO':float,
               'TENENCIA DE VIVIENDA':str,'NUMERO DE HERMANOS':float,
               'NUMERO DE PERSONAS CONVIVE':float,'TIPO_VIVIENDA':str,
               'GRUPO_VULNERABLE':str,'ID':str,'key':str,'lat':float,'lon':float,'localidad2':str}
        data = pd.read_csv(DATA_URL,dtype=types)
        return data    
    data = load_data()
    st.header("Localizacion Estudiantes por Estrato")
    st.write(data.columns)
    ESTRATO = st.slider("Estrato", 1, 6)
    st.map(data[data['ESTRATO']==ESTRATO][['lat','lon']])

if functionality=='Prediccion':
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

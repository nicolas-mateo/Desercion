import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
DATA_URL = ("caracterizacion_estudiantes_clean.csv")
functionality = st.sidebar.radio('Que Visualizacion Desea',('Mapa Demografico','Prediccion')) 

if functionality=='Mapa Demografico':
    @st.cache(persist=True)
    def load_data(nrows):
        data = pd.read_csv(DATA_URL)
        return data    
    data = load_data(100000)
    print(data.describe())
    st.header("Viviendas por Estrato")
    ESTRATO = st.slider("Estrato", 1, 6)
    st.map(data.query("ESTRATO >= @ESTRATO")[["latitude", "longitude"]].dropna(how="any"))

    st.write("Plotly graph")
    fig = px.line(data, x="key", y="EDAD", title='Life expectancy in Canada')
    st.plotly_chart(fig)
    
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

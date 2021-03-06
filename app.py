#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import base64
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

#Function to load the data in files CSV

@st.cache(persist=True)
def load_csv(file_name):
    data = pd.read_csv(file_name)
    return data

#Information displayed in the sidebar
st.sidebar.title("Tablero Deserción Estudiantil ETITC")
st.sidebar.write("A continuación se muestran las diferentes opciones de visualización de datos:")
functionality = st.sidebar.radio('¿Qué visualización desea?',('Información Histórica','Informacion Activos', 'Calculadora')) 

#To show the historical information
if functionality=='Información Histórica':
    
    #Import student data
    data = pd.read_csv("grad_desert.csv")
    
    #Import the information of localities in Bogota
    location_bog = pd.read_csv("georeferencia_localidad_bog.csv",sep=';')

    data['EMPLEO']=data['EMPLEO'].fillna('SIN INFORMACION')
    st.title("Información Histórica Académica y Sociodemográfica")
    st.header("1. Distribución Estudiantes por Localidad y Estado Académico")
    estado1 = st.multiselect(label='Estado de Estudiante', options=['DESERTOR', 'GRADUADO'], default=['DESERTOR', 'GRADUADO']) 

    #Clean primary dataframe data for further plots
    data['EMPLEO']=data['EMPLEO'].fillna('SIN INFORMACION')
    data['LOCALIDAD'] = data['LOCALIDAD'].fillna('')
    data['LOCALIDAD'] = data['LOCALIDAD'].apply(lambda x: x.replace('LA LA CANDELARIA','LA CANDELARIA').replace('RAFAEL URIBE','RAFAEL URIBE URIBE'))

    #Join the information of students with the data of localities
    data_map = data.merge(location_bog, how="left", on="LOCALIDAD").drop(columns=["CODIGO", "gp"], axis=1).rename(columns={"LONGITUD":"long_localidad", "LATITUD":"lat_localidad"})

    #GroupBy by locations and student status
    to_map = data_map[data_map['ESTADO'].isin(estado1)].groupby(["LOCALIDAD", "long_localidad", "lat_localidad"])['key'].count().reset_index().rename(columns={"key":"Num_estudiantes", "LOCALIDAD":"Localidad"})  

    #Create and Show de map
    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig1 = px.scatter_mapbox(to_map, lat="lat_localidad", lon="long_localidad", hover_name="Localidad", size="Num_estudiantes", size_max=20, zoom=10)
    st.plotly_chart(fig1)

    #Add checkbox to show the map data    
    if st.checkbox("Mostrar datos"):
        st.table(to_map[["Localidad", "Num_estudiantes"]].sort_values(by="Num_estudiantes", ascending=False).set_index("Localidad"))

    #SECOND CHART
    #Title and Multiselect for the second graph
    st.header("2. Estudiantes por Estrato y Estado")
    estado=st.multiselect(label='Estado de Estudiante',options=['DESERTOR','GRADUADO'],default=['DESERTOR','GRADUADO'],key=1231245151)
    ciclo=st.multiselect(label='Ciclos Propedeuticos',options=['TECNICO','TECNOLOGIA','PROFESIONAL'],default=['TECNICO','TECNOLOGIA','PROFESIONAL'],key=4239523092)
    
    #Create GroupBy 
    to_plot=data[(data['ESTADO'].isin(estado)) & (data['CICLO'].isin(ciclo))].groupby(['ESTRATO','ESTADO'])['key'].count().reset_index()

    #Create and show the bar chart
    fig2 = px.bar(to_plot,x='ESTRATO', y='key', color='ESTADO',labels={'ESTRATO':'ESTRATO','key':'Total Estudiantes'} )
    st.plotly_chart(fig2)

    #THIRD CHART
    ## Sankey diagram
    st.header("3. Diagrama de Flujo entre Ciclo y Estado")

    #Create and show the sankey diagram
    z1=data.groupby(['ESTADO','CICLO'])['key'].count().reset_index()
    z1['Percentage'] = 100 * z1['key']  / z1['key'].sum()
    z1.replace({'PROFESIONAL':2, 'TECNICO':3,'TECNOLOGIA':4},inplace=True)
    z1['ESTADO']=pd.factorize(z1['ESTADO'])[0]
    z1['ESTADO']=z1['ESTADO'].astype("category")
    z1['CICLO']=z1['CICLO'].astype("category")
    source = z1.CICLO.tolist()
    target1=z1['ESTADO'].tolist()
    value1=z1['Percentage'].tolist()
    label=['DESERTOR','GRADUADO','PROFESIONAL', 'TECNICO' ,'TECNOLOGIA']
    link=dict(source=source,target=target1,value=value1)
    node = dict(label = label, pad=100, line = dict(color = "black", width = 0.5), thickness=5)
    sank = go.Sankey(link = link, node=node)
    fig3=go.Figure(data=sank)
    st.plotly_chart(fig3)

    #FOURTH CHART
    #Title and multiselect
    st.header("4. Histograma de Promedios")
    estado2=st.multiselect(label='Estado de Estudiante',options=['DESERTOR','GRADUADO'],default=['DESERTOR','GRADUADO'],key=32654897123)
    to_plot=data[(data['ESTADO'].isin(estado2)) & (data['PROMEDIO']>0)]

    #Create and show the histrogram
    fig5 = px.histogram(to_plot,x='PROMEDIO', color='ESTADO')
    st.plotly_chart(fig5)

    #FIFTH CHART
    #Title
    st.header('5. Distribucion de Promedios')
    
    #Create and show the line chart
    tecn=data[data['CICLO']=='TECNICO']['PROMEDIO']
    tecnolo=data[data['CICLO']=='TECNOLOGIA']['PROMEDIO']
    prof=data[data['CICLO']=='PROFESIONAL']['PROMEDIO']
    fig9 = ff.create_distplot([tecn,tecnolo,prof], ['TECNICO','TECNOLOGIA','PROFESIONAL'], show_hist=False)
    st.plotly_chart(fig9)

    #SIXTH CHART
    ## Sunburst
    st.header("6. Grafico Sunburst")
    nota=data.groupby(['PROGRAMA','CICLO','ESTADO', 'key'])['PROMEDIO'].mean().reset_index()
    fig4 = px.sunburst(nota, path=['CICLO','ESTADO', 'PROGRAMA'],  color='PROMEDIO')
    st.plotly_chart(fig4)

    #SEVENTH CHART
    st.header('7. Empleo')
    estado3=st.selectbox(label='Estado de Estudiante', options=['DESERTOR', 'GRADUADO'])
    total_por_ciclo=data[data['ESTADO']==estado3].groupby('CICLO')['key'].count().reset_index()
    empleo=data[data['ESTADO']==estado3].groupby(['CICLO','EMPLEO'])['key'].count().reset_index()
    toplot=empleo.merge(total_por_ciclo, on='CICLO',how='inner')
    toplot['PROPORCION']=toplot['key_x']/toplot['key_y']
    fig10 = px.bar(toplot, x="CICLO", y="PROPORCION",
             color='EMPLEO', barmode='group',
             height=400, labels={'PROPORCION':'Proporcion de Estudiantes'})
    st.plotly_chart(fig10)


#To show the calculator
if functionality=='Calculadora':
    #Title
    st.write("""
    # Predicción de la deserción *estudiantil* en IETC
    """)
    
    #Input parameters
    def user_input_features():
        prom = st.slider('Promedio del estudiante', 0.0, 5.0)
        empleo = st.selectbox('Situacion laboral', options = ["DESEMPLEADO", "EMPLEADO", "INDEPENDIENTE", "OTRO", "SIN_INFO"])
        estrato = st.slider('Estrato', 0, 6)
        programa = st.selectbox("Programa que estudia", 
        options =  ['PROGRAMA_INGENIERIA_DE_SISTEMAS',
                    'PROGRAMA_INGENIERIA_ELECTROMECANICA',
                    'PROGRAMA_INGENIERIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES',
                    'PROGRAMA_INGENIERIA_EN_PROCESOS_INDUSTRIALES',
                    'PROGRAMA_INGENIERIA_MECANICA',
                    'PROGRAMA_INGENIERIA_MECATRONICA',
                    'PROGRAMA_TECNICA_PROFESIONAL_EN_DIBUJO_MECANICO_Y_DE_HERRAMIENTAS_INDUSTRIALES',
                    'PROGRAMA_TECNICA_PROFESIONAL_EN_PROCESOS_DE_MANUFACTURA',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_COMPUTACION',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_DISENO_DE_MAQUINAS',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTROMECANICA',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTRONICA_INDUSTRIAL',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_MANTENIMIENTO_INDUSTRIAL',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_MECATRONICA',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_PROCESOS_INDUSTRIALES',
                    'PROGRAMA_TECNICO_PROFESIONAL_EN_SISTEMAS',
                    'PROGRAMA_TECNOLOGIA_EN_AUTOMATIZACION_INDUSTRIAL',
                    'PROGRAMA_TECNOLOGIA_EN_DESARROLLO_DE_SOFTWARE',
                    'PROGRAMA_TECNOLOGIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES',
                    'PROGRAMA_TECNOLOGIA_EN_ELECTROMECANICA',
                    'PROGRAMA_TECNOLOGIA_EN_GESTION_DE_FABRICACION_MECANICA',
                    'PROGRAMA_TECNOLOGIA_EN_MECATRONICA',
                    'PROGRAMA_TECNOLOGIA_EN_MONTAJES_INDUSTRIALES',
                    'PROGRAMA_TECNOLOGIA_EN_PROCESOS_INDUSTRIALES',
                    'PROGRAMA_TECNOLOGIA_EN_PRODUCCION_INDUSTRIAL',
                    'PROGRAMA_TECNOLOGIA_EN_SISTEMAS'])

        data = {
        'PROMEDIO': 0,
        'EMPLEO_DESEMPLEADO': 0,
        'EMPLEO_EMPLEADO': 0,
        'EMPLEO_INDEPENDIENTE': 0,
        'EMPLEO_OTRO': 0,
        'EMPLEO_SIN_INFO': 0,
        'ESTRATO_0': 0,
        'ESTRATO_1': 0,
        'ESTRATO_2': 0,
        'ESTRATO_3': 0,
        'ESTRATO_4': 0,
        'ESTRATO_5': 0,
        'ESTRATO_6': 0,
        'PROGRAMA_INGENIERIA_DE_SISTEMAS': 0,
        'PROGRAMA_INGENIERIA_ELECTROMECANICA': 0,
        'PROGRAMA_INGENIERIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES': 0,
        'PROGRAMA_INGENIERIA_EN_PROCESOS_INDUSTRIALES': 0,
        'PROGRAMA_INGENIERIA_MECANICA': 0,
        'PROGRAMA_INGENIERIA_MECATRONICA': 0,
        'PROGRAMA_TECNICA_PROFESIONAL_EN_DIBUJO_MECANICO_Y_DE_HERRAMIENTAS_INDUSTRIALES': 0,
        'PROGRAMA_TECNICA_PROFESIONAL_EN_PROCESOS_DE_MANUFACTURA': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_COMPUTACION': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_DISENO_DE_MAQUINAS': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTROMECANICA': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTRONICA_INDUSTRIAL': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_MANTENIMIENTO_INDUSTRIAL': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_MECATRONICA': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_PROCESOS_INDUSTRIALES': 0,
        'PROGRAMA_TECNICO_PROFESIONAL_EN_SISTEMAS': 0,
        'PROGRAMA_TECNOLOGIA_EN_AUTOMATIZACION_INDUSTRIAL': 0,
        'PROGRAMA_TECNOLOGIA_EN_DESARROLLO_DE_SOFTWARE': 0,
        'PROGRAMA_TECNOLOGIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES': 0,
        'PROGRAMA_TECNOLOGIA_EN_ELECTROMECANICA': 0,
        'PROGRAMA_TECNOLOGIA_EN_GESTION_DE_FABRICACION_MECANICA': 0,
        'PROGRAMA_TECNOLOGIA_EN_MECATRONICA': 0,
        'PROGRAMA_TECNOLOGIA_EN_MONTAJES_INDUSTRIALES': 0,
        'PROGRAMA_TECNOLOGIA_EN_PROCESOS_INDUSTRIALES': 0,
        'PROGRAMA_TECNOLOGIA_EN_PRODUCCION_INDUSTRIAL': 0,
        'PROGRAMA_TECNOLOGIA_EN_SISTEMAS': 0}
        data["PROMEDIO"] = (prom - 3.261060)/0.891188 
        empleo = "EMPLEO_"+empleo 
        data[empleo] = 1
        estrato = f"ESTRATO_{estrato}"
        data[estrato] = 1
        data[programa] = 1 
        features = pd.DataFrame(data, index=[0])
        return features
    
    #Calculate the prediction
    df=user_input_features()
    modelr=pickle.load(open('logreg.sav', 'rb'))
    predictions=modelr.predict_proba(df)
    decision=modelr.predict(df)
    
    #Show the prediction
    st.subheader('Probabilidades')
    show = pd.DataFrame(data=predictions, columns=["Graduado", "Desertor"])
    st.write(show)


#Display information about current students
if functionality=='Informacion Activos':
    
    #Title
    st.title("Información Académica y Sociodemográfica Estudiantes Actuales")
    
    #Import the data of active students, and load the logistic regression model
    data = pd.read_csv("activos.csv")
    df= pd.read_csv('df_activos_Final.csv')
    modelr=pickle.load(open('logreg.sav', 'rb'))
    
    #Predict probability of desertion for active student base.
    df['PREDICTION']=modelr.predict(df[["PROMEDIO","EMPLEO_DESEMPLEADO","EMPLEO_EMPLEADO","EMPLEO_INDEPENDIENTE","EMPLEO_OTRO",
    "EMPLEO_SIN_INFO","ESTRATO_0","ESTRATO_1","ESTRATO_2","ESTRATO_3","ESTRATO_4","ESTRATO_5",
    "ESTRATO_6","PROGRAMA_INGENIERIA_DE_SISTEMAS","PROGRAMA_INGENIERIA_ELECTROMECANICA","PROGRAMA_INGENIERIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES",
    "PROGRAMA_INGENIERIA_EN_PROCESOS_INDUSTRIALES","PROGRAMA_INGENIERIA_MECANICA","PROGRAMA_INGENIERIA_MECATRONICA",
    "PROGRAMA_TECNICA_PROFESIONAL_EN_DIBUJO_MECANICO_Y_DE_HERRAMIENTAS_INDUSTRIALES","PROGRAMA_TECNICA_PROFESIONAL_EN_PROCESOS_DE_MANUFACTURA",
    "PROGRAMA_TECNICO_PROFESIONAL_EN_COMPUTACION","PROGRAMA_TECNICO_PROFESIONAL_EN_DISENO_DE_MAQUINAS",
    "PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTROMECANICA","PROGRAMA_TECNICO_PROFESIONAL_EN_ELECTRONICA_INDUSTRIAL",
    "PROGRAMA_TECNICO_PROFESIONAL_EN_MANTENIMIENTO_INDUSTRIAL","PROGRAMA_TECNICO_PROFESIONAL_EN_MECATRONICA",
    "PROGRAMA_TECNICO_PROFESIONAL_EN_PROCESOS_INDUSTRIALES","PROGRAMA_TECNICO_PROFESIONAL_EN_SISTEMAS",
    "PROGRAMA_TECNOLOGIA_EN_AUTOMATIZACION_INDUSTRIAL","PROGRAMA_TECNOLOGIA_EN_DESARROLLO_DE_SOFTWARE",
    "PROGRAMA_TECNOLOGIA_EN_DISENO_DE_MAQUINAS_Y_PRODUCTOS_INDUSTRIALES","PROGRAMA_TECNOLOGIA_EN_ELECTROMECANICA",
    "PROGRAMA_TECNOLOGIA_EN_GESTION_DE_FABRICACION_MECANICA","PROGRAMA_TECNOLOGIA_EN_MECATRONICA",
    "PROGRAMA_TECNOLOGIA_EN_MONTAJES_INDUSTRIALES","PROGRAMA_TECNOLOGIA_EN_PROCESOS_INDUSTRIALES",
    "PROGRAMA_TECNOLOGIA_EN_PRODUCCION_INDUSTRIAL","PROGRAMA_TECNOLOGIA_EN_SISTEMAS"]])
    data=data.merge(df[['key','PREDICTION']], on='key',how='inner')
    prediccion=data.groupby(['PROGRAMA','PREDICTION'])['key'].count().reset_index()
    totales=data.groupby('PROGRAMA')['key'].count().reset_index()
    proporciones=prediccion.merge(totales[['PROGRAMA','key']],on='PROGRAMA',how='inner')
    
    #Group desertion and graduation by program, and find their proportions

    proporciones['Proporcion']=100*proporciones['key_x']/proporciones['key_y']
    proporciones=proporciones.drop(columns=['key_x','key_y'])
    proporciones=proporciones.pivot(index='PROGRAMA',columns='PREDICTION')['Proporcion'].reset_index().fillna(0)
    proporciones.columns.name = None
    programas_grad=proporciones.sort_values(by=0,ascending=False).head(5)
    programas_des=proporciones.sort_values(by=1,ascending=False).head(5)

    #Plot best and worst performing programs, according to our predictions.
    fig24=px.bar(programas_grad,x='PROGRAMA',y=[0,1],labels={0:'(%)Graduados',1:'(%) Desertores'},barmode='group')

    fig26 = go.Figure(data=[
    go.Bar(name='(%)Graduados', x=programas_des['PROGRAMA'], y=programas_des[0]),
    go.Bar(name='(%) Desertores', x=programas_des['PROGRAMA'], y=programas_des[1])
    ])
    fig26.update_layout(barmode='group',width=800,height=600)
    
    st.header("1. Programas con Mayor Tendencia a Graduacion")
    st.plotly_chart(fig24)
    st.header("2. Programas con Mayor Tendencia a Desercion")
    st.plotly_chart(fig26)

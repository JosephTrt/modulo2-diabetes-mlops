# Imports

# FastAPI
from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse

# Base
import os
import pandas as pd
import numpy as np
import joblib
import subprocess
from src.configuraciones import config


# from Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler



# API
app = FastAPI()

ruta_actual = os.getcwd()

def clasificacion(pipeline_de_test, datos_de_test):
    
    # Pre-procesamineto
    # Quitar espacios en blanco
    datos_de_test = datos_de_test.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Map del target
    # datos_de_test['class'] = datos_de_test['class'].map({'Positive': 1, 'Negative': 0})

    # Aplicacion de las features
    datos_de_test = datos_de_test[config.FEATURES]


    # Clasificacion
    resultado_clasificacion = pipeline_de_test.predict(datos_de_test)
    
    # Desmapeo
    resultado_clasificacion_map = np.where(resultado_clasificacion == 1, 'Positive', 'Negative')

    return resultado_clasificacion, resultado_clasificacion_map, datos_de_test


@app.post("/obtencion-y-clasificacion-datos")
def publicar_mensaje(file: UploadFile = File(...)):

    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")
    
    try:
        # Guardar el archivo CSV subido temporalmente
        file_location = f"{ruta_actual}/{file.filename}"

        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Leer el archivo CSV
        df_de_los_datos_subidos = pd.read_csv(file_location)

        # Cargar el pipeline de producción
        ruta_modelo = os.path.join(ruta_actual, "src/pipeLineClassificationModel_v2.joblib")  # Actualización de la ruta
        pipeline_de_produccion = joblib.load(ruta_modelo)

        # Hacer la clasificacion
        resultado_clasificacion, resultado_clasificacion_map, datos_test_procesados = clasificacion(pipeline_de_produccion, df_de_los_datos_subidos)

        # Concatenar los datos procesados y las predicciones
        df_concatenado = pd.concat([datos_test_procesados, pd.Series(resultado_clasificacion_map, name="Clasificaciones"), pd.Series(resultado_clasificacion, name="Clasificacion sin mapear")], axis=1)

        # Guardar el archivo de salida
        output_file = f"{ruta_actual}/salida_datos_y_predicciones.csv"
        df_concatenado.to_csv(output_file, index=False)

        # Devolver el archivo resultante
        return FileResponse(output_file, media_type="application/octet-stream", filename="salida_datos_y_predicciones.csv")

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío o tiene un formato incorrecto.")
    
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo de predicción.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

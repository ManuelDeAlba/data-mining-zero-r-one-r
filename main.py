from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.classifier import OneRClassifier

app = FastAPI()
dataframe = None

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
#? http://localhost:8000/?iteraciones=2&modelo=one-r&train_size=70&clase=clase
async def home(request: Request, modelo="zero-r", iteraciones=1, clase="", train_size=70):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={"modelo": modelo.lower(), "iteraciones": iteraciones, "clase": clase, "train_size": train_size}
    )

@app.post("/upload-file")
async def uploadFile(file: UploadFile, modelo: str = Form(...), iteraciones: int = Form(...), clase: str = Form(...), train_size: int = Form(...)):
    global dataframe

    # Leer el archivo CSV pasado por el usuario
    content = await file.read()
    decoded_content = content.decode("utf-8")
    data = StringIO(decoded_content)
    dataframe = pd.read_csv(data)

    # Separar el dataframe en atributos normales y clase
    try:
        X = dataframe.drop(columns=[clase])
        y = dataframe[clase]
    except KeyError:
        return {"error": True, "message": "La clase especificada no existe en el archivo"}

    resultados = []

    # Se repetirá el proceso dependiendo de las iteraciones
    for i in range(iteraciones):
        # Se divide el conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, shuffle=True)

        # Unir los conjuntos de entrenamiento y prueba para mostrarlos en la tabla de resultados
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        # Aplicar el modelo correspondiente
        if modelo == "zero-r":
            # En ZeroR, se predice la clase más frecuente
            mas_frecuente = y_train.value_counts().idxmax()
            
            # Se realizan las "predicciones"
            y_pred = [mas_frecuente] * len(y_test)

            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            resultados.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicciones": y_pred.tolist(),
                "columnas_train": train.columns.to_list(),
                "valores_train": train.values.tolist(),
                "columnas_test": test.columns.to_list(),
                "valores_test": test.values.tolist()
            })
        elif modelo == "one-r":
            # Crear el clasificador OneR
            oner = OneRClassifier()

            # Crear un objeto LabelEncoder para transformar las variables categóricas en numéricas
            label_encoder = LabelEncoder()

            # Codificar las variables categóricas
            #? Para regresar los valores a su forma original, se puede usar label_encoder.inverse_transform([...])
            #! Aquí no se puede usar get_dummies porque se pierde la relación entre las variables al modificarse la estructura de la tabla
            X_train_encoded = X_train.apply(label_encoder.fit_transform)
            X_test_encoded = X_test.apply(label_encoder.fit_transform)
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test) # Se usa solo transform para usar los valores ya codificados con y_train

            # Entrenar el modelo
            oner.fit(X_train_encoded.values, y_train_encoded)

            # Predecir
            y_pred_encoded = oner.predict(X_test_encoded.values)

            # Obtener las predicciones en su forma original
            y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

            # # Calcular métricas
            accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
            precision = precision_score(y_test_encoded, y_pred_encoded, average="macro")
            recall = recall_score(y_test_encoded, y_pred_encoded, average="macro")
            f1 = f1_score(y_test_encoded, y_pred_encoded, average="macro")

            resultados.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicciones": y_pred_decoded.tolist(),
                "columnas_train": train.columns.to_list(),
                "valores_train": train.values.tolist(),
                "columnas_test": test.columns.to_list(),
                "valores_test": test.values.tolist()
            })

    # Resultados
    return {
        "resultados": resultados,
        "modelo": modelo,
        "iteraciones": iteraciones,
        "clase": clase,
        "train_size": train_size
    }

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
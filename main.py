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

            resultados.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
        elif modelo == "one-r":
            # Crear el clasificador OneR
            oner = OneRClassifier()

            # Crear un objeto LabelEncoder para transformar las variables categóricas en numéricas
            label_encoder = LabelEncoder()

            # Codificar las variables categóricas
            #? Para regresar los valores a su forma original, se puede usar label_encoder.inverse_transform([...])
            #! Aquí no se puede usar get_dummies porque se pierde la relación entre las variables al modificarse la estructura de la tabla
            X_train = X_train.apply(label_encoder.fit_transform)
            X_test = X_test.apply(label_encoder.fit_transform)
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.fit_transform(y_test)

            # Entrenar el modelo
            oner.fit(X_train.values, y_train)

            # Predecir
            y_pred = oner.predict(X_test.values)

            # # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            resultados.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

    # Resultados
    return {"resultados": resultados, "modelo": modelo, "iteraciones": iteraciones, "clase": clase, "train_size": train_size}

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI()
dataframe = None

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, modelo="zero-r", iteraciones=1, clase="", test_size=30):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={"modelo": modelo.lower(), "iteraciones": iteraciones, "clase": clase, "test_size": test_size}
    )

@app.post("/upload-file")
async def uploadFile(file: UploadFile, modelo: str = Form(...), iteraciones: int = Form(...), clase: str = Form(...), test_size: int = Form(...)):
    global dataframe

    # Leer el archivo CSV pasado por el usuario
    content = await file.read()
    decoded_content = content.decode("utf-8")
    data = StringIO(decoded_content)
    dataframe = pd.read_csv(data)

    # Separar el dataframe en atributos normales y clase
    X = dataframe.drop(columns=[clase])
    y = dataframe[clase]

    # Se repetirá el proceso dependiendo de las iteraciones
    for i in range(iteraciones):
        # Se divide el conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=True)

        # Aplicar el modelo correspondiente
        if modelo == "zero-r":
            pass
        elif modelo == "one-r":
            pass

    # Resultados
    return {"modelo": modelo, "iteraciones": iteraciones, "clase": clase, "test_size": test_size}

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
from fastapi import FastAPI, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

app = FastAPI()
dataframe = None

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, modelo="zero-r", iteraciones=1, clase=""):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={"modelo": modelo.lower(), "iteraciones": iteraciones, "clase": clase}
    )

@app.post("/upload-file")
async def uploadFile(file: UploadFile, modelo: str = Form(...), iteraciones: int = Form(...), clase: str = Form(...)):
    global dataframe

    content = await file.read()
    decoded_content = content.decode("utf-8")
    data = StringIO(decoded_content)
    dataframe = pd.read_csv(data)
    
    X = dataframe.drop(columns=[clase])
    y = dataframe[clase]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("X_train\n", X_train)
    print("X_test\n", X_test)
    print("y_train\n", y_train)
    print("y_test\n", y_test)

    return dataframe.to_dict()

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
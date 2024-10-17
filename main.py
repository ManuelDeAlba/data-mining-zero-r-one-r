from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

app = FastAPI()
dataframe = None

@app.get("/")
def home():
    content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>5.2 Implementación de algoritmos Zero-R y One-R</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
        </head>
        <body>
            <form action="/upload-file" enctype="multipart/form-data" method="POST">
                <input name="file" type="file">
                <label>Iteraciones:
                    <input name="iteraciones" type="number" value="1">
                </label>
                <label>
                    <input name="modelo" type="radio" value="zero-r" checked>Zero-R
                </label>
                <label>
                    <input name="modelo" type="radio" value="one-r">One-R
                </label>
                <input type="submit">
            </form>
        </body>
        </html>
    """
    return HTMLResponse(content=content)

@app.post("/upload-file")
async def uploadFile(file: UploadFile):
    global dataframe

    content = await file.read()
    decoded_content = content.decode("utf-8")
    data = StringIO(decoded_content)
    dataframe = pd.read_csv(data)

    X_train, X_test, y_train, y_test = train_test_split(dataframe, test_size=0.3)

    return dataframe.to_dict()

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
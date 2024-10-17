from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
import pandas as pd
from io import StringIO

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
            <title>5.2 algo jeje</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
        </head>
        <body>
            <form action="/upload-file" enctype="multipart/form-data" method="POST">
                <input name="file" type="file">
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
    
    return dataframe.to_dict()

@app.get("/read-file")
async def readFile():
    if dataframe is not None:
        print(dataframe)
        return dataframe.to_dict()
    else:
        return {"error": True, "message": "No file uploaded yet"}
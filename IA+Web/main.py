from fastapi import FastAPI, File, UploadFile, Form
from uvicorn import run


app = FastAPI()


@app.get("/")
async def root() -> dict:
    return {"message": "Hello World"}


@app.post('/file')
def _file_upload(
        my_file: UploadFile = File(...),
        first: str = Form(...),
        second: str = Form("default value  for second"),
):
    return {
        "name": my_file.filename,
        "first": first,
        "second": second
    }


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
import numpy as np

from preprocess.ct import preprocess_ct_from_bytes
from preprocess.xray import preprocess_xray_from_bytes

app = FastAPI()

# Load models ONCE
ct_model = tf.keras.models.load_model("models/ct_model.keras")
xray_model = tf.keras.models.load_model("models/xray.keras")

CT_CLASSES = ["Benign", "Malignant", "Normal"]
XRAY_CLASSES = ["Covid", "Tuberculosis", "Pneumonia", "Normal"]

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict/ct")
async def predict_ct(file: UploadFile = File(...)):
    img_bytes = await file.read()
    inp = preprocess_ct_from_bytes(img_bytes)

    pred = ct_model.predict(inp)[0]
    idx = int(np.argmax(pred))

    return JSONResponse({
        "model": "CT",
        "prediction": CT_CLASSES[idx],
        "confidence": float(pred[idx] * 100),
        "probabilities": dict(zip(CT_CLASSES, pred.tolist()))
    })

@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    img_bytes = await file.read()
    inp = preprocess_xray_from_bytes(img_bytes)

    pred = xray_model.predict(inp)[0]
    idx = int(np.argmax(pred))

    return JSONResponse({
        "model": "X-Ray",
        "prediction": XRAY_CLASSES[idx],
        "confidence": float(pred[idx] * 100),
        "probabilities": dict(zip(XRAY_CLASSES, pred.tolist()))
    })

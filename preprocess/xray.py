import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

IMG_SIZE = (224, 224)

def preprocess_xray_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ---- CLAHE (lighter) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    processed = clahe.apply(gray)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    processed = cv2.resize(processed, IMG_SIZE)
    processed = processed.astype(np.float32)
    processed = preprocess_input(processed)

    return np.expand_dims(processed, axis=0)

import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)

def preprocess_ct_from_bytes(image_bytes):
    # Decode bytes → OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ---- CLAHE + Blur (same as training) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    blur = cv2.GaussianBlur(cl, (3, 3), 0)

    rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)

    # ---- Resize + EfficientNet preprocess ----
    rgb = cv2.resize(rgb, IMG_SIZE)
    rgb = rgb.astype(np.float32)
    rgb = preprocess_input(rgb)

    return np.expand_dims(rgb, axis=0)

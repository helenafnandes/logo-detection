from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import numpy as np
import cv2
from detect_logo import detect_and_return_logo  # Importa a função que detecta a logo e retorna a ROI

app = FastAPI()


# Função para ler a imagem do upload e processá-la
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Lê a imagem enviada
    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Salva a imagem temporariamente para passar ao modelo (pode ser otimizado)
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img)

    # Detecta a logo e retorna a ROI (logo recortada)
    rois = detect_and_return_logo(temp_path)

    # Se houver logo detectada, salva e retorna a primeira ROI
    if rois:
        roi_image = rois[0]  # Aqui pegamos apenas a primeira ROI detectada
        roi_path = "detected_logo.jpg"
        cv2.imwrite(roi_path, roi_image)

        return {"filename": "detected_logo.jpg", "message": "Logo detected", "roi_image_path": roi_path}

    return {"filename": file.filename, "message": "No logo detected"}


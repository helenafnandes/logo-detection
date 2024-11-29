import cv2
import torch
import sys
from pathlib import Path
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from ultralytics.utils.ops import non_max_suppression, scale_boxes

# Adicionar o YOLOv5 ao PYTHONPATH
from ultralytics.utils.torch_utils import select_device

import torch


yolo_model2 = torch.hub.load("ultralytics/yolov5", "custom", path="models/yolov5/runs/train/exp12/weights/best.pt", force_reload=True)  # local model
# Caminhos dos modelos e diretórios
yolo_model_path = 'models/yolov5/runs/train/exp12/weights/best.pt'  # YOLO modelo treinado
resnet_model_path = 'models/resnet/logo_classifier.pth'  # ResNet modelo treinado
data_dir = 'models/resnet/brandROIs'  # Diretório com as classes (pastas com nomes das marcas)

# Configurações do dispositivo
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = select_device('cpu')


# Inicializar o YOLO
yolo_device = select_device(device)
#yolo_model = DetectMultiBackend(yolo_model_path, device=yolo_device)
#yolo_stride, yolo_names = yolo_model.stride, yolo_model.names

# Inicializar o ResNet
# Inicializar o modelo ResNet com 788 classes
num_classes = 788  # Número de classes detectado
resnet_model = models.resnet18(pretrained=False)  # Carregar a arquitetura
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)  # Ajustar a última camada

# Carregar os pesos treinados no modelo
resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))

# Transferir o modelo para o dispositivo
resnet_model = resnet_model.to(device)
resnet_model.eval()  # Modo de avaliação

# Lista de classes para a ResNet
class_names = sorted(os.listdir(data_dir))

# Transformação para o modelo ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Função para redimensionar imagem com padding
def resize_with_padding(image, target_size=416):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded_image

# Função para detectar logo
# Função para detectar e retornar ROIs
# Função para detectar e retornar ROIs
def detect_logo(image_path):
    img0 = cv2.imread(image_path)  # Carrega a imagem original
    assert img0 is not None, f"Erro ao carregar a imagem: {image_path}"

    # Redimensiona com padding e prepara para o modelo YOLO
    img = resize_with_padding(img0, target_size=416)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Converte BGR para RGB e rearranja os canais
    img = torch.from_numpy(img.copy()).to(yolo_device).float() / 255.0  # Normaliza para [0, 1]
    img = img[None]  # Adiciona dimensão de batch

    # Realiza a predição

    pred = yolo_model2(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Retornar a primeira ROI detectada
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            x1, y1, x2, y2 = map(int, det[0, :4])
            roi = img0[y1:y2, x1:x2]
            return roi
    return None




# Função para classificar logo
def classify_logo(roi):
    if roi is None or not isinstance(roi, np.ndarray):
        print("ROI inválida ou vazia.")
        return None
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB
    roi_image = Image.fromarray(roi)  # Converte para formato PIL
    roi_transformed = resnet_transform(roi_image).unsqueeze(0).to(device)  # Aplica transformações

    with torch.no_grad():
        output = resnet_model(roi_transformed)  # Predição do ResNet
        _, predicted = torch.max(output, 1)  # Classe com maior probabilidade
        return class_names[predicted.item()]


# Função principal
def detect_and_classify(image_path):
    roi = detect_logo(image_path)
    brand = classify_logo(roi)
    return brand

if __name__ == "__main__":
    image_path = "teste.jpg"  # Caminho da imagem de teste

    brands = detect_and_classify(image_path)  # Detecta e classifica as logos
    if brands:
        print(f"Marcas detectadas: {', '.join(brands)}")
    else:
        print("Nenhuma logo detectada.")

import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from torchvision import models, transforms
from PIL import Image
import os

# Função para redimensionar a imagem mantendo a proporção e adicionar padding
def resize_with_padding(image, target_size=416):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Criando uma imagem preta (padding) com tamanho target_size
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded_image

# Função para detectar o logo com YOLOv5
def detect_logo(image_path):
    model_path = 'runs/train/exp12/weights/best.pt'  # Caminho para o modelo YOLOv5 treinado
    device = select_device('cpu')  # Forçar a usar a CPU

    model = DetectMultiBackend(model_path, device=device)
    stride, names = model.stride, model.names

    img0 = cv2.imread(image_path)  # Imagem original
    assert img0 is not None, f"Erro ao carregar a imagem: {image_path}"

    # Redimensionar imagem com padding
    img = resize_with_padding(img0, target_size=416)

    # Converter de BGR para RGB, e rearranjar para CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0  # Normalizar para [0, 1]
    img = img[None]  # Adicionar dimensão do lote

    # Inferência com YOLOv5
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)  # Coordenadas da ROI
                roi = img0[y1:y2, x1:x2]  # Região de interesse (logo detectado)
                return roi
    return None

# Função para classificar o logo usando ResNet
def classify_logo(roi_image):
    device = torch.device('cpu')

    model = models.resnet18(weights='IMAGENET1K_V1')  # Carregar o modelo pré-treinado
    for param in model.parameters():
        param.requires_grad = False

    data_dir = 'models/resnet/brandROIs'  # Diretório com suas imagens de treinamento
    num_classes = len(os.listdir(data_dir))  # Número de classes com base nas pastas

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('models/resnet/logo_classifier.pth'))
    model = model.to(device)
    model.eval()  # Colocar o modelo em modo de avaliação

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocessar a ROI para a entrada do ResNet
    roi_pil = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    roi_tensor = transform(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(roi_tensor)
        _, predicted = torch.max(output, 1)

    class_names = sorted(os.listdir(data_dir))
    return class_names[predicted.item()]

# Função principal para detectar e classificar a logo
def process_image(image_path):
    roi = detect_logo(image_path)
    if roi is not None:
        brand_name = classify_logo(roi)
        return brand_name
    else:
        return "Logo não detectado"

# Testando com uma imagem
if __name__ == "__main__":
    image_path = 'models/resnet/teste.jpg'
    brand_name = process_image(image_path)
    print(f'A marca detectada é: {brand_name}')

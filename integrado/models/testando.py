import sys

import cv2
import torch
from pathlib import Path
# Adicionar o YOLOv5 ao PYTHONPATH
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolo5'
sys.path.insert(0, str(ROOT))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

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

# Função para detectar objetos e mostrar as ROIs
def detect_and_show(image_path):
    # Carregar modelo
    model_path = 'runs/train/exp12/weights/best.pt'  # Caminho para o modelo treinado
    #device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
    device = select_device('cpu')  # Forçar a usar a CPU

    model = DetectMultiBackend(model_path, device=device)
    stride, names = model.stride, model.names

    # Carregar imagem
    img0 = cv2.imread(image_path)  # Imagem original
    assert img0 is not None, f"Erro ao carregar a imagem: {image_path}"

    # Redimensionar imagem com padding
    img = resize_with_padding(img0, target_size=416)

    # Converter de BGR para RGB, e rearranjar para CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0  # Normalizar para [0, 1]
    img = img[None]  # Adicionar dimensão do lote

    # Inferência
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Processar resultados
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Desenhar caixas na imagem
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)  # Coordenadas da ROI
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa
                cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar a imagem com as detecções
    cv2.imshow('Detections', img0)
    cv2.waitKey(0)  # Aguarda pressionar uma tecla
    cv2.destroyAllWindows()

# Caminho para a imagem de teste
if __name__ == "__main__":
    #image_path = input("Digite o caminho da imagem: ").strip()
    image_path = "../teste.jpg"
    detect_and_show(image_path)

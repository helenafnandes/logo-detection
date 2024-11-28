import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.datasets import letterbox
from utils.torch_utils import select_device

# Configurações
MODEL_PATH = 'yolov5/runs/train/exp12/weights/best.pt'  # Caminho para o modelo treinado
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU se disponível
IMG_SIZE = 416  # Tamanho da imagem de entrada

# Função para detectar objetos e mostrar as ROIs
def detect_and_show(image_path):
    # Carregar modelo
    device = select_device(DEVICE)
    model = DetectMultiBackend(MODEL_PATH, device=device)
    stride, names = model.stride, model.names

    # Carregar imagem
    img0 = cv2.imread(image_path)  # Imagem original
    assert img0 is not None, f"Erro ao carregar a imagem: {image_path}"
    img = letterbox(img0, IMG_SIZE, stride=stride)[0]  # Redimensionar com preenchimento
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR para RGB, HWC para CHW
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalizar para [0, 1]
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
    image_path = "test.jpg"
    detect_and_show(image_path)
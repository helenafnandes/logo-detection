import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

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

# Função para detectar a logo e retornar a ROI (recorte da logo)
def detect_and_return_logo(image_path):
    # Carregar modelo
    model_path = 'runs/train/exp12/weights/best.pt'  # Caminho para o modelo treinado
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

    roi_list = []

    # Processar resultados
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Extrair a ROI da logo detectada
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)  # Coordenadas da ROI
                roi = img0[y1:y2, x1:x2]  # Recorte da logo da imagem original
                roi_list.append(roi)

    return roi_list  # Retorna as ROIs encontradas (logo recortada)

if __name__ == "__main__":
    # Caminho da imagem de teste
    image_path = "../teste.jpg"

    # Chamar a função para detectar e retornar as ROIs
    rois = detect_and_return_logo(image_path)

    # Carregar a imagem original para exibição
    img_original = cv2.imread(image_path)

    # Verificar se alguma ROI foi detectada
    if rois:
        print(f"Total de logos detectadas: {len(rois)}")

        # Salvar as ROIs detectadas e desenhar retângulos na imagem original
        for idx, roi in enumerate(rois):
            # Coordenadas da ROI e outras informações
            x1, y1, x2, y2 = roi["bbox"]
            class_name = roi["class_name"]

            # Desenhar retângulo ao redor da ROI na imagem original
            cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Retângulo verde
            cv2.putText(img_original, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Nome da classe

            # Salvar a ROI como imagem separada
            roi_image = roi["image"]  # Obter a imagem recortada
            output_path = f"logo_detected_{idx + 1}.jpg"
            cv2.imwrite(output_path, roi_image)  # Salvar ROI
            print(f"Logo {idx + 1} salva em: {output_path}")

        # Exibir a imagem original com os retângulos e classes
        cv2.imshow("Detecções", img_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Salvar a imagem com as detecções
        annotated_path = "image_with_detections.jpg"
        cv2.imwrite(annotated_path, img_original)
        print(f"Imagem anotada salva em: {annotated_path}")

    else:
        print("Nenhuma logo foi detectada.")

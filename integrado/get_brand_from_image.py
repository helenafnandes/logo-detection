import torch
from PIL import Image
from torchvision import models, transforms
import os




# Carregar o modelo YOLOv5
yolo_model = torch.hub.load("ultralytics/yolov5", "custom", path="models/yolov5/runs/train/exp12/weights/best.pt", force_reload=True)

# Carregar o modelo ResNet18 na CPU
device = torch.device('cpu')  # Forçar a CPU
resnet_model = models.resnet18(weights='IMAGENET1K_V1')
for param in resnet_model.parameters():
    param.requires_grad = False

# Definir o número de classes (marcas) no seu dataset
data_dir = 'models/resnet/brandROIs'  # Diretório com suas imagens de marcas
num_classes = len(os.listdir(data_dir))  # Número de classes com base no número de pastas (marcas)

# Substituir a última camada para o número de classes do seu dataset
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)

# Carregar os pesos do modelo salvo na CPU
resnet_model.load_state_dict(torch.load('models/resnet/logo_classifier.pth', map_location=torch.device('cpu')))
resnet_model = resnet_model.to(device)
resnet_model.eval()

# Definir as transformações para as imagens de teste
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Função para testar a imagem com ResNet
def test_image(image, model, transform, device):
    image = transform(image).unsqueeze(0)  # Adicionar a dimensão do batch
    image = image.to(device)
    with torch.no_grad():
        output = model(image)  # Passar pela rede
        _, predicted = torch.max(output, 1)  # Pega o índice da classe com maior probabilidade
    return predicted.item()

# Função para integrar YOLOv5 e ResNet
def detect_and_classify(image_path, yolo_model, resnet_model, transform, device):
    # Iniciar a detecção com o YOLOv5
    results = yolo_model(image_path)
    # Extrair as caixas delimitadoras (bounding boxes) de interesse
    detections = results.pandas().xyxy[0]
    brand = ""

    # Iterar sobre as detecções e verificar se há uma logomarca
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        roi = Image.open(image_path).crop((xmin, ymin, xmax, ymax))  # Cortar a região da imagem

        # Classificar a ROI com ResNet
        predicted_class = test_image(roi, resnet_model, transform, device)

        # Listar as classes (nomes das pastas no diretório de treinamento)
        class_names = sorted(os.listdir(data_dir))
        print(f'A marca detectada na região é: {class_names[predicted_class]}')
        brand = class_names[predicted_class]
    return brand

def get_brand(image_path):
    brand = detect_and_classify(image_path, yolo_model, resnet_model, transform, device)
    return brand

# Caminho da imagem a ser processada
#image_path = 'teste.jpg'  # Substitua pelo caminho da imagem

# Chamar a função para detectar e classificar a logomarca
#detect_and_classify(image_path, yolo_model, resnet_model, transform, device)


#get_brand(image_path)
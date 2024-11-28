import torch
from torchvision import models, transforms
from PIL import Image
import os

# Usar GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo pré-treinado (ResNet18) e ajustar a última camada
model = models.resnet18(weights='IMAGENET1K_V1')  # ou 'ResNet18_Weights.DEFAULT'

# Congelar as camadas iniciais (não ajustar os pesos dessas camadas)
for param in model.parameters():
    param.requires_grad = False

# Definir o número de classes no seu dataset
data_dir = 'brandROIs'  # Diretório com suas imagens
num_classes = len(os.listdir(data_dir))  # Número de classes com base no número de pastas (marcas)

# Substituir a última camada para o número de classes do seu dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Carregar os pesos do modelo salvo
model.load_state_dict(torch.load('logo_classifier.pth'))
model = model.to(device)
model.eval()  # Colocar o modelo em modo de avaliação

# Definir as transformações para as imagens de teste
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Função para testar uma imagem
def test_image(image_path, model, transform, device):
    # Abrir a imagem
    image = Image.open(image_path).convert('RGB')

    # Aplicar as transformações
    image = transform(image).unsqueeze(0)  # Adicionar a dimensão do batch (1, C, H, W)
    image = image.to(device)

    # Fazer a previsão
    with torch.no_grad():
        output = model(image)  # Passar pela rede
        _, predicted = torch.max(output, 1)  # Pega o índice da classe com maior probabilidade

    return predicted.item()

# Caminho da imagem que você quer testar
image_path = 'teste.jpg'  # Substitua pelo caminho da imagem a ser testada

# Testar a imagem
predicted_class = test_image(image_path, model, transform, device)

# Lista das classes (nomes das pastas no diretório de treinamento)
class_names = sorted(os.listdir(data_dir))

# Exibir o nome da classe prevista
print(f'A imagem foi classificada como: {class_names[predicted_class]}')

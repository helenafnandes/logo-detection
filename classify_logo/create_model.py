from torchvision import datasets
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except UnidentifiedImageError:
            print(f"Warning: Ignoring invalid image {path}")
            return None  # Retorna None para imagens inválidas

        # Aplica as transformações (como a conversão para tensor) aqui
        sample = self.transform(sample)  # Aplica a transformação (a conversão para tensor)
        return sample, target

# Definir transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Diretório onde o dataset está armazenado
data_dir = 'brandROIs'

# Carregar o dataset com a classe personalizada
dataset = CustomImageFolder(root=data_dir, transform=transform)

# Filtrar imagens inválidas diretamente no DataLoader
def collate_fn(batch):
    # Filtrar imagens inválidas (None) durante o carregamento
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# Criar o DataLoader com a função customizada
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Usar GPU se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo pré-treinado (ResNet18) e ajustar a última camada
model = models.resnet18(pretrained=True)

# Congelar as camadas iniciais (não ajustar os pesos dessas camadas)
for param in model.parameters():
    param.requires_grad = False

# Substituir a última camada (fully connected) para o número de classes do seu dataset
num_classes = len(dataset.classes)  # Número de marcas únicas
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Mover o modelo para o dispositivo (GPU ou CPU)
model = model.to(device)

# Definir o critério de perda e o otimizador
criterion = nn.CrossEntropyLoss()  # Para classificação
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Só treinar a última camada

# Função de treinamento
def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()  # Modo de treinamento
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zerar gradientes

            outputs = model(inputs)  # Passar as imagens pelo modelo
            loss = criterion(outputs, labels)  # Calcular a perda
            loss.backward()  # Backpropagation
            optimizer.step()  # Atualizar os pesos

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print('Training Finished')
    # Salvar o modelo treinado
    torch.save(model.state_dict(), 'logo_classifier.pth')
    print("Modelo salvo com sucesso!")


# Treinar o modelo
train(model, train_loader, criterion, optimizer, epochs=10)

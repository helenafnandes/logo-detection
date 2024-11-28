import os
import shutil
import random

# Caminhos para as pastas de imagens e rótulos
dataset_dir = 'dataset'  # Caminho para a pasta do dataset
images_dir = os.path.join(dataset_dir, 'images')  # Caminho para a pasta de imagens
labels_dir = os.path.join(dataset_dir, 'labels')  # Caminho para a pasta de rótulos

# Caminhos para as pastas de treino e validação
train_images_dir = os.path.join(dataset_dir, 'images/train')
val_images_dir = os.path.join(dataset_dir, 'images/val')
train_labels_dir = os.path.join(dataset_dir, 'labels/train')
val_labels_dir = os.path.join(dataset_dir, 'labels/val')

# Criação das pastas de treino e validação
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Listar todas as imagens no diretório de imagens
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Embaralhar a lista de imagens para garantir divisão aleatória
random.shuffle(image_files)

# Definir a porcentagem de dados para validação (20% para validação e 80% para treino)
val_size = int(len(image_files) * 0.2)  # 20% para validação
train_size = len(image_files) - val_size

# Dividir as imagens e rótulos entre treino e validação
train_images = image_files[:train_size]
val_images = image_files[train_size:]

# Função para mover arquivos para o diretório de treino ou validação
def move_files(files, source_dir, dest_dir):
    for file in files:
        # Mover a imagem
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))
        # Mover o arquivo de rótulo correspondente (com o mesmo nome, mas com extensão .txt)
        label_file = file.replace('.jpg', '.txt')
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(dest_dir.replace('images', 'labels'), label_file))

# Mover os arquivos de treino e validação
move_files(train_images, images_dir, train_images_dir)
move_files(val_images, images_dir, val_images_dir)

print(f"Divisão concluída: {train_size} imagens de treino e {val_size} imagens de validação.")

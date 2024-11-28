import os
import shutil


def organize_dataset(brands_dir, labels_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            image_name = label_file.replace('.txt', '.jpg')

            for root_dir, _, files in os.walk(brands_dir):
                if image_name in files:
                    image_path = os.path.join(root_dir, image_name)
                    shutil.copy(image_path, os.path.join(output_images_dir, image_name))
                    shutil.copy(label_path, os.path.join(output_labels_dir, label_file))
                    break


if __name__ == "__main__":
    brands_dir = "../DATASETS/LogosInTheWild/litw_cleaned/data_cleaned/brands"
    labels_dir = "yolo_labels"
    output_images_dir = "dataset/images"
    output_labels_dir = "dataset/labels"

    organize_dataset(brands_dir, labels_dir, output_images_dir, output_labels_dir)
    print("Dataset organizado com sucesso!")

import os
import xml.etree.ElementTree as ET


def convert_bbox(size, box):
    """Converte coordenadas de XML para o formato YOLO."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(xml_file, output_dir):
    """Converte um único arquivo XML para o formato YOLO."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        yolo_annotations = []
        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Converte coordenadas para o formato YOLO
            bbox = convert_bbox((w, h), (xmin, xmax, ymin, ymax))
            yolo_annotations.append(f"0 {' '.join(map(str, bbox))}")

        # Salva o arquivo .txt correspondente
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        with open(os.path.join(output_dir, f"{base_name}.txt"), 'w') as out_file:
            out_file.write("\n".join(yolo_annotations))
    except Exception as e:
        print(f"Erro ao processar {xml_file}: {e}")


def process_all_annotations(dataset_dir, output_dir):
    """Processa todos os arquivos XML em todas as subpastas."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root_dir, _, files in os.walk(dataset_dir):
        for file_name in files:
            if file_name.endswith('.xml'):
                xml_path = os.path.join(root_dir, file_name)
                convert_annotation(xml_path, output_dir)


if __name__ == "__main__":
    # Caminho para a pasta "brands"
    dataset_dir = "../DATASETS/LogosInTheWild/litw_cleaned/data_cleaned/brands"
    # Pasta onde os arquivos .txt serão salvos
    output_dir = "yolo_labels"

    process_all_annotations(dataset_dir, output_dir)
    print(f"Conversão concluída! Arquivos salvos em: {output_dir}")

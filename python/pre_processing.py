from pathlib import Path

import numpy as np
import cv2

def pre_process_image(image_path:Path, results_path:Path = Path().parent / 'results' / 'pre_processing'):
    image_name = image_path.name
    result_image_path = results_path/image_name

    # Leemos la imagen
    image:np.ndarray = cv2.imread(image_path.__str__())

    # Separamos la imagen por color
    r_img, g_img, b_img = (image[:,:,i] for i in range(2,-1,-1))

    # Definimos la combinacion rgb a trabajar
    gb_img:np.ndarray = cv2.absdiff(g_img, b_img)

    # Binarizacion inicial de la imagen
    bin_img = np.uint8(gb_img>=80)*255

    # Transformacion Morfologica
    closing_streel = np.ones((50,50), np.uint8)
    dilation_streel = np.ones((10,10), np.uint8)

    closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, closing_streel)
    dilation = cv2.dilate(closing, dilation_streel, 1)

    # Para encontrar cajas
    # Encontrar contornos
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del hierarchy

    # Calcular rectangulo mas grande
    boxes = np.array([cv2.boundingRect(contours[i]) for i in range(len(contours))])
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    max_box = max(boxes, key=area)

    # 3. Mascara
    get_masked_image = lambda img, box: img[box[1] : box[1]+box[3], box[0] : box[0]+box[2], :]
    mask = get_masked_image(image, max_box)

    # 4. Guardar imagen
    cv2.imwrite(result_image_path.__str__(), mask)

    return result_image_path, mask
from os import mkdir, remove
from pathlib import Path
from functools import reduce

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import cv2

def segmentation(image_path:Path, image:np.ndarray, results_path: Path = Path().parent / 'results' / 'segmentation'):
    image_name = image_path.name
    result_image_path = results_path / image_name.replace('.jpg', '')
    if not result_image_path.exists(): mkdir(result_image_path)

    # Se generan las cajas para identificar las zonas de interes
    contours, hierarchy = cv2.findContours(image/255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del hierarchy

    # Calcular rectangulo mas grande
    threshold:np.ndarray
    L, A = threshold.shape
    boxes = np.array([cv2.boundingRect(contours[i]) for i in range(len(contours))])
    
    # Filtro de las cajas, que deben cumplir la condicion especifica
    valid_box_condition = lambda box: box[3]>0.4*L and box[3]<=0.7*L and box[2] > 0.08*A and box[2] <= 0.17*A
    valid_boxes = np.array(list(filter(valid_box_condition, boxes)))
    
    # Se eliminan duplicados haciendo un set y selecciono los 6 primeros
    valid_boxes = np.array(list({tuple(box) for box in valid_boxes}))
    valid_boxes = valid_boxes[:6]

    # Aplicar las mascaras a la imagen
    get_masked_image = lambda img, box: img[box[1] : box[1]+box[3], box[0] : box[0]+box[2]]
    masked_images_list = [get_masked_image(image, box) for box in valid_boxes]

    # Guardo las imagenes en una carpeta con el nombre del archivo
    for num, img in enumerate(masked_images_list):
        cv2.imwrite(str(result_image_path/image_name.replace('.jpg', f'_{num+1}.png')),img*255)
        plt.subplot(1, len(masked_images_list), num+1), plt.imshow(img)
    plt.show()

    return result_image_path, masked_images_list

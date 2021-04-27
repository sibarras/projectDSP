from os import mkdir, remove
from pathlib import Path
from functools import reduce

from matplotlib import pyplot as plt
import numpy as np
from numpy.lib import imag
import scipy as sp
import cv2

def segmentation(image_path:Path, image_tresh:np.ndarray, results_path: Path = Path().parent / 'results' / 'segmentation'):
    image_name = image_path.name
    result_image_path = results_path / image_name.replace('.jpg', '')
    if not result_image_path.exists(): mkdir(result_image_path)

    # Se generan las cajas para identificar las zonas de interes
    plt.subplot(1,2,1), plt.imshow(image_tresh), plt.title('Initial')
    image_to_tresh = cv2.dilate(image_tresh[:],None,iterations=2)
    
    plt.subplot(1,2,2), plt.imshow(image_to_tresh), plt.title('Dilated'), plt.show()

    threshold = cv2.convertScaleAbs(image_to_tresh/255)

    plt.plot(), plt.imshow(image_to_tresh), plt.title('With Threshold'), plt.show()

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # im = cv2.drawContours(image_tresh.copy(), contours, -1, (0, 255, 0), 0.5)
    # plt.plot(), plt.imshow(im), plt.title('Contours'), plt.show()

    # Calcular rectangulo mas grande
    threshold:np.ndarray
    L, A = threshold.shape
    boxes = np.array([cv2.boundingRect(contours[i]) for i in range(len(contours))])
    boxes = list(filter(lambda box: len(box)==4, boxes))

    # Filtro de las cajas, que deben cumplir la condicion especifica
    valid_box_condition = lambda box: box[3]>0.4*L and box[3]<=0.7*L and box[2] > 0.08*A and box[2] <= 0.17*A
    valid_boxes = np.array(list(filter(valid_box_condition, boxes)))
    
    # Se eliminan duplicados haciendo un set y selecciono los 6 primeros
    valid_boxes = np.array(list({tuple(box) for box in valid_boxes}))
    valid_boxes:list = list(valid_boxes)

    # Ordenar las cajas por su posicion en la imagen
    x_pos = lambda box: box[0]
    valid_boxes.sort(key=x_pos)
    print(valid_boxes)

    # im = cv2.drawContours(image_tresh.copy(), valid_boxes, -1, (0, 255, 0), 0.5)
    # plt.plot(), plt.imshow(im), plt.title('Contours'), plt.show()

    # Aplicar las mascaras a la imagen
    get_masked_image = lambda img, box: img[box[1] : box[1]+box[3], box[0] : box[0]+box[2]]
    masked_images_list = [get_masked_image(image_tresh, box) for box in valid_boxes]
    

    # Guardo las imagenes en una carpeta con el nombre del archivo
    for num, img in enumerate(masked_images_list):
        img = cv2.convertScaleAbs(img)
        cv2.imwrite(str(result_image_path/f'{image_name}_{num+1}.png'),img)
        plt.subplot(1, len(masked_images_list), num+1), plt.imshow(img)
    plt.suptitle("Segmentation"), plt.show()

    return result_image_path, masked_images_list

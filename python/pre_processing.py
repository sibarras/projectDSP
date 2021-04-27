from pathlib import Path
from os import remove
import numpy as np
import cv2

from matplotlib import pyplot as plt

def pre_process_image(image_path:Path, results_path:Path = Path().parent / 'results' / 'pre_processing'):
    image_name = image_path.name
    result_image_path = results_path/image_name

    # Leemos la imagen
    image:np.ndarray = cv2.imread(image_path.__str__())
    # cv2.imshow('pre processing', image)
    # cv2.waitKey(1000)

    # Separamos la imagen por color
    r_img, g_img, b_img = (image[:,:,i] for i in range(2,-1,-1))

    # Definimos la combinacion rgb a trabajar
    rg_img:np.ndarray = cv2.absdiff(r_img, g_img)
    gb_img:np.ndarray = cv2.absdiff(g_img, b_img)
    br_img:np.ndarray = cv2.absdiff(b_img, r_img)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('Gray Image', gray)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print(gray)

    # Binarizacion inicial de la imagen
    bin_img = np.uint8(gray>=150)*255

    gray = cv2.blur(bin_img,(3,3))
    canny = cv2.Canny(gray,150,200)
    canny = cv2.dilate(canny,None,iterations=1)

    cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image,cnts,-1,(0,255,0),2)

    valid_boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        epsilon = 0.05*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        ratio = float(w)/h

        if len(approx)==4 and area>2000 and ratio>2:
            valid = cv2.boundingRect(c)
            valid_boxes.append(valid)

    area = lambda box : box[2]*box[3]
    valid_boxes.sort(key=area)
    box = valid_boxes[0]

    get_masked_image = lambda img, box: img[box[1] : box[1]+box[3], box[0] : box[0]+box[2], :]
    result_image = get_masked_image(image, box)

    # Guardar imagen
    cv2.imwrite(result_image_path.__str__(), result_image)

    return result_image_path, result_image

    cv2.imshow('pre processing', result_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

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
    boxes = np.array([cv2.boundingRect(contour) for contour in contours])
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    max_box = max(boxes, key=area)

    # 3. Mascara
    get_masked_image = lambda img, box: img[box[1] : box[1]+box[3], box[0] : box[0]+box[2], :]
    mask = get_masked_image(image, max_box)

    # 4. Guardar imagen
    cv2.imwrite(result_image_path.__str__(), mask)

    return result_image_path, mask
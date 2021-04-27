import numpy as np
from pathlib import Path
from os import makedirs
import pytesseract
import cv2

def classification(image_path: Path, images_list:np.ndarray, first_image, results_path: Path = Path().parent / 'results' / 'classification') -> int:
    image_name = image_path.name.replace('.jpg', '')
    results_image_path = results_path/image_name
    if not results_image_path.exists(): makedirs(results_image_path)

    first_image:np.ndarray = cv2.imread(first_image.__str__())
    i = 0
    cv2.imshow('PLACA',first_image)
    cv2.moveWindow('PLACA',780,10)

    for image in images_list:
        text = pytesseract.image_to_string(image, config='--psm 11')
        print('PLACA: ',text)

        
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(first_image,text,(i+20,20),1,2.2,(0,255,0),3)
            
        cv2.imshow(f'Image_{text}',image)
        cv2.moveWindow(f'Image_{text}',45*(1+i),10*(1+i))
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    return text


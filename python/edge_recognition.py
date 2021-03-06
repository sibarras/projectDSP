from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from os import remove, makedirs
import cv2

def edge_recognition(image_path:Path, image:np.ndarray, results_path: Path = Path().parent.absolute() / 'results' / 'edge_recognition'):
    image_name = image_path.name
    result_image_path = results_path / image_name.replace('.jpeg', '')
    if not result_image_path.exists(): makedirs(result_image_path)
    
    # Crear un negativo con los colores maximos de la imagen
    max_color_neg_img = np.array([[1-max(rgb) for rgb in img_row] for img_row in image/255])

    cv2.imshow('Negative Image', max_color_neg_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Guardar como bmp para binarizar la imagen
    bmp_path = result_image_path / image_name.replace('.jpeg', '.bmp')
    cv2.imwrite(str(bmp_path), max_color_neg_img*255)
    bmp_image = cv2.imread(str(bmp_path))
    remove(bmp_path)

    cv2.imshow('Binarized Image', bmp_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Aplicacion de laplaciano para extraccion de bordes / No esta siendo utilizado
    bw2:np.ndarray = cv2.Canny(bmp_image, 100, 200)

   
    # plt.plot(), plt.imshow(bw2), plt.title('Canny'), plt.show()

    # Obtener Bordes de la imagen
    bmp_grayscale = sum(bmp_image[:,:,i] for i in range(3))/3 
    
    plt.plot(), plt.imshow(bmp_grayscale), plt.title('Grayscale Img'), plt.show()
    
    spec_img = get_contours(bmp_grayscale)

    plt.subplot(3,2,(5, 6)), plt.imshow(bw2), plt.title('Image Filtered')
    plt.show()

    # Binarizamos para tener el contorno de la imagen
    # _, threshold = cv2.threshold(bw2, 0, 255, 0)
    _, threshold = cv2.threshold(bw2, 50, 255, 0)
    img:np.ndarray = threshold

    cv2.imshow('Final Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Guardamos la imagen
    cv2.imwrite(str(result_image_path/image_name.replace('.jpeg', '.png')),img)

    return result_image_path, img


def get_contours(grayscale_img:np.ndarray) -> np.ndarray:

    # Obtenemos la transformada de fourier
    frec_domain_image = np.fft.fft2(grayscale_img)
    frec_domain_image_1d = np.fft.fft(grayscale_img.flatten())

    # Movemos el componente de frecuencia cero al centro del espectro
    f_shift_image = np.fft.fftshift(frec_domain_image)
    f_shift_image_1d = np.fft.fftshift(frec_domain_image_1d)

    # Buscamos el espectro de magnitud de la imagen
    spectrum_image:np.ndarray = get_spectrum(f_shift_image)
    spectrum_image_1d = get_spectrum(f_shift_image_1d)
    
    # Crear filtro bidimensional para la imagen
    frec_filter = get_2d_filter(grayscale_img)
    
    # Realizar la multiplicacion
    frec_shift_ans = f_shift_image*frec_filter

    # Encuentra el espectro para graficar
    spectrum_ans = get_spectrum(frec_shift_ans)

    # Regresar a la imagen
    frec_ans = np.fft.ifftshift(frec_shift_ans)
    complex_ans:np.ndarray = np.fft.ifft2(frec_ans)


    real_ans:np.ndarray = np.real(complex_ans)
    ans = real_ans*255/real_ans.max()

    # Mostrar el espectro
    plt.subplot(3,2,1), plt.imshow(grayscale_img), plt.title('Initial Image')
    plt.subplot(3,2,2), plt.imshow(spectrum_image), plt.title(' Image Spectre')
    plt.subplot(3,2,3), plt.imshow(frec_filter), plt.title('Filter')
    plt.subplot(3,2,4), plt.imshow(spectrum_ans), plt.title('Spectre Filtered')


    return ans

def get_2d_filter(img:np.ndarray) -> np.ndarray:
    xf, yf = img.shape
    D = lambda x, y : ((x-xf/2)**2 + (y-yf/2)**2)**0.5
    D0 = 10

    highpass_butterford = lambda x,y: 1 - 1/(1+(D(x,y)/D0)**(2*50))
    filter_kernel_b = np.array([[highpass_butterford(x,y) for y in range(yf)] for x in range(xf)])
    
    highpass_gaussian = lambda x,y: 1-np.exp(-D(x,y)**2/(2*D0**2))
    filter_kernel_g = np.array([[highpass_gaussian(x,y) for y in range(yf)] for x in range(xf)])
    

    filter_kernel = filter_kernel_g * 1/filter_kernel_g.max()

    f_kernel = np.fft.fft2(filter_kernel)
    f_shift_kernel = np.fft.fft2(f_kernel)

    kernel_spectrum = get_spectrum(f_shift_kernel)


    return filter_kernel_g

def get_1d_filter(img:np.ndarray) -> np.ndarray:
    xf, yf = img.flatten().shape
    D = lambda x, y : ((x-xf/2)**2 + (y-yf/2)**2)**0.5
    D0 = 10
    high_bandstop_butterford = lambda x,y: 1 - 1/(1+(D(x,y)/D0)**(2*20))

    filter_kernel = np.array([[high_bandstop_butterford(x,y) for y in range(yf)] for x in range(xf)])

    f_kernel = np.fft.fft(filter_kernel)
    f_shift_kernel = np.fft.fft(f_kernel)

    kernel_spectrum = get_spectrum(f_shift_kernel)

    return f_shift_kernel

def get_spectrum(frec_img: np.ndarray) -> np.ndarray:
    return 20*np.log(np.abs(frec_img))
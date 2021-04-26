from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from os import remove, makedirs
import cv2

def edge_recognition(image_path:Path, image:np.ndarray, results_path: Path = Path().parent.absolute() / 'results' / 'edge_recognition'):
    image_name = image_path.name
    result_image_path = results_path / image_name.replace('.jpg', '')
    if not result_image_path.exists(): makedirs(result_image_path)
    
    # Crear un negativo con los colores maximos de la imagen
    max_color_neg_img = np.array([[1-max(rgb) for rgb in img_row] for img_row in image/255])

    # Guardar como bmp para binarizar la imagen
    bmp_path = result_image_path / image_name.replace('.jpg', '.bmp')
    cv2.imwrite(str(bmp_path), max_color_neg_img)
    bmp_image = cv2.imread(str(bmp_path))
    remove(bmp_path)

    # Aplicacion de laplaciano para extraccion de bordes
    bw2:np.ndarray = cv2.Canny(bmp_image, 0.01, 0.6)

    # Obtener Bordes de la imagen
    bmp_grayscale = sum(bmp_image[:,:,i] for i in range(3))/3 
    spec_img = get_contours(bmp_grayscale)
    img = bw2

    # Binarizamos para tener el contorno de la imagen
    ret, threshold = cv2.threshold(spec_img, 50, 255, 0)
    img:np.ndarray = threshold
    del ret

    # Guardamos la imagen
    cv2.imwrite(str(result_image_path/image_name.replace('.jpg', '.png')),img)

    # Prueba -- Borrar
    # plt.subplot(221), plt.imshow(image), plt.title('ER. Imagen Inicial')
    # plt.subplot(222), plt.imshow(max_color_neg_img), plt.title('Negativo Maximo de La imagen')
    # plt.subplot(223), plt.imshow(bmp_grayscale*255), plt.title('Imagen Binarizada')
    # plt.subplot(224), plt.imshow(img), plt.title('Imagen Utilizando Filtrado de Frec')
    # plt.show()

    print(img.shape, bw2.shape)

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
    
    # Mostrar el espectro
    # plt.subplot(2,2,1), plt.imshow(grayscale_img), plt.title('Imagen para Contornos')
    # plt.subplot(2,2,2), plt.imshow(grayscale_img), plt.title('Imagen para Contornos')
    # plt.subplot(2,2,3), plt.imshow(spectrum_image), plt.title('Espectro a Aplicar')
    # plt.subplot(2,2,4), plt.plot(spectrum_image_1d), plt.title('Espectro 1D a aplicar')
    # plt.show()

    # Crear filtro unidimensional para la imagen
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

    # # Mostrar el espectro
    # plt.subplot(2,2,3), plt.imshow(spectrum_ans), plt.title('Spectrum applied')
    # plt.subplot(2,2,4), plt.imshow(ans), plt.title('Final Image')
    # plt.show()

    print(ans.max())

    return ans

def get_2d_filter(img:np.ndarray) -> np.ndarray:
    xf, yf = img.shape
    D = lambda x, y : ((x-xf/2)**2 + (y-yf/2)**2)**0.5
    D0 = 40
    high_bandstop_butterford = lambda x,y: 1 - 1/(1+(D(x,y)/D0)**(2*20))

    filter_kernel = np.array([[high_bandstop_butterford(x,y) for y in range(yf)] for x in range(xf)])
    filter_kernel = filter_kernel * 1/filter_kernel.max()

    f_kernel = np.fft.fft2(filter_kernel)
    f_shift_kernel = np.fft.fft2(f_kernel)

    kernel_spectrum = get_spectrum(f_shift_kernel)

    # plt.subplot(121), plt.imshow(img), plt.title('Main Image')
    # plt.subplot(122), plt.imshow(kernel_spectrum), plt.title('2D Filter')
    # plt.show()

    return f_shift_kernel

def get_1d_filter(img:np.ndarray) -> np.ndarray:
    xf, yf = img.flatten().shape
    D = lambda x, y : ((x-xf/2)**2 + (y-yf/2)**2)**0.5
    D0 = 20
    high_bandstop_butterford = lambda x,y: 1 - 1/(1+(D(x,y)/D0)**(2*20))

    filter_kernel = np.array([[high_bandstop_butterford(x,y) for y in range(yf)] for x in range(xf)])

    f_kernel = np.fft.fft(filter_kernel)
    f_shift_kernel = np.fft.fft(f_kernel)

    kernel_spectrum = get_spectrum(f_shift_kernel)

    # plt.subplot(121), plt.imshow(img), plt.title('Main Image')
    # plt.subplot(122), plt.imshow(kernel_spectrum), plt.title('2D Filter')
    # plt.show()

    return f_shift_kernel

def get_spectrum(frec_img: np.ndarray) -> np.ndarray:
    return 20*np.log(np.abs(frec_img))
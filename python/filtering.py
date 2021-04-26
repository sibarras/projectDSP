from pathlib import Path
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def filtering(results_image_folder: Path, masked_images:list[np.ndarray], results_path:Path = Path().parent / 'results' / 'filtering') -> Path:

    for num, image, image_path in zip(range(len(masked_images)), masked_images, results_image_folder.glob('*.png')):
        img_name = image_path.name

        image = image*255/max([max(image[i]) for i in range(len(image))])

        # Obtenemos la transformada de fourier para la imagen
        frec_img = np.fft.fft2(image)
        frec_1d = np.fft.fft(image.flatten())
        frec_ysum_1d = np.fft.fft(image.sum(0).flatten())
        frec_xsum_1d = np.fft.fft(image.sum(1).flatten())

        # Movemos el componente de frecuencia cero al centro del espectro
        f_shift_image = np.fft.fftshift(frec_img)
        f_shift_1d = np.fft.fftshift(frec_1d)
        f_shift_y = np.fft.fftshift(frec_ysum_1d)
        f_shift_x = np.fft.fftshift(frec_xsum_1d)

        # Buscamos el espectro de magnitud de la imagen
        magnitude_spectrum_image = 20*np.log(np.abs(f_shift_image))
        # magnitude_spectrum_1d = 20*np.log(np.abs(frec_1d))
        magnitude_spectrum_ysum = 20*np.log(np.abs(f_shift_y))
        magnitude_spectrum_xsum = 20*np.log(np.abs(f_shift_x))
        
        # Mostrar la imagen en el espectro de magnitud
        plt_n = 3 # Plot per file
        # plt.subplot(plt_n, len(masked_images), plt_n*num+1), plt.imshow(image), plt.title(img_name)
        # # plt.subplot(plt_n, len(masked_images), plt_n*num+2), plt.imshow(magnitude_spectrum_image), plt.title('Mag.Spectrum')
        # # plt.subplot(plt_n, len(masked_images), (plt_n*num+2,plt_n*num+3)), plt.plot(magnitude_spectrum_1d)
        # plt.subplot(plt_n, len(masked_images), plt_n*num+2), plt.plot(magnitude_spectrum_ysum)
        # plt.subplot(plt_n, len(masked_images), plt_n*num+3), plt.plot(magnitude_spectrum_xsum)

    plt.show()
    
        
        

import numpy as np
from pathlib import Path
from os import makedirs



def classification(image_path: Path, image:np.ndarray, results_path: Path = Path().parent / 'results' / 'classification') -> int:
    image_name = image_path.name.replace('.jpg', '')
    results_image_path = results_path/image_name
    if not results_image_path.exists(): makedirs(results_image_path)

    result = 123
    return result


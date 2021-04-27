from .pre_processing import pre_process_image, Path
from .edge_recognition import edge_recognition
from .segmentation import segmentation
from .classification import classification

def run() -> None:
    main_path = Path().parent
    images_path = main_path/'images'

    for image_path in images_path.glob('*.jpeg'):
        print(image_path.name)
        processed_img_dir, processed_img = pre_process_image(image_path)
        edge_img_dir, edge_img = edge_recognition(processed_img_dir, processed_img)
        segmentated_img_folder, masked_images = segmentation(edge_img_dir, edge_img)
        result:int = classification(segmentated_img_folder, masked_images, image_path)
        print(result)

    return None

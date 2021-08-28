import sunglasses
import cv2
import glob

# path_to_images = "./Humans/"
# path_to_images = "./selfies/"
path_to_images = "./no_glasses/"
images = glob.glob(path_to_images + "/*")

for image_path in images:
    max_len = 640
    face_image = cv2.imread(image_path)
    scale_factor = max_len / max(face_image.shape)
    face_image = sunglasses.image_resize(face_image, width=int(face_image.shape[0] * scale_factor),
                                         height=int(face_image.shape[1] * scale_factor))
    shape_text, result_image = sunglasses.process_image(face_image)
    if result_image is None:
        continue

    cv2.imshow("face", result_image)
    cv2.waitKey(0)

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import glob
import os
import telebot
from random import randint
from math import acos, degrees

# path_to_glasses = "./glasses/"
# path_to_images = "./Humans/"
# path_to_images = "./selfies/"
# images = glob.glob(path_to_images + "/*")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

token = os.getenv("GLASSES_TOKEN")
bot = telebot.TeleBot(token)
user_states = {}


def exception_catcher(base_function):
    def new_function(*args,
                     **kwargs):  # This allows you to decorate functions without worrying about what arguments they take
        try:
            return base_function(*args, **kwargs)  # base_function is whatever function this decorator is applied to
        except Exception as e:
            err_msg = base_function.__name__ + ' => ' + str(e)
            print(err_msg)

    return new_function


def image_resize(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    if dim[0]*dim[1] < image.shape[0]*image.shape[1]:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def put_on_glasses(face_shape, face_img, glasses_img):
    # берём индексы ориентиров для левого и правого глаз, затем
    # вычисляем координаты каждого глаза
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    left_eye_pts = face_shape[lStart:lEnd]
    right_eye_pts = face_shape[rStart:rEnd]

    # вычисляем центр массы для каждого глаза
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")

    # вычисляем угол между центроидами глаз
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]

    if dx >= 0:
        return face_img

    angle = np.degrees(np.arctan2(dy, dx)) - 180

    # поворачиваем изображение очков на вычисленный угол, чтобы
    # поворот очков соответствовал наклону головы
    glasses_img_cpy = imutils.rotate_bound(glasses_img, angle)

    # очки не должны покрывать *всю* ширину лица, а в идеале
    # только глаза — здесь выполняем примерную оценку и указываем
    # 90% ширины лица в качестве ширины очков
    glasses_width = int((np.linalg.norm(face_shape[0] - face_shape[16])) * 1.1)
    glasses_img_cpy = image_resize(glasses_img_cpy, glasses_width)
    face_img_cpy = face_img.copy()
    rows, cols, channels = glasses_img_cpy.shape

    start_x = max(int((right_eye_center[0] + left_eye_center[0]) / 2 - cols / 2), 0)
    start_y = max(int((right_eye_center[1] + left_eye_center[1]) / 2 - rows / 2), 0)

    rows = min(face_img_cpy.shape[0] - start_y, rows)
    cols = min(face_img_cpy.shape[1] - start_x, cols)

    ret, mask = cv2.threshold(glasses_img_cpy[:, :, 3:], 31, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    face_img_cpy[start_y:start_y + rows,
                 start_x:start_x + cols, 0:3] = cv2.bitwise_and(face_img_cpy[start_y:start_y + rows,
                                                                start_x:start_x + cols, 0:3], (255, 255, 255),
                                                                mask=mask_inv)
    return np.uint8(face_img_cpy)


def process_image(image):
    if image is None:
        return

    img_tmp = image.copy()
    max_len = 640
    scale_factor = max_len / max(image.shape)
    img_tmp = imutils.resize(image, width=int(image.shape[0] * scale_factor),
                             height=int(image.shape[1] * scale_factor))

    min_box_size = int(min(img_tmp.shape[0], img_tmp.shape[1]) * 0.3)
    img_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_box_size, min_box_size))

    if faces is None or len(faces) == 0:
        return

    print("Found {0} Faces!".format(len(faces)))
    (x, y, w, h) = faces[0]
    found_rect = dlib.rectangle(x, y, x + w, y + h)

    shape = predictor(img_gray, found_rect)
    shape = face_utils.shape_to_np(shape)

    top_point = np.array([[x + int(w / 2), y]])
    top_point = np.resize(top_point, (1, 2))
    shape = np.append(shape, top_point, 0)

    # метрики
    face_height = np.linalg.norm(top_point - shape[8])
    face_width = np.linalg.norm(shape[0] - shape[16])
    chin_a = np.linalg.norm(shape[12] - shape[6])
    chin_b = np.linalg.norm(shape[4] - shape[6])
    chin_c = np.linalg.norm(shape[4] - shape[12])
    chin_angle = degrees(acos((-chin_a * chin_a + chin_b * chin_b + chin_c * chin_c) / (2.0 * chin_b * chin_c)))

    face_length_ratio = face_height / face_width  # пропорции
    # face_chin_ratio = jaw_length / np.linalg.norm(shape[1] - shape[15])  # подбородок
    face_forehead_ratio = np.linalg.norm(shape[0] - shape[16]) / np.linalg.norm(shape[3] - shape[13])  # скулы

    print(f'face_length_ratio   = {face_length_ratio}')
    print(f'chin_angle          = {chin_angle}')
    print(f'face_forehead_ratio = {face_forehead_ratio}')

    face_labels = {'square': 0, 'circle': 1, 'rectangle': 2, 'oval': 3, 'triangle': 4, 'heart': 5}
    face_metrics = [0, 0, 0, 0, 0, 0]

    face_length_ratio_thr = 1.1
    chin_angle_thr = 45
    face_forehead_ratio_thr = 1.2

    if face_length_ratio > face_length_ratio_thr:
        face_metrics[face_labels['rectangle']] += 1
        face_metrics[face_labels['oval']] += 1
        face_metrics[face_labels['triangle']] += 1
        face_metrics[face_labels['heart']] += 1
    else:
        face_metrics[face_labels['square']] += 1
        face_metrics[face_labels['circle']] += 1

    if chin_angle > chin_angle_thr:
        face_metrics[face_labels['oval']] += 1
        face_metrics[face_labels['heart']] += 1
        face_metrics[face_labels['circle']] += 1
    else:
        face_metrics[face_labels['rectangle']] += 1
        face_metrics[face_labels['triangle']] += 1
        face_metrics[face_labels['square']] += 1

    if face_forehead_ratio > face_forehead_ratio_thr:
        face_metrics[face_labels['heart']] += 1
        face_metrics[face_labels['triangle']] += 1
    else:
        face_metrics[face_labels['oval']] += 1
        face_metrics[face_labels['circle']] += 1
        face_metrics[face_labels['rectangle']] += 1
        face_metrics[face_labels['square']] += 1

    print(face_labels)
    print(face_metrics)
    face_label = np.argmax(face_metrics)
    shape_text = 'rectangle'
    for shape_text, label in face_labels.items():
        if label == face_label:
            print(shape_text)
            break

    glasses_paths = glob.glob(path_to_glasses + shape_text + "/*")
    if len(glasses_paths) == 0:
        return

    glasses_ctr = randint(0, len(glasses_paths) - 1)
    glasses_image = cv2.imread(glasses_paths[glasses_ctr], -1)
    img_with_glasses = put_on_glasses(shape, img_tmp, glasses_image)
    cv2.putText(img_with_glasses, shape_text, (int(img_with_glasses.shape[0] * 0.1), int(img_with_glasses.shape[1] * 0.1)),
                0, 1, (0, 0, 0))

    return img_with_glasses


@exception_catcher
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, f'Я твой личный помощник. Приятно познакомиться, {message.from_user.first_name}. '
                          f'Для ознакомления с функционалом выполни /help')


@exception_catcher
@bot.message_handler(commands=['help'])
def send_help(message):
    bot.send_chat_action(message.chat.id, 'typing')
    bot.reply_to(message, '/start - начать\n'
                          '/send - отправить селфи')


@exception_catcher
@bot.message_handler(commands=['send'])
def send_photo(message):
    user_states[message.from_user.id] = 1
    bot.register_next_step_handler(message, get_media_message)
    bot.reply_to(message, 'Сделай селфи и старайся смотреть ровно в камеру :)')


@exception_catcher
@bot.message_handler(content_types=['photo'])
def get_media_message(message):
    if message.photo is None or user_states.get(message.from_user.id) != 1:
        return

    user_states.pop(message.from_user.id, None)
    max_size_ctr = -1
    max_size = 0
    photo_ctr = 0
    for photo_size in message.photo:
        if photo_size.width * photo_size.height > max_size:
            max_size = photo_size.width * photo_size.height
            max_size_ctr = photo_ctr

        photo_ctr += 1

    if max_size_ctr < 0:
        return

    photo_size = message.photo[max_size_ctr]
    file_info = bot.get_file(photo_size.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    photo = cv2.imdecode(np.frombuffer(downloaded_file, dtype=np.uint8), 1)
    result_image = process_image(photo)
    if result_image is None:
        return

    result_image_str = cv2.imencode('.jpg', result_image)

    bot.send_photo(chat_id=message.chat.id,
                   photo=result_image_str[1].tobytes(),
                   caption='Взгляни, что я тебе подобрал!',
                   reply_to_message_id=message.id)


bot.polling(none_stop=True)

# for image_path in images:
    # result_image = process_image(cv2.imread(image_path))
    # if result_image is None:
    #         continue
    #
    # cv2.imshow("face", result_image)
    # cv2.waitKey(0)

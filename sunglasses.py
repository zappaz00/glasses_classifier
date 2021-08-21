import cv2
import dlib
from imutils import face_utils
import numpy as np
import glob
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing

path_to_images = "C:\\Users\\filin\\Downloads\\archive\\Humans\\"
#path_to_images = "C:\\Users\\filin\\Downloads\\selfies\\"
images = glob.glob(path_to_images + "/*.jpg")
metrics_num = 3
metrics_full = np.empty((0, metrics_num))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_labels = {'circle': 0, 'oval': 1, 'square': 2, 'rectangular': 3, 'heart': 4, 'trapezoid': 5}

for img in images:
    image = cv2.imread(img)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    print("Found {0} Faces!".format(len(faces)))

    if metrics_full.shape[0] > 100:
        break

    if len(faces) == 0:
        continue

    for face in faces:
        (x, y, w, h) = face
        found_rect = dlib.rectangle(x, y, x+w, y+h)
        # print(found_rect)

        img_tmp = image.copy()
        # вычисляем ключевые точки
        shape = predictor(img_gray, found_rect)
        shape = face_utils.shape_to_np(shape)

        top_point = np.array([[x+int(w/2), y]])
        top_point = np.resize(top_point, (1, 2))
        shape = np.append(shape, top_point, 0)

        point_ctr = 0
        for x, y in shape:
            # cv2.circle(img_tmp, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(img_tmp, f'{point_ctr}', (x, y), 0, 0.25, (0, 255, 0))
            point_ctr = point_ctr + 1

        # метрики
        face_height = np.linalg.norm(top_point - shape[8])
        face_width = np.linalg.norm(shape[0] - shape[16])
        face_square = w * h

        face_length = face_height / face_width
        face_chin = np.linalg.norm(shape[4] - shape[12]) / np.linalg.norm(shape[6] - shape[10])
        face_forehead = np.linalg.norm(shape[0] - shape[16]) / np.linalg.norm(shape[3] - shape[13])

        shape_text = ''
        if 0.75 * face_height > face_width:  # Длина Вашего лица на 1/3 (или больше) больше его ширины
            if face_chin > 1.3:  # Ваш подбородок выглядит округлым?
                shape_text = 'oval'
            else:  # Ваш подбородок выглядит угловатым?
                shape_text = 'rectangular'
        elif 0.85 * face_height > face_width:  # Длина Вашего лица лишь немного больше его ширины
            if face_chin > 1.7:  # Ваш подбородок выглядит заостренным?
                if face_forehead > 1.15:  # Ваш лоб - самая широкая часть лица?
                    shape_text = 'heart'
                else:
                    shape_text = 'trapezoid'  # ромб?
            elif face_chin > 1.3:  # Ваш подбородок выглядит округлым?
                if face_forehead > 0.85:  # Ваш лоб равен скулам по ширине?
                    shape_text = 'oval'
                else:  # Ваш лоб кажется уже скул?
                    shape_text = 'trapezoid'
            else:  # Ваш подбородок выглядит угловатым?
                if face_forehead > 0.85:  # Ваш лоб равен скулам по ширине?
                    shape_text = 'square'
                else:  # Ваш лоб кажется уже скул?
                    shape_text = 'trapezoid'
        else:  # Длина Вашего лица равна его ширине или чуть меньше
            if face_chin > 1.3:  # Ваш подбородок выглядит округлым?
                shape_text = 'circle'
            else:  # Ваш подбородок выглядит угловатым?
                shape_text = 'square'

        label = face_labels[shape_text]
        print(shape_text)

        # metrics = np.array([])
        # metrics = np.append(metrics, face_height / face_width)
        # metrics = np.append(metrics, np.linalg.norm(shape[0] - shape[16]) / np.linalg.norm(shape[3] - shape[13]) )
        # metrics = np.append(metrics, np.linalg.norm(shape[4] - shape[12]) / np.linalg.norm(shape[6] - shape[10]) )
        # print(metrics)
        # metrics_full = np.append(metrics_full, [metrics], 0)

    cv2.imshow("face", img_tmp)
    d = 1
    # if cv2.waitKey(0) & 0xFF == ord("q"):
    # sys.exit(0)

# metrics_full = sklearn.preprocessing.scale(metrics_full)
# fig, axs = plt.subplots(metrics_num, metrics_num)
# for i in range(metrics_num):
# for j in range(metrics_num):
# axs[i, j].set_title(f'Axis [{i}, {j}]')
# axs[i, j].scatter(metrics_full[:, i], metrics_full[:, j], s = 10, color = 'blue', alpha = 0.75)
# axs[i, j].hist2d(metrics_full[:, i], metrics_full[:, j], (150, 150), cmap=plt.cm.jet)

# plt.show()

# kmeans = KMeans(n_clusters=6, random_state=0).fit(metrics_full)
# print(kmeans.labels_)
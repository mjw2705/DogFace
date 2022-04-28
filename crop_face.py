import cv2, os
import numpy as np
import dlib


def cvt_lms_to_np(landmark):
    coords = np.zeros((6, 2), dtype=int)
    for i in range(6):
        x = landmark.part(i).x
        y = landmark.part(i).y
        coords[i] = (x, y)
    return coords

DataPath = './DogFaceNet_Dataset/after_4_bis/'

detector = dlib.cnn_face_detection_model_v1('./saved_model/dogHeadDetector.dat')
predictor = dlib.shape_predictor('./saved_model/landmarkDetector.dat')
output_path = './cropped_img'
save_img_size = (96, 96)
f = open('./cropFail_path.txt', 'a')


cnt_no_find = 0
cnt_crop_image = 0

if not os.path.exists(output_path):
    os.mkdir(output_path)

for root, dirs, files in os.walk(DataPath):
    file_id = root.split('/')[-1]
    file_id_dir = os.path.join(output_path, file_id)
    if not os.path.exists(file_id_dir):
        os.mkdir(file_id_dir)
    
    for file in files:
        img_path = os.path.join(root, file)
        save_file_path = os.path.join(file_id_dir, file)

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        margin = 10
        faces = detector(img_gray, upsample_num_times=1)

        if not faces:
            print(img_path)
            f.write(img_path + '\n')
            cnt_no_find += 1
        else:
            for i, face in enumerate(faces):
                x1, y1 = face.rect.left(), face.rect.top()
                x2, y2 = face.rect.right(), face.rect.bottom()
                
                shape = predictor(img, face.rect)
                lms = cvt_lms_to_np(shape)
                
                if lms[0][1] < y1:
                    y1 = lms[0][1]
                if lms[3][1] > y2:
                    y2 = lms[3][1]

                x1 = max(x1 - margin, 0)
                y1 = max(y1 - margin, 0)
                x2 = min(x2 + margin, img_w)
                y2 = min(y2 + margin, img_h)
                crop_img = img[y1:y2, x1:x2]
                crop_img = cv2.resize(crop_img, save_img_size)

                cv2.imwrite(save_file_path, crop_img)

            cnt_crop_image += 1

print(f"crop successful: {cnt_crop_image}")
print(f"fail to crop: {cnt_no_find}")
f.close()
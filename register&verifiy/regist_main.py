import cv2
import os
import dlib
import numpy as np
import time
import torch
from itertools import combinations
from PIL import Image
import torchvision.transforms as transforms

class register_face:
    def __init__(self, video, user_id):
        self.videoPath = video
        self.user_id = user_id
        self.detector = dlib.cnn_face_detection_model_v1('./save_model/dogHeadDetector.dat')
        self.predictor = dlib.shape_predictor('./save_model/landmarkDetector.dat')
        self.sec = 5
        self.savePath = './frame_save' + f'/{self.user_id}'
        self.face_size = (96, 96)

        pth_file = './save_model/483_0.062.pth'
        self.use_cuda = torch.cuda.is_available()
        device = "cuda" if self.use_cuda else "cpu"
        self.model = torch.load(pth_file).to(device)
        self.model.eval()

    def _frame_num_error(self, fps, length):
        if fps * self.sec <= length:
            return False
        return True

    def _check_dir(self):
        try:
            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)
        except OSError:
            print(f'Error: creating directory {self.savePath}')
    
    def _cvt_lms_to_np(self, landmark):
        coords = np.zeros((6, 2), dtype=int)
        for i in range(6):
            x = landmark.part(i).x
            y = landmark.part(i).y
            coords[i] = (x, y)
        return coords

    def _get_dog_face(self, img, faces, margin):
        img_h, img_w = img.shape[:2]
        for i, face in enumerate(faces):
            x1, y1 = face.rect.left(), face.rect.top()
            x2, y2 = face.rect.right(), face.rect.bottom()
            
            shape = self.predictor(img, face.rect)
            lms = self._cvt_lms_to_np(shape)
            
            if lms[0][1] < y1:
                y1 = lms[0][1]
            if lms[3][1] > y2:
                y2 = lms[3][1]

            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, img_w)
            y2 = min(y2 + margin, img_h)
            crop_img = img[y1:y2, x1:x2]

        return crop_img

    def _save_img(self, crop_img, dog_cnt):
        crop_img = cv2.resize(crop_img, self.face_size)
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{self.savePath}/frame{dog_cnt}.png', crop_img)

    def process(self):
        cap = cv2.VideoCapture(self.videoPath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        dog_cnt = 0
        margin = 10

        if self._frame_num_error(fps, length):
            print("5초 이상의 등록 영상이 필요합니다!")
            return False
        else:
            self._check_dir()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 임시 개 이미지
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not ret:
                    pass
                else:
                    dog_cnt += 1
                    self._save_img(img, dog_cnt)

                # # 개 얼굴 검출
                # faces = self.detector(img, upsample_num_times=1)
                # if not faces:
                #     pass
                # else:
                #     dog_cnt += 1
                #     crop_img = self.get_dog_face(img, faces, margin)
                #     self.save_img(crop_img, dog_cnt)
        
        if int(length * 0.6) <= dog_cnt:
            img_lists = self._img_combinations(self.savePath)
            avg_thres = self._calc_dist(img_lists)
            print("이미지 등록 완료")
            return self.savePath, avg_thres
        else:
            print("좀 더 정확한 개 얼굴이 필요합니다!")
            '''디렉토리 내에 저장된 이미지 삭제?
            self._delete_img()
            '''
            return False

    def _img_combinations(self, img_path):
        img_list = []
        for files in os.listdir(img_path):
            file_path = os.path.join(img_path, files)
            img_list.append(file_path)
        return list(combinations(img_list, 2))

    def _calc_dist(self, img_lists):
        total = 0
        for img1, img2 in img_lists[:2]:
            print(img1, img2)
            distance = matching(img1, img2, self.model, self.use_cuda)
            total += distance
        avg = total / len(img_lists)
        return avg

    def _delete_img(self):
        if os.path.exists(self.savePath):
            for file in os.scandir(self.savePath):
                os.remove(file.path)

    def matching(img1, img2, network):
        transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
        
        img1 = Image.open(img1)
        img2 = Image.open(img2)

        img1 = transform(img1)
        img2 = transform(img2)

        img1 = torch.unsqueeze(img1, 0)
        img2 = torch.unsqueeze(img2, 0)

        if self.use_cuda:
            img1 = img1.cuda()
            img2 = img2.cuda()
        output1, output2 = network(img1, img2)
        
        output_vec1 = np.array(output1.cpu().detach().numpy())
        output_vec2 = np.array(output2.cpu().detach().numpy())

        euclidean_distance = np.sqrt(np.sum(np.square(np.subtract(output_vec1, output_vec2))))

        return euclidean_distance

if __name__ == "__main__":
    video = './user_video/2_5s.mp4'
    user_id = 2
    start = time.time()
    print(start)
    register = register_face(video, user_id)
    result = register.process()
    print(result)
    print(time.time() - start)

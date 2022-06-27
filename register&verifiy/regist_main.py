import cv2
import os
import numpy as np
import time
import torch
from itertools import combinations
from PIL import Image
import torchvision.transforms as transforms
from util import *

class register_face:
    def __init__(self, video, user_id):
        self.videoPath = video
        self.user_id = user_id

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        pt_file = './save_model/best.pt'
        self.face_model = torch.load(pt_file, map_location=self.device)['model'].float().fuse().eval()
        self.input_img_size = (416, 416)

        self.sec = 5
        self.savePath = './frame_save' + f'/{self.user_id}'
        self.face_size = (100, 100)

        pth_file = './save_model/483_0.062.pth'
        self.siamese_model = torch.load(pth_file, map_location=self.device).eval()

    def _frame_num_error(self, fps, length):
        if fps * self.sec <= length:
            return False
        return True

    def _get_dog_face(self, frame, conf_thres, iou_thres):
        img = cv2.resize(frame, self.input_img_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.face_model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        return pred

    def _save_img(self, crop_img, dog_cnt):
        crop_img = cv2.resize(crop_img, self.face_size)
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
            check_dir(self.savePath)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                pred = self._get_dog_face(frame, conf_thres=0.1, iou_thres=0.6)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det)==1:
                        det[:, :4] = scale_coords(self.input_img_size, det[:, :4], frame.shape).round()

                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = xyxy
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            crop_img = frame[y1:y2, x1:x2].copy()

                            dog_cnt += 1
                            self._save_img(crop_img, dog_cnt)
                    else:
                        pass
        
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
            distance = self._matching(img1, img2, self.siamese_model)
            total += distance
        avg = total / len(img_lists)
        return avg

    def _delete_img(self):
        if os.path.exists(self.savePath):
            for file in os.scandir(self.savePath):
                os.remove(file.path)

    def _matching(self, img1, img2, network):
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
    register = register_face(video, user_id)
    result = register.process()
    print(result)
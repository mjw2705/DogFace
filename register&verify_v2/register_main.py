from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                            non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device
import os
import torch
from PIL import Image
import time
from itertools import combinations
from utils.util import face_detect_run, matching

def check_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            delete_img(path)
    except OSError:
        print(f'Error: creating directory {path}')

def delete_img(save_dir):
    if os.path.exists(save_dir):
        for file in os.scandir(save_dir):
            os.remove(file.path)

class Register:
    def __init__(self, user_id, video):
        self.video = video
        self.save_dir = './frame_save' + f'/{user_id}'
        check_dir(self.save_dir)

        cap = cv2.VideoCapture(video)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        device=''
        self.device = select_device(device)
        self.face_weight = './saved_weights/yolov5s_cls_best_400.pt'
        self.siamese_weight = './saved_weights/483_0.062.pth'

    def _frame_num_error(self, sec):
        if self.fps * sec <= self.length:
            return False
        return True

    def _detect_thres_error(self, thres, detect_cnt):
        if self.length * thres > detect_cnt:
            return True
        return False
    
    def process(self):
        '''5초 미만의 영상 등록 거부'''
        if self._frame_num_error(sec=5):
            print("5초 이상의 등록 영상이 필요합니다!")
            return False, False
        else:
            '''5초 이상의 영상인 경우 얼굴 검출 후, save_dir에 저장
            검출된 얼굴 개수가 영상frame * 0.6이하일 경우, save_dir에서 삭제
            '''
            face_detect_run('register', self.save_dir, self.video, self.face_weight, self.device)
            avg_thres = self._regist_thres_run(self.siamese_weight)
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}, threshold {colorstr('bold', avg_thres)}")
            return self.save_dir, avg_thres

    def _regist_thres_run(self, weight):
        siamese_model = torch.load(weight, map_location=self.device).eval()
        dirs = os.listdir(self.save_dir)
        detect_cnt = len(dirs)

        if self._detect_thres_error(thres=0.6, detect_cnt=detect_cnt):
            print("좀 더 정확한 반려동물 얼굴이 필요합니다!")
            delete_img(self.save_dir)
            return None
        else:
            img_list = []
            for files in dirs:
                file_path = os.path.join(self.save_dir, files)
                img_list.append(file_path)
            img_lists = list(combinations(img_list, 2))

            total = 0
            for img1, img2 in img_lists:
                img1 = Image.open(img1)
                img2 = Image.open(img2)
                euclidean_distance = matching(img1, img2, siamese_model, self.device)
                total += euclidean_distance
            avg_thres = total / len(img_lists)
            
            return avg_thres

if __name__ == "__main__":
    # =============입력=============
    user_id = 1
    video = './user_video/1_3s.mp4'
    # ==============================

    start = time.time()
    register = Register(user_id, video)
    regist_result = register.process()
    print(regist_result)
    print("처리시간: ", time.time() - start)
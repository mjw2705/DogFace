# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
# from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
#                             non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
# from utils.plots import get_one_box
from utils.torch_utils import select_device
import os
from pathlib import Path
import torch
from PIL import Image
import time
from utils.util import face_detect_run, matching


'''이미지 들어오면 해당 id의 db에서 이미지들과 비교
평균 threshold보다 작으면 같은 개 
아니면 다른개'''
class Verification:
    def __init__(self, user_id, image, DB_thres):
        self.DB_dir = './frame_save' + f'/{user_id}'
        self.image = image
        self.DB_thres = DB_thres

        device=''
        self.device = select_device(device)
        self.face_weight = './saved_weights/yolov5s_cls_best_400.pt'
        self.siamese_weight = './saved_weights/483_0.062.pth'
    
    def process(self):
        crop_img = face_detect_run('verify', self.DB_dir, self.image, self.face_weight, self.device)
        # cv2.imwrite(f'{DB_dir}/verify.jpg', crop_img)

        if crop_img is None:
            print("좀 더 정확한 반려동물 얼굴이 필요합니다!")
            return None
        else:
            avg_thres = self._verify_thres_run(image=crop_img, weight=self.siamese_weight)
            if avg_thres <= self.DB_thres:
                print("같은 개!")
                return True
            else:
                print("다른 개")
                return False

    def _verify_thres_run(self, image, weight):
        siamese_model = torch.load(weight, map_location=self.device).eval()
        dirs = os.listdir(self.DB_dir)
        
        img_list = []
        for files in dirs:
            file_path = os.path.join(self.DB_dir, files)
            img_list.append(file_path)

        total = 0
        for db_img in img_list:
            img1 = Image.fromarray(image)
            img2 = Image.open(db_img)
            euclidean_distance = matching(img1, img2, siamese_model, self.device)
            total += euclidean_distance
        avg_thres = total / len(img_list)
        
        return avg_thres


if __name__ == "__main__":
    # =============입력===============
    user_id = 1
    image = './1_ex.jpg'
    # ================================

    # =============DB에서 받아오는 거=================
    DB_dir = './frame_save' + f'/{user_id}'
    DB_thres = 0.3018400997912746
    # ================================================

    start = time.time()
    try:
        if len(os.listdir(DB_dir)) < 1:
            raise Exception('DB에 이미지가 등록되지 않았습니다.')
        else:
            verify = Verification(user_id, image, DB_thres)
            verify_result = verify.process()
            print(verify_result)
    except Exception as e:
        print(f'Error: ', e)
    
    print("처리시간: ", time.time() - start)
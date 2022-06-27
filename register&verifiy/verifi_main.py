import cv2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from util import *

'''이미지들어오면 해당 id의 db에서 이미지들과 비교
평균 threshold보다 작으면 같은 개 
아니면 다른개?'''

class verification:
    def __init__(self, user_id, img):
        self.user_id = user_id
        self.img_path = img

        self.face_size = (100, 100)
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.input_img_size = (416, 416)

        pt_file = './save_model/best.pt'
        self.face_model = torch.load(pt_file, map_location=self.device)['model'].float().fuse().eval()
        pth_file = './save_model/483_0.062.pth'
        self.siamese_model = torch.load(pth_file, map_location=self.device).eval()

        # ======DB에서 받아오는 거=======
        self.DB_PATH = './frame_save' + f'/{self.user_id}'
        self.avg_thres = 1.91 
        # ==============================
        

    def process(self):
        face_img = self.get_dog_face(conf_thres=0.1, iou_thres=0.6)
        if face_img is None:
            print("정확한 얼굴이 필요합니다!")
            return None

        real_avg_thres = self._compare_img(face_img)
        if real_avg_thres <= self.avg_thres:
            print("같은 개!")
            return True
        else:
            print("다른 개")
            return False

    def get_dog_face(self, conf_thres, iou_thres):
        frame = cv2.imread(self.img_path)
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
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det)==1:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(self.input_img_size, det[:, :4], frame.shape).round()
                # Write results
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    crop_img = frame[y1:y2, x1:x2].copy()
        return crop_img

    def _get_images(self, path):
        img_lists = []
        images = os.listdir(path)
        for image in images[:5]:
            img_path = os.path.join(self.DB_PATH, image)
            img_lists.append(img_path)
        return img_lists

    def _compare_img(self, image):
        total = 0
        img_lists = self._get_images(self.DB_PATH)

        for img in img_lists:
            distance = self._matching(image, img, self.siamese_model)
            total += distance
        avg = total / len(img_lists)
        return avg

    def _matching(self, img1, img2, network):
        transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
        
        img1 = Image.fromarray(img1)
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
    user_id = 1
    img = './1_ex.png'
    verify = verification(user_id, img)
    print(verify.process())


import cv2
import os
import torch
import numpy as np
import dlib
from PIL import Image
import torchvision.transforms as transforms

'''이미지들어오면 해당 id의 db에서 이미지들과 비교
평균 threshold보다 작으면 같은 개 
아니면 다른개?'''

class verification:
    def __init__(self, user_id, img):
        self.user_id = user_id
        self.img_path = img
        self.face_size = (96, 96)
        self.detector = dlib.cnn_face_detection_model_v1('./save_model/dogHeadDetector.dat')
        self.predictor = dlib.shape_predictor('./save_model/landmarkDetector.dat')

        # ======DB에서 받아오는 거=======
        self.DB_PATH = './frame_save' + f'/{self.user_id}'
        self.avg_thres = 1.91 
        # ==============================
        pth_file = './save_model/483_0.062.pth'
        self.use_cuda = torch.cuda.is_available()
        device = "cuda" if self.use_cuda else "cpu"
        self.model = torch.load(pth_file).to(device)
        self.model.eval()
    
    def process(self):
        face_img = self.get_dog_face(margin=10)
        real_avg_thres = self._compare_img(face_img)
        if real_avg_thres <= self.avg_thres:
            print("같은 개!")
            return True
        else:
            print("다른 개")
            return False
    # 바꿀거
    def _cvt_lms_to_np(self, landmark):
        coords = np.zeros((6, 2), dtype=int)
        for i in range(6):
            x = landmark.part(i).x
            y = landmark.part(i).y
            coords[i] = (x, y)
        return coords

    # 바꿀거
    def get_dog_face(self, margin):
        frame = cv2.imread(self.img_path)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        faces = self.detector(img, upsample_num_times=1)
        if not faces:
            return None
        else:
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

            crop_img = cv2.resize(crop_img, self.face_size)
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
            distance = self._matching(image, img, self.model)
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
    user_id = 2
    img = './ex.png'
    verify = verification(user_id, img)
    print(verify.process())


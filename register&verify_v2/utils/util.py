from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, colorstr, cv2,
                            non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import save_one_box, get_one_box
from utils.torch_utils import smart_inference_mode
import torch
import torchvision.transforms as transforms
import numpy as np


@smart_inference_mode()
def face_detect_run(mode, save_dir, source, weight, device):
    crop_img = None
    conf_thres=0.5  # confidence threshold
    iou_thres=0.45
    save_crop=True
    box_thres = 0.6

    face_weights = weight
    model_type = face_weights.split('/')[-1].split('.')[-1]
    if model_type == 'pt':
        imgsz = (224, 224)
    else:
        imgsz = (256, 256)

    # Load model
    model = DetectMultiBackend(face_weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=False)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=5)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if det is not None and len(det)==1:

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop
                # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_crop:
                        if conf >= box_thres:
                            if mode == 'register':
                                save_one_box(img_size=(100, 100), xyxy=xyxy, im=imc, file=f'{save_dir}/frame{seen}.jpg', BGR=True)
                            else:
                                crop_img = get_one_box(img_size=(100, 100), xyxy=xyxy, im=imc, BGR=True)
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return crop_img


def matching(img1, img2, network, device):
    img_to_tensor = transforms.ToTensor()
    img1 = img_to_tensor(img1)
    img2 = img_to_tensor(img2)

    img1 = torch.unsqueeze(img1, 0).to(device)
    img2 = torch.unsqueeze(img2, 0).to(device)
    output1, output2 = network(img1, img2)
    
    output_vec1 = np.array(output1.cpu().detach().numpy())
    output_vec2 = np.array(output2.cpu().detach().numpy())

    euclidean_distance = np.sqrt(np.sum(np.square(np.subtract(output_vec1, output_vec2))))
    return euclidean_distance
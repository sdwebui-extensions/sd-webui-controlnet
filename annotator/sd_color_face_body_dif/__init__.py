import os
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


face_detector = None 

def unload_sd_color_face_body_dif():
    global face_detector

class FaceDetector:
    def __init__(self):
        self.model = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')

    def __call__(self, img):
        result = self.model(img)
        return result["boxes"]

def resize_image(input_image, resolution, nearest = False, crop264 = True):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    if crop264:
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
    else:
        H = int(H)
        W = int(W)
    if not nearest:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    else:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)
    return img

def apply_sd_color_face_body_dif(img, res=512, blur_ratio=16):
    global face_detector
    split_num       = 7
    expand_ratio    = 1
    face_detector   = FaceDetector()
    h, w            = img.shape[:2]

    img             = resize_image(img, res)
    now_h, now_w    = img.shape[:2]
    img_body_blur   = cv2.resize(img, (int(now_w//blur_ratio), int(now_h//blur_ratio)), interpolation=cv2.INTER_CUBIC)  
    img_body_blur   = cv2.resize(img_body_blur, (now_w, now_h), interpolation=cv2.INTER_NEAREST)

    boxes           = face_detector(img)
    for _, box in enumerate(boxes):
        left, top, right, bottom = box
        left            = int(np.clip(left - expand_ratio, 0, now_w))
        top             = int(np.clip(top - expand_ratio, 0, now_h))
        right           = int(np.clip(right + expand_ratio, 0, now_w))
        bottom          = int(np.clip(bottom + expand_ratio, 0, now_h))
        
        face            = img[top:bottom, left:right]
        face_h, face_w  = face.shape[:2]

        if face_h < face_w:
            w_to_split  = int(face_w / (face_h / split_num))
            face        = cv2.resize(face, (w_to_split, split_num), interpolation=cv2.INTER_CUBIC)  
        else:
            h_to_split  = int(face_h / (face_w / split_num))
            face        = cv2.resize(face, (split_num, h_to_split), interpolation=cv2.INTER_CUBIC)

        face            = cv2.resize(face, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
        img_body_blur[top:bottom, left:right] = face
    img             = img_body_blur
    img             = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img
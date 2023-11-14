import cv2
import numpy as np

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

def apply_sd_color(img, res=512, blur_ratio=16):
    h, w            = img.shape[:2]
    
    img             = resize_image(img, res)
    now_h, now_w    = img.shape[:2]
    img             = cv2.resize(img, (int(now_w//blur_ratio), int(now_h//blur_ratio)), interpolation=cv2.INTER_CUBIC)  
    img             = cv2.resize(img, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
    img             = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  
    return img
import cv2
import numpy as np

def cv2_resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def apply_sd_heatmap(img, res=512):
    h, w    = img.shape[:2]
    img     = cv2_resize_shortest_edge(img, res)

    img     = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, -1]
    img     = cv2.medianBlur(img, 33)
    img     = (img - np.min(img)) / (np.max(img) - np.min(img))
    img     = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)
    img     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img     = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  
    return img
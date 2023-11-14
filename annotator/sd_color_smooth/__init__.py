import cv2

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

def apply_sd_color_smooth(img, res=512):
    h, w = img.shape[:2]

    img = cv2_resize_shortest_edge(img, res)
    img = cv2.medianBlur(img, 33)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  
    return img
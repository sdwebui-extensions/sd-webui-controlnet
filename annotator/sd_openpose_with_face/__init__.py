# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .face import Face
from modules.paths import models_path


body_estimation = None 
hand_estimation = None 
face_estimation = None

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"
modeldir = os.path.join(models_path, "sd_openpose_with_face")
old_modeldir = os.path.dirname(os.path.realpath(__file__))


def unload_sd_openpose_with_face_model():
    global body_estimation, hand_estimation, face_estimation
    # if body_estimation is not None:
    #     body_estimation.model.cpu()
    #     hand_estimation.model.cpu()
    #     face_estimation.model.cpu()


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


def apply_sd_openpose_with_face(oriImg):
    global body_estimation, hand_estimation, face_estimation
    if body_estimation is None:
        body_modelpath = os.path.join(modeldir, "body_pose_model.pth")
        hand_modelpath = os.path.join(modeldir, "hand_pose_model.pth")
        face_modelpath = os.path.join(modeldir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            from annotator.util import load_model
            body_modelpath = load_model("body_pose_model.pth", body_model_path, modeldir)

        if not os.path.exists(hand_modelpath):
            from annotator.util import load_model
            hand_modelpath = load_model("hand_pose_model.pth", hand_model_path, modeldir)

        if not os.path.exists(face_modelpath):
            from annotator.util import load_model
            face_modelpath = load_model("facenet.pth", face_model_path, modeldir)

        body_estimation = Body(body_modelpath)
        hand_estimation = Hand(hand_modelpath)
        face_estimation = Face(face_modelpath)
    
    oriImg = oriImg[:, :, ::-1].copy()
    H, W, C = oriImg.shape
    with torch.no_grad():
        candidate, subset = body_estimation(oriImg)
        hands = []
        faces = []

        # Hand
        hands_list = util.handDetect(candidate, subset, oriImg)
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                hands.append(peaks.tolist())
        # Face
        faces_list = util.faceDetect(candidate, subset, oriImg)
        for x, y, w in faces_list:
            heatmaps = face_estimation(oriImg[y:y+w, x:x+w, :])
            peaks = face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                faces.append(peaks.tolist())

        if candidate.ndim == 2 and candidate.shape[1] == 4:
            candidate = candidate[:, :2]
            candidate[:, 0] /= float(W)
            candidate[:, 1] /= float(H)
        bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
        pose = dict(bodies=bodies, hands=hands, faces=faces)
        return draw_pose(pose, H, W)
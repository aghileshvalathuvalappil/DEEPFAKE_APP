import cv2
import torch
import numpy as np
from typing import Optional
import torchvision.transforms as T


_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112, 112)),             
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


def _detect_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return frame_bgr[y:y+h, x:x+w]


def _center_crop(frame_bgr, size: int = 224):
    h, w, _ = frame_bgr.shape
    min_edge = min(h, w)
    y = (h - min_edge) // 2
    x = (w - min_edge) // 2
    crop = frame_bgr[y:y+min_edge, x:x+min_edge]
    return crop


def _sample_frame_indices(n_total: int, n_target: int):
    if n_total <= 0:
        return []
    if n_total <= n_target:
        return list(range(n_total))
    return list(np.linspace(0, n_total - 1, n_target).astype(int))


def extract_face_frames(video_path: str, target_frames: int = 20) -> Optional[torch.Tensor]:
    """
    Extract up to `target_frames` face crops, resize to 112x112, normalize,
    and return tensor of shape [1, T, 3, 112, 112].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        count = 0
        while count < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            face = _detect_face(frame)
            if face is None:
                face = _center_crop(frame)
            frames.append(_transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
            count += 1
        cap.release()
        if len(frames) == 0:
            return None
        T_tensor = torch.stack(frames, dim=0).unsqueeze(0)  
        return T_tensor

    take = _sample_frame_indices(total, target_frames)

    frames = []
    for idx in take:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        face = _detect_face(frame)
        if face is None:
            face = _center_crop(frame)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        frames.append(_transform(face_rgb))

    cap.release()

    if len(frames) == 0:
        return None

    frames_tensor = torch.stack(frames, dim=0)            
    frames_tensor = frames_tensor.unsqueeze(0)            
    return frames_tensor

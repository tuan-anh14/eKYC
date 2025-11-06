import cv2
import numpy as np


def resize_keep_aspect(image_bgr, height=320):
    """Resize giữ tỷ lệ theo chiều cao mong muốn."""
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return image_bgr
    scale = float(height) / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image_bgr, (new_w, height), interpolation=cv2.INTER_LINEAR)


def to_gray(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def clahe_equalize(image_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def denoise(image_gray):
    return cv2.fastNlMeansDenoising(image_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def sharpen(image_gray):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image_gray, -1, kernel)


def adaptive_thresh(image_gray):
    return cv2.adaptiveThreshold(
        image_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


def preprocess_for_ocr(image_bgr):
    """Chuỗi tiền xử lý: gray -> CLAHE -> denoise -> sharpen -> adaptive threshold."""
    g = to_gray(image_bgr)
    g = clahe_equalize(g)
    g = denoise(g)
    g = sharpen(g)
    th = adaptive_thresh(g)
    return th



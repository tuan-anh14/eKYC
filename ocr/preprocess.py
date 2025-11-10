import cv2
import numpy as np
from typing import Tuple, Optional


def resize_keep_aspect(image_bgr, height=320):
    """Resize giữ tỷ lệ theo chiều cao mong muốn."""
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return image_bgr
    scale = float(height) / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image_bgr, (new_w, height), interpolation=cv2.INTER_LINEAR)


def deskew(image_gray) -> np.ndarray:
    """Sửa độ nghiêng của ảnh (deskewing) - kỹ thuật quan trọng cho OCR."""
    # Tìm các cạnh bằng Canny
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return image_gray
    
    # Tính góc nghiêng trung bình
    angles = []
    for rho, theta in lines[:20]:  # Chỉ lấy 20 dòng đầu
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image_gray
    
    # Lấy góc trung vị để tránh outlier
    median_angle = np.median(angles)
    
    # Chỉ xoay nếu góc nghiêng đáng kể (> 0.5 độ)
    if abs(median_angle) < 0.5:
        return image_gray
    
    # Xoay ảnh
    h, w = image_gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image_gray, M, (w, h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(image_gray, method='clahe'):
    """Tăng cường độ tương phản - có nhiều phương pháp."""
    if method == 'clahe':
        return clahe_equalize(image_gray)
    elif method == 'histogram':
        return cv2.equalizeHist(image_gray)
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image_gray, table)
    else:
        return image_gray


def remove_shadow(image_bgr) -> np.ndarray:
    """Loại bỏ bóng đổ - rất quan trọng cho ảnh CCCD chụp bằng điện thoại."""
    # Chuyển sang LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng Gaussian blur trên kênh L để tạo mask
    l_blur = cv2.GaussianBlur(l, (21, 21), 0)
    
    # Tính toán và chuẩn hóa
    l_norm = cv2.divide(l, l_blur, scale=255)
    
    # Merge lại
    lab_norm = cv2.merge([l_norm, a, b])
    result = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    return result


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

def preprocess_for_ocr_v2(image_bgr):
    """Tiền xử lý phiên bản 2: gray -> CLAHE -> denoise -> morphological operations."""
    g = to_gray(image_bgr)
    g = clahe_equalize(g)
    g = denoise(g)
    # Morphological operations để làm rõ chữ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    # Adaptive threshold với tham số khác
    th = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        8,
    )
    return th

def preprocess_for_ocr_v3(image_bgr):
    """Tiền xử lý phiên bản 3: gray -> CLAHE -> Otsu threshold."""
    g = to_gray(image_bgr)
    g = clahe_equalize(g)
    g = denoise(g)
    # Otsu threshold
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_simple_bw(image_bgr, threshold_value=127):
    """Tiền xử lý đơn giản giống Colab: gray -> threshold đơn giản -> đen trắng.
    Đây là phương pháp chuẩn được dùng trong nhiều notebook Colab hiệu quả.
    
    Args:
        image_bgr: Ảnh BGR đầu vào
        threshold_value: Giá trị ngưỡng (mặc định 127 như Colab)
    
    Returns:
        Ảnh đen trắng (binary) đã được threshold
    """
    # Bước 1: Chuyển sang grayscale (đen trắng)
    gray = to_gray(image_bgr)
    
    # Bước 2: Áp dụng threshold đơn giản (giống Colab: cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC))
    # THRESH_TRUNC: Cắt giá trị pixel > threshold về threshold (giữ nguyên phần tối)
    _, threshed = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TRUNC)
    
    return threshed


def preprocess_simple_binary(image_bgr, threshold_value=127):
    """Tiền xử lý đơn giản: gray -> binary threshold -> đen trắng hoàn toàn.
    Tạo ảnh nhị phân thuần túy (chỉ có đen và trắng).
    
    Args:
        image_bgr: Ảnh BGR đầu vào
        threshold_value: Giá trị ngưỡng
    
    Returns:
        Ảnh nhị phân (binary) - chỉ có đen (0) và trắng (255)
    """
    # Bước 1: Chuyển sang grayscale
    gray = to_gray(image_bgr)
    
    # Bước 2: Binary threshold (chuyển hoàn toàn sang đen trắng)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binary


def preprocess_for_ocr_v4(image_bgr):
    """Tiền xử lý phiên bản 4: shadow removal -> deskew -> CLAHE -> denoise -> adaptive threshold.
    Phiên bản này tốt hơn cho ảnh chụp bằng điện thoại có bóng đổ."""
    # Loại bỏ bóng đổ trước
    img_no_shadow = remove_shadow(image_bgr)
    g = to_gray(img_no_shadow)
    # Sửa độ nghiêng
    g = deskew(g)
    # Tăng cường độ tương phản
    g = clahe_equalize(g)
    # Loại bỏ nhiễu
    g = denoise(g)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        10,
    )
    return th


def preprocess_for_ocr_v5(image_bgr):
    """Tiền xử lý phiên bản 5: shadow removal -> resize -> CLAHE -> bilateral filter -> Otsu.
    Tối ưu cho ảnh chất lượng thấp."""
    # Loại bỏ bóng đổ
    img_no_shadow = remove_shadow(image_bgr)
    # Resize để tăng độ phân giải nếu ảnh quá nhỏ
    h, w = img_no_shadow.shape[:2]
    if h < 400:
        scale = 400 / h
        new_w = int(w * scale)
        img_no_shadow = cv2.resize(img_no_shadow, (new_w, 400), interpolation=cv2.INTER_CUBIC)
    
    g = to_gray(img_no_shadow)
    # Bilateral filter thay vì denoise thông thường (giữ edge tốt hơn)
    g = cv2.bilateralFilter(g, 9, 75, 75)
    # CLAHE
    g = clahe_equalize(g)
    # Otsu threshold
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_multi_scale(image_bgr, scales=[1.0, 1.5, 2.0]):
    """Tiền xử lý đa tỷ lệ - thử nhiều kích thước khác nhau và chọn kết quả tốt nhất.
    Kỹ thuật này thường được dùng trong các notebook Colab hiệu quả."""
    results = []
    h, w = image_bgr.shape[:2]
    
    for scale in scales:
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled = image_bgr.copy()
        
        # Áp dụng preprocessing
        processed = preprocess_for_ocr_v4(scaled)
        results.append((processed, scale))
    
    # Trả về tất cả các kết quả để OCR thử nhiều lần
    return results



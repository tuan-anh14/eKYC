# ocr/ocr_infer.py

from paddleocr import PaddleOCR
import cv2
import re
import unicodedata
from typing import Dict, Optional, Tuple, List
import os
from .preprocess import (
    preprocess_for_ocr, preprocess_for_ocr_v2, preprocess_for_ocr_v3,
    preprocess_for_ocr_v4, preprocess_for_ocr_v5, preprocess_multi_scale,
    preprocess_simple_bw, preprocess_simple_binary,
    resize_keep_aspect, remove_shadow, deskew
)

# Khởi tạo OCR model một lần (singleton pattern)
_ocr_instance = None
_vietocr_instance = None

def get_ocr_instance():
    """Lazy initialization của PaddleOCR để tránh load model nhiều lần"""
    global _ocr_instance
    if _ocr_instance is None:
        # Bật angle classifier (tham số phổ biến, tương thích nhiều phiên bản)
        _ocr_instance = PaddleOCR(lang='vi', use_angle_cls=True)
    return _ocr_instance

def get_vietocr_instance():
    """Lazy init VietOCR; trả về None nếu gói chưa được cài."""
    global _vietocr_instance
    # Mặc định cố gắng bật VietOCR nếu có; có thể tắt bằng USE_VIETOCR=0
    if os.environ.get('USE_VIETOCR', '1') == '0':
        return None
    if _vietocr_instance is not None:
        return _vietocr_instance
    try:
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        # Chọn model nhẹ nếu đặt biến môi trường LIGHT_VIETOCR=1
        model_name = 'vgg_transformer'
        if os.environ.get('LIGHT_VIETOCR', '0') == '1':
            model_name = 'vgg_seq2seq'
        # Cho phép override bằng VIETOCR_MODEL
        model_name = os.environ.get('VIETOCR_MODEL', model_name)
        config = Cfg.load_config_from_name(model_name)
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        _vietocr_instance = Predictor(config)
        return _vietocr_instance
    except Exception:
        return None

def _normalize(s: str) -> str:
    """Chuẩn hóa chuỗi tiếng Việt"""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()

def _dbg(*args):
    if os.environ.get('OCR_DEBUG', '0') == '1':
        try:
            print('[OCR_DEBUG]', *args)
        except Exception:
            pass

def _ensure_bgr(image):
    """Đảm bảo ảnh đầu vào là BGR 3 kênh."""
    if image is None:
        return image
    if len(image.shape) == 2:
        # gray -> BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # BGRA -> BGR
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _ocr_collect_lines(ocr: PaddleOCR, image) -> Tuple[List[str], List[float]]:
    """Chạy OCR trên ảnh và thu thập danh sách (text, confidence)."""
    lines: List[str] = []
    confs: List[float] = []
    try:
        result = ocr.ocr(_ensure_bgr(image))
    except Exception:
        # Fallback cho API mới
        result = ocr.predict(_ensure_bgr(image))

    # Hỗ trợ OCRResult từ PaddleX
    if isinstance(result, list) and len(result) == 1:
        page = result[0]
        # Kiểm tra nếu là OCRResult từ PaddleX
        if type(page).__name__ == 'OCRResult' or hasattr(page, 'rec_texts') or hasattr(page, 'boxes'):
            texts = getattr(page, 'rec_texts', None) or getattr(page, 'texts', None)
            scores = getattr(page, 'rec_scores', None) or getattr(page, 'scores', None)
            
            if texts is not None:
                import numpy as np
                if isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                if scores is not None and isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                
                n = len(texts)
                for i in range(n):
                    txt = str(texts[i]) if texts[i] is not None else ''
                    if txt:
                        lines.append(txt)
                        try:
                            conf = float(scores[i]) if scores is not None and scores[i] is not None else 1.0
                            confs.append(conf)
                        except Exception:
                            confs.append(1.0)
                return lines, confs

    # Hỗ trợ nhiều định dạng trả về của PaddleOCR
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        page = result[0]
        rec_texts = page.get('rec_texts') or []
        rec_scores = page.get('rec_scores') or []
        for txt, score in zip(rec_texts, rec_scores):
            if txt:
                lines.append(str(txt))
                try:
                    confs.append(float(score) if score is not None else 1.0)
                except Exception:
                    confs.append(1.0)
    elif isinstance(result, dict):
        for item in result.get('ocr_result', []):
            if isinstance(item, dict):
                txt = item.get('text')
                score = item.get('confidence', 1.0)
                if txt:
                    lines.append(str(txt))
                    try:
                        confs.append(float(score))
                    except Exception:
                        confs.append(1.0)
    elif isinstance(result, list):
        for page in result:
            if page is None:
                continue
            for line in page:
                if isinstance(line, list) and len(line) >= 2:
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        txt, conf = line[1][0], line[1][1]
                        if txt:
                            lines.append(txt)
                            try:
                                confs.append(float(conf))
                            except Exception:
                                confs.append(1.0)
                    elif isinstance(line[1], dict):
                        txt = line[1].get('text', '')
                        conf = line[1].get('confidence', 1.0)
                        if txt:
                            lines.append(txt)
                            try:
                                confs.append(float(conf))
                            except Exception:
                                confs.append(1.0)
                    elif isinstance(line[1], str):
                        lines.append(line[1])
                        confs.append(1.0)
                elif isinstance(line, dict):
                    txt = line.get('text', '')
                    conf = line.get('confidence', 1.0)
                    if txt:
                        lines.append(txt)
                        try:
                            confs.append(float(conf))
                        except Exception:
                            confs.append(1.0)
                elif isinstance(line, str):
                    lines.append(line)
                    confs.append(1.0)
    return lines, confs

def _parse_paddlex_ocrresult(page, image_bgr) -> List[Tuple[str, float, any, List[List[int]]]]:
    """Parse OCRResult từ PaddleX thành format chuẩn (text, conf, crop, box)."""
    items: List[Tuple[str, float, any, List[List[int]]]] = []
    try:
        # PaddleX OCRResult thường có các thuộc tính:
        #   page.boxes hoặc page.dt_polys -> list/ndarray shape [N, 4, 2] (4 điểm)
        #   page.rec_texts -> list[str]
        #   page.rec_scores -> list[float]
        boxes = getattr(page, 'boxes', None) or getattr(page, 'dt_polys', None)
        texts = getattr(page, 'rec_texts', None) or getattr(page, 'texts', None)
        scores = getattr(page, 'rec_scores', None) or getattr(page, 'scores', None)
        
        # Debug chi tiết
        _dbg('_parse_paddlex_ocrresult: boxes type:', type(boxes), 'is None:', boxes is None)
        _dbg('_parse_paddlex_ocrresult: texts type:', type(texts), 'is None:', texts is None)
        if boxes is not None:
            _dbg('_parse_paddlex_ocrresult: boxes length:', len(boxes) if hasattr(boxes, '__len__') else 'N/A')
        if texts is not None:
            _dbg('_parse_paddlex_ocrresult: texts length:', len(texts) if hasattr(texts, '__len__') else 'N/A')
        
        if boxes is None or texts is None:
            _dbg('_parse_paddlex_ocrresult: boxes or texts is None, trying alternative attributes')
            # Thử các thuộc tính khác
            if hasattr(page, '__dict__'):
                _dbg('_parse_paddlex_ocrresult: page.__dict__ keys:', list(page.__dict__.keys()))
            return items
        
        # Convert numpy array to list if needed
        import numpy as np
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if scores is not None and isinstance(scores, np.ndarray):
            scores = scores.tolist()
        
        n = min(len(boxes), len(texts), len(scores) if scores is not None else len(texts))
        H, W = image_bgr.shape[:2]
        
        _dbg(f'_parse_paddlex_ocrresult: parsing {n} detections')
        
        for i in range(n):
            try:
                box = boxes[i]
                # Chuẩn hóa box về [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                if isinstance(box, np.ndarray):
                    box = box.tolist()
                
                # Box có thể là [4, 2] hoặc [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                xs = []
                ys = []
                for p in box:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        xs.append(int(float(p[0])))
                        ys.append(int(float(p[1])))
                    elif isinstance(p, np.ndarray):
                        xs.append(int(float(p[0])))
                        ys.append(int(float(p[1])))
                
                if not xs or not ys:
                    continue
                
                x1, x2 = max(min(xs), 0), min(max(xs), W - 1)
                y1, y2 = max(min(ys), 0), min(max(ys), H - 1)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                crop = image_bgr[y1:y2, x1:x2]
                txt = str(texts[i]) if texts[i] is not None else ''
                conf = float(scores[i]) if scores is not None and scores[i] is not None else 1.0
                
                # Tạo box format chuẩn
                box_standard = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                items.append((txt, conf, crop, box_standard))
            except Exception as e:
                _dbg(f'_parse_paddlex_ocrresult: exception at index {i}:', str(e))
                continue
        
        _dbg(f'_parse_paddlex_ocrresult: parsed {len(items)} items')
    except Exception as e:
        _dbg('_parse_paddlex_ocrresult: exception:', str(e))
    return items

def _ocr_collect_with_crops(ocr: PaddleOCR, image_bgr) -> List[Tuple[str, float, any, List[List[int]]]]:
    """Thu thập (text, conf, crop_bgr, box) từ PaddleOCR."""
    items: List[Tuple[str, float, any, List[List[int]]]] = []
    
    # Debug: Kiểm tra image
    if image_bgr is None:
        _dbg('_ocr_collect_with_crops: image_bgr is None')
        return items
    _dbg('_ocr_collect_with_crops: image shape:', image_bgr.shape if hasattr(image_bgr, 'shape') else 'no shape')
    
    # Đảm bảo image là BGR
    image_bgr = _ensure_bgr(image_bgr)
    
    result = None
    try:
        # Bỏ cls=True vì không được hỗ trợ trong một số version
        _dbg('_ocr_collect_with_crops: calling ocr.ocr')
        result = ocr.ocr(image_bgr)
        _dbg('_ocr_collect_with_crops: ocr.ocr returned, type:', type(result))
    except Exception as e:
        _dbg('_ocr_collect_with_crops: ocr.ocr failed:', str(e))
        try:
            _dbg('_ocr_collect_with_crops: trying ocr.predict')
            result = ocr.predict(image_bgr)
            _dbg('_ocr_collect_with_crops: ocr.predict returned, type:', type(result))
        except Exception as e2:
            _dbg('_ocr_collect_with_crops: ocr.predict also failed:', str(e2))
            return items
    
    if result is None:
        _dbg('_ocr_collect_with_crops: result is None')
        return items
    
    _dbg('_ocr_collect_with_crops: result type:', type(result), 'is_list:', isinstance(result, list))
    
    if not isinstance(result, list):
        _dbg('_ocr_collect_with_crops: result is not a list, returning empty')
        return items
    
    _dbg('_ocr_collect_with_crops: result length:', len(result))
    
    # Xử lý OCRResult từ PaddleX
    if len(result) == 1:
        page = result[0]
        # Kiểm tra nếu là OCRResult từ PaddleX
        if type(page).__name__ == 'OCRResult' or hasattr(page, 'rec_texts') or hasattr(page, 'boxes'):
            _dbg('_ocr_collect_with_crops: detected OCRResult format, parsing...')
            # Debug: Kiểm tra OCRResult có gì
            _dbg('_ocr_collect_with_crops: OCRResult attributes:', dir(page))
            if hasattr(page, 'boxes'):
                _dbg('_ocr_collect_with_crops: page.boxes type:', type(getattr(page, 'boxes', None)), 'value:', getattr(page, 'boxes', None))
            if hasattr(page, 'rec_texts'):
                _dbg('_ocr_collect_with_crops: page.rec_texts type:', type(getattr(page, 'rec_texts', None)), 'value:', getattr(page, 'rec_texts', None))
            if hasattr(page, 'dt_polys'):
                _dbg('_ocr_collect_with_crops: page.dt_polys type:', type(getattr(page, 'dt_polys', None)), 'value:', getattr(page, 'dt_polys', None))
            
            parsed_items = _parse_paddlex_ocrresult(page, image_bgr)
            if parsed_items:
                items.extend(parsed_items)
                _dbg('_ocr_collect_with_crops: final items count (OCRResult):', len(items))
                return items
            else:
                # Nếu OCRResult không có dữ liệu, fallback về xử lý format chuẩn
                _dbg('_ocr_collect_with_crops: OCRResult parsing returned empty, falling back to standard format')
                # Kiểm tra xem page có phải là list không (format chuẩn)
                if isinstance(page, list):
                    _dbg('_ocr_collect_with_crops: page is list, processing as standard format')
                    # Xử lý như format chuẩn
                    for det_idx, det in enumerate(page):
                        try:
                            if not isinstance(det, list) or len(det) != 2:
                                continue
                            box, txt_conf = det
                            if not (isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 2):
                                continue
                            txt = str(txt_conf[0])
                            conf = float(txt_conf[1]) if txt_conf[1] is not None else 0.0
                            xs = [int(p[0]) for p in box]
                            ys = [int(p[1]) for p in box]
                            x1, x2 = max(min(xs), 0), min(max(xs), image_bgr.shape[1])
                            y1, y2 = max(min(ys), 0), min(max(ys), image_bgr.shape[0])
                            crop = image_bgr[y1:y2, x1:x2]
                            items.append((txt, conf, crop, box))
                        except Exception as e:
                            _dbg(f'_ocr_collect_with_crops: det {det_idx} exception:', str(e))
                            continue
                    if items:
                        _dbg('_ocr_collect_with_crops: collected items from fallback:', len(items))
                        return items
    
    # PaddleOCR format chuẩn: [[[box, (text, conf)], ...]]
    page_count = 0
    for page_idx, page in enumerate(result):
        page_count += 1
        _dbg(f'_ocr_collect_with_crops: page {page_idx}, type:', type(page))
        
        if isinstance(page, dict):
            # Format dict: {'ocr_result': [...]}
            ocr_result = page.get('ocr_result', [])
            _dbg(f'_ocr_collect_with_crops: page {page_idx} is dict, ocr_result length:', len(ocr_result))
            page = ocr_result
        
        if not isinstance(page, list):
            _dbg(f'_ocr_collect_with_crops: page {page_idx} is not a list, skipping')
            continue
        
        _dbg(f'_ocr_collect_with_crops: page {page_idx} length:', len(page))
        
        for det_idx, det in enumerate(page):
            try:
                if not isinstance(det, list):
                    continue
                
                if len(det) != 2:
                    continue
                
                box, txt_conf = det
                
                if not (isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 2):
                    continue
                
                txt = str(txt_conf[0])
                conf = float(txt_conf[1]) if txt_conf[1] is not None else 0.0
                
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1, x2 = max(min(xs), 0), min(max(xs), image_bgr.shape[1])
                y1, y2 = max(min(ys), 0), min(max(ys), image_bgr.shape[0])
                crop = image_bgr[y1:y2, x1:x2]
                items.append((txt, conf, crop, box))
            except Exception as e:
                _dbg(f'_ocr_collect_with_crops: det {det_idx} exception:', str(e))
                continue
    
    _dbg('_ocr_collect_with_crops: final items count:', len(items), 'page_count:', page_count)
    return items


def _count_vietnamese_chars(text: str) -> int:
    """Đếm số ký tự tiếng Việt có dấu trong text"""
    if not text:
        return 0
    count = 0
    for char in text:
        # Ký tự tiếng Việt có dấu: U+00C0 đến U+1EF9
        if ord(char) >= 0x00C0 and ord(char) <= 0x1EF9:
            count += 1
    return count

def _ocr_with_vietocr(image_bgr, use_vietocr: bool = True) -> Optional[str]:
    """Chạy VietOCR trên ảnh. Trả về text hoặc None nếu lỗi."""
    if not use_vietocr:
        return None
    viet = get_vietocr_instance()
    if viet is None:
        return None
    try:
        # VietOCR cần ảnh RGB
        if len(image_bgr.shape) == 3:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_bgr
        result = viet.predict(image_rgb)
        return str(result) if result else None
    except Exception as e:
        _dbg('VietOCR error:', str(e))
        return None

def _hybrid_ocr_roi(image_bgr, ocr: PaddleOCR, use_vietocr: bool = True) -> Tuple[str, float]:
    """Chạy hybrid OCR (PaddleOCR + VietOCR) trên ROI và chọn kết quả tốt nhất.
    Trả về: (best_text, confidence)
    """
    candidates: List[Tuple[str, float, int]] = []  # (text, conf, viet_count)
    
    # PaddleOCR với multipass (không dùng VietOCR trong multipass để tránh trùng)
    paddle_text, paddle_conf, _ = _ocr_multipass_text(ocr, image_bgr, fast_mode=False, use_vietocr=False)
    if paddle_text:
        viet_count = _count_vietnamese_chars(paddle_text)
        candidates.append((paddle_text, paddle_conf, viet_count))
    
    # VietOCR riêng biệt nếu có - chạy trực tiếp trên ROI
    if use_vietocr:
        viet_text = _ocr_with_vietocr(image_bgr, use_vietocr=True)
        if viet_text:
            viet_text = _normalize(viet_text)
            viet_count = _count_vietnamese_chars(viet_text)
            # VietOCR không có confidence, ước tính dựa trên độ dài và số ký tự tiếng Việt
            estimated_conf = min(0.95, 0.75 + (viet_count / max(1, len(viet_text))) * 0.2)
            candidates.append((viet_text, estimated_conf, viet_count))
            _dbg('VietOCR result:', viet_text[:100] + '...' if len(viet_text) > 100 else viet_text, 'viet_count:', viet_count)
        else:
            _dbg('VietOCR returned None or empty')
    else:
        _dbg('VietOCR disabled (use_vietocr=False)')
    
    if not candidates:
        _dbg('No OCR candidates found')
        return "", 0.0
    
    # Chọn kết quả tốt nhất: ưu tiên có nhiều ký tự tiếng Việt có dấu
    # Score = viet_count * 0.7 + confidence * 0.3 (ưu tiên cao hơn cho tiếng Việt)
    best_text, best_conf, best_viet_count = max(candidates, key=lambda x: (x[2] * 0.7 + x[1] * 0.3))
    _dbg('Hybrid OCR selected:', best_text[:100] + '...' if len(best_text) > 100 else best_text, 
         'viet_count:', best_viet_count, 'conf:', f'{best_conf:.2f}')
    return best_text, best_conf

def _ocr_multipass_text(ocr: PaddleOCR, image_bgr, fast_mode: bool = False, use_vietocr: bool = False) -> Tuple[str, float, List[str]]:
    """Chạy OCR nhiều pass (resize + preprocess) và chọn kết quả tốt nhất.
    Ưu tiên kết quả có nhiều ký tự tiếng Việt có dấu và confidence cao.
    Trả về: (full_text, avg_conf, lines)
    
    Args:
        fast_mode: Nếu True, chỉ chạy 1 pass (resize) để tăng tốc
        use_vietocr: Nếu True, thử dùng VietOCR để refine kết quả
    """
    candidates: List[Tuple[str, float, List[str], int]] = []  # (text, conf, lines, viet_count)
    
    # Pass 0: Preprocessing đơn giản giống Colab (gray -> threshold -> đen trắng)
    # Đây là phương pháp chuẩn được dùng trong nhiều notebook Colab hiệu quả
    try:
        img0_bw = preprocess_simple_bw(image_bgr, threshold_value=127)
        # Chuyển từ grayscale về BGR để OCR nhận (PaddleOCR cần BGR)
        if len(img0_bw.shape) == 2:
            img0_bw_bgr = cv2.cvtColor(img0_bw, cv2.COLOR_GRAY2BGR)
        else:
            img0_bw_bgr = img0_bw
        lines0, confs0 = _ocr_collect_lines(ocr, img0_bw_bgr)
        if lines0:
            text0 = " ".join(lines0)
            avg0 = sum(confs0) / max(1, len(confs0))
            viet_count0 = _count_vietnamese_chars(text0)
            candidates.append((text0, avg0, lines0, viet_count0))
            _dbg('Pass 0 (simple BW):', text0[:50] + '...' if len(text0) > 50 else text0, 'conf:', f'{avg0:.2f}')
    except Exception as e:
        _dbg('Pass 0 (simple BW) failed:', str(e))
    
    # Pass 1: resize mềm để tăng chiều cao chữ (pass này thường tốt nhất cho tiếng Việt)
    img1 = resize_keep_aspect(image_bgr, height=256)
    lines1, confs1 = _ocr_collect_lines(ocr, img1)
    if lines1:
        text1 = " ".join(lines1)
        avg1 = sum(confs1) / max(1, len(confs1))
        viet_count1 = _count_vietnamese_chars(text1)
        candidates.append((text1, avg1, lines1, viet_count1))
    
    # Nếu fast_mode, chỉ chạy 1 pass
    if fast_mode:
        if candidates:
            best_text, best_conf, best_lines, _ = candidates[0]
            # Nếu use_vietocr và kết quả có ít ký tự tiếng Việt, thử VietOCR
            if use_vietocr and _count_vietnamese_chars(best_text) < len(best_text) * 0.1:
                viet_text = _ocr_with_vietocr(image_bgr, use_vietocr=True)
                if viet_text and _count_vietnamese_chars(viet_text) > _count_vietnamese_chars(best_text):
                    return _normalize(viet_text), float(best_conf), [viet_text]
            return _normalize(best_text), float(best_conf), best_lines
        return "", 0.0, []
    
    # Pass 2: tiền xử lý mạnh (CLAHE + denoise + sharpen + adaptive threshold)
    img2 = preprocess_for_ocr(image_bgr)
    lines2, confs2 = _ocr_collect_lines(ocr, img2)
    if lines2:
        text2 = " ".join(lines2)
        avg2 = sum(confs2) / max(1, len(confs2))
        viet_count2 = _count_vietnamese_chars(text2)
        candidates.append((text2, avg2, lines2, viet_count2))
    
    # Pass 3: Preprocessing v4 (shadow removal + deskew) - tốt cho ảnh chụp bằng điện thoại
    try:
        img3 = preprocess_for_ocr_v4(image_bgr)
        # Chuyển từ grayscale về BGR để OCR nhận
        if len(img3.shape) == 2:
            img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
        lines3, confs3 = _ocr_collect_lines(ocr, img3)
        if lines3:
            text3 = " ".join(lines3)
            avg3 = sum(confs3) / max(1, len(confs3))
            viet_count3 = _count_vietnamese_chars(text3)
            candidates.append((text3, avg3, lines3, viet_count3))
    except Exception as e:
        _dbg('Preprocess v4 failed:', str(e))
    
    # Pass 4: Preprocessing v5 (bilateral filter) - tốt cho ảnh chất lượng thấp
    try:
        img4 = preprocess_for_ocr_v5(image_bgr)
        if len(img4.shape) == 2:
            img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
        lines4, confs4 = _ocr_collect_lines(ocr, img4)
        if lines4:
            text4 = " ".join(lines4)
            avg4 = sum(confs4) / max(1, len(confs4))
            viet_count4 = _count_vietnamese_chars(text4)
            candidates.append((text4, avg4, lines4, viet_count4))
    except Exception as e:
        _dbg('Preprocess v5 failed:', str(e))
    
    # Pass 5: thử VietOCR nếu được yêu cầu và có ít ký tự tiếng Việt
    if use_vietocr:
        viet_text = _ocr_with_vietocr(image_bgr, use_vietocr=True)
        if viet_text:
            viet_text = _normalize(viet_text)
            viet_count = _count_vietnamese_chars(viet_text)
            # Ước tính confidence cho VietOCR
            estimated_conf = min(0.95, 0.75 + (viet_count / max(1, len(viet_text))) * 0.2)
            candidates.append((viet_text, estimated_conf, [viet_text], viet_count))

    if not candidates:
        return "", 0.0, []
    
    # Chọn kết quả tốt nhất: ưu tiên có nhiều ký tự tiếng Việt có dấu, sau đó mới xét confidence
    # Score = viet_count * 0.7 + confidence * 0.3 (ưu tiên cao hơn cho ký tự tiếng Việt)
    best_text, best_conf, best_lines, _ = max(candidates, key=lambda x: (x[3] * 0.7 + x[1] * 0.3))
    return _normalize(best_text), float(best_conf), best_lines


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text


def _uppercase_ratio(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for ch in letters if ch.isupper())
    return upp / float(len(letters))


def _remove_anchors(text: str, anchors: List[str]) -> str:
    """Xóa nhãn (accent-insensitive) khỏi text."""
    if not text:
        return text
    ascii_text = _strip_accents(text).lower()
    for a in anchors:
        a_ascii = _strip_accents(a).lower()
        if a_ascii in ascii_text:
            idx = ascii_text.find(a_ascii)
            # cắt phần trước và cả nhãn
            cut = idx + len(a_ascii)
            text = text[cut:]
            ascii_text = ascii_text[cut:]
    # loại các dấu phân cách sau nhãn
    text = re.sub(r"^[\s:：\-\|/]+", "", text)
    return _normalize(text)


def _cut_before_next_label(text: str, stop_keywords: List[str]) -> str:
    """Cắt text trước nhãn tiếp theo (accent-insensitive)."""
    if not text:
        return text
    ascii_text = _strip_accents(text).lower()
    cut_pos = None
    for kw in stop_keywords:
        kw_ascii = _strip_accents(kw).lower()
        pos = ascii_text.find(kw_ascii)
        if pos != -1:
            cut_pos = pos if cut_pos is None else min(cut_pos, pos)
    if cut_pos is not None:
        return _normalize(text[:cut_pos])
    return _normalize(text)


def _slice_between(raw_text: str, start_keywords: List[str], stop_keywords: List[str]) -> str:
    """Từ full text, lấy đoạn sau một trong các start_keywords cho tới trước stop_keywords (accent-insensitive)."""
    if not raw_text:
        return ""
    ascii_full = _strip_accents(raw_text).lower()
    start_idx = None
    for kw in start_keywords:
        pos = ascii_full.find(_strip_accents(kw).lower())
        if pos != -1:
            start_idx = pos + len(_strip_accents(kw))
            break
    if start_idx is None:
        return ""
    sub = raw_text[start_idx:]
    # tìm stop gần nhất
    ascii_sub = _strip_accents(sub).lower()
    end_idx = None
    for kw in stop_keywords:
        pos = ascii_sub.find(_strip_accents(kw).lower())
        if pos != -1:
            end_idx = pos
            break
    if end_idx is not None:
        sub = sub[:end_idx]
    return _normalize(sub)

# Anchor sets for labels (Vietnamese, English, no-accent, common OCR variants)
ANCHORS = {
    "name": ["họ và tên", "ho va ten", "full name", "ho và tên", "full-name"],
    "dob": ["ngày sinh", "ngay sinh", "date of birth", "birth"],
    "gender": ["giới tính", "gioi tinh", "sex"],
    "nationality": ["quốc tịch", "quoc tich", "nationality"],
    "origin": ["quê quán", "que quan", "place of origin", "origin", "place of", "origin:"],
    "residence": ["nơi thường trú", "noi thuong tru", "place of residence", "residence", "residen", "place of"],
    "expiry": ["có giá trị đến", "co gia tri den", "valid until", "good thru", "valid thru", "date of expiry"],
    "issue": ["ngày cấp", "ngay cap", "issued on", "issue date", "issued by", "nơi cấp", "noi cap"],
}

def _norm_no_accents(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()

def _is_anchor(text: str, anchors: List[str]) -> bool:
    """Kiểm tra xem text có chứa anchor không. 
    Tìm anchor ở đầu text hoặc bất kỳ đâu trong text (vì có thể có format như "Họ và tên / Full name:")"""
    if not text:
        return False
    t = _norm_no_accents(text.lower())
    for a in anchors:
        a_norm = _norm_no_accents(a.lower())
        # Tìm anchor ở đầu text (có thể có khoảng trắng hoặc dấu phân cách trước)
        # Hoặc tìm anchor ở bất kỳ đâu trong text
        if a_norm in t:
            # Kiểm tra xem anchor có ở đầu text không (sau khi bỏ khoảng trắng và dấu phân cách)
            t_clean = re.sub(r'^[\s:：\-\|/]+', '', t)
            if t_clean.startswith(a_norm):
                return True
            # Hoặc anchor nằm ở bất kỳ đâu trong text (nhưng không phải là một phần của từ khác)
            # Tìm anchor với word boundary
            pattern = r'\b' + re.escape(a_norm) + r'\b'
            if re.search(pattern, t):
                return True
            # Fallback: tìm đơn giản nếu anchor đủ dài
            if len(a_norm) >= 5 and a_norm in t:
                return True
    return False

def _find_anchor_box(items, anchors: List[str], alt_tokens: List[str]) -> Optional[List[List[int]]]:
    """Tìm box nhãn: ưu tiên khớp full anchor; nếu không, khớp theo từ khoá rời (alt_tokens)."""
    candidates = []
    
    # Debug: Log một số items đầu tiên để xem text được OCR ra
    if len(items) > 0:
        sample_texts = [txt[:50] for txt, _, _, _ in items[:5]]
        _dbg('Sample OCR texts:', sample_texts)
    
    # Ưu tiên 1: Tìm khớp full anchor trong từng item
    for txt, conf, crop, box in items:
        if _is_anchor(txt, anchors):
            _dbg('Found anchor (priority 1):', txt[:50], 'for anchors:', anchors[:2])
            candidates.append((box, 1))  # Priority 1: full match
    
    # Ưu tiên 2: Tìm theo token rời trong từng item
    if not candidates:
        for txt, conf, crop, box in items:
            nt = _norm_no_accents(txt)
            matched_tokens = [tok for tok in alt_tokens if tok in nt]
            if matched_tokens:
                _dbg('Found anchor (priority 2):', txt[:50], 'matched tokens:', matched_tokens)
                candidates.append((box, 2))  # Priority 2: token match
    
    # Ưu tiên 3: Ghép các item thành dòng và tìm anchor trên dòng
    if not candidates:
        grouped_lines = _group_lines(items)
        _dbg('Grouped lines count:', len(grouped_lines))
        for line_text, rects in grouped_lines:
            line_norm = _norm_no_accents(line_text)
            if _is_anchor(line_text, anchors):
                _dbg('Found anchor (priority 3 - line):', line_text[:80], 'for anchors:', anchors[:2])
                # lấy union rect của các rects thuộc dòng
                xs1 = [r[0] for r in rects]; ys1 = [r[1] for r in rects]
                xs2 = [r[2] for r in rects]; ys2 = [r[3] for r in rects]
                ax1, ay1, ax2, ay2 = min(xs1), min(ys1), max(xs2), max(ys2)
                anchor_box = [[ax1, ay1], [ax2, ay1], [ax2, ay2], [ax1, ay2]]
                candidates.append((anchor_box, 3))  # Priority 3: line match
            else:
                # Thử tìm theo token trong dòng
                matched_tokens = [tok for tok in alt_tokens if tok in line_norm]
                if matched_tokens:
                    _dbg('Found anchor (priority 3 - line token):', line_text[:80], 'matched tokens:', matched_tokens)
                    xs1 = [r[0] for r in rects]; ys1 = [r[1] for r in rects]
                    xs2 = [r[2] for r in rects]; ys2 = [r[3] for r in rects]
                    ax1, ay1, ax2, ay2 = min(xs1), min(ys1), max(xs2), max(ys2)
                    anchor_box = [[ax1, ay1], [ax2, ay1], [ax2, ay2], [ax1, ay2]]
                    candidates.append((anchor_box, 3))
    
    if candidates:
        # Sắp xếp theo priority (thấp hơn = tốt hơn), sau đó theo x
        candidates.sort(key=lambda x: (x[1], _box_to_rect(x[0])[0]))
        _dbg('Anchor found, returning box')
        return candidates[0][0]
    
    _dbg('No anchor found after all attempts, items count:', len(items))
    return None

def _box_to_rect(box: List[List[int]]) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)

def _group_lines(items) -> List[Tuple[str, List[Tuple[int,int,int,int]]]]:
    """Nhóm các box theo dòng (dựa trên tâm y) và ghép text theo thứ tự x.
    Trả về [(line_text, [rects...])]."""
    lines: List[List[Tuple[int,int,int,int,str]]] = []
    for txt, conf, crop, box in items:
        x1, y1, x2, y2 = _box_to_rect(box)
        my = (y1 + y2) // 2
        placed = False
        for line in lines:
            # so sánh với tâm dòng đầu tiên
            lx1, ly1, lx2, ly2, _ = line[0]
            lmy = (ly1 + ly2) // 2
            if abs(my - lmy) <= max(8, (ly2 - ly1) // 2 + 10):
                line.append((x1, y1, x2, y2, txt))
                placed = True
                break
        if not placed:
            lines.append([(x1, y1, x2, y2, txt)])
    merged: List[Tuple[str, List[Tuple[int,int,int,int]]]] = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        text = " ".join([t[4] for t in line])
        rects = [(t[0], t[1], t[2], t[3]) for t in line]
        merged.append((_normalize(text), rects))
    return merged

def _is_date_pattern(text: str) -> bool:
    """Kiểm tra xem text có phải là pattern ngày tháng không (dd/mm/yyyy, dd-mm-yyyy, dd mm yyyy)"""
    if not text:
        return False
    # Pattern: 2 số, separator (/, -, space), 2 số, separator, 4 số
    patterns = [
        r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}\b',  # dd/mm/yyyy, dd-mm-yyyy, dd mm yyyy
        r'\b\d{1,2}\s+\d{1,2}\s+\d{4}\b',  # dd mm yyyy với nhiều space
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

def _read_right_of_anchor(image_bgr, ocr: PaddleOCR, items, anchors: List[str], max_gap_px: int = 12, right_pad: int = 40, max_lines: int = 1, stop_at_date: bool = False, stop_after_anchors: Optional[List[str]] = None) -> str:
    """Tìm box là anchor rồi đọc vùng bên phải trên cùng hàng và các dòng tiếp theo, OCR ROI theo multipass.
    
    Args:
        max_lines: Số dòng tối đa để đọc (1 = chỉ dòng hiện tại, >1 = đọc thêm các dòng dưới)
        stop_at_date: Nếu True, dừng khi gặp pattern ngày tháng (dùng cho residence để tránh lấy ngày)
    """
    H, W = image_bgr.shape[:2]
    # alt token heuristics cho name/origin/residence
    alt_map = {
        'name': ['full', 'name', 'ho', 'ten'],
        'origin': ['place', 'origin', 'que', 'quan'],
        'residence': ['place', 'residen', 'residence', 'noi', 'thuong', 'tru'],
    }
    key = 'name' if 'full name' in ' '.join(anchors) or 'ho va ten' in ' '.join(anchors) else (
        'origin' if 'origin' in ' '.join(anchors) or 'que quan' in ' '.join(anchors) else 'residence'
    )
    anchor_box = _find_anchor_box(items, anchors, alt_map.get(key, []))
    if anchor_box is None:
        _dbg('no_anchor_found_for', anchors[:2])
        # Nếu không tìm thấy anchor, vẫn cố gắng tạo ROI dựa trên vị trí ước tính
        # Dựa trên vị trí trung bình của các items để tạo ROI
        if not items:
            return ""
        # Tìm item đầu tiên có text chứa một phần anchor
        anchor_box = None
        for txt, conf, crop, box in items:
            nt = _norm_no_accents(txt)
            for anchor in anchors:
                anchor_norm = _norm_no_accents(anchor)
                # Tìm nếu có một phần của anchor trong text
                if len(anchor_norm) >= 3 and any(word in nt for word in anchor_norm.split() if len(word) >= 3):
                    anchor_box = box
                    break
            if anchor_box is not None:
                break
        
        # Nếu vẫn không tìm thấy, dùng box đầu tiên làm anchor ước tính
        if anchor_box is None and items:
            anchor_box = items[0][3]  # Lấy box của item đầu tiên
            _dbg('using_first_item_as_anchor_estimate')
        
        if anchor_box is None:
            return ""
    ax1, ay1, ax2, ay2 = _box_to_rect(anchor_box)
    mid_y = (ay1 + ay2) // 2
    line_height = ay2 - ay1
    
    # Thu thập các box trên cùng dòng và các dòng tiếp theo
    line_boxes: List[Tuple[int, int, int, int]] = []
    all_candidate_boxes = []
    
    # Tìm box chứa ngày tháng để dừng lại (nếu stop_at_date=True)
    date_box_y = None
    if stop_at_date:
        for txt, conf, crop, box in items:
            if _is_date_pattern(txt):
                x1, y1, x2, y2 = _box_to_rect(box)
                date_box_y = (y1 + y2) // 2
                _dbg('found_date_box', txt, date_box_y)
                break

    # Nếu có stop_after_anchors (ví dụ: "Có giá trị đến"), dừng trước nhãn đó
    stop_anchor_y = None
    stop_anchor_boxes_y = []  # Lưu tất cả các box chứa "Có giá trị đến"
    if stop_after_anchors:
        stop_box = _find_anchor_box(items, stop_after_anchors, [])
        if stop_box is not None:
            sx1, sy1, sx2, sy2 = _box_to_rect(stop_box)
            stop_anchor_y = sy1
        
        # Tìm TẤT CẢ các box có text chứa "Có giá trị đến" (không chỉ anchor box)
        for txt, conf, crop, box in items:
            txt_norm = _norm_no_accents(txt.lower())
            if any(exp in txt_norm for exp in ['co gia tri den', 'co gia tri', 'valid until', 'good thru', 'co gia tr', 'co gia tr dn', 'co gia tr den']):
                bx1, by1, bx2, by2 = _box_to_rect(box)
                stop_anchor_boxes_y.append((by1, by2))
                # Cập nhật stop_anchor_y nếu box này nằm cao hơn (gần anchor hơn)
                if stop_anchor_y is None or by1 < stop_anchor_y:
                    stop_anchor_y = by1
    
    for txt, conf, crop, box in items:
        x1, y1, x2, y2 = _box_to_rect(box)
        if x1 <= ax2:
            continue
        my = (y1 + y2) // 2
        
        # Kiểm tra xem box này có nằm trong vùng của "Có giá trị đến" không
        is_in_stop_zone = False
        for stop_y1, stop_y2 in stop_anchor_boxes_y:
            if y1 < stop_y2 and y2 > stop_y1:  # Có overlap với box "Có giá trị đến"
                is_in_stop_zone = True
                break
        if is_in_stop_zone:
            _dbg('skipping_box_in_stop_zone', txt)
            continue
        
        # Nếu có date_box_y hoặc stop_anchor_y và box này nằm sau mốc dừng, bỏ qua
        stop_y = None
        if date_box_y is not None:
            stop_y = date_box_y - line_height * 0.3
        if stop_anchor_y is not None:
            # Cho phép lấy đủ các dòng trước "Có giá trị đến" (giảm threshold)
            stop_y = min(stop_y, stop_anchor_y - 5) if stop_y is not None else (stop_anchor_y - 5)
        if stop_y is not None and my >= stop_y:
            _dbg('skipping_box_after_stop', txt)
            continue
        
        # Tính khoảng cách theo chiều dọc từ anchor
        vertical_dist = abs(my - mid_y)
        # Cải thiện threshold: dựa trên chiều cao dòng và khoảng cách thực tế
        # Sử dụng tỷ lệ động thay vì giá trị cố định
        line_threshold = max(line_height * 0.6, line_height // 2 + max_gap_px)
        
        # Nếu max_lines > 1, thu thập cả các dòng dưới
        if max_lines > 1:
            # Cho phép các box trên cùng dòng hoặc các dòng tiếp theo
            # Tính toán chính xác hơn dựa trên chiều cao dòng
            max_vertical_dist = line_height * max_lines * 1.2  # 1.2x để lấy đủ dòng nhưng không quá xa
            if vertical_dist <= max_vertical_dist:
                all_candidate_boxes.append((x1, y1, x2, y2, my, txt))
        else:
            # Chỉ lấy box trên cùng dòng - threshold chặt chẽ hơn
            if vertical_dist <= line_threshold:
                line_boxes.append((x1, y1, x2, y2))
    
    # Nếu có nhiều dòng, sắp xếp theo y và nhóm lại
    if max_lines > 1 and all_candidate_boxes:
        # Sắp xếp theo y
        all_candidate_boxes.sort(key=lambda b: b[4])
        # Nhóm thành các dòng
        lines_groups = []
        current_line = [all_candidate_boxes[0]]
        current_y = all_candidate_boxes[0][4]
        
        for box in all_candidate_boxes[1:]:
            # Kiểm tra nếu box này chứa ngày tháng, dừng lại
            if stop_at_date and _is_date_pattern(box[5]):
                _dbg('stopping_at_date_in_line', box[5])
                break
            # Cải thiện logic nhóm dòng: sử dụng threshold động dựa trên chiều cao dòng
            line_gap_threshold = line_height * 0.7  # 70% chiều cao dòng
            if abs(box[4] - current_y) <= line_gap_threshold:
                current_line.append(box)
            else:
                lines_groups.append(current_line)
                current_line = [box]
                current_y = box[4]
        if current_line:
            lines_groups.append(current_line)
        
        # Lấy tối đa max_lines dòng đầu tiên, nhưng ưu tiên lấy đủ các dòng trước khi dừng
        # Nếu có stop_anchor_y, lấy TẤT CẢ các dòng trước đó (không giới hạn bởi max_lines)
        lines_to_take = max_lines
        if stop_after_anchors and stop_anchor_y is not None:
            # Đếm số dòng nằm trước stop_anchor_y (lấy TẤT CẢ các dòng trước đó)
            lines_before_stop = 0
            for line_group in lines_groups:
                # Kiểm tra y của dòng đầu tiên trong group
                if line_group and line_group[0][4] < stop_anchor_y - 5:  # -5 để chắc chắn lấy đủ
                    lines_before_stop += 1
                else:
                    break
            # Lấy TẤT CẢ các dòng trước "Có giá trị đến", không giới hạn bởi max_lines
            if lines_before_stop > 0:
                lines_to_take = lines_before_stop  # Lấy tất cả các dòng trước đó
        
        for line_group in lines_groups[:lines_to_take]:
            for x1, y1, x2, y2, _, _ in line_group:
                line_boxes.append((x1, y1, x2, y2))
    
    if not line_boxes:
        # Fallback: tạo ROI từ anchor
        rx1 = min(ax2 + 2, W - 1)
        rx2 = min(W - 1, ax2 + W // 2 + right_pad)
        ry1 = max(ay1 - 8, 0)
        # Mở rộng xuống dưới nếu cần đọc nhiều dòng, nhưng dừng trước mốc dừng nếu có
        if max_lines > 1:
            stop_y2 = None
            if date_box_y is not None:
                stop_y2 = date_box_y - line_height // 2
            if stop_anchor_y is not None:
                stop_y2 = min(stop_y2, stop_anchor_y) if stop_y2 is not None else stop_anchor_y
            if stop_y2 is not None:
                ry2 = min(H, stop_y2)
            else:
                ry2 = min(H, ay2 + line_height * max_lines + 8)
        else:
            ry2 = min(ay2 + 8, H)
        roi = image_bgr[ry1:ry2, rx1:rx2]
    else:
        xs1 = [b[0] for b in line_boxes]
        ys1 = [b[1] for b in line_boxes]
        xs2 = [b[2] for b in line_boxes]
        ys2 = [b[3] for b in line_boxes]
        rx1 = min(xs1)
        ry1 = max(0, min(ys1) - 2)
        rx2 = min(W - 1, max(xs2) + right_pad)
        # Mở rộng xuống dưới nhưng dừng trước mốc dừng nếu có
        stop_y2 = None
        if date_box_y is not None:
            stop_y2 = date_box_y - line_height // 2
        if stop_anchor_y is not None:
            stop_y2 = min(stop_y2, stop_anchor_y) if stop_y2 is not None else stop_anchor_y
        if stop_y2 is not None:
            ry2 = min(H, stop_y2)
        else:
            ry2 = min(H, max(ys2) + 2)
        roi = image_bgr[ry1:ry2, rx1:rx2]
    
    # Sử dụng hybrid OCR cho các trường quan trọng (name, origin, residence)
    # Với các trường này, dùng hybrid OCR để nhận diện dấu tốt hơn
    # Luôn dùng hybrid OCR cho các trường có nhiều dòng hoặc khi cần nhận diện dấu tốt
    use_hybrid = max_lines >= 2  # Dùng hybrid cho các trường có nhiều dòng
    if use_hybrid:
        text, conf = _hybrid_ocr_roi(roi, ocr, use_vietocr=True)
    else:
        # Với ROI nhỏ, vẫn dùng multipass nhưng có thể thử VietOCR nếu cần
        use_fast = roi.shape[0] * roi.shape[1] < 50000
        text, conf, _ = _ocr_multipass_text(ocr, roi, fast_mode=use_fast, use_vietocr=False)
    _dbg('roi_text', text, 'max_lines:', max_lines, 'use_hybrid:', use_hybrid)
    return _normalize(text)


def extract_id_fields(image_bgr) -> Dict:
    """
    Input: ảnh BGR (cv2.imread hoặc numpy array)
    Output: dict {
        id_number, name, dob, gender, nationality, 
        place_of_origin, residence, issue_date, expiry_date
    }
    """
    ocr = get_ocr_instance()
    # Pass 1: full text phục vụ số/ngày (dùng fast_mode để tăng tốc)
    full, avg_conf, lines = _ocr_multipass_text(ocr, image_bgr, fast_mode=True)
    out = {}
    
    # CCCD 12 số
    m = re.search(r"\b(\d{12})\b", full)
    if m:
        out["id_number"] = m.group(1)
    
    # Ngày (dd/mm/yyyy hoặc dd-mm-yyyy)
    dates = re.findall(
        r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-]((?:19|20)\d{2})\b",
        full
    )
    
    def pack(d):
        return f"{d[0].zfill(2)}/{d[1].zfill(2)}/{d[2]}"
    
    if dates:
        out["dob"] = pack(dates[0])
        # Expiry date sẽ được extract bằng ROI ở phần dưới
    
    # Giới tính
    if re.search(r"\bNam\b", full, re.I):
        out["gender"] = "Nam"
    elif re.search(r"\bNữ\b|Nu|Nữ", full, re.I):
        out["gender"] = "Nữ"
    
    # Quốc tịch
    if re.search(r"Vi[eê]t\s*Nam", full, re.I):
        out["nationality"] = "Việt Nam"
    
    # Họ tên / Name: Tìm anchor và đọc trực tiếp ROI với multipass đầy đủ
    items = _ocr_collect_with_crops(ocr, image_bgr)
    _dbg('Total OCR items collected:', len(items))
    name_anchor_box = _find_anchor_box(items, ANCHORS["name"], ['full', 'name', 'ho', 'ten', 'họ', 'tên'])
    
    if name_anchor_box:
        H, W = image_bgr.shape[:2]
        ax1, ay1, ax2, ay2 = _box_to_rect(name_anchor_box)
        # Tìm anchor tiếp theo (dob hoặc gender) để biết điểm dừng
        dob_anchor_box = _find_anchor_box(items, ANCHORS["dob"], ['ngay', 'sinh', 'date', 'birth'])
        
        # Tạo ROI từ bên phải của "Họ và tên" đến trước "Ngày sinh"
        roi_x1 = min(ax2 + 2, W - 1)
        if dob_anchor_box:
            dx1, dy1, dx2, dy2 = _box_to_rect(dob_anchor_box)
            roi_x2 = min(W - 1, dx1 - 5)
            roi_y2 = min(H, dy1 + 10)
        else:
            roi_x2 = min(W - 1, ax2 + W // 2 + 40)
            roi_y2 = min(H, ay2 + 20)
        roi_y1 = max(0, ay1 - 8)
        
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            name_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            # Resize LỚN HƠN để nhận diện chữ in hoa tiếng Việt tốt hơn
            # Chữ in hoa cần độ phân giải cao hơn nhiều để giữ dấu (Ề, Ấ, etc.)
            h_roi, w_roi = name_roi_img.shape[:2]
            if h_roi < 150:  # Tăng từ 100 lên 150 để nhận diện chữ in hoa tốt hơn
                scale = 150 / h_roi
                new_w = int(w_roi * scale)
                name_roi_img = cv2.resize(name_roi_img, (new_w, 150), interpolation=cv2.INTER_CUBIC)
            
            # Preprocessing đặc biệt cho chữ in hoa: dùng preprocessing v5 (bilateral filter) để giữ edge tốt
            try:
                name_roi_processed = preprocess_for_ocr_v5(name_roi_img)
                if len(name_roi_processed.shape) == 2:
                    name_roi_processed = cv2.cvtColor(name_roi_processed, cv2.COLOR_GRAY2BGR)
                # Thử OCR với ảnh đã preprocessing
                lines_proc, confs_proc = _ocr_collect_lines(ocr, name_roi_processed)
                if lines_proc:
                    name_text_proc = " ".join(lines_proc)
                    # So sánh với kết quả từ hybrid OCR
                    name_text_hybrid, _ = _hybrid_ocr_roi(name_roi_img, ocr, use_vietocr=True)
                    # Chọn kết quả có nhiều ký tự tiếng Việt có dấu hơn
                    if _count_vietnamese_chars(name_text_proc) > _count_vietnamese_chars(name_text_hybrid):
                        name_text = name_text_proc
                    else:
                        name_text = name_text_hybrid
                else:
                    name_text, _ = _hybrid_ocr_roi(name_roi_img, ocr, use_vietocr=True)
            except Exception:
                name_text, _ = _hybrid_ocr_roi(name_roi_img, ocr, use_vietocr=True)
            
            # Debug: Kiểm tra text OCR đọc được (trước khi xử lý)
            if name_text:
                _dbg('Name OCR raw text (before cleaning):', name_text)
                _dbg('Name has uppercase Vietnamese diacritics:', any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 and c.isupper() for c in name_text))
                # Kiểm tra từng ký tự
                for char in name_text:
                    if char.isupper() and ord(char) >= 0x00C0 and ord(char) <= 0x1EF9:
                        _dbg(f'Found uppercase Vietnamese char: {char} (U+{ord(char):04X})')
            
            if name_text:
                # Loại bỏ nhãn MẠNH HƠN - loại bỏ cả "H và tên" ở đầu (không phân biệt hoa thường)
                # Loại bỏ từ đầu text
                name_text = re.sub(r"^(?i)(h\s*và\s*tên|ho\s*va\s*ten|họ\s*và\s*tên|full\s*name)[\s:：\-|/]*", "", name_text).strip()
                # Loại bỏ nhãn ở bất kỳ đâu trong text (bao gồm cả "H và tên" không dấu)
                name_text = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|h\s*va\s*ten|ho\s*va\s*ten|họ\s*và\s*tên|full\s*name|ful\s*nmne|ful\s*name)[\s:：\-|/]*", " ", name_text)
                # Loại bỏ các ký tự không phải chữ cái và khoảng trắng, nhưng giữ nguyên dấu tiếng Việt (bao gồm cả chữ in hoa có dấu)
                name_text = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name_text)
                # Chuẩn hóa khoảng trắng
                name_text = re.sub(r"\s+", " ", name_text).strip()
                # Lọc các từ hợp lệ - loại bỏ các từ ngắn có thể là label
                if name_text:
                    words = name_text.split()
                    valid_words = []
                    for word in words:
                        # Bỏ qua các từ có thể là label
                        word_lower = word.lower()
                        if word_lower in ['h', 'và', 'tên', 'ho', 'va', 'ten', 'full', 'name']:
                            continue
                        # Chấp nhận từ nếu:
                        # 1. Có nhiều chữ HOA (>50%) HOẶC
                        # 2. Có ký tự tiếng Việt có dấu (bao gồm cả chữ in hoa có dấu như Ề, Ấ)
                        has_vietnamese = any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 for c in word)
                        uppercase_ratio = _uppercase_ratio(word)
                        if uppercase_ratio > 0.5 or has_vietnamese:
                            valid_words.append(word)
                    if valid_words:
                        out["name"] = " ".join(valid_words)
    
    # Fallback: dùng cách cũ nếu không tìm thấy anchor
    if not out.get("name"):
        name_roi = _read_right_of_anchor(image_bgr, ocr, items, ANCHORS["name"], max_lines=1)
        if name_roi:
            # Tạo ROI từ name_roi để dùng preprocessing tốt hơn
            # Tìm anchor box để tạo ROI
            name_anchor_box = _find_anchor_box(items, ANCHORS["name"], ['full', 'name', 'ho', 'ten', 'họ', 'tên'])
            if name_anchor_box:
                H, W = image_bgr.shape[:2]
                ax1, ay1, ax2, ay2 = _box_to_rect(name_anchor_box)
                roi_x1 = min(ax2 + 2, W - 1)
                roi_x2 = min(W - 1, ax2 + W // 2 + 40)
                roi_y1 = max(0, ay1 - 8)
                roi_y2 = min(H, ay2 + 20)
                if roi_x1 < roi_x2 and roi_y1 < roi_y2:
                    name_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                    # Resize và preprocessing cho chữ in hoa
                    h_roi, w_roi = name_roi_img.shape[:2]
                    if h_roi < 150:
                        scale = 150 / h_roi
                        new_w = int(w_roi * scale)
                        name_roi_img = cv2.resize(name_roi_img, (new_w, 150), interpolation=cv2.INTER_CUBIC)
                    # Dùng hybrid OCR với preprocessing
                    name_text, _ = _hybrid_ocr_roi(name_roi_img, ocr, use_vietocr=True)
                    if name_text:
                        name_clean = re.sub(r"^(?i)(h\s*và\s*tên|ho\s*va\s*ten|họ\s*và\s*tên|full\s*name)[\s:：\-|/]*", "", name_text).strip()
                        name_clean = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|ho\s*va\s*ten|họ\s*và\s*tên|full\s*name)[\s:：\-|/]*", " ", name_clean)
                        name_clean = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name_clean)
                        name_clean = re.sub(r"\s+", " ", name_clean).strip()
                        words = name_clean.split()
                        valid_words = []
                        for word in words:
                            word_lower = word.lower()
                            if word_lower in ['h', 'và', 'tên', 'ho', 'va', 'ten', 'full', 'name']:
                                continue
                            if _uppercase_ratio(word) > 0.5 or any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 for c in word):
                                valid_words.append(word)
                        if valid_words:
                            out["name"] = " ".join(valid_words)
            
            # Nếu vẫn chưa có, dùng text từ name_roi
            if not out.get("name"):
                name_clean = re.sub(r"^(?i)(h\s*và\s*tên|ho\s*va\s*ten|họ\s*và\s*tên|full\s*name)[\s:：\-|/]*", "", name_roi).strip()
                name_clean = re.sub(r"(?i)(^|\s+)(ho\s*va\s*ten|họ\s*và\s*tên|full\s*name)[\s:：\-|/]*", " ", name_clean)
                name_clean = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name_clean)
                name_clean = re.sub(r"\s+", " ", name_clean).strip()
                if name_clean:
                    words = name_clean.split()
                    valid_words = []
                    for word in words:
                        word_lower = word.lower()
                        if word_lower in ['h', 'và', 'tên', 'ho', 'va', 'ten', 'full', 'name']:
                            continue
                        if _uppercase_ratio(word) > 0.5 or any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 for c in word):
                            valid_words.append(word)
                    if valid_words:
                        out["name"] = " ".join(valid_words)
        
        # Nếu vẫn chưa có, thử dùng hybrid OCR trên toàn bộ vùng name ước tính
        if not out.get("name") and items:
            # Tìm dòng có chứa "Họ và tên" hoặc "Full name" trong full text
            for i, line in enumerate(lines):
                line_norm = _norm_no_accents(line.lower())
                if any(anchor_norm in line_norm for anchor_norm in [_norm_no_accents(a).lower() for a in ANCHORS["name"]]):
                    # Tìm dòng tiếp theo (có thể là name)
                    if i + 1 < len(lines):
                        candidate_name = lines[i + 1]
                        # Dùng hybrid OCR trên dòng này
                        # Tạo ROI ước tính từ vị trí dòng
                        H, W = image_bgr.shape[:2]
                        # Ước tính vị trí dòng dựa trên index
                        estimated_y = int(H * (i + 1) / max(len(lines), 1))
                        roi_y1 = max(0, estimated_y - 30)
                        roi_y2 = min(H, estimated_y + 30)
                        roi_x1 = int(W * 0.4)  # Bắt đầu từ 40% chiều rộng
                        roi_x2 = W - 10
                        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
                            name_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                            name_text, _ = _hybrid_ocr_roi(name_roi_img, ocr, use_vietocr=True)
                            if name_text:
                                name_text = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name_text)
                                name_text = re.sub(r"\s+", " ", name_text).strip()
                                words = name_text.split()
                                valid_words = [w for w in words if _uppercase_ratio(w) > 0.5 or any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 for c in w)]
                                if valid_words:
                                    out["name"] = " ".join(valid_words)
                                    break

    # Nếu chưa có name, fallback text-based (giữ nguyên logic cũ)
    def _normalize_line(s: str) -> str:
        return _normalize(unicodedata.normalize("NFC", s)).lower()

    norm_lines = [_normalize(l) for l in lines]
    norm_lines_lc = [_normalize_line(l) for l in lines]

    def collect_after(anchors: List[str], stop_keywords: List[str], max_lines: int = 3) -> str:
        for i, line_lc in enumerate(norm_lines_lc):
            # Tìm anchor trong dòng (có thể là một phần của anchor)
            anchor_found = False
            for anchor in anchors:
                anchor_norm = _norm_no_accents(anchor).lower()
                # Tìm nếu anchor hoặc một phần của anchor có trong dòng
                if anchor_norm in line_lc or any(word in line_lc for word in anchor_norm.split() if len(word) >= 3):
                    anchor_found = True
                    break
            
            if anchor_found:
                collected: List[str] = []
                # lấy phần còn lại của dòng hiện tại sau nhãn
                current = norm_lines[i]
                # bỏ phần nhãn nếu có - cải thiện regex để bỏ nhiều loại nhãn hơn
                for anchor in anchors:
                    anchor_pattern = re.escape(_norm_no_accents(anchor))
                    current_clean = re.sub(rf"(?i){anchor_pattern}[:：\-\|/]*", "", current, flags=re.IGNORECASE)
                    current = current_clean
                current_clean = current.strip()
                if current_clean:
                    collected.append(current_clean)
                # lấy thêm các dòng tiếp theo cho đến khi gặp stop
                for j in range(i + 1, min(len(norm_lines), i + 1 + max_lines)):
                    next_lc = norm_lines_lc[j]
                    if any(sk in next_lc for sk in stop_keywords):
                        break
                    collected.append(norm_lines[j])
                result = _normalize(" ".join(collected))
                if result:
                    return result
        return ""

    # Heuristic 1: lấy sau nhãn (accent-insensitive), loại tiêu đề nếu dính trên cùng dòng
    name = collect_after(
        anchors=ANCHORS["name"],
        stop_keywords=ANCHORS["dob"] + ANCHORS["gender"],
        max_lines=2,
    )
    if name:
        name = _remove_anchors(name, ANCHORS["name"]) 

    # Heuristic 2: nếu chưa có, tìm dòng HOA nhất giữa số CCCD và "Ngày sinh"
    if not name:
        id_val = out.get("id_number")
        id_idx = None
        dob_idx = None
        for i, l in enumerate(norm_lines):
            if id_val and id_val in l:
                id_idx = i
            if re.search(r"(?i)ngày\s*sinh|date\s*of\s*birth", _normalize_line(l)):
                dob_idx = i
                break
        search_start = 0 if id_idx is None else id_idx
        search_end = len(norm_lines) if dob_idx is None else dob_idx
        best = ""
        best_score = 0.0
        for i in range(search_start, search_end):
            candidate = norm_lines[i]
            cleaned = re.sub(r"(?i)(họ\s*và\s*tên|ho\s*va\s*ten|full\s*name)[:：\-\|]*", "", candidate).strip()
            if len(cleaned) < 5:
                continue
            score = _uppercase_ratio(cleaned)
            # Ưu tiên dòng có nhiều chữ HOA
            if score > 0.6 and score >= best_score:
                best = cleaned
                best_score = score
        if best:
            name = best

    if name and not out.get("name"):
        # Chỉ giữ chữ cái Latin có dấu và khoảng trắng; chuẩn hoá khoảng trắng
        name = re.sub(r"(?i)(^|\s+)(ho\s*va\s*ten|họ\s*và\s*tên|full\s*name|ful\s*nmne|ful\s*name)[\s:：\-|/]*", " ", name)
        name = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        # Lọc các từ hợp lệ (có nhiều chữ HOA hoặc có dấu tiếng Việt)
        if name:
            words = name.split()
            valid_words = []
            for word in words:
                if _uppercase_ratio(word) > 0.5 or any(ord(c) >= 0x00C0 and ord(c) <= 0x1EF9 for c in word):
                    valid_words.append(word)
            if valid_words:
                out["name"] = " ".join(valid_words)
    
    # Quê quán (Place of origin): ưu tiên anchor+bbox với hybrid OCR
    origin_anchor_box = _find_anchor_box(items, ANCHORS["origin"], ['place', 'origin', 'que', 'quan', 'quê', 'quán'])
    if origin_anchor_box:
        H, W = image_bgr.shape[:2]
        ox1, oy1, ox2, oy2 = _box_to_rect(origin_anchor_box)
        # Tìm anchor tiếp theo (residence hoặc expiry) để biết điểm dừng
        residence_anchor_box = _find_anchor_box(items, ANCHORS["residence"], ['place', 'residen', 'residence', 'noi', 'thuong', 'tru'])
        expiry_anchor_box = _find_anchor_box(items, ANCHORS["expiry"], ['co', 'gia', 'tri', 'den', 'valid', 'until'])
        
        # Tạo ROI từ bên phải của "Quê quán" đến trước anchor tiếp theo
        roi_x1 = min(ox2 + 2, W - 1)
        if residence_anchor_box:
            rx1, ry1, rx2, ry2 = _box_to_rect(residence_anchor_box)
            roi_x2 = min(W - 1, rx1 - 5)
            roi_y2 = min(H, ry1 - 5)
        elif expiry_anchor_box:
            ex1, ey1, ex2, ey2 = _box_to_rect(expiry_anchor_box)
            roi_x2 = min(W - 1, ex1 - 5)
            roi_y2 = min(H, ey1 - 5)
        else:
            roi_x2 = min(W - 1, ox2 + W // 2 + 40)
            roi_y2 = min(H, oy2 + 60)
        roi_y1 = max(0, oy1 - 8)
        
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            origin_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            # Dùng hybrid OCR để nhận diện dấu tốt hơn
            origin_roi, _ = _hybrid_ocr_roi(origin_roi_img, ocr, use_vietocr=True)
            
            if origin_roi:
                # loại phần nhãn tiếng Anh và tiếng Việt còn sót
                origin_roi = re.sub(r"(?i)(^|\s+)(place\s*of\s*origin|quê\s*quán|que\s*quan)[\s:：\-|/]*", " ", origin_roi)
                # Giữ nguyên dấu tiếng Việt, loại bỏ ký tự đặc biệt không cần thiết
                origin_roi = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", origin_roi)
                origin_roi = re.sub(r"\s+", " ", origin_roi).strip()
                out["place_of_origin"] = _normalize(origin_roi)
    
    # Fallback: dùng cách cũ nếu không tìm thấy anchor
    if not out.get("place_of_origin"):
        origin_roi = _read_right_of_anchor(image_bgr, ocr, items, ANCHORS["origin"], max_lines=2)
        if origin_roi:
            # loại phần nhãn tiếng Anh và tiếng Việt còn sót
            origin_roi = re.sub(r"(?i)(^|\s+)(place\s*of\s*origin|quê\s*quán|que\s*quan)[\s:：\-|/]*", " ", origin_roi)
            # Giữ nguyên dấu tiếng Việt, loại bỏ ký tự đặc biệt không cần thiết
            origin_roi = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", origin_roi)
            origin_roi = re.sub(r"\s+", " ", origin_roi).strip()
            out["place_of_origin"] = _normalize(origin_roi)
    if not out.get("place_of_origin"):
        # fallback text-based
        origin = collect_after(
            anchors=ANCHORS["origin"],
            stop_keywords=ANCHORS["residence"] + ANCHORS["expiry"] + ANCHORS["issue"],
            max_lines=3,  # Tăng số dòng để lấy đủ thông tin
        )
        if origin:
            origin_clean = _remove_anchors(origin, ANCHORS["origin"]) 
            origin_clean = _cut_before_next_label(origin_clean, ANCHORS["residence"] + ANCHORS["expiry"] + ANCHORS["issue"])
            # Làm sạch và giữ nguyên dấu tiếng Việt
            origin_clean = re.sub(r"(?i)(^|\s+)(place\s*of\s*origin|quê\s*quán|que\s*quan)[\s:：\-|/]*", " ", origin_clean)
            origin_clean = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", origin_clean)
            origin_clean = re.sub(r"\s+", " ", origin_clean)
            origin_clean = re.sub(r"\s*,\s*", ", ", origin_clean).strip()
            out["place_of_origin"] = _normalize(origin_clean)
    # Nơi thường trú (Place of residence)
    # Cách tiếp cận mới: Tìm các box text nằm DƯỚI label "Nơi thường trú" và TRÊN label "Có giá trị đến"
    # CHỈ lấy text ở dưới label, không lấy text từ "Có giá trị đến"
    residence_anchor_box = _find_anchor_box(items, ANCHORS["residence"], ['place', 'residen', 'residence', 'noi', 'thuong', 'tru', 'nơi', 'thường', 'trú'])
    expiry_anchor_box = _find_anchor_box(items, ANCHORS["expiry"], ['co', 'gia', 'tri', 'den', 'valid', 'until', 'expiry'])
    
    if residence_anchor_box:
        H, W = image_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = _box_to_rect(residence_anchor_box)
        
        # Tìm các box text nằm DƯỚI label "Nơi thường trú" (y > ry2)
        # và trong vùng ngang từ bên phải label (x > rx2)
        residence_text_boxes = []
        expiry_y = None
        expiry_y_bottom = None
        if expiry_anchor_box:
            ex1, ey1, ex2, ey2 = _box_to_rect(expiry_anchor_box)
            expiry_y = ey1  # Dừng trước dòng "Có giá trị đến"
            expiry_y_bottom = ey2
        
        # Tìm tất cả các box có thể chứa text "Có giá trị đến" để tránh lấy nhầm
        expiry_text_boxes_y = []
        for txt, conf, crop, box in items:
            txt_norm = _norm_no_accents(txt.lower())
            if any(exp in txt_norm for exp in ['co gia tri den', 'co gia tri', 'valid until', 'good thru', 'co gia tr', 'co gia tr dn', 'co gia tr den']):
                x1, y1, x2, y2 = _box_to_rect(box)
                expiry_text_boxes_y.append((y1, y2))
        
        for txt, conf, crop, box in items:
            x1, y1, x2, y2 = _box_to_rect(box)
            mid_y = (y1 + y2) // 2
            mid_x = (x1 + x2) // 2
            
            # Loại bỏ box có text là label "Có giá trị đến" TRƯỚC
            txt_norm = _norm_no_accents(txt.lower())
            if any(exp in txt_norm for exp in ['co gia tri den', 'co gia tri', 'valid until', 'good thru', 'co gia tr', 'co gia tr dn', 'co gia tr den']):
                continue  # Bỏ qua box này hoàn toàn
            
            # Kiểm tra xem box này có nằm trong vùng của "Có giá trị đến" không
            # Chỉ loại bỏ nếu box này THỰC SỰ nằm trong vùng expiry (overlap đáng kể)
            is_in_expiry_zone = False
            for exp_y1, exp_y2 in expiry_text_boxes_y:
                # Chỉ coi là overlap nếu box này có phần lớn nằm trong vùng expiry
                # Hoặc nếu box expiry nằm hoàn toàn trong box này (box này chứa expiry)
                box_height = y2 - y1
                expiry_height = exp_y2 - exp_y1
                overlap_height = min(y2, exp_y2) - max(y1, exp_y1)
                
                # Chỉ loại bỏ nếu overlap đáng kể (>= 50% chiều cao box nhỏ hơn)
                if overlap_height > 0:
                    min_height = min(box_height, expiry_height)
                    if overlap_height >= min_height * 0.5:  # Overlap >= 50%
                        is_in_expiry_zone = True
                        break
            if is_in_expiry_zone:
                continue  # Bỏ qua box nằm trong vùng "Có giá trị đến"
            
            # Chỉ lấy box nằm DƯỚI label "Nơi thường trú"
            # Nới lỏng điều kiện để lấy đủ các dòng địa chỉ (có thể có nhiều dòng)
            if mid_y > ry1:  # Lấy tất cả box nằm từ label trở xuống (không chỉ dưới label)
                # Lấy box trong vùng ngang từ bên phải label
                # Nhưng loại bỏ box của chính label "Nơi thường trú"
                txt_norm_check = _norm_no_accents(txt.lower())
                is_residence_label = any(res in txt_norm_check for res in ['noi thuong tru', 'nơi thường trú', 'place of residen'])
                if not is_residence_label and mid_x > rx2 - 10:  # Nới lỏng: cho phép box nằm từ bên phải label
                    # KHÔNG dừng sớm dựa vào expiry_y - chỉ dựa vào việc kiểm tra overlap với expiry zone
                    # Điều này đảm bảo lấy TẤT CẢ các dòng địa chỉ trước "Có giá trị đến"
                    residence_text_boxes.append((txt, conf, mid_y))
        
        # Sắp xếp theo y để giữ thứ tự (từ trên xuống dưới)
        residence_text_boxes.sort(key=lambda x: x[2])
        
        if residence_text_boxes:
            # Debug: Log số lượng box tìm được
            _dbg(f'Found {len(residence_text_boxes)} residence text boxes')
            for i, (txt, conf, y) in enumerate(residence_text_boxes):
                _dbg(f'  Box {i}: y={y}, text="{txt[:50]}"')
            
            # Ghép text từ các box, loại bỏ label
            residence_parts = []
            for txt, conf, y_pos in residence_text_boxes:
                # Fix cứng: Loại bỏ "Có giá tr đn" và các biến thể TRƯỚC TIÊN
                # (bao gồm cả không dấu và có dấu, với các khoảng trắng khác nhau)
                txt_clean = re.sub(r"(?i)(có\s*giá\s*tr\s*đn|co\s*gia\s*tr\s*đn|co\s*gia\s*tr\s*dn|co\s*gia\s*tr\s*den|có\s*giá\s*trị\s*đến|co\s*gia\s*tri\s*den|valid\s*until|good\s*thru)", "", txt)
                # Fix cứng: Loại bỏ "H và tên" và các biến thể (ở đầu hoặc giữa text)
                txt_clean = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|h\s*va\s*ten|h\s*và\s*ten|h\s*va\s*tên|ho\s*va\s*ten|họ\s*và\s*tên)[\s:：\-|/]*", " ", txt_clean)
                # Loại bỏ label "Nơi thường trú" nếu có
                txt_clean = re.sub(r"(?i)(^|\s+)(nơi\s*thường\s*trú|noi\s*thuong\s*tru|place\s*of\s*residen[ce])[\s:：\-|/]*", " ", txt_clean)
                # Loại bỏ các ký tự đặc biệt và chuẩn hóa khoảng trắng
                txt_clean = re.sub(r"\s+", " ", txt_clean).strip()
                if txt_clean:
                    residence_parts.append(txt_clean)
            
            if residence_parts:
                # Ghép các phần với khoảng trắng, giữ nguyên thứ tự
                residence_text = " ".join(residence_parts)
                _dbg(f'Combined residence text: "{residence_text[:100]}"')
                # Loại bỏ ngày tháng nếu có
                residence_text = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', residence_text).strip()
                # Làm sạch
                residence_text = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", residence_text)
                residence_text = re.sub(r"\s+", " ", residence_text)
                residence_text = re.sub(r"\s*,\s*", ", ", residence_text).strip()
                out["residence"] = _normalize(residence_text)
    
    # Fallback: dùng ROI nếu không tìm thấy bằng box
    if not out.get("residence") and residence_anchor_box and expiry_anchor_box:
        H, W = image_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = _box_to_rect(residence_anchor_box)
        ex1, ey1, ex2, ey2 = _box_to_rect(expiry_anchor_box)
        
        # Tìm tất cả các box "Có giá trị đến" để xác định vùng cần tránh chính xác hơn
        expiry_boxes_y_range = []
        for txt, conf, crop, box in items:
            txt_norm = _norm_no_accents(txt.lower())
            if any(exp in txt_norm for exp in ['co gia tri den', 'co gia tri', 'valid until', 'good thru', 'co gia tr', 'co gia tr dn', 'co gia tr den']):
                bx1, by1, bx2, by2 = _box_to_rect(box)
                expiry_boxes_y_range.append((by1, by2))
        
        # Xác định y tối thiểu của vùng "Có giá trị đến" (để dừng trước đó)
        min_expiry_y = min([y1 for y1, y2 in expiry_boxes_y_range]) if expiry_boxes_y_range else ey1
        
        # Tạo ROI CHỈ lấy text ở DƯỚI label "Nơi thường trú"
        roi_x1 = min(rx2 + 2, W - 1)  # Bắt đầu từ bên phải label
        roi_x2 = min(W - 1, ex1 - 15)  # Dừng trước anchor "Có giá trị đến" với padding lớn
        roi_y1 = max(0, ry2 + 5)  # Bắt đầu từ DƯỚI label (ry2 + 5)
        # Dừng ngay trước vùng "Có giá trị đến" (padding tối thiểu) để lấy TẤT CẢ các dòng địa chỉ
        roi_y2 = min(H, min_expiry_y - 2)  # Padding tối thiểu để lấy đủ tất cả các dòng
        
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            residence_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            residence_roi, _ = _hybrid_ocr_roi(residence_roi_img, ocr, use_vietocr=True)
            
            if residence_roi:
                # Loại bỏ ngày tháng
                residence_roi = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', residence_roi).strip()
                # Fix cứng: Loại bỏ "Có giá tr đn" và các biến thể TRƯỚC TIÊN
                residence_roi = re.sub(r"(?i)(có\s*giá\s*tr\s*đn|co\s*gia\s*tr\s*đn|co\s*gia\s*tr\s*dn|co\s*gia\s*tr\s*den|có\s*giá\s*trị\s*đến|co\s*gia\s*tri\s*den|valid\s*until|good\s*thru).*$", "", residence_roi).strip()
                # Fix cứng: Loại bỏ "H và tên" và các biến thể
                residence_roi = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|h\s*va\s*ten|h\s*và\s*ten|h\s*va\s*tên|ho\s*va\s*ten|họ\s*và\s*tên)[\s:：\-|/]*", " ", residence_roi)
                # Làm sạch
                residence_roi = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", residence_roi)
                residence_roi = re.sub(r"\s+", " ", residence_roi)
                residence_roi = re.sub(r"\s*,\s*", ", ", residence_roi).strip()
                out["residence"] = _normalize(residence_roi)
    
    # Fallback: dùng cách cũ nếu không tìm thấy cả 2 anchor
    # Tăng max_lines để lấy đủ các dòng
    if not out.get("residence"):
        residence_roi = _read_right_of_anchor(
            image_bgr,
            ocr,
            items,
            ANCHORS["residence"],
            max_lines=10,  # Tăng từ 8 lên 10 để lấy đủ các dòng
            max_gap_px=25,  # Tăng gap để lấy được các dòng xa hơn
            right_pad=80,  # Tăng padding để lấy đủ chiều rộng
            stop_at_date=True,
            stop_after_anchors=ANCHORS["expiry"],
        ) 
        if residence_roi:
            # Fix cứng: Loại bỏ "Có giá tr đn" và các biến thể TRƯỚC TIÊN
            residence_roi = re.sub(r"(?i).*?(có\s*giá\s*tr\s*đn|co\s*gia\s*tr\s*đn|co\s*gia\s*tr\s*dn|co\s*gia\s*tr\s*den|có\s*giá\s*trị\s*đến|co\s*gia\s*tri\s*den|valid\s*until|good\s*thru).*$", "", residence_roi).strip()
            # Fix cứng: Loại bỏ "H và tên" và các biến thể
            residence_roi = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|h\s*va\s*ten|h\s*và\s*ten|h\s*va\s*tên|ho\s*va\s*ten|họ\s*và\s*tên)[\s:：\-|/]*", " ", residence_roi)
            # Loại bỏ ngày tháng ở cuối (nhiều format: dd/mm/yyyy, dd-mm-yyyy, dd mm yyyy)
            residence_roi = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', residence_roi).strip()
            residence_roi = re.sub(r'\b\d{1,2}\s+\d{1,2}\s+\d{4}.*$', '', residence_roi).strip()
            # Loại bỏ các pattern ngày tháng khác
            residence_roi = re.sub(r'\b\d{2}\s+\d{2}\s+\d{4}.*$', '', residence_roi).strip()
            # loại phần nhãn tiếng Anh và tiếng Việt còn sót
            residence_roi = re.sub(r"(?i)(^|\s+)(place\s*of\s*residen[ce]|nơi\s*thường\s*trú|noi\s*thuong\s*tru)[\s:：\-|/]*", " ", residence_roi)
            # Giữ nguyên dấu tiếng Việt, số, dấu phẩy, dấu chấm (cho địa chỉ)
            residence_roi = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", residence_roi)
            # Chuẩn hóa khoảng trắng nhưng giữ dấu phẩy
            residence_roi = re.sub(r"\s+", " ", residence_roi)
            residence_roi = re.sub(r"\s*,\s*", ", ", residence_roi).strip()
            out["residence"] = _normalize(residence_roi)
    if not out.get("residence"):
        # fallback text-based - tăng max_lines để lấy đủ các dòng
        residence = collect_after(
            anchors=ANCHORS["residence"],
            stop_keywords=ANCHORS["expiry"] + ANCHORS["issue"] + ANCHORS["gender"],
            max_lines=8,  # Tăng từ 5 lên 8 để lấy đủ địa chỉ dài
        )
        if residence:
            # Fix cứng: Loại bỏ "Có giá tr đn" và các biến thể TRƯỚC TIÊN
            residence = re.sub(r"(?i).*?(có\s*giá\s*tr\s*đn|co\s*gia\s*tr\s*đn|co\s*gia\s*tr\s*dn|co\s*gia\s*tr\s*den|có\s*giá\s*trị\s*đến|co\s*gia\s*tri\s*den|valid\s*until|good\s*thru).*$", "", residence).strip()
            # Fix cứng: Loại bỏ "H và tên" và các biến thể
            residence = re.sub(r"(?i)(^|\s+)(h\s*và\s*tên|h\s*va\s*ten|h\s*và\s*ten|h\s*va\s*tên|ho\s*va\s*ten|họ\s*và\s*tên)[\s:：\-|/]*", " ", residence)
            residence = _remove_anchors(residence, ANCHORS["residence"]) 
            residence = _cut_before_next_label(residence, ANCHORS["expiry"] + ANCHORS["issue"])
            # Loại bỏ ngày tháng ở cuối (nhiều format)
            residence = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', residence).strip()
            residence = re.sub(r'\b\d{1,2}\s+\d{1,2}\s+\d{4}.*$', '', residence).strip()
            # Làm sạch và giữ nguyên dấu tiếng Việt
            residence = re.sub(r"(?i)(^|\s+)(place\s*of\s*residen[ce]|nơi\s*thường\s*trú|noi\s*thuong\s*tru)[\s:：\-|/]*", " ", residence)
            residence = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", residence)
            residence = re.sub(r"\s+", " ", residence)
            residence = re.sub(r"\s*,\s*", ", ", residence).strip()
            out["residence"] = _normalize(residence)
    # Fallback mạnh từ full text nếu vẫn chưa có
    if not out.get("residence"):
        res2 = _slice_between(full, start_keywords=ANCHORS["residence"], stop_keywords=ANCHORS["expiry"] + ANCHORS["issue"]) 
        if res2:
            # Loại bỏ bất kỳ ngày/thời gian ở cuối do OCR lẫn (nhiều format)
            res2 = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', res2).strip()
            res2 = re.sub(r'\b\d{1,2}\s+\d{1,2}\s+\d{4}.*$', '', res2).strip()
            # Làm sạch và giữ nguyên dấu tiếng Việt
            res2 = re.sub(r"(?i)(^|\s+)(place\s*of\s*residen[ce]|nơi\s*thường\s*trú|noi\s*thuong\s*tru)[\s:：\-|/]*", " ", res2)
            res2 = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", res2)
            res2 = re.sub(r"\s+", " ", res2)
            res2 = re.sub(r"\s*,\s*", ", ", res2).strip()
            out["residence"] = _normalize(res2)
    # Địa chỉ tổng quát (fallback)
    if not out.get("residence"):
        m = re.search(r"Địa\s*chỉ[: ]+(.+?)(?:Ngày\s*cấp|$)", full, re.I | re.DOTALL)
        if m:
            res3 = m.group(1)
            # Loại bỏ ngày tháng nếu có
            res3 = re.sub(r'\b\d{1,2}[\/\- ]\d{1,2}[\/\- ]\d{4}.*$', '', res3).strip()
            res3 = re.sub(r'\b\d{1,2}\s+\d{1,2}\s+\d{4}.*$', '', res3).strip()
            # Làm sạch và giữ nguyên dấu tiếng Việt
            res3 = re.sub(r"[^A-Za-zÀ-ỹ0-9\s,\.]", " ", res3)
            res3 = re.sub(r"\s+", " ", res3)
            res3 = re.sub(r"\s*,\s*", ", ", res3).strip()
            out["residence"] = _normalize(res3)

    # "Có giá trị đến" (Expiry date) - Extract bằng ROI để tránh lẫn label
    expiry_anchor_box = _find_anchor_box(items, ANCHORS["expiry"], ['co', 'gia', 'tri', 'den', 'valid', 'until', 'expiry'])
    if expiry_anchor_box:
        H, W = image_bgr.shape[:2]
        ex1, ey1, ex2, ey2 = _box_to_rect(expiry_anchor_box)
        # Tạo ROI từ bên phải của "Có giá trị đến" đến hết dòng
        roi_x1 = min(ex2 + 2, W - 1)
        roi_x2 = min(W - 1, ex2 + 150)  # Đủ rộng để lấy ngày
        roi_y1 = max(0, ey1 - 5)
        roi_y2 = min(H, ey2 + 5)
        
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            expiry_roi_img = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            expiry_text, _ = _hybrid_ocr_roi(expiry_roi_img, ocr, use_vietocr=True)
            
            if expiry_text:
                # Loại bỏ label nếu có
                expiry_text = re.sub(r"(?i)(^|\s+)(có\s*giá\s*trị\s*đến|co\s*gia\s*tri\s*den|valid\s*until|good\s*thru)[\s:：\-|/]*", " ", expiry_text)
                # Tìm ngày trong text đã làm sạch
                m_exp = re.search(r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-]((?:19|20)\d{2})\b", expiry_text)
                if m_exp:
                    out["expiry_date"] = f"{m_exp.group(1).zfill(2)}/{m_exp.group(2).zfill(2)}/{m_exp.group(3)}"
    
    # Fallback: dùng regex từ full text nếu không tìm thấy bằng ROI
    if not out.get("expiry_date"):
        m_exp = re.search(r"(Có\s*giá\s*trị\s*đến|Good\s*thru|Valid\s*(?:until|thru))\s*[: ]+((?:0?[1-9]|[12]\d|3[01])[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{2})",
                          full, re.I)
        if m_exp:
            out["expiry_date"] = _normalize(m_exp.group(2))
        # Nếu chưa phân loại mà có >1 date, gán date thứ 2 là expiry_date (CCCD mẫu mới)
        if not out.get("expiry_date") and len(dates) > 1:
            out["expiry_date"] = pack(dates[1])
    
    # Ngày cấp + Nơi cấp (ưu tiên theo anchor-box "issue")
    issue_roi = _read_right_of_anchor(image_bgr, ocr, items, ANCHORS["issue"])
    if issue_roi:
        # Tách ngày trong chuỗi issue
        m2 = re.search(r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-]((?:19|20)\d{2})\b", issue_roi)
        if m2:
            out["issue_date"] = f"{m2.group(1).zfill(2)}/{m2.group(2).zfill(2)}/{m2.group(3)}"
            issue_rest = issue_roi.replace(m2.group(0), " ").strip()
        else:
            issue_rest = issue_roi
        # Loại nhãn
        issue_rest = re.sub(r"(?i)(ngay\s*cap|ngày\s*cấp|issued\s*on|issue\s*date|issued\s*by|noi\s*cap|nơi\s*cấp)[:：\-|/]*", "", issue_rest)
        issue_rest = re.sub(r"\s+", " ", issue_rest).strip()
        if issue_rest:
            out["issuer"] = issue_rest

    # Refine bằng VietOCR nếu có cho các trường dài
    viet = get_vietocr_instance()
    need_refine = viet is not None and (
        not out.get("name") or _uppercase_ratio(out.get("name", "")) > 0.8 or not out.get("place_of_origin") or not out.get("residence")
    )
    if need_refine:
        try:
            det_items = _ocr_collect_with_crops(ocr, resize_keep_aspect(image_bgr, height=320))
            texts = [t for (t, c, cr) in det_items]
            texts_lc = [_strip_accents(_normalize(t)).lower() for t in texts]

            def viet_join_after(anchor_keywords: List[str], stop_keywords: List[str], max_lines: int = 3) -> str:
                for i, t in enumerate(texts_lc):
                    if any(a in t for a in anchor_keywords):
                        collected: List[str] = []
                        for j in range(i + 1, min(len(det_items), i + 1 + max_lines)):
                            if any(sk in texts_lc[j] for sk in stop_keywords):
                                break
                            crop = det_items[j][2]
                            if crop is None or crop.size == 0:
                                continue
                            try:
                                recog = viet.predict(crop)
                                if recog:
                                    collected.append(str(recog))
                            except Exception:
                                continue
                        return _normalize(" ".join(collected))
                return ""

            # Name
            if not out.get('name') or _uppercase_ratio(out.get('name', '')) > 0.8:
                vname = viet_join_after(
                    anchor_keywords=["ho va ten", "ho và ten", "họ va ten", "họ và tên", "full name"],
                    stop_keywords=["ngay sinh", "date of birth", "gioi tinh", "sex"],
                    max_lines=2,
                )
                if vname:
                    vname = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", vname)
                    vname = re.sub(r"\s+", " ", vname).strip()
                    if vname:
                        out['name'] = vname

            # Place of origin
            if not out.get('place_of_origin'):
                vorig = viet_join_after(
                    anchor_keywords=["que quan", "quê quan", "quê quán", "place of origin"],
                    stop_keywords=["noi thuong tru", "nơi thuong tru", "nơi thường trú", "place of residence", "co gia tri", "valid"],
                    max_lines=2,
                )
                if vorig:
                    out['place_of_origin'] = _normalize(vorig)

            # Residence
            if not out.get('residence'):
                vres = viet_join_after(
                    anchor_keywords=["noi thuong tru", "nơi thuong tru", "nơi thường trú", "place of residence"],
                    stop_keywords=["co gia tri", "valid"],
                    max_lines=3,
                )
                if vres:
                    out['residence'] = _normalize(vres)
        except Exception:
            pass
    # Địa chỉ tổng quát (fallback)
    if not out.get("residence"):
        m = re.search(r"Địa\s*chỉ[: ]+(.+?)(?:Ngày\s*cấp|$)", full, re.I | re.DOTALL)
        if m:
            out["residence"] = _normalize(m.group(1))
    
    # Nơi cấp (heuristic)
    issuer_patterns = [
        r"Nơi\s*cấp[: ]+(.+?)(?:$)",
        r"Cơ\s*quan\s*cấp[: ]+(.+?)(?:$)",
    ]
    if not out.get("issuer"):
        for pattern in issuer_patterns:
            m = re.search(pattern, full, re.I | re.DOTALL)
            if m:
                out["issuer"] = _normalize(m.group(1))
                break
    
    return out

def validate_ocr_result(fields: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate kết quả OCR - kiểm tra các trường bắt buộc
    Returns: (is_valid, error_message)
    """
    if not fields.get("id_number"):
        return False, "Không tìm thấy số CCCD. Vui lòng chụp lại ảnh rõ ràng, thẳng góc."
    
    if not fields.get("dob"):
        return False, "Không tìm thấy ngày sinh. Vui lòng chụp lại ảnh rõ ràng, thẳng góc."
    
    return True, None

if __name__ == "__main__":
    import argparse
    import json
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Đường dẫn đến ảnh CCCD")
    args = ap.parse_args()
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ {args.image}")
        exit(1)
    
    data = extract_id_fields(img)
    is_valid, error_msg = validate_ocr_result(data)
    
    if not is_valid:
        print(f"Lỗi: {error_msg}")
    
    print(json.dumps(data, ensure_ascii=False, indent=2))


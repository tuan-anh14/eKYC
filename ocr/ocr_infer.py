# ocr/ocr_infer.py

from paddleocr import PaddleOCR
import cv2
import re
import unicodedata
from typing import Dict, Optional, Tuple, List
import os
from .preprocess import preprocess_for_ocr, resize_keep_aspect

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
    # Mặc định KHÔNG bật VietOCR để nhẹ; chỉ bật khi USE_VIETOCR=1
    if os.environ.get('USE_VIETOCR', '0') != '1':
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

def _ocr_collect_with_crops(ocr: PaddleOCR, image_bgr) -> List[Tuple[str, float, any]]:
    """Thu thập (text, conf, crop_bgr) bằng PaddleOCR để refine với VietOCR."""
    items: List[Tuple[str, float, any]] = []
    try:
        result = ocr.ocr(_ensure_bgr(image_bgr))
    except Exception:
        try:
            result = ocr.predict(_ensure_bgr(image_bgr))
        except Exception:
            return items
    if isinstance(result, list):
        for page in result:
            if not page:
                continue
            for line in page:
                try:
                    box = line[0]
                    rec = line[1]
                    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                        txt = str(rec[0])
                        conf = float(rec[1]) if rec[1] is not None else 1.0
                    elif isinstance(rec, dict):
                        txt = str(rec.get('text', ''))
                        conf = float(rec.get('confidence', 1.0))
                    else:
                        txt = str(rec)
                        conf = 1.0
                    x_coords = [int(p[0]) for p in box]
                    y_coords = [int(p[1]) for p in box]
                    x1, y1 = max(0, min(x_coords)), max(0, min(y_coords))
                    x2, y2 = max(x_coords), max(y_coords)
                    pad = 4
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(image_bgr.shape[1], x2 + pad)
                    y2 = min(image_bgr.shape[0], y2 + pad)
                    crop = image_bgr[y1:y2, x1:x2]
                    items.append((txt, conf, crop))
                except Exception:
                    continue
    return items


def _ocr_multipass_text(ocr: PaddleOCR, image_bgr) -> Tuple[str, float, List[str]]:
    """Chạy OCR nhiều pass (resize + preprocess) và chọn kết quả có avg confidence cao nhất.
    Trả về: (full_text, avg_conf, lines)
    """
    candidates: List[Tuple[str, float, List[str]]] = []
    # Pass 1: resize mềm để tăng chiều cao chữ
    img1 = resize_keep_aspect(image_bgr, height=256)
    lines1, confs1 = _ocr_collect_lines(ocr, img1)
    if lines1:
        avg1 = sum(confs1) / max(1, len(confs1))
        candidates.append((" ".join(lines1), avg1, lines1))
    # Pass 2: tiền xử lý mạnh (CLAHE + denoise + sharpen + adaptive threshold)
    img2 = preprocess_for_ocr(image_bgr)
    lines2, confs2 = _ocr_collect_lines(ocr, img2)
    if lines2:
        avg2 = sum(confs2) / max(1, len(confs2))
        candidates.append((" ".join(lines2), avg2, lines2))

    if not candidates:
        return "", 0.0, []
    # Chọn theo avg confidence
    best_text, best_conf, best_lines = max(candidates, key=lambda x: x[1])
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


def extract_id_fields(image_bgr) -> Dict:
    """
    Input: ảnh BGR (cv2.imread hoặc numpy array)
    Output: dict {
        id_number, name, dob, gender, nationality, 
        place_of_origin, residence, issue_date, expiry_date
    }
    """
    ocr = get_ocr_instance()
    full, avg_conf, lines = _ocr_multipass_text(ocr, image_bgr)
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
        # Phân loại ngày cấp / hạn sử dụng dựa theo ngữ cảnh
        # Ưu tiên bắt "Có giá trị đến" làm expiry_date
        m_exp = re.search(r"(Có\s*giá\s*trị\s*đến|Good\s*thru|Valid\s*(?:until|thru))\s*[: ]+((?:0?[1-9]|[12]\d|3[01])[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{2})",
                          full, re.I)
        if m_exp:
            out["expiry_date"] = _normalize(m_exp.group(2))
        # Ngày cấp nếu có nhãn
        m_issue = re.search(r"Ngày\s*cấp\s*[: ]+((?:0?[1-9]|[12]\d|3[01])[\/\-](?:0?[1-9]|1[0-2])[\/\-](?:19|20)\d{2})",
                            full, re.I)
        if m_issue:
            out["issue_date"] = _normalize(m_issue.group(1))
        # Nếu chưa phân loại mà có >1 date, gán date thứ 2 là expiry_date (CCCD mẫu mới)
        if not out.get("expiry_date") and len(dates) > 1:
            out["expiry_date"] = pack(dates[1])
    
    # Giới tính
    if re.search(r"\bNam\b", full, re.I):
        out["gender"] = "Nam"
    elif re.search(r"\bNữ\b|Nu|Nữ", full, re.I):
        out["gender"] = "Nữ"
    
    # Quốc tịch
    if re.search(r"Vi[eê]t\s*Nam", full, re.I):
        out["nationality"] = "Việt Nam"
    
    # Họ tên / Name: dùng anchors dòng để lấy phần sau nhãn đến trước "Ngày sinh"
    def _normalize_line(s: str) -> str:
        return _normalize(unicodedata.normalize("NFC", s)).lower()

    norm_lines = [_normalize(l) for l in lines]
    norm_lines_lc = [_normalize_line(l) for l in lines]

    def collect_after(anchors: List[str], stop_keywords: List[str], max_lines: int = 3) -> str:
        for i, line_lc in enumerate(norm_lines_lc):
            if any(a in line_lc for a in anchors):
                collected: List[str] = []
                # lấy phần còn lại của dòng hiện tại sau nhãn
                current = norm_lines[i]
                # bỏ phần nhãn nếu có
                current_clean = re.sub(r"(?i)(họ\s*và\s*tên|ho\s*va\s*ten|full\s*name)[:：\-\|]*", "", current).strip()
                if current_clean:
                    collected.append(current_clean)
                # lấy thêm 1-2 dòng tiếp theo cho đến khi gặp stop
                for j in range(i + 1, min(len(norm_lines), i + 1 + max_lines)):
                    next_lc = norm_lines_lc[j]
                    if any(sk in next_lc for sk in stop_keywords):
                        break
                    collected.append(norm_lines[j])
                return _normalize(" ".join(collected))
        return ""

    # Heuristic 1: lấy sau nhãn (accent-insensitive), loại tiêu đề nếu dính trên cùng dòng
    name = collect_after(
        anchors=["họ và tên", "ho va ten", "full name", "ho và tên"],
        stop_keywords=["ngày sinh", "date of birth", "giới tính", "sex"],
        max_lines=2,
    )
    if name:
        name = _remove_anchors(name, ["họ và tên", "ho va ten", "full name", "ho và tên"]) 

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

    if name:
        # Chỉ giữ chữ cái Latin có dấu và khoảng trắng; chuẩn hoá khoảng trắng
        name = re.sub(r"(?i)(^ho\s*va\s*ten|^họ\s*và\s*tên|^full\s*name)[\s:：\-|/]*", "", name)
        name = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        out["name"] = name
    
    # Quê quán (Place of origin): lấy từ anchors đến trước "Nơi thường trú"
    origin = collect_after(
        anchors=["quê quán", "que quan", "place of origin"],
        stop_keywords=["nơi thường trú", "noi thuong tru", "place of residence", "có giá trị", "co gia tri", "ngày cấp", "issue date"],
        max_lines=2,
    )
    if origin:
        origin_clean = _remove_anchors(origin, ["quê quán", "que quan", "place of origin"]) 
        # Cắt cứng khi gặp dấu phân tách + nhãn kế tiếp (bao gồm bản thiếu chữ/không dấu)
        ascii_o = _strip_accents(origin_clean).lower()
        cut_candidates = [
            "/ place of residen", "/ place of residence", "| place of residen", "| place of residence",
            " place of residen", " place of residence", " noi thuong", " noi ",
        ]
        cut_pos = None
        for tok in cut_candidates:
            pos = ascii_o.find(tok)
            if pos != -1:
                cut_pos = pos if cut_pos is None else min(cut_pos, pos)
        if cut_pos is not None:
            origin_clean = origin_clean[:cut_pos]
        # Cắt mềm theo danh sách nhãn dừng
        origin_clean = _cut_before_next_label(origin_clean, [
            "nơi thường trú", "noi thuong tru", "place of residence", "place of residen",
            "có giá trị", "co gia tri", "ngày cấp", "issue date",
        ])
        out["place_of_origin"] = origin_clean
    # Nơi thường trú (Place of residence)
    residence = collect_after(
        anchors=["nơi thường trú", "noi thuong tru", "place of residence"],
        stop_keywords=["có giá trị", "co gia tri", "valid", "valid until", "ngày cấp", "issue date", "sex", "giới tính"],
        max_lines=3,
    )
    if residence:
        residence = _remove_anchors(residence, ["nơi thường trú", "noi thuong tru", "place of residence"]) 
        # Nếu còn dính "Place of residen..." ở cuối, cắt bỏ
        ascii_r = _strip_accents(residence).lower()
        for tok in ["/ place of residen", "| place of residen", " place of residen", " place of residence"]:
            pos = ascii_r.find(tok)
            if pos != -1:
                residence = residence[:pos]
                break
        residence = _cut_before_next_label(residence, [
            "có giá trị", "co gia tri", "valid", "valid until", "ngày cấp", "issue date"
        ])
        out["residence"] = residence
    # Fallback mạnh từ full text nếu vẫn chưa có
    if not out.get("residence"):
        res2 = _slice_between(
            full,
            start_keywords=["nơi thường trú", "noi thuong tru", "place of residence", "place of residen"],
            stop_keywords=["có giá trị", "co gia tri", "valid", "valid until", "ngày cấp", "issue date"],
        )
        if res2:
            # Loại bỏ bất kỳ ngày/thời gian ở cuối do OCR lẫn
            res2 = re.sub(r"\b\d{2}[\/-]\d{2}[\/-]\d{4}.*$", "", res2).strip()
            out["residence"] = res2
    # Địa chỉ tổng quát (fallback)
    if not out.get("residence"):
        m = re.search(r"Địa\s*chỉ[: ]+(.+?)(?:Ngày\s*cấp|$)", full, re.I | re.DOTALL)
        if m:
            out["residence"] = _normalize(m.group(1))

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
                    stop_keywords=["noi thuong tru", "nơi thuong tru", "nơi thường trú", "place of residence", "co gia tri", "valid", "ngay cap", "issue"],
                    max_lines=2,
                )
                if vorig:
                    out['place_of_origin'] = _normalize(vorig)

            # Residence
            if not out.get('residence'):
                vres = viet_join_after(
                    anchor_keywords=["noi thuong tru", "nơi thuong tru", "nơi thường trú", "place of residence"],
                    stop_keywords=["co gia tri", "valid", "ngay cap", "issue"],
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


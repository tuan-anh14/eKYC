# ocr/ocr_infer.py

from paddleocr import PaddleOCR
import cv2
import re
import unicodedata
from typing import Dict, Optional, Tuple

# Khởi tạo OCR model một lần (singleton pattern)
_ocr_instance = None

def get_ocr_instance():
    """Lazy initialization của PaddleOCR để tránh load model nhiều lần"""
    global _ocr_instance
    if _ocr_instance is None:
        # Phiên bản PaddleOCR mới (3.3.1+) đã thay đổi API
        # Chỉ sử dụng tham số cơ bản
        _ocr_instance = PaddleOCR(lang='vi')
    return _ocr_instance

def _normalize(s: str) -> str:
    """Chuẩn hóa chuỗi tiếng Việt"""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_id_fields(image_bgr) -> Dict:
    """
    Input: ảnh BGR (cv2.imread hoặc numpy array)
    Output: dict {
        id_number, name, dob, gender, nationality, 
        place_of_origin, residence, issue_date, expiry_date
    }
    """
    ocr = get_ocr_instance()
    # Phiên bản mới của PaddleOCR tự động xử lý angle classification
    result = ocr.predict(image_bgr)

    lines = []
    # Xử lý các format kết quả khác nhau của PaddleOCR 3.x
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        # Định dạng OCRResult (dict-like) mới: dùng rec_texts/rec_scores
        page = result[0]
        rec_texts = page.get('rec_texts') or []
        rec_scores = page.get('rec_scores') or []
        for txt, score in zip(rec_texts, rec_scores):
            if txt and (score is None or float(score) >= 0.3):
                lines.append(str(txt))
    elif isinstance(result, dict):
        # Một số biến thể trả về dict có ocr_result
        for item in result.get('ocr_result', []):
            if isinstance(item, dict):
                txt = item.get('text')
                score = item.get('confidence', 1.0)
                if txt and float(score) >= 0.3:
                    lines.append(str(txt))
    elif isinstance(result, list):
        # Định dạng cũ của PaddleOCR 2.x
        for page in result:
            if page is None:
                continue
            for line in page:
                if isinstance(line, list) and len(line) >= 2:
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        txt, conf = line[1][0], float(line[1][1])
                        if conf >= 0.3:
                            lines.append(txt)
                    elif isinstance(line[1], dict):
                        txt = line[1].get('text', '')
                        conf = float(line[1].get('confidence', 1.0))
                        if conf >= 0.3 and txt:
                            lines.append(txt)
                    elif isinstance(line[1], str):
                        lines.append(line[1])
                elif isinstance(line, dict):
                    txt = line.get('text', '')
                    conf = float(line.get('confidence', 1.0))
                    if conf >= 0.3 and txt:
                        lines.append(txt)
                elif isinstance(line, str):
                    lines.append(line)
    
    full = _normalize(" ".join(lines))
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
    
    # Họ tên (heuristic theo nhãn phổ biến)
    # Xử lý lỗi OCR: "Ho và tên" (thiếu dấu), "I" thay vì "|" hoặc ":"
    # Text thực tế: "Ho và tên I Full name NGUYN ĐC DÜNG Ngày sinh"
    # Pattern: lấy text HOA sau "Full name" đến trước "Ngày sinh"
    m = re.search(r'Full\s*name\s+([A-ZĂÂĐÊÔƠƯĐ][A-ZĂÂĐÊÔƠƯĐ\s]{5,}?)\s+Ngày\s*sinh', full, re.I)
    if m:
        out["name"] = _normalize(m.group(1))
    # Fallback: text HOA giữa số CCCD và "Ngày sinh" (bỏ qua "Full name" nếu có)
    if not out.get("name"):
        m = re.search(r'\b\d{12}\b\s+.*?Full\s*name\s+([A-ZĂÂĐÊÔƠƯĐ][A-ZĂÂĐÊÔƠƯĐ\s]{5,}?)\s+Ngày\s*sinh', full, re.I)
        if m:
            out["name"] = _normalize(m.group(1))
    # Fallback 2: text HOA thuần túy giữa số CCCD và "Ngày sinh"
    if not out.get("name"):
        m = re.search(r'\b\d{12}\b\s+([A-ZĂÂĐÊÔƠƯĐ][A-ZĂÂĐÊÔƠƯĐ\s]{6,}?)\s+Ngày\s*sinh', full)
        if m:
            name_text = m.group(1).strip()
            # Loại bỏ "Full name" và "Ho và tên I" nếu có
            name_text = re.sub(r'(?:Ho\s+và\s+tên\s*[I|/]\s*)?Full\s*name\s*', '', name_text, flags=re.I).strip()
            if name_text and not re.search(r'\b(birth|date|of)\b', name_text, re.I):
                out["name"] = _normalize(name_text)
    
    # Quê quán
    # Xử lý lỗi OCR: "Qué quán" (có lỗi), "Que quán"
    # Text thực tế: "Qué quán / Place of origin: Ha Binh, Vũ Thư, Thái Binh Noi thuròng trú"
    # Pattern: lấy text sau "Place of origin:" đến trước "Noi thuròng trú"
    m = re.search(r'Place\s*of\s*origin:\s*([^N]+?)\s+Noi\s*thuròng\s*trú', full, re.I | re.DOTALL)
    if m:
        place_text = m.group(1).strip()
        out["place_of_origin"] = _normalize(place_text)
    # Fallback: không có "Place of origin:"
    if not out.get("place_of_origin"):
        m = re.search(r'Qu[êeé]\s*quán\s*[/|I]?\s*[:：]?\s+([^N]+?)\s+N[ơo]i\s*th[ưu]ờng\s*trú', full, re.I | re.DOTALL)
        if m:
            place_text = m.group(1).strip()
            place_text = re.sub(r'Place\s*of\s*origin:?\s*', '', place_text, flags=re.I).strip()
            out["place_of_origin"] = _normalize(place_text)
    # Nơi thường trú / Địa chỉ
    # Xử lý lỗi OCR: "Noi thuròng trú" (thiếu dấu), "I" thay vì "|"
    # Text thực tế: "Noi thuròng trú I Place of residen TDP 2 Ngoc Truc Co già tn dn 07/07/2028 Dai Mê Nam Tù Liêm. Hà Nôi"
    m = re.search(r'N[ơo]i\s*thuròng\s*trú\s*[I|/]\s*Place\s*of\s*residen[ce]?\s+(.+?)\s+Co\s*già\s*tn\s*dn', full, re.I | re.DOTALL)
    if m:
        residence_text = m.group(1).strip()
        # Gộp phần còn lại sau "07/07/2028" nếu có
        m2 = re.search(r'Co\s*già\s*tn\s*dn\s+\d{2}/\d{2}/\d{4}\s+(.+?)(?:\s*$|\s*Nam)', full, re.I | re.DOTALL)
        if m2:
            residence_text += " " + m2.group(1).strip()
        out["residence"] = _normalize(residence_text)
    # Fallback: không có "Place of residen"
    if not out.get("residence"):
        m = re.search(r'N[ơo]i\s*th[ưu]ờng\s*trú\s*[I|/]?\s*[:：]?\s+(.+?)(?=\s+C[oó]\s*giá\s*trị|\s+Co\s*già|\s+\d{2}/\d{2}/\d{4}|\s*$)', full, re.I | re.DOTALL)
        if m:
            residence_text = m.group(1).strip()
            residence_text = re.sub(r'Place\s*of\s*residen[ce]?\s*', '', residence_text, flags=re.I).strip()
            out["residence"] = _normalize(residence_text)
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

